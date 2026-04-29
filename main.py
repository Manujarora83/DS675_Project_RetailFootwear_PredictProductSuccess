"""Main entry point for the organized DS675 footwear classification project."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from get_args import get_args
from src.config import RAW_CATEGORICAL_FEATURES, RAW_NUMERIC_FEATURES
from src.data_loader import FootwearDataLoader
from src.evaluation import ModelEvaluator
from src.models import ClassicalModelTrainer, TorchMLPTrainer
from src.preprocessing import FootwearPreprocessor
from src.utils import ensure_dirs, save_json, set_seed
from src.visualization import (
    plot_categorical_distributions,
    plot_correlation_heatmap,
    plot_numeric_distributions,
    plot_pca_analysis,
    plot_target_distribution,
    plot_training_curves,
)


def resolve_path(path: Path) -> Path:
    """Resolve paths relative to the project root when needed."""
    if path.exists():
        return path
    project_relative = Path(__file__).resolve().parent / path
    return project_relative


def run_eda(df: pd.DataFrame, output_dir: Path, show_plots: bool) -> None:
    print("\nRunning EDA...")
    plot_categorical_distributions(df, RAW_CATEGORICAL_FEATURES, output_dir, show_plots)
    plot_numeric_distributions(df, RAW_NUMERIC_FEATURES, output_dir, show_plots)
    plot_correlation_heatmap(df, RAW_NUMERIC_FEATURES, output_dir, show_plots)
    plot_target_distribution(df, output_dir, show_plots)
    print(f"EDA figures saved to: {output_dir / 'figures'}")


def main() -> None:
    args = get_args()
    set_seed(args.random_state)

    output_dir = resolve_path(args.output_dir)
    ensure_dirs(output_dir)

    data_path = resolve_path(args.data_path)
    loader = FootwearDataLoader(data_path)
    df = loader.load()
    df = loader.create_target(df)

    print("=" * 80)
    print("DS675 Sports Footwear Product Success Classification")
    print("=" * 80)
    print(f"Dataset path: {data_path}")
    print(f"Dataset shape: {df.shape}")
    print("Target distribution:")
    print(df["product_success"].value_counts())

    if args.mode in {"eda", "all"}:
        run_eda(df, output_dir, args.show_plots)

    X, y = loader.build_features(df)
    y_encoded, label_encoder = loader.encode_target(y)
    class_names = label_encoder.classes_.tolist()

    preprocessor = FootwearPreprocessor(random_state=args.random_state)
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split(
        X, y_encoded, val_size=args.val_size, test_size=args.test_size
    )
    X_train_p, X_val_p, X_test_p = preprocessor.fit_transform(X_train, X_val, X_test)

    split_info = {
        "train_samples": len(y_train),
        "validation_samples": len(y_val),
        "test_samples": len(y_test),
        "processed_feature_count": X_train_p.shape[1],
        "numeric_features": preprocessor.numeric_features,
        "categorical_features": preprocessor.categorical_features,
        "class_mapping": dict(zip(class_names, label_encoder.transform(class_names).tolist())),
    }
    save_json(split_info, output_dir / "reports" / "split_and_feature_info.json")
    print("\nSplit and preprocessing summary:")
    print(split_info)

    if args.mode in {"pca", "all"}:
        print("\nRunning PCA analysis...")
        n_90, n_95 = plot_pca_analysis(X_train_p, y_train, class_names, output_dir, args.show_plots)
        print(f"Components needed for 90% variance: {n_90}")
        print(f"Components needed for 95% variance: {n_95}")

    if args.mode not in {"train", "all"}:
        print("\nDone.")
        return

    evaluator = ModelEvaluator(class_names=class_names, output_dir=output_dir, show_plots=args.show_plots)
    classical_trainer = ClassicalModelTrainer(
        random_state=args.random_state,
        cv_folds=args.cv_folds,
        n_jobs=args.n_jobs,
        quick=args.quick,
    )

    trained_models = {}

    if args.model in {"logistic", "all"}:
        print("\nTraining Logistic Regression...")
        lr_model = classical_trainer.train_logistic_regression(X_train_p, y_train)
        trained_models["logistic_regression"] = lr_model
        y_pred = lr_model.best_estimator_.predict(X_val_p)
        y_proba = lr_model.best_estimator_.predict_proba(X_val_p)
        evaluator.evaluate("Logistic Regression", y_val, y_pred, y_proba, split="Validation")
        classical_trainer.save_model(lr_model, output_dir / "models" / "logistic_regression.joblib")

    if args.model in {"xgboost", "all"}:
        print("\nTraining XGBoost...")
        try:
            xgb_model = classical_trainer.train_xgboost(X_train_p, y_train)
            trained_models["xgboost"] = xgb_model
            y_pred = xgb_model.best_estimator_.predict(X_val_p)
            y_proba = xgb_model.best_estimator_.predict_proba(X_val_p)
            evaluator.evaluate("XGBoost", y_val, y_pred, y_proba, split="Validation")
            classical_trainer.save_model(xgb_model, output_dir / "models" / "xgboost.joblib")
        except ImportError as exc:
            print(f"Skipping XGBoost: {exc}")

    if args.model in {"mlp", "all"}:
        print("\nTraining PyTorch MLP...")
        mlp_trainer = TorchMLPTrainer(
            input_dim=X_train_p.shape[1],
            learning_rate=args.learning_rate,
            dropout=args.dropout,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            epochs=args.epochs,
            early_stop_patience=args.early_stop_patience,
            random_state=args.random_state,
        )
        mlp_trainer.fit(X_train_p, y_train, X_val_p, y_val)
        plot_training_curves(
            mlp_trainer.history["train_loss"],
            mlp_trainer.history["val_loss"],
            mlp_trainer.history["train_f1"],
            mlp_trainer.history["val_f1"],
            output_dir,
            args.show_plots,
        )
        trained_models["pytorch_mlp"] = mlp_trainer
        y_pred = mlp_trainer.predict(X_val_p)
        y_proba = mlp_trainer.predict_proba(X_val_p)
        evaluator.evaluate("PyTorch MLP", y_val, y_pred, y_proba, split="Validation")
        mlp_trainer.save(output_dir / "models" / "pytorch_mlp.pt")

    print("\nFinal test-set evaluation for trained models...")
    for name, model in trained_models.items():
        display_name = name.replace("_", " ").title()
        if name in {"logistic_regression", "xgboost"}:
            estimator = model.best_estimator_
            y_pred = estimator.predict(X_test_p)
            y_proba = estimator.predict_proba(X_test_p)
        else:
            y_pred = model.predict(X_test_p)
            y_proba = model.predict_proba(X_test_p)
        evaluator.evaluate(display_name, y_test, y_pred, y_proba, split="Test")

    comparison = evaluator.comparison_table()
    comparison_path = output_dir / "reports" / "model_comparison.csv"
    comparison.to_csv(comparison_path)
    print("\nModel comparison:")
    print(comparison.to_string())
    print(f"\nSaved comparison table to: {comparison_path}")
    print("Done.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(130)
