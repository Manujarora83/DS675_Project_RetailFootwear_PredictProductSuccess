from __future__ import annotations
""" DS675 Mini-Project Milestone 4: Predicting Product Success in Sports Footwear Retail: A Multi-Model Classification Approach

**Dataset:** Sports Footwear Sales and Consumer Behavior (Kaggle)  
**Authors:** Manuj Arora, Vivek Gupta <br>
**Date:** Spring 2026  

**Objective:** Before stocking a new shoe, can a retailer predict whether it will be a **Hit**, **Average**, or **Flop** — based on brand, category, price, discount, channel, and market? This model extends beyond footwear and is applicable to all retail sales scenarios.

**Supervised Models:** Logistic Regression → XGBoost → PyTorch MLP  
**Unsupervised Comparison:** K-Means Clustering  
**Dimensionality Reduction:** PCA / Kernel PCA  
**Evaluation Metrics:** Accuracy, Macro F1, Confusion Matrix, ROC-AUC (One-vs-Rest)"""

"""Main entry point for the organized DS675 footwear classification project."""

# import sys library
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
# get command-line arguments
from get_args import get_args
from src.config import RAW_CATEGORICAL_FEATURES, RAW_NUMERIC_FEATURES
from src.data_loader import FootwearDataLoader
from src.evaluation import ModelEvaluator
from src.models import ClassicalModelTrainer, KMeansComparison, TorchMLPTrainer, run_mlp_architecture_experiments
from src.preprocessing import FootwearPreprocessor
from src.utils import ensure_dirs, save_json, set_seed
from src.visualization import (
    plot_categorical_distributions,
    plot_correlation_heatmap,
    plot_kmeans_comparison,
    plot_mlp_architecture_comparison,
    plot_model_comparison_bars,
    plot_all_confusion_matrices,
    plot_all_roc_curves,
    plot_numeric_distributions,
    plot_pca_analysis,
    plot_target_distribution,
    plot_training_curves,
)

# Resolve paths relative to the project root when needed.
def resolve_path(path: Path) -> Path:
    """Resolve paths relative to the project root when needed."""
    if path.exists():
        return path
    project_relative = Path(__file__).resolve().parent / path
    return project_relative

# Run Exploratory Data Analysis (EDA)
def run_eda(df: pd.DataFrame, output_dir: Path, show_plots: bool) -> None:
    print("\nRunning EDA...")
    plot_categorical_distributions(df, RAW_CATEGORICAL_FEATURES, output_dir, show_plots)
    plot_numeric_distributions(df, RAW_NUMERIC_FEATURES, output_dir, show_plots)
    plot_correlation_heatmap(df, RAW_NUMERIC_FEATURES, output_dir, show_plots)
    plot_target_distribution(df, output_dir, show_plots)
    print(f"EDA figures saved to: {output_dir / 'figures'}")

# Main entry point for the organized DS675 footwear classification project.
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
    # Train classical models
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

    # ── MLP Architecture Experiments (Section 8.4) ──────────────────────────
    if args.model in {"mlp", "all"}:
        print("\nRunning MLP architecture experiments...")
        arch_results = run_mlp_architecture_experiments(
            X_train_p, y_train, X_val_p, y_val,
            input_dim=X_train_p.shape[1],
            epochs=30 if args.quick else 80,
            batch_size=args.batch_size,
            random_state=args.random_state,
        )
        if arch_results:
            plot_mlp_architecture_comparison(arch_results, output_dir, args.show_plots)
            # Save as CSV
            import csv
            arch_path = output_dir / "reports" / "mlp_architecture_comparison.csv"
            arch_path.parent.mkdir(parents=True, exist_ok=True)
            with open(arch_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Architecture", "Best Val F1", "Parameters"])
                for name, f1, params in arch_results:
                    writer.writerow([name, f"{f1:.4f}", params])
            print(f"Architecture comparison saved to: {arch_path}")

    # ── K-Means Clustering Comparison (Section 9 in notebook) ─────────────
    print("\nRunning K-Means clustering comparison...")
    kmeans_comp = KMeansComparison(n_clusters=3, random_state=args.random_state, n_init=10)
    kmeans_labels = kmeans_comp.fit_predict(X_train_p)
    kmeans_metrics = kmeans_comp.evaluate_with_X(X_train_p, y_train)

    print(f"K-Means Results:")
    print(f"  Adjusted Rand Index (vs true labels): {kmeans_metrics['ARI']:.4f}")
    print(f"  Silhouette Score: {kmeans_metrics['Silhouette']:.4f}")
    print(f"  Cluster sizes: {kmeans_metrics['cluster_sizes']}")

    save_json(kmeans_metrics, output_dir / "reports" / "kmeans_comparison.json")
    plot_kmeans_comparison(
        X_train_p, y_train, kmeans_labels,
        kmeans_comp.kmeans.cluster_centers_,
        class_names, kmeans_metrics["ARI"],
        output_dir, args.show_plots,
    )
    # Analyze the results of K-Means clustering
    if kmeans_metrics["ARI"] < 0.1:
        print("  → Low ARI: product success structure does NOT emerge naturally.")
        print("    Supervised learning is essential — validating our approach.")
    else:
        print(f"  → ARI of {kmeans_metrics['ARI']:.3f} suggests some natural structure exists.")

    # ── Final Test-Set Evaluation ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  FINAL TEST SET EVALUATION")
    print("=" * 70)

    test_predictions = {}   # name -> (y_true, y_pred)
    test_probas = {}        # name -> (y_true, y_proba)

    # Logistic Regression and XGBoost
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
        test_predictions[display_name] = (y_test, y_pred)
        test_probas[display_name] = (y_test, y_proba)

    # ── Section 11: Grand Model Comparison ────────────────────────────────
    print("\nGenerating grand comparison visualizations...")

    # Build comparison table from test results only
    test_results = {k: v for k, v in evaluator.results.items() if "(Test)" in k}
    if test_results:
        import pandas as pd
        test_df = pd.DataFrame(test_results).T.round(4)
        # Clean up index names (remove " (Test)" suffix)
        test_df.index = [idx.replace(" (Test)", "") for idx in test_df.index]

        plot_model_comparison_bars(test_df, output_dir, args.show_plots)

    # Section 11.1: All confusion matrices and ROC curves side-by-side
    if test_predictions:
        plot_all_confusion_matrices(test_predictions, class_names, output_dir, args.show_plots)
    if test_probas:
        plot_all_roc_curves(test_probas, class_names, output_dir, args.show_plots)
    # ── Showcase: What the Retailer Sees ──────────────────────────────────
    print("\nGenerating retailer showcase predictions...")

    # Use the best available model for showcase
    best_model_name = list(test_predictions.keys())[0]  # first model
    best_y_pred = None
    best_y_proba = None

    for name, model in trained_models.items():
        display_name = name.replace("_", " ").title()
        if name in {"logistic_regression", "xgboost"}:
            best_y_pred = model.best_estimator_.predict(X_test_p)
            best_y_proba = model.best_estimator_.predict_proba(X_test_p)
        else:
            best_y_pred = model.predict(X_test_p)
            best_y_proba = model.predict_proba(X_test_p)
        best_model_name = display_name

    # Sample 15 test products
    import pandas as pd
    import numpy as np
    rng = np.random.RandomState(args.random_state)
    sample_idx = rng.choice(len(y_test), 15, replace=False)

    showcase = X_test.iloc[sample_idx].copy()
    showcase["Actual"] = label_encoder.inverse_transform(y_test[sample_idx])
    showcase["Predicted"] = label_encoder.inverse_transform(best_y_pred[sample_idx])
    showcase["Confidence"] = best_y_proba[sample_idx].max(axis=1)
    showcase["Correct"] = showcase["Actual"] == showcase["Predicted"]

    display_cols = ["brand", "category", "gender", "base_price_usd", "discount_percent",
                    "sales_channel", "country", "Predicted", "Confidence", "Actual", "Correct"]
    display_df = showcase[display_cols].reset_index(drop=True)
    display_df["Confidence"] = display_df["Confidence"].apply(lambda x: f"{x:.1%}")

    print("\n" + "=" * 90)
    print(f"  WHAT THE RETAILER SEES: Sample Predictions ({best_model_name})")
    print("=" * 90)
    print(display_df.to_string(index=False))

    # Save to CSV
    showcase_path = output_dir / "reports" / "sample_predictions.csv"
    display_df.to_csv(showcase_path, index=False)
    print(f"\nSaved sample predictions to: {showcase_path}")

    # Prediction distribution: Actual vs Predicted
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    actual_labels = pd.Series(label_encoder.inverse_transform(y_test))
    pred_labels = pd.Series(label_encoder.inverse_transform(best_y_pred))

    actual_labels.value_counts().reindex(class_names).plot(
        kind="bar", ax=axes[0], color=["#e74c3c", "#f39c12", "#27ae60"], edgecolor="black")
    axes[0].set_title("Actual Labels (Test Set)", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis="x", rotation=0)

    pred_labels.value_counts().reindex(class_names).plot(
        kind="bar", ax=axes[1], color=["#e74c3c", "#f39c12", "#27ae60"], edgecolor="black")
    axes[1].set_title("Predicted Labels (Test Set)", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Count")
    axes[1].tick_params(axis="x", rotation=0)

    plt.suptitle("Actual vs Predicted Label Distribution", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / "figures" / "actual_vs_predicted_distribution.png",
                dpi=150, bbox_inches="tight")
    if args.show_plots:
        plt.show()
    plt.close()

    correct = (best_y_pred == y_test).sum()
    print(f"\nOut of {len(y_test)} test products:")
    print(f"  Correctly classified: {correct} ({correct/len(y_test)*100:.1f}%)")
    print(f"  Misclassified: {len(y_test)-correct} ({(len(y_test)-correct)/len(y_test)*100:.1f}%)")
    # ── Save final comparison table ───────────────────────────────────────
    comparison = evaluator.comparison_table()
    comparison_path = output_dir / "reports" / "model_comparison.csv"
    comparison.to_csv(comparison_path)
    print("\nModel comparison:")
    print(comparison.to_string())
    print(f"\nSaved comparison table to: {comparison_path}")
    print("Done.")

#Check if the script is being run directly or interrupted
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(130)
