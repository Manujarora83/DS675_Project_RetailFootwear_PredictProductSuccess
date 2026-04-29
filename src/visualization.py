"""Plotting functions for EDA, PCA, and model evaluation."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, KernelPCA
from sklearn.metrics import ConfusionMatrixDisplay, auc, roc_curve
from sklearn.preprocessing import label_binarize

try:
    import seaborn as sns
except Exception:  # pragma: no cover
    sns = None


def _save_or_show(path: Path, show: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def plot_categorical_distributions(df: pd.DataFrame, columns: list[str], output_dir: Path, show: bool = False) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    axes = axes.flatten()
    for i, col in enumerate(columns):
        df[col].value_counts().plot(kind="bar", ax=axes[i], color="steelblue", edgecolor="black")
        axes[i].set_title(col, fontsize=12, fontweight="bold")
        axes[i].tick_params(axis="x", rotation=45)
    plt.suptitle("Distribution of Categorical Features", fontsize=16, fontweight="bold", y=1.02)
    _save_or_show(output_dir / "figures" / "categorical_distributions.png", show)


def plot_numeric_distributions(df: pd.DataFrame, columns: list[str], output_dir: Path, show: bool = False) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    axes = axes.flatten()
    for i, col in enumerate(columns):
        axes[i].hist(df[col], bins=30, color="steelblue", edgecolor="black", alpha=0.7)
        axes[i].set_title(col, fontsize=12, fontweight="bold")
        axes[i].axvline(df[col].mean(), color="red", linestyle="--", label=f"Mean: {df[col].mean():.1f}")
        axes[i].legend(fontsize=8)
    axes[-1].set_visible(False)
    plt.suptitle("Distribution of Numeric Features", fontsize=16, fontweight="bold", y=1.02)
    _save_or_show(output_dir / "figures" / "numeric_distributions.png", show)


def plot_correlation_heatmap(df: pd.DataFrame, columns: list[str], output_dir: Path, show: bool = False) -> None:
    plt.figure(figsize=(10, 8))
    corr = df[columns].corr()
    if sns is not None:
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True, linewidths=0.5)
    else:
        plt.imshow(corr, aspect="auto")
        plt.xticks(range(len(columns)), columns, rotation=45, ha="right")
        plt.yticks(range(len(columns)), columns)
        plt.colorbar()
    plt.title("Correlation Heatmap of Numeric Features", fontsize=14, fontweight="bold")
    _save_or_show(output_dir / "figures" / "correlation_heatmap.png", show)


def plot_target_distribution(df: pd.DataFrame, output_dir: Path, show: bool = False) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(df["success_score"], bins=50, color="steelblue", edgecolor="black", alpha=0.7)
    q_low = df.loc[df["product_success"] == "Underperforming", "success_score"].max()
    q_high = df.loc[df["product_success"] == "Average", "success_score"].max()
    axes[0].axvline(q_low, color="red", linestyle="--", linewidth=2, label=f"Threshold 1: {q_low:.2f}")
    axes[0].axvline(q_high, color="orange", linestyle="--", linewidth=2, label=f"Threshold 2: {q_high:.2f}")
    axes[0].set_title("Composite Success Score Distribution", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Success Score")
    axes[0].legend()

    df["product_success"].value_counts().plot(kind="bar", ax=axes[1], color=["#e74c3c", "#f39c12", "#27ae60"], edgecolor="black")
    axes[1].set_title("Product Success Class Distribution", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Count")
    axes[1].tick_params(axis="x", rotation=0)
    _save_or_show(output_dir / "figures" / "target_distribution.png", show)


def plot_pca_analysis(X_train_processed: np.ndarray, y_train: np.ndarray, class_names: list[str], output_dir: Path, show: bool = False) -> tuple[int, int]:
    pca_full = PCA(random_state=42)
    pca_full.fit(X_train_processed)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    n_90 = int(np.argmax(cumvar >= 0.90) + 1)
    n_95 = int(np.argmax(cumvar >= 0.95) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(range(1, len(pca_full.explained_variance_ratio_) + 1), pca_full.explained_variance_ratio_, color="steelblue", edgecolor="black", alpha=0.7)
    axes[0].set_xlabel("Principal Component")
    axes[0].set_ylabel("Explained Variance Ratio")
    axes[0].set_title("Scree Plot", fontsize=12, fontweight="bold")

    axes[1].plot(range(1, len(cumvar) + 1), cumvar, "bo-", linewidth=2)
    axes[1].axhline(y=0.90, color="red", linestyle="--", label="90% variance")
    axes[1].axhline(y=0.95, color="orange", linestyle="--", label="95% variance")
    axes[1].set_xlabel("Number of Components")
    axes[1].set_ylabel("Cumulative Explained Variance")
    axes[1].set_title("Cumulative Variance Explained", fontsize=12, fontweight="bold")
    axes[1].legend()
    _save_or_show(output_dir / "figures" / "pca_scree_cumulative_variance.png", show)

    pca_2d = PCA(n_components=2, random_state=42)
    X_pca_2d = pca_2d.fit_transform(X_train_processed)
    kpca_2d = KernelPCA(n_components=2, kernel="rbf", gamma=0.01, random_state=42)
    X_kpca_2d = kpca_2d.fit_transform(X_train_processed)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    colors = {0: "#e74c3c", 1: "#f39c12", 2: "#27ae60"}
    for ax, data, title in [(axes[0], X_pca_2d, "Linear PCA"), (axes[1], X_kpca_2d, "Kernel PCA (RBF)")]:
        for cls_idx in range(len(class_names)):
            mask = y_train == cls_idx
            ax.scatter(data[mask, 0], data[mask, 1], c=colors.get(cls_idx, "gray"), label=class_names[cls_idx], alpha=0.3, s=10)
        ax.set_title(f"{title} — 2D Projection", fontsize=12, fontweight="bold")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.legend(markerscale=3)
    _save_or_show(output_dir / "figures" / "linear_vs_kernel_pca.png", show)
    return n_90, n_95


def plot_training_curves(train_losses, val_losses, train_f1s, val_f1s, output_dir: Path, show: bool = False) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(train_losses, label="Training Loss", linewidth=2)
    axes[0].plot(val_losses, label="Validation Loss", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].set_title("Training vs Validation Loss", fontsize=12, fontweight="bold")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(train_f1s, label="Training F1", linewidth=2)
    axes[1].plot(val_f1s, label="Validation F1", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro F1 Score")
    axes[1].set_title("Training vs Validation F1", fontsize=12, fontweight="bold")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    _save_or_show(output_dir / "figures" / "mlp_training_curves.png", show)


def plot_confusion_matrix(y_true, y_pred, class_names, title: str, output_dir: Path, show: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=class_names, cmap="Blues", ax=ax)
    ax.set_title(title, fontsize=12, fontweight="bold")
    safe_name = title.lower().replace(" ", "_").replace("—", "_").replace("/", "_")
    _save_or_show(output_dir / "figures" / f"{safe_name}_confusion_matrix.png", show)


def plot_roc_curves(y_true, y_proba, class_names, title: str, output_dir: Path, show: bool = False) -> None:
    y_true_bin = label_binarize(y_true, classes=list(range(len(class_names))))
    colors = ["#e74c3c", "#f39c12", "#27ae60"]
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, cls_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
        roc_auc_i = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors[i % len(colors)], linewidth=2, label=f"{cls_name} (AUC={roc_auc_i:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    safe_name = title.lower().replace(" ", "_").replace("—", "_").replace("/", "_")
    _save_or_show(output_dir / "figures" / f"{safe_name}_roc_curves.png", show)
