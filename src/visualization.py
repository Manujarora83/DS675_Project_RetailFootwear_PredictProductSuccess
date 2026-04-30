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


# ── Section 11: Grand Model Comparison ──────────────────────────────────────


def plot_model_comparison_bars(
    comparison_df: "pd.DataFrame",
    output_dir: Path,
    show: bool = False,
) -> None:
    """Side-by-side bar chart of Accuracy, Macro F1, and ROC-AUC for all models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    models = comparison_df.index.tolist()
    x = np.arange(len(models))
    width = 0.25

    # Accuracy / F1 / AUC
    axes[0].bar(x - width, comparison_df["Accuracy"], width, label="Accuracy", color="steelblue", edgecolor="black")
    axes[0].bar(x, comparison_df["Macro F1"], width, label="Macro F1", color="darkorange", edgecolor="black")
    if "ROC-AUC" in comparison_df.columns and comparison_df["ROC-AUC"].notna().all():
        axes[0].bar(x + width, comparison_df["ROC-AUC"], width, label="ROC-AUC", color="#27ae60", edgecolor="black")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=15, ha="right")
    axes[0].set_ylabel("Score")
    axes[0].set_title("Model Comparison: Accuracy, F1 & ROC-AUC", fontsize=12, fontweight="bold")
    axes[0].legend()
    axes[0].set_ylim(0, 1)
    axes[0].grid(axis="y", alpha=0.3)

    # Per-class F1
    f1_cols = [c for c in comparison_df.columns if c.startswith("F1 (")]
    if f1_cols:
        w = 0.25
        class_colors = ["#e74c3c", "#f39c12", "#27ae60"]
        for j, col in enumerate(f1_cols):
            offset = (j - len(f1_cols) / 2 + 0.5) * w
            label = col.replace("F1 (", "").replace(")", "")
            axes[1].bar(x + offset, comparison_df[col], w, label=label,
                        color=class_colors[j % len(class_colors)], edgecolor="black")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(models, rotation=15, ha="right")
        axes[1].set_ylabel("F1 Score")
        axes[1].set_title("Per-Class F1 Score by Model", fontsize=12, fontweight="bold")
        axes[1].legend()
        axes[1].set_ylim(0, 1)
        axes[1].grid(axis="y", alpha=0.3)

    _save_or_show(output_dir / "figures" / "model_comparison_bars.png", show)


# ── Section 11.1: All Confusion Matrices & ROC Curves side-by-side ──────────


def plot_all_confusion_matrices(
    all_predictions: dict[str, tuple[np.ndarray, np.ndarray]],
    class_names: list[str],
    output_dir: Path,
    show: bool = False,
) -> None:
    """1×N panel of confusion matrices for all models."""
    n = len(all_predictions)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (name, (y_true, y_pred)) in zip(axes, all_predictions.items()):
        ConfusionMatrixDisplay.from_predictions(
            y_true, y_pred, display_labels=class_names,
            cmap="Blues", ax=ax, colorbar=False,
        )
        ax.set_title(name, fontsize=11, fontweight="bold")

    plt.suptitle("Confusion Matrices — Test Set", fontsize=16, fontweight="bold", y=1.05)
    _save_or_show(output_dir / "figures" / "all_confusion_matrices_test.png", show)


def plot_all_roc_curves(
    all_probas: dict[str, tuple[np.ndarray, np.ndarray]],
    class_names: list[str],
    output_dir: Path,
    show: bool = False,
) -> None:
    """1×N panel of ROC curves for all models."""
    n = len(all_probas)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    colors = ["#e74c3c", "#f39c12", "#27ae60"]

    for ax, (name, (y_true, y_proba)) in zip(axes, all_probas.items()):
        y_true_bin = label_binarize(y_true, classes=list(range(len(class_names))))
        for i, (cls_name, color) in enumerate(zip(class_names, colors)):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
            roc_auc_i = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, linewidth=2, label=f"{cls_name} (AUC={roc_auc_i:.3f})")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle("ROC Curves (One-vs-Rest) — Test Set", fontsize=16, fontweight="bold", y=1.05)
    _save_or_show(output_dir / "figures" / "all_roc_curves_test.png", show)


# ── K-Means Visualization ───────────────────────────────────────────────────


def plot_kmeans_comparison(
    X_train_processed: np.ndarray,
    y_train: np.ndarray,
    kmeans_labels: np.ndarray,
    cluster_centers: np.ndarray,
    class_names: list[str],
    ari: float,
    output_dir: Path,
    show: bool = False,
) -> None:
    """Actual labels vs K-Means clusters in PCA 2D space."""
    pca_2d = PCA(n_components=2, random_state=42)
    X_pca = pca_2d.fit_transform(X_train_processed)
    centroids_pca = pca_2d.transform(cluster_centers)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    colors = {0: "#e74c3c", 1: "#f39c12", 2: "#27ae60"}

    # Actual labels
    for cls_idx in range(len(class_names)):
        mask = y_train == cls_idx
        axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1],
                        c=colors.get(cls_idx, "gray"), label=class_names[cls_idx], alpha=0.3, s=10)
    axes[0].set_title("Actual Product Success Labels", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[0].legend(markerscale=3)

    # K-Means clusters
    for c_idx in range(3):
        mask = kmeans_labels == c_idx
        axes[1].scatter(X_pca[mask, 0], X_pca[mask, 1],
                        c=colors.get(c_idx, "gray"), label=f"Cluster {c_idx}", alpha=0.3, s=10)
    axes[1].scatter(centroids_pca[:, 0], centroids_pca[:, 1],
                    c="black", marker="X", s=200, edgecolors="white", linewidth=2, label="Centroids")
    axes[1].set_title(f"K-Means Clusters (ARI = {ari:.3f})", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    axes[1].legend(markerscale=3)

    _save_or_show(output_dir / "figures" / "kmeans_vs_actual_labels.png", show)


# ── MLP Architecture Comparison ─────────────────────────────────────────────


def plot_mlp_architecture_comparison(
    variants: list[tuple[str, float, int]],
    output_dir: Path,
    show: bool = False,
) -> None:
    """Bar chart of MLP architecture variants by validation F1."""
    names = [v[0] for v in variants]
    f1s = [v[1] for v in variants]
    params = [v[2] for v in variants]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(names, f1s, color="steelblue", edgecolor="black")
    best_idx = np.argmax(f1s)
    bars[best_idx].set_color("darkorange")

    for bar, f1, p in zip(bars, f1s, params):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"F1={f1:.4f} ({p:,} params)", va="center", fontsize=9)

    ax.set_xlabel("Best Validation Macro F1")
    ax.set_title("MLP Architecture Comparison", fontsize=14, fontweight="bold")
    ax.set_xlim(0, max(f1s) + 0.05)
    _save_or_show(output_dir / "figures" / "mlp_architecture_comparison.png", show)
