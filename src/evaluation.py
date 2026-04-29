"""Evaluation helpers for classification models."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize

from .visualization import plot_confusion_matrix, plot_roc_curves


class ModelEvaluator:
    """Evaluate models and collect comparable metrics."""

    def __init__(self, class_names: list[str], output_dir: Path, show_plots: bool = False):
        self.class_names = list(class_names)
        self.output_dir = Path(output_dir)
        self.show_plots = show_plots
        self.results: dict[str, dict[str, Any]] = {}

    def evaluate(self, name: str, y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None = None, split: str = "Validation") -> dict[str, Any]:
        metrics: dict[str, Any] = {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Macro F1": f1_score(y_true, y_pred, average="macro"),
        }
        if y_proba is not None:
            y_true_bin = label_binarize(y_true, classes=list(range(len(self.class_names))))
            metrics["ROC-AUC"] = roc_auc_score(y_true_bin, y_proba, multi_class="ovr", average="macro")
        else:
            metrics["ROC-AUC"] = None

        per_class_f1 = f1_score(y_true, y_pred, average=None)
        for idx, class_name in enumerate(self.class_names):
            metrics[f"F1 ({class_name})"] = per_class_f1[idx]

        key = f"{name} ({split})"
        self.results[key] = metrics

        print("=" * 70)
        print(f"{key}")
        print("=" * 70)
        print(f"Accuracy: {metrics['Accuracy']:.4f}")
        print(f"Macro F1: {metrics['Macro F1']:.4f}")
        if metrics["ROC-AUC"] is not None:
            print(f"ROC-AUC:  {metrics['ROC-AUC']:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=self.class_names))

        plot_confusion_matrix(y_true, y_pred, self.class_names, f"{key} — Confusion Matrix", self.output_dir, self.show_plots)
        if y_proba is not None:
            plot_roc_curves(y_true, y_proba, self.class_names, f"{key} — ROC Curves OvR", self.output_dir, self.show_plots)
        return metrics

    def comparison_table(self) -> pd.DataFrame:
        return pd.DataFrame(self.results).T.round(4)
