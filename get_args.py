"""Command-line arguments for the sports footwear product-success project."""
from __future__ import annotations

import argparse
from pathlib import Path


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DS675 Sports Footwear Product Success Classification Project"
    )

    # Paths
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/global_sports_footwear_sales_2018_2026.csv"),
        help="Path to the footwear sales CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory where figures, reports, and models are saved.",
    )

    # Execution controls
    parser.add_argument(
        "--mode",
        choices=["eda", "pca", "train", "all"],
        default="all",
        help="Which portion of the project to run.",
    )
    parser.add_argument(
        "--model",
        choices=["logistic", "xgboost", "mlp", "all"],
        default="all",
        help="Which supervised model to train/evaluate.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)

    # Neural network settings
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.30)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--early-stop-patience", type=int, default=15)

    # Grid-search controls
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use smaller hyperparameter grids for faster classroom/demo runs.",
    )
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=-1)

    # Plot controls
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Show plots interactively in addition to saving them.",
    )

    return parser.parse_args()
