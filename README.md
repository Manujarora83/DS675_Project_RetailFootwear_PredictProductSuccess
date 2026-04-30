# DS675 Footwear Product Success Classification Project

Predicting Product Success in Sports Footwear Retail: A Multi-Model Classification Approach.

**Authors:** Manuj Arora, Vivek Gupta
**Course:** DS675 Machine Learning, NJIT, Spring 2026

## Overview

This project builds a supervised ML system that classifies sports footwear products into three success tiers — **High Performing**, **Average**, and **Underperforming** — based solely on attributes available before a product is sold.

The pipeline includes:

- Data loading, target engineering, and leakage-safe preprocessing
- Exploratory Data Analysis with visualizations
- PCA / Kernel PCA dimensionality reduction analysis
- Logistic Regression, XGBoost, and PyTorch MLP training with Grid Search
- MLP architecture experiments (6 variants)
- K-Means clustering comparison (unsupervised vs supervised)
- Grand model comparison: confusion matrices, ROC-AUC, F1, accuracy
- All plots and metrics saved to `outputs/`

## Project Structure

```text
footwear_ml_project/
├── data/
│   └── global_sports_footwear_sales_2018_2026.csv
├── outputs/
│   ├── figures/          # All generated plots
│   ├── models/           # Saved model files
│   └── reports/          # CSV/JSON metrics and comparison tables
├── src/
│   ├── config.py         # Constants and feature definitions
│   ├── data_loader.py    # Data loading and target creation
│   ├── evaluation.py     # Metrics, classification reports
│   ├── models.py         # LogReg, XGBoost, MLP, K-Means, architecture experiments
│   ├── preprocessing.py  # Leakage-safe split and encoding
│   ├── utils.py          # Seed setting, file helpers
│   └── visualization.py  # All plotting functions
├── get_args.py           # Command-line argument definitions
├── main.py               # Main entry point
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

Run the complete pipeline:
```bash
python main.py --mode all --model all
```

Quick classroom/demo version (smaller grids, fewer epochs):
```bash
python main.py --mode all --model all --quick --epochs 30 --cv-folds 3
```

Run only EDA:
```bash
python main.py --mode eda
```

Train a single model:
```bash
python main.py --mode train --model mlp --epochs 100
python main.py --mode train --model xgboost --quick
python main.py --mode train --model logistic
```

## Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--mode` | `all` | `eda`, `pca`, `train`, or `all` |
| `--model` | `all` | `logistic`, `xgboost`, `mlp`, or `all` |
| `--quick` | off | Use smaller hyperparameter grids |
| `--epochs` | 100 | MLP training epochs |
| `--cv-folds` | 5 | Cross-validation folds |
| `--test-size` | 0.15 | Test set proportion |
| `--show-plots` | off | Display plots interactively |

## Data Leakage Prevention

The target `product_success` is created from `revenue_usd` and `customer_rating`, so these columns are removed from model features before training. `final_price_usd` is also dropped as it is deterministic. StandardScaler and OneHotEncoder are fit only on the training split.

## Output Files

```text
outputs/
├── figures/
│   ├── categorical_distributions.png
│   ├── numeric_distributions.png
│   ├── correlation_heatmap.png
│   ├── target_distribution.png
│   ├── pca_scree_cumulative_variance.png
│   ├── linear_vs_kernel_pca.png
│   ├── mlp_training_curves.png
│   ├── mlp_architecture_comparison.png
│   ├── kmeans_vs_actual_labels.png
│   ├── model_comparison_bars.png
│   ├── all_confusion_matrices_test.png
│   ├── all_roc_curves_test.png
│   └── [individual model confusion matrices and ROC curves]
├── models/
│   ├── logistic_regression.joblib
│   ├── xgboost.joblib
│   └── pytorch_mlp.pt
└── reports/
    ├── split_and_feature_info.json
    ├── model_comparison.csv
    ├── mlp_architecture_comparison.csv
    └── kmeans_comparison.json
```
