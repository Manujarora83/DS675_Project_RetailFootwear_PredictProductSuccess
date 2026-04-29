# DS675 Footwear Product Success Classification Project

This is a clean Python project version of the original notebook. It keeps the core workflow intact while organizing the code into reusable files:

- data loading and target creation
- leakage-safe preprocessing
- EDA visualizations
- PCA / Kernel PCA analysis
- Logistic Regression, XGBoost, and PyTorch MLP training
- confusion matrices, ROC-AUC, F1, and final model comparison

## Project structure

```text
footwear_ml_project/
├── data/
│   └── global_sports_footwear_sales_2018_2026.csv
├── outputs/
│   ├── figures/
│   ├── models/
│   └── reports/
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── evaluation.py
│   ├── models.py
│   ├── preprocessing.py
│   ├── utils.py
│   └── visualization.py
├── get_args.py
├── main.py
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

For PyTorch GPU support, install PyTorch from the official selector if needed.

## Common runs

Run the complete pipeline:

```bash
python main.py --mode all --model all
```

Run faster classroom/demo version:

```bash
python main.py --mode all --model all --quick --epochs 30 --cv-folds 3
```

Run only EDA:

```bash
python main.py --mode eda
```

Run only PCA:

```bash
python main.py --mode pca
```

Train only the neural network:

```bash
python main.py --mode train --model mlp --epochs 100
```

Train only Logistic Regression:

```bash
python main.py --mode train --model logistic
```

Train only XGBoost:

```bash
python main.py --mode train --model xgboost --quick
```

## Notes on leakage prevention

The target `product_success` is created from `revenue_usd` and `customer_rating`, so these columns are removed from model features before training. StandardScaler and OneHotEncoder are fit only on the training split, then applied to validation and test sets.

## Output files

Generated plots are saved under:

```text
outputs/figures/
```

Models are saved under:

```text
outputs/models/
```

Metrics are saved under:

```text
outputs/reports/
```
