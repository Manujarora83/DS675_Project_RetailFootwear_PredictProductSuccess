"""Model definitions and training classes."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold

try:
    import xgboost as xgb
except Exception:  # pragma: no cover
    xgb = None

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
except Exception:  # pragma: no cover
    torch = None
    nn = None
    optim = None
    DataLoader = None
    TensorDataset = None

from sklearn.metrics import f1_score


@dataclass
class ClassicalModelTrainer:
    random_state: int = 42
    cv_folds: int = 5
    n_jobs: int = -1
    quick: bool = False

    def train_logistic_regression(self, X_train: np.ndarray, y_train: np.ndarray) -> GridSearchCV:
        param_grid = {
            "C": [0.01, 0.1, 1, 10] if self.quick else [0.001, 0.01, 0.1, 1, 10, 100],
            "solver": ["lbfgs"],
            "max_iter": [1000],
        }
        search = GridSearchCV(
            LogisticRegression(random_state=self.random_state),
            param_grid=param_grid,
            cv=StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state),
            scoring="f1_macro",
            n_jobs=self.n_jobs,
        )
        search.fit(X_train, y_train)
        print(f"Best Logistic Regression params: {search.best_params_}")
        print(f"Best CV Macro F1: {search.best_score_:.4f}")
        return search

    def train_xgboost(self, X_train: np.ndarray, y_train: np.ndarray) -> GridSearchCV:
        if xgb is None:
            raise ImportError("xgboost is not installed. Install it with: pip install xgboost")
        if self.quick:
            param_grid = {
                "n_estimators": [100],
                "max_depth": [4, 6],
                "learning_rate": [0.05, 0.1],
                "subsample": [0.8],
            }
        else:
            param_grid = {
                "n_estimators": [100, 300, 500],
                "max_depth": [4, 6, 8],
                "learning_rate": [0.01, 0.05, 0.1],
                "subsample": [0.8, 1.0],
            }
        search = GridSearchCV(
            xgb.XGBClassifier(
                objective="multi:softprob",
                num_class=3,
                eval_metric="mlogloss",
                random_state=self.random_state,
            ),
            param_grid=param_grid,
            cv=StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state),
            scoring="f1_macro",
            n_jobs=self.n_jobs,
            verbose=1,
        )
        search.fit(X_train, y_train)
        print(f"Best XGBoost params: {search.best_params_}")
        print(f"Best CV Macro F1: {search.best_score_:.4f}")
        return search

    @staticmethod
    def save_model(model, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, path)


if nn is not None:
    class FootwearMLP(nn.Module):
        """PyTorch MLP for three-class footwear success prediction."""

        def __init__(self, input_dim: int, dropout_rate: float = 0.30):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Dropout(dropout_rate),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.BatchNorm1d(64),
                nn.Dropout(dropout_rate),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.BatchNorm1d(32),
                nn.Linear(32, 3),
            )

        def forward(self, x):
            return self.network(x)
else:
    class FootwearMLP:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is not installed. Install it with: pip install torch")


class TorchMLPTrainer:
    """Train and evaluate the PyTorch MLP."""

    def __init__(
        self,
        input_dim: int,
        learning_rate: float = 0.001,
        dropout: float = 0.30,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        epochs: int = 100,
        early_stop_patience: int = 15,
        random_state: int = 42,
    ):
        if torch is None:
            raise ImportError("PyTorch is not installed. Install it with: pip install torch")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stop_patience = early_stop_patience
        self.random_state = random_state
        self.model = FootwearMLP(input_dim, dropout).to(self.device)
        self.history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": []}

    def _make_loader(self, X: np.ndarray, y: np.ndarray, shuffle: bool) -> DataLoader:
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)
        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray):
        train_loader = self._make_loader(X_train, y_train, shuffle=True)
        val_loader = self._make_loader(X_val, y_val, shuffle=False)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)

        best_val_f1 = -np.inf
        best_state = None
        patience_counter = 0

        for epoch in range(self.epochs):
            self.model.train()
            train_loss_sum = 0.0
            train_preds, train_labels = [], []

            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                outputs = self.model(xb)
                loss = criterion(outputs, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss_sum += loss.item() * len(yb)
                train_preds.extend(outputs.argmax(dim=1).detach().cpu().numpy())
                train_labels.extend(yb.detach().cpu().numpy())

            avg_train_loss = train_loss_sum / len(y_train)
            train_f1 = f1_score(train_labels, train_preds, average="macro")

            val_loss, val_f1 = self._validate(val_loader, y_val, criterion)
            self.history["train_loss"].append(avg_train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_f1"].append(train_f1)
            self.history["val_f1"].append(val_f1)
            scheduler.step(val_loss)

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch + 1:3d}/{self.epochs} | Train Loss: {avg_train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f}"
                )
            if patience_counter >= self.early_stop_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        print(f"Best Validation Macro F1: {best_val_f1:.4f}")
        return self

    def _validate(self, loader: DataLoader, y_val: np.ndarray, criterion) -> tuple[float, float]:
        self.model.eval()
        loss_sum = 0.0
        preds, labels = [], []
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                outputs = self.model(xb)
                loss = criterion(outputs, yb)
                loss_sum += loss.item() * len(yb)
                preds.extend(outputs.argmax(dim=1).cpu().numpy())
                labels.extend(yb.cpu().numpy())
        return loss_sum / len(y_val), f1_score(labels, preds, average="macro")

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            return self.model(X_tensor).argmax(dim=1).cpu().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            return torch.softmax(self.model(X_tensor), dim=1).cpu().numpy()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)


# ── K-Means Comparison ──────────────────────────────────────────────────────


class KMeansComparison:
    """Run K-Means clustering and compare with actual labels."""

    def __init__(self, n_clusters: int = 3, random_state: int = 42, n_init: int = 10):
        from sklearn.cluster import KMeans as _KMeans
        self.kmeans = _KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
        self.n_clusters = n_clusters

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        self.labels_ = self.kmeans.fit_predict(X)
        return self.labels_

    def evaluate(self, y_true: np.ndarray) -> dict[str, float]:
        from sklearn.metrics import adjusted_rand_score, silhouette_score
        ari = adjusted_rand_score(y_true, self.labels_)
        sil = silhouette_score(X=None, labels=self.labels_, metric="precomputed") if False else 0.0
        # compute silhouette on a sample for speed
        n = min(5000, len(y_true))
        rng = np.random.RandomState(42)
        idx = rng.choice(len(y_true), n, replace=False)
        # Need X for silhouette — caller should pass it
        return {"ARI": ari, "cluster_sizes": np.bincount(self.labels_).tolist()}

    def evaluate_with_X(self, X: np.ndarray, y_true: np.ndarray) -> dict[str, float]:
        from sklearn.metrics import adjusted_rand_score, silhouette_score
        ari = adjusted_rand_score(y_true, self.labels_)
        n = min(5000, len(y_true))
        rng = np.random.RandomState(42)
        idx = rng.choice(len(y_true), n, replace=False)
        sil = silhouette_score(X[idx], self.labels_[idx], random_state=42)
        return {
            "ARI": ari,
            "Silhouette": sil,
            "cluster_sizes": np.bincount(self.labels_).tolist(),
        }


# ── MLP Architecture Experiments ────────────────────────────────────────────


def run_mlp_architecture_experiments(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    input_dim: int,
    epochs: int = 80,
    batch_size: int = 256,
    random_state: int = 42,
) -> list[tuple[str, float, int]]:
    """Train multiple MLP variants and return (name, best_val_f1, n_params)."""
    if torch is None:
        print("Skipping MLP architecture experiments: PyTorch not installed.")
        return []

    variants = [
        ("Shallow (64)", [64], 0.3),
        ("Medium (128 -> 64)", [128, 64], 0.3),
        ("Deep (128 -> 64 -> 32)", [128, 64, 32], 0.3),
        ("Wide (256 -> 128 -> 64)", [256, 128, 64], 0.3),
        ("Deep + Low Dropout (0.1)", [128, 64, 32], 0.1),
        ("Deep + High Dropout (0.5)", [128, 64, 32], 0.5),
    ]

    torch.manual_seed(random_state)
    X_t = torch.FloatTensor(X_train)
    y_t = torch.LongTensor(y_train)
    X_v = torch.FloatTensor(X_val)
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)

    results = []
    print("MLP Architecture Comparison:")
    print("=" * 80)

    for name, hidden_layers, dropout_rate in variants:
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, h_dim), nn.ReLU(),
                nn.BatchNorm1d(h_dim), nn.Dropout(dropout_rate),
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 3))

        model = nn.Sequential(*layers)
        opt = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        crit = nn.CrossEntropyLoss()
        best_f1 = 0.0

        model.train()
        for epoch in range(epochs):
            for xb, yb in loader:
                out = model(xb)
                loss = crit(out, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()

            model.eval()
            with torch.no_grad():
                preds = model(X_v).argmax(dim=1).numpy()
                f1 = f1_score(y_val, preds, average="macro")
                if f1 > best_f1:
                    best_f1 = f1
            model.train()

        n_params = sum(p.numel() for p in model.parameters())
        print(f"  {name:45s} | Best Val F1: {best_f1:.4f} | Params: {n_params:,}")
        results.append((name, best_f1, n_params))

    return results
