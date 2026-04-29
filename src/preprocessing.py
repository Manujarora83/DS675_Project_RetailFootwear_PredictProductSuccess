"""Train/validation/test split and preprocessing pipeline."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class FootwearPreprocessor:
    """Create leakage-safe split and preprocessing transformer."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.numeric_features: list[str] = []
        self.categorical_features: list[str] = []
        self.transformer: ColumnTransformer | None = None
        self.feature_names_: list[str] = []

    def split(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        val_size: float = 0.15,
        test_size: float = 0.15,
    ):
        """Perform 70/15/15 stratified split by default."""
        temp_size = val_size + test_size
        relative_test_size = test_size / temp_size

        X_train, X_temp, y_train, y_temp = train_test_split(
            X,
            y,
            test_size=temp_size,
            random_state=self.random_state,
            stratify=y,
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=relative_test_size,
            random_state=self.random_state,
            stratify=y_temp,
        )
        return X_train, X_val, X_test, y_train, y_val, y_test

    def fit_transform(self, X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame):
        """Fit scaler/encoder on training data only, then transform all splits."""
        self.numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features = X_train.select_dtypes(include=["object", "string", "category"]).columns.tolist()

        self.transformer = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), self.numeric_features),
                (
                    "cat",
                    OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"),
                    self.categorical_features,
                ),
            ]
        )

        X_train_p = self.transformer.fit_transform(X_train)
        X_val_p = self.transformer.transform(X_val)
        X_test_p = self.transformer.transform(X_test)

        cat_names = self.transformer.named_transformers_["cat"].get_feature_names_out(
            self.categorical_features
        ).tolist()
        self.feature_names_ = self.numeric_features + cat_names
        return X_train_p, X_val_p, X_test_p
