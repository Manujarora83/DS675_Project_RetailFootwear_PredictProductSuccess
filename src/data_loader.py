"""Data loading and target creation."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import LabelEncoder

from .config import DROP_COLUMNS, TARGET_COLUMN


class FootwearDataLoader:
    """Load the CSV and create the product-success classification target."""

    def __init__(self, data_path: Path):
        self.data_path = Path(data_path)

    def load(self) -> pd.DataFrame:
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.data_path}")
        df = pd.read_csv(self.data_path)
        return df

    @staticmethod
    def create_target(df: pd.DataFrame) -> pd.DataFrame:
        """Create product_success from z-scored revenue and customer rating.

        The target follows the original notebook logic:
        revenue_z and rating_z are averaged, then quantile-binned into
        Underperforming / Average / High Performing.
        """
        df = df.copy()
        df["revenue_z"] = stats.zscore(df["revenue_usd"])
        df["rating_z"] = stats.zscore(df["customer_rating"])
        df["success_score"] = (df["revenue_z"] + df["rating_z"]) / 2
        df[TARGET_COLUMN] = pd.qcut(
            df["success_score"],
            q=3,
            labels=["Underperforming", "Average", "High Performing"],
        )
        return df

    @staticmethod
    def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Engineer features and remove leaky columns."""
        df = df.copy()
        df["order_date"] = pd.to_datetime(df["order_date"])
        df["order_year"] = df["order_date"].dt.year
        df["order_month"] = df["order_date"].dt.month
        df["order_quarter"] = df["order_date"].dt.quarter
        df["price_sensitivity"] = df["base_price_usd"] * df["discount_percent"]

        existing_drop_cols = [col for col in DROP_COLUMNS if col in df.columns]
        X = df.drop(columns=existing_drop_cols)
        y = df[TARGET_COLUMN]
        return X, y

    @staticmethod
    def encode_target(y: pd.Series) -> tuple[np.ndarray, LabelEncoder]:
        encoder = LabelEncoder()
        return encoder.fit_transform(y), encoder
