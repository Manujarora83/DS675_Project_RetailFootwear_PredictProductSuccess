"""Shared constants for the footwear classification project."""
from __future__ import annotations

RANDOM_STATE = 42

TARGET_COLUMN = "product_success"

RAW_NUMERIC_FEATURES = [
    "base_price_usd",
    "discount_percent",
    "final_price_usd",
    "units_sold",
    "revenue_usd",
    "customer_rating",
    "size",
]

RAW_CATEGORICAL_FEATURES = [
    "brand",
    "category",
    "gender",
    "color",
    "payment_method",
    "sales_channel",
    "country",
    "customer_income_level",
]

DROP_COLUMNS = [
    "order_id",
    "order_date",
    "model_name",
    "final_price_usd",      # derived from base price and discount
    "revenue_usd",          # used to create target; keep out to avoid leakage
    "customer_rating",      # used to create target; keep out to avoid leakage
    "revenue_z",
    "rating_z",
    "success_score",
    TARGET_COLUMN,
]

CLASS_LABELS = ["Average", "High Performing", "Underperforming"]
