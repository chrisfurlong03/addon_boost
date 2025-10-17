from typing import List
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CAT_BASE: List[str] = ["route_od", "payment_type", "loyalty_tier", "season"]
ITEM_COL: List[str] = ["addon_id"]
CATEGORICAL: List[str] = CAT_BASE + ITEM_COL

NUMERIC: List[str] = [
    "flight_duration_min",
    "dep_hour_local",
    "pax_count",
    "days_to_departure",
    "purchased_any_addon",
    "used_upgrade",
]

PRICE_NUMERIC: List[str] = NUMERIC + [
    "price_offered", "price_list", "discount_pct", "price_x_days", "price_x_pax"
]

TARGET = "label_purchase"
GROUP_KEY = "booking_id"

preprocessor_propensity = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("num", StandardScaler(), NUMERIC),
    ]
)

preprocessor_price = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("num", StandardScaler(), PRICE_NUMERIC),
    ]
)

def assert_unique_columns(df: pd.DataFrame, cols: List[str]) -> None:
    idx = pd.Index(cols)
    if not idx.is_unique:
        dupes = idx[idx.duplicated()].tolist()
        raise AssertionError(f"Duplicate column selections detected: {dupes}")