"""
Lightweight offline eval helpers.

These are proxy metrics for quick iteration — for production you should add:
- Policy value estimation with IPS / Doubly Robust (needs logged propensities)
- Calibrated probability checks (e.g., Brier, calibration curves)
- Per-segment metrics (route, cabin, tier, channel)
"""
import pandas as pd
from .features import TARGET


def offline_value_estimate(df: pd.DataFrame) -> float:
    """
    Naive realized revenue per impression from historical logs.
    NOTE: This is *not* counterfactual-safe — use only as a sanity metric.
    """
    revenue = (df[TARGET] * df["price_offered"]).mean()
    print(f"[OFFLINE] Avg revenue per impression: {revenue:.2f}")
    return float(revenue)
