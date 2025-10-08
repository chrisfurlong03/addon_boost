"""
Model training:
- M1: Propensity / ranking classifier (predicts P(purchase | context, addon))
- M2: Price / elasticity classifier (predicts P(purchase | price, context, addon))
Notes:
- Interaction features (price_x_days, price_x_pax) are CREATED BEFORE column selection
  to avoid KeyError during preprocessing.
- GroupKFold on booking_id prevents leakage across the same booking.
"""
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.pipeline import Pipeline

from .features import (
    CAT_BASE, ITEM_COL, NUMERIC, CATEGORICAL, PRICE_NUMERIC,
    TARGET, GROUP_KEY,
    preprocessor_propensity, preprocessor_price,
    assert_unique_columns,
)

# --- M1: Propensity (context + addon) ---
def train_propensity_model(df: pd.DataFrame) -> Pipeline:
    assert TARGET in df, "Missing target column"
    X_cols = CAT_BASE + ITEM_COL + NUMERIC
    assert_unique_columns(df, X_cols)

    X = df[X_cols].copy()
    y = df[TARGET].astype(int).values

    gkf = GroupKFold(n_splits=5)
    groups = df[GROUP_KEY]

    best_auc = -np.inf
    best_pipe: Pipeline | None = None

    for fold, (tr, va) in enumerate(gkf.split(X, y, groups=groups)):
        pipe = Pipeline([
            ("prep", preprocessor_propensity),
            ("clf", XGBClassifier(
                n_estimators=400,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                eval_metric="logloss",
                n_jobs=-1,
            )),
        ])
        pipe.fit(X.iloc[tr], y[tr])
        proba = pipe.predict_proba(X.iloc[va])[:, 1]
        auc = roc_auc_score(y[va], proba)
        ap = average_precision_score(y[va], proba)
        print(f"[M1][Fold {fold}] AUC={auc:.4f} AP={ap:.4f}")
        if auc > best_auc:
            best_auc = auc
            best_pipe = pipe

    print(f"[M1] Selected model AUC={best_auc:.4f}")
    assert best_pipe is not None
    return best_pipe

# --- M2: Price / Elasticity (context + addon + price/interactions) ---
def train_price_elasticity_model(df: pd.DataFrame) -> Pipeline:
    # Create interactions BEFORE selecting columns to avoid KeyError
    df_local = df.copy()
    df_local["price_x_days"] = df_local["price_offered"] * df_local["days_to_departure"].astype(float)
    df_local["price_x_pax"]  = df_local["price_offered"] * df_local["pax_count"].astype(float)

    X_cols = CATEGORICAL + PRICE_NUMERIC
    assert_unique_columns(df_local, X_cols)

    X = df_local[X_cols].copy()
    y = df_local[TARGET].astype(int).values

    gkf = GroupKFold(n_splits=5)
    groups = df_local[GROUP_KEY]

    best_auc = -np.inf
    best_pipe: Pipeline | None = None

    for fold, (tr, va) in enumerate(gkf.split(X, y, groups=groups)):
        pipe = Pipeline([
            ("prep", preprocessor_price),
            ("clf", XGBClassifier(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                eval_metric="logloss",
                n_jobs=-1,
            )),
        ])
        pipe.fit(X.iloc[tr], y[tr])
        proba = pipe.predict_proba(X.iloc[va])[:, 1]
        auc = roc_auc_score(y[va], proba)
        ap = average_precision_score(y[va], proba)
        print(f"[M2][Fold {fold}] AUC={auc:.4f} AP={ap:.4f}")
        if auc > best_auc:
            best_auc = auc
            best_pipe = pipe

    print(f"[M2] Selected model AUC={best_auc:.4f}")
    assert best_pipe is not None
    return best_pipe
