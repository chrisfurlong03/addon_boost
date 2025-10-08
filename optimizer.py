"""
Offer optimization:
- Enforces policy guardrails:
  • price <= per-add-on list price (from required list_price_map)
  • max discount and min margin
- One price per add-on: pick the bucket with highest predicted purchase probability
- Rank add-ons by probability and return top_k
"""
from dataclasses import dataclass
from typing import Dict, List
import pandas as pd

from .features import CAT_BASE, ITEM_COL, NUMERIC, CATEGORICAL, PRICE_NUMERIC
from .config import Policy


@dataclass
class AddonOffer:
    addon_id: str
    price: float
    predicted_prob: float
    expected_profit: float  # kept for telemetry; ranking uses probability


def feasible(policy: Policy, list_price: float, offer_price: float, cost: float = 0.0) -> bool:
    # Do not exceed list
    if offer_price > list_price:
        return False
    # Respect max discount
    discount_pct = (list_price - offer_price) / max(list_price, 1e-6)
    if discount_pct > policy.max_discount_pct:
        return False
    # Respect min margin
    margin_pct = (offer_price - cost) / max(offer_price, 1e-6)
    return margin_pct >= policy.min_margin_pct


def optimize_offers(
    context_rows: pd.DataFrame,
    propensity_model,   # sklearn Pipeline (currently unused for ranking but available)
    price_model,        # sklearn Pipeline
    price_grid: List[float],
    policy: Policy,
    addon_costs: Dict[str, float],
    addon_candidates: List[str],
    top_k: int = 2,
    list_price_map: Dict[str, float] | None = None,
) -> List[AddonOffer]:
    """
    One price per add-on (max probability), then rank add-ons by probability.
    `list_price_map` is REQUIRED and must contain entries for all candidate add-ons.
    """
    if not list_price_map:
        raise ValueError("list_price_map is required and cannot be empty")

    missing = [a for a in addon_candidates if a not in list_price_map]
    if missing:
        raise ValueError(f"list_price_map missing add-ons: {missing}")

    per_addon_best: Dict[str, AddonOffer] = {}

    for addon in addon_candidates:
        row = context_rows.copy()
        row["addon_id"] = addon

        list_price = float(list_price_map[addon])
        cost = addon_costs.get(addon, 0.0)
        best_offer: AddonOffer | None = None

        for p in price_grid:
            if not feasible(policy, list_price=list_price, offer_price=p, cost=cost):
                continue

            # M1 design (not used for ranking now, placeholder for later blending)
            _ = row[CAT_BASE + ITEM_COL + NUMERIC].copy()

            # M2 design with interactions (ensure columns exist before selection)
            X2 = row[(CAT_BASE + ITEM_COL)].copy()
            X2 = X2.assign(
                price_offered=p,
                price_list=list_price,
                discount_pct=(list_price - p) / max(list_price, 1e-6),
                price_x_days=p * float(row["days_to_departure"].iloc[0]),
                price_x_pax=p * float(row["pax_count"].iloc[0]),
                flight_duration_min=float(row["flight_duration_min"].iloc[0]),
                dep_hour_local=int(row["dep_hour_local"].iloc[0]),
                pax_count=int(row["pax_count"].iloc[0]),
                days_to_departure=int(row["days_to_departure"].iloc[0]),
                purchased_any_addon=int(row["purchased_any_addon"].iloc[0]),
                used_upgrade=int(row["used_upgrade"].iloc[0]),
            )
            X2 = X2[(CATEGORICAL + PRICE_NUMERIC)]

            prob = float(price_model.predict_proba(X2)[:, 1][0])
            offer = AddonOffer(
                addon_id=addon,
                price=p,
                predicted_prob=prob,
                expected_profit=prob * (p - cost),
            )

            if (best_offer is None) or (offer.predicted_prob > best_offer.predicted_prob):
                best_offer = offer

        if best_offer is not None:
            per_addon_best[addon] = best_offer

    # Rank by purchase probability (desc)
    offers = sorted(per_addon_best.values(), key=lambda o: o.predicted_prob, reverse=True)
    return offers[:top_k]
