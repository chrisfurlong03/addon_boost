from typing import Dict, List
import numpy as np
import pandas as pd
from .config import RNG_SEED, ADDON_META

RNG = np.random.default_rng(RNG_SEED)

ROUTES = ["ORD_SFO", "SFO_ORD", "LAX_JFK", "JFK_LAX", "IAH_DEN", "DEN_IAH", "EWR_MCO", "MCO_EWR"]
PAYMENT_TYPES = ["credit_card", "points", "mixed"]
TIERS = ["None", "Silver", "Gold", "Platinum"]
SEASONS = ["Q1", "Q2", "Q3", "Q4"]

AFFINITY = {
    "seat_upgrade": 0.28,
    "baggage_bundle": 0.18,
    "lounge_access": 0.12,
    "wifi": 0.22,
    "priority_boarding": 0.16,
}

ELASTICITY = {
    "seat_upgrade": 0.045,
    "baggage_bundle": 0.055,
    "lounge_access": 0.050,
    "wifi": 0.040,
    "priority_boarding": 0.065,
}

def _route_mix(route: str) -> float:
    if route in {"LAX_JFK", "JFK_LAX"}:
        return 0.25
    if route.startswith("EWR") or route.endswith("EWR"):
        return 0.15
    return 0.1

def generate_synthetic_training(n_bookings: int = 4000, price_jitter: float = 0.3) -> pd.DataFrame:
    rows: List[Dict] = []
    for i in range(n_bookings):
        b = f"B{100000+i}"
        route = RNG.choice(ROUTES)
        flight_duration_min = int(RNG.normal(210, 60))
        dep_hour_local = int(RNG.integers(5, 22))
        pax_count = int(RNG.integers(1, 5))
        days_to_departure = int(np.clip(RNG.normal(21, 14), 0, 120))
        payment_type = RNG.choice(PAYMENT_TYPES, p=[0.65, 0.25, 0.10])
        loyalty_tier = RNG.choice(TIERS, p=[0.5, 0.25, 0.18, 0.07])
        season = RNG.choice(SEASONS)
        purchased_any_addon = int(RNG.random() < 0.25)
        used_upgrade = int(RNG.random() < 0.12)

        for addon_id, meta in ADDON_META.items():
            list_price = meta["base_price"] * (1.0 + 0.2 * RNG.normal(0, 1))
            offered_price = max(1.0, list_price * (1.0 - price_jitter*RNG.random()))
            discount_pct = (list_price - offered_price) / max(list_price, 1e-6)

            base = -1.2
            base += AFFINITY[addon_id]
            base += 0.002 * (flight_duration_min - 180)
            base += 0.08 * (pax_count - 1)
            base += -0.02 * (days_to_departure - 14) / 7
            base += 0.25 if payment_type == "points" else 0.0
            base += 0.20 if loyalty_tier in {"Gold", "Platinum"} else 0.0
            base += 0.15 if purchased_any_addon else 0.0
            base += 0.12 if used_upgrade and addon_id == "lounge_access" else 0.0
            base += _route_mix(route)

            price_term = -ELASTICITY[addon_id] * offered_price
            price_term += 0.002 * offered_price * (days_to_departure < 7)
            price_term += -0.002 * offered_price * (payment_type == "points")

            logit = base + price_term
            prob = 1 / (1 + np.exp(-logit))
            label_purchase = int(RNG.random() < prob)

            rows.append({
                "booking_id": b,
                "addon_id": addon_id,
                "label_purchase": label_purchase,
                "price_offered": float(offered_price),
                "price_list": float(list_price),
                "discount_pct": float(discount_pct),
                "route_od": route,
                "flight_duration_min": float(flight_duration_min),
                "dep_hour_local": int(dep_hour_local),
                "pax_count": int(pax_count),
                "days_to_departure": int(days_to_departure),
                "payment_type": payment_type,
                "loyalty_tier": loyalty_tier,
                "season": season,
                "purchased_any_addon": int(purchased_any_addon),
                "used_upgrade": int(used_upgrade),
            })
    return pd.DataFrame(rows)