from dataclasses import dataclass
from typing import Dict

RNG_SEED = 42
PRICE_BUCKETS = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0]

ADDON_META: Dict[str, Dict[str, float]] = {
    "seat_upgrade": {"base_price": 30.0, "cost": 3.0},
    "baggage_bundle": {"base_price": 20.0, "cost": 1.5},
    "lounge_access": {"base_price": 25.0, "cost": 5.0},
    "wifi": {"base_price": 15.0, "cost": 0.6},
    "priority_boarding": {"base_price": 10.0, "cost": 0.2},
}

@dataclass
class Policy:
    min_margin_pct: float = 0.1
    max_discount_pct: float = 0.5
    fairness_block_cc_specific: bool = True