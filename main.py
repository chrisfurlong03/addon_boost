import pandas as pd
from .config import Policy, PRICE_BUCKETS, ADDON_META
from .data_gen import generate_synthetic_training
from .models import train_propensity_model, train_price_elasticity_model
from .optimizer import optimize_offers

ADDON_CANDIDATES = list(ADDON_META.keys())

def run_demo():
    # 1) Data
    df = generate_synthetic_training(n_bookings=4000)
    print("Synthetic training head:\n", df.head())

    # 2) Train
    propensity_model = train_propensity_model(df)
    price_model = train_price_elasticity_model(df)

    addon_costs = {k: v["cost"] for k, v in ADDON_META.items()}

    # 3) Inference context (no single price_list; we require per-add-on price_list_map)
    ctx = pd.DataFrame([{
        "booking_id": "B_demo",
        "route_od": "ORD_SFO",
        "flight_duration_min": 270,
        "dep_hour_local": 9,
        "pax_count": 2,
        "days_to_departure": 14,
        "payment_type": "credit_card",
        "loyalty_tier": "Gold",
        "season": "Q4",
        "purchased_any_addon": 0,
        "used_upgrade": 0,
    }])

    policy = Policy(min_margin_pct=0.1, max_discount_pct=0.5)

    # REQUIRED: per-add-on list prices (demo uses base_price as list ceiling)
    price_list_map = {k: float(v["base_price"]) for k, v in ADDON_META.items()}

    offers = optimize_offers(
        context_rows=ctx,
        propensity_model=propensity_model,
        price_model=price_model,
        price_grid=PRICE_BUCKETS,
        policy=policy,
        addon_costs=addon_costs,
        addon_candidates=ADDON_CANDIDATES,
        top_k=2,
        list_price_map=price_list_map,
    )

    print("Suggested offers (ranked by purchase probability):")
    for o in offers:
        print(o)

if __name__ == "__main__":
    run_demo()
