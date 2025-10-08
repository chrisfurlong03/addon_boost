import os
from typing import Any, Dict, List

import pandas as pd
from flask import Flask, jsonify, request

from .config import Policy, PRICE_BUCKETS, ADDON_META
from .data_gen import generate_synthetic_training
from .models import train_price_elasticity_model, train_propensity_model
from .optimizer import optimize_offers

app = Flask(__name__)

# Lazy, in-memory models for demo purposes
PROP_MODEL = None
PRICE_MODEL = None
ADDON_COSTS = {k: v["cost"] for k, v in ADDON_META.items()}
ADDON_CANDIDATES = list(ADDON_META.keys())

def get_models():
    global PROP_MODEL, PRICE_MODEL
    if PROP_MODEL is None or PRICE_MODEL is None:
        n = int(os.getenv("TRAIN_N_BOOKINGS", "3000"))
        df = generate_synthetic_training(n_bookings=n)
        PROP_MODEL = train_propensity_model(df)
        PRICE_MODEL = train_price_elasticity_model(df)
    return PROP_MODEL, PRICE_MODEL

@app.post("/warmup")
def warmup():
    get_models()
    return jsonify({"status": "warmed"}), 200

def _validate_context(ctx: Dict[str, Any]):
    required = [
        "booking_id",
        "route_od",
        "flight_duration_min",
        "dep_hour_local",
        "pax_count",
        "days_to_departure",
        "payment_type",
        "loyalty_tier",
        "season",
        "purchased_any_addon",
        "used_upgrade",
    ]
    missing = [k for k in required if k not in ctx]
    if missing:
        return False, f"Missing fields: {missing}"
    return True, None

@app.post("/recommend")
def recommend():
    try:
        payload = request.get_json(force=True, silent=False) or {}
        ctx = payload.get("context", {})
        ok, err = _validate_context(ctx)
        if not ok:
            return jsonify({"error": err}), 400

        # REQUIRED: price_list_map must exist and cover all add-ons requested
        raw_map = payload.get("price_list_map")
        if not isinstance(raw_map, dict) or not raw_map:
            return jsonify({"error": "price_list_map is required and must be a non-empty object"}), 400

        addons: List[str] = list(map(str, payload.get("addons") or ADDON_CANDIDATES))
        price_list_map = {str(k): float(v) for k, v in raw_map.items() if v is not None}
        missing = [a for a in addons if a not in price_list_map]
        if missing:
            return jsonify({"error": f"price_list_map missing add-ons: {missing}"}), 400

        top_k = int(payload.get("top_k", 2))
        price_buckets = payload.get("price_buckets") or PRICE_BUCKETS
        policy_dict = payload.get("policy") or {}
        policy = Policy(
            min_margin_pct=float(policy_dict.get("min_margin_pct", Policy.min_margin_pct)),
            max_discount_pct=float(policy_dict.get("max_discount_pct", Policy.max_discount_pct)),
            fairness_block_cc_specific=bool(policy_dict.get("fairness_block_cc_specific", True)),
        )

        # Build one-row context DataFrame with proper types
        row = {
            "booking_id": str(ctx.get("booking_id")),
            "route_od": str(ctx.get("route_od")),
            "flight_duration_min": float(ctx.get("flight_duration_min")),
            "dep_hour_local": int(ctx.get("dep_hour_local")),
            "pax_count": int(ctx.get("pax_count")),
            "days_to_departure": int(ctx.get("days_to_departure")),
            "payment_type": str(ctx.get("payment_type")),
            "loyalty_tier": str(ctx.get("loyalty_tier")),
            "season": str(ctx.get("season")),
            "purchased_any_addon": int(ctx.get("purchased_any_addon")),
            "used_upgrade": int(ctx.get("used_upgrade")),
        }
        context_df = pd.DataFrame([row])

        prop, price = get_models()
        offers = optimize_offers(
            context_rows=context_df,
            propensity_model=prop,
            price_model=price,
            price_grid=list(map(float, price_buckets)),
            policy=policy,
            addon_costs=ADDON_COSTS,
            addon_candidates=addons,
            top_k=top_k,
            list_price_map=price_list_map,
        )

        return jsonify(
            {
                "offers": [o.__dict__ for o in offers],
                "meta": {
                    "top_k": top_k,
                    "candidates_considered": len(addons),
                    "price_buckets": price_buckets,
                    "list_price_map_used": True,
                },
            }
        ), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    if os.getenv("WARMUP_ON_START", "1") == "1":
        get_models()
    app.run(host="0.0.0.0", port=port, debug=bool(int(os.getenv("DEBUG", "0"))))
