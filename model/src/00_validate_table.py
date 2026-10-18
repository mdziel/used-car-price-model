"""Check that a modelling table has the columns and value sets the pipeline expects (see model/INPUT_FORMAT.md).
Run:  python 00_validate_table.py [path/to/table.csv]      (default: data/example_table.csv)"""
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + "/"
path = sys.argv[1] if len(sys.argv) > 1 else ROOT + "data/example_table.csv"
df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)

REQUIRED = {
    "ad_id": "unique id", "price": "asking price (> 0)", "model_year": "int", "age_years": "years, >= 0", "km": "odometer in km",
    "power_hp": "engine power", "fuel": "one of the fuel labels below", "drive": "'All-wheel drive' or another label",
    "engine_code": "category label", "trim": "category label ('unknown' allowed)", "colour": "category label",
    "seller_id": "grouping key for dealers (private ads: one id per ad)", "is_private": "0/1", "group_id": "dealer-group label or empty",
    "dist_big3": "distance (km) to the nearest big city", "vat_deductible": "0/1", "n_equip_raw": "count", "n_eq_ticked": "count",
    "service_followed": "0/1", "right_to_exchange": "0/1", "known_damage_yes": "0/1",
    "warranty_incl_months": "months or empty", "warranty_option_max_months": "months or empty", "warranty_mentioned": "0/1",
    "warranty_certified": "0/1", "warranty_trade_assoc": "0/1", "warranty_option_offered": "0/1",
    "owners": "count or empty", "co2": "g/km or empty", "weight_kg": "kg or empty", "engine_l": "litres or empty",
}
OPTIONAL = {"title_spec": "free-text title (adds title keywords to the boosted models)", "seats": "", "doors": "", "boot_l": "", "tow_kg": "",
            "flag_damage_weak": "0/1", "mileage_fixed": "0/1", "age_from_model_year": "0/1", "tuned_yes": "0/1", "location": "any label"}
FUELS = {"Plug-in hybrid petrol", "Diesel", "Petrol", "Hybrid petrol (mild)", "Hybrid diesel"}
problems = []
for c, why in REQUIRED.items():
    if c not in df.columns:
        problems.append(f"missing column: {c} ({why})")
eq = [c for c in df.columns if c.startswith("eq_")]
if len(eq) < 10:
    problems.append(f"expected equipment flag columns eq_<feature> (0/1); found {len(eq)}")
if not problems:
    if not df["ad_id"].is_unique:
        problems.append("ad_id is not unique")
    if not (df["price"] > 0).all():
        problems.append("price must be > 0")
    bad = set(df["fuel"].dropna().unique()) - FUELS
    if bad:
        problems.append(f"unknown fuel labels: {sorted(bad)} (expected {sorted(FUELS)})")
    for c in ["is_private", "vat_deductible", "service_followed", "right_to_exchange", "known_damage_yes", *eq]:
        if not set(df[c].dropna().unique()) <= {0, 1}:
            problems.append(f"{c} must be 0/1")
    if df["seller_id"].isna().any():
        problems.append("seller_id has missing values")
print(f"{path}: {len(df)} rows, {df.shape[1]} columns, {len(eq)} equipment flags")
print("optional columns present:", [c for c in OPTIONAL if c in df.columns])
if problems:
    print("PROBLEMS:")
    for p in problems:
        print(" -", p)
    sys.exit(1)
print("OK")
