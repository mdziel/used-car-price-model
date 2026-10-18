"""E08 (part 2): tree-model view of the same effects. Fit the chosen boosted model on all ads, then
 - finite-difference age and mileage effects (compare with OLS in 06_inference.py),
 - SHAP: importance by feature group and per-ad explanations for ads of interest,
 - equipment effect = mean SHAP when listed minus when not listed (compare with OLS).
Usage: python 06b_shap.py <boosted_model>     e.g. lightgbm_mono
"""
import os
import glob
import json
import sys

import numpy as np
import pandas as pd
import shap
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib  # noqa: E402

name = sys.argv[1]
df, names = lib.load()
y = df["y"].values
F = json.load(open(lib.RES + "ols_formulas.json"))
EQ = F["EQ"]
cols_T = json.load(open(lib.RES + "tree_feature_names.json"))
XT_df = lib.tree_features(df, names, EQ)
assert list(XT_df.columns) == cols_T
mono = [-1 if c in ("age_years", "km10k") else 0 for c in cols_T]
p = {}
for f in glob.glob(lib.RES + "e03_e06_models_*.csv"):
    for _, r in pd.read_csv(f).iterrows():
        p[r["model"]] = json.loads(r["best_params_full_dev"])
kw = {k: (int(v) if float(v).is_integer() and k != "learning_rate" else v) for k, v in p[name].items()}
if name.startswith("lightgbm"):
    model = lgb.LGBMRegressor(n_jobs=6, verbose=-1, subsample_freq=1, subsample=0.8, colsample_bytree=0.7, random_state=lib.SEED,
                              **({"monotone_constraints": mono} if name.endswith("mono") else {}), **kw)
else:
    import xgboost as xgb
    model = xgb.XGBRegressor(n_jobs=6, subsample=0.8, colsample_bytree=0.7, random_state=lib.SEED, verbosity=0, **kw)
model.fit(XT_df, y)
print("fitted", name, kw)

# ---- finite-difference age / km effects
ok = (df["age_years"] <= 11) & (df["km10k"] <= 38)
base = XT_df[ok].copy()
f0 = model.predict(base)
rows = []
for label, d in [("+1 year age, mileage fixed", {"age_years": 1}), ("+10,000 km, age fixed", {"km10k": 1}),
                 ("+1 year and +15,000 km", {"age_years": 1, "km10k": 1.5})]:
    b = base.copy()
    for k, v in d.items():
        b[k] = b[k] + v
    if "age_years" in d and "model_year" in b.columns:
        b["model_year"] = b["model_year"] - d["age_years"]  # one year older also means one model year earlier
    rows.append({"effect": label, "boosting_pct": 100 * (np.exp((model.predict(b) - f0).mean()) - 1)})
fd = pd.DataFrame(rows)
fd.to_csv(lib.RES + "e08_boosting_age_km_effects.csv", index=False)
print(fd.round(2).to_string(index=False))

# ---- SHAP
ex = shap.TreeExplainer(model)
sv = ex.shap_values(XT_df)
base_val = float(np.ravel(ex.expected_value)[0])
groups = {}
for c in cols_T:
    if c in ("age_years",):
        g = "age"
    elif c in ("km10k",):
        g = "mileage"
    elif c == "power_hp":
        g = "power"
    elif c.startswith(("fuel_grp", "engine_grp", "awd")):
        g = "fuel / engine / drive"
    elif c.startswith("trim_grp"):
        g = "trim"
    elif c.startswith("colour_grp"):
        g = "colour"
    elif c in ("gen1", "my_ge2022"):
        g = "generation"
    elif c.startswith("warranty_"):
        g = "warranty"
    elif c in ("dist_big3", "lat", "lon"):
        g = "location"
    elif c in ("n_equip_raw", "n_eq_ticked", "service_followed", "right_to_exchange", "known_damage_yes"):
        g = "seller style / listing quality"
    elif c in ("is_private", "vat_deductible"):
        g = "seller type / VAT"
    elif c == "model_year":
        g = "model year"
    elif c in ("owners", "co2", "weight_kg", "engine_l"):
        g = "other specs (owners, CO2, weight, engine size)"
    elif c.startswith("t_"):
        g = "title keywords"
    else:
        g = "equipment (individual ticks)"
    groups[c] = g
gdf = pd.DataFrame(sv, columns=cols_T)
by_group = pd.DataFrame({g: gdf[[c for c in cols_T if groups[c] == g]].sum(axis=1) for g in sorted(set(groups.values()))})
imp = by_group.abs().mean().sort_values(ascending=False)
imp.to_csv(lib.RES + "e08_shap_group_importance.csv", header=["mean_abs_shap_log"])
print("\nmean |SHAP| by group (log points):")
print(imp.round(4).to_string())

# equipment: mean SHAP listed vs not listed
rows = []
for k in EQ:
    on = XT_df[k] == 1
    if on.sum() >= 20 and (~on).sum() >= 20:
        rows.append({"feature": names[k], "prevalence": float(on.mean()), "shap_pct_listed_vs_not": 100 * (np.exp(gdf.loc[on, k].mean() - gdf.loc[~on, k].mean()) - 1)})
pd.DataFrame(rows).sort_values("shap_pct_listed_vs_not", key=np.abs, ascending=False).to_csv(lib.RES + "e08_shap_equipment.csv", index=False)

# per-ad explanations (grouped), for the two ads: explained with a model that did NOT see these two ads
# (the model fitted on all ads reproduces their asking price almost exactly, which would hide the question)
TWO = [int(a) for a in os.environ.get("ADS_OF_INTEREST", "").split(",") if a]  # optional: ad ids to explain
keep_rows = ~df["ad_id"].isin(TWO).values
model2 = type(model)(**model.get_params())
model2.fit(XT_df[keep_rows], y[keep_rows])
ex2 = shap.TreeExplainer(model2)
sv2_all = ex2.shap_values(XT_df)
base_val = float(np.ravel(ex2.expected_value)[0])
sv = sv2_all  # rows of the two ads are now out-of-sample
gdf2 = pd.DataFrame(sv2_all, columns=cols_T)
by_group = pd.DataFrame({g: gdf2[[c for c in cols_T if groups[c] == g]].sum(axis=1) for g in sorted(set(groups.values()))})
out = []
for i in TWO:
    j = int(np.where(df["ad_id"].values == i)[0][0])
    pred_log = base_val + sv[j].sum()
    tot = by_group.iloc[j].sort_values(key=np.abs, ascending=False)
    print(f"\nad {i}: model price {np.exp(pred_log):,.0f} (average car {np.exp(base_val):,.0f}); asking {df['price'].iloc[j]:,.0f}")
    for g, v in tot.items():
        print(f"   {g:32s} {100 * (np.exp(v) - 1):+6.1f}%")
        out.append({"ad_id": i, "group": g, "effect_pct": 100 * (np.exp(v) - 1), "log": v})
    fe = pd.Series(sv[j], index=cols_T)
    eqf = fe[[c for c in cols_T if groups[c] == "equipment (individual ticks)"]]
    eqf.index = [names[c] for c in eqf.index]
    print("   equipment ticks, largest:", {k: f"{100 * (np.exp(v) - 1):+.1f}%" for k, v in eqf.reindex(eqf.abs().sort_values(ascending=False).index).head(5).items()})
pd.DataFrame(out).to_csv(lib.RES + "e08_shap_two_ads.csv", index=False)
