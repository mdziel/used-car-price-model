"""E06 + E09: blend selection on out-of-fold predictions, then final fits on the development set,
one-time hold-out evaluation, and fair values for the hold-out ads.

Usage: python 04_final_fit.py <linear_model> <boosted_model>   e.g.  lasso lightgbm_mono
Inputs : results/oof_<name>.npy (from 02/03), best hyper-parameters in results/e03_e06_models_*.csv
Outputs: results/oof_blend.npy, results/holdout_pred_<name>.npy, results/e09_holdout.csv, results/blend_weights.csv
"""
import os
import glob
import json
import sys

import numpy as np
import pandas as pd
from patsy import dmatrix
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib  # noqa: E402

lin_name, gb_name = sys.argv[1], sys.argv[2]
df, names = lib.load()
des = lib.load_design()
dev = (des["holdout"] == 0).values
hold = ~dev
y = df["y"].values
F = json.load(open(lib.RES + "ols_formulas.json"))
EQ = F["EQ"]
cols_T = json.load(open(lib.RES + "tree_feature_names.json"))
XL = dmatrix(F["E"], df, return_type="dataframe", NA_action="raise").drop(columns=["Intercept"]).values
XT_df = lib.tree_features(df, names, EQ)
assert list(XT_df.columns) == cols_T
XT = XT_df.values
mono = [-1 if c in ("age_years", "km10k") else 0 for c in cols_T]

params = {}
for f in glob.glob(lib.RES + "e03_e06_models_*.csv"):
    for _, r in pd.read_csv(f).iterrows():
        params[r["model"]] = json.loads(r["best_params_full_dev"])
print("hyper-parameters from the full-development grid search:", {k: params[k] for k in (lin_name, gb_name)})


def build(name):
    p = params[name]
    if name == "ols_E":
        return LinearRegression(), XL
    if name == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=p["ridge__alpha"])), XL
    if name == "lasso":
        return make_pipeline(StandardScaler(), Lasso(alpha=p["lasso__alpha"], max_iter=20000)), XL
    kw = {k: (int(v) if float(v).is_integer() and k != "learning_rate" else v) for k, v in p.items()}
    if name.startswith("lightgbm"):
        return lgb.LGBMRegressor(n_jobs=4, verbose=-1, subsample_freq=1, subsample=0.8, colsample_bytree=0.7,
                                 random_state=lib.SEED, **({"monotone_constraints": mono} if name.endswith("mono") else {}), **kw), XT
    if name == "xgboost":
        return xgb.XGBRegressor(n_jobs=4, subsample=0.8, colsample_bytree=0.7, random_state=lib.SEED, verbosity=0, **kw), XT
    raise ValueError(name)


# ---------------------------------------------------------------- blend weight on out-of-fold predictions
oof = {n: np.load(lib.RES + f"oof_{n}.npy") for n in (lin_name, gb_name)} if lin_name != "ols_E" else {
    lin_name: np.load(lib.RES + "oof_ols_E.npy"), gb_name: np.load(lib.RES + f"oof_{gb_name}.npy")}
rows = []
for w in np.linspace(0, 1, 11):
    b = w * oof[lin_name] + (1 - w) * oof[gb_name]
    s = lib.summarise_oof(y, b, dev)
    rows.append({"weight_linear": round(w, 1), **s})
bw = pd.DataFrame(rows)
bw.to_csv(lib.RES + "blend_weights.csv", index=False)
print(bw[["weight_linear", "rmse_log", "mae_abs", "mape", "r2_log"]].round(4).to_string(index=False))
w_best = float(bw.loc[bw["rmse_log"].idxmin(), "weight_linear"])
w_use = 0.5 if abs(w_best - 0.5) <= 0.2 else w_best  # prefer the simple average unless clearly lopsided
print(f"best weight on linear = {w_best}; using {w_use}")
blend = w_use * oof[lin_name] + (1 - w_use) * oof[gb_name]
np.save(lib.RES + "oof_blend.npy", blend)

# ---------------------------------------------------------------- final fits on development set; hold-out evaluation
res, hp = [], {}
for n in (lin_name, gb_name):
    est, X = build(n)
    est.fit(X[dev], y[dev])
    p = np.full(len(df), np.nan)
    p[hold] = est.predict(X[hold])
    hp[n] = p
    np.save(lib.RES + f"holdout_pred_{n}.npy", p)
    m = lib.metrics(y[hold], p[hold])
    m["model"] = n
    res.append(m)
hb = w_use * hp[lin_name] + (1 - w_use) * hp[gb_name]
np.save(lib.RES + "holdout_pred_blend.npy", hb)
m = lib.metrics(y[hold], hb[hold])
m["model"] = f"blend ({w_use:.1f} {lin_name} + {1 - w_use:.1f} {gb_name})"
res.append(m)
for n in (lin_name, gb_name):
    m = lib.summarise_oof(y, oof[n], dev)
    m["model"] = f"{n} (CV, development)"
    res.append(m)
m = lib.summarise_oof(y, blend[None, :].repeat(1, axis=0) if blend.ndim == 1 else blend, dev)
m["model"] = "blend (CV, development)"
res.append(m)
out = pd.DataFrame(res)[["model", "rmse_log", "mae_log", "mae_abs", "mape", "r2_log", "bias_log", "n"]]
out.to_csv(lib.RES + "e09_holdout.csv", index=False)
print("\nHOLD-OUT (touched once) vs CV on development:")
print(out.round(4).to_string(index=False))
