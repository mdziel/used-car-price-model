"""E11: do extra features help? Model year, owners/CO2/weight/engine size (missing allowed), title keywords (if the table has a
'title_spec' column), listing order (ad id), and trim x age interactions for the linear model. XGBoost with the tuned
hyper-parameters fixed (no re-tuning), same grouped folds.
Decision rule (set before running): adopt a feature group only if CV RMSE improves by at least 1% in both repeats."""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from patsy import dmatrix
from sklearn.linear_model import LinearRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib  # noqa: E402

warnings.filterwarnings("ignore")
df, names = lib.load()
des = lib.load_design()
dev = (des["holdout"] == 0).values
y = df["y"].values
F = json.load(open(lib.RES + "ols_formulas.json"))
EQ = F["EQ"]
base = lib.tree_features(df, names, EQ, extended=False)

extra = {
    "base": base,
    "+ model year": pd.concat([base, df[["model_year"]].astype(float)], axis=1),
    "+ owners, CO2, weight, engine size (NaN allowed)": pd.concat([base, df[["owners", "co2", "weight_kg", "engine_l"]].astype(float)], axis=1),
    "+ listing order (ad id)": pd.concat([base, df[["ad_id"]].astype(float)], axis=1),
}
specs = df[["model_year", "owners", "co2", "weight_kg", "engine_l"]].astype(float)
if "title_spec" in df.columns:
    full = lib.tree_features(df, names, EQ, extended=True)
    tok_cols = [c for c in full.columns if c.startswith("t_")]
    extra["+ title keywords"] = pd.concat([base, full[tok_cols]], axis=1)
    extra["+ all except listing order"] = pd.concat([base, specs, full[tok_cols]], axis=1)
else:
    extra["+ all except listing order"] = pd.concat([base, specs], axis=1)
P = dict(learning_rate=0.03, max_depth=5, min_child_weight=3, n_estimators=800)
rows = []
for label, X in extra.items():
    Xv = X.values

    def fp(tr, te, Xv=Xv):
        m = xgb.XGBRegressor(n_jobs=1, subsample=0.8, colsample_bytree=0.7, random_state=lib.SEED, verbosity=0, **P)
        return m.fit(Xv[tr], y[tr]).predict(Xv[te])

    oof, _ = lib.cv_oof(fp, des, len(df), n_jobs=10)
    per = [lib.metrics(y[dev], oof[r][dev])["rmse_log"] for r in range(lib.N_REPEATS)]
    rows.append({"model": "xgboost " + label, "rmse_log": float(np.mean(per)), "rmse_r0": per[0], "rmse_r1": per[1], "n_features": X.shape[1]})
    print(f"xgboost {label:55s} rmse_log={np.mean(per):.4f} (r0 {per[0]:.4f}, r1 {per[1]:.4f}) features={X.shape[1]}", flush=True)

# linear: trim x age interactions and engine x age
for label, extra_terms in [("OLS E (reference)", ""), ("OLS E + trim x age", " + C(trim_grp):age_years"),
                           ("OLS E + trim x age + trim x km", " + C(trim_grp):age_years + C(trim_grp):km10k"),
                           ("OLS E + engine x age", " + C(engine_grp):age_years")]:
    X = np.asarray(dmatrix(F["E"] + extra_terms, df, return_type="dataframe", NA_action="raise"))

    def fpl(tr, te, X=X):
        return LinearRegression(fit_intercept=False).fit(X[tr], y[tr]).predict(X[te])

    oof, _ = lib.cv_oof(fpl, des, len(df))
    per = [lib.metrics(y[dev], oof[r][dev])["rmse_log"] for r in range(lib.N_REPEATS)]
    rows.append({"model": label, "rmse_log": float(np.mean(per)), "rmse_r0": per[0], "rmse_r1": per[1], "n_features": X.shape[1]})
    print(f"{label:60s} rmse_log={np.mean(per):.4f} (r0 {per[0]:.4f}, r1 {per[1]:.4f}) cols={X.shape[1]}", flush=True)
res = pd.DataFrame(rows)
b = res.loc[res["model"] == "xgboost base", "rmse_log"].iloc[0]
res["change_vs_base_pct"] = 100 * (res["rmse_log"] / b - 1)
res.to_csv(lib.RES + "e11_extras.csv", index=False)
