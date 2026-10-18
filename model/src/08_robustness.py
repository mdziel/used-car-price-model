"""E10: robustness of the headline effects (OLS set E) to sample and specification choices.
For each variant: refit, report +1 year (mileage fixed), +10,000 km (age fixed), a few equipment/warranty terms, and the
grouped-CV error on the same subset. Writes results/e10_robustness.csv."""
import os
import json
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from patsy import build_design_matrices, dmatrix
from sklearn.linear_model import LinearRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib  # noqa: E402

df, names = lib.load()
des = lib.load_design()
df["fold"] = des["fold_r0"].values
df["dev"] = (des["holdout"] == 0).values
F = json.load(open(lib.RES + "ols_formulas.json"))
nm2key = {v: k for k, v in names.items()}
focus = ["Head-up display", "Air suspension", "Sunroof / glass roof", "Panoramic roof", "360° camera", "Electric seats with memory"]
focus_keys = [nm2key[f] for f in focus]
tow = df.copy()
fair = pd.read_csv(lib.RES + "fair_values.csv")
df = df.merge(fair[["ad_id", "credible"]], on="ad_id", how="left")


def fit_variant(sub, label, cluster="seller_id", formula=None):
    sub = sub.reset_index(drop=True)
    formula = formula or F["E"]
    dm = dmatrix(formula, sub, return_type="dataframe", NA_action="raise")
    X, y = dm.values, sub["y"].values
    grp = pd.factorize(sub[cluster])[0]
    fit = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": grp})
    beta, cov, cols = fit.params, fit.cov_params(), list(dm.columns)
    base = sub[(sub["age_years"] <= 11) & (sub["km10k"] <= 38)]

    def D(frame):
        return np.asarray(build_design_matrices([dm.design_info], frame, return_type="dataframe", NA_action="raise")[0])

    out = {"variant": label, "n_ads": len(sub), "n_sellers": sub["seller_id"].nunique()}
    for nme, da, dk in [("age_pct_per_year", 1, 0), ("km_pct_per_10k", 0, 1)]:
        f1 = base.copy()
        f1["age_years"] += da
        f1["km10k"] += dk
        g = (D(f1) - D(base)).mean(axis=0)
        e, sd = float(g @ beta), float(np.sqrt(g @ cov @ g))
        out[nme] = 100 * (np.exp(e) - 1)
        out[nme + "_se"] = 100 * sd * np.exp(e)
    for f_, k in zip(focus, focus_keys):
        if k in cols:
            j = cols.index(k)
            out[f"eq: {f_}"] = 100 * (np.exp(beta[j]) - 1)
    for c in ["warranty_incl_ge12", "warranty_incl_ge24"]:
        if c in cols:
            out[c] = 100 * (np.exp(beta[cols.index(c)]) - 1)
    # grouped CV on the subset (development rows, first repeat's folds)
    m = sub["dev"].values & (sub["fold"].values >= 0)
    pr = np.full(len(sub), np.nan)
    for k in range(lib.N_SPLITS):
        te, tr = m & (sub["fold"].values == k), m & (sub["fold"].values != k)
        if te.sum() and tr.sum():
            pr[te] = LinearRegression(fit_intercept=False).fit(X[tr], y[tr]).predict(X[te])
    out["cv_rmse_log"] = float(np.sqrt(np.nanmean((pr[m] - y[m]) ** 2)))
    return out


rows = [fit_variant(df, "baseline: all ads")]
rows.append(fit_variant(df[df["is_private"] == 0], "dealers only (drop private)"))
rows.append(fit_variant(df[df["gen1"] == 0], "drop first generation (model year <= 2017)"))
rows.append(fit_variant(df[df["fuel_grp"] != "PHEV"], "drop plug-in hybrids"))
rows.append(fit_variant(df[df["fuel_grp"] == "PHEV"], "plug-in hybrids only"))
rows.append(fit_variant(df[df["credible"] == True], "drop flagged ads (damage, unit fixes, ...)"))  # noqa: E712
rows.append(fit_variant(df[df["mileage_fixed"] == 0], "drop ads with corrected mileage"))
rows.append(fit_variant(df[df["age_from_model_year"] == 0], "drop ads with unreliable registration date"))
rows.append(fit_variant(df[df["km"] < 250000], "drop cars above 250,000 km"))
rows.append(fit_variant(df[df["age_years"] > 1.0], "drop cars younger than 1 year"))
rows.append(fit_variant(df, "standard errors clustered by dealer group instead of seller", cluster="chain_or_seller"))
# winsorise the target: clip log price residuals of the baseline fit at the 1st/99th percentile, refit on clipped y
dm0 = dmatrix(F["E"], df, return_type="dataframe")
b0 = np.linalg.lstsq(dm0.values, df["y"].values, rcond=None)[0]
r0 = df["y"].values - dm0.values @ b0
lo, hi = np.quantile(r0, [0.01, 0.99])
dfw = df.copy()
dfw["y"] = dm0.values @ b0 + np.clip(r0, lo, hi)
rows.append(fit_variant(dfw, "winsorised target (residuals clipped at 1%/99%)"))
# simpler functional form: linear age/km instead of splines (shows what the spline buys)
Elin = F["E"].replace(lib.AGE_S, "age_years").replace(lib.KM_S, "km10k")
rows.append(fit_variant(df, "linear age and km (no splines)", formula=Elin))
res = pd.DataFrame(rows)
res.to_csv(lib.RES + "e10_robustness.csv", index=False)
pd.set_option("display.width", 250, "display.max_columns", 20)
print(res.round(2).to_string(index=False))
