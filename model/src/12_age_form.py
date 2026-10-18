"""E12: functional form of age. Spline (current) vs year dummies vs finer splines vs dummies by fuel group.
Same grouped CV as E02. Also prints the implied age-year profile: mean predicted log-price drop going from age a to a+1
with every other variable (incl. mileage) held at its actual value."""
import os
import json
import sys

import numpy as np
import pandas as pd
from patsy import dmatrix, build_design_matrices
from sklearn.linear_model import LinearRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib  # noqa: E402

df, names = lib.load()
des = lib.load_design()
dev = (des["holdout"] == 0).values
y = df["y"].values
F = json.load(open(lib.RES + "ols_formulas.json"))
df["age_yr"] = np.floor(df["age_years"]).clip(0, 10).astype(int)
E = F["E"]
assert lib.AGE_S in E
V = {
    "spline, 4 knots (current)": E,
    "year dummies (0,1,...,10+) + fuel x linear age": E.replace(lib.AGE_S, "C(age_yr)"),
    "year dummies, no fuel x age term": E.replace(lib.AGE_S, "C(age_yr)").replace(" + C(fuel_grp):age_years", ""),
    "finer spline (df = 9)": E.replace(lib.AGE_S, "bs(age_years, df=9, lower_bound=0, upper_bound=12)"),
    "year dummies x fuel group": E.replace(lib.AGE_S, "C(age_yr):C(fuel_grp) + C(age_yr)"),
    "spline + year dummies": E + " + C(age_yr)",
}
rows, prof = [], {}
for label, f in V.items():
    dm = dmatrix(f, df, return_type="dataframe", NA_action="raise")
    X = dm.values

    def fp(tr, te, X=X):
        return LinearRegression(fit_intercept=False).fit(X[tr], y[tr]).predict(X[te])

    oof, _ = lib.cv_oof(fp, des, len(df))
    per = [lib.metrics(y[dev], oof[r][dev])["rmse_log"] for r in range(lib.N_REPEATS)]
    rows.append({"variant": label, "rmse_log": float(np.mean(per)), "rmse_r0": per[0], "rmse_r1": per[1], "n_columns": X.shape[1]})
    print(f"{label:52s} rmse_log={np.mean(per):.4f} (r0 {per[0]:.4f}, r1 {per[1]:.4f}) cols={X.shape[1]}", flush=True)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    di = dm.design_info

    def level(a):
        fr = df.copy()
        fr["age_years"], fr["age_yr"] = a + 0.5, min(a, 10)
        return float((np.asarray(build_design_matrices([di], fr, return_type="dataframe")[0]) @ beta).mean())

    lv = np.array([level(a) for a in range(0, 11)])
    prof[label] = 100 * (np.exp(np.diff(lv)) - 1)
res = pd.DataFrame(rows)
res.to_csv(lib.RES + "e12_age_form.csv", index=False)
p = pd.DataFrame(prof, index=[f"{a}->{a + 1}" for a in range(0, 10)]).round(1)
p.to_csv(lib.RES + "e12_age_profile.csv")
print("\n% price change from age a to a+1 at fixed mileage (mean over all cars):")
print(p.to_string())
