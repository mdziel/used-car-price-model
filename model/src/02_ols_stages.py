"""E01 + E02: baselines and staged OLS feature sets under repeated grouped CV (development set only)."""
import os
import json
import sys

import numpy as np
import pandas as pd
from patsy import dmatrix
from sklearn.linear_model import LinearRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib  # noqa: E402

df, names = lib.load()
des = lib.load_design()
assert (des["ad_id"].values == df["ad_id"].values).all()
dev = (des["holdout"] == 0).values
y = df["y"].values
EQ = lib.equipment_cols(df[dev], names)
print("equipment features with 2-98% prevalence:", len(EQ), "of", len(names))
eq_terms = " + ".join(EQ)
WARR = "warranty_incl_ge12 + warranty_incl_ge24 + warranty_certified + warranty_trade_assoc + warranty_option_offered + warranty_mentioned"
SELLER_STYLE = "I(n_equip_raw / 100) + service_followed + right_to_exchange + known_damage_yes"
LOC = "I(dist_big3 / 100)"

A = f"{lib.AGE_S} + {lib.KM_S} + power_hp"
B = A + " + C(fuel_grp) + awd + C(engine_grp) + gen1 + my_ge2022 + C(fuel_grp):age_years + C(fuel_grp):km10k"
C_ = B + " + C(trim_grp) + C(colour_grp) + is_private + vat_deductible"
D = C_ + " + " + eq_terms
E = D + " + " + WARR + " + " + SELLER_STYLE + " + " + LOC
SETS = [
    ("B1 mean by model year", "C(model_year)"),
    ("B2 OLS age+km+power (linear)", "age_years + km10k + power_hp"),
    ("A  splines age,km + power", A),
    ("B  + fuel, engine, drive, generation (+ fuel x age/km)", B),
    ("C  + trim, colour, seller type, VAT", C_),
    ("D  + equipment ticks", D),
    ("E  + warranty, seller style, location", E),
]

rows, oofs = [], {}
for label, f in SETS:
    X = np.asarray(dmatrix(f, df, return_type="dataframe", NA_action="raise"))
    rank = np.linalg.matrix_rank(X)

    def fit_predict(tr, te, X=X):
        m = LinearRegression(fit_intercept=False).fit(X[tr], y[tr])
        return m.predict(X[te])

    oof, jobs = lib.cv_oof(fit_predict, des, len(df))
    s = lib.summarise_oof(y, oof, dev)
    fold_rmse = [float(np.sqrt(np.mean((oof[r][te] - y[te]) ** 2))) for r, k, te in jobs]
    s.update({"model": label, "n_columns": X.shape[1], "rank": rank, "fold_rmse_sd": float(np.std(fold_rmse))})
    rows.append(s)
    oofs[label] = oof
    print(f"{label:58s} rmse_log={s['rmse_log']:.4f} (sd rep {s['rmse_sd_repeats']:.4f}) mae_abs={s['mae_abs']:.0f} mape={s['mape']:.3f} r2={s['r2_log']:.4f} cols={X.shape[1]} rank={rank}")
res = pd.DataFrame(rows)[["model", "rmse_log", "rmse_sd_repeats", "fold_rmse_sd", "mae_log", "mae_abs", "mape", "r2_log", "bias_log", "n", "n_columns", "rank"]]
res.to_csv(lib.RES + "e01_e02_ols_cv.csv", index=False)
np.save(lib.RES + "oof_ols_E.npy", oofs[SETS[-1][0]])
json.dump({"A": A, "B": B, "C": C_, "D": D, "E": E, "EQ": EQ}, open(lib.RES + "ols_formulas.json", "w"))

# error by segment for the richest set
best = oofs[SETS[-1][0]].mean(axis=0)
seg = df.loc[dev, ["fuel_grp", "age_years", "price"]].copy()
seg["err"] = best[dev] - y[dev]
seg["age_band"] = pd.cut(seg["age_years"], [0, 2, 4, 6, 8, 12])
seg["price_band"] = pd.qcut(seg["price"], 5)
for k in ["fuel_grp", "age_band", "price_band"]:
    print(f"\nRMSE(log) of set E by {k}:")
    print(seg.groupby(k, observed=True)["err"].agg(lambda e: float(np.sqrt(np.mean(e ** 2)))).round(4).to_dict())
