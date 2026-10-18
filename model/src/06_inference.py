"""E08 (part 1): structural OLS (feature set E) on all ads, seller-clustered standard errors.
Marginal effects of age, mileage, equipment and warranty with delta-method and cluster-bootstrap intervals.
Writes results/e08_*.csv."""
import os
import json
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from patsy import dmatrix, build_design_matrices

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib  # noqa: E402

df, names = lib.load()
F = json.load(open(lib.RES + "ols_formulas.json"))
y = df["y"].values
grp = pd.factorize(df["seller_id"])[0]
dm = dmatrix(F["E"], df, return_type="dataframe", NA_action="raise")
cols = list(dm.columns)
X = dm.values
fit = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": grp})
cov = fit.cov_params()
beta = fit.params
print(f"OLS E on all {len(df)} ads: R2={fit.rsquared:.4f}, resid sd={np.sqrt(fit.scale):.4f}, rank={np.linalg.matrix_rank(X)} of {X.shape[1]}")
info = dm.design_info


def design_for(frame):
    return np.asarray(build_design_matrices([info], frame, return_type="dataframe", NA_action="raise")[0])


def effect(frame0, frame1):
    """Mean change in log price between two versions of the same cars, with delta-method sd."""
    g = (design_for(frame1) - design_for(frame0)).mean(axis=0)
    est = float(g @ beta)
    sd = float(np.sqrt(g @ cov @ g))
    return est, sd


def pct(est, sd=None):
    lo = hi = np.nan
    if sd is not None:
        lo, hi = 100 * (np.exp(est - 1.96 * sd) - 1), 100 * (np.exp(est + 1.96 * sd) - 1)
    return 100 * (np.exp(est) - 1), lo, hi


rows = []
base = df[(df["age_years"] <= 11) & (df["km10k"] <= 38)].copy()


def shift(frame, dage=0.0, dkm=0.0):
    f = frame.copy()
    f["age_years"] = f["age_years"] + dage
    f["km10k"] = f["km10k"] + dkm
    return f


def add(label, sub, dage, dkm):
    e, sd = effect(sub, shift(sub, dage, dkm))
    p, lo, hi = pct(e, sd)
    rows.append({"effect": label, "n_cars": len(sub), "pct_change": p, "ci95_low": lo, "ci95_high": hi, "log_change": e, "se_log": sd})


add("+1 year age, mileage fixed (all cars)", base, 1, 0)
add("+10,000 km, age fixed (all cars)", base, 0, 1)
add("+1 year and +15,000 km (a typical year of ownership)", base, 1, 1.5)
for fg in ["PHEV", "Diesel", "Petrol", "HybridPetrol"]:
    s = base[base["fuel_grp"] == fg]
    add(f"+1 year age, mileage fixed — {fg}", s, 1, 0)
    add(f"+10,000 km, age fixed — {fg}", s, 0, 1)
    add(f"+1 year and +15,000 km — {fg}", s, 1, 1.5)
for lo_, hi_ in [(0, 2), (2, 4), (4, 6), (6, 8), (8, 11)]:
    s = base[(base["age_years"] > lo_) & (base["age_years"] <= hi_)]
    add(f"+1 year age, mileage fixed — cars aged {lo_}-{hi_}", s, 1, 0)
    add(f"+10,000 km, age fixed — cars aged {lo_}-{hi_}", s, 0, 1)
eff = pd.DataFrame(rows)

# cluster bootstrap for the headline effects (sellers resampled)
rng = np.random.RandomState(lib.SEED)
sellers = df["seller_id"].values
uniq = np.unique(sellers)
idx_by = {s: np.where(sellers == s)[0] for s in uniq}
G_age = (design_for(shift(base, 1, 0)) - design_for(base)).mean(axis=0)
G_km = (design_for(shift(base, 0, 1)) - design_for(base)).mean(axis=0)
boot = []
for b in range(300):
    pick = rng.choice(uniq, len(uniq), replace=True)
    ii = np.concatenate([idx_by[s] for s in pick])
    bb = np.linalg.lstsq(X[ii], y[ii], rcond=None)[0]
    boot.append((G_age @ bb, G_km @ bb))
boot = np.array(boot)
for j, nm in enumerate(["+1 year age, mileage fixed (all cars)", "+10,000 km, age fixed (all cars)"]):
    lo_, hi_ = np.percentile(boot[:, j], [2.5, 97.5])
    eff.loc[eff["effect"] == nm, "boot_ci95_low"] = 100 * (np.exp(lo_) - 1)
    eff.loc[eff["effect"] == nm, "boot_ci95_high"] = 100 * (np.exp(hi_) - 1)
eff.to_csv(lib.RES + "e08_age_km_effects.csv", index=False)
print(eff.round(2).to_string(index=False))

# ---------------------------------------------------------------- equipment and warranty coefficients
rows = []
share = df[F["EQ"]].mean()
for k in F["EQ"]:
    j = cols.index(k)
    b, se = beta[j], np.sqrt(cov[j, j])
    rows.append({"feature": names[k], "prevalence": share[k], "pct_effect": 100 * (np.exp(b) - 1), "se_pct": 100 * se * np.exp(b),
                 "ci95_low": 100 * (np.exp(b - 1.96 * se) - 1), "ci95_high": 100 * (np.exp(b + 1.96 * se) - 1), "t": b / se})
eq = pd.DataFrame(rows)
from scipy import stats  # noqa: E402
eq["p"] = 2 * (1 - stats.norm.cdf(np.abs(eq["t"])))
# Benjamini-Hochberg q-values
o = eq["p"].argsort().values
q = np.empty(len(eq))
ranked = eq["p"].values[o] * len(eq) / (np.arange(len(eq)) + 1)
q[o] = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1)
eq["q_bh"] = q
eq = eq.sort_values("t", key=np.abs, ascending=False)
eq.to_csv(lib.RES + "e08_equipment_effects.csv", index=False)
print("\nTop equipment effects (OLS E, % of price, cluster SE):")
print(eq.head(25)[["feature", "prevalence", "pct_effect", "ci95_low", "ci95_high", "q_bh"]].round(3).to_string(index=False))
print("features with q<0.10:", int((eq["q_bh"] < 0.10).sum()), "of", len(eq), "| expected false discoveries at q<0.10 ~", round(0.10 * (eq["q_bh"] < 0.10).sum(), 1))

wc = [c for c in cols if c.startswith(("warranty_", "I(n_equip", "service_", "right_to", "known_damage", "I(dist", "is_private", "vat_ded"))]
rows = []
for c in wc:
    j = cols.index(c)
    b, se = beta[j], np.sqrt(cov[j, j])
    rows.append({"term": c, "coef_log": b, "pct_effect_per_unit": 100 * (np.exp(b) - 1), "ci95_low": 100 * (np.exp(b - 1.96 * se) - 1),
                 "ci95_high": 100 * (np.exp(b + 1.96 * se) - 1), "t": b / se})
w = pd.DataFrame(rows)
w.to_csv(lib.RES + "e08_warranty_seller_effects.csv", index=False)
print("\nwarranty / seller-style / location terms:")
print(w.round(3).to_string(index=False))

# trim and engine terms
tt = [c for c in cols if c.startswith(("C(trim_grp)", "C(engine_grp)", "C(colour_grp)", "C(fuel_grp)[T"))]
rows = []
for c in tt:
    j = cols.index(c)
    b, se = beta[j], np.sqrt(cov[j, j])
    rows.append({"term": c, "pct_effect": 100 * (np.exp(b) - 1), "ci95_low": 100 * (np.exp(b - 1.96 * se) - 1), "ci95_high": 100 * (np.exp(b + 1.96 * se) - 1), "t": b / se})
pd.DataFrame(rows).to_csv(lib.RES + "e08_trim_engine_effects.csv", index=False)
print("\ntrim / engine / colour / fuel terms (vs base level):")
print(pd.DataFrame(rows).round(2).to_string(index=False))

# depreciation schedule for reference cars (typical mileage 15,000 km per year)
ref = []
for fg in ["PHEV", "Diesel", "Petrol", "HybridPetrol"]:
    s = df[(df["fuel_grp"] == fg)]
    row = s.iloc[[0]].copy()
    sched = {}
    for a in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        f = s.copy()
        f["age_years"], f["km10k"] = a, 1.5 * a
        sched[a] = float(np.exp(design_for(f) @ beta).mean() if False else np.exp((design_for(f) @ beta)).mean())
    ref.append({"fuel": fg, **{f"age{a}": round(v) for a, v in sched.items()}})
pd.DataFrame(ref).to_csv(lib.RES + "e08_depreciation_schedule.csv", index=False)
print("\nmean predicted price of the fuel group's own cars set to age a, 15,000 km/yr:")
print(pd.DataFrame(ref).to_string(index=False))
