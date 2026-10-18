"""E09/E10 outputs: fair price for every ad, 80% interval, price-vs-fair gap, dealer premium, flags,
value-for-money rankings and the assessment of two ads.

Fair price = car-only prediction (blend) that never saw the ad: out-of-fold for development ads, development-fit for hold-out ads.
Writes results/fair_values.csv, results/rank_best_value.csv, results/rank_worst_value.csv, results/comps_two_ads.csv.
"""
import os
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib  # noqa: E402

df, names = lib.load()
des = lib.load_design()
dev = (des["holdout"] == 0).values
y = df["y"].values
pred = np.where(dev, np.load(lib.RES + "oof_blend.npy").mean(axis=0), np.load(lib.RES + "holdout_pred_blend.npy"))
assert not np.isnan(pred).any()
res = y - pred
print("residual mean/sd on dev:", round(res[dev].mean(), 4), round(res[dev].std(), 4), "| hold-out:", round(res[~dev].mean(), 4), round(res[~dev].std(), 4))

# ------------------------------------------------------------ 80% interval: empirical residual quantiles by fuel group x predicted-price band
df["pred"] = pred
df["resid"] = res
# Residual spread is not constant: it is about twice as large for 200,000+ km cars as for cars under 50,000 km
# (explore_resid.py). A small log-linear variance model on |residual| gives each ad its own scale; intervals and
# z-scores use standardised residuals.
import statsmodels.api as sm  # noqa: E402
from patsy import dmatrix  # noqa: E402
Xv = dmatrix("bs(age_years, df=4) + bs(km10k, df=4) + C(fuel_grp) + is_private + bs(pred, df=3)", df, return_type="dataframe")
vfit = sm.OLS(np.log(np.abs(res[dev]) + 0.003), Xv[dev]).fit()
scale = np.exp(vfit.predict(Xv).values)
scale *= np.sqrt(np.mean((res[dev] / scale[dev]) ** 2))  # unit variance of the standardised residual on development ads
zres = res / scale
q10, q90 = np.quantile(zres[dev], [0.10, 0.90])
df["scale"] = scale
df["fair"] = np.exp(pred)
df["fair_lo"] = np.exp(pred + q10 * scale)  # about 10% of asks are below this
df["fair_hi"] = np.exp(pred + q90 * scale)
df["gap_log"] = pred - y  # > 0: asking price below the car-only fair value
df["gap_pct"] = 100 * (df["fair"] / df["price"] - 1)  # how much above the ask the fair value is
df["z"] = df["gap_log"] / df["scale"]
print("typical residual scale: <50k km:", round(float(np.median(scale[df['km'] < 50000])), 4), "| 100-150k:", round(float(np.median(scale[(df['km'] >= 100000) & (df['km'] < 150000)])), 4), "| >200k:", round(float(np.median(scale[df['km'] > 200000])), 4))
df["gap_rank_pct"] = df["gap_log"].rank(pct=True) * 100

# ------------------------------------------------------------ dealer premium (shrunk) from E07
dt = pd.read_csv(lib.RES + "e07_dealer_effects.csv")[["seller_id", "premium_pct", "n_ads", "dealer_effect_sd"]].rename(
    columns={"premium_pct": "dealer_premium_pct", "n_ads": "dealer_n_ads"})
df = df.merge(dt, on="seller_id", how="left")
df["dealer_premium_pct"] = df["dealer_premium_pct"].fillna(0.0)
df["fair_dealer_adj"] = df["fair"] * (1 + df["dealer_premium_pct"] / 100)
df["gap_dealer_pct"] = 100 * (df["fair_dealer_adj"] / df["price"] - 1)  # gap relative to what this dealer usually charges

# ------------------------------------------------------------ flags that explain apparent bargains
df["tuned_yes"] = df["tuned_yes"] if "tuned_yes" in df.columns else 0
for _c in ["known_damage_yes", "flag_damage_weak", "mileage_fixed", "age_from_model_year", "flag_reserved"]:
    if _c not in df.columns:
        df[_c] = 0
flag_cols = {"known_damage_yes": "seller reports damage", "flag_damage_weak": "text mentions defect/spare parts",
             "mileage_fixed": "mileage unit corrected", "age_from_model_year": "unreliable registration date",
             "tuned_yes": "tuned engine", "flag_reserved": "reserved/sold wording"}
df["flags"] = df.apply(lambda r: "; ".join(v for k, v in flag_cols.items() if r[k] == 1), axis=1)
df["credible"] = (df[["known_damage_yes", "flag_damage_weak", "mileage_fixed", "age_from_model_year", "tuned_yes"]].sum(axis=1) == 0)
df["link"] = df["ad_id"].astype(str)
df["split"] = np.where(dev, "dev (out-of-fold)", "hold-out")

keep = ["ad_id", "link", "trim_grp", "model_year", "age_years", "km", "fuel_grp", "engine_grp", "trim_grp", "power_hp", "price", "fair", "fair_lo",
        "fair_hi", "gap_pct", "z", "gap_rank_pct", "dealer_premium_pct", "dealer_n_ads", "fair_dealer_adj", "gap_dealer_pct", "seller_name", "location",
        "n_eq_ticked", "flags", "credible", "split"]
keep = [c for c in dict.fromkeys(keep) if c in df.columns]
df[keep].to_csv(lib.RES + "fair_values.csv", index=False, encoding="utf-8-sig")
print("fair values written:", len(df), "| share of asks inside the 80% interval:", round(float(((df["price"] >= df["fair_lo"]) & (df["price"] <= df["fair_hi"]))[dev].mean()), 3),
      "(dev) /", round(float(((df["price"] >= df["fair_lo"]) & (df["price"] <= df["fair_hi"]))[~dev].mean()), 3), "(hold-out)")

# ------------------------------------------------------------ rankings
pool = df[df["credible"]]
print("credible ads:", len(pool), "of", len(df), "| excluded by flags:", int((~df["credible"]).sum()))
cols = ["ad_id", "link", "trim_grp", "model_year", "km", "price", "fair", "gap_pct", "z", "fair_lo", "fair_hi", "dealer_premium_pct", "gap_dealer_pct",
        "seller_name", "location", "flags", "split"]
cols = [c for c in dict.fromkeys(cols) if c in df.columns]
# companion lists: (a) gap relative to the dealer's usual pricing, (b) petrol / mild-hybrid cars, 2-8 years, 40-150k km
alt = pool.sort_values("gap_dealer_pct", ascending=False).head(20)[cols]
alt.to_csv(lib.RES + "rank_best_value_dealer_adjusted.csv", index=False, encoding="utf-8-sig")
like = pool[pool["fuel_grp"].isin(["Petrol", "HybridPetrol"]) & pool["age_years"].between(2, 8) & pool["km"].between(40000, 150000)]
likebest = like.sort_values("gap_log", ascending=False).head(10)[cols]
likebest.to_csv(lib.RES + "rank_best_value_similar_cars.csv", index=False, encoding="utf-8-sig")
print("\nbest value among petrol / mild-hybrid cars, 2-8 yrs, 40-150k km, pool of", len(like), ":")
print(likebest[["ad_id", "trim_grp", "model_year", "km", "price", "fair", "gap_pct", "z", "dealer_premium_pct", "seller_name"]].round(1).to_string(index=False))
best = pool.sort_values("gap_log", ascending=False).head(20)[cols]
worst = pool.sort_values("gap_log", ascending=True).head(10)[cols]
best.to_csv(lib.RES + "rank_best_value.csv", index=False, encoding="utf-8-sig")
worst.to_csv(lib.RES + "rank_worst_value.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 250, "display.max_colwidth", 34)
show = ["ad_id", "trim_grp", "model_year", "km", "price", "fair", "gap_pct", "z", "dealer_premium_pct", "seller_name"]
print("\nTOP 20 best value (credible ads):")
print(best[show].round(1).to_string(index=False))
print("\nTOP 10 worst value:")
print(worst[show].round(1).to_string(index=False))
# how many of the extremes are flagged ones
allbest = df.sort_values("gap_log", ascending=False).head(20)
print("\nof the 20 largest gaps overall, flagged (not credible):", int((~allbest["credible"]).sum()))

# ------------------------------------------------------------ the two ads of interest
ids = [int(a) for a in os.environ.get("ADS_OF_INTEREST", "").split(",") if a]  # optional: ad ids to examine
print("\n=== ads of interest (set ADS_OF_INTEREST) ===")
comps = []
for i in ids:
    r = df[df["ad_id"] == i].iloc[0]
    print(f"\nad {i}: {r['trim_grp']} | {int(r['model_year'])}, {r['km']:.0f} km, {r['fuel_grp']} {r['engine_grp']} {r['trim_grp']}")
    print(f"  ask {r['price']:,.0f} | fair {r['fair']:,.0f} (80% interval {r['fair_lo']:,.0f}-{r['fair_hi']:,.0f}) | gap {r['gap_pct']:+.1f}% | z {r['z']:+.2f} | "
          f"percentile of gap {r['gap_rank_pct']:.0f} | dealer premium {r['dealer_premium_pct']:+.1f}% (n={r['dealer_n_ads']:.0f}) | dealer-adjusted fair {r['fair_dealer_adj']:,.0f} | {r['split']}")
    s = df[(df["fuel_grp"] == r["fuel_grp"]) & (df["ad_id"] != i)].copy()
    s["dist"] = np.sqrt(((s["age_years"] - r["age_years"]) / 1.0) ** 2 + ((s["km10k"] - r["km10k"]) / 2.5) ** 2 + ((s["power_hp"] - r["power_hp"]) / 40) ** 2
                        + 1.5 * (s["trim_grp"] != r["trim_grp"]) + 0.8 * (s["engine_grp"] != r["engine_grp"]))
    c = s.nsmallest(6, "dist")[["ad_id", "trim_grp", "model_year", "km", "price", "fair", "gap_pct", "seller_name"]].copy()
    c.insert(0, "for_ad", i)
    comps.append(c)
    print(c.drop(columns=["for_ad"]).round(0).to_string(index=False))
if comps:
    pd.concat(comps).to_csv(lib.RES + "comps_ads_of_interest.csv", index=False, encoding="utf-8-sig")
