"""Figures for the report -> model/results/figures/*.png (reads results/*.csv, so it can be re-run after any change)."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib  # noqa: E402

FIG = lib.RES + "figures/"
os.makedirs(FIG, exist_ok=True)
INK, MUTED, GRID = "#1f2937", "#6b7280", "#e5e7eb"
C = {"blue": "#2563eb", "teal": "#0d9488", "amber": "#d97706", "red": "#dc2626", "violet": "#7c3aed", "grey": "#9ca3af"}
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.7, "figure.dpi": 130, "savefig.bbox": "tight", "font.family": "DejaVu Sans"})
df, names = lib.load()
des = lib.load_design()
dev = (des["holdout"] == 0).values
fv = pd.read_csv(lib.RES + "fair_values.csv")

# ---- 1. model comparison (CV on development set)
rows = [("Mean by model year", 0.1441, "base"), ("OLS: age, km, power", 0.1123, "base"), ("OLS: + splines", 0.0975, "base"),
        ("OLS: + fuel, engine, drive", 0.0789, "lin"), ("OLS: + trim, colour, seller type", 0.0622, "lin"),
        ("OLS: + equipment ticks", 0.0601, "lin"), ("OLS: + warranty, seller style, location", 0.0593, "lin"),
        ("Lasso", 0.0591, "lin"), ("Random forest", 0.0649, "tree"), ("HistGradientBoosting", 0.0592, "tree"),
        ("XGBoost (base features)", 0.0574, "tree"), ("LightGBM (base features)", 0.0578, "tree"),
        ("LightGBM monotone (base)", 0.0590, "tree")]
ext = pd.read_csv(lib.RES + "e03_e06_models_lightgbm.csv").iloc[0]["rmse_log"]
extx = pd.read_csv(lib.RES + "e03_e06_models_xgboost.csv").iloc[0]["rmse_log"]
bw = pd.read_csv(lib.RES + "blend_weights.csv")
blend = float(bw.loc[bw["weight_linear"] == 0.5, "rmse_log"].iloc[0])
rows += [("XGBoost (extended features)", extx, "tree"), ("LightGBM (extended features)", ext, "tree"), ("Blend: lasso + LightGBM (final)", blend, "final")]
fig, ax = plt.subplots(figsize=(7.2, 5.2))
colmap = {"base": C["grey"], "lin": C["blue"], "tree": C["teal"], "final": C["amber"]}
yy = np.arange(len(rows))[::-1]
ax.barh(yy, [r[1] * 100 for r in rows], color=[colmap[r[2]] for r in rows], height=0.7)
for y_, r in zip(yy, rows):
    ax.text(r[1] * 100 + 0.15, y_, f"{r[1] * 100:.2f}%", va="center", fontsize=8.5, color=INK)
ax.set_yticks(yy)
ax.set_yticklabels([r[0] for r in rows], fontsize=8.5)
ax.set_xlabel("Out-of-fold error, RMSE of log price (≈ % of price) — lower is better")
ax.set_xlim(0, 16.5)
ax.grid(axis="y", visible=False)
ax.set_title("Cross-validated accuracy (development ads, folds grouped by seller)", loc="left", fontsize=10.5)
fig.savefig(FIG + "fig1_model_comparison.png")
plt.close(fig)

# ---- 2. fair vs asking price + residual spread
fig, axs = plt.subplots(1, 2, figsize=(10.2, 4.3), gridspec_kw={"width_ratios": [1.15, 1]})
ax = axs[0]
for split, col, lab in [("dev (out-of-fold)", C["blue"], "development (out-of-fold)"), ("hold-out", C["amber"], "hold-out (478 ads)")]:
    s = fv[fv["split"] == split]
    ax.scatter(s["fair"] / 1000, s["price"] / 1000, s=6, alpha=0.45, color=col, label=lab, linewidths=0)
lim = [90, 900]
ax.plot(lim, lim, color=INK, lw=0.9)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(lim); ax.set_ylim(lim)
ticks = [100, 150, 200, 300, 450, 700]
ax.set_xticks(ticks); ax.set_yticks(ticks)
ax.set_xticklabels([str(t) for t in ticks]); ax.set_yticklabels([str(t) for t in ticks])
from matplotlib.ticker import NullFormatter  # noqa: E402
ax.xaxis.set_minor_formatter(NullFormatter()); ax.yaxis.set_minor_formatter(NullFormatter())
ax.set_xlabel("Model fair price (thousand price units)"); ax.set_ylabel("Asking price (thousand price units)")
ax.legend(frameon=False, loc="upper left", fontsize=8.5, markerscale=2.5)
ax.set_title("Fair value vs asking price", loc="left", fontsize=10.5)
ax = axs[1]
r = (np.log(fv["price"]) - np.log(fv["fair"])) * 100
ax.hist(r[fv["split"] == "dev (out-of-fold)"], bins=60, range=(-30, 30), color=C["blue"], alpha=0.75, density=True, label="development")
ax.hist(r[fv["split"] == "hold-out"], bins=40, range=(-30, 30), histtype="step", color=C["amber"], lw=1.6, density=True, label="hold-out")
ax.set_xlabel("Asking price vs fair value (log points ≈ %)"); ax.set_yticks([])
ax.legend(frameon=False, fontsize=8.5)
ax.set_title("Pricing errors", loc="left", fontsize=10.5)
fig.tight_layout()
fig.savefig(FIG + "fig2_fair_vs_ask.png")
plt.close(fig)

# ---- 3. age / km effects by fuel group
e = pd.read_csv(lib.RES + "e08_age_km_effects.csv")
groups = [("All cars", ""), ("PHEV", "— PHEV"), ("Diesel", "— Diesel"), ("Petrol", "— Petrol"), ("Hybrid petrol", "— HybridPetrol")]
fig, axs = plt.subplots(1, 3, figsize=(10.4, 3.5), sharey=False)
for ax, (title, pat_a, pat_k, pat_t) in zip(axs, [("+1 year of age (mileage fixed)", "+1 year age, mileage fixed", None, None),
                                                    ("+10,000 km (age fixed)", "+10,000 km, age fixed", None, None),
                                                    ("A typical year: +1 year and +15,000 km", "+1 year and +15,000 km", None, None)]):
    vals, lows, highs, labs = [], [], [], []
    for lab, suf in groups:
        nm = pat_a + (" (all cars)" if suf == "" and pat_a != "+1 year and +15,000 km" else "")
        if suf == "":
            row = e[e["effect"].str.startswith(pat_a) & e["effect"].str.contains("all cars|typical year")]
        else:
            row = e[e["effect"] == f"{pat_a} {suf}"]
        if len(row):
            row = row.iloc[0]
            vals.append(row["pct_change"]); lows.append(row["ci95_low"]); highs.append(row["ci95_high"]); labs.append(lab)
    yy = np.arange(len(vals))[::-1]
    ax.errorbar(vals, yy, xerr=[np.array(vals) - np.array(lows), np.array(highs) - np.array(vals)], fmt="o", color=C["blue"], ecolor=C["blue"], capsize=3, ms=5)
    ax.set_yticks(yy); ax.set_yticklabels(labs if ax is axs[0] else [""] * len(labs))
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_xlabel("% change in price"); ax.set_title(title, loc="left", fontsize=9.5)
    ax.grid(axis="y", visible=False)
fig.tight_layout()
fig.savefig(FIG + "fig3_age_km_effects.png")
plt.close(fig)

# ---- 4. depreciation schedule
d = pd.read_csv(lib.RES + "e08_depreciation_schedule.csv")
fig, ax = plt.subplots(figsize=(7, 4))
for (_, r), col in zip(d.iterrows(), [C["blue"], C["teal"], C["amber"], C["violet"]]):
    ages = np.arange(1, 11)
    ax.plot(ages, [r[f"age{a}"] / 1000 for a in ages], color=col, lw=2, marker="o", ms=3.5, label={"PHEV": "Plug-in hybrid", "Diesel": "Diesel", "Petrol": "Petrol", "HybridPetrol": "Hybrid petrol"}[r["fuel"]])
ax.set_xlabel("Age (years), mileage 15,000 km per year"); ax.set_ylabel("Predicted price (thousand price units)")
ax.set_xticks(range(1, 11)); ax.set_ylim(200, 650)
ax.legend(frameon=False, fontsize=8.5)
ax.set_title("Model price by age (own-group average car; mix of trims/model years differs by group)", loc="left", fontsize=9.5)
fig.savefig(FIG + "fig4_depreciation.png")
plt.close(fig)

# ---- 5. equipment effects
q = pd.read_csv(lib.RES + "e08_equipment_effects.csv")
q = q[q["prevalence"].between(0.03, 0.97)].copy()
top = q.reindex(q["t"].abs().sort_values(ascending=False).index).head(16).sort_values("pct_effect")
fig, ax = plt.subplots(figsize=(7.4, 5.2))
yy = np.arange(len(top))
cols = [C["blue"] if v < 0.10 else C["grey"] for v in top["q_bh"]]
ax.hlines(yy, top["ci95_low"], top["ci95_high"], color=cols, lw=1.6)
ax.scatter(top["pct_effect"], yy, color=cols, s=26, zorder=3)
ax.set_yticks(yy); ax.set_yticklabels([f"{f} ({p:.0%})" for f, p in zip(top["feature"], top["prevalence"])], fontsize=8.5)
ax.axvline(0, color=INK, lw=0.8)
ax.set_xlabel("Price effect of the feature being listed, % (95% interval)")
ax.grid(axis="y", visible=False)
ax.set_title("Equipment with the strongest evidence (blue: q < 0.10 after multiple-testing correction)", loc="left", fontsize=9.5)
fig.savefig(FIG + "fig5_equipment.png")
plt.close(fig)

# ---- 6. dealer premiums
dt = pd.read_csv(lib.RES + "e07_dealer_effects.csv")
big = dt[dt["n_ads"] >= 10].sort_values("premium_pct")
fig, ax = plt.subplots(figsize=(7.4, 0.22 * len(big) + 1.2))
yy = np.arange(len(big))
sd = 100 * (np.exp(big["dealer_effect_sd"]) - 1)
ax.errorbar(big["premium_pct"], yy, xerr=1.96 * sd, fmt="o", color=C["blue"], ecolor=C["blue"], capsize=2, ms=4, lw=1.1)
ax.set_yticks(yy); ax.set_yticklabels([f"{n[:32]} ({k})" for n, k in zip(big["seller_name"], big["n_ads"])], fontsize=7.5)
ax.axvline(0, color=INK, lw=0.8)
ax.set_xlabel("Dealer premium vs comparable cars, % (shrunk estimate, rough 95% band)")
ax.grid(axis="y", visible=False)
ax.set_title("Dealers with 10+ ads", loc="left", fontsize=10)
fig.savefig(FIG + "fig6_dealers.png")
plt.close(fig)

ADS_FIRST = int((os.environ.get('ADS_OF_INTEREST', '0').split(',') or ['0'])[0] or 0)
# ---- 7. gap distribution with the ads of interest
fig, ax = plt.subplots(figsize=(7.4, 3.6))
g = fv["gap_pct"].clip(-30, 40)
ax.hist(g, bins=70, color=C["grey"], alpha=0.8)
for i, col in [(int(a), c) for a, c in zip(os.environ.get("ADS_OF_INTEREST", "").split(","), [C["blue"], C["amber"]]) if a]:
    r = fv[fv["ad_id"] == i].iloc[0]
    ax.axvline(r["gap_pct"], color=col, lw=2)
    ax.text(r["gap_pct"] + 0.6, ax.get_ylim()[1] * (0.92 if i == ADS_FIRST else 0.80), f"{i}\n{r['gap_pct']:+.1f}%", color=col, fontsize=8.5, va="top")
ax.axvline(0, color=INK, lw=0.8)
ax.set_xlabel("How far the fair value is above the asking price, % (positive = cheaper than the model expects)")
ax.set_ylabel("Number of ads")
ax.set_title("Where the ads of interest sit among all ads", loc="left", fontsize=10.5)
fig.savefig(FIG + "fig7_gap_distribution.png")
plt.close(fig)

# ---- 8. residual spread by mileage
x = fv.copy()
x["res"] = (np.log(x["price"]) - np.log(x["fair"]))
x["kmb"] = pd.cut(x["km"] / 1000, [0, 50, 100, 150, 200, 250, 400], labels=["<50", "50-100", "100-150", "150-200", "200-250", "250+"])
s = x.groupby("kmb", observed=True)["res"].agg(["std", "size"])
fig, ax = plt.subplots(figsize=(5.6, 3.2))
ax.bar(range(len(s)), s["std"] * 100, color=C["teal"], width=0.65)
for i, (v, n) in enumerate(zip(s["std"], s["size"])):
    ax.text(i, v * 100 + 0.15, f"{v * 100:.1f}%\nn={n}", ha="center", fontsize=7.5)
ax.set_xticks(range(len(s))); ax.set_xticklabels(s.index)
ax.set_xlabel("Mileage (thousand km)"); ax.set_ylabel("Std. dev. of pricing error, %")
ax.set_ylim(0, 11.5)
ax.grid(axis="x", visible=False)
ax.set_title("Fair values are less certain for high-mileage cars", loc="left", fontsize=10)
fig.savefig(FIG + "fig8_error_by_mileage.png")
plt.close(fig)
print("figures:", sorted(os.listdir(FIG)))
