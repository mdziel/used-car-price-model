"""E07: car + dealer model. Dealer/chain premiums are empirical-Bayes (BLUP) effects estimated from the
out-of-fold residuals of a car-only model; evaluation uses leave-one-ad-out within dealer, so an ad's own residual
never feeds its own dealer effect.

Structure: log price = car part + chain effect (chain members only) + dealer effect + noise; effects ~ N(0, tau^2).
Run:  python 05_dealer.py [oof_name]       (default: blend)
"""
import os
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib  # noqa: E402

name = sys.argv[1] if len(sys.argv) > 1 else "blend"
df, names = lib.load()
des = lib.load_design()
dev = (des["holdout"] == 0).values
y = df["y"].values
pred = np.load(lib.RES + f"oof_{name}.npy").mean(axis=0)  # average of the repeats' out-of-fold predictions
r_all = y - pred
mask = dev & ~np.isnan(pred)
d = df.loc[mask, ["ad_id", "seller_id", "seller_name", "chain", "is_private", "price"]].copy()
d["r"] = r_all[mask]
mu_dev = float(d["r"].mean())
d["r"] -= mu_dev
d["chain_key"] = np.where(d["chain"] != "", d["chain"], "none")
dl = d[d["is_private"] == 0].copy()  # private ads get no dealer effect

# ---------------------------------------------------------------- variance components (moments)
g = dl.groupby("seller_id")["r"]
n_s, m_s, ss = g.size(), g.mean(), g.apply(lambda v: ((v - v.mean()) ** 2).sum())
multi = n_s >= 2
sig2 = float(ss[multi].sum() / (n_s[multi].sum() - multi.sum()))  # pooled within-dealer variance
sel = pd.DataFrame({"n": n_s, "m": m_s, "chain": dl.groupby("seller_id")["chain_key"].first()})
nc = sel[(sel["chain"] == "none") & (sel["n"] >= 2)]
tau_nc2 = float(max(1e-5, np.mean(nc["m"] ** 2 - sig2 / nc["n"])))
ch = sel[(sel["chain"] != "none") & (sel["n"] >= 2)]
tau_tot_chain2 = float(max(1e-5, np.mean(ch["m"] ** 2 - sig2 / ch["n"])))
within = []
for c, grp in ch.groupby("chain"):
    if len(grp) >= 2:
        within.append(np.var(grp["m"], ddof=1) - np.mean(sig2 / grp["n"]))
tau_s2 = float(max(1e-5, np.mean(within)))
tau_c2 = float(max(1e-5, tau_tot_chain2 - tau_s2))
print(f"car model: {name}")
print(f"sigma_e (within-dealer) = {np.sqrt(sig2):.4f} | dealer sd non-chain = {np.sqrt(tau_nc2):.4f} | "
      f"chain sellers: total sd {np.sqrt(tau_tot_chain2):.4f} = chain sd {np.sqrt(tau_c2):.4f} (+) branch sd {np.sqrt(tau_s2):.4f}")

# ---------------------------------------------------------------- BLUPs
sel["V"] = np.where(sel["chain"] == "none", tau_nc2, tau_s2) + sig2 / sel["n"]
chains = {}
for c, grp in sel[sel["chain"] != "none"].groupby("chain"):
    chains[c] = (float((1 / grp["V"]).sum()), float((grp["m"] / grp["V"]).sum()))


def effect(chain, seller_m, seller_n, others_A, others_B):
    """Return (chain effect u, dealer effect v) given seller stats (m, n; n may be 0) and the other sellers' sums."""
    if chain == "none":
        if seller_n == 0:
            return 0.0, 0.0
        V = tau_nc2 + sig2 / seller_n
        return 0.0, tau_nc2 / V * seller_m
    A, B = others_A, others_B
    if seller_n > 0:
        V = tau_s2 + sig2 / seller_n
        A, B = A + 1 / V, B + seller_m / V
    u = tau_c2 * B / (1 + tau_c2 * A)
    v = 0.0 if seller_n == 0 else tau_s2 / (tau_s2 + sig2 / seller_n) * (seller_m - u)
    return u, v


# leave-one-ad-out evaluation
rows = []
S = dl.groupby("seller_id")["r"].sum()
for i, row in dl.iterrows():
    s, c = row["seller_id"], row["chain_key"]
    n, tot = int(n_s[s]), float(S[s])
    n1 = n - 1
    m1 = (tot - row["r"]) / n1 if n1 > 0 else 0.0
    if c != "none":
        A, B = chains[c]
        V_full = tau_s2 + sig2 / n
        A_oth, B_oth = A - 1 / V_full, B - float(m_s[s]) / V_full
    else:
        A_oth = B_oth = 0.0
    u, v = effect(c, m1, n1, A_oth, B_oth)
    rows.append((i, u, v, n1))
loo = pd.DataFrame(rows, columns=["idx", "u_chain", "v_dealer", "n_other"]).set_index("idx")
dl = dl.join(loo)
dl["prem"] = dl["u_chain"] + dl["v_dealer"]
out = {}
for label, m in [("all dealer ads", np.ones(len(dl), bool)), ("dealer has >=1 other ad", dl["n_other"] >= 1),
                 ("dealer has >=4 other ads", dl["n_other"] >= 4), ("dealer is single-ad, chain info only", (dl["n_other"] == 0) & (dl["chain_key"] != "none")),
                 ("dealer single-ad, no chain", (dl["n_other"] == 0) & (dl["chain_key"] == "none"))]:
    sub = dl[m]
    if len(sub) < 20:
        continue
    e0 = np.sqrt(np.mean(sub["r"] ** 2))
    e1 = np.sqrt(np.mean((sub["r"] - sub["prem"]) ** 2))
    out[label] = {"n": int(len(sub)), "rmse_car_only": float(e0), "rmse_with_dealer": float(e1), "gain_pct": float(100 * (1 - e1 / e0))}
    print(f"{label:40s} n={len(sub):5d} rmse car-only={e0:.4f} | + dealer effect={e1:.4f} | gain {100 * (1 - e1 / e0):+.1f}%")
json.dump({"car_model": name, "sigma_within": np.sqrt(sig2), "sd_dealer_nonchain": np.sqrt(tau_nc2), "sd_chain": np.sqrt(tau_c2),
           "sd_branch_in_chain": np.sqrt(tau_s2), "loo_eval": out}, open(lib.RES + "e07_summary.json", "w"), indent=1)

# ---------------------------------------------------------------- final (all-data) dealer and chain effects
# final effects use every ad: development out-of-fold residuals plus hold-out predictions (variance components stay as
# estimated on the development set; the hold-out accuracy numbers in E09 do not use these effects)
import os  # noqa: E402
hp_path = lib.RES + "holdout_pred_blend.npy"
r_final = r_all.copy()
if os.path.exists(hp_path):
    hp = np.load(hp_path)
    r_final[~dev] = (y - hp)[~dev]
ok = ~np.isnan(r_final)
fa = df.loc[ok, ["ad_id", "seller_id", "seller_name", "chain", "is_private"]].copy()
fa["r"] = r_final[ok] - mu_dev
fa = fa[fa["is_private"] == 0]
fa["chain_key"] = np.where(fa["chain"] != "", fa["chain"], "none")
g2 = fa.groupby("seller_id")["r"]
sel = pd.DataFrame({"n": g2.size(), "m": g2.mean(), "chain": fa.groupby("seller_id")["chain_key"].first()})
sel["V"] = np.where(sel["chain"] == "none", tau_nc2, tau_s2) + sig2 / sel["n"]
chains = {}
for c, grp in sel[sel["chain"] != "none"].groupby("chain"):
    chains[c] = (float((1 / grp["V"]).sum()), float((grp["m"] / grp["V"]).sum()))
dl = fa  # for the seller-name merge below
tab = []
for s, row in sel.iterrows():
    c = row["chain"]
    A, B = chains.get(c, (0.0, 0.0))
    if c != "none":
        V = tau_s2 + sig2 / row["n"]
        A_oth, B_oth = A - 1 / V, B - row["m"] / V
    else:
        A_oth = B_oth = 0.0
    u, v = effect(c, row["m"], int(row["n"]), A_oth, B_oth)
    sd_v = np.sqrt(max(0.0, (tau_nc2 if c == "none" else tau_s2) * (1 - (tau_nc2 if c == "none" else tau_s2) / row["V"])))
    tab.append({"seller_id": s, "chain": c, "n_ads": int(row["n"]), "mean_resid": row["m"], "chain_effect": u, "dealer_effect": v,
                "premium": u + v, "dealer_effect_sd": sd_v})
tab = pd.DataFrame(tab).merge(dl.groupby("seller_id")["seller_name"].first().reset_index(), on="seller_id")
tab["premium_pct"] = 100 * (np.exp(tab["premium"]) - 1)
tab.sort_values("premium", ascending=False).to_csv(lib.RES + "e07_dealer_effects.csv", index=False)
ctab = []
for c, (A, B) in chains.items():
    u = tau_c2 * B / (1 + tau_c2 * A)
    ctab.append({"chain": c, "label": df.loc[df["chain"] == c, "chain_label"].iloc[0], "n_ads": int(sel[sel["chain"] == c]["n"].sum()), "n_branches": int((sel["chain"] == c).sum()),
                 "chain_effect_pct": 100 * (np.exp(u) - 1), "sd_pct": 100 * np.sqrt(tau_c2 / (1 + tau_c2 * A))})
pd.DataFrame(ctab).sort_values("chain_effect_pct", ascending=False).to_csv(lib.RES + "e07_chain_effects.csv", index=False)
print("\nchain effects (shrunk, % of price):")
print(pd.DataFrame(ctab).sort_values("chain_effect_pct", ascending=False).round(2).to_string(index=False))
t5 = tab[tab["n_ads"] >= 5].sort_values("premium")
print("\nlowest premium dealers (n>=5):", t5.head(5)[["seller_name", "n_ads", "premium_pct"]].round(1).values.tolist())
print("highest premium dealers (n>=5):", t5.tail(5)[["seller_name", "n_ads", "premium_pct"]].round(1).values.tolist())
