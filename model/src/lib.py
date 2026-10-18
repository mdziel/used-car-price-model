"""Shared helpers for the hedonic price model (phases 1-9).

Everything reads one modelling table (default: data/example_table.csv; set the environment variable MODEL_TABLE to use another file).
The table format is documented in model/INPUT_FORMAT.md. Fixed seeds live here.
Import with:  sys.path.insert(0, <repo>/model/src); import lib
"""
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + "/"
MODEL = ROOT + "model/"
RES = MODEL + "results/"
TABLE = os.environ.get("MODEL_TABLE", ROOT + "data/example_table.csv")
SEED = 2026
N_SPLITS, N_REPEATS = 5, 2
HOLDOUT_SHARE = 0.15

# spline definitions (fixed knots: no data-dependent transform inside CV)
AGE_S = "bs(age_years, knots=(1.5, 3, 5, 8), degree=3, lower_bound=0, upper_bound=12)"
KM_S = "bs(km10k, knots=(3, 6, 10, 15), degree=3, lower_bound=0, upper_bound=40)"

FUEL_GROUPS = {"Plug-in hybrid petrol": "PHEV", "Diesel": "Diesel", "Petrol": "Petrol",
               "Hybrid petrol (mild)": "HybridPetrol", "Hybrid diesel": "HybridPetrol"}


def load():
    """Modelling table with the derived columns used by all experiments. Returns (table, {E000: feature name})."""
    df = pd.read_csv(TABLE, encoding="utf-8-sig", low_memory=False)
    df["y"] = np.log(df["price"])
    df["km10k"] = df["km"] / 10000
    df["fuel_grp"] = df["fuel"].map(FUEL_GROUPS)
    df["awd"] = (df["drive"] == "All-wheel drive").astype(int)
    df["gen1"] = (df["model_year"] <= 2017).astype(int)       # older model years (earlier generation)
    df["my_ge2022"] = (df["model_year"] >= 2022).astype(int)  # newer model years (later facelift / equipment era)
    df["engine_grp"] = df["engine_code"]
    df["trim_grp"] = df["trim"]
    top_col = df["colour"].value_counts().head(8).index
    df["colour_grp"] = df["colour"].where(df["colour"].isin(top_col), "Other").fillna("Other")
    df["dist_big3"] = df["dist_big3"].fillna(df["dist_big3"].median())
    # dealer group id (given by the data source) is the chain; private ads and ads without a group have none
    has_g = (df["is_private"] == 0) & df["group_id"].notna()
    df["chain"] = np.where(has_g, df["group_id"].astype(str), "")
    df["chain_label"] = df["chain"]
    df["seller_name"] = df["seller_id"].where(df["is_private"] == 0)  # dealers are identified by their (anonymous) id
    df["chain_or_seller"] = np.where(df["chain"] != "", "chain_" + df["chain"], df["seller_id"])
    df["warranty_incl_months"] = df["warranty_incl_months"].fillna(0)
    df["warranty_option_max_months"] = df["warranty_option_max_months"].fillna(0)
    df["warranty_incl_ge12"] = (df["warranty_incl_months"] >= 12).astype(int)
    df["warranty_incl_ge24"] = (df["warranty_incl_months"] >= 24).astype(int)
    # equipment columns (eq_<feature name>) renamed to safe names; mapping kept
    eq_cols = [c for c in df.columns if c.startswith("eq_")]
    names = {f"E{i:03d}": c[3:] for i, c in enumerate(eq_cols)}
    df = df.rename(columns={c: f"E{i:03d}" for i, c in enumerate(eq_cols)})
    return df, names


TREE_NUM = ["age_years", "km10k", "power_hp", "awd", "gen1", "my_ge2022", "dist_big3", "is_private",
            "vat_deductible", "n_equip_raw", "n_eq_ticked", "service_followed", "right_to_exchange", "known_damage_yes",
            "warranty_incl_months", "warranty_option_max_months", "warranty_incl_ge12", "warranty_incl_ge24",
            "warranty_certified", "warranty_trade_assoc", "warranty_option_offered", "warranty_mentioned"]


def tree_features(df, names, EQ, extended=True):
    """Feature matrix for the boosted models. extended=True adds model year, owners/CO2/weight/engine size (missing values allowed)
    and, if the table has a 'title_spec' text column, binary title keywords (experiment E11: about -3% CV error). Listing order
    (ad id) is deliberately NOT used: it reflects how long an ad has survived on the market, not the car."""
    cat = pd.get_dummies(df[["fuel_grp", "engine_grp", "trim_grp", "colour_grp"]], dtype=float)
    parts = [df[TREE_NUM].astype(float), cat, df[EQ].astype(float)]
    if extended:
        parts.append(df[["model_year", "owners", "co2", "weight_kg", "engine_l"]].astype(float))
        if "title_spec" in df.columns:
            from sklearn.feature_extraction.text import CountVectorizer
            cv = CountVectorizer(token_pattern=r"[^\W\d_]{3,}", lowercase=True, min_df=60, binary=True)
            tok = cv.fit_transform(df["title_spec"].fillna("").str.lower())
            parts.append(pd.DataFrame(tok.toarray(), columns=["t_" + t for t in cv.get_feature_names_out()], index=df.index))
    return pd.concat(parts, axis=1)


def equipment_cols(df, names, min_share=0.02, max_share=0.98):
    return [k for k in names if min_share <= df[k].mean() <= max_share]


# ------------------------------------------------------------------ validation design
def assign_groups(groups, n_splits, rng):
    """Random group K-fold: shuffled groups go to the currently smallest fold."""
    sizes = groups.value_counts()
    order = rng.permutation(sizes.index.values)
    load = np.zeros(n_splits)
    fold_of = {}
    for g in order:
        k = int(np.argmin(load + rng.rand(n_splits) * 1e-6))
        fold_of[g] = k
        load[k] += sizes[g]
    return groups.map(fold_of).values


def make_design(df):
    """Locked hold-out (about 15% of rows, drawn by seller) and repeated grouped folds on the rest."""
    rng = np.random.RandomState(SEED)
    sizes = df["seller_id"].value_counts()
    order = rng.permutation(sizes.index.values)
    hold, n = set(), 0
    for g in order:
        if n >= HOLDOUT_SHARE * len(df):
            break
        hold.add(g)
        n += sizes[g]
    out = pd.DataFrame({"ad_id": df["ad_id"], "seller_id": df["seller_id"], "holdout": df["seller_id"].isin(hold).astype(int)})
    dev = out["holdout"] == 0
    for r in range(N_REPEATS):
        rr = np.random.RandomState(SEED + 100 + r)
        f = np.full(len(df), -1)
        f[dev.values] = assign_groups(out.loc[dev, "seller_id"], N_SPLITS, rr)
        out[f"fold_r{r}"] = f
    return out


def load_design():
    return pd.read_csv(RES + "folds.csv")


# ------------------------------------------------------------------ metrics
def metrics(y_log, pred_log):
    e = pred_log - y_log
    p, a = np.exp(pred_log), np.exp(y_log)
    return {"rmse_log": float(np.sqrt(np.mean(e ** 2))), "mae_log": float(np.mean(np.abs(e))),
            "mae_abs": float(np.mean(np.abs(p - a))), "mape": float(np.mean(np.abs(p - a) / a)),
            "r2_log": float(1 - np.sum(e ** 2) / np.sum((y_log - y_log.mean()) ** 2)),
            "bias_log": float(np.mean(e)), "n": int(len(e))}


def cv_oof(fit_predict, design, n, n_jobs=1):
    """Repeated grouped CV on the development rows.

    fit_predict(train_idx, test_idx) -> predictions (log price) for test_idx.
    Returns oof array (R x n; NaN on hold-out rows) and the list of (repeat, fold, test_idx).
    """
    from joblib import Parallel, delayed
    jobs = []
    for r in range(N_REPEATS):
        f = design[f"fold_r{r}"].values
        for k in range(N_SPLITS):
            jobs.append((r, k, np.where(f == k)[0], np.where((f != k) & (f >= 0))[0]))
    res = Parallel(n_jobs=n_jobs)(delayed(fit_predict)(tr, te) for _, _, te, tr in jobs)
    oof = np.full((N_REPEATS, n), np.nan)
    for (r, k, te, tr), p in zip(jobs, res):
        oof[r, te] = p
    return oof, [(r, k, te) for r, k, te, _ in jobs]


def summarise_oof(y, oof, dev_mask):
    """Mean over repeats of the metrics; sd of RMSE over repeats."""
    rows = [metrics(y[dev_mask], oof[r][dev_mask]) for r in range(oof.shape[0])]
    m = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    m["rmse_sd_repeats"] = float(np.std([r["rmse_log"] for r in rows]))
    return m
