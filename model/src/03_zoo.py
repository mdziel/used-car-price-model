"""E03-E06: regularised linear models, random forest, boosting (HistGB, LightGBM, XGBoost), monotone variants.

Nested validation: outer = the frozen repeated grouped folds; inner = 3-fold GroupKFold by seller on the
training part, used only to choose hyperparameters from small grids. Out-of-fold predictions are saved.
Run:  python -I 03_zoo.py [name ...]     (no names = all)
"""
import os
import json
import sys
import time
import warnings

import numpy as np
import pandas as pd
from patsy import dmatrix
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib  # noqa: E402

warnings.filterwarnings("ignore")
df, names = lib.load()
des = lib.load_design()
dev = (des["holdout"] == 0).values
y = df["y"].values
groups = df["seller_id"].values
F = json.load(open(lib.RES + "ols_formulas.json"))
EQ = F["EQ"]

# ---- matrices
XL = dmatrix(F["E"], df, return_type="dataframe", NA_action="raise")
XL = XL.drop(columns=["Intercept"]).values  # linear models: spline design + dummies (E)
XT_df = lib.tree_features(df, names, EQ)
XT = XT_df.values
cols_T = list(XT_df.columns)
mono = [-1 if c in ("age_years", "km10k") else 0 for c in cols_T]
json.dump(cols_T, open(lib.RES + "tree_feature_names.json", "w"))
print("linear design", XL.shape, "| tree design", XT.shape)


def nested(estimator, grid, X):
    """fit_predict with inner grouped grid search."""
    def fp(tr, te):
        gs = GridSearchCV(estimator, grid, cv=GroupKFold(3), scoring="neg_root_mean_squared_error", n_jobs=1)
        gs.fit(X[tr], y[tr], groups=groups[tr])
        fp.best.append(gs.best_params_)
        return gs.predict(X[te])
    fp.best = []
    return fp


def lgbm(**kw):
    return lgb.LGBMRegressor(n_jobs=1, verbose=-1, subsample_freq=1, subsample=0.8, colsample_bytree=0.7,
                             random_state=lib.SEED, **kw)


MODELS = {
    "ridge": (make_pipeline(StandardScaler(), Ridge()), {"ridge__alpha": [1, 3, 10, 30, 100, 300]}, XL),
    "lasso": (make_pipeline(StandardScaler(), Lasso(max_iter=20000)), {"lasso__alpha": [3e-4, 1e-3, 3e-3, 1e-2]}, XL),
    "elasticnet": (make_pipeline(StandardScaler(), ElasticNet(max_iter=20000)),
                   {"elasticnet__alpha": [1e-3, 3e-3, 1e-2], "elasticnet__l1_ratio": [0.2, 0.5, 0.8]}, XL),
    "rf": (RandomForestRegressor(n_estimators=300, n_jobs=1, random_state=lib.SEED),
           {"max_features": [0.3, 0.5], "min_samples_leaf": [1, 3]}, XT),
    "histgb": (HistGradientBoostingRegressor(random_state=lib.SEED),
               {"learning_rate": [0.05, 0.1], "max_leaf_nodes": [8, 16], "max_iter": [300, 600], "l2_regularization": [1.0]}, XT),
    "lightgbm": (lgbm(), {"n_estimators": [400, 800], "learning_rate": [0.03, 0.06], "num_leaves": [8, 16],
                          "min_child_samples": [10, 25]}, XT),
    "lightgbm_mono": (lgbm(monotone_constraints=mono), {"n_estimators": [400, 800], "learning_rate": [0.03, 0.06],
                                                      "num_leaves": [8, 16], "min_child_samples": [10, 25]}, XT),
    "xgboost": (xgb.XGBRegressor(n_jobs=1, subsample=0.8, colsample_bytree=0.7, random_state=lib.SEED, verbosity=0),
                {"max_depth": [3, 5], "n_estimators": [400, 800], "learning_rate": [0.03, 0.06], "min_child_weight": [3]}, XT),
}

want = sys.argv[1:] or list(MODELS)
rows = []
for name in want:
    est, grid, X = MODELS[name]
    t0 = time.time()
    fp = nested(est, grid, X)
    oof, jobs = lib.cv_oof(fp, des, len(df), n_jobs=10)
    # fp.best is filled inside worker processes, so recompute best params on the full dev set for reporting
    gs = GridSearchCV(est, grid, cv=GroupKFold(5), scoring="neg_root_mean_squared_error", n_jobs=10)
    gs.fit(X[dev], y[dev], groups=groups[dev])
    s = lib.summarise_oof(y, oof, dev)
    s.update({"model": name, "seconds": round(time.time() - t0), "best_params_full_dev": json.dumps({k: (float(v) if not isinstance(v, str) else v) for k, v in gs.best_params_.items()})})
    rows.append(s)
    np.save(lib.RES + f"oof_{name}.npy", oof)
    print(f"{name:14s} rmse_log={s['rmse_log']:.4f} (sd rep {s['rmse_sd_repeats']:.4f}) mae_abs={s['mae_abs']:.0f} mape={s['mape']:.3f} r2={s['r2_log']:.4f} [{s['seconds']}s] {s['best_params_full_dev']}", flush=True)
    pd.DataFrame([s]).to_csv(lib.RES + f"e03_e06_models_{name}.csv", index=False)  # one file per model
