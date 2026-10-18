# Experiment log

Entry format: `ID | date | hypothesis and expected sign | feature set | model | CV scheme and seed | metrics | decision | files`.
Hypotheses are written before the run where possible. Metrics are filled in after.

---

## E00 — Phase 0: data audit and cleaning (2026-10-17)

- **Question:** is the collected data usable as a modelling table, and what has to be fixed first?
- **Scripts:** source-specific cleaning code (not included); the rules and counts are in `CLEANING_LOG.md`.
- **Result:** 3,012 ads in, **2,995** out (17 ads removed; details in `CLEANING_LOG.md`):
  - dropped 1 with price 0, 1 export-net price, 4 damaged/as-is, 2 duplicate chassis numbers,
    7 with no mileage, 1 with implausibly low mileage, 1 with missing power;
  - fixed 9 mileages entered in the wrong unit (all private ads);
  - 16 ads have an unreliable registration date, so their age comes from the model year.
- **Findings that shape the next phases:**
  - Asking prices are VAT-inclusive throughout. The 1,199 company cars with an ex-VAT amount in the listing show
    a constant fraction of the price. Text about excluding VAT is leasing wording, not a different price basis
    (a first version of rule R2 wrongly dropped 13 such ads; corrected and noted in the log).
  - `log(price)` and age are strongly negatively correlated (-0.91), so age is the dominant term and the
    functional form (nonlinear, fuel-specific) will matter.
  - 726 dealers (62 with 10+ ads, 91 with 5–9, 220 with 2–4, 353 with a single ad) plus 236 private ads:
    dealer effects are only well identified for about 150 dealers. Chain ids exist for 119 groups covering
    1,943 ads (65%), so a chain-level effect is feasible.
  - Owners (66% missing), CO2 (32%), weight (27%) are sparse; treated as optional features with missing flags.
  - Ad coordinates are available and can serve as an explicit location feature (the public extract only has a rounded distance to the nearest big city).
- **Decision:** proceed to Phase 1 (validation design) with the modelling table frozen.
  The table must not change without a new entry here.

---

## Pre-registered hypotheses for Phases 1–8 (written 2026-10-17, before any model was fitted)

Common setup: target `log(price)`; development set = all sellers except a locked 15% hold-out; outer validation
= 5-fold grouped by seller, repeated (seeds fixed in `src/lib.py`); errors reported as RMSE of log price
(≈ proportional error), MAE in price units, MAPE and R². Hold-out used once at the end (E09).

| ID | Experiment | Expectation (before seeing results) |
|---|---|---|
| E01 | Baselines: mean log price by model year; OLS(age, km, power); OLS with age/km splines | Age explains most variance (R² around 0.85–0.9 for model year alone); splines beat linear terms; log RMSE of the best baseline roughly 0.12–0.15 |
| E02 | Staged OLS feature sets A→E (age/km/power → +fuel, engine, drive, generation → +trim, colour, seller type → +equipment → +warranty, location, seller style) | Fuel/engine type gives the largest gain after age/km; trim helps; equipment adds little once trim is in (collinear bundles); warranty positive and small; location tiny. Each stage improves CV error by less than the previous one |
| E03 | Ridge / lasso / elastic net on the full design | Regularisation helps over plain OLS with 100+ equipment dummies, by a small margin |
| E04 | Random forest, HistGradientBoosting, LightGBM, XGBoost, same features | Boosting beats forest; boosting beats the best linear model modestly (a few % in log RMSE) because of interactions (fuel × age, trim × age) |
| E05 | Monotone constraints (price non-increasing in age and km) on boosting | Little change in accuracy, smoother effects, more credible marginal effects |
| E06 | Nested-CV tuning and a blend of the best linear and best boosted model | Blend is at least as good as the best single model |
| E07 | Car + dealer model (random dealer/chain effect) | Dealer/chain effect sd of roughly 3–6% of price; known-dealer prediction error clearly lower than unseen-dealer error; most of the gain comes from chains and large dealers |
| E08 | Inference: price effect of +1 year, +10,000 km, equipment, warranty | +1 year about −5% to −9% at fixed mileage; +10,000 km about −1% to −3%; panoramic roof, 360° camera, Harman Kardon, air suspension positive and individually small; effects of rare or bundle-collinear features imprecise |
| E09 | Hold-out check of the chosen model | Hold-out error within about one standard error of the CV error (no overfitting from model selection) |
| E10 | Robustness: drop private ads / first generation / plug-ins; winsorised target; alternative mileage rules; chain-clustered SEs | Age and mileage effects stable in sign and roughly in size; equipment effects less stable |

---

## E00b — Phase 1: validation design (2026-10-17, `src/01_split.py`)

- Hold-out: 16% of ads (478 ads, 136 sellers) drawn by seller with seed 2026; never used for tuning.
- Development set: 2,517 ads. Outer CV: 5 folds grouped by seller, 2 repeats (different seller-to-fold assignments);
  inner CV for tuning: 3-fold grouped. Fold assignments frozen in `results/folds.csv`.
- Check: no seller in two folds of a repeat; hold-out sellers disjoint from development sellers.
- Chain definition (`lib.py`): first word of the dealer name if at least 3 different dealers share it (large dealer groups).

## E01 + E02 — baselines and staged OLS (`src/02_ols_stages.py`, `results/e01_e02_ols_cv.csv`)

Out-of-fold error on the development set (RMSE of log price, mean of 2 repeats; 0.06 ≈ 6% of price):

| Model | RMSE log | MAE | MAPE | R² (log) |
|---|---:|---:|---:|---:|
| B1 mean by model year | 0.1441 | 38,316 | 11.4% | 0.868 |
| B2 OLS age + km + power (linear) | 0.1123 | 29,538 | 8.7% | 0.920 |
| A splines in age and km + power | 0.0975 | 26,888 | 7.7% | 0.940 |
| B + fuel, engine, drive, generation, fuel × age/km | 0.0789 | 22,768 | 6.3% | 0.961 |
| C + trim, colour, private/dealer, VAT | 0.0622 | 16,037 | 4.6% | 0.975 |
| D + 86 equipment ticks | 0.0601 | 15,565 | 4.5% | 0.977 |
| E + warranty, seller style, location | **0.0593** | 15,429 | 4.4% | 0.978 |

- **Versus expectation:** baseline matched (0.144; expected 0.12–0.15); the best baseline (0.098) was better than expected
  because splines capture the depreciation curve well. Fuel/engine was the largest step after age/km, **but trim
  (0.079 → 0.062) mattered more than expected**. Equipment (0.062 → 0.060) and warranty/location (→ 0.059) added little,
  as expected. Error by segment (set E): PHEV 0.051, diesel 0.067, petrol 0.066; nearly new (<2 years) 0.043, 8+ years 0.070.
- **Decision:** keep set E as the linear benchmark.

## E03 — regularised linear (`src/03_zoo.py`)

Ridge 0.0593, lasso 0.0591, elastic net 0.0591 (nested grouped tuning). **Essentially identical to OLS E (0.0593):**
with 2,500 ads and 150 columns, shrinkage does not buy accuracy. Expectation ("small margin") met. Lasso is useful for
selecting equipment (see E08).

## E04 + E05 — tree models, monotone variant (`src/03_zoo.py`)

Nested grouped tuning (inner 3-fold by seller; small grids). Base features (140 columns):

| Model | RMSE log | MAE | Best parameters (full development set) |
|---|---:|---:|---|
| Random forest | 0.0649 | 15,799 | max_features 0.5, min leaf 1 |
| HistGradientBoosting | 0.0592 | 15,021 | lr 0.05, 16 leaves, 600 iterations |
| LightGBM | 0.0578 | 14,641 | lr 0.03, 16 leaves, 800 trees, min child 25 |
| LightGBM, monotone in age and km | 0.0590 | 14,903 | same grid |
| XGBoost | 0.0574 | 14,487 | depth 5, lr 0.03, 800 trees |

- **Versus expectation:** boosting beat the linear model by about 3% (expected "modestly"), forest worse than both.
  Monotone constraints cost about 2% accuracy (expected "little change"). Base-feature results are kept in
  `results/e03_e06_models_*_basefeatures.csv`.

## E11 — do extra features help? (`src/09_extras.py`, `results/e11_extras.csv`)

Decision rule fixed before running: adopt a feature group only if CV error improves by ≥1% in both repeats.
XGBoost with fixed parameters, base RMSE 0.0571:

| Added | RMSE | Change |
|---|---:|---:|
| model year | 0.0565 | −1.1% |
| owners, CO₂, weight, engine size (missing allowed) | 0.0565 | −1.1% |
| 28 title keywords (equipment words from the ad title) | 0.0566 | −0.9% |
| listing order (ad id) | 0.0570 | −0.2% |
| all of the above | 0.0549 | −3.9% |
| **all except listing order (adopted)** | **0.0554** | **−3.0%** (repeat 0: −2.6%, repeat 1: −3.3%) |

OLS with trim × age, trim × km or engine × age interactions: no improvement (0.0593 → 0.0591–0.0594).
**Decision:** extended features adopted for the boosted models (`lib.tree_features`); listing order excluded on
principle (it measures how long an ad has survived, not the car) even though adding it gave another −0.9%.
After re-tuning: XGBoost 0.0554, **LightGBM 0.0549**.

## E06 — blend (`src/04_final_fit.py`)

Blend of lasso and LightGBM on out-of-fold predictions: weight on lasso 0.0 → 0.0549, 0.3 → 0.0539 (best), 0.5 → 0.0543,
1.0 → 0.0591. Pre-set rule: use the simple 50/50 average unless the best weight is more than 0.2 away. **Final: 50/50, CV RMSE 0.0543,
MAE 13,759, MAPE 4.0%, R² 0.981.** The first blend (lasso + XGBoost on base features) had CV 0.0558.

## E09 — hold-out (touched once; 478 ads, 136 sellers)

| Model | Hold-out RMSE | Hold-out MAE | CV RMSE (development) |
|---|---:|---:|---:|
| Lasso | 0.0596 | 17,247 | 0.0591 |
| LightGBM | 0.0495 | 13,934 | 0.0549 |
| **Blend 50/50 (selected on development data)** | **0.0516** | **14,717** | **0.0543** |

- No sign of overfitting from model selection (hold-out error is lower than CV error; sampling error of a hold-out
  RMSE with 478 ads is about ±0.0017). LightGBM alone was better on the hold-out than the blend; the blend stays because
  the selection rule was fixed beforehand.
- Interval check: the 80% ranges contain 80% of development asks and 74% of hold-out asks (slightly too narrow on new data).

## E07 — car + dealer model (`src/05_dealer.py`)

Empirical-Bayes dealer and chain effects estimated from out-of-fold residuals; leave-one-ad-out evaluation.

- **First attempt (failed to be sensible):** chains defined by the first word of the dealer name created spurious
  chains from generic one-word names. Replaced by the data source's own dealer-group id (119 groups, 65% of ads).
- **Variance components (blend residuals):** within-dealer sd 4.2%; independent dealers sd 4.1%; branches inside groups sd 2.0%;
  **group-level sd 0.3%** (not distinguishable from zero).
- **Leave-one-out error reduction from adding the dealer effect:** all dealer ads −4.9%; dealer has ≥1 other ad −6.4%
  (2,005 ads); ≥4 other ads −7.9% (1,528 ads); single-ad dealers 0% (no information). Robust to the car model:
  lasso only −5.9% / −7.6% / −8.2%; LightGBM only −3.7% / −4.9% / −7.2%.
- **Versus expectation:** the dealer sd (2–4%) is lower than expected (3–6%), and the expected chain-level effect did not appear.

## E08 — inference (`src/06_inference.py`, `src/06b_shap.py`)

Structural OLS (set E, all 2,995 ads, seller-clustered SE) and the boosted model:

- **Age, mileage (all cars):** +1 year at fixed km −4.41% (delta-method 95% −4.77 to −4.06; cluster bootstrap −4.68 to −3.98);
  +10,000 km at fixed age −2.51% (−2.59 to −2.43); a typical year (+1 year, +15,000 km) −8.00%. LightGBM finite differences: −3.78%, −2.35%, −7.19%.
- **By fuel (+1 year, km fixed):** plug-in hybrid −5.55%, diesel −2.93%, petrol −3.21%, hybrid petrol −4.40%. By age band: 0–2 yrs −3.5%, 2–4 −6.4%, 4–6 −6.0%, 6–8 −3.1%, 8–11 −2.9%.
- **Trim vs the base trim ("Basic"):** Sport +15.4%, Elite +12.9%, Executive +11.7%, Comfort +9.6%, Prestige +9.1%, Classic +5.3%, trim not stated +7.3% (trim labels are invented, see `INPUT_FORMAT.md`).
- **Equipment:** 18 of 86 features have q<0.10 (about 2 expected false positives): air suspension +5.7%, auto park +3.3%, sport steering wheel +2.8%,
  light sensor +2.5%, head-up display +2.1%, glass roof +1.9%, electric seats +1.6–1.8%, navigation +1.2%; negative for Isofix −1.2%, CarPlay −1.3%, app integration −1.4%, trip computer −1.7% (listing-style artefacts).
  Panoramic roof +0.8% (−0.05 to +1.6), not significant.
- **Warranty and seller:** 12+ months included −1.5% (−2.7 to −0.3); 24+ months +2.0% (−0.4 to +4.5); optional warranty offered −1.7%; private vs dealer +0.3% (n.s.); +0.65% per 100 km from the three big cities.
- **SHAP (mean |log contribution|):** mileage 0.140, age 0.094, model year 0.064, generation 0.050, trim 0.026, equipment ticks 0.018.
- **Versus expectation:** the per-year age effect at fixed mileage (−4.4%) is smaller than expected (−5% to −9%); mileage (−2.5% per 10,000 km) within the expected range.
- **Pitfalls found and fixed:** (1) per-ad SHAP from a model trained on the ad reproduced its ask almost exactly, so the two ads are now explained with a model that excludes them;
  (2) finite-difference age effects must also shift model year when model year is a feature.

## E10 — robustness (`src/08_robustness.py`)

Thirteen variants (drop private ads, first generation, plug-ins, flagged ads, mileage-corrected ads, cars above 250,000 km, cars under 1 year;
dealer-group clustering; winsorised target; linear instead of spline age/km; plug-ins only). Mileage effect −2.4% to −2.8% in all; the age effect varies mainly with
the powertrain mix (−2.8% without plug-ins, −5.6% for plug-ins only). Head-up display +1.7% to +2.3% and air suspension +3.7% to +5.9% keep their sign;
panoramic roof between −0.5% and +2.0%.

## E12 — functional form of age (asked after the report; `src/12_age_form.py`, `results/e12_age_form.csv`, `results/e12_age_profile.csv`)

Question: is a flexible continuous age (cubic spline with knots at 1.5, 3, 5, 8 years, plus fuel × linear age) adequate, or are year dummies better?
Expectation before running: dummies fit about equally well; steep early decline, plateau later.

| Variant (OLS set E, grouped CV) | RMSE log |
|---|---:|
| Spline, 4 knots (current) | 0.0593 |
| Year dummies (0,1,…,10+) + fuel × linear age | 0.0595 |
| Year dummies, no fuel × age term | 0.0613 |
| Finer spline (9 df) | 0.0594 |
| Spline + year dummies | 0.0594 |
| Year dummies × fuel group | not estimable (empty cells, e.g. no diesel under 1 year; predictions explode) |

- **Result:** dummies do not beat the spline; the spline is as good and smoother. Implied price change from age a to a+1 **at fixed mileage**
  (spline): 0→1 −2.4%, 1→2 −1.9%, 2→3 −3.1%, 3→4 −7.5%, 4→5 −7.4%, 5→6 −5.2%, then about −3.5% to −4%. **Not** steepest in year 1: with
  mileage held fixed the steepest years are 3–6. Adding the usual 15,000 km per year (−3.8%) gives a *total* yearly loss of about 6% in years 1–2,
  11% in years 4–5 and 7–8% later.
- **Caveat:** cars younger than 1 year are rare (78 ads) and mostly plug-in hybrids, so the very first year is poorly identified, and the data
  cannot show the new-car-to-used drop (first registration price versus the first used ask).
- **Decision:** keep the spline (no change to the model or the report numbers).

## Deliverables (Phase 9)

A report with figures, per-ad fair values and rankings, and a model workbook were produced from the private data; this repository contains the code, documentation, aggregate results, figures and an anonymised data extract.

## Template

```
## E## — title (date)
- Hypothesis / expected sign:
- Feature set:
- Model and hyperparameters:
- CV scheme and seed:
- Metrics (mean ± sd over folds):
- Decision (keep / drop / investigate):
- Files:
```
