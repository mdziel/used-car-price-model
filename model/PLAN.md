# Hedonic price model for used cars — plan

Status: draft v1 (2026-10-17). Nothing in this plan has been run yet.
Data snapshot: 3,012 used-car ads collected 2026-10-16/17 (raw data not included; an anonymised extract is in `data/` and `spreadsheet/`).

## 1. Questions

1. **Marginal effects (hedonic coefficients):** what is the price effect of one more year of age,
   +10,000 km, and each piece of equipment, holding other things equal?
2. **Fair value:** for each ad, what *should* it cost given its characteristics, and how far is the
   asking price above/below that (with an uncertainty band)?

Both questions are answered from one dataset but need different tools: question 1 needs an
interpretable model with honest standard errors; question 2 needs the best out-of-sample predictor.
We therefore keep a **structural track** (log-linear regression family) and a **prediction track**
(regularised and tree-based models), and compare them under the same validation scheme.

## 2. What the data look like (profile of the current snapshot)

| Item | Finding | Consequence |
|---|---|---|
| Rows | 3,012 ads, 709 sellers; 2,769 dealers, 243 private | Dealer effects matter; cluster by seller |
| Asking price | median 370k, 5–95% range 166k–589k, max 859k; one ad with price 0 | Model log price; drop/flag price 0 |
| Model year | 2016–2026, 2025 is the largest cohort (521) | Many nearly-new cars; age/mileage not linear |
| Fuel | plug-in petrol 1,582, diesel 1,162, petrol 179, hybrid 89 | Three very different markets; need interactions or segment models |
| Engine code in title | ten engine codes; the great majority of ads state one | Engine/power must be a feature; titles are messy |
| Trim in title | seven trim levels, one dominant; **17% state none** | Trim extraction is noisy; fall back to equipment signals |
| Mileage | median 113,000 km; max 2,300,000 km | The max is surely a unit error (entered in the wrong unit): needs a cleaning rule |
| Missing | CO2 35%, weight 27%, owners 66%, mileage 0.2% | Use CO2/weight only as optional features; owners probably unusable |
| Generations | the oldest model years belong to an earlier generation | Structural break; add generation indicator or restrict sample |

## 3. Known pitfalls (decide up front, document each)

- **Asking price is not transaction price.** It is a ceiling, and stale ads that don't sell stay
  listed. We model asking prices; the "fair price" is a fair *asking* price.
- **VAT.** Prices include VAT for private buyers; some dealer ads carry an ex-VAT amount in the
  listing data. Check which price we hold and keep it consistent.
- **Equipment ticks are noisy.** Absence of a tick means "seller didn't list it", not "car lacks it".
  This measurement error biases equipment coefficients toward zero. Sellers using the manufacturer's factory
  lists (about a third of ads) tick far more than those who write by hand, so the number of ticks
  is partly a **seller-style proxy**. Mitigation: seller-style control, grouped features,
  and later the variant-based standard-equipment lookup.
- **Collinearity.** Equipment bundles by trim (trim levels). Individual features
  will be hard to separate; regularisation and grouped tests are needed, and we should report
  what is identified and what is not.
- **Dealer and chain effects.** Chains price differently (large dealer groups). Repeated
  dealer boilerplate and duplicate cars (2 duplicate VINs) can leak across folds.
- **Selection / survivorship.** Only unsold cars are visible. Effects are conditional on being
  listed; a price-vs-time-on-market analysis needs repeated crawls (future option).
- **Warranty included vs optional** is itself a priced feature; our Garanti column is a rule-based
  summary (about 85–90% accurate on review), so treat it as a noisy regressor.

## 4. Workflow (phases, each ends with a logged decision)

**Phase 0 — Data audit and freeze.** Build a clean modelling table `model/data/model_table.parquet`
(or csv) from `ads.json` with a documented cleaning log: price 0, mileage unit errors,
registration-date gaps, duplicate VINs, impossible values. Record the snapshot date and
row counts before/after each rule. Write `DATA_DICTIONARY.md`.

**Phase 1 — Validation design (fixed before looking at results).**
- Target: `log(price)`; report errors also in price units and in %.
- Primary scheme: **repeated GroupKFold by seller** (5 folds × 3 repeats). This measures how well
  we price a car at a dealer the model has not seen, which is the honest case for judging ads.
- Locked **hold-out set** (about 15% of sellers, drawn once with a fixed seed) used only for the final
  comparison, never for tuning.
- Fair values for all ads come from **out-of-fold predictions**, so no ad is priced by a model that saw it.
- Metrics: RMSE of log price, MAE in price units, MAPE, R², calibration of intervals, and error by segment
  (fuel type, age band, price band).

**Phase 2 — Baselines.** Mean-by-year; OLS on age + mileage + power; then + fuel/drive/trim.
These set the bar; every complex model must beat them by a stated margin.

**Phase 3 — Feature engineering (feature sets A–F, nested).**
- A: age (years from registration date to snapshot), mileage in km (log and linear), power.
- B: + fuel type, drive, engine code (anonymised labels), generation, facelift/infotainment era.
- C: + trim (parsed from title, with "unknown" level), colour, body/seats, seller type.
- D: + equipment ticks (144, plus category counts, plus `x*` flagged separately).
- E: + warranty (included months, optional), seller-style proxy, region.
- F: + dealer fixed/random effects or chain indicators.
- Interactions to test: age × mileage, fuel × age, fuel × mileage, trim × age.
- Nonlinearities: splines or piecewise terms for age and mileage; check the depreciation curve visually.

**Phase 4 — Model families.**
1. *Structural / interpretable:* OLS on log price (HC3 and seller-clustered standard errors);
   with splines; mixed model with seller random effect if feasible.
2. *Regularised linear:* ridge, lasso, elastic net with CV-chosen penalty; group lasso style
   by equipment category via stability selection.
3. *Trees:* random forest, gradient boosting (scikit-learn HistGradientBoosting; xgboost is installed),
   with monotonic constraints on age and mileage in one variant.
4. *Quantile / interval models:* quantile gradient boosting or conformal intervals for the fair-price band.
5. *Car + dealer (hierarchical) models:* log price = car effect + dealer effect. See section 4b.
6. *Ensembles:* simple average of the best linear and best tree model, if it helps.
LightGBM 4.7 and SHAP 0.49 were installed on 2026-10-17 (in addition to xgboost); SHAP gives
per-car explanations of the fair price. Permutation importance and partial dependence stay as
cross-checks.

### 4b. Car + dealer model ("this car should cost X, but this dealer charges X + 5%")

- **Specification:** `log(price_ij) = f(car_ij) + u_j + e_ij`, where `u_j` is a dealer-specific
  intercept (random effect, shrunk toward zero for dealers with few ads) and `f` is the car part
  (linear or boosted trees). Implemented as (a) a mixed model (statsmodels MixedLM) for the linear `f`,
  and (b) for tree models, a two-step/stacked version: fit `f` out-of-fold, then estimate shrunken
  dealer effects from the residuals, iterating once or twice (backfitting).
- **Outputs per ad:** car-only fair price X; dealer premium `u_j` in % with an interval; and the
  full prediction X × (1 + premium). For dealers we have never seen, the premium is 0 with the
  population spread as uncertainty.
- **Validation (two questions, two schemes):** (1) unseen dealers: grouped CV by seller, car part only;
  (2) known dealers: hold out random ads *within* dealers, use the dealer effect, and check whether it
  improves accuracy over the car-only model. The gain from (2) over (1) is the value of knowing the dealer.
- **Interpretation caveat:** the dealer premium is not pure pricing power. It also absorbs anything
  we do not observe, such as condition, service history, certified status, higher-quality
  reconditioning, included warranty or a better location. We report it as "price relative to
  comparable cars", not as overcharging.
- **Limits of the data:** 709 sellers, many with only 1–2 ads, so most dealer effects are weakly
  identified. Chains (large dealer groups) have many ads, so a *chain-level* effect
  is estimated more reliably than the effect of each branch; we will test a chain level above
  the branch level (nested random effects).

**Phase 5 — Tuning and model selection.** Nested CV for hyperparameters; compare on the repeated
GroupKFold; one-standard-error rule to prefer the simpler model; final check once on the hold-out.

**Phase 6 — Inference for question 1.** Report, with confidence intervals (cluster bootstrap by seller):
- price change (% and price units) for +1 year of age, +10,000 km, at a few reference cars
  (e.g. 3 years / 4 years old, different fuels);
- value of key equipment (panoramic roof, 360 camera, Harman Kardon, heated steering wheel,
  adaptive cruise, air suspension ...), flagged with how well each is identified;
- value of included warranty months;
- comparison of effects between the linear model and tree-based partial dependence.

**Phase 7 — Fair values for question 2.** For every ad: predicted fair asking price, 80% interval,
gap in price units and %, and a "deal score". Review the largest apparent bargains by hand (the usual
explanation is missing information: damage, high ownership count, data errors).
Add columns to the workbook (`Rimligt pris`, `Avvikelse %`) and a `Modell` sheet with coefficients.

**Phase 8 — Robustness.** Drop private sellers; drop the first generation; drop plug-ins and
refit; winsorise prices; different mileage cleaning; alternative trim parsing; seller-clustered vs
chain-clustered errors; stability of coefficients across folds.

**Phase 9 — Report and handover.** `REPORT.md` (and an HTML version) with method, results,
limitations and how to refresh the model on a new crawl.

## 5. Documentation and reproducibility

```
model/
  PLAN.md                 this file (versioned; changes recorded below)
  README.md               how to rerun everything, environment, folder map
  DATA_DICTIONARY.md      every modelling column: source, unit, cleaning, missingness
  EXPERIMENT_LOG.md       one entry per experiment (template below)
  CLEANING_LOG.md         every dropped/changed row with reason
  src/                    numbered scripts (00_clean.py, 01_baselines.py, ...) + shared helpers
  results/                metrics tables (csv), coefficient tables, OOF predictions, figures
  REPORT.md               final write-up
```

**Experiment log template** (one entry per run, written *before* looking at the result where possible):
`ID | date | hypothesis and expected sign | feature set | model and hyperparameters | CV scheme and seed |
metrics (mean ± sd over folds) | decision (keep/drop/investigate) | file references`.

Rules: fixed random seeds; every script reads the same frozen table; results written to files, not
only printed; no tuning on the hold-out; any change to the plan is appended to the change log below.

## 6. Decisions I need from you (defaults in bold)

1. **Scope:** model all powertrains together with interactions, plus segment models for the
   categories you care about, or restrict to a segment (e.g. petrol/mild-hybrid, similar to the ad you added)?
   Default: all, then segments.
2. **Installs:** decided — SHAP and LightGBM installed (see change log).
3. **Crawl refresh:** use today's snapshot only (default), or recrawl later to add time-on-market and price-change signals?

## 7. Change log

- 2026-10-17: v1 drafted.
- 2026-10-17: installed LightGBM 4.7.0 and SHAP 0.49.1 into the base Anaconda environment. The first
  `pip install` upgraded numpy to 2.2.6, which broke pandas, scikit-learn, statsmodels, matplotlib and
  xgboost; numpy was restored to 1.26.4 (the Anaconda version) and SHAP pinned to 0.49.1, after which all
  imports work. numba now reports 0.67.0 (Anaconda's own record says 0.60.0).
  Pre-existing pip conflicts (rembg, streamlit) were not caused by this.
- 2026-10-17: added section 4b (car + dealer hierarchical model).
- 2026-10-17: user approved scope defaults (all fuel types together, today's snapshot) and asked to run Phases 1–9 to the report.
- 2026-10-17: phases executed; results in `EXPERIMENT_LOG.md` (and a private report). Deviations from the plan: (a) chains defined by the data source's dealer-group id after a name-based
  rule proved spurious; (b) extended tree features adopted after E11 (decision rule fixed beforehand); (c) the interval/z-score uses a heteroskedastic scale
  because errors double for 200,000+ km cars; (d) per-ad SHAP uses a model that excludes the ad. Not done: monotone constraints for the final blend, quantile-regression intervals
  (empirical standardised-residual quantiles used instead), a second crawl for time-on-market.
