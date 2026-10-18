# Input format

The pipeline reads one modelling table (CSV, one row per ad), by default `data/example_table.csv`; set `MODEL_TABLE=/path/to/table.csv` to use another file.
`python model/src/00_validate_table.py [table.csv]` checks that a table has what the scripts expect. Every column is described in `DATA_DICTIONARY.md`.

## Required columns

`ad_id` (unique), `price` (asking price incl. VAT, > 0), `model_year`, `age_years`, `km` (odometer in km), `power_hp`, `fuel`, `drive`, `engine_code`, `trim`, `colour`,
`seller_id`, `is_private`, `group_id`, `dist_big3`, `vat_deductible`, `n_equip_raw`, `n_eq_ticked`, `service_followed`, `right_to_exchange`, `known_damage_yes`,
`warranty_incl_months`, `warranty_option_max_months`, `warranty_mentioned`, `warranty_certified`, `warranty_trade_assoc`, `warranty_option_offered`, `owners`, `co2`, `weight_kg`, `engine_l`,
and any number of equipment flag columns `eq_<feature name>` (0/1; 1 = the seller listed the feature).

Value sets: `fuel` ∈ {`Petrol`, `Diesel`, `Plug-in hybrid petrol`, `Hybrid petrol (mild)`, `Hybrid diesel`}; `drive` = `All-wheel drive` for AWD (any other label means not AWD);
`trim`, `engine_code` and `colour` are free category labels (use `unknown` for missing). The first label in alphabetical order is the base level in the regressions
(in the example data: trim "Basic"). `seller_id` is the grouping key for dealer effects and for the grouped cross-validation (one id per dealer; one id per ad for private sellers).

## Optional columns

`title_spec` (free-text title; adds binary title keywords to the boosted models), `seats`, `doors`, `boot_l`, `tow_kg`, `flag_damage_weak`, `mileage_fixed`,
`age_from_model_year`, `tuned_yes` (0/1 flags used to exclude doubtful ads from the rankings), `location` (any label).

## The example table

`data/example_table.csv` is an anonymised extract of the data behind the reported results: random ad IDs, shuffled rows, anonymous dealer / dealer-group / location codes,
invented trim labels, anonymised engine labels, mileage rounded to 100 km, distance to the nearest big city rounded to 10 km. It has no titles, so the title keywords are skipped.

## Building a table from your own source

Typical steps (the source-specific code is not part of this repository): convert odometer units to km; compute `age_years` from the registration date; map the seller's equipment lines to a fixed
feature list and write one 0/1 column per feature; summarise warranty information into the numeric `warranty_*` columns; set `seller_id` per dealer; drop or correct implausible rows
(see `CLEANING_LOG.md` for the rules used).
