# Data dictionary — `data/example_table.csv`

One row per ad (2,995 rows, snapshot 2026-10-16/17). Values are anonymised or rounded as described in `INPUT_FORMAT.md`.

| Column | Unit | Meaning |
|---|---|---|
| `ad_id` | id | random identifier (not the source's ad number) |
| `price` | price units | asking price incl. VAT (the regression target is its log) |
| `model_year` | year | model year |
| `age_years` | years | age at the snapshot (from the registration date; from the model year if that date was missing or inconsistent) |
| `km` | km | odometer, rounded to 100 km |
| `power_hp` | hp | engine power |
| `fuel` | category | Petrol, Diesel, Plug-in hybrid petrol, Hybrid petrol (mild), Hybrid diesel |
| `drive` | category | All-wheel drive, Front-wheel drive, Two-wheel drive |
| `engine_l`, `co2`, `weight_kg`, `seats`, `doors`, `boot_l`, `tow_kg` | various | specifications (some often missing: owners 66%, CO2 32%, weight 27%) |
| `colour` | category | colour |
| `owners` | count | number of owners, if stated |
| `engine_code` | category | anonymised engine label ("Engine 1" ...), parsed from the ad title; `unknown` if not stated |
| `trim` | category | invented trim label (Basic, Classic, Comfort, Elite, Executive, Prestige, Sport), parsed from the ad title; `unknown` if not stated (17% of ads) |
| `seller_id` | id | anonymous dealer code, or `private_<id>` for private sellers (grouping key for dealer effects and CV) |
| `is_private` | 0/1 | private seller |
| `group_id` | id | anonymous dealer-group code (empty if none) |
| `location` | id | anonymous location code |
| `dist_big3` | km | distance to the nearest of the three largest cities, rounded to 10 km |
| `vat_deductible` | 0/1 | company car with an ex-VAT price shown |
| `n_equip_raw` | count | number of equipment lines the seller listed (a seller-style proxy) |
| `n_eq_ticked` | count | number of the 146 equipment features matched |
| `service_followed` | 0/1 | ad states the service programme has been followed |
| `right_to_exchange` | 0/1 | ad offers a 30-day right to exchange |
| `known_damage_yes` | 0/1 | seller reports known damage |
| `flag_damage_weak` | 0/1 | ad text mentions a defect or spare parts |
| `mileage_fixed` | 0/1 | the odometer value was corrected (entered in the wrong unit) |
| `age_from_model_year` | 0/1 | age was derived from the model year |
| `tuned_yes` | 0/1 | seller reports a tuned engine |
| `warranty_incl_months` | months | longest included warranty found in the ad text (empty = none/unknown) |
| `warranty_option_max_months` | months | longest optional warranty offered |
| `warranty_mentioned`, `warranty_included_any`, `warranty_option_offered` | 0/1 | warranty information present / something included / an option or offer |
| `warranty_certified`, `warranty_trade_assoc` | 0/1 | manufacturer-certified warranty / trade-association warranty mentioned |
| `eq_<feature>` (146 columns) | 0/1 | the seller listed this equipment feature (after mapping the seller's wording to a fixed list). 0 means *not listed*, not *absent* |

Derived inside `lib.py` (not stored): `km10k`, `fuel_grp`, `awd`, `gen1` (model year ≤ 2017), `my_ge2022`, `colour_grp` (top eight colours), `chain`, warranty threshold flags.
