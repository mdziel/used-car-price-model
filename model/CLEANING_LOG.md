# Cleaning log

Rules applied to the raw collection (source-specific parsing code is not included). Rules run in order; counts are rows hit by each rule at that point.

| Rule | Description | Ads | Action |
|---|---|---:|---|
| R0 | ads in the raw collection | 3012 | start |
| R1 | price missing or zero | 1 | dropped |
| R2 | export-net price wording in title or description | 1 | dropped |
| R3 | damaged / sold-as-is wording in title or description | 4 | dropped |
| R4 | same chassis number listed twice; keep the most recently updated ad | 2 | dropped |
| R5 | mileage above 500,000 km (values entered in the wrong unit) | 9 | corrected (divided by 10), flag mileage_fixed=1 |
| R6 | mileage missing | 7 | dropped |
| R7 | registration date missing or inconsistent with model year | 16 | age taken from model year, flag age_from_model_year=1 |
| R8 | very low mileage (< 1,000 km) on a car older than 1.5 years | 1 | dropped |
| R9 | fuel, power or model year missing | 1 | dropped |
| END | ads in the modelling table | 2995 | final |

Notes: asking prices are VAT-inclusive throughout (ads with an ex-VAT amount show a constant fraction of the price); text about excluding VAT is leasing wording and is not treated as a different price basis. All nine mileage corrections are private ads; after correction the values are plausible for the age.
