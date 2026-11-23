# Preprocessing Improvements Design

## Goal

Improve `src/preprocess_functions.py` to handle 33 new parser fields, fix data quality issues, and improve filtering logic while preserving data quantity (~88% retention).

## Design Decisions

- Keep newly built properties in, flag with `is_newly_built` feature
- Only include new fields with >30% fill rate
- Two-stage outlier filtering: hard geographic/logical bounds first, then percentile
- Use `publication_date` for listing duration instead of `date_obtained`
- Parser stays pure extraction, all cleaning in preprocessor

## Pipeline Steps

### Stage 1: Data Cleaning & Filtering

1. **Parse dates** - Convert date_obtained, publication_date, last_modification_date to datetime
2. **Listing duration** - `last_modification_date - publication_date` (replaces date_obtained groupby)
3. **Deduplicate** - Keep latest record per property_id
4. **Drop non-modelable** - Remove life annuity (10) and public sales (109)
5. **Hard bounds filtering** - Lat 50.9-51.2, Lon 3.5-4.0, area 10-1000 sqm, price >25K. Drop missing price/area/coords.

### Stage 2: Feature Engineering

6. **Age** - `datetime.now().year - year_built` (dynamic, not hardcoded)
7. **Price per sqm** - `price / livable_area`
8. **Scale price** - Divide by 1000
9. **Booleans** - All `has_*` and `is_*` flags to bool, NaN → False
10. **New numeric fields** - p_score, g_score, monthly_costs, floor_number, kitchen_surface, shower_room_count, facade_count
11. **New categorical fields** - renovation_obligation, location_type → fill Unknown
12. **Drop dead columns** - parking_closed_box, terrace_orientation, has_dining_room, transaction_type, region, province, district, external_reference, title, description

### Stage 3: Imputation & Spatial

13. **Poisson GLM imputation** - bedrooms → toilets → bathrooms → cadastral_income (enriched predictors)
14. **Categorical fill** - heating_type, building_condition, flood_zone_type, renovation_obligation, location_type, kitchen_type, energy_label → 'Unknown'
15. **One-hot encode** property_type → property_type_HOUSE
16. **Neighbourhood** spatial join (unchanged)
17. **OSM street type** matching (unchanged)
18. **Distance from center** (unchanged)
19. **Percentile outlier filtering** - log_price and price_sq at 99th percentile

## Expected Output

~17,000-17,500 rows, ~50+ features, 88% retention from deduplicated set.
