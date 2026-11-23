# Parser Improvements Design

## Goal

Improve `src/parser.py` to extract all available fields from the `window.classified` JSON in Immoweb HTML files. Fix existing bugs. Keep parser as pure extraction (no validation/cleaning).

## Bug Fixes

### 1. `floor_number` always null
- **Root cause:** Line 124 reads `property_details.get('floor')` but JSON stores it at `property.location.floor`
- **Fix:** Change to `location_data.get('floor')`

### 2. `parking_closed_box` always null
- **Action:** Verify correct JSON path exists; fix path or remove column if field doesn't exist in Immoweb's data model

## New Fields

### Tier 1: 100% Fill Rate (12 fields)

| Field Name | JSON Path | Type |
|------------|-----------|------|
| `publication_date` | `publication.creationDate` | string (ISO datetime) |
| `last_modification_date` | `publication.lastModificationDate` | string (ISO datetime) |
| `transaction_type` | `transaction.type` | string |
| `transaction_subtype` | `transaction.subtype` | string |
| `region` | `property.location.region` | string |
| `province` | `property.location.province` | string |
| `district` | `property.location.district` | string |
| `is_sold_or_rented` | `flags.isSoldOrRented` | bool |
| `is_under_option` | `flags.isUnderOption` | bool |
| `is_public_sale` | `flags.isPublicSale` | bool |
| `is_newly_built` | `flags.isNewlyBuilt` | bool |
| `is_life_annuity_sale` | `flags.isLifeAnnuitySale` | bool |

### Tier 2: 50-80% Fill Rate (9 fields)

| Field Name | JSON Path | Type |
|------------|-----------|------|
| `external_reference` | `externalReference` | string |
| `p_score` | `property.constructionPermit.pScore` | numeric |
| `g_score` | `property.constructionPermit.gScore` | numeric |
| `renovation_obligation` | `transaction.certificates.renovationObligation` | string/bool |
| `has_lift` | `property.hasLift` | bool |
| `has_heat_pump` | `property.energy.hasHeatPump` | bool |
| `has_thermic_panels` | `property.energy.hasThermicPanels` | bool |
| `facade_count` | `property.building.facadeCount` | int |

### Tier 3: 20-50% Fill Rate (12 fields)

| Field Name | JSON Path | Type |
|------------|-----------|------|
| `kitchen_surface` | `property.kitchen.surface` | numeric |
| `living_room_surface` | `property.livingRoom.surface` | numeric |
| `shower_room_count` | `property.showerRoomCount` | int |
| `has_dressing_room` | `property.hasDressingRoom` | bool |
| `has_air_conditioning` | `property.hasAirConditioning` | bool |
| `has_caretaker` | `property.hasCaretakerOrConcierge` | bool |
| `street_facade_width` | `property.building.streetFacadeWidth` | numeric |
| `terrace_orientation` | `property.terraceOrientation` | string |
| `is_furnished` | `transaction.sale.isFurnished` | bool |
| `location_type` | `property.location.type` | string |
| `monthly_costs` | `property.monthlyCosts` | numeric |

## Code Organization

Regroup `extract_info()` field extraction into labeled sections:
1. Property identification (id, date, type)
2. Pricing (price, min, max)
3. Location (address, coordinates, region, province, district)
4. Building (condition, year, floors, facades, facade width)
5. Rooms (bedrooms, bathrooms, toilets, showers, kitchen, living room)
6. Amenities (terrace, garden, pool, lift, AC, dressing room, caretaker)
7. Energy (label, consumption, heating, solar, heat pump, glazing)
8. Parking (indoor, outdoor, closed box)
9. Transaction (type, subtype, cadastral income, furnished, renovation, flood)
10. Publication (dates, office, reference)
11. Flags (sold, option, public sale, new build, biddit, life annuity)
12. Text (title, description)

## What Does NOT Change

- `parse_data_folder_parallel()` logic
- `ProcessPoolExecutor` parallelization
- No validation or cleaning in parser (stays in `preprocess_functions.py`)
- Output format: flat dict -> DataFrame -> CSV

## Post-Update Steps

1. Re-run `run_parsing_only.py` to re-parse all 457K HTML files (~20 min)
2. Update `preprocess_functions.py` to handle new fields downstream
