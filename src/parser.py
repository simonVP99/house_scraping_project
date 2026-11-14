"""HTML -> structured dataset extraction for Immoweb classified pages.

Design notes
------------
*   Extraction uses a raw string scan + ``json.JSONDecoder.raw_decode`` instead of
    BeautifulSoup.  The only thing we need from the DOM is the payload of
    ``window.classified = {...}``, and building a full element tree to find one
    script tag costs ~25ms/file (~1 hour over the full corpus) versus ~0.3ms.
    ``raw_decode`` also terminates on the real closing brace, so it is immune to
    descriptions containing ``};`` where the old non-greedy regex would truncate.
*   Immoweb sends ``null`` (not a missing key) for amenities the agency did not
    fill in.  ``null`` means *unknown*, not *absent*, so tri-state flags are
    preserved as ``None`` instead of being coerced to ``False``.  Downstream
    code (``preprocess_functions.convert_boolean_columns``) adds explicit
    ``*_missing`` indicators so a model can still tell "no garden" from
    "garden not declared".
*   When the form field is ``null`` we try to recover the amenity from the
    free-text description and record which columns were recovered, so the
    provenance stays auditable.
"""

import os
import re
import sys
import json
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# =============================================================================
# Text helpers
# =============================================================================

# Control characters (incl. NUL) and zero-width junk that would otherwise end up
# in the CSV.  A single NUL byte makes Python's ``csv`` module reject the whole
# file, so this matters more than it looks.
_CONTROL_CHARS = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_WHITESPACE = re.compile(r'[ \t\u00a0]+')


def clean_text(value, max_len=None):
    """Strip control characters and collapse whitespace. Returns None if empty.

    ``max_len`` truncates on a word boundary when possible.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    value = _CONTROL_CHARS.sub('', value)
    value = _WHITESPACE.sub(' ', value).strip()
    if not value:
        return None
    if max_len is not None and len(value) > max_len:
        cut = value[:max_len]
        space = cut.rfind(' ')
        if space > max_len * 0.8:
            cut = cut[:space]
        value = cut.strip()
    return value or None


def tri_state_bool(value):
    """Coerce to True/False/None without inventing a value for unknown."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value == 1:
            return True
        if value == 0:
            return False
        return None
    if isinstance(value, str):
        s = value.strip().lower()
        if s in ('true', '1', 'yes', 'y'):
            return True
        if s in ('false', '0', 'no', 'n'):
            return False
    return None


# =============================================================================
# Amenity recovery from free text
# =============================================================================

# NL / FR / EN keywords.  Matched case-insensitively on a description that the
# agency wrote by hand, so recall matters more than precision here.
_AMENITY_PATTERNS = {
    'has_garden': re.compile(r'\b(tuin|jardin|garden|voortuin|achtertuin)\b', re.I),
    'has_terrace': re.compile(r'\b(terras|terrasse|terrace|dakterras)\b', re.I),
    'has_swimming_pool': re.compile(r'\b(zwembad|piscine|swimming pool)\b', re.I),
    'has_basement': re.compile(r'\b(kelder|sous-sol|basement)\b', re.I),
    'has_attic': re.compile(r'\b(zolder|grenier|attic)\b', re.I),
    'has_lift': re.compile(r'\b(lift|ascenseur|elevator)\b', re.I),
}

# "geen tuin", "zonder terras", "sans jardin", "no garden", "pas de jardin"
_NEGATION = re.compile(
    r'\b(geen|zonder|sans|no|pas de|aucun[e]?|niet)\b[\s\w]{0,15}$', re.I)


def recover_amenities_from_text(description, row):
    """Fill tri-state *unknown* amenity flags from the description text.

    Only touches columns whose value is currently ``None`` -- an explicit form
    value always wins.  Returns the list of column names that were recovered.
    """
    recovered = []
    if not description:
        return recovered
    for column, pattern in _AMENITY_PATTERNS.items():
        if row.get(column) is not None:
            continue
        match = pattern.search(description)
        if not match:
            continue
        prefix = description[max(0, match.start() - 25):match.start()]
        if _NEGATION.search(prefix):
            continue
        row[column] = True
        recovered.append(column)
    return recovered


# =============================================================================
# City normalisation
# =============================================================================

# Case/whitespace variants of one locality ("Gent" vs "GENT") are collapsed.
# Distinct municipalities are deliberately NOT merged -- that is a modelling
# decision, not a parsing one.
_CITY_ALIASES = {
    'gent': 'Gent',
    'ghent': 'Gent',
    'gand': 'Gent',
}


def normalize_city(value):
    if not value or not isinstance(value, str):
        return None
    collapsed = _WHITESPACE.sub(' ', value).strip()
    if not collapsed:
        return None
    key = collapsed.lower()
    if key in _CITY_ALIASES:
        return _CITY_ALIASES[key]
    # Title-case only when the source is shouting (e.g. "WONDELGEM").
    if collapsed.isupper():
        return collapsed.title()
    return collapsed


def normalize_street(value):
    """Collapse whitespace and strip a trailing house number that leaked in.

    ``location.number`` already carries the house number separately, but some
    listings have it embedded in the street string.
    """
    if not value or not isinstance(value, str):
        return None
    collapsed = _WHITESPACE.sub(' ', value).strip().rstrip(',')
    if not collapsed:
        return None
    return collapsed or None


# =============================================================================
# JSON extraction
# =============================================================================

def extract_classified_json(html):
    """Pull the ``window.classified = {...}`` payload out of raw HTML.

    Uses ``raw_decode`` so the object ends at the real closing brace.  Returns
    a dict or ``None``.
    """
    decoder = json.JSONDecoder()
    search_from = 0
    while True:
        idx = html.find('window.classified', search_from)
        if idx == -1:
            return None
        eq = html.find('=', idx)
        # The assignment must be close to the identifier; otherwise we matched
        # an unrelated mention (e.g. inside a JS bundle).
        if eq != -1 and eq - idx < 40:
            brace = html.find('{', eq)
            if brace != -1 and brace - eq < 40:
                try:
                    obj, _ = decoder.raw_decode(html, brace)
                    if isinstance(obj, dict):
                        return obj
                except json.JSONDecodeError:
                    pass
        search_from = idx + len('window.classified')


def _safe_get(mapping, *keys, default=None):
    """First non-None value among ``keys`` in ``mapping``."""
    if not isinstance(mapping, dict):
        return default
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value
    return default


# =============================================================================
# File identity
# =============================================================================

_FILENAME_RE = re.compile(r'^full_source_(\d{8})\.html$', re.I)


def parse_file_identity(file_path):
    """Return ``(property_id, date_str)`` or ``(None, None)`` if unparseable.

    ``date_str`` is the DDMMYYYY tag from ``full_source_DDMMYYYY.html``.
    """
    directory = os.path.basename(os.path.dirname(file_path))
    file_name = os.path.basename(file_path)
    match = _FILENAME_RE.match(file_name)
    if not match:
        # Tolerate other names but still validate the tag shape.
        parts = file_name.split('_')
        if len(parts) < 3:
            return directory or None, None
        tag = parts[2].split('.')[0]
        return directory or None, tag if tag.isdigit() else None
    return directory or None, match.group(1)


# =============================================================================
# Main extraction
# =============================================================================

def extract_info(file_path):
    """Extract structured property data from one Immoweb HTML file.

    Returns a flat dict, or ``None`` if the file could not be parsed.
    """
    property_id, date_str = parse_file_identity(file_path)

    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as handle:
            html = handle.read()
    except OSError:
        return None

    data = extract_classified_json(html)
    if data is None:
        return None

    # --- Navigate to major JSON sections ---
    property_details = data.get('property') or {}
    price_data = data.get('price') or {}
    transaction_data = data.get('transaction') or {}
    sale_data = transaction_data.get('sale') or {}
    certificates_data = transaction_data.get('certificates') or {}
    publication_data = data.get('publication') or {}
    flags_data = data.get('flags') or {}
    statistics_data = data.get('statistics') or {}
    customers_data = data.get('customers') or []
    media_data = data.get('media') or {}

    location_data = property_details.get('location') or {}
    building_data = property_details.get('building') or {}
    energy_data = property_details.get('energy') or {}
    land_data = property_details.get('land') or {}
    construction_permit_data = property_details.get('constructionPermit') or {}
    kitchen_data = property_details.get('kitchen') or {}
    living_room_data = property_details.get('livingRoom') or {}

    description = (
        property_details.get('description')
        or (property_details.get('alternativeDescriptions') or {}).get('en')
        or (property_details.get('alternativeDescriptions') or {}).get('nl')
        or (property_details.get('alternativeDescriptions') or {}).get('fr')
    )

    row = {
        # ---- Property identification ----
        'property_id': property_id,
        'date_obtained': date_str,

        # ---- Pricing ----
        'price': _safe_get(price_data, 'mainValue', 'price'),
        'price_min': price_data.get('minRangeValue'),
        'price_max': price_data.get('maxRangeValue'),
        'price_old_value': price_data.get('oldValue'),

        # ---- Property type ----
        'property_type': _safe_get(property_details, 'type', default=data.get('type')),
        'property_subtype': property_details.get('subtype'),

        # ---- Location ----
        'postal_code': location_data.get('postalCode'),
        'city': normalize_city(location_data.get('locality')),
        'city_raw': clean_text(location_data.get('locality'), 80),
        'street': normalize_street(location_data.get('street')),
        'house_number': location_data.get('number'),
        'box': location_data.get('box'),
        'latitude': location_data.get('latitude'),
        'longitude': location_data.get('longitude'),
        'region': location_data.get('region'),
        'province': location_data.get('province'),
        'district': location_data.get('district'),
        'location_type': location_data.get('type'),
        'floor_number': location_data.get('floor'),

        # ---- Building ----
        'year_built': building_data.get('constructionYear'),
        'building_condition': building_data.get('condition'),
        'total_floors': building_data.get('floorCount'),
        'facade_count': building_data.get('facadeCount'),
        'street_facade_width': building_data.get('streetFacadeWidth'),

        # ---- Dimensions ----
        'livable_area': _safe_get(property_details, 'netHabitableSurface', 'indoor_surface'),
        'property_size': land_data.get('surface'),
        'kitchen_surface': kitchen_data.get('surface'),
        'living_room_surface': living_room_data.get('surface'),

        # ---- Rooms ----
        'number_of_rooms': property_details.get('roomCount'),
        'number_of_bedrooms': property_details.get('bedroomCount'),
        'number_of_bathrooms': property_details.get('bathroomCount'),
        'number_of_toilets': property_details.get('toiletCount'),
        'shower_room_count': property_details.get('showerRoomCount'),
        'fireplace_count': property_details.get('fireplaceCount'),

        # ---- Amenities (tri-state: True / False / None=unknown) ----
        'has_living_room': tri_state_bool(property_details.get('hasLivingRoom')),
        'has_dining_room': tri_state_bool(property_details.get('hasDiningRoom')),
        'has_attic': tri_state_bool(property_details.get('hasAttic')),
        'has_basement': tri_state_bool(property_details.get('hasBasement')),
        'has_terrace': tri_state_bool(property_details.get('hasTerrace')),
        'terrace_surface': property_details.get('terraceSurface'),
        'terrace_orientation': property_details.get('terraceOrientation'),
        'has_garden': tri_state_bool(property_details.get('hasGarden')),
        'garden_surface': property_details.get('gardenSurface'),
        'garden_orientation': property_details.get('gardenOrientation'),
        'has_swimming_pool': tri_state_bool(property_details.get('hasSwimmingPool')),
        'has_lift': tri_state_bool(property_details.get('hasLift')),
        'has_dressing_room': tri_state_bool(property_details.get('hasDressingRoom')),
        'has_air_conditioning': tri_state_bool(property_details.get('hasAirConditioning')),
        'has_caretaker': tri_state_bool(property_details.get('hasCaretakerOrConcierge')),
        'kitchen_type': kitchen_data.get('type'),
        'monthly_costs': property_details.get('monthlyCosts'),

        # ---- Parking ----
        'parking_indoor': property_details.get('parkingCountIndoor'),
        'parking_outdoor': property_details.get('parkingCountOutdoor'),
        'parking_closed_box': property_details.get('parkingCountClosedBox'),

        # ---- Energy ----
        'energy_label': certificates_data.get('epcScore'),
        'primary_energy_consumption': certificates_data.get('primaryEnergyConsumptionPerSqm'),
        'heating_type': energy_data.get('heatingType'),
        'has_photovoltaic_panels': tri_state_bool(energy_data.get('hasPhotovoltaicPanels')),
        'has_double_glazing': tri_state_bool(energy_data.get('hasDoubleGlazing')),
        'has_heat_pump': tri_state_bool(energy_data.get('hasHeatPump')),
        'has_thermic_panels': tri_state_bool(energy_data.get('hasThermicPanels')),

        # ---- Flood risk / planning ----
        'flood_zone_type': construction_permit_data.get('floodZoneType'),
        'p_score': construction_permit_data.get('pScore'),
        'g_score': construction_permit_data.get('gScore'),
        'permit_is_obtained': tri_state_bool(construction_permit_data.get('isObtained')),
        'permit_has_obligation_to_construct': tri_state_bool(
            construction_permit_data.get('hasObligationToConstruct')),
        'breaches_urban_planning': tri_state_bool(
            construction_permit_data.get('isBreachingUrbanPlanningRegulation')),

        # ---- Land (plot) attributes ----
        'land_is_flat': tri_state_bool(land_data.get('isFlat')),
        'land_is_wooded': tri_state_bool(land_data.get('isWooded')),
        'land_facing_street': tri_state_bool(land_data.get('isFacingStreet')),
        'land_width': land_data.get('landWidth'),
        'land_use_designation': land_data.get('latestUseDesignation'),
        'land_sewer_connection': tri_state_bool(land_data.get('sewerConnection')),

        # ---- Transaction ----
        'transaction_type': transaction_data.get('type'),
        'transaction_subtype': transaction_data.get('subtype'),
        'cadastral_income': sale_data.get('cadastralIncome'),
        'is_furnished': tri_state_bool(sale_data.get('isFurnished')),
        'renovation_obligation': certificates_data.get('renovationObligation'),

        # ---- Demand signals ----
        # NOTE: view/bookmark counts accumulate over a listing's life, so they
        # proxy time-on-market.  Useful for an attractiveness model scored late
        # in a listing's life; unsafe as a launch-time price feature.
        'view_count': statistics_data.get('viewCount'),
        'bookmark_count': statistics_data.get('bookmarkCount'),

        # ---- Media ----
        'picture_count': len(media_data.get('pictures') or []),
        'floor_plan_count': len(media_data.get('floorPlans') or []),
        'has_virtual_tour': tri_state_bool(media_data.get('virtualTourUrl')),

        # ---- Publication ----
        # publication_date is the true first-publication timestamp (100% present);
        # real time-on-market can be derived from it.
        'publication_date': publication_data.get('creationDate'),
        'last_modification_date': publication_data.get('lastModificationDate'),
        'publication_expiration_date': publication_data.get('expirationDate'),
        'external_reference': clean_text(data.get('externalReference'), 120),
        'immo_office_name': clean_text(
            (customers_data[0] or {}).get('name'), 120) if customers_data else None,

        # ---- Flags ----
        'is_biddit_sale': tri_state_bool(flags_data.get('isAnInteractiveSale')),
        'is_sold_or_rented': tri_state_bool(flags_data.get('isSoldOrRented')),
        'is_under_option': tri_state_bool(flags_data.get('isUnderOption')),
        'is_public_sale': tri_state_bool(flags_data.get('isPublicSale')),
        'is_newly_built': tri_state_bool(flags_data.get('isNewlyBuilt')),
        'is_life_annuity_sale': tri_state_bool(flags_data.get('isLifeAnnuitySale')),

        # ---- Text ----
        # Control characters (incl. NUL) are stripped at the source: a single
        # NUL byte makes Python's csv module reject the entire output file.
        'title': clean_text(property_details.get('title'), 500),
        'description': clean_text(description),
    }

    # ---- Recover amenities the agency left blank, from the description ----
    recovered = recover_amenities_from_text(row.get('description'), row)
    row['text_recovered_flags'] = '|'.join(sorted(recovered)) if recovered else None

    return row


# =============================================================================
# Folder parsing
# =============================================================================

def collect_html_files(root_directory):
    """Return a sorted list of HTML file paths under ``root_directory``.

    Sorted so that ``stop_after`` selects a stable subset across runs.
    """
    paths = []
    for dirpath, _dirnames, filenames in os.walk(root_directory):
        for filename in filenames:
            if filename.endswith('.html'):
                paths.append(os.path.join(dirpath, filename))
    paths.sort()
    return paths


def _atomic_write_json(path, payload):
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle)
    os.replace(tmp, path)


def load_manifest(manifest_file):
    """Load the set of ``property_id|date_obtained`` keys already parsed."""
    if not manifest_file or not os.path.exists(manifest_file):
        return set()
    try:
        with open(manifest_file, 'r', encoding='utf-8') as handle:
            payload = json.load(handle)
        return set(payload.get('parsed_keys', []))
    except (json.JSONDecodeError, OSError):
        return set()


def save_manifest(manifest_file, keys):
    if not manifest_file:
        return
    _atomic_write_json(manifest_file, {'parsed_keys': sorted(keys)})


def parse_data_folder_parallel(root_directory, output_file, stop_after=None,
                               failures_file=None, manifest_file=None,
                               incremental=False, workers=None):
    """Parse every HTML file under ``root_directory`` into a DataFrame.

    Args:
        root_directory: Folder holding ``<property_id>/full_source_*.html``.
        output_file: CSV path used as a fallback when no files are found.
        stop_after: Optional cap on how many files are parsed *in this run*.
            Applied after incremental filtering, so with ``incremental=True``
            it means "parse at most N files that are not yet parsed" rather
            than "look at the first N files".
        failures_file: Optional CSV path for a per-file failure manifest.
        manifest_file: Optional JSON path tracking already-parsed files.
        incremental: When True, skip files recorded in ``manifest_file``.
        workers: Optional worker-process count.

    Returns:
        pd.DataFrame of extracted rows.
    """
    all_files = collect_html_files(root_directory)
    print(f"Found {len(all_files)} HTML files.")

    parsed_keys = load_manifest(manifest_file) if incremental else set()
    if parsed_keys:
        before = len(all_files)
        all_files = [f for f in all_files
                     if _manifest_key_for(f) not in parsed_keys]
        print(f"Incremental mode: skipping {before - len(all_files)} already-parsed files.")

    if stop_after is not None:
        all_files = all_files[:stop_after]

    if not all_files:
        print("No new HTML files to process.")
        if incremental and manifest_file and parsed_keys:
            print("Nothing new; keeping existing output unchanged.")
            return pd.DataFrame()
        try:
            return pd.read_csv(output_file)
        except FileNotFoundError:
            return pd.DataFrame()

    print(f"Parsing {len(all_files)} HTML files in parallel...")
    rows = []
    failures = []

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(extract_info, path): path for path in all_files}
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="Parsing HTML files",
                           disable=not sys.stderr.isatty()):
            path = futures[future]
            try:
                result = future.result()
                if result:
                    rows.append(result)
                else:
                    failures.append({'file': path, 'reason': 'no_classified_payload'})
            except Exception as exc:  # noqa: BLE001 - report, don't abort the run
                failures.append({'file': path, 'reason': f'{type(exc).__name__}: {exc}'})

    print(f"Parsed {len(rows)} files, {len(failures)} failures "
          f"({len(failures) / max(len(all_files), 1) * 100:.3f}%).")

    if failures:
        if failures_file:
            pd.DataFrame(failures).to_csv(failures_file, index=False)
            print(f"  Failure manifest written to {failures_file}")
        else:
            for item in failures[:10]:
                print(f"  FAILED {item['file']}: {item['reason']}")

    if manifest_file:
        new_keys = set(parsed_keys)
        for row in rows:
            if row.get('property_id') and row.get('date_obtained'):
                new_keys.add(f"{row['property_id']}|{row['date_obtained']}")
        save_manifest(manifest_file, new_keys)
        print(f"  Manifest now tracks {len(new_keys)} parsed files.")

    return pd.DataFrame(rows)


def _manifest_key_for(file_path):
    property_id, date_str = parse_file_identity(file_path)
    return f"{property_id}|{date_str}"
