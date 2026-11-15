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
import polars as pl
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


def _surface_values(bedroom_list):
    """Numeric surfaces from the ``property.bedrooms`` array, dropping junk."""
    out = []
    for bedroom in bedroom_list or []:
        if not isinstance(bedroom, dict):
            continue
        value = bedroom.get('surface')
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
            out.append(float(value))
    return out


def _sum_bedroom_surfaces(bedroom_list):
    values = _surface_values(bedroom_list)
    return sum(values) if values else None


def _min_bedroom_surface(bedroom_list):
    values = _surface_values(bedroom_list)
    return min(values) if values else None


def _max_bedroom_surface(bedroom_list):
    values = _surface_values(bedroom_list)
    return max(values) if values else None


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


def _nested_number(mapping, parent, child, default=None):
    """Read ``mapping[parent][child]`` as a scalar.

    Immoweb wraps several surfaces in a one-key object (``office: {"surface": 40}``
    rather than ``office: 40``).  Returning the dict itself would put a Struct
    column in the dataset, which breaks CSV export and is useless downstream.
    """
    node = mapping.get(parent) if isinstance(mapping, dict) else None
    if isinstance(node, dict):
        value = node.get(child)
        return value if isinstance(value, (int, float)) else default
    if isinstance(node, (int, float)):
        return node
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

# Auto-generated boilerplate Immoweb writes into the description field when the
# agency supplies nothing. They are 25-37 characters, so a length-only check
# accepts them and discards the real prose in an alternative language.
_DESCRIPTION_STUBS = re.compile(
    r'^(?:renovation obligation|obligation de r[ée]novation|'
    r'renovatieverplichting)\s*:\s*[\w\s.\-/]*$',
    re.I)


def _usable_text(value, min_chars=20):
    """True if ``value`` carries real prose rather than a blank/placeholder.

    Agencies submit ``'-'``, ``'.'`` or a single space for "no description".
    Those are truthy, so a plain ``or`` chain stops on them and discards the
    real text sitting in an alternative language: measured on 500 listings,
    46 of 171 (9.2% of the sample) had a placeholder main description with a
    1-2k character alternative that was being thrown away.

    Immoweb also emits auto-generated stubs such as
    ``'Renovation obligation: no'``.  Those clear a length check but carry no
    listing content, so they are rejected by pattern rather than by size.
    """
    if not isinstance(value, str):
        return False
    stripped = value.strip()
    if len(stripped) < min_chars:
        return False
    if not any(ch.isalnum() for ch in stripped):
        return False
    if _DESCRIPTION_STUBS.match(stripped):
        return False
    return True


def pick_description(property_details):
    """Best available long-form description across the language fields."""
    if not isinstance(property_details, dict):
        return None
    alternatives = property_details.get('alternativeDescriptions') or {}
    candidates = [
        property_details.get('description'),
        alternatives.get('nl') if isinstance(alternatives, dict) else None,
        alternatives.get('en') if isinstance(alternatives, dict) else None,
        alternatives.get('fr') if isinstance(alternatives, dict) else None,
    ]
    for candidate in candidates:
        if _usable_text(candidate):
            return candidate
    # Nothing passed the length/placeholder bar: keep the longest non-empty
    # value so short-but-real descriptions are not silently dropped.
    non_empty = [c for c in candidates if isinstance(c, str) and c.strip()]
    return max(non_empty, key=len) if non_empty else None


def pick_description_language(property_details, chosen):
    """Which language field ``chosen`` came from, for embedding provenance.

    Reciprocal to pick_description: the merged ``description`` column keeps one
    language only, and an embedder that is told the language can avoid
    code-switching artefacts when listings mix NL and FR.
    """
    if not isinstance(chosen, str) or not chosen.strip():
        return None
    if not isinstance(property_details, dict):
        return None
    alternatives = property_details.get('alternativeDescriptions') or {}
    if not isinstance(alternatives, dict):
        alternatives = {}
    if chosen == property_details.get('description'):
        return 'main'
    for language in ('nl', 'en', 'fr'):
        if chosen == alternatives.get(language):
            return language
    return 'main'


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

    return build_row(data, property_id, date_str)


def build_row(data, property_id, date_str):
    """Turn a ``window.classified`` payload into one flat row.

    Separated from :func:`extract_info` so a row can be rebuilt from a stored
    payload without a page on disk -- see ``snapshot_store``, which keeps the
    payload at ~0.3% of the HTML size. ``extract_info`` is still the HTML entry
    point and every existing caller is unaffected.
    """
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
    sections_data = data.get('hasSectionsArray') or {}
    alternative_text = property_details.get('alternativeDescriptions') or {}
    if not isinstance(alternative_text, dict):
        alternative_text = {}
    kitchen_data = property_details.get('kitchen') or {}
    living_room_data = property_details.get('livingRoom') or {}
    certificates_property_data = property_details.get('propertyCertificates') or {}
    specificities_data = property_details.get('specificities') or {}
    attic_data = property_details.get('attic') or {}
    basement_data = property_details.get('basement') or {}
    bedroom_list = property_details.get('bedrooms') or []
    investor_data = transaction_data.get('investor') or {}

    description = pick_description(property_details)
    description_language = pick_description_language(property_details, description)

    row = {
        # ---- Property identification ----
        'property_id': property_id,
        'date_obtained': date_str,

        # ---- Pricing ----
        'price': _safe_get(price_data, 'mainValue', 'price'),
        # Unpriced listings are real: price.type == 'group_sale' carries a range
        # (mainValue null, min/maxRangeValue set), and 'annuity_monthly_amount'
        # carries a monthly figure that must never be read as a sale price.
        'price_type': price_data.get('type'),
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
        'annex_count': building_data.get('annexCount'),

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

        # ---- Interior surfaces (per-room breakdown) ----
        # These are free whenever the agency filled the interior section, and
        # they allow a real consistency check: living + kitchen + bedrooms should
        # not exceed netHabitableSurface (observed to hold for 98.5% of listings).
        'bedroom_surface_sum': _sum_bedroom_surfaces(bedroom_list),
        'bedroom_surface_min': _min_bedroom_surface(bedroom_list),
        'bedroom_surface_max': _max_bedroom_surface(bedroom_list),
        'bedrooms_with_surface': sum(
            1 for b in bedroom_list
            if isinstance(b, dict) and b.get('surface') is not None),
        'attic_surface': attic_data.get('surface'),
        'basement_surface': basement_data.get('surface'),

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
        'land_sewer_connection_raw': clean_text(land_data.get('sewerConnection'), 40),
        'land_has_gas_water_electricity': tri_state_bool(
            land_data.get('hasGasWaterElectricityConnection')),
        'land_has_plot_to_rear': tri_state_bool(land_data.get('hasPlotToRear')),

        # ---- Property certificates (EPC-adjacent, agency-declared) ----
        'cert_has_electrical_installation': tri_state_bool(
            certificates_property_data.get(
                'hasElectricalInstallationComplianceCertificate')),
        'cert_has_asbestos': tri_state_bool(
            certificates_property_data.get('hasAsbestosCertificate')),
        'cert_built_plan_status': clean_text(
            certificates_property_data.get('builtPlanStatus'), 40),
        'cert_oil_tank_status': clean_text(
            certificates_property_data.get('oilTankCertificateStatus'), 40),
        'cert_global_insulation_level': certificates_property_data.get(
            'globalInsulationLevel'),
        'cert_primary_energy_consumption_level': certificates_property_data.get(
            'primaryEnergyConsumptionLevel'),

        # ---- Interior facilities (office / workspace) ----
        'has_office': tri_state_bool(specificities_data.get('hasOffice')),
        'office_surface': _nested_number(specificities_data, 'office', 'surface'),
        'has_workspace': tri_state_bool(specificities_data.get('hasWorkspace')),
        'workspace_surface': _nested_number(specificities_data, 'workspace', 'surface'),

        # ---- Availability ----
        'availability_period_type': clean_text(
            transaction_data.get('availabilityPeriodType'), 40),
        'availability_date': clean_text(transaction_data.get('availabilityDate'), 20),
        'is_subject_to_vat': tri_state_bool(sale_data.get('isSubjectToVat')),

        # ---- Investment (rental yield, rarely declared) ----
        'is_investment_property': tri_state_bool(
            investor_data.get('isInvestmentProperty')),
        'expected_monthly_rental_income': investor_data.get(
            'expectedMonthlyRentalIncome'),

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
        # Paid promotion tier: XL/L/M. Premium exposure is a legitimate price
        # signal, not a leak (it is known at listing time).
        'visualisation_option': clean_text(
            publication_data.get('visualisationOption'), 10),
        'external_reference': clean_text(data.get('externalReference'), 120),
        'immo_office_name': clean_text(
            (customers_data[0] or {}).get('name'), 120) if customers_data else None,

        # ---- Listing state (over a listing's life) ----
        # displayFlags.main is the canonical state machine: None / 'new' /
        # 'under_option'.  secondary carries new_construction, new_price,
        # life_annuity, notary_sale, biddit_sale, percent_sold.
        'display_flag_main': clean_text(
            ((data.get('displayFlags') or {}).get('main')), 30),
        'display_flags_secondary': '|'.join(
            sorted(str(x) for x in
                   ((data.get('displayFlags') or {}).get('secondary') or []))) or None,

        # ---- Form completeness ----
        # Which sections the agency filled in.  Doubles as the explanation for
        # *why* sparse fields are sparse, so missingness becomes a signal.
        'sections_filled_count': sum(
            1 for value in sections_data.values() if value is True),
        # The count alone conflates very different listings: 6 sections of
        # interior data is not the same as 6 sections of financial data.  Measured
        # over 400 listings, the per-section rates are far from uniform
        # (general 99.8%, interior 98.2%, exterior 84.8%, facilities 65.8%,
        # public sale 0.8%), so the individual flags carry information the sum
        # throws away.  Stable across snapshots for 98.7% of properties, i.e. a
        # static attribute of the listing rather than scrape-timing noise.
        'section_general': tri_state_bool(sections_data.get('hasGeneralSection')),
        'section_interior': tri_state_bool(sections_data.get('hasInteriorSection')),
        'section_exterior': tri_state_bool(sections_data.get('hasExteriorSection')),
        'section_facilities': tri_state_bool(sections_data.get('hasFacilitiesSection')),
        'section_energy': tri_state_bool(sections_data.get('hasEnergySection')),
        'section_planning': tri_state_bool(sections_data.get('hasPlanningSection')),
        'section_financial': tri_state_bool(sections_data.get('hasFinancialSection')),
        'section_public_sale': tri_state_bool(sections_data.get('hasPublicSaleSection')),

        # ---- New-build / project context -------------------------------
        # constructionPermit carries detail no other field exposes.  Fill rates
        # over 400 listings: hasPossiblePriorityPurchaseRight 40.5%,
        # constructionType 25.8%, hasPlotDivisionAuthorization 18.2%,
        # totalBuildableGroundFloorSurface 7.2%.
        'construction_type': _safe_get(construction_permit_data, 'constructionType'),
        'has_plot_division_authorization': tri_state_bool(
            construction_permit_data.get('hasPlotDivisionAuthorization')),
        'has_priority_purchase_right': tri_state_bool(
            construction_permit_data.get('hasPossiblePriorityPurchaseRight')),
        'buildable_ground_floor_surface': _nested_number(
            construction_permit_data, 'totalBuildableGroundFloorSurface', 'surface'),

        # ---- Alternate-language descriptions -----------------------------
        # The merged `description` above picks one language.  Embeds and text
        # mining see whichever won, so keep the originals to (a) detect how much
        # text the chosen branch dropped and (b) allow a language-aware embed.
        # NL is present on 100% of listings and FR on 76%.
        'description_nl': clean_text(alternative_text.get('nl')),
        'description_fr': clean_text(alternative_text.get('fr')),
        'description_en': clean_text(alternative_text.get('en')),
        'description_language': description_language,

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

    Implemented with ``os.scandir`` rather than ``os.walk``: DirEntry already
    carries ``is_dir``/``is_file`` from the readdir call, so the ~30k directory
    stat calls that ``os.walk`` would issue are avoided.
    """
    paths = []
    stack = [root_directory]
    while stack:
        current = stack.pop()
        try:
            entries = list(os.scandir(current))
        except OSError:
            continue
        for entry in entries:
            try:
                if entry.is_dir(follow_symlinks=False):
                    # Skip dot-directories: data/.snapshots holds the compressed
                    # snapshot stores, which are not HTML input.
                    if not entry.name.startswith('.'):
                        stack.append(entry.path)
                elif entry.name.endswith('.html') and entry.is_file(follow_symlinks=False):
                    paths.append(entry.path)
            except OSError:
                continue
    paths.sort()
    return paths


def parse_snapshot_store(data_dir, workers=None, progress_every=5000, log=print):
    """Build the full dataset from ``data/.snapshots`` instead of the HTML tree.

    Equivalent output to :func:`parse_data_folder_parallel`, but reads ~0.3% as
    many bytes: the store holds only the ``window.classified`` payload, already
    decompressed in one stream per property. Each row is rebuilt through the same
    :func:`build_row`, so field values are identical to an HTML parse.

    Returns a polars DataFrame with the same columns as the HTML path.
    """
    import gzip
    import json

    store_dir = os.path.join(data_dir, '.snapshots')
    if not os.path.isdir(store_dir):
        raise FileNotFoundError(
            f'no snapshot store at {store_dir}; build one with '
            f'python src/snapshot_store.py --data-dir {data_dir}')

    names = sorted(n for n in os.listdir(store_dir) if n.endswith('.jsonl.gz'))
    rows = []
    errors = []
    for index, name in enumerate(names, 1):
        property_id = name[:-len('.jsonl.gz')]
        try:
            with gzip.open(os.path.join(store_dir, name), 'rt', encoding='utf-8') as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    payload = record.get('js')
                    if not isinstance(payload, dict):
                        errors.append((property_id, 'payload not an object'))
                        continue
                    rows.append(build_row(payload, property_id, record.get('date')))
        except Exception as error:  # noqa: BLE001 - report, never abort the run
            errors.append((property_id, f'{type(error).__name__}: {error}'))
        if progress_every and index % progress_every == 0:
            log(f'  {index:,}/{len(names):,} properties, {len(rows):,} rows')

    log(f'Built {len(rows):,} rows from snapshot store, {len(errors)} errors')
    return rows_to_frame(rows), errors


def _effective_workers(requested=None):
    """Worker count tuned for this workload.

    Parsing a 379 KB file costs 0.31 ms with 4 workers and 0.99 ms with 16 --
    it is memory-bandwidth bound, not CPU bound, so oversubscribing the 18-core
    machine actively hurts.  Measured on an M5 Pro.
    """
    if requested is not None:
        return max(int(requested), 1)
    return min(4, os.cpu_count() or 1)


def rows_to_frame(rows):
    """Assemble extracted rows into a Polars DataFrame with stable schema.

    ``infer_schema_length=None`` scans every value: the default samples only the
    first 100 rows, which silently mistypes sparse columns (a column null for
    the first 100 rows gets String instead of Float64) and then either errors or
    coerces lossily.  ``strict=False`` lets genuinely mixed columns fall back to
    String instead of aborting the whole corpus parse.
    """
    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows, infer_schema_length=None, strict=False)


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
        workers: Optional worker-process count (default 4; see _effective_workers).

    Returns:
        A Polars DataFrame of extracted rows.
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
            return pl.DataFrame()
        try:
            return pl.read_csv(output_file)
        except FileNotFoundError:
            return pl.DataFrame()

    print(f"Parsing {len(all_files)} HTML files in parallel...")
    rows = []
    failures = []

    with ProcessPoolExecutor(max_workers=_effective_workers(workers)) as executor:
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

    # A broken pool fails EVERY future with the same exception, so without this
    # check the caller gets a plausible-looking "0 rows parsed" and a manifest
    # of thousands of identical errors instead of one clear diagnosis.
    broken_pool = sum(1 for f in failures if 'BrokenProcessPool' in f['reason'])
    if broken_pool and broken_pool >= 0.9 * max(len(all_files), 1):
        raise RuntimeError(
            f"{broken_pool}/{len(all_files)} files failed with BrokenProcessPool "
            f"and no rows were produced. This is usually an unguarded "
            f"multiprocessing entry point: the code that calls "
            f"parse_data_folder_parallel must be behind "
            f"`if __name__ == '__main__':` on this platform (spawn-based "
            f"multiprocessing).")

    print(f"Parsed {len(rows)} files, {len(failures)} failures "
          f"({len(failures) / max(len(all_files), 1) * 100:.3f}%).")

    if failures:
        if failures_file:
            pl.DataFrame(failures).write_csv(failures_file)
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

    return rows_to_frame(rows)


def _manifest_key_for(file_path):
    property_id, date_str = parse_file_identity(file_path)
    return f"{property_id}|{date_str}"
