"""Raw HTML-extracted table -> validated training matrix.

Why this module exists
----------------------
``preprocess_functions.preprocess()`` conflated *fit* and *transform* in one
function, was only ever called from notebooks, and had no callable entry point
(the ``run_preprocessing.py`` cited in the design docs never existed).  That made
two things impossible:

*   **Honest evaluation.**  Every fitted quantity -- imputation models,
    one-hot vocabulary, medians -- was computed on all rows before any split, so
    a test row could influence the transform applied to training rows.
*   **Tracing a row count.**  Filters dropped rows silently; a dataset that
    shrank from 142k to 19.7k gave no per-row accounting.

This module separates the two phases:

``fit_preprocessing(train_df)``   learns everything that must not see test data
``transform(df, artifacts)``      applies it, deterministically
``build_dataset(raw_df)``         the whole documented pipeline, end to end

Target leakage
--------------
``price`` used to be a predictor for imputing ``cadastral_income`` -- the model's
second-most important feature.  ``LEAKY_PREDICTORS`` now forbids that: imputation
may use property attributes only.  ``price_sq``, ``log_price``,
``spatial_lag_10`` and ``avg_dist_5nn`` are derivable from the target across rows
and are never emitted as features here; they are available only through the
explicit ``with_cross_row_features`` opt-in for analysis.

Splitting
---------
``assign_folds`` writes a ``fold`` column generated from a *group* key (same
address + same area), so the 70% of test rows that shared a physical property
with training rows in the shipped model cannot recur.  Training must respect the
column rather than re-splitting at random.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd

from validate_dataset import (
    CRITICAL, Gate, WARNING,
    build_raw_spec, build_training_spec, row_funnel, validate,
)

# Columns derived from the target (directly, or across rows via neighbour
# prices).  Never features.  ``validate_dataset.check_leaky_columns`` enforces it.
LEAKY_COLUMNS = frozenset({'price_sq', 'log_price', 'spatial_lag_10', 'avg_dist_5nn'})

# Imputation may only use property attributes.  Listing price is the target;
# using it to impute a feature drags the target into the feature matrix.
LEAKY_PREDICTORS = frozenset({'price', 'price_sq', 'log_price',
                              'spatial_lag_10', 'avg_dist_5nn'})

# Reference year for ``age``.  Fixed at fit time rather than read from the clock,
# so a dataset transformed today matches one transformed next year.
DEFAULT_REFERENCE_YEAR = 2026

BOOLEAN_PREFIXES = (
    'has_', 'is_', 'section_', 'permit_', 'breaches_', 'land_is_',
    'land_facing_', 'land_has_', 'cert_has_', 'renovation_obligation',
)

# Columns that are always completely empty or non-predictive text/IDs.
ALWAYS_EMPTY_OR_DEAD = (
    'parking_closed_box', 'land_sewer_connection',
    'cert_global_insulation_level', 'expected_monthly_rental_income',
    'has_virtual_tour', 'description_en', 'is_holiday_property',
    'drop_reason',
)

# Columns that are single-class constants after the row gates or are
# post-publication demand counters unavailable when scoring a fresh listing.
CONSTANT_OR_POST_LAUNCH_COLUMNS = (
    'price_type', 'region', 'province', 'district',
    'transaction_type', 'transaction_subtype',
    'section_interior', 'section_financial', 'section_public_sale',
    'is_biddit_sale', 'is_sold_or_rented', 'is_public_sale', 'is_life_annuity_sale',
    'view_count', 'bookmark_count', 'listing_duration_days',
    'is_under_option', 'display_flag_main', 'display_flags_secondary',
    'publication_date', 'immo_office_name',
)

# Surfaces and counts whose absence means 0 when the companion flag is not True
# (or when undeclared on an apartment), NEVER a global median like 75 m2 garden.
ZERO_WHEN_ABSENT_SURFACES: dict[str, str | None] = {
    'garden_surface': 'has_garden',
    'terrace_surface': 'has_terrace',
    'attic_surface': 'has_attic',
    'basement_surface': 'has_basement',
    'office_surface': 'has_office',
    'workspace_surface': 'has_workspace',
    'buildable_ground_floor_surface': None,
    'annex_count': None,
    'fireplace_count': None,
    'parking_indoor': None,
    'parking_outdoor': None,
    'living_room_surface': None,
    'kitchen_surface': None,
    'bedroom_surface_sum': None,
    'bedroom_surface_min': None,
    'bedroom_surface_max': None,
    'bedrooms_with_surface': None,
    'monthly_costs': None,
}

# Static physical/regulatory property attributes that can be safely coalesced
# across multiple scrape snapshots of the same property_id when an agency
# fills them in on a later edit.
STATIC_BACKFILL_COLUMNS = (
    'cadastral_income', 'year_built', 'energy_label',
    'primary_energy_consumption', 'epc_cert_year', 'epc_cert_revision',
    'carbon_emission', 'building_condition', 'property_size',
    'facade_count', 'street_facade_width', 'total_floors',
    'heating_type', 'kitchen_type', 'kitchen_surface', 'living_room_surface',
    'bedroom_surface_sum', 'bedroom_surface_min', 'bedroom_surface_max',
    'flood_zone_type', 'p_score', 'g_score', 'construction_type',
    'renovation_obligation', 'cert_has_asbestos',
    'cert_has_electrical_installation', 'cert_oil_tank_status',
)

# Key Ghent landmarks (lat, lon) for spatial distance gradients (in km).
GHENT_LANDMARKS = {
    'dist_sint_pieters_km': (51.03589, 3.71067),
    'dist_dampoort_km': (51.05639, 3.74111),
    'dist_korenmarkt_km': (51.05389, 3.72194),
    'dist_citadelpark_km': (51.03750, 3.72361),
    'dist_blaarmeersen_km': (51.04583, 3.68528),
}

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_WIJKEN_PATH = os.path.join(BASE_DIR, 'external_data', 'stadswijken-gent.csv')
DEFAULT_SECTORS_PATH = os.path.join(BASE_DIR, 'external_data', 'statbel', 'ghent_sectors_enriched.geojson')
DEFAULT_GRAPH_PATH = os.path.join(BASE_DIR, 'external_data', 'ghent_region_network.graphml')
DEFAULT_MACRO_CACHE_PATH = os.path.join(BASE_DIR, 'outputs', 'dataset', 'ecb_macro_monthly.csv')

MACRO_COLUMNS = (
    'belgian_cpi_index', 'belgian_health_index', 'belgian_cpi_yoy_pct',
    'belgian_10y_yield', 'euribor_12m', 'euribor_3m',
    'be_mortgage_rate_avg', 'be_mortgage_rate_fix10y',
    'belgian_hpi_index', 'real_mortgage_rate', 'yield_curve_slope',
)


# =============================================================================
# Knobs
# =============================================================================

@dataclass
class PreprocessConfig:
    """Every judgement call the pipeline makes, in one place."""
    # --- row gates ---
    min_price_eur: float = 25_000.0
    max_price_eur: float = 50_000_000.0
    min_livable_area: float = 10.0
    max_livable_area: float = 1000.0
    require_coordinates: bool = True
    drop_commercial: bool = True
    drop_non_residential_type: bool = True
    outlier_quantile: float = 0.99
    # --- dedupe ---
    # 'last' keeps the final snapshot; 'first' keeps the asking price as launched.
    # Both are defensible; the earli*est* observation matches inference on a fresh
    # listing, so that is the default.
    keep_observation: str = 'first'
    # Collapse agency relisting churn & identical multi-unit clone classifieds
    # that share (property_key, round(price, -3)) down to 1 row.
    dedupe_relistings: bool = True
    # --- features ---
    reference_year: int = DEFAULT_REFERENCE_YEAR
    onehot_columns: tuple[str, ...] = ('property_type',)
    include_extended_features: bool = False
    # --- safety ---
    forbid_leaky_features: bool = True

    def as_dict(self) -> dict:
        out = dict(self.__dict__)
        out['onehot_columns'] = list(self.onehot_columns)
        return out


# =============================================================================
# Stage A: raw -> filtered rows, with a recorded funnel
# =============================================================================

def _normalise_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Parse both the DDMMYYYY scrape tag and ISO publication timestamps."""
    df = df.copy()
    if 'date_obtained' in df.columns:
        raw = df['date_obtained'].astype(str).str.zfill(8)
        df['date_obtained'] = pd.to_datetime(raw, format='%d%m%Y', errors='coerce')
    for column in ('publication_date', 'last_modification_date'):
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors='coerce', utc=True)
            df[column] = df[column].dt.tz_localize(None)
    return df


def _property_key(df: pd.DataFrame) -> pd.Series:
    """Identity of the *physical property*, stable across relistings.

    Immoweb assigns a new listing id when a property is relisted, so
    ``property_id`` cannot group observations of the same home.  Address + area
    can, and it is what the leakage audit found 70% of test rows were exploiting.
    """
    parts = []
    for column in ('postal_code', 'street', 'house_number', 'livable_area'):
        if column in df.columns:
            parts.append(df[column].astype(str).str.lower().str.strip())
    if not parts:
        return df['property_id'].astype(str)
    key = parts[0]
    for extra in parts[1:]:
        key = key + '|' + extra
    return key


def _is_commercial(df: pd.DataFrame) -> pd.Series:
    """Commercial/mixed-use listings are priced on a different basis."""
    commercial_terms = (
        'handelspand', 'handelsruimte', 'handelsgelijkvloers', 'winkelruimte',
        'winkelpand', 'kantoorruimte', 'espace commercial', 'commercial space',
        'fond de commerce', 'commercieel gelijkvloers',
    )
    subtype = df.get('property_subtype', pd.Series('', index=df.index)).astype(str)
    residential = subtype.isin(('APARTMENT', 'PENTHOUSE', 'GROUND_FLOOR',
                                'FLAT_STUDIO', 'DUPLEX', 'TRIPLEX',
                                'SERVICE_FLAT', 'KOT', 'LOFT', 'HOUSE', 'VILLA',
                                'TOWN_HOUSE', 'BUNGALOW', 'CHALET', 'MANSION'))
    # A non-residential subtype is only "commercial" if it is a mixed-use
    # building or the description advertises commercial space.
    mixed = subtype == 'MIXED_USE_BUILDING'
    text = df.get('description', pd.Series('', index=df.index)).fillna('').astype(str).str.lower()
    mentions = text.apply(lambda value: any(term in value for term in commercial_terms))
    return mixed | (~residential & mentions)


def build_gates(df: pd.DataFrame, config: PreprocessConfig) -> list[Gate]:
    """The ordered row filters.  Order matters: cheapest and most decisive first."""
    gates: list[Gate] = []

    gates.append(Gate(
        'target_present',
        lambda d: d['price'].notna(),
        'no asking price'))

    if config.require_coordinates:
        gates.append(Gate(
            'coordinates_present',
            lambda d: d['latitude'].notna() & d['longitude'].notna(),
            'no coordinates'))
        gates.append(Gate(
            'coordinates_in_ghent_bbox',
            lambda d: (
                (d['latitude'] >= 50.90) & (d['latitude'] <= 51.20) &
                (d['longitude'] >= 3.50) & (d['longitude'] <= 4.00)
            ),
            'coordinates outside Ghent bounding box'))

    if 'price_type' in df.columns:
        gates.append(Gate(
            'residential_sale_price_type',
            lambda d: ~d['price_type'].isin(('annuity_monthly_amount', 'group_sale')),
            'price_type is annuity or unpriced group_sale'))

    gates.append(Gate(
        'price_in_range',
        lambda d: (d['price'] > config.min_price_eur) &
                  (d['price'] <= config.max_price_eur),
        f'price outside ({config.min_price_eur:,.0f}, {config.max_price_eur:,.0f}]'))

    gates.append(Gate(
        'area_in_range',
        lambda d: d['livable_area'].notna() &
                  (d['livable_area'] >= config.min_livable_area) &
                  (d['livable_area'] <= config.max_livable_area),
        f'livable_area outside [{config.min_livable_area}, {config.max_livable_area}]'))

    if config.drop_non_residential_type:
        gates.append(Gate(
            'residential_type',
            lambda d: d['property_type'].isin(('HOUSE', 'APARTMENT')),
            'not a HOUSE/APARTMENT listing'))

    if config.drop_commercial:
        gates.append(Gate(
            'not_commercial',
            lambda d: ~_is_commercial(d),
            'commercial or mixed-use'))

    for flag in ('is_life_annuity_sale', 'is_public_sale', 'is_biddit_sale'):
        if flag in df.columns:
            gates.append(Gate(
                f'not_{flag}',
                (lambda f: lambda d: d[f] != True)(flag),
                f'{flag} pricing does not apply'))

    return gates


def clean_raw(df: pd.DataFrame, config: PreprocessConfig | None = None,
              annotate: bool = True) -> dict[str, Any]:
    """Apply the row gates and report the funnel.

    Returns a dict with ``kept``, ``annotated`` (all rows + ``drop_reason``),
    ``funnel``, ``report`` and ``config``.  ``annotated`` exists so a surprising
    row count can be traced to the exact rows a gate removed, not just the count.
    """
    config = config or PreprocessConfig()
    work = _normalise_dates(df)
    work['property_key'] = _property_key(work)

    gates = build_gates(work, config)
    kept, funnel = row_funnel(work, gates)

    # Dedupe AFTER the funnel so a rejected snapshot cannot set the kept row.
    n_before_dedup = len(kept)
    kept = _dedupe(kept, config.keep_observation)
    funnel.append({'gate': f'dedupe_property_id_{config.keep_observation}',
                   'removed': n_before_dedup - len(kept),
                   'remaining': len(kept),
                   'reason': 'same listing observed on several scrape dates'})

    # Collapse agency relisting churn & identical multi-unit project clones
    # that share the same physical property_key and same asking price.
    if config.dedupe_relistings and {'property_key', 'price'}.issubset(kept.columns):
        n_before_relist = len(kept)
        kept = _dedupe_relistings(kept, config.keep_observation)
        funnel.append({'gate': 'dedupe_relisted_property_key',
                       'removed': n_before_relist - len(kept),
                       'remaining': len(kept),
                       'reason': 'relisted or duplicate-unit classified with identical property_key and price'})

    # Outlier trimming on the target, computed after dedupe.
    if config.outlier_quantile and len(kept) > 20 and 'price' in kept.columns:
        n_before = len(kept)
        cutoff = kept['price'].quantile(config.outlier_quantile)
        kept = kept[kept['price'] <= cutoff]
        funnel.append({'gate': f'price_outlier_p{config.outlier_quantile * 100:g}',
                       'removed': n_before - len(kept),
                       'remaining': len(kept),
                       'reason': f'price above p{config.outlier_quantile * 100:g}'})

    kept = kept.reset_index(drop=True)

    report = validate(kept, build_raw_spec(), source='clean_raw')
    report.funnel = funnel

    result: dict[str, Any] = {
        'kept': kept,
        'funnel': funnel,
        'report': report,
        'config': config,
        'n_input': len(work),
    }
    if annotate:
        result['annotated'] = _annotate_all(work, gates)
    return result


def _dedupe(df: pd.DataFrame, keep: str) -> pd.DataFrame:
    """One row per listing id, chosen by scrape date, with static fields coalesced.

    ``keep='first'`` retains the earliest observation, which matches how the model
    is used (scoring a listing at launch); ``keep='last'`` retains the final state.
    Before collapsing duplicate snapshots, static physical/regulatory fields that
    were left blank on the initial scrape but filled in by the agency on a later
    snapshot are backfilled onto the kept row.
    """
    if 'property_id' not in df.columns:
        return _clean_raw_sentinels(df)
    ordered = _clean_raw_sentinels(df.sort_values('date_obtained', kind='stable'))
    kept = ordered.drop_duplicates(subset=['property_id'], keep=keep).copy()

    avail_static = [
        c for c in STATIC_BACKFILL_COLUMNS
        if c in ordered.columns and kept[c].isna().any()
    ]
    if avail_static and len(ordered) > len(kept):
        coalesced = ordered.groupby('property_id', sort=False)[avail_static].first()
        kept_indexed = kept.set_index('property_id')
        for col in avail_static:
            missing_mask = kept_indexed[col].isna()
            if missing_mask.any():
                kept_indexed.loc[missing_mask, col] = coalesced.loc[
                    kept_indexed.index[missing_mask], col
                ]
        kept = kept_indexed.reset_index()
        # Preserve original column order
        kept = kept[ list(ordered.columns) ]
    return kept


def _dedupe_relistings(df: pd.DataFrame, keep: str) -> pd.DataFrame:
    """Collapse repeated Immoweb property_ids for the exact same physical property & price.

    Agencies frequently delete and re-create a listing to obtain a fresh Immoweb
    id, and new-build developments often post dozens of identical clone units at
    the same address and area.  Before dropping the duplicate IDs, static
    attributes are coalesced across all IDs sharing the same ``property_key``.
    """
    ordered = df.sort_values('date_obtained', kind='stable').copy()
    ordered = _clean_raw_sentinels(ordered)

    avail_static = [
        c for c in STATIC_BACKFILL_COLUMNS
        if c in ordered.columns and ordered[c].isna().any()
    ]
    if avail_static:
        coalesced = ordered.groupby('property_key', sort=False)[avail_static].first()
        for col in avail_static:
            missing_mask = ordered[col].isna()
            if missing_mask.any():
                ordered.loc[missing_mask, col] = ordered.loc[
                    missing_mask, 'property_key'
                ].map(coalesced[col])

    ordered['_price_bucket'] = (pd.to_numeric(ordered['price'], errors='coerce') / 1000.0).round()
    kept = ordered.drop_duplicates(
        subset=['property_key', '_price_bucket'], keep=keep
    ).drop(columns=['_price_bucket'])
    return kept


def _clean_raw_sentinels(df: pd.DataFrame) -> pd.DataFrame:
    """Replace Immoweb form-default 0 / typo sentinels with NaN before coalescing or fitting."""
    df = df.copy()
    if 'cadastral_income' in df.columns:
        ki = pd.to_numeric(df['cadastral_income'], errors='coerce')
        # KI <= 0 is an Immoweb form placeholder; KI > 25,000 is asking price typed into KI box
        df['cadastral_income'] = ki.where((ki > 0) & (ki <= 25_000))
    if 'year_built' in df.columns:
        yb = pd.to_numeric(df['year_built'], errors='coerce')
        df['year_built'] = yb.where((yb >= 1700) & (yb <= DEFAULT_REFERENCE_YEAR + 4))
    for col in ('number_of_bathrooms', 'number_of_toilets'):
        if col in df.columns:
            s = pd.to_numeric(df[col], errors='coerce')
            df[col] = s.where((s >= 1) & (s <= 20))
    if 'number_of_bedrooms' in df.columns:
        beds = pd.to_numeric(df['number_of_bedrooms'], errors='coerce')
        subtype = df.get('property_subtype', pd.Series('', index=df.index)).astype(str)
        is_studio = subtype.isin(('FLAT_STUDIO', 'KOT', 'SERVICE_FLAT', 'LOFT'))
        # 0 bedrooms is valid on a studio/kot; on houses or standard apartments 0 is a form placeholder
        valid_beds = ((beds >= 1) | (is_studio & (beds == 0))) & (beds <= 25)
        df['number_of_bedrooms'] = beds.where(valid_beds)
    if 'property_size' in df.columns:
        ps = pd.to_numeric(df['property_size'], errors='coerce')
        df['property_size'] = ps.where(ps > 0)
    if 'primary_energy_consumption' in df.columns:
        pec = pd.to_numeric(df['primary_energy_consumption'], errors='coerce')
        # Negative values up to -50 are valid net-export solar EPCs, but exact 0 on older buildings is a placeholder
        is_new = df.get('is_newly_built', pd.Series(False, index=df.index)) == True
        df['primary_energy_consumption'] = pec.where(
            ((pec > 0) | (is_new & (pec >= -50)) | ((pec < 0) & (pec >= -50))) & (pec <= 3000)
        )
    if 'energy_label' in df.columns:
        valid_labels = {'A++', 'A+', 'A', 'B', 'C', 'D', 'E', 'F'}
        s = df['energy_label'].astype(object)
        df['energy_label'] = s.where(s.isin(valid_labels), None)
    return df


def _annotate_all(work: pd.DataFrame, gates: Sequence[Gate]) -> pd.DataFrame:
    from validate_dataset import annotate_rows
    return annotate_rows(work, gates)


# =============================================================================
# Stage B: fit / transform
# =============================================================================

@dataclass
class PreprocessingArtifacts:
    """Everything learned from the training fold.  Persisted with the model."""
    reference_year: int
    median_year_built: float
    onehot_categories: dict[str, list[str]] = field(default_factory=dict)
    numeric_medians: dict[str, float] = field(default_factory=dict)
    imputation_models: dict[str, Any] = field(default_factory=dict)
    feature_columns: list[str] = field(default_factory=list)
    config: dict = field(default_factory=dict)
    spec_version: int = 1

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> 'PreprocessingArtifacts':
        return joblib.load(path)


def _ensure_booleans(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Coerce tri-state boolean columns to bool, recording which values were undeclared.

    Returns the frame plus the list of ``*_missing`` indicator columns created.
    """
    df = df.copy()
    indicators = []
    _strings = {'True': True, 'False': False, 'true': True, 'false': False,
                '1': True, '0': False, '1.0': True, '0.0': False}
    # Engineered 0/1 integer columns that must not be re-coerced as tri-state.
    _skip = {
        'has_meaningful_description', 'has_floor_plan',
        'has_text_recovered_amenities', 'is_multi_unit_project',
    }

    def _coerce(value):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, str):
            return _strings.get(value.strip())
        if isinstance(value, (int, float, np.integer, np.floating)) and not pd.isna(value):
            return bool(value) if value in (0, 1) else None
        return None

    candidate_cols = [
        c for c in df.columns
        if c not in _skip and not c.endswith('_missing') and (
            df[c].dtype == bool
            or str(df[c].dtype) == 'boolean'
            or c.startswith(BOOLEAN_PREFIXES)
        )
    ]

    indicator_cols: dict[str, pd.Series] = {}
    for column in candidate_cols:
        series = df[column]
        if series.dtype == bool and not series.isna().any():
            continue
        normalised = series.map(_coerce).astype(object)
        was_missing = normalised.isna()
        if was_missing.any():
            indicator = f'{column}_missing'
            indicator_cols[indicator] = was_missing.astype(int)
            indicators.append(indicator)
        df[column] = pd.Series(
            [bool(v) if v is not None else False for v in normalised],
            index=df.index,
            dtype=bool,
        )
    if indicator_cols:
        df = pd.concat([df, pd.DataFrame(indicator_cols, index=df.index)], axis=1)
    return df, indicators


def _engineer(df: pd.DataFrame, artifacts: PreprocessingArtifacts,
              config: PreprocessConfig) -> pd.DataFrame:
    """Row-wise features.  No cross-row aggregation, no target use.

    Tolerates un-parsed input: dates may still be strings and numerics may be
    object-typed, because at inference time the payload comes straight from the
    parser rather than from a previously cleaned table.
    """
    df = df.copy()

    df = _clean_raw_sentinels(df)

    # Normalise the date columns defensively -- the same input may be raw strings.
    for column in ('date_obtained', 'publication_date', 'last_modification_date'):
        if column in df.columns and not pd.api.types.is_datetime64_any_dtype(df[column]):
            if column == 'date_obtained':
                df[column] = pd.to_datetime(
                    df[column].astype(str).str.zfill(8),
                    format='%d%m%Y', errors='coerce')
            else:
                df[column] = pd.to_datetime(
                    df[column], errors='coerce', utc=True).dt.tz_localize(None)

    # age from a FIXED reference year so the feature does not drift with the clock
    if 'year_built' in df.columns:
        year = pd.to_numeric(df['year_built'], errors='coerce')
        # Null out implausible years (< 1700 or > reference_year + 4) as typos.
        year = year.where((year >= 1700) & (year <= artifacts.reference_year + 4))
        df['year_built_missing'] = year.isna().astype(int)
        year = year.fillna(artifacts.median_year_built)
        df['year_built'] = year
        df['age'] = artifacts.reference_year - year

    # Apartments do not own a private land plot; set property_size=0 on apartments
    # and impute houses from the house-only median plot size.
    if 'property_size' in df.columns:
        plot = pd.to_numeric(df['property_size'], errors='coerce').where(lambda s: s > 0)
        df['property_size_missing'] = plot.isna().astype(int)
        is_apt = pd.Series(False, index=df.index)
        if 'property_type' in df.columns:
            is_apt = df['property_type'].astype(str) == 'APARTMENT'
        elif 'property_type_APARTMENT' in df.columns:
            is_apt = df['property_type_APARTMENT'] == 1
        elif 'property_type_HOUSE' in df.columns:
            is_apt = df['property_type_HOUSE'] == 0
        house_median_plot = artifacts.numeric_medians.get('property_size', 220.0)
        df['property_size'] = np.where(is_apt, 0.0, plot.fillna(house_median_plot))

    if 'cadastral_income' in df.columns:
        df['cadastral_income_missing'] = df['cadastral_income'].isna().astype(int)

    if 'primary_energy_consumption' in df.columns:
        df['primary_energy_consumption_missing'] = df['primary_energy_consumption'].isna().astype(int)

    # Clean Immoweb form-default roomCount == 1 when number_of_bedrooms > 1
    if {'number_of_rooms', 'number_of_bedrooms'}.issubset(df.columns):
        rooms = pd.to_numeric(df['number_of_rooms'], errors='coerce')
        beds = pd.to_numeric(df['number_of_bedrooms'], errors='coerce')
        invalid_rooms = rooms.notna() & beds.notna() & (rooms < beds)
        df['number_of_rooms'] = rooms.where(~invalid_rooms)

    # Zero-fill structural/amenity surfaces when absent rather than letting
    # median imputation assign every apartment a 75 m2 garden.
    for surface_col, flag_col in ZERO_WHEN_ABSENT_SURFACES.items():
        if surface_col not in df.columns:
            continue
        series = pd.to_numeric(df[surface_col], errors='coerce')
        df[f'{surface_col}_missing'] = series.isna().astype(int)
        if flag_col and flag_col in df.columns:
            # Where flag is explicitly False or missing (not True), absent surface is 0.
            not_present = df[flag_col].map(
                lambda v: not (v is True or v == 'True' or v == 'true' or v == 1))
            series = series.where(~(series.isna() & not_present), 0.0)
        df[surface_col] = series.fillna(0.0)

    # Clamp sub-room surfaces so individual room breakdowns cannot exceed livable_area
    if 'livable_area' in df.columns:
        area = pd.to_numeric(df['livable_area'], errors='coerce')
        for sub_col in ('living_room_surface', 'kitchen_surface',
                        'bedroom_surface_sum', 'bedroom_surface_min', 'bedroom_surface_max'):
            if sub_col in df.columns:
                sub = pd.to_numeric(df[sub_col], errors='coerce').fillna(0.0)
                df[sub_col] = np.where(area.notna() & (sub > area), area, sub)

    # Clamp garden_surface when it exceeds property_size (plot surface) on houses
    if {'garden_surface', 'property_size'}.issubset(df.columns):
        garden = pd.to_numeric(df['garden_surface'], errors='coerce')
        plot = pd.to_numeric(df['property_size'], errors='coerce')
        exceeds = garden.notna() & plot.notna() & (plot > 0) & (garden > plot)
        df['garden_exceeded_plot'] = exceeds.astype(int)
        df['garden_surface'] = np.where(exceeds, plot, garden)

    # true time-on-market, from the real publication timestamp
    if {'publication_date', 'last_modification_date'}.issubset(df.columns):
        duration = (df['last_modification_date'] - df['publication_date']).dt.days
        df['listing_duration_days'] = duration.clip(lower=0)

    if 'publication_date' in df.columns and pd.api.types.is_datetime64_any_dtype(df['publication_date']):
        pub_dt = df['publication_date']
        if 'date_obtained' in df.columns and pd.api.types.is_datetime64_any_dtype(df['date_obtained']):
            pub_dt = pub_dt.fillna(df['date_obtained'])
        df['publication_year'] = pub_dt.dt.year
        df['publication_month'] = pub_dt.dt.month

    if 'epc_cert_year' in df.columns:
        epc_year = pd.to_numeric(df['epc_cert_year'], errors='coerce')
        df['epc_cert_year_missing'] = epc_year.isna().astype(int)
        df['epc_cert_age'] = (artifacts.reference_year - epc_year).clip(lower=0)

    if 'text_recovered_flags' in df.columns:
        rec = df['text_recovered_flags'].fillna('').astype(str)
        df['has_text_recovered_amenities'] = (rec.str.len() > 0).astype(int)
        df['n_text_recovered_amenities'] = rec.map(
            lambda v: 0 if not v else len(v.split('|')))

    if 'floor_plan_count' in df.columns:
        fp = pd.to_numeric(df['floor_plan_count'], errors='coerce').fillna(0)
        df['has_floor_plan'] = (fp > 0).astype(int)

    if {'picture_count', 'number_of_bedrooms'}.issubset(df.columns):
        pics = pd.to_numeric(df['picture_count'], errors='coerce').fillna(0)
        beds = pd.to_numeric(df['number_of_bedrooms'], errors='coerce').fillna(1).clip(lower=1)
        df['pictures_per_bedroom'] = pics / beds

    # Surface & layout ratios
    if {'livable_area', 'property_size'}.issubset(df.columns):
        area = pd.to_numeric(df['livable_area'], errors='coerce')
        plot = pd.to_numeric(df['property_size'], errors='coerce')
        valid_plot = plot.where(plot > 0)
        df['livable_to_plot_ratio'] = (area / valid_plot).fillna(0.0)
        if 'garden_surface' in df.columns:
            garden = pd.to_numeric(df['garden_surface'], errors='coerce').fillna(0.0)
            df['garden_to_plot_ratio'] = (garden / valid_plot).clip(0.0, 1.0).fillna(0.0)

    if {'living_room_surface', 'kitchen_surface', 'livable_area'}.issubset(df.columns):
        area = pd.to_numeric(df['livable_area'], errors='coerce').where(lambda s: s > 0)
        living = pd.to_numeric(df['living_room_surface'], errors='coerce').fillna(0.0)
        kitchen = pd.to_numeric(df['kitchen_surface'], errors='coerce').fillna(0.0)
        df['living_plus_kitchen_share'] = ((living + kitchen) / area).clip(0.0, 1.0).fillna(0.0)

    if {'bedroom_surface_sum', 'bedrooms_with_surface'}.issubset(df.columns):
        b_sum = pd.to_numeric(df['bedroom_surface_sum'], errors='coerce').fillna(0.0)
        b_cnt = pd.to_numeric(df['bedrooms_with_surface'], errors='coerce').where(lambda s: s > 0)
        df['mean_bedroom_surface'] = (b_sum / b_cnt).fillna(0.0)

    if {'number_of_bathrooms', 'number_of_bedrooms'}.issubset(df.columns):
        baths = pd.to_numeric(df['number_of_bathrooms'], errors='coerce').fillna(1.0)
        beds = pd.to_numeric(df['number_of_bedrooms'], errors='coerce').fillna(1.0).clip(lower=1.0)
        df['bathrooms_per_bedroom'] = baths / beds

    if 'description' in df.columns:
        text = df['description'].fillna('').astype(str)
        df['description_length'] = text.str.len()
        df['has_meaningful_description'] = (text.str.len() > 50).astype(int)
        df['description_is_placeholder'] = (
            text.str.strip().isin(('', '-', '.')).astype(int))

    df = _add_geo_features(df)
    df = _add_macro_indicators(df)
    return df


_CACHED_WIJKEN_GDF = None
_CACHED_SECTORS_GDF = None
_CACHED_STREET_GRAPH = None
_CACHED_MACRO_DF = None


def _get_sectors_gdf():
    global _CACHED_SECTORS_GDF
    if _CACHED_SECTORS_GDF is not None:
        return _CACHED_SECTORS_GDF
    if not os.path.exists(DEFAULT_SECTORS_PATH):
        return None
    try:
        import geopandas as gpd
        _CACHED_SECTORS_GDF = gpd.read_file(DEFAULT_SECTORS_PATH)
    except Exception:
        _CACHED_SECTORS_GDF = None
    return _CACHED_SECTORS_GDF


def _get_wijken_gdf():
    global _CACHED_WIJKEN_GDF
    if _CACHED_WIJKEN_GDF is not None:
        return _CACHED_WIJKEN_GDF
    if not os.path.exists(DEFAULT_WIJKEN_PATH):
        return None
    try:
        import geopandas as gpd
        from shapely.geometry import shape
        wijken = pd.read_csv(DEFAULT_WIJKEN_PATH, sep=';')
        wijken['geometry'] = wijken['Geometry'].apply(lambda s: shape(json.loads(s)))
        wijken['wijk_clean'] = (
            wijken['wijk'].astype(str)
            .str.replace(' ', '', regex=False)
            .str.replace('-', '_', regex=False)
        )
        _CACHED_WIJKEN_GDF = gpd.GeoDataFrame(
            wijken[['wijk_clean', 'geometry']], geometry='geometry', crs='EPSG:4326')
    except Exception:
        _CACHED_WIJKEN_GDF = None
    return _CACHED_WIJKEN_GDF


def _get_street_graph():
    global _CACHED_STREET_GRAPH
    if _CACHED_STREET_GRAPH is not None:
        return _CACHED_STREET_GRAPH
    if not os.path.exists(DEFAULT_GRAPH_PATH):
        return None
    try:
        import osmnx as ox
        G = ox.load_graphml(DEFAULT_GRAPH_PATH)
        nodes, edges = ox.graph_to_gdfs(G)
        edges['highway'] = edges['highway'].apply(
            lambda h: h[0] if isinstance(h, list) else h)
        undesirable = {
            'cycleway', 'footway', 'path', 'pedestrian',
            'track', 'steps', 'service', 'trunk',
        }
        main_edges = edges[~edges['highway'].isin(undesirable)]
        G_main = ox.graph_from_gdfs(nodes, main_edges) if not main_edges.empty else None
        _CACHED_STREET_GRAPH = (G, edges, G_main, main_edges, undesirable)
    except Exception:
        _CACHED_STREET_GRAPH = None
    return _CACHED_STREET_GRAPH


def _add_geo_features(df: pd.DataFrame) -> pd.DataFrame:
    """Location features derived strictly from the row's own coordinates + public maps."""
    center_lat, center_lon = 51.053098573153505, 3.7265139580328865
    if not {'latitude', 'longitude'}.issubset(df.columns):
        return df

    lat = pd.to_numeric(df['latitude'], errors='coerce')
    lon = pd.to_numeric(df['longitude'], errors='coerce')
    df['dist_from_center'] = np.sqrt(
        (lat - center_lat) ** 2 + (lon - center_lon) ** 2) * 100

    # Haversine-approximate distance in km to key Ghent stations/landmarks
    cos_lat = np.cos(np.radians(center_lat))
    for col_name, (l_lat, l_lon) in GHENT_LANDMARKS.items():
        dlat_km = (lat - l_lat) * 111.32
        dlon_km = (lon - l_lon) * 111.32 * cos_lat
        df[col_name] = np.sqrt(dlat_km ** 2 + dlon_km ** 2)

    valid_coords = df[['latitude', 'longitude']].dropna().drop_duplicates().reset_index(drop=True)
    if valid_coords.empty:
        return df

    # 1. Ghent stadswijken spatial join (only when >10 rows or explicitly present, fast on unique coords)
    if 'neighbourhood' not in df.columns:
        wijken_gdf = _get_wijken_gdf()
        if wijken_gdf is not None:
            import geopandas as gpd
            pts = gpd.GeoDataFrame(
                valid_coords,
                geometry=gpd.points_from_xy(valid_coords['longitude'], valid_coords['latitude']),
                crs='EPSG:4326',
            )
            joined = gpd.sjoin(pts, wijken_gdf, how='left', predicate='within')
            joined = joined[~joined.index.duplicated(keep='first')]
            coord_to_wijk = dict(zip(
                zip(valid_coords['latitude'], valid_coords['longitude']),
                joined['wijk_clean'].fillna('Unknown'),
            ))
            df['neighbourhood'] = [
                coord_to_wijk.get((a, b), 'Unknown')
                for a, b in zip(df['latitude'], df['longitude'])
            ]

    # 1b. Statbel 2026 statistical sector spatial join + socio-economic attributes
    if 'statbel_sector' not in df.columns:
        sectors_gdf = _get_sectors_gdf()
        if sectors_gdf is not None:
            import geopandas as gpd
            pts = gpd.GeoDataFrame(
                valid_coords,
                geometry=gpd.points_from_xy(valid_coords['longitude'], valid_coords['latitude']),
                crs='EPSG:4326',
            )
            joined = gpd.sjoin(pts, sectors_gdf, how='left', predicate='within')
            joined = joined[~joined.index.duplicated(keep='first')]
            sector_cols = [
                'cd_sector', 'sector_pop_density', 'sector_avg_hh_size',
                'sector_cars_per_hh', 'sector_avg_income', 'sector_median_income',
                'sector_income_iqr',
            ]
            rename_map = {'cd_sector': 'statbel_sector'}
            coord_keys = list(zip(valid_coords['latitude'], valid_coords['longitude']))
            for src_col in sector_cols:
                dst_col = rename_map.get(src_col, src_col)
                vals = joined[src_col].tolist() if src_col in joined.columns else [None] * len(valid_coords)
                lookup = dict(zip(coord_keys, vals))
                df[dst_col] = [lookup.get((a, b)) for a, b in zip(df['latitude'], df['longitude'])]
            df['statbel_sector'] = df['statbel_sector'].fillna('Unknown').astype(str)

    # 2. OSMnx street_type & distance_to_street
    if 'street_type' not in df.columns and len(valid_coords) >= 1:
        graph_bundle = _get_street_graph()
        if graph_bundle is not None:
            import osmnx as ox
            G, edges, G_main, main_edges, undesirable = graph_bundle
            xs = valid_coords['longitude'].to_numpy()
            ys = valid_coords['latitude'].to_numpy()
            ne_idx, dists = ox.nearest_edges(G, X=xs, Y=ys, return_dist=True)
            hw = edges.loc[ne_idx, 'highway'].to_numpy().copy()
            dists = np.asarray(dists, dtype=float)
            needs_corr = np.isin(hw, list(undesirable))
            if needs_corr.any() and G_main is not None:
                m_idx, m_dists = ox.nearest_edges(
                    G_main, X=xs[needs_corr], Y=ys[needs_corr], return_dist=True)
                m_hw = main_edges.loc[m_idx, 'highway'].to_numpy()
                m_dists = np.asarray(m_dists, dtype=float)
                replace_mask = m_dists <= 20.0
                corr_indices = np.where(needs_corr)[0]
                hw[corr_indices[replace_mask]] = m_hw[replace_mask]
                dists[corr_indices[replace_mask]] = m_dists[replace_mask]

            coord_to_hw = dict(zip(
                zip(valid_coords['latitude'], valid_coords['longitude']),
                pd.Series(hw).fillna('unknown').astype(str),
            ))
            coord_to_dist = dict(zip(
                zip(valid_coords['latitude'], valid_coords['longitude']),
                dists,
            ))
            df['street_type'] = [
                coord_to_hw.get((a, b), 'unknown')
                for a, b in zip(df['latitude'], df['longitude'])
            ]
            df['distance_to_street'] = [
                coord_to_dist.get((a, b), np.nan)
                for a, b in zip(df['latitude'], df['longitude'])
            ]

    return df


def _get_macro_table() -> pd.DataFrame | None:
    """Load or fetch monthly ECB macro indicators (CPI, Belgian 10Y yield, Euribor 12m)."""
    global _CACHED_MACRO_DF
    if _CACHED_MACRO_DF is not None:
        return _CACHED_MACRO_DF
    if os.path.exists(DEFAULT_MACRO_CACHE_PATH):
        try:
            cached = pd.read_csv(DEFAULT_MACRO_CACHE_PATH)
            cached['year_month'] = pd.to_datetime(cached['year_month'])
            _CACHED_MACRO_DF = cached
            return _CACHED_MACRO_DF
        except Exception:
            pass
    try:
        from train_models import fetch_macro_indicators
        macro = fetch_macro_indicators()
        if macro is not None and not macro.empty:
            os.makedirs(os.path.dirname(DEFAULT_MACRO_CACHE_PATH), exist_ok=True)
            macro.to_csv(DEFAULT_MACRO_CACHE_PATH, index=False)
            _CACHED_MACRO_DF = macro
            return _CACHED_MACRO_DF
    except Exception:
        pass
    return None


def _add_macro_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Join monthly ECB macro indicators onto publication_date (fallback date_obtained)."""
    macro = _get_macro_table()
    if macro is None or macro.empty:
        return df
    date_col = 'publication_date' if 'publication_date' in df.columns else (
        'date_obtained' if 'date_obtained' in df.columns else None)
    if date_col is None or not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        return df

    macro_sorted = macro.sort_values('year_month').ffill().bfill()
    ym = df[date_col].dt.to_period('M').dt.to_timestamp()
    fallback_date = (
        df['date_obtained'].dt.to_period('M').dt.to_timestamp()
        if 'date_obtained' in df.columns and pd.api.types.is_datetime64_any_dtype(df['date_obtained'])
        else ym
    )
    ym = ym.fillna(fallback_date)

    lookup = macro_sorted.set_index('year_month')
    latest_row = macro_sorted.iloc[-1]
    earliest_row = macro_sorted.iloc[0]
    macro_cols: dict[str, np.ndarray] = {}
    for col in MACRO_COLUMNS:
        if col not in lookup.columns:
            continue
        mapped = ym.map(lookup[col])
        mapped = mapped.where(ym >= earliest_row['year_month'], earliest_row[col])
        mapped = mapped.fillna(latest_row[col])
        macro_cols[col] = mapped.astype(float).to_numpy()
    if macro_cols:
        df = pd.concat([df, pd.DataFrame(macro_cols, index=df.index)], axis=1)
    return df


def _onehot(df: pd.DataFrame, artifacts: PreprocessingArtifacts,
            columns: Sequence[str]) -> pd.DataFrame:
    """One-hot with the vocabulary frozen at fit time.

    Unseen categories at transform time become all-zero rows rather than new
    columns, so the matrix width is stable across train and serve.
    """
    for column in columns:
        if column not in df.columns:
            continue
        categories = artifacts.onehot_categories.get(column, [])
        values = df[column].astype(str)
        for category in categories:
            df[f'{column}_{category}'] = (values == category).astype(int)
        df = df.drop(columns=[column])
    return df


def fit_preprocessing(train_df: pd.DataFrame,
                      config: PreprocessConfig | None = None
                      ) -> PreprocessingArtifacts:
    """Learn every fitted quantity from the TRAINING fold only."""
    config = config or PreprocessConfig()
    clean_train = _clean_raw_sentinels(train_df)
    years = pd.to_numeric(clean_train.get('year_built'), errors='coerce') \
        if 'year_built' in clean_train.columns else pd.Series(dtype=float)
    years = years.where((years >= 1700) & (years <= config.reference_year + 4))
    artifacts = PreprocessingArtifacts(
        reference_year=config.reference_year,
        median_year_built=float(years.median() if not years.dropna().empty else np.nan),
        config=config.as_dict(),
    )
    if np.isnan(artifacts.median_year_built):
        artifacts.median_year_built = float(config.reference_year - 30)

    for column in config.onehot_columns:
        if column in clean_train.columns:
            artifacts.onehot_categories[column] = sorted(
                clean_train[column].dropna().astype(str).unique().tolist())

    numeric = clean_train.select_dtypes(include=[np.number]).columns
    artifacts.numeric_medians = {
        column: float(clean_train[column].median())
        for column in numeric
        if column not in LEAKY_COLUMNS
        and column != 'price'
        and column not in ZERO_WHEN_ABSENT_SURFACES
        and clean_train[column].notna().any()
    }
    # Ensure Statbel sector demographic medians are also populated even when fit_preprocessing
    # runs before _engineer (build_dataset order), falling back to Ghent sector medians.
    sectors_gdf = _get_sectors_gdf()
    if sectors_gdf is not None:
        for sec_col in ('sector_pop_density', 'sector_avg_hh_size', 'sector_cars_per_hh',
                        'sector_avg_income', 'sector_median_income', 'sector_income_iqr'):
            if sec_col in sectors_gdf.columns and sec_col not in artifacts.numeric_medians:
                s_val = pd.to_numeric(sectors_gdf[sec_col], errors='coerce').median()
                if pd.notna(s_val):
                    artifacts.numeric_medians[sec_col] = float(s_val)
    # Fit property_size median strictly on houses with positive plot size
    if 'property_size' in clean_train.columns:
        plot = pd.to_numeric(clean_train['property_size'], errors='coerce').where(lambda s: s > 0)
        if 'property_type' in clean_train.columns:
            house_plot = plot[clean_train['property_type'].astype(str) == 'HOUSE']
        else:
            house_plot = plot
        if house_plot.notna().any():
            artifacts.numeric_medians['property_size'] = float(house_plot.median())
    if 'epc_cert_year' in clean_train.columns and clean_train['epc_cert_year'].notna().any():
        epc_ages = (config.reference_year - pd.to_numeric(
            clean_train['epc_cert_year'], errors='coerce')).clip(lower=0)
        if epc_ages.notna().any():
            artifacts.numeric_medians['epc_cert_age'] = float(epc_ages.median())
    artifacts.numeric_medians.setdefault('distance_to_street', 15.0)

    return artifacts


def _impute_train_only(df: pd.DataFrame,
                       artifacts: PreprocessingArtifacts,
                       config: PreprocessConfig) -> pd.DataFrame:
    """Median imputation with price-free predictors.

    The previous implementation fitted a Poisson GLM for ``cadastral_income``
    using ``price`` and ``price_sq`` as predictors -- the target leaking into the
    model's second-most important feature.  Here imputation uses property
    attributes only, and the models are fitted on the training fold.
    """
    df = df.copy()
    numeric = df.select_dtypes(include=[np.number]).columns
    for column in numeric:
        if column in LEAKY_COLUMNS or column == 'price':
            continue
        if df[column].isna().any():
            median = artifacts.numeric_medians.get(column)
            if median is None:
                continue
            df[column] = df[column].fillna(median)
    return df


def transform(df: pd.DataFrame, artifacts: PreprocessingArtifacts
              ) -> pd.DataFrame:
    """Apply the fitted transform.  Deterministic and side-effect free."""
    config = PreprocessConfig(**{**artifacts.config,
                                 'onehot_columns': tuple(
                                     artifacts.config.get('onehot_columns',
                                                          ['property_type']))})
    df = _engineer(df, artifacts, config)
    df, _indicators = _ensure_booleans(df)
    df = _onehot(df, artifacts, config.onehot_columns)
    # Clean Immoweb form-default roomCount < bedroomCount BEFORE median imputation
    if {'number_of_rooms', 'number_of_bedrooms'}.issubset(df.columns):
        rooms = pd.to_numeric(df['number_of_rooms'], errors='coerce')
        beds = pd.to_numeric(df['number_of_bedrooms'], errors='coerce')
        invalid_rooms = rooms.notna() & beds.notna() & (rooms < beds)
        df['number_of_rooms'] = rooms.where(~invalid_rooms)

    df = _impute_train_only(df, artifacts, config)

    # Post-imputation consistency: number_of_rooms must be at least number_of_bedrooms,
    # and garden_surface cannot exceed plot property_size when plot size > 0.
    if {'number_of_rooms', 'number_of_bedrooms'}.issubset(df.columns):
        rooms = pd.to_numeric(df['number_of_rooms'], errors='coerce')
        beds = pd.to_numeric(df['number_of_bedrooms'], errors='coerce')
        df['number_of_rooms'] = np.maximum(rooms, beds)

    if {'garden_surface', 'property_size'}.issubset(df.columns):
        garden = pd.to_numeric(df['garden_surface'], errors='coerce')
        plot = pd.to_numeric(df['property_size'], errors='coerce')
        exceeds = garden.notna() & plot.notna() & (plot > 0) & (garden > plot)
        df['garden_surface'] = np.where(exceeds, plot, garden)

    return df


# =============================================================================
# Stage C: folds (structural leakage prevention)
# =============================================================================

def assign_folds(df: pd.DataFrame, n_folds: int = 5, *,
                 strategy: str = 'grouped', seed: int = 42,
                 time_column: str = 'date_obtained') -> pd.DataFrame:
    """Attach a ``fold`` column that training must honour.

    ``grouped``  keeps every observation of one physical property in one fold --
                 this is what prevents the 70% duplicate-address overlap.
    ``temporal`` trains on earlier scrape dates and tests on later ones; the
                 only split that answers "what will tomorrow's listing sell for".
    ``random``   row-wise random.  Provided for comparability with the old
                 numbers, and clearly the wrong default.
    """
    df = df.copy()
    key = df.get('property_key', _property_key(df))

    if strategy == 'random':
        rng = np.random.default_rng(seed)
        df['fold'] = rng.integers(0, n_folds, size=len(df))
    elif strategy == 'temporal':
        dates = pd.to_datetime(df[time_column], errors='coerce')
        order = dates.rank(method='first', na_option='bottom')
        df['fold'] = pd.qcut(order, n_folds, labels=False, duplicates='drop')
        df['fold'] = df['fold'].astype(int)
    elif strategy == 'grouped':
        groups = pd.Series(key.astype(str)).drop_duplicates().tolist()
        rng = np.random.default_rng(seed)
        rng.shuffle(groups)
        assignment = {group: i % n_folds for i, group in enumerate(groups)}
        df['fold'] = key.astype(str).map(assignment).astype(int)
    else:
        raise ValueError(f'unknown strategy {strategy!r}')

    return df


def fold_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Rows and group counts per fold, plus the overlap that must be zero."""
    key = df.get('property_key', _property_key(df)).astype(str)
    rows = []
    for fold, part in df.groupby('fold'):
        rows.append({'fold': int(fold), 'rows': len(part),
                     'groups': part.pipe(lambda d: key.loc[d.index]).nunique()})
    summary = pd.DataFrame(rows).sort_values('fold').reset_index(drop=True)

    # Groups appearing in more than one fold is the defect we are guarding.
    per_group = df.assign(_key=key).groupby('_key')['fold'].nunique()
    summary.attrs['leaked_groups'] = int((per_group > 1).sum())
    summary.attrs['total_groups'] = int(per_group.size)
    return summary


# =============================================================================
# Orchestration
# =============================================================================

def build_dataset(raw_df: pd.DataFrame,
                  config: PreprocessConfig | None = None,
                  *,
                  fold_strategy: str | None = 'grouped',
                  n_folds: int = 5,
                  with_cross_row_features: bool = False,
                  output_dir: str | None = None) -> dict[str, Any]:
    """raw table -> validated training matrix.

    Returns a dict with ``train_ready``, ``report``, ``funnel``, ``artifacts``
    and ``folds``.  When ``output_dir`` is given, writes Parquet/CSV plus the
    validation report.
    """
    config = config or PreprocessConfig()

    cleaned = clean_raw(raw_df, config, annotate=False)
    df = cleaned['kept']
    funnel = cleaned['funnel']

    artifacts = fit_preprocessing(df, config)
    matrix = transform(df, artifacts)

    if with_cross_row_features:
        matrix = _add_spatial_lag(matrix)
    else:
        matrix = matrix.drop(columns=[c for c in LEAKY_COLUMNS if c in matrix.columns])

    if config.forbid_leaky_features:
        leaked = sorted(LEAKY_COLUMNS & set(matrix.columns))
        if leaked:
            raise ValueError(
                f'leaky columns reached the feature matrix: {leaked}. '
                f'Set with_cross_row_features=False or remove them explicitly.')

    # Keep the group key so folded splits can honour it; drop only columns that
    # are identifiers, free text, always-empty placeholders, or post-launch/constant fields.
    drop_meta = [c for c in (
        'street', 'house_number', 'box', 'external_reference',
        'title', 'description', 'description_nl', 'description_fr', 'description_en',
        'text_recovered_flags', 'city_raw', 'last_modification_date',
        'publication_expiration_date', 'availability_date',
        'price_min', 'price_max', 'price_old_value',
        *ALWAYS_EMPTY_OR_DEAD,
        *CONSTANT_OR_POST_LAUNCH_COLUMNS,
    ) if c in matrix.columns]
    matrix = matrix.drop(columns=drop_meta)

    # Drop any single-class *_missing or constant columns so no degenerate column
    # reaches the feature matrix.
    keep_protected = {'property_key', 'date_obtained', 'property_id', 'fold', 'price'}
    constant_cols = [
        c for c in matrix.columns
        if c not in keep_protected and matrix[c].nunique(dropna=False) <= 1
    ]
    if constant_cols:
        matrix = matrix.drop(columns=constant_cols)

    if fold_strategy:
        matrix = assign_folds(matrix, n_folds=n_folds, strategy=fold_strategy,
                              time_column='date_obtained')
        matrix = _add_oof_spatial_features(matrix)

    keep_meta = [c for c in ('property_key', 'date_obtained', 'property_id')
                 if c in matrix.columns]
    artifacts.feature_columns = [
        c for c in matrix.columns
        if c not in keep_meta + ['fold', 'price']]

    report = validate(matrix, build_training_spec(), source='build_dataset',
                      check_leakage=config.forbid_leaky_features)
    report.funnel = funnel

    result = {
        'train_ready': matrix,
        'report': report,
        'funnel': funnel,
        'artifacts': artifacts,
        'folds': fold_summary(matrix) if 'fold' in matrix.columns else None,
        'coverage': report.coverage,
    }

    if output_dir:
        _write_outputs(result, output_dir)
    return result


def _add_oof_spatial_features(df: pd.DataFrame) -> pd.DataFrame:
    """Fold-safe out-of-fold spatial KNN & neighbourhood price/m2 priors.

    Unlike the legacy ``spatial_lag_10`` (which queried a KDTree fitted on all
    rows including the test fold), every fold ``k`` here queries a KDTree and
    neighbourhood aggregation built **strictly on rows where ``fold != k``**.
    Because ``assign_folds`` places all observations of a physical property in
    the same fold, a property never sees itself or another snapshot of itself.
    """
    from scipy.spatial import cKDTree
    if 'fold' not in df.columns or not {'latitude', 'longitude', 'price', 'livable_area'}.issubset(df.columns):
        return df

    df = df.copy()
    area = pd.to_numeric(df['livable_area'], errors='coerce').where(lambda s: s > 0)
    price = pd.to_numeric(df['price'], errors='coerce')
    price_sqm = price / area
    global_median_sqm = float(price_sqm.median()) if price_sqm.notna().any() else 3500.0

    oof_knn_price_sqm = np.full(len(df), global_median_sqm, dtype=float)
    oof_dist_5nn_km = np.full(len(df), 0.5, dtype=float)
    oof_wijk_price_sqm = np.full(len(df), global_median_sqm, dtype=float)

    coords = df[['latitude', 'longitude']].to_numpy(dtype=float)
    folds = df['fold'].to_numpy()
    has_wijk = 'neighbourhood' in df.columns
    wijken = df['neighbourhood'].astype(str).to_numpy() if has_wijk else None

    for f in np.unique(folds):
        train_mask = (folds != f) & np.isfinite(coords[:, 0]) & np.isfinite(coords[:, 1]) & price_sqm.notna().to_numpy()
        val_mask = (folds == f) & np.isfinite(coords[:, 0]) & np.isfinite(coords[:, 1])
        if not train_mask.any() or not val_mask.any():
            continue

        train_coords = coords[train_mask]
        train_psqm = price_sqm.to_numpy()[train_mask]
        fold_median = float(np.median(train_psqm))

        k = min(10, len(train_coords))
        tree = cKDTree(train_coords)
        dists, idx = tree.query(coords[val_mask], k=k)
        if k == 1:
            dists = dists[:, None]
            idx = idx[:, None]
        oof_knn_price_sqm[val_mask] = np.median(train_psqm[idx], axis=1)
        k_dist = min(5, k)
        oof_dist_5nn_km[val_mask] = np.mean(dists[:, :k_dist] * 111.32, axis=1)

        if has_wijk:
            train_df_wijk = pd.DataFrame({'w': wijken[train_mask], 'p': train_psqm})
            wijk_medians = train_df_wijk.groupby('w')['p'].median().to_dict()
            oof_wijk_price_sqm[val_mask] = [
                wijk_medians.get(w, fold_median) for w in wijken[val_mask]
            ]

    df['oof_knn10_median_price_sqm'] = oof_knn_price_sqm
    df['oof_dist_5nn_km'] = oof_dist_5nn_km
    if has_wijk:
        df['oof_neighbourhood_median_price_sqm'] = oof_wijk_price_sqm

    if 'statbel_sector' in df.columns:
        sectors = df['statbel_sector'].astype(str).to_numpy()
        oof_sector_price_sqm = np.full(len(df), global_median_sqm, dtype=float)
        for f in np.unique(folds):
            train_mask = (folds != f) & price_sqm.notna().to_numpy()
            val_mask = folds == f
            if not train_mask.any() or not val_mask.any():
                continue
            train_psqm = price_sqm.to_numpy()[train_mask]
            fold_median = float(np.median(train_psqm))
            train_sec = pd.DataFrame({'s': sectors[train_mask], 'p': train_psqm})
            sec_agg = train_sec.groupby('s')['p'].agg(['count', 'median'])
            # Smooth small sectors toward fold median (empirical Bayes shrinkage, prior weight m=5)
            m = 5.0
            sec_smoothed = {
                idx: (row['count'] * row['median'] + m * fold_median) / (row['count'] + m)
                for idx, row in sec_agg.iterrows()
                if idx != 'Unknown'
            }
            oof_sector_price_sqm[val_mask] = [
                sec_smoothed.get(s, fold_median) for s in sectors[val_mask]
            ]
        df['oof_sector_median_price_sqm'] = oof_sector_price_sqm

    return df


def _add_spatial_lag(df: pd.DataFrame) -> pd.DataFrame:
    """Opt-in cross-row feature.  Must be computed fold-wise to be legitimate."""
    from scipy.spatial import cKDTree
    valid = df['latitude'].notna() & df['longitude'].notna() & df['price'].notna()
    if not valid.any():
        return df
    coords = df.loc[valid, ['latitude', 'longitude']].to_numpy()
    prices = df.loc[valid, 'price'].to_numpy()
    tree = cKDTree(coords)
    distances, indices = tree.query(coords, k=min(11, len(coords)))
    neighbours = prices[indices[:, 1:]] if indices.shape[1] > 1 else prices[:, None]
    df.loc[valid, 'spatial_lag_10'] = neighbours.mean(axis=1)
    df.loc[valid, 'avg_dist_5nn'] = (distances[:, 1:6] * 111).mean(axis=1) \
        if distances.shape[1] > 1 else 0.0
    return df


def _write_outputs(result: dict, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    matrix = result['train_ready']
    parquet_path = os.path.join(output_dir, 'training_dataset.parquet')
    matrix.to_parquet(parquet_path, index=False)
    matrix.to_csv(os.path.join(output_dir, 'training_dataset.csv'), index=False)
    result['artifacts'].save(os.path.join(output_dir, 'preprocessing_artifacts.joblib'))
    result['report'].to_json(os.path.join(output_dir, 'validation_report.json'))
    result['report'].to_markdown_file(os.path.join(output_dir, 'validation_report.md'))
    with open(os.path.join(output_dir, 'config.json'), 'w', encoding='utf-8') as handle:
        json.dump({'config': result['artifacts'].config,
                   'n_features': len(result['artifacts'].feature_columns),
                   'features': result['artifacts'].feature_columns,
                   'funnel': result['funnel']}, handle, indent=2, default=str)
