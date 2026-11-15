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

BOOLEAN_PREFIXES = ('has_', 'is_')


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
    """One row per listing id, chosen by scrape date.

    ``keep='first'`` retains the earliest observation, which matches how the model
    is used (scoring a listing at launch); ``keep='last'`` retains the final state.
    Rows are sorted by date first so ``first``/``last`` mean what they say.
    """
    if 'property_id' not in df.columns:
        return df
    ordered = df.sort_values('date_obtained', kind='stable')
    return ordered.drop_duplicates(subset=['property_id'], keep=keep)


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
    """Coerce has_*/is_* to bool, recording which values were undeclared.

    Returns the frame plus the list of ``*_missing`` indicator columns created.
    """
    df = df.copy()
    indicators = []
    _strings = {'True': True, 'False': False, 'true': True, 'false': False,
                '1': True, '0': False}
    for column in [c for c in df.columns if c.startswith(BOOLEAN_PREFIXES)]:
        series = df[column]
        if series.dtype == bool:
            continue
        # Map once instead of chaining replace(): pandas emits a downcasting
        # FutureWarning for replace() on object columns, and a single pass makes
        # the tri-state handling explicit.
        def _coerce(value):
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return _strings.get(value.strip())
            if isinstance(value, (int, float)) and not pd.isna(value):
                return bool(value) if value in (0, 1) else None
            return None
        normalised = series.map(_coerce).astype(object)
        was_missing = normalised.isna()
        if was_missing.any():
            indicator = f'{column}_missing'
            df[indicator] = was_missing.astype(int)
            indicators.append(indicator)
        df[column] = pd.Series(normalised.notna().to_numpy(),
                               index=df.index, dtype=bool)
    return df, indicators


def _engineer(df: pd.DataFrame, artifacts: PreprocessingArtifacts,
              config: PreprocessConfig) -> pd.DataFrame:
    """Row-wise features.  No cross-row aggregation, no target use.

    Tolerates un-parsed input: dates may still be strings and numerics may be
    object-typed, because at inference time the payload comes straight from the
    parser rather than from a previously cleaned table.
    """
    df = df.copy()

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
        year = year.fillna(artifacts.median_year_built)
        df['year_built'] = year
        df['age'] = artifacts.reference_year - year

    # true time-on-market, from the real publication timestamp
    if {'publication_date', 'last_modification_date'}.issubset(df.columns):
        duration = (df['last_modification_date'] - df['publication_date']).dt.days
        df['listing_duration_days'] = duration.clip(lower=0)

    if 'description' in df.columns:
        text = df['description'].fillna('').astype(str)
        df['description_length'] = text.str.len()
        df['has_meaningful_description'] = (text.str.len() > 50).astype(int)
        df['description_is_placeholder'] = (
            text.str.strip().isin(('', '-', '.')).astype(int))

    df = _add_geo_features(df)
    return df


def _add_geo_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cheap location features derived from the row's own coordinates only."""
    center_lat, center_lon = 51.053098573153505, 3.7265139580328865
    if {'latitude', 'longitude'}.issubset(df.columns):
        df['dist_from_center'] = np.sqrt(
            (df['latitude'] - center_lat) ** 2 +
            (df['longitude'] - center_lon) ** 2) * 100
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
    artifacts = PreprocessingArtifacts(
        reference_year=config.reference_year,
        median_year_built=float(pd.to_numeric(
            train_df.get('year_built'), errors='coerce').median()
            if 'year_built' in train_df.columns else np.nan),
        config=config.as_dict(),
    )
    if np.isnan(artifacts.median_year_built):
        artifacts.median_year_built = float(config.reference_year - 30)

    for column in config.onehot_columns:
        if column in train_df.columns:
            artifacts.onehot_categories[column] = sorted(
                train_df[column].dropna().astype(str).unique().tolist())

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
    df = _impute_train_only(df, artifacts, config)
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
    numeric = df.select_dtypes(include=[np.number]).columns
    artifacts.numeric_medians = {
        column: float(df[column].median())
        for column in numeric
        if column not in LEAKY_COLUMNS and column != 'price'
        and df[column].notna().any()
    }

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
    # are identifiers or free text.  ``number_of_rooms`` is retained deliberately:
    # 12.5% of rows carry a placeholder roomCount of 1 alongside 2+ bedrooms, which
    # is only auditable if the column survives.
    drop_meta = [c for c in ('street', 'house_number', 'box', 'external_reference',
                             'title', 'description', 'text_recovered_flags',
                             'city_raw', 'last_modification_date')
                 if c in matrix.columns]
    matrix = matrix.drop(columns=drop_meta)

    if fold_strategy:
        matrix = assign_folds(matrix, n_folds=n_folds, strategy=fold_strategy,
                              time_column='date_obtained')

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
