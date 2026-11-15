"""Full exploratory data analysis over a parsed Immoweb dataset.

Produces a single JSON document describing *every* field plus spatial,
temporal and target-relationship aggregates, so a UI can browse the dataset
without re-reading 143k rows of HTML-derived columns.

Design notes
------------
*   Polars, not pandas: the corpus is ~144k rows x 130 columns read straight
    from parquet, and the per-field work (null counts, cardinality, quantiles)
    is a columnar aggregation.
*   Every field is classified into a ``kind`` (numeric/categorical/boolean/
    text/date/geo) because that decides which summary is meaningful.  A single
    generic describe() produces garbage for identifiers and free text.
*   Statistics are computed over *declared* values only.  Immoweb sends null
    for "agency did not fill this in", so a null rate is a finding, not an
    error -- it is reported next to every field rather than being imputed away.
*   Nothing here mutates the dataset or drops rows: this is observation.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import numpy as np
import polars as pl

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs', 'eda')

# Columns that encode the answer rather than a predictor.  Flagged in the
# report so a reader cannot accidentally treat them as features.
LEAKY_COLUMNS = {'price', 'price_sq', 'log_price', 'spatial_lag_10', 'avg_dist_5nn',
                 'price_min', 'price_max', 'delta_eur', 'delta_pct',
                 'predicted_k', 'predicted_eur', 'asking_eur'}

# Columns that identify a record rather than describe a property.
ID_COLUMNS = {'property_id', 'external_reference', 'street', 'house_number', 'box',
              'title', 'description', 'url'}

TARGET = 'price'
TARGET_MIN, TARGET_MAX = 25_000, 3_000_000

# Same bounds the validation layer enforces (src/validate_dataset.py).  A
# handful of listings carry coordinates from elsewhere entirely -- 77 rows
# (0.055%) include 1.31/103.86, which is Singapore -- and a single such row
# stretches a map viewport across two continents.  The EDA reports how many it
# excluded rather than silently dropping them.
GHENT_BBOX = ((50.90, 51.20), (3.50, 4.00))


# ---------------------------------------------------------------------------
# Field classification
# ---------------------------------------------------------------------------

def classify_field(name: str, dtype: pl.DataType, series: pl.Series | None = None) -> str:
    """Map a column to the summary that is actually meaningful for it."""
    if isinstance(dtype, (pl.Date, pl.Datetime)):
        return 'date'
    if dtype == pl.Boolean:
        return 'boolean'
    if name in ('latitude', 'longitude'):
        return 'geo'
    if dtype.is_numeric():
        # A numeric column holding a handful of distinct values is an encoded
        # category (energy label score, condition rank), not a measurement.
        if series is not None:
            try:
                n_unique = series.n_unique()
                if n_unique <= 12:
                    return 'categorical'
            except Exception:  # noqa: BLE001 - classification must not fail the run
                pass
        return 'numeric'
    # Object/string columns: distinguish short repeated labels from free text.
    if name in ('description', 'title'):
        return 'text'
    return 'categorical'


def _scalar(value):
    """JSON-safe scalar (polars returns numpy/Decimal types)."""
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, bool):
        return value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (datetime, np.datetime64)):
        return str(value)
    if isinstance(value, (int, float, str)):
        return value
    return str(value)


# ---------------------------------------------------------------------------
# Per-field summaries
# ---------------------------------------------------------------------------

def _numeric_summary(series: pl.Series) -> dict:
    s = series.drop_nulls()
    if s.len() == 0:
        return {'count': 0}
    stats = {
        'count': int(s.len()),
        'mean': _scalar(s.mean()),
        'std': _scalar(s.std()),
        'min': _scalar(s.min()),
        'p01': _scalar(s.quantile(0.01)),
        'p10': _scalar(s.quantile(0.10)),
        'p25': _scalar(s.quantile(0.25)),
        'median': _scalar(s.quantile(0.50)),
        'p75': _scalar(s.quantile(0.75)),
        'p90': _scalar(s.quantile(0.90)),
        'p99': _scalar(s.quantile(0.99)),
        'max': _scalar(s.max()),
        'zeros': int((s == 0).sum()),
        'negatives': int((s < 0).sum()),
    }
    # IQR-based outlier count: robust to the heavy right tail in property prices.
    q1, q3 = stats['p25'], stats['p75']
    if q1 is not None and q3 is not None and q3 > q1:
        iqr = q3 - q1
        lo, hi = q1 - 3 * iqr, q3 + 3 * iqr
        stats['outliers_iqr3'] = int(((s < lo) | (s > hi)).sum())
    return stats


def _value_counts(series: pl.Series, limit: int = 25) -> list[dict]:
    """Top values with counts and share-of-declared. Truncates long text."""
    s = series.drop_nulls()
    total = s.len()
    if total == 0:
        return []
    counts = s.value_counts(sort=True).head(limit)
    out = []
    for row in counts.iter_rows():
        value, count = row[0], row[1]
        if isinstance(value, str) and len(value) > 80:
            value = value[:77] + '...'
        out.append({'value': _scalar(value),
                    'count': int(count),
                    'pct': round(int(count) / total * 100, 2)})
    return out


def field_profile(df: pl.DataFrame, name: str) -> dict:
    series = df[name]
    dtype = series.dtype
    kind = classify_field(name, dtype, series)
    n = df.height
    non_null = int(series.is_not_null().sum())
    try:
        n_unique = int(series.n_unique())
    except Exception:  # noqa: BLE001
        n_unique = None

    profile = {
        'name': name,
        'dtype': str(dtype),
        'kind': kind,
        'non_null': non_null,
        'null_count': n - non_null,
        'null_pct': round((n - non_null) / n * 100, 2) if n else 0.0,
        'n_unique': n_unique,
        'is_leaky': name in LEAKY_COLUMNS,
        'is_id_like': name in ID_COLUMNS,
        'flags': [],
    }

    if profile['null_pct'] >= 95:
        profile['flags'].append('near_empty')
    if n_unique == 1:
        profile['flags'].append('constant')
    if n_unique is not None and n and n_unique == non_null and non_null > 100:
        profile['flags'].append('unique_per_row')

    if kind == 'numeric':
        profile['stats'] = _numeric_summary(series)
    elif kind in ('categorical', 'boolean', 'date', 'geo'):
        profile['values'] = _value_counts(series)
        if kind == 'date':
            non_null_s = series.drop_nulls()
            if non_null_s.len():
                profile['range'] = {'min': _scalar(non_null_s.min()),
                                    'max': _scalar(non_null_s.max())}
    elif kind == 'text':
        lengths = series.drop_nulls().str.len_chars()
        profile['length'] = {
            'count': int(lengths.len()),
            'mean': _scalar(lengths.mean()) if lengths.len() else None,
            'median': _scalar(lengths.quantile(0.5)) if lengths.len() else None,
            'max': _scalar(lengths.max()) if lengths.len() else None,
            'at_2500_cap': int((lengths == 2500).sum()) if lengths.len() else 0,
            'placeholder_dash': int((series == '-').sum()),
        }
        profile['values'] = _value_counts(series, limit=5)

    return profile


# ---------------------------------------------------------------------------
# Cross-field sections
# ---------------------------------------------------------------------------

def target_correlations(df: pl.DataFrame, target: str = TARGET,
                        top_n: int = 40) -> list[dict]:
    """Pearson and Spearman correlation of every numeric field with the target.

    Computed on a target-trimmed frame so a handful of 13M-euro typos do not
    dominate the Pearson coefficient.
    """
    if target not in df.columns:
        return []
    work = _target_frame(df)
    if work.height < 30:
        return []

    out = []
    for name in work.columns:
        if name == target or not work.schema[name].is_numeric():
            continue
        s = work[name].cast(pl.Float64)
        pair = pl.DataFrame({'x': s, 'y': work[target].cast(pl.Float64)}).drop_nulls()
        if pair.height < 30 or pair['x'].n_unique() < 2:
            continue
        pearson = float(np.corrcoef(pair['x'].to_numpy(),
                                    pair['y'].to_numpy())[0, 1])
        try:
            spearman = float(pair.select(
                pl.corr('x', 'y', method='spearman')).item() or 0.0)
        except Exception:  # noqa: BLE001 - spearman needs scipy; degrade gracefully
            spearman = None
        out.append({
            'field': name,
            'pearson': round(pearson, 4) if pearson == pearson else None,
            'spearman': round(spearman, 4) if spearman is not None else None,
            'n': int(pair.height),
            'is_leaky': name in LEAKY_COLUMNS,
        })
    out.sort(key=lambda r: -abs(r['spearman'] if r['spearman'] is not None
                                else (r['pearson'] or 0)))
    return out[:top_n]


def _target_frame(df: pl.DataFrame, target: str = TARGET) -> pl.DataFrame:
    if target not in df.columns:
        return df.head(0)
    return df.filter(
        pl.col(target).is_not_null()
        & (pl.col(target) >= TARGET_MIN)
        & (pl.col(target) <= TARGET_MAX))


def target_by_group(df: pl.DataFrame, columns: list[str],
                    min_group: int = 10) -> dict:
    """Median/mean price per category, for the categorical fields that matter."""
    work = _target_frame(df)
    out = {}
    if work.height == 0:
        return out
    for col in columns:
        if col not in work.columns:
            continue
        try:
            grouped = (work.filter(pl.col(col).is_not_null())
                       .group_by(col)
                       .agg(pl.col(TARGET).median().alias('median'),
                            pl.col(TARGET).mean().alias('mean'),
                            pl.len().alias('n'))
                       .filter(pl.col('n') >= min_group)
                       .sort('median', descending=True))
        except Exception:  # noqa: BLE001
            continue
        out[col] = [{'value': _scalar(r[0]),
                     'median': _scalar(round(r[1])),
                     'mean': _scalar(round(r[2])),
                     'n': int(r[3])} for r in grouped.iter_rows()]
    return out


def missingness_section(df: pl.DataFrame, top_n: int = 40) -> dict:
    """Per-field null rates plus co-missingness blocks.

    Immoweb reveals optional sections only when the agency fills them, so
    missingness is clustered by agency behaviour (sections_filled_count) rather
    than scattered at random.  Measuring that is the point.
    """
    n = df.height
    rows = []
    for name in df.columns:
        null_count = int(df[name].is_null().sum())
        rows.append({'field': name,
                     'null_pct': round(null_count / n * 100, 2) if n else 0.0,
                     'null_count': null_count})
    rows.sort(key=lambda r: -r['null_pct'])

    bands = {'0%': 0, '0-10%': 0, '10-25%': 0, '25-50%': 0,
             '50-75%': 0, '75-95%': 0, '95-100%': 0}
    for r in rows:
        p = r['null_pct']
        if p == 0:
            bands['0%'] += 1
        elif p < 10:
            bands['0-10%'] += 1
        elif p < 25:
            bands['10-25%'] += 1
        elif p < 50:
            bands['25-50%'] += 1
        elif p < 75:
            bands['50-75%'] += 1
        elif p < 95:
            bands['75-95%'] += 1
        else:
            bands['95-100%'] += 1

    return {
        'total_fields': len(df.columns),
        'fields': rows[:top_n],
        'bands': bands,
        'rows_with_no_nulls': int((df.null_count().sum_horizontal() == 0).sum()),
    }


def temporal_section(df: pl.DataFrame, date_col: str = 'date_obtained') -> dict:
    """Observation counts over time (the corpus has known collection gaps)."""
    if date_col not in df.columns:
        return {}
    try:
        dates = pl.Series('d', [str(v) for v in df[date_col].to_list()])
        # date_obtained arrives as DDMMYYYY (sometimes losing a leading zero).
        parsed = dates.str.zfill(8).str.strptime(pl.Date, '%d%m%Y', strict=False)
    except Exception:  # noqa: BLE001
        return {}

    work = pl.DataFrame({'d': parsed}).drop_nulls()
    if work.height == 0:
        return {}
    daily = (work.group_by('d').len().sort('d').rename({'len': 'count'}))
    monthly = (work.with_columns(pl.col('d').dt.strftime('%Y-%m').alias('m'))
               .group_by('m').len().sort('m').rename({'len': 'count'}))
    return {
        'min': str(daily['d'].min()),
        'max': str(daily['d'].max()),
        'distinct_days': int(daily.height),
        'daily': [{'date': str(r[0]), 'count': int(r[1])}
                  for r in daily.iter_rows()],
        'monthly': [{'month': str(r[0]), 'count': int(r[1])}
                    for r in monthly.iter_rows()],
    }


def spatial_section(df: pl.DataFrame, grid_deg: float = 0.005) -> dict:
    """Aggregate to a lat/lon grid and to postal codes for map rendering.

    A 0.005 degree cell is ~550 m N-S, small enough to show neighbourhood
    structure and coarse enough that each cell holds enough properties for a
    stable median.
    """
    if 'latitude' not in df.columns or 'longitude' not in df.columns:
        return {}
    any_coords = df.filter(pl.col('latitude').is_not_null()
                           & pl.col('longitude').is_not_null()
                           & (pl.col('latitude') > 0) & (pl.col('longitude') > 0))
    work = any_coords.filter(
        pl.col('latitude').is_between(*GHENT_BBOX[0])
        & pl.col('longitude').is_between(*GHENT_BBOX[1]))
    if work.height == 0:
        return {}

    priced = _target_frame(work)
    if 'livable_area' in priced.columns:
        priced = priced.with_columns(
            (pl.col(TARGET) / pl.col('livable_area')).alias('price_sqm'))

    cells = (priced
             .with_columns(
                 (pl.col('latitude') / grid_deg).floor().alias('gy'),
                 (pl.col('longitude') / grid_deg).floor().alias('gx'))
             .group_by(['gx', 'gy'])
             .agg(pl.col(TARGET).median().alias('median_price'),
                  pl.col(TARGET).mean().alias('mean_price'),
                  pl.col(TARGET).count().alias('n'),
                  (pl.col(TARGET) / pl.col('livable_area')).median().alias('median_price_sqm')
                  if 'livable_area' in priced.columns else pl.lit(None).alias('median_price_sqm'),
                  pl.col('latitude').mean().alias('lat'),
                  pl.col('longitude').mean().alias('lon'))
             .filter(pl.col('n') >= 3)
             .sort('n', descending=True))

    postal = (priced.group_by('postal_code')
              .agg(pl.col(TARGET).median().alias('median_price'),
                   pl.col(TARGET).count().alias('n'),
                   pl.col('latitude').mean().alias('lat'),
                   pl.col('longitude').mean().alias('lon'))
              .sort('n', descending=True)) if 'postal_code' in priced.columns else pl.DataFrame()

    neighbourhood = pl.DataFrame()
    if 'neighbourhood' in priced.columns:
        neighbourhood = (priced.filter(pl.col('neighbourhood').is_not_null())
                         .group_by('neighbourhood')
                         .agg(pl.col(TARGET).median().alias('median_price'),
                              pl.col(TARGET).count().alias('n'),
                              pl.col('latitude').mean().alias('lat'),
                              pl.col('longitude').mean().alias('lon'))
                         .sort('n', descending=True))

    def _rows(frame):
        return [{k: _scalar(v) for k, v in zip(frame.columns, r)}
                for r in frame.iter_rows()] if frame.height else []

    return {
        'grid_size_deg': grid_deg,
        'n_with_coords': int(work.height),
        'n_with_coords_any': int(any_coords.height),
        'n_outside_bbox': int(any_coords.height - work.height),
        'n_priced_with_coords': int(priced.height),
        'bbox': {
            'lat_min': _scalar(work['latitude'].min()),
            'lat_max': _scalar(work['latitude'].max()),
            'lon_min': _scalar(work['longitude'].min()),
            'lon_max': _scalar(work['longitude'].max()),
        },
        'cells': _rows(cells),
        'postal_codes': _rows(postal),
        'neighbourhoods': _rows(neighbourhood),
    }


def duplicate_section(df: pl.DataFrame) -> dict:
    """Snapshot and relisting structure.

    Distinguishes 'same listing re-observed on another day' (legitimate, and the
    dataset is one row per scrape) from 'same address under a new property_id'
    (a relisting that will leak across a random split).
    """
    out = {}
    if 'property_id' in df.columns:
        out['unique_property_ids'] = int(df['property_id'].n_unique())
        out['rows'] = df.height
        out['rows_per_property'] = round(
            df.height / max(df['property_id'].n_unique(), 1), 2)

    address_cols = [c for c in ('postal_code', 'street', 'house_number', 'livable_area')
                    if c in df.columns]
    if len(address_cols) == 4:
        work = df.filter(pl.col('street').is_not_null()
                         & pl.col('house_number').is_not_null())
        if work.height:
            grouped = (work.group_by(address_cols)
                       .agg(pl.col('property_id').n_unique().alias('ids'),
                            pl.len().alias('rows')))
            multi = grouped.filter(pl.col('ids') > 1)
            out['address_groups'] = int(grouped.height)
            out['address_groups_with_multiple_ids'] = int(multi.height)
            out['ids_in_multi_id_groups'] = int(multi['ids'].sum()) if multi.height else 0
    return out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def build_report(df: pl.DataFrame, label: str = 'full') -> dict:
    """Assemble the complete EDA document."""
    fields = [field_profile(df, name) for name in df.columns]

    # Categorical group-by is only meaningful for low-cardinality columns.
    group_cols = [f['name'] for f in fields
                  if f['kind'] in ('categorical', 'boolean')
                  and f['n_unique'] is not None and 1 < f['n_unique'] <= 40]

    report = {
        'label': label,
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'dataset': {
            'rows': df.height,
            'columns': df.width,
            'target': TARGET,
            'target_trim': [TARGET_MIN, TARGET_MAX],
            'priced_rows': int(_target_frame(df).height),
        },
        'fields': fields,
        'correlations': target_correlations(df),
        'target_by_group': target_by_group(df, group_cols),
        'missingness': missingness_section(df),
        'temporal': temporal_section(df),
        'spatial': spatial_section(df),
        'duplicates': duplicate_section(df),
    }
    return report


def write_report(report: dict, output_dir: str = DEFAULT_OUTPUT_DIR) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'eda_report.json')
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as handle:
        json.dump(report, handle, allow_nan=False)
    os.replace(tmp, path)
    size_mb = os.path.getsize(path) / 1e6
    print(f"  EDA report -> {path} ({size_mb:.1f} MB)")
    return path


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate a full EDA report.')
    parser.add_argument('--input', default=os.path.join(BASE_DIR, 'outputs',
                                                        'property_data_full.parquet'))
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--label', default='full')
    parser.add_argument('--sample', type=int, default=None,
                        help='Only profile the first N rows (fast smoke test).')
    args = parser.parse_args()

    print(f"Loading {args.input} ...")
    if args.input.endswith('.parquet'):
        df = pl.read_parquet(args.input)
    else:
        df = pl.read_csv(args.input, infer_schema_length=0)
    if args.sample:
        df = df.head(args.sample)
    print(f"  {df.height:,} rows x {df.width} columns")

    report = build_report(df, label=args.label)
    write_report(report, args.output_dir)

    print("\nField inventory by kind:")
    kinds = {}
    for f in report['fields']:
        kinds[f['kind']] = kinds.get(f['kind'], 0) + 1
    for kind, count in sorted(kinds.items(), key=lambda kv: -kv[1]):
        print(f"  {kind:<12} {count:>4}")

    empty = [f['name'] for f in report['fields'] if f['null_pct'] >= 99]
    print(f"\nNear-empty fields ({len(empty)}): {', '.join(empty[:12])}"
          f"{' ...' if len(empty) > 12 else ''}")

    print("\nTop correlations with price:")
    for row in report['correlations'][:10]:
        flag = ' [LEAKY]' if row['is_leaky'] else ''
        print(f"  {row['field']:<34} spearman={row['spearman']:>7}  "
              f"pearson={row['pearson']:>7}{flag}")

    spatial = report['spatial']
    if spatial:
        print(f"\nSpatial: {spatial['n_priced_with_coords']:,} priced with coords, "
              f"{len(spatial['cells'])} grid cells, "
              f"{len(spatial['postal_codes'])} postal codes")


if __name__ == '__main__':
    main()
