"""Dataset validation: is this table trustworthy enough to train on?

The contract this module enforces
---------------------------------
1.  **Every dropped row has a recorded reason.**  Filters that silently shrink a
    dataset are indistinguishable from bugs.  ``row_funnel`` attributes each
    dropped row to the *first* gate it failed and reports the funnel, so a
    surprising row count can always be traced to a named rule.
2.  **Checks are declarative and reusable.**  ``DatasetSpec`` is the single
    source of truth for column expectations and is the same object used to
    validate at training time and to validate an inference payload.
3.  **Validation never mutates.**  It inspects and reports.  Cleaning lives in
    ``preprocessing``.
4.  **The report is a build artifact.**  Machine-readable JSON for gating CI and
    a Markdown summary for humans, both written next to the dataset.

Severity
--------
``critical``  a row/product that violates this must not reach training.
``warning``   suspicious but usable; counted and surfaced, never dropped.
``info``      descriptive coverage statistics, no pass/fail meaning.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

CRITICAL = 'critical'
WARNING = 'warning'
INFO = 'info'

# Ghent metropolitan area.  Deliberately a rectangle rather than a polygon: the
# corpus contains sub-municipalities (Oostakker, Wondelgem, Drongen, ...) and a
# tight polygon would reject them.
GHENT_BBOX = {'lat_min': 50.90, 'lat_max': 51.20, 'lon_min': 3.50, 'lon_max': 4.00}


# =============================================================================
# Column specification
# =============================================================================

@dataclass
class ColumnSpec:
    """Expectations for one column.

    Floats, not sentinels: ``None`` means "no constraint".
    """
    name: str
    required: bool = False
    minimum: float | None = None
    maximum: float | None = None
    allowed: Sequence[str] | None = None
    null_rate_max: float | None = None
    description: str = ''


@dataclass
class DatasetSpec:
    """The column contract for a dataset."""
    name: str
    columns: dict[str, ColumnSpec] = field(default_factory=dict)

    def add(self, *specs: ColumnSpec) -> 'DatasetSpec':
        for spec in specs:
            self.columns[spec.name] = spec
        return self

    def required_columns(self) -> list[str]:
        return [s.name for s in self.columns.values() if s.required]


def build_raw_spec() -> DatasetSpec:
    """Contract for ``property_data_raw.*`` -- the parser's output.

    Ranges here are *hard* domain bounds: a violation means the parse is wrong
    or the listing is not a residential for-sale property.  Plausibility bounds
    (is this price typical?) belong in the quality checks, not here.
    """
    spec = DatasetSpec('property_data_raw')
    spec.add(
        ColumnSpec('property_id', required=True, description='Immoweb listing id'),
        ColumnSpec('date_obtained', required=True, description='DDMMYYYY scrape tag'),
        ColumnSpec('price', minimum=1.0, maximum=50_000_000.0,
                   description='asking price in EUR'),
        ColumnSpec('livable_area', minimum=1.0, maximum=10_000.0,
                   description='net habitable surface in m2'),
        ColumnSpec('latitude', minimum=-90.0, maximum=90.0),
        ColumnSpec('longitude', minimum=-180.0, maximum=180.0),
        ColumnSpec('year_built', minimum=1700.0, maximum=2100.0),
        ColumnSpec('number_of_bedrooms', minimum=0.0, maximum=50.0),
        ColumnSpec('postal_code', description='4-digit Belgian postal code'),
        ColumnSpec('property_type', allowed=('HOUSE', 'APARTMENT'),
                   description='residential type; others are excluded'),
        ColumnSpec('energy_label', null_rate_max=0.60),
        ColumnSpec('building_condition', null_rate_max=0.60),
        ColumnSpec('description', null_rate_max=0.95),
    )
    return spec


def build_training_spec() -> DatasetSpec:
    """Contract for the matrix handed to the model."""
    spec = DatasetSpec('training')
    spec.add(
        # price is in EUR at this stage (scaling to thousands is a modelling choice,
        # not a data property, and scaling before validation hides the units).
        ColumnSpec('price', required=True, minimum=1.0, maximum=50_000_000.0,
                   description='target, asking price in EUR'),
        ColumnSpec('livable_area', required=True, minimum=10.0, maximum=1000.0),
        ColumnSpec('latitude', required=True),
        ColumnSpec('longitude', required=True),
        ColumnSpec('dist_from_center', required=True, minimum=0.0),
    )
    return spec


# =============================================================================
# Check results
# =============================================================================

@dataclass
class CheckResult:
    name: str
    severity: str
    passed: bool
    n_affected: int
    n_total: int
    message: str
    detail: dict = field(default_factory=dict)

    @property
    def rate(self) -> float:
        return self.n_affected / self.n_total if self.n_total else 0.0

    def as_dict(self) -> dict:
        return {
            'name': self.name,
            'severity': self.severity,
            'passed': bool(self.passed),
            'n_affected': int(self.n_affected),
            'n_total': int(self.n_total),
            'rate': round(self.rate, 6),
            'message': self.message,
            'detail': self.detail,
        }


class ValidationReport:
    """Accumulates check results and renders JSON + Markdown."""

    def __init__(self, dataset_name: str, n_rows: int, n_cols: int,
                 source: str | None = None):
        self.dataset_name = dataset_name
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.source = source
        self.checks: list[CheckResult] = []
        self.funnel: list[dict] = []
        self.coverage: dict = {}

    def add(self, result: CheckResult) -> CheckResult:
        self.checks.append(result)
        return result

    @property
    def critical_failures(self) -> list[CheckResult]:
        return [c for c in self.checks if c.severity == CRITICAL and not c.passed]

    @property
    def warnings(self) -> list[CheckResult]:
        return [c for c in self.checks if c.severity == WARNING and not c.passed]

    def ok(self) -> bool:
        return not self.critical_failures

    def require_ok(self) -> None:
        """Raise if any critical check failed.  Call before training."""
        failed = self.critical_failures
        if not failed:
            return
        lines = [f'{c.name}: {c.message} ({c.n_affected}/{c.n_total})'
                 for c in failed]
        raise DatasetValidationError(
            f'{len(failed)} critical check(s) failed for {self.dataset_name}:\n  '
            + '\n  '.join(lines))

    def as_dict(self) -> dict:
        return {
            'dataset': self.dataset_name,
            'source': self.source,
            'generated_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'shape': {'rows': int(self.n_rows), 'columns': int(self.n_cols)},
            'ok': self.ok(),
            'critical_failures': len(self.critical_failures),
            'warnings': len(self.warnings),
            'funnel': self.funnel,
            'coverage': self.coverage,
            'checks': [c.as_dict() for c in self.checks],
        }

    def to_json(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w', encoding='utf-8') as handle:
            json.dump(self.as_dict(), handle, indent=2)

    def to_markdown(self) -> str:
        lines = [
            f'# Dataset validation: `{self.dataset_name}`',
            '',
            f'- **Rows**: {self.n_rows:,}  **Columns**: {self.n_cols}',
        ]
        if self.source:
            lines.append(f'- **Source**: `{self.source}`')
        status = 'PASS' if self.ok() else 'FAIL'
        lines.append(f'- **Status**: **{status}** '
                     f'({len(self.critical_failures)} critical, '
                     f'{len(self.warnings)} warnings)')
        lines.append('')

        if self.funnel:
            lines += ['## Row funnel', '',
                      'Every dropped row is attributed to the first gate it failed.',
                      '',
                      '| # | Gate | Removed | Remaining |',
                      '|---|---|---:|---:|']
            for i, step in enumerate(self.funnel, 1):
                lines.append(f"| {i} | {step['gate']} | "
                             f"{step['removed']:,} | {step['remaining']:,} |")
            lines.append('')

        if self.coverage:
            lines += ['## Coverage', '']
            for label, table in self.coverage.items():
                lines += [f'**{label}**', '']
                if isinstance(table, dict) and table.get('_rows'):
                    lines += ['| value | rows | share |', '|---|---:|---:|']
                    for row in table['_rows']:
                        lines.append(f"| {row['value']} | {row['rows']:,} | "
                                     f"{row['share'] * 100:.1f}% |")
                lines.append('')

        for severity, title in ((CRITICAL, 'Critical checks'),
                                (WARNING, 'Warnings'),
                                (INFO, 'Information')):
            group = [c for c in self.checks if c.severity == severity]
            if not group:
                continue
            lines += [f'## {title}', '', '| Check | Result | Affected | Note |',
                      '|---|---|---:|---|']
            for check in group:
                mark = 'PASS' if check.passed else '**FAIL**'
                lines.append(f"| `{check.name}` | {mark} | "
                             f"{check.n_affected:,} ({check.rate * 100:.2f}%) | "
                             f"{check.message} |")
            lines.append('')
        return '\n'.join(lines)

    def to_markdown_file(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w', encoding='utf-8') as handle:
            handle.write(self.to_markdown())


class DatasetValidationError(RuntimeError):
    """Raised when a dataset fails a critical validation check."""


# =============================================================================
# Row funnel
# =============================================================================

@dataclass
class Gate:
    """A named row filter.  ``predicate`` returns True for rows to KEEP."""
    name: str
    predicate: Callable[[pd.DataFrame], pd.Series]
    reason: str = ''

    def mask(self, df: pd.DataFrame) -> pd.Series:
        mask = self.predicate(df)
        if not isinstance(mask, pd.Series):
            mask = pd.Series(mask, index=df.index)
        # Treat NaN in the mask as "do not keep" rather than raising.
        return mask.fillna(False).astype(bool)


def row_funnel(df: pd.DataFrame, gates: Sequence[Gate],
               reason_column: str = 'drop_reason') -> tuple[pd.DataFrame, list[dict]]:
    """Apply gates in order, recording which rows each gate removed.

    Returns ``(kept_df, funnel)`` where ``funnel`` is a list of step dicts.  The
    kept frame carries ``reason_column`` set to ``None``; a variant of the input
    with per-row reasons is available via ``annotate_rows``.

    Rows are attributed to the *first* gate that rejects them, so the funnel
    columns sum to the input row count exactly.
    """
    work = df.copy()
    work[reason_column] = None
    remaining = pd.Series(True, index=work.index)
    funnel: list[dict] = []

    for gate in gates:
        eligible = remaining
        if not eligible.any():
            funnel.append({'gate': gate.name, 'removed': 0,
                           'remaining': int(remaining.sum())})
            continue
        keep = gate.mask(work.loc[eligible])
        rejects = eligible & ~keep.reindex(work.index, fill_value=True)
        work.loc[rejects, reason_column] = gate.name
        remaining &= ~rejects
        funnel.append({
            'gate': gate.name,
            'removed': int(rejects.sum()),
            'remaining': int(remaining.sum()),
            'reason': gate.reason or gate.name,
        })

    return work.loc[remaining].copy(), funnel


def annotate_rows(df: pd.DataFrame, gates: Sequence[Gate],
                  reason_column: str = 'drop_reason') -> pd.DataFrame:
    """Return the full frame with ``reason_column`` recording why each row dropped.

    Useful for auditing exactly which rows a gate removed, rather than only how
    many.
    """
    work = df.copy()
    work[reason_column] = None
    remaining = pd.Series(True, index=work.index)
    for gate in gates:
        eligible = remaining
        if not eligible.any():
            continue
        keep = gate.mask(work.loc[eligible])
        rejects = eligible & ~keep.reindex(work.index, fill_value=True)
        work.loc[rejects, reason_column] = gate.name
        remaining &= ~rejects
    return work


# =============================================================================
# Individual checks
# =============================================================================

def check_columns(df: pd.DataFrame, spec: DatasetSpec) -> list[CheckResult]:
    """Presence + dtype-independent range/allow-list checks."""
    results = []
    for name in spec.required_columns():
        present = name in df.columns
        results.append(CheckResult(
            name=f'column_present:{name}',
            severity=CRITICAL,
            passed=present,
            n_affected=0 if present else len(df),
            n_total=len(df),
            message='present' if present else 'REQUIRED COLUMN MISSING',
        ))

    for col_spec in spec.columns.values():
        if col_spec.name not in df.columns:
            continue
        series = df[col_spec.name]

        if col_spec.minimum is not None or col_spec.maximum is not None:
            numeric = pd.to_numeric(series, errors='coerce')
            violations = pd.Series(False, index=df.index)
            if col_spec.minimum is not None:
                violations |= numeric < col_spec.minimum
            if col_spec.maximum is not None:
                violations |= numeric > col_spec.maximum
            violations &= series.notna()
            results.append(CheckResult(
                name=f'range:{col_spec.name}', severity=CRITICAL,
                passed=not violations.any(),
                n_affected=int(violations.sum()), n_total=len(df),
                message=f'outside [{col_spec.minimum}, {col_spec.maximum}]',
                detail={'min': _safe_float(numeric.min()),
                        'max': _safe_float(numeric.max())},
            ))

        if col_spec.allowed is not None:
            observed = series.dropna().astype(str)
            unexpected = ~observed.isin([str(a) for a in col_spec.allowed])
            results.append(CheckResult(
                name=f'allowed:{col_spec.name}', severity=CRITICAL,
                passed=not unexpected.any(),
                n_affected=int(unexpected.sum()), n_total=len(df),
                message=f'not in {list(col_spec.allowed)}',
                detail={'observed': sorted(observed[unexpected].unique()[:10])},
            ))

        if col_spec.null_rate_max is not None:
            nulls = int(series.isna().sum())
            rate = nulls / len(df) if len(df) else 0.0
            results.append(CheckResult(
                name=f'null_rate:{col_spec.name}', severity=WARNING,
                passed=rate <= col_spec.null_rate_max,
                n_affected=nulls, n_total=len(df),
                message=f'{rate * 100:.1f}% null (limit '
                        f'{col_spec.null_rate_max * 100:.0f}%)',
            ))
    return results


def check_key_uniqueness(df: pd.DataFrame,
                         keys: Sequence[str] = ('property_id', 'date_obtained')
                         ) -> list[CheckResult]:
    """The parser emits one row per (listing, scrape date); duplicates are a bug."""
    present = [k for k in keys if k in df.columns]
    if len(present) != len(keys):
        return [CheckResult('key_uniqueness', CRITICAL, False, len(df), len(df),
                            f'missing key columns: {set(keys) - set(present)}')]
    dup = int(df.duplicated(subset=list(keys), keep=False).sum())
    return [CheckResult(
        'key_uniqueness', CRITICAL, dup == 0, dup, len(df),
        f'duplicate ({", ".join(keys)}) rows' if dup else 'unique',
    )]


def check_geo_bounds(df: pd.DataFrame, bbox: Mapping[str, float] = GHENT_BBOX
                     ) -> list[CheckResult]:
    """Coordinates should land inside the Ghent region."""
    if 'latitude' not in df.columns or 'longitude' not in df.columns:
        return [CheckResult('geo_bounds', CRITICAL, False, len(df), len(df),
                            'latitude/longitude missing')]
    lat, lon = df['latitude'], df['longitude']
    outside = lat.notna() & lon.notna() & (
        (lat < bbox['lat_min']) | (lat > bbox['lat_max']) |
        (lon < bbox['lon_min']) | (lon > bbox['lon_max']))
    total = int((lat.notna() & lon.notna()).sum())
    return [CheckResult(
        'geo_bounds', WARNING, not outside.any(), int(outside.sum()), total,
        f'outside Ghent bbox {dict(bbox)}', detail=dict(bbox),
    )]


def check_price_per_sqm(df: pd.DataFrame) -> list[CheckResult]:
    """Flag implausible EUR/m2 without dropping the rows."""
    if not {'price', 'livable_area'}.issubset(df.columns):
        return []
    price = pd.to_numeric(df['price'], errors='coerce')
    area = pd.to_numeric(df['livable_area'], errors='coerce')
    valid = (price > 0) & (area > 0)
    if not valid.any():
        return []
    per_sqm = (price[valid] / area[valid])
    low, high = per_sqm.quantile([0.001, 0.999])
    extreme = valid & ((price / area < low) | (price / area > high))
    return [CheckResult(
        'price_per_sqm_extremes', WARNING, not extreme.any(),
        int(extreme.sum()), int(valid.sum()),
        f'outside [EUR{low:,.0f}, EUR{high:,.0f}]/m2',
        detail={'p0.1': _safe_float(low), 'p99.9': _safe_float(high),
                'median': _safe_float(per_sqm.median())},
    )]


def check_cross_field_consistency(df: pd.DataFrame) -> list[CheckResult]:
    """Fields that must agree with each other for the record to make sense."""
    results = []

    # Land plot smaller than the building footprint is normal for multi-storey
    # housing -- livable_area sums every floor -- so it is not a defect and is
    # deliberately not checked.  A garden larger than the whole plot is.
    if {'garden_surface', 'property_size'}.issubset(df.columns):
        garden = pd.to_numeric(df['garden_surface'], errors='coerce')
        plot = pd.to_numeric(df['property_size'], errors='coerce')
        bad = garden.notna() & plot.notna() & (plot > 0) & (garden > plot)
        results.append(CheckResult(
            'garden_larger_than_plot', WARNING, not bad.any(),
            int(bad.sum()), int((garden.notna() & plot.notna()).sum()),
            'garden_surface > property_size',
        ))

    # More bedrooms than rooms is impossible.
    if {'number_of_bedrooms', 'number_of_rooms'}.issubset(df.columns):
        beds = pd.to_numeric(df['number_of_bedrooms'], errors='coerce')
        rooms = pd.to_numeric(df['number_of_rooms'], errors='coerce')
        bad = beds.notna() & rooms.notna() & (rooms > 0) & (beds > rooms)
        results.append(CheckResult(
            'bedrooms_exceed_rooms', WARNING, not bad.any(),
            int(bad.sum()), int((beds.notna() & rooms.notna()).sum()),
            'number_of_bedrooms > number_of_rooms',
        ))

    # Build year beyond this year is legitimate for new-build listings that carry
    # an expected completion year, so this is a warning rather than a defect.
    if 'year_built' in df.columns:
        year = pd.to_numeric(df['year_built'], errors='coerce')
        this_year = datetime.now().year
        bad = year.notna() & (year > this_year)
        results.append(CheckResult(
            'year_built_in_future', WARNING, not bad.any(),
            int(bad.sum()), int(year.notna().sum()),
            f'year_built > {this_year} (expected completion year?)',
        ))
    return results


def check_dates(df: pd.DataFrame) -> list[CheckResult]:
    """Scrape dates must be parseable and not in the future."""
    results = []
    if 'date_obtained' in df.columns:
        parsed = pd.to_datetime(df['date_obtained'], errors='coerce')
        if parsed.isna().all():
            # DDMMYYYY strings need explicit parsing.
            parsed = pd.to_datetime(
                df['date_obtained'].astype(str).str.zfill(8),
                format='%d%m%Y', errors='coerce')
        unparseable = int(parsed.isna().sum())
        results.append(CheckResult(
            'date_parseable', CRITICAL, unparseable == 0, unparseable, len(df),
            'unparseable date_obtained',
        ))
        future = int((parsed > pd.Timestamp.now().normalize()).sum())
        results.append(CheckResult(
            'date_not_future', CRITICAL, future == 0, future, len(df),
            'date_obtained in the future',
        ))
    return results


def check_fabricated_flags(df: pd.DataFrame) -> list[CheckResult]:
    """Guard the defect fixed in the parser: a flag that is never True or never
    False while its partner surface is populated usually means fabrication."""
    results = []
    for column in df.columns:
        if not column.startswith('has_'):
            continue
        series = df[column]
        if series.dtype != bool:
            continue
        n_false = int((~series).sum())
        n_true = int(series.sum())
        # A flag with exactly one observed class carries no information; if the
        # companion *_missing indicator is absent, that class was invented.
        degenerate = (n_true == 0 or n_false == 0)
        has_indicator = f'{column}_missing' in df.columns
        results.append(CheckResult(
            f'flag_not_degenerate:{column}',
            INFO if has_indicator else WARNING,
            not (degenerate and not has_indicator),
            n_true if degenerate else 0, len(df),
            ('only one class observed and no _missing indicator'
             if degenerate and not has_indicator
             else f'{n_true} True / {n_false} False'),
        ))
    return results


def check_leaky_columns(df: pd.DataFrame,
                        forbidden: Iterable[str] = ('price_sq', 'log_price',
                                                     'spatial_lag_10',
                                                     'avg_dist_5nn')
                        ) -> list[CheckResult]:
    """Target-derived or cross-row-derivable columns must not be features."""
    forbidden = set(forbidden)
    present = sorted(forbidden & set(df.columns))
    if not present:
        return [CheckResult('no_leaky_columns', CRITICAL, True, 0, len(df),
                            'none present')]
    return [CheckResult(
        'no_leaky_columns', CRITICAL, False, len(present), len(df),
        f'target-derived columns present: {present}. These must be computed '
        f'fold-wise or excluded before training.',
        detail={'columns': present},
    )]


# =============================================================================
# Coverage
# =============================================================================

def coverage_tables(df: pd.DataFrame) -> dict:
    """Distribution of the slicing variables, so blind spots are visible."""
    out = {}
    for column in ('property_type', 'property_subtype', 'neighbourhood',
                   'postal_code', 'energy_label', 'building_condition', 'city'):
        if column not in df.columns:
            continue
        counts = df[column].astype(str).value_counts(dropna=False)
        total = len(df)
        out[column] = {
            'distinct': int(counts.size),
            '_rows': [{'value': str(idx), 'rows': int(value),
                       'share': round(value / total, 4)}
                      for idx, value in counts.head(15).items()],
        }
    if 'date_obtained' in df.columns:
        dates = pd.to_datetime(df['date_obtained'], errors='coerce')
        if dates.isna().all():
            dates = pd.to_datetime(
                df['date_obtained'].astype(str).str.zfill(8),
                format='%d%m%Y', errors='coerce')
        periods = dates.dt.to_period('M').astype(str).value_counts().sort_index()
        total = len(df)
        out['period'] = {
            'distinct': int(periods.size),
            '_rows': [{'value': str(idx), 'rows': int(value),
                       'share': round(value / total, 4)}
                      for idx, value in periods.items()],
        }
    return out


# =============================================================================
# Combined validation
# =============================================================================

def validate(df: pd.DataFrame, spec: DatasetSpec, *,
             source: str | None = None,
             check_leakage: bool = False,
             coverage: bool = True) -> ValidationReport:
    """Run the full check suite and return a report."""
    report = ValidationReport(spec.name, len(df), len(df.columns), source=source)

    for result in check_columns(df, spec):
        report.add(result)
    for result in check_key_uniqueness(df):
        report.add(result)
    for result in check_dates(df):
        report.add(result)
    for result in check_geo_bounds(df):
        report.add(result)
    for result in check_price_per_sqm(df):
        report.add(result)
    for result in check_cross_field_consistency(df):
        report.add(result)
    for result in check_fabricated_flags(df):
        report.add(result)
    if check_leakage:
        for result in check_leaky_columns(df):
            report.add(result)
    if coverage:
        report.coverage = coverage_tables(df)
    return report


def _safe_float(value) -> float | None:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
