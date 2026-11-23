"""Entry point: raw HTML-derived table -> validated training dataset.

    python src/build_dataset.py --input outputs/property_data_raw.csv

This is the stage the design docs referenced as ``run_preprocessing.py`` but
which never existed, so preprocessing was only ever run from notebooks.

Outputs (under ``outputs/dataset/`` by default):
    training_dataset.parquet     typed matrix, ready for training
    training_dataset.csv         same data, for tools without Arrow
    preprocessing_artifacts.joblib   fitted transform, reused at inference
    validation_report.json       machine-readable gate results
    validation_report.md         human-readable summary + row funnel
    config.json                  knobs, feature list, funnel
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessing import PreprocessConfig, build_dataset  # noqa: E402

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read_input(path: str) -> pd.DataFrame:
    if path.endswith('.parquet'):
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def _default_input_path() -> str:
    full_parquet = os.path.join(BASE_DIR, 'outputs/property_data_full.parquet')
    if os.path.exists(full_parquet):
        return full_parquet
    return os.path.join(BASE_DIR, 'outputs/property_data_raw.csv')


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Build a validated training dataset from parser output.')
    parser.add_argument('--input',
                        default=_default_input_path(),
                        help='parser output (CSV or Parquet)')
    parser.add_argument('--output-dir',
                        default=os.path.join(BASE_DIR, 'outputs/dataset'),
                        help='where to write the dataset and reports')
    parser.add_argument('--folds', type=int, default=5,
                        help='number of folds (0 disables fold assignment)')
    parser.add_argument('--fold-strategy', default='grouped',
                        choices=('grouped', 'temporal', 'random'),
                        help='grouped prevents the same property appearing in two folds')
    parser.add_argument('--keep-observation', default='first',
                        choices=('first', 'last'),
                        help='which scrape snapshot represents a listing')
    parser.add_argument('--outlier-quantile', type=float, default=0.99,
                        help='trim prices above this quantile (0 disables)')
    parser.add_argument('--with-cross-row-features', action='store_true',
                        help='include spatial lag features (leaky unless computed '
                             'fold-wise; off by default)')
    parser.add_argument('--allow-leaky', action='store_true',
                        help='do not fail when target-derived columns are present')
    parser.add_argument('--keep-commercial', action='store_true',
                        help='do not drop commercial/mixed-use listings')
    parser.add_argument('--strict', action='store_true',
                        help='exit non-zero if any critical validation check fails')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f'ERROR: input not found: {args.input}', file=sys.stderr)
        return 2

    print(f'Loading {args.input} ...')
    raw = _read_input(args.input)
    print(f'  {len(raw):,} rows x {len(raw.columns)} columns')

    config = PreprocessConfig(
        keep_observation=args.keep_observation,
        outlier_quantile=args.outlier_quantile,
        drop_commercial=not args.keep_commercial,
        forbid_leaky_features=not args.allow_leaky,
    )

    result = build_dataset(
        raw,
        config,
        fold_strategy=args.fold_strategy if args.folds else None,
        n_folds=max(args.folds, 2),
        with_cross_row_features=args.with_cross_row_features,
        output_dir=args.output_dir,
    )

    report = result['report']
    matrix = result['train_ready']

    print()
    print('Row funnel')
    print('----------')
    for i, step in enumerate(result['funnel'], 1):
        print(f"  {i}. {step['gate']:<34} -{step['removed']:>7,}  "
              f"-> {step['remaining']:>7,}")
    print(f"  {'INPUT':<34}  {result_note(len(raw))}")

    if result['folds'] is not None:
        folds = result['folds']
        print()
        print('Folds')
        print('-----')
        print(folds.to_string(index=False))
        leaked = folds.attrs.get('leaked_groups', 0)
        total = folds.attrs.get('total_groups', 0)
        status = 'OK' if leaked == 0 else 'LEAK'
        print(f"  groups spanning >1 fold: {leaked}/{total}  [{status}]")

    print()
    print(f"Training matrix: {matrix.shape[0]:,} rows x {matrix.shape[1]} columns")
    print(f"Validation: {'PASS' if report.ok() else 'FAIL'} "
          f"({len(report.critical_failures)} critical, "
          f"{len(report.warnings)} warnings)")
    for check in report.critical_failures[:10]:
        print(f"  CRITICAL {check.name}: {check.message} ({check.n_affected:,})")
    for check in report.warnings[:10]:
        print(f"  warning  {check.name}: {check.message} ({check.n_affected:,})")

    print()
    print(f"Wrote dataset + reports to {args.output_dir}")

    if args.strict and not report.ok():
        return 1
    return 0


def result_note(n: int) -> str:
    return f'{n:>7,} rows in'


if __name__ == '__main__':
    raise SystemExit(main())
