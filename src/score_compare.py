"""
Compare scoring with vs without today's data in training.
Trains on ALL data, scores today's rows, compares to holdout results.
"""

import pandas as pd
import numpy as np
import os
import warnings
import urllib.request
import csv
import io

from autogluon.tabular import TabularPredictor

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs', 'models')
TODAY = '2026-02-23'


def _fetch_ecb_series(url, value_col_name):
    """Fetch a single monthly time series from ECB SDW API."""
    req = urllib.request.Request(url, headers={'Accept': 'text/csv'})
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = resp.read().decode()
    reader = csv.DictReader(io.StringIO(data))
    result = {}
    for row in reader:
        result[row['TIME_PERIOD']] = float(row['OBS_VALUE'])
    return pd.DataFrame([
        {'year_month': pd.Timestamp(k), value_col_name: v}
        for k, v in result.items()
    ])


def fetch_macro_indicators():
    """Fetch monthly macro indicators from ECB: CPI index, Belgian 10Y yield, Euribor 12m."""
    indicators = {
        'belgian_cpi_index': 'https://data-api.ecb.europa.eu/service/data/ICP/M.BE.N.000000.4.INX?startPeriod=2024-01&endPeriod=2026-12&format=csvdata',
        'belgian_10y_yield': 'https://data-api.ecb.europa.eu/service/data/IRS/M.BE.L.L40.CI.0000.EUR.N.Z?startPeriod=2024-01&endPeriod=2026-12&format=csvdata',
        'euribor_12m': 'https://data-api.ecb.europa.eu/service/data/FM/M.U2.EUR.RT.MM.EURIBOR1YD_.HSTA?startPeriod=2024-01&endPeriod=2026-12&format=csvdata',
    }
    merged = None
    for col_name, url in indicators.items():
        try:
            series = _fetch_ecb_series(url, col_name)
            if merged is None:
                merged = series
            else:
                merged = merged.merge(series, on='year_month', how='outer')
            print(f"  Fetched {len(series)} months of {col_name}")
        except Exception as e:
            print(f"  Could not fetch {col_name}: {e}")
    if merged is not None:
        merged = merged.sort_values('year_month')
        merged = merged.ffill()
    return merged


def prepare_features(df, rates):
    df = df.copy()
    df['date_obtained'] = pd.to_datetime(df['date_obtained'])

    if rates is not None:
        df['year_month'] = df['date_obtained'].dt.to_period('M').dt.to_timestamp()
        df = df.merge(rates, on='year_month', how='left')
        df.drop(columns=['year_month'], inplace=True)

    drop_cols = [
        'property_id', 'date_obtained', 'street', 'house_number', 'box',
        'city', 'immo_office_name', 'log_price', 'price_sq',
    ]
    drop_cols = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=drop_cols)

    bool_cols = df.select_dtypes(include=['bool']).columns.tolist()
    for col in bool_cols:
        df[col] = df[col].astype(int)

    return df


def main():
    print("=" * 80)
    print("COMPARING: HOLDOUT vs FULL-DATA TRAINING")
    print("=" * 80)

    # --- Load data ---
    emb_path = os.path.join(BASE_DIR, 'outputs', 'property_data_with_embeddings.csv')
    df_all = pd.read_csv(emb_path)
    df_all['date_obtained'] = pd.to_datetime(df_all['date_obtained'])
    print(f"Total rows: {len(df_all):,}")

    mask_today = df_all['date_obtained'] == TODAY
    df_today = df_all[mask_today].copy()
    print(f"Today's rows: {len(df_today)}")

    # --- Fetch Euribor ---
    print("\nFetching macro indicators from ECB...")
    rates = fetch_macro_indicators()

    # --- Prepare features ---
    today_meta = df_today[['property_id', 'price', 'date_obtained']].copy()
    for col in ['postal_code', 'livable_area', 'property_size', 'number_of_bedrooms',
                'building_condition', 'energy_label', 'neighbourhood',
                'property_type_HOUSE', 'has_meaningful_description']:
        if col in df_today.columns:
            today_meta[col] = df_today[col].values

    all_features = prepare_features(df_all, rates)
    today_features = prepare_features(df_today, rates)

    # --- Train on ALL data (including today) ---
    print("\n" + "=" * 80)
    print("TRAINING ON ALL DATA (including today)")
    print("=" * 80)

    model_path = os.path.join(OUTPUT_DIR, 'autogluon_full')

    predictor = TabularPredictor(
        label='price',
        path=model_path,
        eval_metric='root_mean_squared_error',
        problem_type='regression',
    )

    predictor.fit(
        train_data=all_features,
        time_limit=600,
        presets='best_quality',
        verbosity=2,
    )

    # --- Score today with full model ---
    predictions_full = predictor.predict(today_features)

    results = today_meta.copy()
    results['pred_full_k'] = predictions_full.values
    results['asking_eur'] = results['price'] * 1000
    results['pred_full_eur'] = results['pred_full_k'] * 1000
    results['delta_full_eur'] = results['pred_full_eur'] - results['asking_eur']
    results['delta_full_pct'] = results['delta_full_eur'] / results['asking_eur'] * 100

    # --- Load holdout results ---
    holdout_path = os.path.join(BASE_DIR, 'outputs', 'today_deals.csv')
    holdout = pd.read_csv(holdout_path)
    holdout_map = holdout.set_index('property_id')[['predicted_eur', 'delta_eur', 'delta_pct']].rename(
        columns={'predicted_eur': 'pred_holdout_eur', 'delta_eur': 'delta_holdout_eur', 'delta_pct': 'delta_holdout_pct'}
    )
    results = results.merge(holdout_map, left_on='property_id', right_index=True, how='left')

    results['shift_eur'] = results['pred_full_eur'] - results['pred_holdout_eur']
    results['shift_pct'] = results['shift_eur'] / results['asking_eur'] * 100

    # --- Summary ---
    print("\n" + "=" * 80)
    print("COMPARISON: HOLDOUT vs FULL MODEL on today's 853 properties")
    print("=" * 80)

    print(f"\n{'Metric':<35} {'Holdout':>12} {'Full':>12} {'Shift':>12}")
    print("-" * 75)
    print(f"{'Mean delta (EUR)':<35} {results['delta_holdout_eur'].mean():>+12,.0f} {results['delta_full_eur'].mean():>+12,.0f} {results['shift_eur'].mean():>+12,.0f}")
    print(f"{'Median delta (EUR)':<35} {results['delta_holdout_eur'].median():>+12,.0f} {results['delta_full_eur'].median():>+12,.0f} {results['shift_eur'].median():>+12,.0f}")
    print(f"{'Mean delta (%)':<35} {results['delta_holdout_pct'].mean():>+11.1f}% {results['delta_full_pct'].mean():>+11.1f}% {results['shift_pct'].mean():>+11.1f}%")
    print(f"{'Median delta (%)':<35} {results['delta_holdout_pct'].median():>+11.1f}% {results['delta_full_pct'].median():>+11.1f}% {results['shift_pct'].median():>+11.1f}%")

    n_under_h = (results['delta_holdout_eur'] > 0).sum()
    n_under_f = (results['delta_full_eur'] > 0).sum()
    print(f"{'Underpriced count':<35} {n_under_h:>12} {n_under_f:>12} {n_under_f - n_under_h:>+12}")
    print(f"{'Overpriced count':<35} {len(results) - n_under_h:>12} {len(results) - n_under_f:>12}")

    print(f"\nShift statistics (full_pred - holdout_pred):")
    print(f"  Mean:   {results['shift_eur'].mean():>+10,.0f} EUR ({results['shift_pct'].mean():>+.2f}%)")
    print(f"  Median: {results['shift_eur'].median():>+10,.0f} EUR ({results['shift_pct'].median():>+.2f}%)")
    print(f"  Std:    {results['shift_eur'].std():>10,.0f} EUR")
    print(f"  Min:    {results['shift_eur'].min():>+10,.0f} EUR")
    print(f"  Max:    {results['shift_eur'].max():>+10,.0f} EUR")

    # Correlation of rankings
    from scipy.stats import spearmanr
    rho, p = spearmanr(results['delta_holdout_pct'], results['delta_full_pct'])
    print(f"\n  Rank correlation (Spearman): {rho:.4f} (p={p:.2e})")

    # Biggest movers
    results_sorted = results.sort_values('shift_eur', ascending=False)
    print(f"\nTop 5 properties where full model predicts HIGHER:")
    for _, row in results_sorted.head(5).iterrows():
        print(f"  {int(row['property_id'])}: holdout {row['pred_holdout_eur']:>10,.0f} -> full {row['pred_full_eur']:>10,.0f} ({row['shift_eur']:>+10,.0f} EUR)")

    print(f"\nTop 5 properties where full model predicts LOWER:")
    for _, row in results_sorted.tail(5).iterrows():
        print(f"  {int(row['property_id'])}: holdout {row['pred_holdout_eur']:>10,.0f} -> full {row['pred_full_eur']:>10,.0f} ({row['shift_eur']:>+10,.0f} EUR)")

    # By property type
    if 'property_type_HOUSE' in results.columns:
        results['type'] = results['property_type_HOUSE'].map({True: 'House', False: 'Apt'})
        print(f"\nShift by type:")
        for t in ['House', 'Apt']:
            sub = results[results['type'] == t]
            print(f"  {t:>5}s: mean shift {sub['shift_eur'].mean():>+10,.0f} EUR ({sub['shift_pct'].mean():>+.2f}%)")

    # Save
    out_path = os.path.join(BASE_DIR, 'outputs', 'today_deals_comparison.csv')
    results.to_csv(out_path, index=False)
    print(f"\nFull comparison saved to {out_path}")


if __name__ == '__main__':
    main()
