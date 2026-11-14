"""
Score today's scraped properties against a model trained on all OTHER data.
Ensures clean holdout — model never sees today's rows during training.
Also runs sanity checks on the model.
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
        # Forward-fill for months where latest data isn't published yet
        merged = merged.ffill()
    return merged


def prepare_features(df, rates):
    """Apply same feature engineering as train_models.py prepare_data()."""
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
    print("SCORING TODAY'S PROPERTIES (CLEAN HOLDOUT)")
    print("=" * 80)

    # --- Load full embeddings data ---
    emb_path = os.path.join(BASE_DIR, 'outputs', 'property_data_with_embeddings.csv')
    df_all = pd.read_csv(emb_path)
    df_all['date_obtained'] = pd.to_datetime(df_all['date_obtained'])
    print(f"Total rows: {len(df_all):,}")

    # --- Split: today vs rest ---
    mask_today = df_all['date_obtained'] == TODAY
    df_today = df_all[mask_today].copy()
    df_train = df_all[~mask_today].copy()
    print(f"Today's rows ({TODAY}): {len(df_today)}")
    print(f"Training rows (all other dates): {len(df_train):,}")

    if len(df_today) == 0:
        print("No data from today found! Exiting.")
        return

    # --- Also check raw scrape counts ---
    parsed_path = os.path.join(BASE_DIR, 'outputs', 'property_data_parsed.csv')
    df_parsed = pd.read_csv(parsed_path, usecols=['property_id', 'date_obtained'], low_memory=False)
    df_parsed['date_str'] = df_parsed['date_obtained'].astype(str).str.zfill(8)
    df_parsed['date_parsed'] = pd.to_datetime(df_parsed['date_str'], format='%d%m%Y', errors='coerce')
    today_parsed = df_parsed[df_parsed['date_parsed'] == TODAY]
    print(f"Today in parsed CSV: {len(today_parsed)} rows (before preprocessing filters)")
    print(f"Today in preprocessed/embeddings: {len(df_today)} rows (after filters)")

    # --- Fetch Euribor ---
    print("\nFetching macro indicators from ECB...")
    rates = fetch_macro_indicators()

    # --- Prepare features ---
    print("\nPreparing features...")
    # Save today's metadata before dropping
    today_meta = df_today[['property_id', 'price', 'date_obtained']].copy()
    for col in ['postal_code', 'livable_area', 'property_size', 'number_of_bedrooms',
                'building_condition', 'energy_label', 'year_built', 'neighbourhood',
                'property_type_HOUSE', 'has_meaningful_description']:
        if col in df_today.columns:
            today_meta[col] = df_today[col].values

    train_features = prepare_features(df_train, rates)
    today_features = prepare_features(df_today, rates)

    # --- Train model on non-today data ---
    print("\n" + "=" * 80)
    print("TRAINING MODEL (excluding today's data)")
    print("=" * 80)

    model_path = os.path.join(OUTPUT_DIR, 'autogluon_holdout')

    predictor = TabularPredictor(
        label='price',
        path=model_path,
        eval_metric='root_mean_squared_error',
        problem_type='regression',
    )

    predictor.fit(
        train_data=train_features,
        time_limit=600,
        presets='best_quality',
        verbosity=2,
    )

    # --- Score today's properties ---
    print("\n" + "=" * 80)
    print("SCORING TODAY'S PROPERTIES")
    print("=" * 80)

    predictions = predictor.predict(today_features)

    results = today_meta.copy()
    results['predicted_k'] = predictions.values
    results['asking_eur'] = results['price'] * 1000
    results['predicted_eur'] = results['predicted_k'] * 1000
    results['delta_eur'] = results['predicted_eur'] - results['asking_eur']
    results['delta_pct'] = results['delta_eur'] / results['asking_eur'] * 100

    if 'property_type_HOUSE' in results.columns:
        results['type'] = results['property_type_HOUSE'].map({1: 'House', 0: 'Apt'})

    results = results.sort_values('delta_pct', ascending=False)

    # --- Display rankings ---
    print(f"\n{'=' * 100}")
    print(f"TODAY'S PROPERTIES RANKED (predicted - asking)")
    print(f"  Positive = model thinks it's worth MORE than asking (potential deal)")
    print(f"  Negative = model thinks it's overpriced")
    print(f"{'=' * 100}\n")

    cols = ['Rank', 'ID', 'Type', 'Asking', 'Predicted', 'Delta', 'Delta%',
            'Area', 'Beds', 'Neighbourhood', 'Condition']
    header = f"{'Rank':<5} {'ID':<12} {'Type':<6} {'Asking':>10} {'Predicted':>10} " \
             f"{'Delta':>10} {'D%':>7} {'Area':>6} {'Beds':>5} {'Neighbourhood':<20} {'Condition':<15}"
    print(header)
    print("-" * len(header))

    for i, (_, row) in enumerate(results.iterrows()):
        ptype = row.get('type', '?')
        area = row.get('livable_area', 0)
        beds = row.get('number_of_bedrooms', 0)
        nbh = str(row.get('neighbourhood', ''))[:19]
        cond = str(row.get('building_condition', ''))[:14]
        area_str = f"{area:.0f}m" if pd.notna(area) else "?"
        beds_str = f"{beds:.0f}" if pd.notna(beds) else "?"
        print(f"{i+1:<5} {int(row['property_id']):<12} {ptype:<6} "
              f"{row['asking_eur']:>10,.0f} {row['predicted_eur']:>10,.0f} "
              f"{row['delta_eur']:>+10,.0f} {row['delta_pct']:>+6.1f}% "
              f"{area_str:>5} {beds_str:>5} {nbh:<20} {cond:<15}")

    # --- Summary ---
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(f"Total scored:     {len(results)}")
    n_under = (results['delta_eur'] > 0).sum()
    n_over = (results['delta_eur'] < 0).sum()
    print(f"Underpriced:      {n_under} ({n_under/len(results)*100:.0f}%)")
    print(f"Overpriced:       {n_over} ({n_over/len(results)*100:.0f}%)")
    print(f"Mean delta:       {results['delta_eur'].mean():+,.0f} EUR ({results['delta_pct'].mean():+.1f}%)")
    print(f"Median delta:     {results['delta_eur'].median():+,.0f} EUR ({results['delta_pct'].median():+.1f}%)")

    if 'type' in results.columns:
        print(f"\nBy type:")
        for t in ['House', 'Apt']:
            sub = results[results['type'] == t]
            if len(sub) > 0:
                print(f"  {t:>5}s: n={len(sub)}, mean delta {sub['delta_pct'].mean():+.1f}%, "
                      f"underpriced {(sub['delta_eur'] > 0).sum()}/{len(sub)}")

    # --- Immoweb links for top 10 deals ---
    print(f"\nTop 10 potential deals (immoweb links):")
    for i, (_, row) in enumerate(results.head(10).iterrows()):
        print(f"  {i+1}. https://www.immoweb.be/en/classified/{int(row['property_id'])} "
              f"({row['delta_pct']:+.1f}%, asking {row['asking_eur']:,.0f})")

    # --- Save full results ---
    results_path = os.path.join(BASE_DIR, 'outputs', 'today_deals.csv')
    results.to_csv(results_path, index=False)
    print(f"\nFull results saved to {results_path}")

    # =====================================================================
    # SANITY CHECKS
    # =====================================================================
    print(f"\n\n{'=' * 80}")
    print("MODEL SANITY CHECKS")
    print(f"{'=' * 80}")
    print("Testing: does the model respond sensibly to feature changes?\n")

    # Use median property as baseline
    baseline = today_features.iloc[:1].copy()  # take first row as template

    def predict_one(row):
        return predictor.predict(row).values[0] * 1000  # return EUR

    base_price = predict_one(baseline)
    print(f"Baseline property prediction: {base_price:,.0f} EUR\n")

    checks = []

    # 1. Livable area: +20 sqm should increase price
    if 'livable_area' in baseline.columns:
        modified = baseline.copy()
        original_area = modified['livable_area'].values[0]
        modified['livable_area'] = original_area + 20
        new_price = predict_one(modified)
        delta = new_price - base_price
        ok = delta > 0
        checks.append(('livable_area +20m²', delta, ok))
        print(f"  livable_area {original_area:.0f} -> {original_area+20:.0f} m²: "
              f"{delta:+,.0f} EUR {'OK' if ok else 'UNEXPECTED'}")

    # 2. Garden: set has_garden=1 should increase or stay
    if 'has_garden' in baseline.columns:
        modified = baseline.copy()
        modified['has_garden'] = 1
        if 'garden_surface' in modified.columns:
            modified['garden_surface'] = 100
        new_price = predict_one(modified)
        delta = new_price - base_price
        ok = delta >= 0
        checks.append(('has_garden=1, 100m²', delta, ok))
        print(f"  has_garden=1, garden_surface=100: {delta:+,.0f} EUR {'OK' if ok else 'UNEXPECTED'}")

    # 3. Garden surface: increase
    if 'garden_surface' in baseline.columns and 'has_garden' in baseline.columns:
        base_garden = baseline.copy()
        base_garden['has_garden'] = 1
        base_garden['garden_surface'] = 50
        p1 = predict_one(base_garden)
        base_garden['garden_surface'] = 200
        p2 = predict_one(base_garden)
        delta = p2 - p1
        ok = delta > 0
        checks.append(('garden 50m² -> 200m²', delta, ok))
        print(f"  garden_surface 50 -> 200 m²: {delta:+,.0f} EUR {'OK' if ok else 'UNEXPECTED'}")

    # 4. Terrace: set has_terrace=1
    if 'has_terrace' in baseline.columns:
        modified = baseline.copy()
        modified['has_terrace'] = 1
        if 'terrace_surface' in modified.columns:
            modified['terrace_surface'] = 20
        new_price = predict_one(modified)
        delta = new_price - base_price
        ok = delta >= 0
        checks.append(('has_terrace=1, 20m²', delta, ok))
        print(f"  has_terrace=1, terrace_surface=20: {delta:+,.0f} EUR {'OK' if ok else 'UNEXPECTED'}")

    # 5. Bedrooms: +1 should increase
    if 'number_of_bedrooms' in baseline.columns:
        modified = baseline.copy()
        original_beds = modified['number_of_bedrooms'].values[0]
        modified['number_of_bedrooms'] = original_beds + 1
        new_price = predict_one(modified)
        delta = new_price - base_price
        ok = delta > 0
        checks.append(('+1 bedroom', delta, ok))
        print(f"  bedrooms {original_beds:.0f} -> {original_beds+1:.0f}: "
              f"{delta:+,.0f} EUR {'OK' if ok else 'UNEXPECTED'}")

    # 6. Swimming pool should increase
    if 'has_swimming_pool' in baseline.columns:
        modified = baseline.copy()
        modified['has_swimming_pool'] = 1
        new_price = predict_one(modified)
        delta = new_price - base_price
        ok = delta > 0
        checks.append(('has_swimming_pool=1', delta, ok))
        print(f"  has_swimming_pool=1: {delta:+,.0f} EUR {'OK' if ok else 'UNEXPECTED'}")

    # 7. Cadastral income increase (proxy for value)
    if 'cadastral_income' in baseline.columns:
        modified = baseline.copy()
        original_ci = modified['cadastral_income'].values[0]
        modified['cadastral_income'] = original_ci * 1.5
        new_price = predict_one(modified)
        delta = new_price - base_price
        ok = delta > 0
        checks.append(('cadastral_income +50%', delta, ok))
        print(f"  cadastral_income {original_ci:.0f} -> {original_ci*1.5:.0f}: "
              f"{delta:+,.0f} EUR {'OK' if ok else 'UNEXPECTED'}")

    # 8. Property size increase
    if 'property_size' in baseline.columns:
        modified = baseline.copy()
        original_ps = modified['property_size'].values[0]
        if pd.notna(original_ps) and original_ps > 0:
            modified['property_size'] = original_ps * 2
            new_price = predict_one(modified)
            delta = new_price - base_price
            ok = delta > 0
            checks.append(('property_size x2', delta, ok))
            print(f"  property_size {original_ps:.0f} -> {original_ps*2:.0f}: "
                  f"{delta:+,.0f} EUR {'OK' if ok else 'UNEXPECTED'}")

    # 9. Photovoltaic panels
    if 'has_photovoltaic_panels' in baseline.columns:
        modified = baseline.copy()
        modified['has_photovoltaic_panels'] = 1
        new_price = predict_one(modified)
        delta = new_price - base_price
        ok = delta >= 0
        checks.append(('has_photovoltaic_panels=1', delta, ok))
        print(f"  has_photovoltaic_panels=1: {delta:+,.0f} EUR {'OK' if ok else 'UNEXPECTED'}")

    # Summary of sanity checks
    passed = sum(1 for _, _, ok in checks if ok)
    total = len(checks)
    print(f"\nSanity check summary: {passed}/{total} passed")
    if passed < total:
        print("Failed checks:")
        for name, delta, ok in checks:
            if not ok:
                print(f"  - {name}: {delta:+,.0f} EUR")

    print(f"\n{'=' * 80}")
    print("DONE")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    main()
