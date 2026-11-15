"""
AutoGluon-based model training for Ghent real estate price prediction.
Uses TabularPredictor with all available model backends.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:  # seaborn is only used for plot colour palettes
    import seaborn as sns
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    sns = None


def _palette(name, n):
    """Colour palette that works with or without seaborn installed."""
    if sns is not None:
        return sns.color_palette(name, n)
    try:
        return plt.get_cmap(name).resampled(max(n, 1))(range(max(n, 1)))
    except ValueError:
        cmap = plt.get_cmap('viridis')
        return cmap([i / max(n - 1, 1) for i in range(max(n, 1))])
import os
import warnings
import urllib.request
import csv
import io
from datetime import datetime

from autogluon.tabular import TabularPredictor

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs', 'models')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================================
# DATA PREPARATION
# ============================================================================

def _fetch_ecb_series(url, value_col_name):
    """Fetch a single monthly time series from ECB SDW API."""
    req = urllib.request.Request(url, headers={'Accept': 'text/csv'})
    with urllib.request.urlopen(req, timeout=30) as resp:
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


def prepare_data(filepath):
    """Load and prepare data for AutoGluon."""
    print("=" * 70)
    print("DATA PREPARATION")
    print("=" * 70)

    df = pd.read_csv(filepath)
    df['date_obtained'] = pd.to_datetime(df['date_obtained'])
    print(f"Loaded {len(df):,} rows x {len(df.columns)} columns")

    # --- Add macro indicators (CPI index, Belgian 10Y yield, Euribor 12m) ---
    print("\nFetching macro indicators from ECB...")
    macro = fetch_macro_indicators()
    if macro is not None:
        df['year_month'] = df['date_obtained'].dt.to_period('M').dt.to_timestamp()
        df = df.merge(macro, on='year_month', how='left')
        df.drop(columns=['year_month'], inplace=True)
        for col in ['belgian_cpi_index', 'belgian_10y_yield', 'euribor_12m']:
            if col in df.columns:
                matched = df[col].notna().sum()
                print(f"  {col}: matched {matched:,} rows ({matched/len(df)*100:.1f}%)")

    # --- Drop columns that shouldn't be features ---
    drop_cols = [
        'property_id',          # identifier, not a feature
        'date_obtained',        # temporal leak (use euribor instead)
        'street',               # too many unique values, noisy
        'house_number',         # same
        'box',                  # mostly missing, not useful
        'city',                 # redundant with neighbourhood/postal_code
        'immo_office_name',     # too many categories, leaky
        'log_price',            # derived from target
        'price_sq',             # derived from target
    ]
    drop_cols = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=drop_cols)
    print(f"Dropped {len(drop_cols)} non-feature columns")

    # --- Convert booleans to int for cleaner handling ---
    bool_cols = df.select_dtypes(include=['bool']).columns.tolist()
    for col in bool_cols:
        df[col] = df[col].astype(int)
    print(f"Converted {len(bool_cols)} boolean columns to int")

    # --- Feature summary ---
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    print(f"\nFeature summary:")
    print(f"  Numeric features: {len(numeric_cols) - 1}")  # minus target
    print(f"  Categorical features: {len(cat_cols)}")
    print(f"  Total features: {len(df.columns) - 1}")  # minus target

    return df


# ============================================================================
# TRAINING
# ============================================================================

def train_autogluon(df, target='price', time_limit=600, preset='best_quality',
                    holdout_fold=None):
    """Train AutoGluon TabularPredictor.

    Honours a precomputed ``fold`` column when present.  The previous
    implementation called ``train_test_split(test_size=0.2, random_state=42)``
    with no grouping, which put the same physical property (same address and
    area) on both sides for 70% of test rows and inflated reported accuracy by
    roughly 0.07 R^2.  When ``fold`` exists, one whole fold is held out instead,
    and because folds are built from the property group key a property cannot
    appear on both sides.
    """
    print("\n" + "=" * 70)
    print(f"AUTOGLUON TRAINING (preset={preset}, time_limit={time_limit}s)")
    print("=" * 70)

    if 'fold' in df.columns:
        folds = sorted(df['fold'].unique())
        holdout = holdout_fold if holdout_fold in folds else folds[-1]
        train_df = df[df['fold'] != holdout].drop(columns=['fold'])
        test_df = df[df['fold'] == holdout].drop(columns=['fold'])
        print(f"Holding out fold {holdout} ({len(test_df):,} rows); "
              f"train {len(train_df):,} rows.")
        if 'property_key' in train_df.columns and 'property_key' in test_df.columns:
            overlap = set(train_df['property_key']) & set(test_df['property_key'])
            print(f"  property overlap across the split: {len(overlap)} "
                  f"({'OK' if not overlap else 'LEAKAGE'})")
    else:
        # Refuse to silently reproduce the leaky default.
        raise ValueError(
            "No 'fold' column found. Build the dataset with "
            "`python src/build_dataset.py` so folds are grouped by property; "
            "a random row-wise split leaks duplicate properties across the "
            "train/test boundary.")

    print(f"Train: {len(train_df):,} rows, Test: {len(test_df):,} rows")

    # Save test set for later evaluation
    test_df.to_csv(os.path.join(OUTPUT_DIR, 'test_set.csv'), index=False)

    # Train
    predictor = TabularPredictor(
        label=target,
        path=os.path.join(OUTPUT_DIR, 'autogluon'),
        eval_metric='root_mean_squared_error',
        problem_type='regression',
    )

    predictor.fit(
        train_data=train_df,
        time_limit=time_limit,
        presets=preset,
        verbosity=2,
    )

    return predictor, train_df, test_df


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate(predictor, test_df, target='price'):
    """Comprehensive evaluation of AutoGluon predictor."""
    print("\n" + "=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)

    # --- Leaderboard ---
    leaderboard = predictor.leaderboard(test_df, silent=True)
    print("\nModel Leaderboard:")
    print(leaderboard.to_string(index=False))
    leaderboard.to_csv(os.path.join(OUTPUT_DIR, 'leaderboard.csv'), index=False)

    # --- Best model metrics ---
    y_pred = predictor.predict(test_df)
    y_true = test_df[target]

    # Prices are in K€, convert to EUR for interpretable metrics
    y_true_eur = y_true * 1000
    y_pred_eur = y_pred * 1000

    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, \
        mean_absolute_percentage_error

    mae = mean_absolute_error(y_true_eur, y_pred_eur)
    rmse = np.sqrt(mean_squared_error(y_true_eur, y_pred_eur))
    r2 = r2_score(y_true_eur, y_pred_eur)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    median_ae = np.median(np.abs(y_true_eur - y_pred_eur))

    print(f"\nBest Model Performance (on test set):")
    print(f"  R²:          {r2:.4f}")
    print(f"  MAE:         {mae:,.0f}€")
    print(f"  Median AE:   {median_ae:,.0f}€")
    print(f"  RMSE:        {rmse:,.0f}€")
    print(f"  MAPE:        {mape:.1f}%")

    # --- Error analysis by price bracket ---
    test_eval = test_df.copy()
    test_eval['predicted'] = y_pred
    test_eval['error'] = y_pred - y_true
    test_eval['abs_error'] = np.abs(test_eval['error'])
    test_eval['pct_error'] = test_eval['abs_error'] / y_true * 100

    brackets = [(0, 200), (200, 350), (350, 500), (500, 750), (750, 2000)]
    print(f"\nError by price bracket:")
    print(f"  {'Bracket (K€)':<20s}  {'N':>5s}  {'MAE (€)':>10s}  {'Med AE (€)':>10s}  {'MAPE':>6s}")
    print(f"  {'-'*20}  {'-'*5}  {'-'*10}  {'-'*10}  {'-'*6}")
    for low, high in brackets:
        mask = (y_true >= low) & (y_true < high)
        if mask.sum() > 0:
            bracket_mae = test_eval.loc[mask, 'abs_error'].mean() * 1000
            bracket_medae = test_eval.loc[mask, 'abs_error'].median() * 1000
            bracket_mape = test_eval.loc[mask, 'pct_error'].mean()
            print(f"  {low:>3d}K - {high:>4d}K         {mask.sum():>5d}  "
                  f"{bracket_mae:>10,.0f}  {bracket_medae:>10,.0f}  {bracket_mape:>5.1f}%")

    # --- Error by property type ---
    if 'property_type_HOUSE' in test_df.columns:
        print(f"\nError by property type:")
        for ptype, label in [(1, 'Houses'), (0, 'Apartments')]:
            mask = test_df['property_type_HOUSE'] == ptype
            if mask.sum() > 0:
                type_mae = test_eval.loc[mask, 'abs_error'].mean() * 1000
                type_mape = test_eval.loc[mask, 'pct_error'].mean()
                print(f"  {label:<15s}  MAE: {type_mae:>10,.0f}€  MAPE: {type_mape:.1f}%  (n={mask.sum()})")

    # --- Feature importance ---
    try:
        importance = predictor.feature_importance(test_df, num_shuffle_sets=3)
        print(f"\nTop 25 Feature Importance (permutation-based):")
        for feat, row in importance.head(25).iterrows():
            bar = "█" * int(row['importance'] / importance['importance'].max() * 40)
            print(f"  {feat:<35s}  {row['importance']:.4f}  {bar}")
        importance.to_csv(os.path.join(OUTPUT_DIR, 'feature_importance.csv'))
    except Exception as e:
        print(f"  Could not compute feature importance: {e}")
        importance = None

    return test_eval, importance


# ============================================================================
# PLOTS
# ============================================================================

def plot_results(predictor, test_eval, importance, target='price'):
    """Generate evaluation plots."""
    print("\nGenerating plots...")

    y_true = test_eval[target]
    y_pred = test_eval['predicted']

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # 1. Predicted vs Actual
    ax = axes[0, 0]
    ax.scatter(y_true * 1000, y_pred * 1000, alpha=0.1, s=8, c='steelblue')
    lims = [0, max(y_true.max(), y_pred.max()) * 1000 * 1.05]
    ax.plot(lims, lims, 'r--', linewidth=2, label='Perfect prediction')
    ax.set_xlabel('Actual Price (€)')
    ax.set_ylabel('Predicted Price (€)')
    ax.set_title('Predicted vs Actual Price')
    ax.legend()
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    # 2. Residual distribution
    ax = axes[0, 1]
    errors_eur = test_eval['error'] * 1000
    ax.hist(errors_eur, bins=80, color='steelblue', edgecolor='white', alpha=0.8)
    ax.axvline(0, color='red', linestyle='--', linewidth=2)
    ax.axvline(errors_eur.median(), color='orange', linestyle='--', linewidth=2,
               label=f'Median: {errors_eur.median():,.0f}€')
    ax.set_xlabel('Prediction Error (€)')
    ax.set_ylabel('Count')
    ax.set_title('Residual Distribution')
    ax.legend()

    # 3. Error by price bracket
    ax = axes[1, 0]
    test_eval['price_bracket'] = pd.cut(y_true, bins=[0, 200, 350, 500, 750, 2000],
                                        labels=['<200K', '200-350K', '350-500K', '500-750K', '>750K'])
    bracket_errors = test_eval.groupby('price_bracket', observed=True)['pct_error'].median()
    colors = _palette('RdYlGn_r', len(bracket_errors))
    ax.bar(range(len(bracket_errors)), bracket_errors.values, color=colors)
    ax.set_xticks(range(len(bracket_errors)))
    ax.set_xticklabels(bracket_errors.index)
    ax.set_ylabel('Median Absolute % Error')
    ax.set_title('Prediction Accuracy by Price Bracket')

    # 4. Feature importance (top 20)
    ax = axes[1, 1]
    if importance is not None:
        top20 = importance.head(20).sort_values('importance')
        colors = _palette('viridis', len(top20))
        ax.barh(range(len(top20)), top20['importance'].values, color=colors)
        ax.set_yticks(range(len(top20)))
        ax.set_yticklabels(top20.index, fontsize=9)
        ax.set_xlabel('Permutation Importance')
        ax.set_title('Top 20 Features (Permutation Importance)')
    else:
        ax.text(0.5, 0.5, 'Feature importance\nnot available', ha='center', va='center', fontsize=14)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'autogluon_evaluation.png'), dpi=150)
    plt.close()

    # --- Leaderboard comparison plot ---
    try:
        lb = pd.read_csv(os.path.join(OUTPUT_DIR, 'leaderboard.csv'))
        fig, ax = plt.subplots(figsize=(12, 6))
        lb_sorted = lb.sort_values('score_test')
        # AutoGluon reports negative RMSE as score for regression
        lb_sorted['rmse'] = -lb_sorted['score_test']
        ax.barh(range(len(lb_sorted)), lb_sorted['rmse'], color=_palette('viridis', len(lb_sorted)))
        ax.set_yticks(range(len(lb_sorted)))
        ax.set_yticklabels(lb_sorted['model'], fontsize=9)
        ax.set_xlabel('RMSE (K€)')
        ax.set_title('AutoGluon Model Comparison (Test RMSE)')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'model_comparison.png'), dpi=150)
        plt.close()
    except Exception as e:
        print(f"  Could not plot leaderboard: {e}")

    print(f"Plots saved to {OUTPUT_DIR}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    input_file = os.path.join(BASE_DIR, 'outputs', 'property_data_with_embeddings.csv')
    df = prepare_data(input_file)

    predictor, train_df, test_df = train_autogluon(
        df,
        target='price',
        time_limit=600,       # 10 minutes
        preset='best_quality'
    )

    test_eval, importance = evaluate(predictor, test_df)
    plot_results(predictor, test_eval, importance)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Model saved to: {os.path.join(OUTPUT_DIR, 'autogluon')}")
    print(f"To load: TabularPredictor.load('{os.path.join(OUTPUT_DIR, 'autogluon')}')")


if __name__ == "__main__":
    main()
