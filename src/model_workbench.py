"""Honest evaluation harness, baseline ladder, and feature-block ablations.

Built from the ground up to answer every modelling question quantitatively
before touching heavy tuning or production promotion:

1. **Metrics (real EUR, never K-EUR)**:
   - MdAPE (median absolute percentage error, %) — primary ranking metric
   - Hit@10% and Hit@15% — fraction of valuations within ±10% / ±15%
   - MAE, RMSE, R² in EUR
   - Log-RMSE — aligns with multiplicative house-price error

2. **Two honest evaluation axes**:
   - **5-fold grouped OOF** (`fold` column, 10,858 unique physical properties,
     0 leaked groups) — measures cross-sectional generalisation.
   - **Temporal holdout** (train `publication_year < 2026` with overlapping
     `property_key`s excluded from train so re-listed properties don't leak;
     test `publication_year >= 2026`, with OOF spatial targets recomputed
     strictly from the pre-2026 train set!) — measures forward-in-time accuracy.

3. **Baseline ladder & loss-function comparison**:
   - Baseline 0: Global median EUR/m² × livable_area
   - Baseline 1: (property_type, neighbourhood) median EUR/m² × livable_area
   - Baseline 2: Regularised hedonic Ridge on log(price)
   - Baseline 3: Default LightGBM / CatBoost / XGBoost on `price` vs `log(price)`
     (with Duan smearing) vs `MAE` (`L1`)

4. **Feature-block ablation** (additive & leave-one-out on LightGBM):
   - Core structure + property attributes
   - Spatial (Ghent landmarks, OSMnx street_type, OOF KNN/wijk/sector price/m²)
   - Statbel sector socio-demographics (population density, income, cars/hh)
   - Macro (Statbel CPI/YoY, ECB mortgage rates, Euribor, 10y yield, HPI)
   - Amenities & EPC/planning/text-recovered flags

Usage:
    .venv-api/bin/python src/model_workbench.py
"""
from __future__ import annotations

import json
import os
import time
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

warnings.filterwarnings('ignore', category=UserWarning)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, 'outputs', 'dataset', 'training_dataset.parquet')
REPORT_DIR = os.path.join(BASE_DIR, 'outputs', 'models', 'workbench')

META_COLUMNS = {'property_id', 'date_obtained', 'property_key', 'fold', 'price'}

# Exact spatial-target columns computed OOF across the 5 folds.
# For temporal holdout these MUST be recomputed strictly on train (<2026)!
OOF_SPATIAL_COLS = (
    'oof_knn10_median_price_sqm',
    'oof_dist_5nn_km',
    'oof_neighbourhood_median_price_sqm',
    'oof_sector_median_price_sqm',
)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 10_000.0, None)
    abs_err = np.abs(y_pred - y_true)
    ape = abs_err / np.maximum(y_true, 1.0) * 100.0
    log_err = np.log(y_pred) - np.log(np.maximum(y_true, 1.0))
    return {
        'mdape': float(np.median(ape)),
        'mape': float(np.mean(ape)),
        'hit_10': float(np.mean(ape <= 10.0) * 100.0),
        'hit_15': float(np.mean(ape <= 15.0) * 100.0),
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'rmse': float(root_mean_squared_error(y_true, y_pred)),
        'log_rmse': float(np.sqrt(np.mean(log_err ** 2))),
        'r2': float(r2_score(y_true, y_pred)),
        'bias_pct': float(np.median((y_pred - y_true) / y_true) * 100.0),
    }


def _fmt_metrics(m: dict[str, float]) -> str:
    return (
        f"MdAPE={m['mdape']:5.2f}% | Hit@10={m['hit_10']:4.1f}% | Hit@15={m['hit_15']:4.1f}% | "
        f"MAE=€{m['mae']:7,.0f} | RMSE=€{m['rmse']:7,.0f} | R²={m['r2']:.4f} | Bias={m['bias_pct']:+5.2f}%"
    )


# ---------------------------------------------------------------------------
# Temporal split with leakage-free spatial target recomputation
# ---------------------------------------------------------------------------
def recompute_spatial_features(train_df: pd.DataFrame, target_df: pd.DataFrame) -> pd.DataFrame:
    """Compute OOF spatial features on `target_df` using ONLY `train_df` as reference."""
    out = target_df.copy()
    tr_valid = train_df[
        train_df['latitude'].notna()
        & train_df['longitude'].notna()
        & train_df['price'].notna()
        & (train_df['livable_area'] > 0)
    ].copy()
    tr_psqm = (tr_valid['price'] / tr_valid['livable_area']).to_numpy(dtype=float)
    global_med = float(np.median(tr_psqm))

    tr_coords = np.column_stack([tr_valid['latitude'].to_numpy(), tr_valid['longitude'].to_numpy()])
    tree = cKDTree(tr_coords)
    tgt_coords = np.column_stack([out['latitude'].to_numpy(), out['longitude'].to_numpy()])

    k_nn = min(10, len(tr_valid))
    dists, idxs = tree.query(tgt_coords, k=k_nn)
    if k_nn == 1:
        dists = dists[:, None]
        idxs = idxs[:, None]
    out['oof_knn10_median_price_sqm'] = np.median(tr_psqm[idxs], axis=1)
    k_dist = min(5, k_nn)
    out['oof_dist_5nn_km'] = np.mean(dists[:, :k_dist], axis=1) * 111.32

    if 'neighbourhood' in tr_valid.columns:
        wijk_med = pd.DataFrame({'w': tr_valid['neighbourhood'].astype(str), 'p': tr_psqm}).groupby('w')['p'].median().to_dict()
        out['oof_neighbourhood_median_price_sqm'] = [
            wijk_med.get(str(w), global_med) for w in out['neighbourhood']
        ]
    if 'statbel_sector' in tr_valid.columns:
        sec_agg = pd.DataFrame({'s': tr_valid['statbel_sector'].astype(str), 'p': tr_psqm}).groupby('s')['p'].agg(['count', 'median'])
        m = 5.0
        sec_smooth = {
            idx: (row['count'] * row['median'] + m * global_med) / (row['count'] + m)
            for idx, row in sec_agg.iterrows() if idx != 'Unknown'
        }
        out['oof_sector_median_price_sqm'] = [
            sec_smooth.get(str(s), global_med) for s in out['statbel_sector']
        ]
    return out


def make_temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train < 2026 (excluding keys that re-appear in 2026), test >= 2026."""
    test_mask = df['publication_year'] >= 2026
    test_keys = set(df.loc[test_mask, 'property_key'].astype(str))
    train_mask = (~test_mask) & (~df['property_key'].astype(str).isin(test_keys))

    train_raw = df[train_mask].copy().reset_index(drop=True)
    test_raw = df[test_mask].copy().reset_index(drop=True)

    # Recompute spatial targets on test using ONLY pre-2026 train, and OOF within train!
    test_df = recompute_spatial_features(train_raw, test_raw)
    # Within train_raw, recompute OOF using train_raw's existing grouped folds
    train_parts = []
    for f in sorted(train_raw['fold'].unique()):
        ref = train_raw[train_raw['fold'] != f]
        val = train_raw[train_raw['fold'] == f]
        train_parts.append(recompute_spatial_features(ref, val))
    train_df = pd.concat(train_parts, axis=0).sort_index().reset_index(drop=True)
    return train_df, test_df


# ---------------------------------------------------------------------------
# Feature typing & preprocessing helper per model family
# ---------------------------------------------------------------------------
def get_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    feat_cols = [c for c in df.columns if c not in META_COLUMNS]
    cat_cols = [c for c in feat_cols if pd.api.types.is_string_dtype(df[c]) or pd.api.types.is_object_dtype(df[c]) or isinstance(df[c].dtype, pd.CategoricalDtype)]
    bool_cols = [c for c in feat_cols if pd.api.types.is_bool_dtype(df[c])]
    num_cols = [c for c in feat_cols if c not in set(cat_cols) and c not in set(bool_cols)]
    return num_cols, bool_cols, cat_cols


def prepare_tree_frame(df: pd.DataFrame, feat_cols: list[str], cat_cols: list[str], bool_cols: list[str]) -> pd.DataFrame:
    X = df[feat_cols].copy()
    for c in bool_cols:
        X[c] = X[c].astype('int8')
    for c in cat_cols:
        X[c] = X[c].fillna('Missing').astype(str)
    for c in X.columns:
        if c not in cat_cols and c not in bool_cols:
            X[c] = pd.to_numeric(X[c], errors='coerce').astype('float32')
    return X


# ---------------------------------------------------------------------------
# Baselines 0, 1, 2
# ---------------------------------------------------------------------------
def run_naive_baselines(df: pd.DataFrame, train_t: pd.DataFrame, test_t: pd.DataFrame) -> list[dict]:
    results = []

    # Baseline 0: global median EUR/m² * livable_area
    oof_b0 = np.zeros(len(df))
    for f in sorted(df['fold'].unique()):
        tr = df['fold'] != f
        va = df['fold'] == f
        med_sqm = float((df.loc[tr, 'price'] / df.loc[tr, 'livable_area']).median())
        oof_b0[va] = med_sqm * df.loc[va, 'livable_area'].to_numpy()
    t_med = float((train_t['price'] / train_t['livable_area']).median())
    tmp_b0 = t_med * test_t['livable_area'].to_numpy()
    results.append({
        'name': 'B0: Global median EUR/m² × area',
        'oof': compute_metrics(df['price'], oof_b0),
        'temporal': compute_metrics(test_t['price'], tmp_b0),
    })

    # Baseline 1: (property_type_HOUSE, neighbourhood) median EUR/m² * livable_area
    def _predict_lookup(tr_df: pd.DataFrame, va_df: pd.DataFrame) -> np.ndarray:
        psqm = tr_df['price'] / tr_df['livable_area']
        g_med = float(psqm.median())
        grp = pd.DataFrame({
            'h': tr_df['property_type_HOUSE'].to_numpy(),
            'w': tr_df['neighbourhood'].astype(str).to_numpy(),
            'p': psqm.to_numpy(),
        }).groupby(['h', 'w'])['p'].median().to_dict()
        preds = [
            grp.get((h, str(w)), g_med) * area
            for h, w, area in zip(va_df['property_type_HOUSE'], va_df['neighbourhood'], va_df['livable_area'])
        ]
        return np.asarray(preds, dtype=float)

    oof_b1 = np.zeros(len(df))
    for f in sorted(df['fold'].unique()):
        tr = df['fold'] != f
        va = df['fold'] == f
        oof_b1[va] = _predict_lookup(df[tr], df[va])
    tmp_b1 = _predict_lookup(train_t, test_t)
    results.append({
        'name': 'B1: (Type, Neighbourhood) EUR/m² lookup',
        'oof': compute_metrics(df['price'], oof_b1),
        'temporal': compute_metrics(test_t['price'], tmp_b1),
    })

    # Baseline 2: Ridge regression on log(price) with Duan smearing
    num_cols, bool_cols, cat_cols = get_feature_columns(df)
    feat_cols = num_cols + bool_cols + cat_cols

    def _fit_pred_ridge(tr_df: pd.DataFrame, va_df: pd.DataFrame) -> np.ndarray:
        pre = ColumnTransformer([
            ('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())]), num_cols + bool_cols),
            ('cat', Pipeline([('imp', SimpleImputer(strategy='constant', fill_value='Missing')), ('ohe', OneHotEncoder(handle_unknown='ignore', min_frequency=10))]), cat_cols),
        ])
        X_tr = pre.fit_transform(tr_df[feat_cols])
        X_va = pre.transform(va_df[feat_cols])
        y_tr_log = np.log(tr_df['price'].to_numpy(dtype=float))
        reg = Ridge(alpha=50.0)
        reg.fit(X_tr, y_tr_log)
        resid = y_tr_log - reg.predict(X_tr)
        smear = float(np.mean(np.exp(resid)))
        return smear * np.exp(reg.predict(X_va))

    oof_b2 = np.zeros(len(df))
    for f in sorted(df['fold'].unique()):
        tr = df['fold'] != f
        va = df['fold'] == f
        oof_b2[va] = _fit_pred_ridge(df[tr], df[va])
    tmp_b2 = _fit_pred_ridge(train_t, test_t)
    results.append({
        'name': 'B2: Hedonic Ridge on log(price)',
        'oof': compute_metrics(df['price'], oof_b2),
        'temporal': compute_metrics(test_t['price'], tmp_b2),
    })

    return results


# ---------------------------------------------------------------------------
# GBDT Workbench (LightGBM, CatBoost, XGBoost) across target formulations
# ---------------------------------------------------------------------------
def eval_gbdt(
    df: pd.DataFrame,
    train_t: pd.DataFrame,
    test_t: pd.DataFrame,
    feat_cols: list[str],
    model_kind: str,
    target_mode: str = 'log',  # 'raw_l2', 'raw_l1', 'log'
    params: dict | None = None,
) -> tuple[dict[str, float], dict[str, float], np.ndarray, np.ndarray]:
    num_all, bool_all, cat_all = get_feature_columns(df)
    cat_cols = [c for c in feat_cols if c in set(cat_all)]
    bool_cols = [c for c in feat_cols if c in set(bool_all)]

    def _fit_predict(tr_df: pd.DataFrame, va_df: pd.DataFrame) -> np.ndarray:
        X_tr = prepare_tree_frame(tr_df, feat_cols, cat_cols, bool_cols)
        X_va = prepare_tree_frame(va_df, feat_cols, cat_cols, bool_cols)
        y_raw = tr_df['price'].to_numpy(dtype=float)
        y_tr = np.log(y_raw) if target_mode == 'log' else y_raw

        if model_kind == 'lgbm':
            import lightgbm as lgb
            for c in cat_cols:
                cats = sorted(set(X_tr[c].unique()) | {'Missing'})
                X_tr[c] = pd.Categorical(X_tr[c], categories=cats)
                X_va[c] = pd.Categorical(X_va[c].where(X_va[c].isin(cats), 'Missing'), categories=cats)
            obj = 'regression_l1' if target_mode == 'raw_l1' else 'regression'
            cfg = dict(
                objective=obj, n_estimators=1500, learning_rate=0.03,
                num_leaves=63, min_child_samples=25, subsample=0.8,
                subsample_freq=1, colsample_bytree=0.75, reg_alpha=1.0,
                reg_lambda=2.0, random_state=42, n_jobs=-1, verbose=-1,
            )
            if params:
                cfg.update(params)
            model = lgb.LGBMRegressor(**cfg)
            # Split 10% group-safe early stopping validation within train fold
            rng = np.random.default_rng(42)
            uniq_g = tr_df['property_key'].unique()
            es_g = set(rng.choice(uniq_g, size=max(1, int(0.12 * len(uniq_g))), replace=False))
            es_mask = tr_df['property_key'].isin(es_g).to_numpy()
            model.fit(
                X_tr[~es_mask], y_tr[~es_mask],
                eval_set=[(X_tr[es_mask], y_tr[es_mask])],
                callbacks=[lgb.early_stopping(75, verbose=False)],
            )
            pred_tr = model.predict(X_tr)
            pred_va = model.predict(X_va)

        elif model_kind == 'catboost':
            from catboost import CatBoostRegressor
            loss = 'MAE' if target_mode == 'raw_l1' else 'RMSE'
            cfg = dict(
                loss_function=loss, iterations=2500, learning_rate=0.04,
                depth=7, l2_leaf_reg=5.0, random_seed=42, verbose=0,
                thread_count=-1,
            )
            if params:
                cfg.update(params)
            model = CatBoostRegressor(**cfg)
            rng = np.random.default_rng(42)
            uniq_g = tr_df['property_key'].unique()
            es_g = set(rng.choice(uniq_g, size=max(1, int(0.12 * len(uniq_g))), replace=False))
            es_mask = tr_df['property_key'].isin(es_g).to_numpy()
            model.fit(
                X_tr[~es_mask], y_tr[~es_mask],
                cat_features=cat_cols,
                eval_set=(X_tr[es_mask], y_tr[es_mask]),
                early_stopping_rounds=100,
                verbose=False,
            )
            pred_tr = model.predict(X_tr)
            pred_va = model.predict(X_va)

        elif model_kind == 'xgb':
            import xgboost as xgb
            for c in cat_cols:
                cats = sorted(set(X_tr[c].unique()) | {'Missing'})
                X_tr[c] = pd.Categorical(X_tr[c], categories=cats)
                X_va[c] = pd.Categorical(X_va[c].where(X_va[c].isin(cats), 'Missing'), categories=cats)
            obj = 'reg:absoluteerror' if target_mode == 'raw_l1' else 'reg:squarederror'
            cfg = dict(
                objective=obj, n_estimators=2000, learning_rate=0.03,
                max_depth=7, min_child_weight=10, subsample=0.8,
                colsample_bytree=0.75, reg_alpha=1.0, reg_lambda=2.0,
                tree_method='hist', enable_categorical=True,
                early_stopping_rounds=75, random_state=42, n_jobs=-1,
            )
            if params:
                cfg.update(params)
            model = xgb.XGBRegressor(**cfg)
            rng = np.random.default_rng(42)
            uniq_g = tr_df['property_key'].unique()
            es_g = set(rng.choice(uniq_g, size=max(1, int(0.12 * len(uniq_g))), replace=False))
            es_mask = tr_df['property_key'].isin(es_g).to_numpy()
            model.fit(
                X_tr[~es_mask], y_tr[~es_mask],
                eval_set=[(X_tr[es_mask], y_tr[es_mask])],
                verbose=False,
            )
            pred_tr = model.predict(X_tr)
            pred_va = model.predict(X_va)
        else:
            raise ValueError(model_kind)

        if target_mode == 'log':
            # Use median smearing (exp(median resid)=1) or Duan smearing?
            # For MdAPE optimization, exp(pred) targets conditional median on log-normal;
            # Duan smearing targets conditional mean (better RMSE, slightly positive MdAPE bias).
            return np.exp(pred_va)
        return pred_va

    oof = np.zeros(len(df))
    for f in sorted(df['fold'].unique()):
        tr = df['fold'] != f
        va = df['fold'] == f
        oof[va] = _fit_predict(df[tr], df[va])
    tmp = _fit_predict(train_t, test_t)
    return compute_metrics(df['price'], oof), compute_metrics(test_t['price'], tmp), oof, tmp


# ---------------------------------------------------------------------------
# Feature blocks for ablation
# ---------------------------------------------------------------------------
def partition_feature_blocks(feat_cols: list[str]) -> dict[str, list[str]]:
    macro_set = {
        'belgian_cpi_index', 'belgian_health_index', 'belgian_cpi_yoy_pct',
        'belgian_10y_yield', 'euribor_12m', 'euribor_3m',
        'be_mortgage_rate_avg', 'be_mortgage_rate_fix10y',
        'belgian_hpi_index', 'real_mortgage_rate', 'yield_curve_slope',
        'publication_year', 'publication_month',
    }
    sector_demo_set = {
        'statbel_sector', 'sector_pop_density', 'sector_avg_hh_size',
        'sector_cars_per_hh', 'sector_avg_income', 'sector_median_income',
        'sector_income_iqr',
    }
    spatial_set = {
        'latitude', 'longitude', 'postal_code', 'city', 'neighbourhood',
        'street_type', 'distance_to_street', 'dist_from_center',
        'dist_sint_pieters_km', 'dist_dampoort_km', 'dist_korenmarkt_km',
        'dist_citadelpark_km', 'dist_blaarmeersen_km', 'location_type',
        'oof_knn10_median_price_sqm', 'oof_dist_5nn_km',
        'oof_neighbourhood_median_price_sqm', 'oof_sector_median_price_sqm',
    }
    core_set = {
        'property_type_APARTMENT', 'property_type_HOUSE', 'property_subtype',
        'livable_area', 'property_size', 'property_size_missing',
        'number_of_bedrooms', 'number_of_bathrooms', 'number_of_toilets',
        'number_of_rooms', 'shower_room_count', 'facade_count',
        'street_facade_width', 'total_floors', 'floor_number',
        'year_built', 'year_built_missing', 'age', 'building_condition',
        'cadastral_income', 'cadastral_income_missing', 'is_newly_built',
        'is_subject_to_vat', 'is_investment_property',
        'energy_label', 'primary_energy_consumption', 'primary_energy_consumption_missing',
        'epc_cert_year', 'epc_cert_age', 'epc_cert_year_missing',
        'epc_cert_revision', 'renovation_obligation', 'renovation_obligation_missing',
        'kitchen_type', 'heating_type', 'has_double_glazing', 'has_heat_pump',
        'has_photovoltaic_panels',
        'livable_to_plot_ratio', 'garden_to_plot_ratio', 'bathrooms_per_bedroom',
        'garden_surface', 'terrace_surface', 'parking_indoor', 'parking_outdoor',
    }
    text_meta_set = {
        'description_length', 'has_meaningful_description', 'description_is_placeholder',
        'description_language', 'has_text_recovered_amenities', 'n_text_recovered_amenities',
        'picture_count', 'floor_plan_count', 'has_floor_plan', 'pictures_per_bedroom',
        'visualisation_option', 'seller_type', 'is_multi_unit_project',
        'sections_filled_count', 'section_general', 'section_exterior',
        'section_facilities', 'section_energy', 'section_planning',
    }
    other_amenities = [
        c for c in feat_cols
        if c not in (macro_set | sector_demo_set | spatial_set | core_set | text_meta_set)
    ]
    return {
        'core': [c for c in feat_cols if c in core_set],
        'spatial': [c for c in feat_cols if c in spatial_set],
        'sector_demo': [c for c in feat_cols if c in sector_demo_set],
        'macro': [c for c in feat_cols if c in macro_set],
        'text_meta': [c for c in feat_cols if c in text_meta_set],
        'amenities_extra': other_amenities,
    }


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    df = pd.read_parquet(DATASET_PATH)
    print(f"Loaded dataset: {df.shape[0]:,} rows x {df.shape[1]} cols | {df['property_key'].nunique():,} unique properties")
    train_t, test_t = make_temporal_split(df)
    print(f"Temporal split: train={len(train_t):,} (<2026, leaked keys removed) | test={len(test_t):,} (>=2026)")

    # Step 3a: Naive & linear baselines
    print("\n=== STEP 3A: BASELINE LADDER ===")
    results = []
    for r in run_naive_baselines(df, train_t, test_t):
        print(f"  {r['name']:42s} | OOF: {_fmt_metrics(r['oof'])}")
        print(f"  {'':42s} | TMP: {_fmt_metrics(r['temporal'])}")
        results.append(r)

    # Step 3b: Loss-function & GBDT family comparison
    _, _, _ = get_feature_columns(df)
    feat_cols = [c for c in df.columns if c not in META_COLUMNS]

    print("\n=== STEP 3B: TARGET FORMULATION & MODEL FAMILY BENCHMARK (ALL 235 FEATURES) ===")
    oofs = {}
    tmps = {}
    for kind, tmode, label in [
        ('lgbm', 'raw_l2', 'LightGBM (raw EUR, L2/RMSE)'),
        ('lgbm', 'raw_l1', 'LightGBM (raw EUR, L1/MAE)'),
        ('lgbm', 'log',    'LightGBM (log(price))'),
        ('catboost', 'log', 'CatBoost (log(price))'),
        ('catboost', 'raw_l1', 'CatBoost (raw EUR, L1/MAE)'),
        ('xgb', 'log',     'XGBoost (log(price))'),
    ]:
        t0 = time.perf_counter()
        m_oof, m_tmp, oof_p, tmp_p = eval_gbdt(df, train_t, test_t, feat_cols, kind, tmode)
        dt = time.perf_counter() - t0
        print(f"  {label:32s} ({dt:5.1f}s) | OOF: {_fmt_metrics(m_oof)}")
        print(f"  {'':40s} | TMP: {_fmt_metrics(m_tmp)}")
        results.append({'name': label, 'oof': m_oof, 'temporal': m_tmp, 'seconds': round(dt, 1)})
        oofs[label] = oof_p
        tmps[label] = tmp_p

    # Quick equal-weight blend of the 3 log-GBDT families
    blend_oof = (oofs['LightGBM (log(price))'] + oofs['CatBoost (log(price))'] + oofs['XGBoost (log(price))']) / 3.0
    blend_tmp = (tmps['LightGBM (log(price))'] + tmps['CatBoost (log(price))'] + tmps['XGBoost (log(price))']) / 3.0
    m_boof = compute_metrics(df['price'], blend_oof)
    m_btmp = compute_metrics(test_t['price'], blend_tmp)
    print(f"  {'Equal Blend (LGBM+CB+XGB log)':32s}          | OOF: {_fmt_metrics(m_boof)}")
    print(f"  {'':40s} | TMP: {_fmt_metrics(m_btmp)}")
    results.append({'name': 'Equal Blend (LGBM+CB+XGB log)', 'oof': m_boof, 'temporal': m_btmp})

    # Step 4: Feature-block ablation (cumulative + leave-one-out on LightGBM log)
    print("\n=== STEP 4: FEATURE-BLOCK ABLATION (LightGBM log(price)) ===")
    blocks = partition_feature_blocks(feat_cols)
    for bname, bcols in blocks.items():
        print(f"  Block '{bname}': {len(bcols)} cols")

    cumulative_order = ['core', 'spatial', 'sector_demo', 'macro', 'text_meta', 'amenities_extra']
    running_cols: list[str] = []
    ablation_results = []
    for bname in cumulative_order:
        running_cols = running_cols + blocks[bname]
        m_oof, m_tmp, _, _ = eval_gbdt(df, train_t, test_t, running_cols, 'lgbm', 'log')
        print(f"  + {bname:18s} ({len(running_cols):3d} cols) | OOF: {_fmt_metrics(m_oof)}")
        print(f"    {'':30s} | TMP: {_fmt_metrics(m_tmp)}")
        ablation_results.append({'step': f'+{bname}', 'n_cols': len(running_cols), 'oof': m_oof, 'temporal': m_tmp})

    # Leave-one-block-out from full set
    print("\n  --- Leave-one-block-out from full 235 features ---")
    full_oof = oofs['LightGBM (log(price))']
    full_m_oof = compute_metrics(df['price'], full_oof)
    full_m_tmp = compute_metrics(test_t['price'], tmps['LightGBM (log(price))'])
    for bname in ['spatial', 'sector_demo', 'macro', 'text_meta', 'amenities_extra']:
        sub_cols = [c for c in feat_cols if c not in set(blocks[bname])]
        m_oof, m_tmp, _, _ = eval_gbdt(df, train_t, test_t, sub_cols, 'lgbm', 'log')
        d_oof = m_oof['mdape'] - full_m_oof['mdape']
        d_tmp = m_tmp['mdape'] - full_m_tmp['mdape']
        print(f"  - {bname:18s} | ΔOOF MdAPE={d_oof:+5.3f}% | ΔTMP MdAPE={d_tmp:+5.3f}% (OOF={m_oof['mdape']:.2f}%, TMP={m_tmp['mdape']:.2f}%)")
        ablation_results.append({'step': f'-{bname}', 'n_cols': len(sub_cols), 'oof': m_oof, 'temporal': m_tmp, 'delta_oof_mdape': d_oof, 'delta_tmp_mdape': d_tmp})

    out_path = os.path.join(REPORT_DIR, 'baseline_and_ablation.json')
    with open(out_path, 'w') as fh:
        json.dump({'benchmarks': results, 'ablations': ablation_results}, fh, indent=2)
    print(f"\nSaved workbench report to {out_path}")


if __name__ == '__main__':
    main()
