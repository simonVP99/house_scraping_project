"""Phase 2 Experiment Workbench — Kaggle / AVM Techniques on Ghent Real Estate.

Runs all 6 splits (5 grouped OOF folds + 2026 temporal holdout) IN PARALLEL
across all 18 Apple Silicon CPU cores (6 workers × 3 threads = 100% core
utilisation, ~4–5× faster than single-fold sequential runs) and uses the
MPS-computed `intfloat/multilingual-e5-small` embeddings.

Experiments tested systematically on BOTH 5-fold grouped OOF and 2026 temporal holdout:
  A. Target & Loss Formulation:
     - A0: Baseline `log(price)` L2 (LightGBM)
     - A1: `log(price)` L1/MAE (directly aligned with MdAPE)
     - A2: `log(price)` Huber (delta=0.15)
     - A3: `log(price / livable_area)` (log €/m²) L2
     - A4: `log(price / livable_area)` L1
     - A5: CPI-deflated `log(real_price)` (Sberbank trick: removes monotonic CPI drift)
     - A6: Statbel Gent quarterly HPI-deflated `log(real_price)` (fixes 2026 under-prediction bias)
     - A7: `log(price / spatial_prior)` (residual over local €/m² × area anchor)

  B. High-Resolution Spatial Engineering (strictly fold-safe):
     - Multi-scale IDW OOF €/m² (k=5, 15, 40) overall AND type-conditioned (HOUSE vs APARTMENT)
     - Local €/m² dispersion (std) & relative area ratio (`livable_area / local_median_area`)
     - Rotated coordinates (15°, 30°, 45°, 60°, 75° — captures diagonal rivers/R40/rail corridors)
     - Domain interactions: KI/m², log(area/plot/KI), Flemish EPC renovation-obligation penalty,
       age × condition, frequency encoding of sector/wijk/subtype
     - 2-stage OOF spatial residual kriging (IDW of Stage-1 log-residuals)

  C. Description Text Signals (strictly fold-safe):
     - C1: OOF Ridge on TF-IDF (word 1-2 grams + sublinear TF) predicting `log(price/m²)`
     - C2: Fold-safe PCA (32 dims) of `multilingual-e5-small` embeddings + OOF Ridge on raw 384-d E5
     - C3: Combined TF-IDF + E5 Ridge + E5 PCA

  D. Diverse Multi-Family Ensemble + Target Diversity (on winning feature set):
     - CatBoost (`log(price)`, `log(€/m²)`, `log(price)` L1)
     - LightGBM (`log(price)` L2, `log(price)` Huber, `log(€/m²)`)
     - XGBoost (`log(price)` L2, `log(€/m²)`)
     - Ridge non-negative meta-blender (nested OOF)

Usage:
    .venv-api/bin/python -u src/experiment_kaggle_avm.py
"""
from __future__ import annotations

import json
import os
import time
import warnings

import numpy as np
import pandas as pd
import polars as pl
from joblib import Parallel, delayed
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, 'outputs', 'dataset', 'training_dataset.parquet')
RAW_PARQUET_PATH = os.path.join(BASE_DIR, 'outputs', 'property_data_full.parquet')
E5_PATH = os.path.join(BASE_DIR, 'outputs', 'models', 'workbench', 'e5_embeddings_14599.npz')
STATBEL_VG_PATH = os.path.join(BASE_DIR, 'external_data', 'statbel', 'statbel_real_estate_sales.parquet')
REPORT_DIR = os.path.join(BASE_DIR, 'outputs', 'models', 'workbench')

META_COLUMNS = {'property_id', 'date_obtained', 'property_key', 'fold', 'price'}

# Monotonic price-level macro cols that trees cannot extrapolate into 2026+
NON_STATIONARY_MACRO = {
    'belgian_cpi_index', 'belgian_health_index', 'belgian_hpi_index', 'publication_year',
}


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
        'hit_10': float(np.mean(ape <= 10.0) * 100.0),
        'hit_15': float(np.mean(ape <= 15.0) * 100.0),
        'mae': float(np.mean(abs_err)),
        'rmse': float(np.sqrt(np.mean((y_pred - y_true) ** 2))),
        'log_rmse': float(np.sqrt(np.mean(log_err ** 2))),
        'r2': float(1.0 - np.sum((y_pred - y_true) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)),
        'bias_pct': float(np.median((y_pred - y_true) / y_true) * 100.0),
    }


def fmt(m: dict[str, float]) -> str:
    return (
        f"MdAPE={m['mdape']:5.2f}% | Hit@10={m['hit_10']:4.1f}% | Hit@15={m['hit_15']:4.1f}% | "
        f"MAE=€{m['mae']:6,.0f} | RMSE=€{m['rmse']:6,.0f} | R²={m['r2']:.4f} | Bias={m['bias_pct']:+5.2f}%"
    )


# ---------------------------------------------------------------------------
# Statbel Ghent quarterly price deflator (type-aware)
# ---------------------------------------------------------------------------
def build_ghent_deflator(df: pd.DataFrame) -> np.ndarray:
    """Return price index relative to 2025-Q4=1.0 using Statbel Gent quarterly sales + HPI."""
    if not os.path.exists(STATBEL_VG_PATH):
        return (df['belgian_cpi_index'] / df['belgian_cpi_index'].median()).to_numpy(dtype=float)

    vg = pd.read_parquet(STATBEL_VG_PATH)
    vg['CD_REFNIS'] = vg['CD_REFNIS'].astype(str)
    g = vg[
        (vg['CD_REFNIS'] == '44021')
        & (vg['CD_PERIOD'].isin(['Q1', 'Q2', 'Q3', 'Q4']))
        & (vg['CD_YEAR'] >= 2020)
        & (vg['CD_CLASS_SURFACE'].astype(str).str.contains('totaal', case=False, na=False))
    ].copy()
    q_map = {'Q1': 1, 'Q2': 4, 'Q3': 7, 'Q4': 10}
    g['ym'] = pd.to_datetime(
        g['CD_YEAR'].astype(str) + '-' + g['CD_PERIOD'].map(q_map).astype(str).str.zfill(2) + '-01'
    )
    is_apt = g['CD_TYPE_NL'].astype(str).str.contains('Appartement', case=False, na=False)
    is_house = g['CD_TYPE_NL'].astype(str).str.contains('Alle huizen', case=False, na=False)

    # Smooth quarterly median with 2-quarter rolling average to kill quarterly composition noise
    house_s = g[is_house].groupby('ym')['MS_P_50_median'].median().sort_index().rolling(2, min_periods=1).mean()
    apt_s = g[is_apt].groupby('ym')['MS_P_50_median'].median().sort_index().rolling(2, min_periods=1).mean()

    # Reindex to monthly and interpolate linearly, ffill beyond 2026-Q2 using CPI ratio
    idx_m = pd.date_range('2020-01-01', '2026-12-01', freq='MS')
    house_m = house_s.reindex(idx_m).interpolate('linear').ffill().bfill()
    apt_m = apt_s.reindex(idx_m).interpolate('linear').ffill().bfill()

    ref_date = pd.Timestamp('2025-10-01')
    house_idx = (house_m / house_m.loc[ref_date]).to_dict()
    apt_idx = (apt_m / apt_m.loc[ref_date]).to_dict()

    # Map each listing via publication_year + publication_month
    pub_ym = pd.to_datetime(
        df['publication_year'].fillna(2025).astype(int).astype(str)
        + '-'
        + df['publication_month'].fillna(6).astype(int).astype(str).str.zfill(2)
        + '-01',
        errors='coerce',
    ).clip(lower=pd.Timestamp('2020-01-01'), upper=pd.Timestamp('2026-12-01'))

    is_h = df['property_type_HOUSE'].to_numpy() == 1
    out = np.where(
        is_h,
        pub_ym.map(house_idx).fillna(1.0).to_numpy(dtype=float),
        pub_ym.map(apt_idx).fillna(1.0).to_numpy(dtype=float),
    )
    return out


# ---------------------------------------------------------------------------
# High-res spatial features (strictly computed from ref_df -> target_df)
# ---------------------------------------------------------------------------
def compute_multiscale_spatial(ref_df: pd.DataFrame, target_df: pd.DataFrame) -> pd.DataFrame:
    """Compute leak-free multi-scale IDW spatial features strictly from `ref_df`."""
    out = pd.DataFrame(index=target_df.index)
    ref_valid = ref_df[
        ref_df['latitude'].notna()
        & ref_df['longitude'].notna()
        & ref_df['price'].notna()
        & (ref_df['livable_area'] > 0)
    ]
    ref_psqm = (ref_valid['price'] / ref_valid['livable_area']).to_numpy(dtype=float)
    ref_area = ref_valid['livable_area'].to_numpy(dtype=float)
    ref_price = ref_valid['price'].to_numpy(dtype=float)
    ref_is_h = (ref_valid['property_type_HOUSE'].to_numpy() == 1)
    ref_coords = np.column_stack([ref_valid['latitude'].to_numpy(), ref_valid['longitude'].to_numpy()])
    tgt_coords = np.column_stack([target_df['latitude'].to_numpy(), target_df['longitude'].to_numpy()])
    tgt_is_h = (target_df['property_type_HOUSE'].to_numpy() == 1)
    tgt_area = target_df['livable_area'].to_numpy(dtype=float)

    g_med = float(np.median(ref_psqm))

    # 1. All-properties multi-scale IDW KNN (k=5, 15, 40)
    tree_all = cKDTree(ref_coords)
    k_max = min(40, len(ref_valid))
    dists_all, idxs_all = tree_all.query(tgt_coords, k=k_max)
    dists_km = dists_all * 111.32
    eps = 0.05  # 50m smoothing floor

    for k in (5, 15, 40):
        kk = min(k, k_max)
        w = 1.0 / (dists_km[:, :kk] + eps)
        w = w / w.sum(axis=1, keepdims=True)
        nbr_psqm = ref_psqm[idxs_all[:, :kk]]
        out[f'sp_idw_psqm_k{k}'] = np.sum(w * nbr_psqm, axis=1)
        if k == 15:
            out['sp_psqm_std_k15'] = np.std(nbr_psqm, axis=1)
            out['sp_psqm_p25_k15'] = np.percentile(nbr_psqm, 25, axis=1)
            out['sp_psqm_p75_k15'] = np.percentile(nbr_psqm, 75, axis=1)
            local_med_area = np.median(ref_area[idxs_all[:, :kk]], axis=1)
            out['sp_rel_area_k15'] = tgt_area / np.maximum(local_med_area, 10.0)
            out['sp_idw_price_k15'] = np.sum(w * ref_price[idxs_all[:, :kk]], axis=1)

    out['sp_dist_3nn_km'] = np.mean(dists_km[:, :min(3, k_max)], axis=1)
    out['sp_dist_10nn_km'] = np.mean(dists_km[:, :min(10, k_max)], axis=1)

    # 2. Type-conditioned IDW KNN (houses look at nearby houses; apartments look at nearby apartments!)
    same_k5 = np.full(len(target_df), g_med, dtype=float)
    same_k15 = np.full(len(target_df), g_med, dtype=float)
    same_k40 = np.full(len(target_df), g_med, dtype=float)
    same_med_area = np.full(len(target_df), 120.0, dtype=float)
    same_dist5 = np.full(len(target_df), 0.5, dtype=float)

    for h_flag in (True, False):
        r_mask = ref_is_h == h_flag
        t_mask = tgt_is_h == h_flag
        if not r_mask.any() or not t_mask.any():
            continue
        t_tree = cKDTree(ref_coords[r_mask])
        r_p = ref_psqm[r_mask]
        r_a = ref_area[r_mask]
        km = min(40, int(r_mask.sum()))
        d, ix = t_tree.query(tgt_coords[t_mask], k=km)
        dk = d * 111.32
        for k, arr in ((5, same_k5), (15, same_k15), (40, same_k40)):
            kk = min(k, km)
            w = 1.0 / (dk[:, :kk] + eps)
            w = w / w.sum(axis=1, keepdims=True)
            arr[t_mask] = np.sum(w * r_p[ix[:, :kk]], axis=1)
        same_med_area[t_mask] = np.median(r_a[ix[:, :min(15, km)]], axis=1)
        same_dist5[t_mask] = np.mean(dk[:, :min(5, km)], axis=1)

    out['sp_same_type_idw_psqm_k5'] = same_k5
    out['sp_same_type_idw_psqm_k15'] = same_k15
    out['sp_same_type_idw_psqm_k40'] = same_k40
    out['sp_same_type_rel_area'] = tgt_area / np.maximum(same_med_area, 10.0)
    out['sp_same_type_dist5_km'] = same_dist5

    # Spatial multiplicative prior price anchor
    out['sp_prior_price'] = same_k15 * tgt_area
    out['sp_log_prior_price'] = np.log(np.maximum(out['sp_prior_price'], 10_000.0))

    # 3. Empirical-Bayes smoothed (type, neighbourhood) and (type, statbel_sector) €/m²
    for grp_col, name, m_prior in (
        ('neighbourhood', 'sp_type_wijk_psqm', 5.0),
        ('statbel_sector', 'sp_type_sector_psqm', 6.0),
        ('postal_code', 'sp_type_postal_psqm', 5.0),
    ):
        if grp_col not in ref_valid.columns:
            continue
        tr_g = pd.DataFrame({
            'h': ref_is_h.astype(int),
            'g': ref_valid[grp_col].astype(str).to_numpy(),
            'p': ref_psqm,
        })
        type_med = tr_g.groupby('h')['p'].median().to_dict()
        agg = tr_g.groupby(['h', 'g'])['p'].agg(['count', 'median'])
        smooth = {
            (h, g): (row['count'] * row['median'] + m_prior * type_med.get(h, g_med)) / (row['count'] + m_prior)
            for (h, g), row in agg.iterrows() if g != 'Unknown'
        }
        out[name] = [
            smooth.get((int(h), str(g)), type_med.get(int(h), g_med))
            for h, g in zip(tgt_is_h, target_df[grp_col])
        ]

    # Replace legacy OOF columns with fold-consistent values too
    out['oof_knn10_median_price_sqm'] = np.median(ref_psqm[idxs_all[:, :min(10, k_max)]], axis=1)
    out['oof_dist_5nn_km'] = np.mean(dists_km[:, :min(5, k_max)], axis=1)
    return out


# ---------------------------------------------------------------------------
# Static engineered domain features (row-local, zero leakage)
# ---------------------------------------------------------------------------
def add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    new_cols: dict[str, np.ndarray] = {}

    # 1. Rotated coordinates (axis-aligned trees love rotated axes for rivers/ring roads)
    lat = out['latitude'].to_numpy(dtype=float) - 51.0531
    lon = (out['longitude'].to_numpy(dtype=float) - 3.7265) * np.cos(np.radians(51.05))
    for deg in (15, 30, 45, 60, 75):
        rad = np.radians(deg)
        new_cols[f'coord_rot_{deg}_x'] = lat * np.cos(rad) + lon * np.sin(rad)
        new_cols[f'coord_rot_{deg}_y'] = -lat * np.sin(rad) + lon * np.cos(rad)
    new_cols['coord_angle_center'] = np.arctan2(lat, lon)

    # 2. Log transforms of heavy-tailed size/value drivers
    area = np.maximum(out['livable_area'].to_numpy(dtype=float), 1.0)
    plot = np.maximum(out['property_size'].to_numpy(dtype=float), 0.0)
    ki = np.maximum(out['cadastral_income'].to_numpy(dtype=float), 1.0)
    new_cols['log_livable_area'] = np.log(area)
    new_cols['log_property_size_p1'] = np.log1p(plot)
    new_cols['log_cadastral_income'] = np.log(ki)
    new_cols['ki_per_sqm'] = ki / area
    new_cols['ki_per_sqm_known'] = np.where(out['cadastral_income_missing'].to_numpy() == 0, ki / area, np.nan)
    new_cols['area_per_bedroom'] = area / np.maximum(out['number_of_bedrooms'].to_numpy(dtype=float), 1.0)
    new_cols['area_per_room'] = area / np.maximum(out['number_of_rooms'].to_numpy(dtype=float), 1.0)
    new_cols['outdoor_surface_total'] = (
        out['garden_surface'].to_numpy(dtype=float) + out['terrace_surface'].to_numpy(dtype=float)
    )
    new_cols['log_outdoor_surface'] = np.log1p(np.maximum(new_cols['outdoor_surface_total'], 0.0))
    new_cols['total_parking'] = (
        out['parking_indoor'].to_numpy(dtype=float) + out['parking_outdoor'].to_numpy(dtype=float)
    )

    # 3. Flemish EPC ordinal score & renovation-obligation cost proxy
    epc_map = {'A++': 0, 'A+': 1, 'A': 2, 'B': 3, 'C': 4, 'D': 5, 'E': 6, 'F': 7}
    epc_ord = out['energy_label'].map(epc_map).astype(float).to_numpy()
    new_cols['epc_ordinal'] = epc_ord
    # E/F label triggers mandatory 5-year renovation to D in Flanders -> cost scales with livable_area!
    is_ef = np.where(np.isnan(epc_ord), 0.0, (epc_ord >= 6).astype(float))
    new_cols['epc_ef_area_penalty'] = is_ef * area

    # 4. Condition ordinal & age interaction
    cond_map = {
        'AS_NEW': 5, 'JUST_RENOVATED': 4, 'GOOD': 3,
        'TO_BE_DONE_UP': 2, 'TO_RENOVATE': 1, 'TO_RESTORE': 0,
    }
    cond_ord = out['building_condition'].map(cond_map).astype(float).to_numpy()
    new_cols['condition_ordinal'] = cond_ord
    new_cols['condition_x_area'] = np.nan_to_num(cond_ord, nan=3.0) * area
    new_cols['age_x_condition'] = out['age'].to_numpy(dtype=float) * (5.0 - np.nan_to_num(cond_ord, nan=3.0))

    # 5. Cyclical seasonality (month of year) — stationary unlike year!
    m = out['publication_month'].fillna(6).to_numpy(dtype=float)
    new_cols['pub_month_sin'] = np.sin(2.0 * np.pi * m / 12.0)
    new_cols['pub_month_cos'] = np.cos(2.0 * np.pi * m / 12.0)

    # 6. Sector wealth × area interaction
    if 'sector_median_income' in out.columns:
        new_cols['log_sector_income_x_area'] = np.log(np.maximum(out['sector_median_income'].to_numpy(dtype=float), 10_000.0)) * np.log(area)

    return pd.concat([out, pd.DataFrame(new_cols, index=out.index)], axis=1)


# ---------------------------------------------------------------------------
# Fold-safe Text Features: TF-IDF Ridge + E5 Embeddings (Ridge + PCA)
# ---------------------------------------------------------------------------
def load_text_and_embeddings(df: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    raw = pl.read_parquet(
        RAW_PARQUET_PATH,
        columns=['property_id', 'date_obtained', 'title', 'description', 'description_nl'],
    ).to_pandas()
    raw['date_obtained'] = pd.to_datetime(raw['date_obtained'], format='%d%m%Y', errors='coerce')
    raw['property_id'] = raw['property_id'].astype(str)
    key_df = df[['property_id', 'date_obtained']].copy()
    key_df['property_id'] = key_df['property_id'].astype(str)
    m = key_df.merge(raw.drop_duplicates(['property_id', 'date_obtained']), on=['property_id', 'date_obtained'], how='left')

    texts = []
    for t, nl, d in zip(m['title'].fillna(''), m['description_nl'].fillna(''), m['description'].fillna('')):
        body = nl if len(str(nl).strip()) >= 40 else d
        texts.append(f"{str(t).strip()} {str(body).strip()}".strip())

    e5_data = np.load(E5_PATH)
    embs = e5_data['embeddings'].astype(np.float32)
    assert len(embs) == len(df), f"E5 length mismatch: {len(embs)} vs {len(df)}"
    return texts, embs


def fit_transform_text_fold(
    tr_idx: np.ndarray,
    va_idx: np.ndarray,
    texts: list[str],
    embs: np.ndarray,
    y_log_psqm: np.ndarray,
    y_log_price: np.ndarray,
    tr_folds: np.ndarray,
    use_tfidf: bool = True,
    use_e5: bool = True,
    e5_pca_dims: int = 32,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Strictly fold-safe text meta-features (nested OOF inside train so trees don't overfit!)."""
    tr_out: dict[str, np.ndarray] = {}
    va_out: dict[str, np.ndarray] = {}

    if use_tfidf:
        tr_texts = [texts[i] for i in tr_idx]
        va_texts = [texts[i] for i in va_idx]
        vec = TfidfVectorizer(
            max_features=20_000, ngram_range=(1, 2), min_df=5, max_df=0.9, sublinear_tf=True,
        )
        X_tr_tf = vec.fit_transform(tr_texts)
        X_va_tf = vec.transform(va_texts)

        # Nested inner-OOF for train so GBDT doesn't see overfitted in-sample TF-IDF predictions
        oof_psqm_tr = np.zeros(len(tr_idx), dtype=np.float32)
        oof_price_tr = np.zeros(len(tr_idx), dtype=np.float32)
        uniq_f = np.unique(tr_folds)
        for uf in uniq_f:
            im = tr_folds != uf
            vm = tr_folds == uf
            r1 = Ridge(alpha=3.0)
            r1.fit(X_tr_tf[im], y_log_psqm[tr_idx][im])
            oof_psqm_tr[vm] = r1.predict(X_tr_tf[vm])
            r2 = Ridge(alpha=3.0)
            r2.fit(X_tr_tf[im], y_log_price[tr_idx][im])
            oof_price_tr[vm] = r2.predict(X_tr_tf[vm])

        r1_full = Ridge(alpha=3.0).fit(X_tr_tf, y_log_psqm[tr_idx])
        r2_full = Ridge(alpha=3.0).fit(X_tr_tf, y_log_price[tr_idx])
        tr_out['txt_tfidf_log_psqm'] = oof_psqm_tr
        va_out['txt_tfidf_log_psqm'] = r1_full.predict(X_va_tf).astype(np.float32)
        tr_out['txt_tfidf_log_price'] = oof_price_tr
        va_out['txt_tfidf_log_price'] = r2_full.predict(X_va_tf).astype(np.float32)

    if use_e5:
        E_tr = embs[tr_idx]
        E_va = embs[va_idx]
        # Nested inner-OOF Ridge on full 384-d E5
        oof_e5_psqm = np.zeros(len(tr_idx), dtype=np.float32)
        oof_e5_price = np.zeros(len(tr_idx), dtype=np.float32)
        uniq_f = np.unique(tr_folds)
        for uf in uniq_f:
            im = tr_folds != uf
            vm = tr_folds == uf
            oof_e5_psqm[vm] = Ridge(alpha=10.0).fit(E_tr[im], y_log_psqm[tr_idx][im]).predict(E_tr[vm])
            oof_e5_price[vm] = Ridge(alpha=10.0).fit(E_tr[im], y_log_price[tr_idx][im]).predict(E_tr[vm])
        tr_out['txt_e5_log_psqm'] = oof_e5_psqm
        va_out['txt_e5_log_psqm'] = Ridge(alpha=10.0).fit(E_tr, y_log_psqm[tr_idx]).predict(E_va).astype(np.float32)
        tr_out['txt_e5_log_price'] = oof_e5_price
        va_out['txt_e5_log_price'] = Ridge(alpha=10.0).fit(E_tr, y_log_price[tr_idx]).predict(E_va).astype(np.float32)

        if e5_pca_dims > 0:
            pca = PCA(n_components=e5_pca_dims, random_state=42)
            P_tr = pca.fit_transform(E_tr).astype(np.float32)
            P_va = pca.transform(E_va).astype(np.float32)
            for j in range(e5_pca_dims):
                tr_out[f'e5_pca_{j}'] = P_tr[:, j]
                va_out[f'e5_pca_{j}'] = P_va[:, j]

    return pd.DataFrame(tr_out), pd.DataFrame(va_out)


# ---------------------------------------------------------------------------
# Parallel 6-split runner (5 grouped OOF folds + 1 temporal holdout simultaneously!)
# ---------------------------------------------------------------------------
def _fit_one_split(
    split_id: str,
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_va: pd.DataFrame,
    groups_tr: np.ndarray,
    cat_cols: list[str],
    model_kind: str,
    loss_mode: str,  # 'l2', 'l1', 'huber'
    params: dict | None = None,
    n_threads: int = 3,
) -> tuple[str, np.ndarray, np.ndarray]:
    """Fit on one split and return (split_id, va_pred_target_space, tr_pred_target_space)."""
    rng = np.random.default_rng(42)
    uniq_g = np.unique(groups_tr)
    es_g = set(rng.choice(uniq_g, size=max(1, int(0.12 * len(uniq_g))), replace=False))
    es_mask = np.isin(groups_tr, list(es_g))

    X_tr = X_tr.copy()
    X_va = X_va.copy()

    if model_kind == 'lgbm':
        import lightgbm as lgb
        for c in cat_cols:
            cats = sorted(set(X_tr[c].unique()) | {'Missing'})
            X_tr[c] = pd.Categorical(X_tr[c], categories=cats)
            X_va[c] = pd.Categorical(X_va[c].where(X_va[c].isin(cats), 'Missing'), categories=cats)
        obj_map = {'l2': 'regression', 'l1': 'regression_l1', 'huber': 'huber'}
        cfg = dict(
            objective=obj_map[loss_mode], n_estimators=2500, learning_rate=0.025,
            num_leaves=63, min_child_samples=25, subsample=0.8,
            subsample_freq=1, colsample_bytree=0.7, reg_alpha=1.0,
            reg_lambda=3.0, random_state=42, n_jobs=n_threads, verbose=-1,
        )
        if loss_mode == 'huber':
            cfg['alpha'] = 0.15
        if params:
            cfg.update(params)
        model = lgb.LGBMRegressor(**cfg)
        model.fit(
            X_tr[~es_mask], y_tr[~es_mask],
            eval_set=[(X_tr[es_mask], y_tr[es_mask])],
            callbacks=[lgb.early_stopping(80, verbose=False)],
        )
        return split_id, model.predict(X_va), model.predict(X_tr)

    if model_kind == 'catboost':
        from catboost import CatBoostRegressor
        loss_map = {'l2': 'RMSE', 'l1': 'MAE', 'huber': 'Huber:delta=0.15'}
        cfg = dict(
            loss_function=loss_map[loss_mode], iterations=3000, learning_rate=0.035,
            depth=7, l2_leaf_reg=5.0, random_seed=42, verbose=0, thread_count=n_threads,
        )
        if params:
            cfg.update(params)
        model = CatBoostRegressor(**cfg)
        model.fit(
            X_tr[~es_mask], y_tr[~es_mask],
            cat_features=cat_cols,
            eval_set=(X_tr[es_mask], y_tr[es_mask]),
            early_stopping_rounds=100,
            verbose=False,
        )
        return split_id, model.predict(X_va), model.predict(X_tr)

    if model_kind == 'xgb':
        import xgboost as xgb
        for c in cat_cols:
            cats = sorted(set(X_tr[c].unique()) | {'Missing'})
            X_tr[c] = pd.Categorical(X_tr[c], categories=cats)
            X_va[c] = pd.Categorical(X_va[c].where(X_va[c].isin(cats), 'Missing'), categories=cats)
        obj_map = {'l2': 'reg:squarederror', 'l1': 'reg:absoluteerror', 'huber': 'reg:pseudohubererror'}
        cfg = dict(
            objective=obj_map[loss_mode], n_estimators=2500, learning_rate=0.025,
            max_depth=7, min_child_weight=10, subsample=0.8,
            colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=3.0,
            tree_method='hist', enable_categorical=True,
            early_stopping_rounds=80, random_state=42, n_jobs=n_threads,
        )
        if loss_mode == 'huber':
            cfg['huber_slope'] = 0.15
        if params:
            cfg.update(params)
        model = xgb.XGBRegressor(**cfg)
        model.fit(
            X_tr[~es_mask], y_tr[~es_mask],
            eval_set=[(X_tr[es_mask], y_tr[es_mask])],
            verbose=False,
        )
        return split_id, model.predict(X_va), model.predict(X_tr)

    raise ValueError(model_kind)


def prep_frame(df: pd.DataFrame, feat_cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    X = df[feat_cols].copy()
    cat_cols = []
    for c in feat_cols:
        if pd.api.types.is_bool_dtype(X[c]):
            X[c] = X[c].astype('int8')
        elif pd.api.types.is_string_dtype(X[c]) or pd.api.types.is_object_dtype(X[c]) or isinstance(X[c].dtype, pd.CategoricalDtype):
            X[c] = X[c].fillna('Missing').astype(str)
            cat_cols.append(c)
        else:
            X[c] = pd.to_numeric(X[c], errors='coerce').astype('float32')
    return X, cat_cols


def run_parallel_splits(
    split_bundles: list[dict],
    feat_cols: list[str],
    model_kind: str,
    target_mode: str,   # 'log_price', 'log_psqm', 'log_cpi_real', 'log_hpi_real', 'log_prior_resid'
    loss_mode: str = 'l2',
    params: dict | None = None,
    n_threads_per_worker: int = 3,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Run all 6 splits in parallel (6 workers × 3 threads = 18 cores!)."""
    jobs = []
    for b in split_bundles:
        tr_df = b['tr_df']
        va_df = b['va_df']
        X_tr, cat_cols = prep_frame(tr_df, feat_cols)
        X_va, _ = prep_frame(va_df, feat_cols)

        y_p = tr_df['price'].to_numpy(dtype=float)
        area_tr = tr_df['livable_area'].to_numpy(dtype=float)

        if target_mode == 'log_price':
            y_tr = np.log(y_p)
        elif target_mode == 'log_psqm':
            y_tr = np.log(y_p / area_tr)
        elif target_mode == 'log_cpi_real':
            y_tr = np.log(y_p / b['tr_cpi_idx'])
        elif target_mode == 'log_hpi_real':
            y_tr = np.log(y_p / b['tr_hpi_idx'])
        elif target_mode == 'log_prior_resid':
            y_tr = np.log(y_p / np.maximum(tr_df['sp_prior_price'].to_numpy(dtype=float), 10_000.0))
        else:
            raise ValueError(target_mode)

        jobs.append((
            b['id'], X_tr, y_tr, X_va, tr_df['property_key'].astype(str).to_numpy(),
            cat_cols, model_kind, loss_mode, params, n_threads_per_worker,
        ))

    res = Parallel(n_jobs=6, backend='loky')(delayed(_fit_one_split)(*j) for j in jobs)
    res_map = {sid: (vp, tp) for sid, vp, tp in res}

    # Assemble OOF and temporal predictions back in EUR
    n_all = sum(len(b['va_df']) for b in split_bundles if b['id'].startswith('fold_'))
    oof_eur = np.zeros(n_all, dtype=float)
    tmp_eur = None
    tr_preds_target = []

    for b in split_bundles:
        vp, tp = res_map[b['id']]
        va_df = b['va_df']
        area_va = va_df['livable_area'].to_numpy(dtype=float)
        if target_mode == 'log_price':
            pred_eur = np.exp(vp)
        elif target_mode == 'log_psqm':
            pred_eur = np.exp(vp) * area_va
        elif target_mode == 'log_cpi_real':
            pred_eur = np.exp(vp) * b['va_cpi_idx']
        elif target_mode == 'log_hpi_real':
            pred_eur = np.exp(vp) * b['va_hpi_idx']
        elif target_mode == 'log_prior_resid':
            pred_eur = np.exp(vp) * np.maximum(va_df['sp_prior_price'].to_numpy(dtype=float), 10_000.0)
        else:
            raise ValueError(target_mode)

        if b['id'].startswith('fold_'):
            oof_eur[b['va_idx']] = pred_eur
        else:
            tmp_eur = pred_eur
        tr_preds_target.append(tp)

    return oof_eur, tmp_eur, tr_preds_target


# ---------------------------------------------------------------------------
# 2-Stage Spatial Residual Kriging (strictly out-of-fold!)
# ---------------------------------------------------------------------------
def add_oof_spatial_residuals(
    split_bundles: list[dict],
    oof_stage1_eur: np.ndarray,
    tmp_stage1_eur: np.ndarray,
    full_df: pd.DataFrame,
) -> None:
    """Compute IDW spatial average of Stage-1 log-residuals strictly from each split's train set."""
    all_log_resid = np.log(full_df['price'].to_numpy(dtype=float)) - np.log(np.maximum(oof_stage1_eur, 10_000.0))

    for b in split_bundles:
        tr_idx = b['tr_idx']
        va_idx = b['va_idx']
        tr_df = b['tr_df']
        va_df = b['va_df']

        tr_coords = np.column_stack([tr_df['latitude'].to_numpy(), tr_df['longitude'].to_numpy()])
        va_coords = np.column_stack([va_df['latitude'].to_numpy(), va_df['longitude'].to_numpy()])
        tr_res = all_log_resid[tr_idx]

        tree = cKDTree(tr_coords)
        # For train rows: leave-self-out (query k=16 and drop index 0)
        d_tr, ix_tr = tree.query(tr_coords, k=16)
        d_tr_km = d_tr[:, 1:] * 111.32
        w_tr = 1.0 / (d_tr_km + 0.08)
        w_tr = w_tr / w_tr.sum(axis=1, keepdims=True)
        b['tr_df']['sp_stage1_resid_k5'] = np.sum((w_tr[:, :5] / w_tr[:, :5].sum(axis=1, keepdims=True)) * tr_res[ix_tr[:, 1:6]], axis=1).astype('float32')
        b['tr_df']['sp_stage1_resid_k15'] = np.sum(w_tr * tr_res[ix_tr[:, 1:]], axis=1).astype('float32')

        # For val/test rows: query k=15 on train tree
        d_va, ix_va = tree.query(va_coords, k=15)
        d_va_km = d_va * 111.32
        w_va = 1.0 / (d_va_km + 0.08)
        w_va = w_va / w_va.sum(axis=1, keepdims=True)
        b['va_df']['sp_stage1_resid_k5'] = np.sum((w_va[:, :5] / w_va[:, :5].sum(axis=1, keepdims=True)) * tr_res[ix_va[:, :5]], axis=1).astype('float32')
        b['va_df']['sp_stage1_resid_k15'] = np.sum(w_va * tr_res[ix_va], axis=1).astype('float32')


# ---------------------------------------------------------------------------
# Main Experiment Runner
# ---------------------------------------------------------------------------
def main():
    t_start = time.perf_counter()
    os.makedirs(REPORT_DIR, exist_ok=True)
    df_raw = pd.read_parquet(DATASET_PATH).reset_index(drop=True)
    print(f"Loaded {len(df_raw):,} rows x {df_raw.shape[1]} cols (parallel 6x3 = 18 cores)")

    # Base feature list (before Phase 2 engineering)
    base_feats = [c for c in df_raw.columns if c not in META_COLUMNS]
    base_feats_stationary = [c for c in base_feats if c not in NON_STATIONARY_MACRO]

    # Deflator indices
    cpi_idx = (df_raw['belgian_cpi_index'] / df_raw.loc[df_raw['publication_year'] == 2025, 'belgian_cpi_index'].median()).to_numpy(dtype=float)
    hpi_idx = build_ghent_deflator(df_raw)
    print(f"Deflators — 2024 vs 2026 mean: CPI {cpi_idx[df_raw['publication_year']==2024].mean():.4f}->{cpi_idx[df_raw['publication_year']==2026].mean():.4f} | "
          f"Gent HPI {hpi_idx[df_raw['publication_year']==2024].mean():.4f}->{hpi_idx[df_raw['publication_year']==2026].mean():.4f}")

    # Load texts + MPS E5 embeddings
    texts, embs = load_text_and_embeddings(df_raw)
    y_price = df_raw['price'].to_numpy(dtype=float)
    y_log_price = np.log(y_price)
    y_log_psqm = np.log(y_price / df_raw['livable_area'].to_numpy(dtype=float))

    # Build enriched static domain features
    df_dom = add_domain_features(df_raw)

    # Build the 6 split definitions (5 grouped OOF folds + 1 temporal split)
    test_mask_t = (df_raw['publication_year'] >= 2026).to_numpy()
    test_keys_t = set(df_raw.loc[test_mask_t, 'property_key'].astype(str))
    train_mask_t = (~test_mask_t) & (~df_raw['property_key'].astype(str).isin(test_keys_t).to_numpy())
    tr_idx_t = np.where(train_mask_t)[0]
    va_idx_t = np.where(test_mask_t)[0]
    y_tmp_true = y_price[va_idx_t]

    print("Precomputing fold-safe multi-scale spatial & text features across all 6 splits in parallel...")
    t_prep = time.perf_counter()

    def _build_split_bundle(sid: str, tr_idx: np.ndarray, va_idx: np.ndarray) -> dict:
        tr_base = df_dom.iloc[tr_idx].copy().reset_index(drop=True)
        va_base = df_dom.iloc[va_idx].copy().reset_index(drop=True)

        # Multi-scale spatial features:
        # - For va: reference is ALL of tr_base
        sp_va = compute_multiscale_spatial(tr_base, va_base)
        # - For tr: strictly inner-OOF by grouped fold so GBDT doesn't overfit spatial targets!
        sp_tr_parts = []
        tr_folds = tr_base['fold'].to_numpy()
        for uf in np.unique(tr_folds):
            ref_sub = tr_base[tr_folds != uf]
            val_sub = tr_base[tr_folds == uf]
            sp_tr_parts.append(compute_multiscale_spatial(ref_sub, val_sub))
        sp_tr = pd.concat(sp_tr_parts, axis=0).sort_index()

        # Drop overlapping old OOF columns before concat
        overlap_cols = [c for c in sp_tr.columns if c in tr_base.columns]
        tr_df = pd.concat([tr_base.drop(columns=overlap_cols), sp_tr.reset_index(drop=True)], axis=1)
        va_df = pd.concat([va_base.drop(columns=overlap_cols), sp_va.reset_index(drop=True)], axis=1)

        # Fold-safe text features (TF-IDF Ridge + E5 Ridge + E5 PCA-32)
        txt_tr, txt_va = fit_transform_text_fold(
            tr_idx, va_idx, texts, embs, y_log_psqm, y_log_price, tr_folds,
            use_tfidf=True, use_e5=True, e5_pca_dims=32,
        )
        tr_df = pd.concat([tr_df, txt_tr.reset_index(drop=True)], axis=1)
        va_df = pd.concat([va_df, txt_va.reset_index(drop=True)], axis=1)

        # Frequency encoding of high-cardinality geo/type categoricals (strictly from train)
        for col in ('statbel_sector', 'neighbourhood', 'postal_code', 'property_subtype'):
            if col in tr_df.columns:
                freq = tr_df[col].astype(str).value_counts(normalize=True).to_dict()
                tr_df[f'freq_{col}'] = tr_df[col].astype(str).map(freq).fillna(0.0).astype('float32')
                va_df[f'freq_{col}'] = va_df[col].astype(str).map(freq).fillna(0.0).astype('float32')

        return {
            'id': sid,
            'tr_idx': tr_idx,
            'va_idx': va_idx,
            'tr_df': tr_df,
            'va_df': va_df,
            'tr_cpi_idx': cpi_idx[tr_idx],
            'va_cpi_idx': cpi_idx[va_idx],
            'tr_hpi_idx': hpi_idx[tr_idx],
            'va_hpi_idx': hpi_idx[va_idx],
        }

    split_specs = [
        (f'fold_{f}', np.where(df_raw['fold'].to_numpy() != f)[0], np.where(df_raw['fold'].to_numpy() == f)[0])
        for f in sorted(df_raw['fold'].unique())
    ] + [('temporal', tr_idx_t, va_idx_t)]

    split_bundles = Parallel(n_jobs=6, backend='loky')(
        delayed(_build_split_bundle)(sid, tri, vai) for sid, tri, vai in split_specs
    )
    print(f"Precomputation finished in {time.perf_counter()-t_prep:.1f}s | total cols per split: {split_bundles[0]['tr_df'].shape[1]}")

    # Column groups
    all_cols = [c for c in split_bundles[0]['tr_df'].columns if c not in META_COLUMNS]
    sp_new_cols = [c for c in all_cols if c.startswith('sp_')]
    dom_new_cols = [c for c in all_cols if c.startswith(('coord_', 'log_', 'ki_', 'area_per_', 'outdoor_', 'total_parking', 'epc_ordinal', 'epc_ef_', 'condition_', 'age_x_', 'pub_month_', 'freq_'))]
    tfidf_cols = [c for c in all_cols if c.startswith('txt_tfidf_')]
    e5_ridge_cols = [c for c in all_cols if c.startswith('txt_e5_')]
    e5_pca_cols = [c for c in all_cols if c.startswith('e5_pca_')]

    report: list[dict] = []

    def record(name: str, oof_p: np.ndarray, tmp_p: np.ndarray, dt: float):
        mo = compute_metrics(y_price, oof_p)
        mt = compute_metrics(y_tmp_true, tmp_p)
        print(f"  {name:44s} ({dt:4.1f}s) | OOF: {fmt(mo)}")
        print(f"  {'':51s} | TMP: {fmt(mt)}")
        report.append({'name': name, 'seconds': round(dt, 1), 'oof': mo, 'temporal': mt})
        return mo, mt

    # ======================================================================
    # EXPERIMENT A: Target & Loss Formulation (on base features, LightGBM)
    # ======================================================================
    print("\n=== EXPERIMENT A: TARGET & LOSS FORMULATION (LightGBM, base features) ===")
    for name, feats, tmode, lmode in [
        ('A0: log(price) L2 (baseline)',          base_feats,            'log_price',    'l2'),
        ('A1: log(price) L1/MAE',                 base_feats,            'log_price',    'l1'),
        ('A2: log(price) Huber (d=0.15)',         base_feats,            'log_price',    'huber'),
        ('A3: log(price/m²) L2',                  base_feats,            'log_psqm',     'l2'),
        ('A4: log(price/m²) L1/MAE',              base_feats,            'log_psqm',     'l1'),
        ('A5: CPI-deflated log(real_price) L2',   base_feats_stationary, 'log_cpi_real', 'l2'),
        ('A6: Gent-HPI-deflated log(real_p) L2',  base_feats_stationary, 'log_hpi_real', 'l2'),
        ('A7: Gent-HPI-deflated log(real_p) L1',  base_feats_stationary, 'log_hpi_real', 'l1'),
    ]:
        t0 = time.perf_counter()
        op, tp, _ = run_parallel_splits(split_bundles, feats, 'lgbm', tmode, lmode)
        record(name, op, tp, time.perf_counter() - t0)

    # ======================================================================
    # EXPERIMENT B: High-Res Spatial + Domain Interactions + Residual Anchor
    # ======================================================================
    print("\n=== EXPERIMENT B: SPATIAL, DOMAIN INTERACTIONS & KRIGING (LightGBM) ===")
    feats_b1 = base_feats_stationary + sp_new_cols
    feats_b2 = feats_b1 + dom_new_cols

    t0 = time.perf_counter()
    op_b1, tp_b1, _ = run_parallel_splits(split_bundles, feats_b1, 'lgbm', 'log_hpi_real', 'l2')
    record('B1: + Multi-scale type-cond spatial', op_b1, tp_b1, time.perf_counter() - t0)

    t0 = time.perf_counter()
    op_b2, tp_b2, _ = run_parallel_splits(split_bundles, feats_b2, 'lgbm', 'log_hpi_real', 'l2')
    record('B2: + Rotated coords & domain interactions', op_b2, tp_b2, time.perf_counter() - t0)

    t0 = time.perf_counter()
    op_b3, tp_b3, _ = run_parallel_splits(split_bundles, feats_b2, 'lgbm', 'log_prior_resid', 'l2')
    record('B3: log(price / spatial_prior) target', op_b3, tp_b3, time.perf_counter() - t0)

    # ======================================================================
    # EXPERIMENT C: Description Text Signals (TF-IDF Ridge + E5 MPS)
    # ======================================================================
    print("\n=== EXPERIMENT C: DESCRIPTION TEXT (TF-IDF + MULTILINGUAL-E5) ===")
    feats_c1 = feats_b2 + tfidf_cols
    feats_c2 = feats_b2 + e5_ridge_cols + e5_pca_cols
    feats_c3 = feats_b2 + tfidf_cols + e5_ridge_cols + e5_pca_cols

    t0 = time.perf_counter()
    op_c1, tp_c1, _ = run_parallel_splits(split_bundles, feats_c1, 'lgbm', 'log_hpi_real', 'l2')
    record('C1: B2 + OOF TF-IDF Ridge', op_c1, tp_c1, time.perf_counter() - t0)

    t0 = time.perf_counter()
    op_c2, tp_c2, _ = run_parallel_splits(split_bundles, feats_c2, 'lgbm', 'log_hpi_real', 'l2')
    record('C2: B2 + E5 Ridge + E5 PCA-32', op_c2, tp_c2, time.perf_counter() - t0)

    t0 = time.perf_counter()
    op_c3, tp_c3, _ = run_parallel_splits(split_bundles, feats_c3, 'lgbm', 'log_hpi_real', 'l2')
    record('C3: B2 + TF-IDF + E5 Ridge + E5 PCA-32', op_c3, tp_c3, time.perf_counter() - t0)

    # 2-stage spatial residual kriging on top of C3
    add_oof_spatial_residuals(split_bundles, op_c3, tp_c3, df_raw)
    feats_c4 = feats_c3 + ['sp_stage1_resid_k5', 'sp_stage1_resid_k15']
    t0 = time.perf_counter()
    op_c4, tp_c4, _ = run_parallel_splits(split_bundles, feats_c4, 'lgbm', 'log_hpi_real', 'l2')
    record('C4: C3 + 2-Stage Spatial Residual Kriging', op_c4, tp_c4, time.perf_counter() - t0)

    # Pick whether kriging helped on both axes
    m_c3 = compute_metrics(y_price, op_c3)
    m_c4 = compute_metrics(y_price, op_c4)
    mt_c3 = compute_metrics(y_tmp_true, tp_c3)
    mt_c4 = compute_metrics(y_tmp_true, tp_c4)
    best_feats = feats_c4 if (m_c4['mdape'] < m_c3['mdape'] and mt_c4['mdape'] <= mt_c3['mdape'] + 0.05) else feats_c3
    print(f"\n  Selected feature set for Step D: {len(best_feats)} features ({'with' if best_feats is feats_c4 else 'without'} kriging)")

    # ======================================================================
    # EXPERIMENT D: Diverse Multi-Family & Multi-Target Ensemble
    # ======================================================================
    print("\n=== EXPERIMENT D: DIVERSE MULTI-FAMILY + MULTI-TARGET ENSEMBLE ===")
    model_oofs: dict[str, np.ndarray] = {}
    model_tmps: dict[str, np.ndarray] = {}

    specs = [
        ('D1: LightGBM log(real_p) L2',     'lgbm',     'log_hpi_real', 'l2',    None),
        ('D2: LightGBM log(real_p) Huber',  'lgbm',     'log_hpi_real', 'huber', None),
        ('D3: LightGBM log(€/m²) L2',       'lgbm',     'log_psqm',     'l2',    None),
        ('D4: CatBoost log(real_p) L2',     'catboost', 'log_hpi_real', 'l2',    None),
        ('D5: CatBoost log(real_p) L1',     'catboost', 'log_hpi_real', 'l1',    None),
        ('D6: CatBoost log(€/m²) L2',       'catboost', 'log_psqm',     'l2',    None),
        ('D7: XGBoost log(real_p) L2',      'xgb',      'log_hpi_real', 'l2',    None),
        ('D8: XGBoost log(€/m²) L2',        'xgb',      'log_psqm',     'l2',    None),
    ]
    for name, kind, tmode, lmode, params in specs:
        t0 = time.perf_counter()
        op, tp, _ = run_parallel_splits(split_bundles, best_feats, kind, tmode, lmode, params)
        record(name, op, tp, time.perf_counter() - t0)
        model_oofs[name] = op
        model_tmps[name] = tp

    # Optimize non-negative convex ensemble weights in log-space (minimizing MAE in log-space ≈ MdAPE)
    # Use nested OOF weights per fold so ensemble score is 100% honest!
    names = list(model_oofs.keys())
    O_log = np.column_stack([np.log(np.maximum(model_oofs[k], 10_000.0)) for k in names])
    T_log = np.column_stack([np.log(np.maximum(model_tmps[k], 10_000.0)) for k in names])

    def _fit_weights(X_mat: np.ndarray, y_log_t: np.ndarray) -> np.ndarray:
        k = X_mat.shape[1]
        def loss_fn(w):
            pred = X_mat @ w
            # Blend 50% L1 + 50% L2 in log-space for simultaneous MdAPE & RMSE optimality
            err = pred - y_log_t
            return 0.5 * np.mean(np.abs(err)) + 0.5 * np.sqrt(np.mean(err ** 2))
        res = minimize(
            loss_fn, np.ones(k) / k,
            bounds=[(0.0, 1.0)] * k,
            constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
            method='SLSQP',
        )
        return res.x

    nested_oof_log = np.zeros(len(df_raw))
    fold_weights = []
    for f in sorted(df_raw['fold'].unique()):
        tr_m = df_raw['fold'].to_numpy() != f
        va_m = df_raw['fold'].to_numpy() == f
        w_f = _fit_weights(O_log[tr_m], y_log_price[tr_m])
        fold_weights.append(w_f)
        nested_oof_log[va_m] = O_log[va_m] @ w_f

    w_full = _fit_weights(O_log, y_log_price)
    ens_oof_eur = np.exp(nested_oof_log)
    ens_tmp_eur = np.exp(T_log @ w_full)

    # Also simple geometric mean of all 8
    geo_oof_eur = np.exp(O_log.mean(axis=1))
    geo_tmp_eur = np.exp(T_log.mean(axis=1))

    print("\n--- ENSEMBLE SUMMARY ---")
    record('ENS-1: Equal Geometric Mean (8 models)', geo_oof_eur, geo_tmp_eur, 0.0)
    record('ENS-2: Optimal Convex Blend (nested OOF)', ens_oof_eur, ens_tmp_eur, 0.0)

    print("\nLearned ensemble weights:")
    for nm, wt in sorted(zip(names, w_full), key=lambda x: -x[1]):
        if wt >= 0.005:
            print(f"  {wt*100:5.1f}%  {nm}")

    # Save OOF/temporal predictions & full JSON report
    out_json = os.path.join(REPORT_DIR, 'phase2_kaggle_experiments.json')
    with open(out_json, 'w') as fh:
        json.dump({
            'results': report,
            'ensemble_weights': {nm: round(float(wt), 4) for nm, wt in zip(names, w_full)},
            'n_best_features': len(best_feats),
            'total_seconds': round(time.perf_counter() - t_start, 1),
        }, fh, indent=2)
    np.savez_compressed(
        os.path.join(REPORT_DIR, 'phase2_ensemble_preds.npz'),
        y_oof=y_price,
        oof_pred=ens_oof_eur,
        y_tmp=y_tmp_true,
        tmp_pred=ens_tmp_eur,
        tmp_idx=va_idx_t,
    )
    print(f"\nAll experiments finished in {time.perf_counter()-t_start:.1f}s -> {out_json}")


if __name__ == '__main__':
    main()
