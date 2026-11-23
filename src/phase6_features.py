"""Phase-6 Unified Feature Factory — Clean, Leak-Free, Normal-Home-First!

Combines ALL winning research outcomes from Issues #1, #2, #3, #4, #9 into ONE canonical
strictly leak-free 6-split feature cache (`outputs/models/workbench/phase6_splits_cache.joblib`):

  1. **Issue #4 — Anti-Chaining Spatial+Text Cluster CV (`fold_cluster`):**
     - ZERO leaked clusters across the 5 OOF folds (`8,354` clusters, balanced `2,920/2,920/2,920/2,920/2,919`).
     - Temporal 2026 holdout drops ANY train row sharing a `cluster_id` with 2026 (`0` leaked clusters!).
  2. **Issue #9 — Winning Temporal Deflator (`0.5 * CPI + 0.5 * HPI`):**
     - Beat raw `CPI^1.0` by `-0.23pp` MdAPE (`8.31% -> 8.08%`) and fixed Q4 inflation overshoot (`+1.78% -> -0.14%`).
  3. **Issue #2 — Winning Wave-2 Micro-Geo (compact 6 cols only, zero bloat):**
     - IRCEL 10m Air Quality (`fl_no2_2025_ugm3`, `fl_bc_2025_ugm3`, `fl_pm25_2025_ugm3`)
     - VHA Navigable Waterfront (`fl_dist_navigable_water_m`, `fl_near_water_100m`, `fl_near_water_300m`)
  4. **Issue #3 — Winning High-Residual Text & Apartment-Floor Features (13 cols):**
     - `log_txt_to_struct_m2_ratio`, `txt_struct_m2_mismatch_20pct`, `txt_epc_kwh_diff`,
       `txt_is_handelshuis`, `txt_is_casco`, `txt_is_herenhuis`, `txt_has_high_ceilings`,
       `txt_is_rented`, `txt_mentions_underfloor`, `apt_is_ground_floor`, `apt_is_top_floor`,
       `apt_high_floor_no_lift`, `apt_floor_ratio`.
  5. **Issue #1 — Full-Corpus Render-Filtered Vision Features (`SigLIP2-base` + `SO400M-384`):**
     - Automatically included when `external_data/vision_full/vision_full_embeddings.npz` exists!
     - Strictly nested inner-OOF per cluster-fold:
       * `vis_sig_real_psqm`, `vis_sig_real_tabresid` (RidgeCV on 768-d REAL-photo-filtered SigLIP2!)
       * `vis_so400m_psqm`, `vis_so400m_tabresid` (RidgeCV on 1152-d SO400M contact sheet!)
       * `vis_blend_psqm`, `vis_blend_tabresid`
       * 8 explicit Render/Media flags (`vis_cgi_margin_mean/max`, `vis_frac_cgi`, `vis_frac_vstage`, `vis_frac_any_render`, `vis_frac_clean_real`, `vis_frac_plan`, `vis_frac_shell`) — the decisive `-0.19pp` interaction key!
       * Top compact real-photo Q-scores & Room-Q (mean + room-specific kitchen/bath/living/facade/worst_fixer) + 10 SO400M Q-scores + PCA-8 on SigLIP2-real!
  6. **Upgraded Text & Tabular-Residual Meta-Features (strictly nested inner-OOF on `cluster_id`!):**
     - Word+Char TF-IDF multi-alpha Ridge + Huber SGD + SVD-16
     - E5-large (1024-d) + Qwen3-0.6B (1024-d) Ridge on `log_real_psqm`, `log_real_price`, and `r_tab` (residual of a fast structured-only LightGBM anchor!)
     - Normalized agency frequency & OOF target-encoding (fixes the 8 split-casing Ghent offices!).

All 6 splits (5 cluster folds + cluster-clean 2026 temporal) run in parallel on CPU (~45s total).

Usage:
    .venv-api/bin/python -u src/phase6_features.py
"""
from __future__ import annotations

import os
import sys
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.sparse import hstack
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, RidgeCV, SGDRegressor

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'src'))

from phase3_features import (
    META_COLUMNS,
    NON_STATIONARY_MACRO,
    _add_domain_features,
    _compute_spatial_and_te,
    _extract_text_flags,
)
from phase4_features import attach_flanders_and_grb, compute_anomaly_flag

WORK_DIR = os.path.join(BASE_DIR, 'outputs', 'models', 'workbench')
DATASET_PATH = os.path.join(BASE_DIR, 'outputs', 'dataset', 'training_dataset.parquet')
RAW_PARQUET = os.path.join(BASE_DIR, 'outputs', 'property_data_full.parquet')
CLUSTER_PATH = os.path.join(BASE_DIR, 'outputs', 'dataset', 'cluster_folds.parquet')
ISSUE3_PATH = os.path.join(BASE_DIR, 'outputs', 'dataset', 'issue3_extra_features.parquet')
W2_PATH = os.path.join(BASE_DIR, 'external_data', 'flanders', 'ghent_coord_enrichment_v2.parquet')
E5_LARGE_PATH = os.path.join(WORK_DIR, 'e5_large_14599.npz')
E5_SMALL_PATH = os.path.join(WORK_DIR, 'e5_embeddings_14599.npz')
QWEN3_PATH = os.path.join(WORK_DIR, 'qwen3_emb_14599.npz')
VIS_FULL_PATH = os.path.join(BASE_DIR, 'external_data', 'vision_full', 'vision_full_embeddings.npz')
OUT_CACHE = os.path.join(WORK_DIR, 'phase6_splits_cache.joblib')

# Strictly curated winners from Wave-2 Geo (#2) and Text/Floor (#3) ablations
KEEP_W2_COLS = [
    'fl_no2_2025_ugm3', 'fl_bc_2025_ugm3', 'fl_pm25_2025_ugm3',
    'fl_dist_navigable_water_m', 'fl_near_water_100m', 'fl_near_water_300m',
]
KEEP_ISSUE3_COLS = [
    'log_txt_to_struct_m2_ratio', 'txt_struct_m2_mismatch_20pct', 'txt_epc_kwh_diff',
    'txt_is_handelshuis', 'txt_is_casco', 'txt_is_herenhuis', 'txt_has_high_ceilings',
    'txt_is_rented', 'txt_mentions_underfloor',
    'apt_is_ground_floor', 'apt_is_top_floor', 'apt_high_floor_no_lift', 'apt_floor_ratio',
]
DROP_W2_BLOATED = [
    'fl_flood_fluvial_class', 'fl_flood_pluvial_class', 'fl_flood_max_class',
    'fl_buurtgroen_10m', 'fl_urban_heat_c', 'gent_trees_100m', 'gent_mature_trees_100m',
    'gent_tree_girth_sum_100m', 'fl_dist_motorway_m', 'fl_dist_highway_ramp_m',
    'gent_dist_tram_m', 'gent_tram_stops_400m', 'gent_dist_rail_station_m',
    'gent_circulatie_sector', 'gent_in_autovrij', 'gent_cambio_500m', 'gent_ev_chargers_400m',
]

RENDER_COL_NAMES = [
    'vis_cgi_margin_mean', 'vis_cgi_margin_max', 'vis_frac_cgi', 'vis_frac_vstage',
    'vis_frac_any_render', 'vis_frac_clean_real', 'vis_frac_plan', 'vis_frac_shell',
]
Q_COMPACT_IDXS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]  # 10 means + 10 room-specific/worst
Q_COMPACT_NAMES = [f'vis_q_{j}' for j in Q_COMPACT_IDXS]
SO_Q_NAMES = [f'vis_so_q_{j}' for j in range(10)]


def _inner_cluster_folds(cids: np.ndarray, n_inner: int = 5, seed: int = 42) -> np.ndarray:
    uniq = np.unique(cids)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    c2f = {c: i % n_inner for i, c in enumerate(uniq)}
    return np.array([c2f[c] for c in cids], dtype=int)


def _fit_tabular_anchor(tr_df: pd.DataFrame, va_df: pd.DataFrame, y_tr: np.ndarray, inner_f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fast structured-only LightGBM anchor (no text/vision) so text & vision learn true residual!"""
    import lightgbm as lgb

    drop_pfx = ('txt_', 'e5_', 'e5l_', 'tfidf_', 'qwen3_', 'vis_')
    drop_ex = set(META_COLUMNS) | set(NON_STATIONARY_MACRO) | {'cluster_id', 'fold_cluster', 'street_norm'}
    cols = [c for c in tr_df.columns if c not in drop_ex and not c.startswith(drop_pfx)]

    def _p(d):
        o = d[cols].copy()
        for c in o.columns:
            if o[c].dtype == 'bool':
                o[c] = o[c].astype('int8')
            elif not pd.api.types.is_numeric_dtype(o[c]):
                o[c] = o[c].astype(str).fillna('Missing').astype('category')
            else:
                o[c] = o[c].astype('float32')
        return o

    Xtr, Xva = _p(tr_df), _p(va_df)
    oof = np.zeros(len(Xtr), dtype=np.float32)
    params = dict(
        objective='huber', alpha=0.15, n_estimators=450, learning_rate=0.04,
        num_leaves=47, min_child_samples=30, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.5, reg_alpha=1.0, reg_lambda=3.0, random_state=42, n_jobs=3, verbose=-1,
    )
    for f in np.unique(inner_f):
        m_tr, m_va = inner_f != f, inner_f == f
        oof[m_va] = lgb.LGBMRegressor(**params).fit(Xtr.iloc[m_tr], y_tr[m_tr]).predict(Xtr.iloc[m_va])
    va_pred = lgb.LGBMRegressor(**params).fit(Xtr, y_tr).predict(Xva).astype(np.float32)
    return oof, va_pred


def _build_one_split(
    split_id: str,
    tr_idx: np.ndarray,
    va_idx: np.ndarray,
    df: pd.DataFrame,
    texts: list[str],
    e5l: np.ndarray,
    e5s: np.ndarray,
    qwen3: np.ndarray,
    deflator: np.ndarray,
    cids: np.ndarray,
    anomaly_flags: np.ndarray,
    vis_bundle: dict | None,
):
    t_sp = time.perf_counter()
    tr_raw = df.iloc[tr_idx].copy().reset_index(drop=True)
    va_raw = df.iloc[va_idx].copy().reset_index(drop=True)
    tr_def = deflator[tr_idx]
    va_def = deflator[va_idx]
    tr_cids = cids[tr_idx]
    inner_f = _inner_cluster_folds(tr_cids, n_inner=5, seed=42)

    # 1. Strictly cluster-safe spatial + TE (inner-OOF on train, full-train on val!)
    va_sp = _compute_spatial_and_te(tr_raw, va_raw).reset_index(drop=True)
    sp_parts = []
    for uf in np.unique(inner_f):
        sp_parts.append(_compute_spatial_and_te(tr_raw[inner_f != uf], tr_raw[inner_f == uf]))
    tr_sp = pd.concat(sp_parts, axis=0).sort_index().reset_index(drop=True)

    tr_base = pd.concat([tr_raw.drop(columns=[c for c in tr_sp.columns if c in tr_raw.columns]), tr_sp], axis=1)
    va_base = pd.concat([va_raw.drop(columns=[c for c in va_sp.columns if c in va_raw.columns]), va_sp], axis=1)

    # Normalized agency frequency (fixes 8 split-casing offices)
    for col in ('statbel_sector', 'neighbourhood', 'postal_code', 'property_subtype', 'agency_norm'):
        if col in tr_base.columns:
            vc = tr_base[col].fillna('Missing').astype(str).value_counts()
            tr_base[f'freq_{col}'] = tr_base[col].fillna('Missing').astype(str).map(vc).fillna(0).astype('float32')
            va_base[f'freq_{col}'] = va_base[col].fillna('Missing').astype(str).map(vc).fillna(0).astype('float32')

    # Targets deflated by winning 0.5*CPI + 0.5*HPI!
    tr_y_psqm = np.log((tr_raw['price'].to_numpy(float) / tr_def) / tr_raw['livable_area'].to_numpy(float))
    tr_y_price = np.log(tr_raw['price'].to_numpy(float) / tr_def)
    tr_sp_anchor = np.log(np.maximum(tr_sp['sp_same_type_idw_psqm_k15'].to_numpy(float), 500.0))
    tr_y_sp_resid = tr_y_psqm - tr_sp_anchor

    # 2. Structured-only tabular anchor -> true tabular residual `r_tab`!
    tr_tab_hat, va_tab_hat = _fit_tabular_anchor(tr_base, va_base, tr_y_psqm, inner_f)
    tr_r_tab = tr_y_psqm - tr_tab_hat

    # 3. Strictly cluster-safe Text Meta-Features (TF-IDF Word+Char, E5-large, Qwen3-0.6B, SVD-16, E5-PCA-24)
    tr_txts = [texts[i] for i in tr_idx]
    va_txts = [texts[i] for i in va_idx]
    vec_w = TfidfVectorizer(max_features=35000, ngram_range=(1, 2), min_df=4, max_df=0.90, sublinear_tf=True,
                            token_pattern=r'(?u)\b[a-zA-Z0-9éèêëàâïîôùûç€²]{2,}\b')
    vec_c = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), max_features=30000, min_df=10, max_df=0.85, sublinear_tf=True)
    X_tr_tf = hstack([vec_w.fit_transform(tr_txts), vec_c.fit_transform(tr_txts)]).tocsr()
    X_va_tf = hstack([vec_w.transform(va_txts), vec_c.transform(va_txts)]).tocsr()

    tr_e5l, va_e5l = e5l[tr_idx], e5l[va_idx]
    tr_qw, va_qw = qwen3[tr_idx], qwen3[va_idx]
    tr_dense = np.hstack([tr_e5l, tr_qw])
    va_dense = np.hstack([va_e5l, va_qw])

    n_tr, n_va = len(tr_idx), len(va_idx)
    t_names = [
        'txt_tfidf_a2_psqm', 'txt_tfidf_a2_price', 'txt_tfidf_a14_psqm', 'txt_tfidf_huber_psqm',
        'txt_tfidf_tabresid', 'txt_tfidf_spresid',
        'txt_e5l_a5_psqm', 'txt_e5l_a5_price', 'txt_e5l_tabresid',
        'txt_qwen3_a8_psqm', 'txt_dense_both_psqm', 'txt_dense_both_tabresid',
    ]
    tr_meta = {k: np.zeros(n_tr, dtype=np.float32) for k in t_names}
    va_meta = {}

    def _huber_sgd(Xtr, ytr, Xte):
        return SGDRegressor(loss='huber', epsilon=0.15, alpha=1e-4, max_iter=60, tol=1e-4, random_state=42).fit(Xtr, ytr).predict(Xte)

    for f in range(5):
        m_tr, m_va = inner_f != f, inner_f == f
        tr_meta['txt_tfidf_a2_psqm'][m_va] = Ridge(alpha=2.5, solver='lsqr', max_iter=200).fit(X_tr_tf[m_tr], tr_y_psqm[m_tr]).predict(X_tr_tf[m_va])
        tr_meta['txt_tfidf_a2_price'][m_va] = Ridge(alpha=2.5, solver='lsqr', max_iter=200).fit(X_tr_tf[m_tr], tr_y_price[m_tr]).predict(X_tr_tf[m_va])
        tr_meta['txt_tfidf_a14_psqm'][m_va] = Ridge(alpha=14.0, solver='lsqr', max_iter=200).fit(X_tr_tf[m_tr], tr_y_psqm[m_tr]).predict(X_tr_tf[m_va])
        tr_meta['txt_tfidf_huber_psqm'][m_va] = _huber_sgd(X_tr_tf[m_tr], tr_y_psqm[m_tr], X_tr_tf[m_va])
        tr_meta['txt_tfidf_tabresid'][m_va] = Ridge(alpha=6.0, solver='lsqr', max_iter=200).fit(X_tr_tf[m_tr], tr_r_tab[m_tr]).predict(X_tr_tf[m_va])
        tr_meta['txt_tfidf_spresid'][m_va] = Ridge(alpha=2.5, solver='lsqr', max_iter=200).fit(X_tr_tf[m_tr], tr_y_sp_resid[m_tr]).predict(X_tr_tf[m_va])
        tr_meta['txt_e5l_a5_psqm'][m_va] = Ridge(alpha=5.0).fit(tr_e5l[m_tr], tr_y_psqm[m_tr]).predict(tr_e5l[m_va])
        tr_meta['txt_e5l_a5_price'][m_va] = Ridge(alpha=5.0).fit(tr_e5l[m_tr], tr_y_price[m_tr]).predict(tr_e5l[m_va])
        tr_meta['txt_e5l_tabresid'][m_va] = Ridge(alpha=12.0).fit(tr_e5l[m_tr], tr_r_tab[m_tr]).predict(tr_e5l[m_va])
        tr_meta['txt_qwen3_a8_psqm'][m_va] = Ridge(alpha=8.0).fit(tr_qw[m_tr], tr_y_psqm[m_tr]).predict(tr_qw[m_va])
        tr_meta['txt_dense_both_psqm'][m_va] = Ridge(alpha=15.0).fit(tr_dense[m_tr], tr_y_psqm[m_tr]).predict(tr_dense[m_va])
        tr_meta['txt_dense_both_tabresid'][m_va] = Ridge(alpha=25.0).fit(tr_dense[m_tr], tr_r_tab[m_tr]).predict(tr_dense[m_va])

    va_meta['txt_tfidf_a2_psqm'] = Ridge(alpha=2.5, solver='lsqr', max_iter=200).fit(X_tr_tf, tr_y_psqm).predict(X_va_tf).astype(np.float32)
    va_meta['txt_tfidf_a2_price'] = Ridge(alpha=2.5, solver='lsqr', max_iter=200).fit(X_tr_tf, tr_y_price).predict(X_va_tf).astype(np.float32)
    va_meta['txt_tfidf_a14_psqm'] = Ridge(alpha=14.0, solver='lsqr', max_iter=200).fit(X_tr_tf, tr_y_psqm).predict(X_va_tf).astype(np.float32)
    va_meta['txt_tfidf_huber_psqm'] = _huber_sgd(X_tr_tf, tr_y_psqm, X_va_tf).astype(np.float32)
    va_meta['txt_tfidf_tabresid'] = Ridge(alpha=6.0, solver='lsqr', max_iter=200).fit(X_tr_tf, tr_r_tab).predict(X_va_tf).astype(np.float32)
    va_meta['txt_tfidf_spresid'] = Ridge(alpha=2.5, solver='lsqr', max_iter=200).fit(X_tr_tf, tr_y_sp_resid).predict(X_va_tf).astype(np.float32)
    va_meta['txt_e5l_a5_psqm'] = Ridge(alpha=5.0).fit(tr_e5l, tr_y_psqm).predict(va_e5l).astype(np.float32)
    va_meta['txt_e5l_a5_price'] = Ridge(alpha=5.0).fit(tr_e5l, tr_y_price).predict(va_e5l).astype(np.float32)
    va_meta['txt_e5l_tabresid'] = Ridge(alpha=12.0).fit(tr_e5l, tr_r_tab).predict(va_e5l).astype(np.float32)
    va_meta['txt_qwen3_a8_psqm'] = Ridge(alpha=8.0).fit(tr_qw, tr_y_psqm).predict(va_qw).astype(np.float32)
    va_meta['txt_dense_both_psqm'] = Ridge(alpha=15.0).fit(tr_dense, tr_y_psqm).predict(va_dense).astype(np.float32)
    va_meta['txt_dense_both_tabresid'] = Ridge(alpha=25.0).fit(tr_dense, tr_r_tab).predict(va_dense).astype(np.float32)

    # Compact high-signal blends & anchor reconstruction
    tr_meta['tab_anchor_log_psqm'] = tr_tab_hat
    va_meta['tab_anchor_log_psqm'] = va_tab_hat
    for d in (tr_meta, va_meta):
        d['txt_blend_psqm'] = (0.55 * d['txt_tfidf_a2_psqm'] + 0.30 * d['txt_e5l_a5_psqm'] + 0.15 * d['txt_qwen3_a8_psqm']).astype(np.float32)
        d['txt_super_tabresid'] = (0.60 * d['txt_tfidf_tabresid'] + 0.40 * d['txt_dense_both_tabresid']).astype(np.float32)
        d['txt_tab_plus_text_psqm'] = (d['tab_anchor_log_psqm'] + d['txt_super_tabresid']).astype(np.float32)
        d['txt_tfidf_minus_e5l_psqm'] = (d['txt_tfidf_a2_psqm'] - d['txt_e5l_a5_psqm']).astype(np.float32)

    # Compact PCA/SVD (24 E5-large + 16 TF-IDF SVD — cuts 56 noisy dims vs Phase 4!)
    pca_e = PCA(n_components=24, random_state=42)
    tr_pe, va_pe = pca_e.fit_transform(tr_e5l).astype(np.float32), pca_e.transform(va_e5l).astype(np.float32)
    for j in range(24):
        tr_meta[f'e5l_pca_{j}'] = tr_pe[:, j]
        va_meta[f'e5l_pca_{j}'] = va_pe[:, j]

    svd_t = TruncatedSVD(n_components=16, random_state=42)
    tr_st, va_st = svd_t.fit_transform(X_tr_tf).astype(np.float32), svd_t.transform(X_va_tf).astype(np.float32)
    for j in range(16):
        tr_meta[f'tfidf_svd_{j}'] = tr_st[:, j]
        va_meta[f'tfidf_svd_{j}'] = va_st[:, j]

    # 4. Strictly cluster-safe Full-Corpus Vision Meta-Features (Recipe 11 + Compact Q + Render Flags!)
    if vis_bundle is not None:
        sig_all = vis_bundle['sig_real']        # (N, 768)
        so_all = vis_bundle['so_sheet']         # (N, 1152)
        ren_all = vis_bundle['render_feats']    # (N, 8)
        q_all = vis_bundle['q_feats']           # (N, 30)
        soq_all = vis_bundle['so_q_feats']      # (N, 10)
        has_v = vis_bundle['has_vis']           # (N,) bool

        tr_sig = np.nan_to_num(sig_all[tr_idx], nan=0.0)
        va_sig = np.nan_to_num(sig_all[va_idx], nan=0.0)
        tr_so = np.nan_to_num(so_all[tr_idx], nan=0.0)
        va_so = np.nan_to_num(so_all[va_idx], nan=0.0)
        tr_hv = has_v[tr_idx]
        va_hv = has_v[va_idx]

        # Residual AFTER tabular + text so vision learns PURE orthogonal visual signal!
        tr_r_tabtxt = tr_y_psqm - tr_meta['txt_tab_plus_text_psqm']

        alphas = np.logspace(0.5, 4.5, 13)
        v_cols = ['vis_sig_psqm', 'vis_sig_tabtxt_resid', 'vis_so_psqm', 'vis_so_tabtxt_resid']
        for vc in v_cols:
            tr_meta[vc] = np.full(n_tr, np.nan, dtype=np.float32)
            va_meta[vc] = np.full(n_va, np.nan, dtype=np.float32)

        for f in range(5):
            m_tr = (inner_f != f) & tr_hv
            m_va = (inner_f == f) & tr_hv
            if m_tr.sum() > 50 and m_va.sum() > 0:
                tr_meta['vis_sig_psqm'][m_va] = RidgeCV(alphas=alphas).fit(tr_sig[m_tr], tr_y_psqm[m_tr]).predict(tr_sig[m_va])
                tr_meta['vis_sig_tabtxt_resid'][m_va] = RidgeCV(alphas=alphas).fit(tr_sig[m_tr], tr_r_tabtxt[m_tr]).predict(tr_sig[m_va])
                tr_meta['vis_so_psqm'][m_va] = RidgeCV(alphas=alphas).fit(tr_so[m_tr], tr_y_psqm[m_tr]).predict(tr_so[m_va])
                tr_meta['vis_so_tabtxt_resid'][m_va] = RidgeCV(alphas=alphas).fit(tr_so[m_tr], tr_r_tabtxt[m_tr]).predict(tr_so[m_va])

        if tr_hv.sum() > 50 and va_hv.sum() > 0:
            va_meta['vis_sig_psqm'][va_hv] = RidgeCV(alphas=alphas).fit(tr_sig[tr_hv], tr_y_psqm[tr_hv]).predict(va_sig[va_hv])
            va_meta['vis_sig_tabtxt_resid'][va_hv] = RidgeCV(alphas=alphas).fit(tr_sig[tr_hv], tr_r_tabtxt[tr_hv]).predict(va_sig[va_hv])
            va_meta['vis_so_psqm'][va_hv] = RidgeCV(alphas=alphas).fit(tr_so[tr_hv], tr_y_psqm[tr_hv]).predict(va_so[va_hv])
            va_meta['vis_so_tabtxt_resid'][va_hv] = RidgeCV(alphas=alphas).fit(tr_so[tr_hv], tr_r_tabtxt[tr_hv]).predict(va_so[va_hv])

        for d in (tr_meta, va_meta):
            d['vis_blend_psqm'] = (0.55 * d['vis_sig_psqm'] + 0.45 * d['vis_so_psqm']).astype(np.float32)
            d['vis_blend_tabtxt_resid'] = (0.55 * d['vis_sig_tabtxt_resid'] + 0.45 * d['vis_so_tabtxt_resid']).astype(np.float32)

        # Compact SigLIP2-real PCA-8
        pca_v = PCA(n_components=8, random_state=42)
        pca_v.fit(tr_sig[tr_hv])
        tr_pv = np.full((n_tr, 8), np.nan, dtype=np.float32)
        va_pv = np.full((n_va, 8), np.nan, dtype=np.float32)
        tr_pv[tr_hv] = pca_v.transform(tr_sig[tr_hv]).astype(np.float32)
        va_pv[va_hv] = pca_v.transform(va_sig[va_hv]).astype(np.float32)
        for j in range(8):
            tr_meta[f'vis_sig_pca_{j}'] = tr_pv[:, j]
            va_meta[f'vis_sig_pca_{j}'] = va_pv[:, j]

        # 8 Render/Media flags (make-or-break -0.19pp interaction signal!) + compact Q-scores
        for j, cname in enumerate(RENDER_COL_NAMES):
            tr_meta[cname] = ren_all[tr_idx, j].astype(np.float32)
            va_meta[cname] = ren_all[va_idx, j].astype(np.float32)
        for j, cname in zip(Q_COMPACT_IDXS, Q_COMPACT_NAMES):
            tr_meta[cname] = q_all[tr_idx, j].astype(np.float32)
            va_meta[cname] = q_all[va_idx, j].astype(np.float32)
        for j, cname in enumerate(SO_Q_NAMES):
            tr_meta[cname] = soq_all[tr_idx, j].astype(np.float32)
            va_meta[cname] = soq_all[va_idx, j].astype(np.float32)
        tr_meta['vis_has_photos'] = tr_hv.astype(np.int8)
        va_meta['vis_has_photos'] = va_hv.astype(np.int8)

    tr_df = pd.concat([tr_base, pd.DataFrame(tr_meta)], axis=1)
    va_df = pd.concat([va_base, pd.DataFrame(va_meta)], axis=1)
    for drop_c in ('street_norm', 'cluster_id', 'fold_cluster'):
        if drop_c in tr_df.columns:
            tr_df.drop(columns=[drop_c], inplace=True)
            va_df.drop(columns=[drop_c], inplace=True)

    print(f"  [Split {split_id:13s}] tr={len(tr_df):5d} va={len(va_df):5d} | cols={tr_df.shape[1]} | {time.perf_counter()-t_sp:.1f}s", flush=True)
    return {
        'id': split_id,
        'tr_idx': tr_idx,
        'va_idx': va_idx,
        'tr_df': tr_df,
        'va_df': va_df,
        'tr_deflator': tr_def,
        'va_deflator': va_def,
        'tr_cids': tr_cids,
        'va_cids': cids[va_idx],
        'tr_anomaly': anomaly_flags[tr_idx],
        'va_anomaly': anomaly_flags[va_idx],
    }


def build_phase6_splits():
    t0 = time.perf_counter()
    print("=== PHASE-6 UNIFIED FEATURE FACTORY (Cluster CV + 0.5*CPI+0.5*HPI + GeoW2 + Issue#3 + Vision) ===", flush=True)

    df = pd.read_parquet(DATASET_PATH)
    df['property_id'] = df['property_id'].astype(str)
    cl = pd.read_parquet(CLUSTER_PATH)
    i3 = pd.read_parquet(ISSUE3_PATH)
    assert (df['property_id'].values == cl['property_id'].astype(str).values).all()
    assert (df['property_id'].values == i3['property_id'].astype(str).values).all()

    # Join raw texts, street, agency
    raw = pd.read_parquet(
        RAW_PARQUET,
        columns=['property_id', 'date_obtained', 'title', 'description', 'description_nl', 'immo_office_name', 'street'],
    )
    raw['property_id'] = raw['property_id'].astype(str)
    raw['date_obtained'] = pd.to_datetime(raw['date_obtained'], format='%d%m%Y', errors='coerce')
    raw = raw.drop_duplicates(subset=['property_id', 'date_obtained'], keep='last')
    df = df.merge(raw, on=['property_id', 'date_obtained'], how='left')

    # Normalized agency name (fixes 8 split-casing Ghent offices!)
    df['agency_norm'] = df['immo_office_name'].fillna('Unknown').astype(str).str.strip().str.lower()
    df['street_norm'] = (
        df['postal_code'].fillna('0000').astype(str).str.strip() + '|'
        + df['street'].fillna('unknown').astype(str).str.strip().str.lower()
    )

    # Build Dutch-preferred text strings
    titles = df['title'].fillna('').astype(str).to_numpy()
    d_nl = df['description_nl'].fillna('').astype(str).to_numpy()
    d_mn = df['description'].fillna('').astype(str).to_numpy()
    texts = [f"{t}. {nl if len(nl) >= 40 else mn}"[:2400] for t, nl, mn in zip(titles, d_nl, d_mn)]
    anomaly_flags = compute_anomaly_flag(titles, np.array(texts), df['price'].to_numpy(float))
    df.drop(columns=['title', 'description', 'description_nl', 'street'], inplace=True)

    # Domain + Text Flags + Flanders/GRB (Wave-1 + ONLY winning Wave-2 cols!)
    df = _add_domain_features(df)
    df = pd.concat([df, _extract_text_flags(texts)], axis=1)
    df = attach_flanders_and_grb(df)
    # Drop bloated Wave-2 columns (keep ONLY winning IRCEL air quality + VHA waterfront!)
    drop_b = [c for c in DROP_W2_BLOATED if c in df.columns]
    if drop_b:
        df.drop(columns=drop_b, inplace=True)

    # Attach curated Issue #3 features
    i3_sub = i3[[c for c in KEEP_ISSUE3_COLS if c in i3.columns]].reset_index(drop=True)
    df = pd.concat([df.reset_index(drop=True), i3_sub], axis=1)

    # Attach cluster folds (Issue #4)
    cids = cl['cluster_id'].to_numpy(int)
    f_cl = cl['fold_cluster'].to_numpy(int)
    df['fold'] = f_cl  # override legacy fold everywhere downstream!

    # Load text embeddings (E5-large + E5-small + Qwen3)
    e5l = np.load(E5_LARGE_PATH)['embeddings'].astype(np.float32)
    e5s = np.load(E5_SMALL_PATH)['embeddings'].astype(np.float32)
    qwen3 = np.load(QWEN3_PATH)['embeddings'].astype(np.float32)
    assert len(e5l) == len(df) and len(qwen3) == len(df)

    # Load Full-Corpus Vision embeddings if ready!
    vis_bundle = None
    if os.path.exists(VIS_FULL_PATH):
        vz = np.load(VIS_FULL_PATH, allow_pickle=True)
        v_pids = [str(p) for p in vz['property_id']]
        p2i = {p: idx for idx, p in enumerate(v_pids)}
        N = len(df)
        sig_real = np.full((N, 768), np.nan, dtype=np.float32)
        so_sheet = np.full((N, 1152), np.nan, dtype=np.float32)
        render_f = np.full((N, 8), np.nan, dtype=np.float32)
        q_f = np.full((N, 30), np.nan, dtype=np.float32)
        so_q_f = np.full((N, 10), np.nan, dtype=np.float32)
        has_vis = np.zeros(N, dtype=bool)
        for row_i, pid in enumerate(df['property_id'].values):
            vi = p2i.get(pid)
            if vi is not None and not np.isnan(vz['sig_real'][vi, 0]):
                sig_real[row_i] = vz['sig_real'][vi]
                so_sheet[row_i] = vz['so_sheet'][vi]
                render_f[row_i] = vz['render_feats'][vi]
                q_f[row_i] = vz['q_feats'][vi]
                so_q_f[row_i] = vz['so_q_feats'][vi]
                has_vis[row_i] = True
        vis_bundle = {
            'sig_real': sig_real, 'so_sheet': so_sheet,
            'render_feats': render_f, 'q_feats': q_f, 'so_q_feats': so_q_f,
            'has_vis': has_vis,
        }
        print(f"  Loaded FULL VISION embeddings: {has_vis.sum():,}/{N:,} homes ({100*has_vis.mean():.1f}%)!", flush=True)
    else:
        print(f"  [WARN] {VIS_FULL_PATH} not found yet — building WITHOUT vision!", flush=True)

    # Winning Issue #9 Deflator: 0.5 * CPI + 0.5 * HPI (relative to 2025 median)
    cpi_raw = df['belgian_cpi_index'].to_numpy(float)
    hpi_raw = df['belgian_hpi_index'].to_numpy(float)
    m25 = (df['publication_year'] == 2025).to_numpy()
    cpi_idx = cpi_raw / np.nanmedian(cpi_raw[m25])
    hpi_idx = hpi_raw / np.nanmedian(hpi_raw[m25])
    deflator = (0.5 * cpi_idx + 0.5 * hpi_idx).astype(np.float64)

    # Build 6 strictly leak-free splits: 5 anti-chaining cluster folds + cluster-clean 2026 temporal!
    split_specs = []
    for k in range(5):
        va_idx = np.where(f_cl == k)[0]
        tr_idx = np.where(f_cl != k)[0]
        assert len(set(cids[tr_idx]) & set(cids[va_idx])) == 0
        split_specs.append((f'fold_{k}', tr_idx, va_idx))

    va_tmp = np.where((df['publication_year'] >= 2026).to_numpy())[0]
    va_tmp_cids = set(cids[va_tmp])
    tr_tmp = np.where((df['publication_year'] < 2026).to_numpy() & (~np.isin(cids, list(va_tmp_cids))))[0]
    assert len(set(cids[tr_tmp]) & set(cids[va_tmp])) == 0
    split_specs.append(('temporal_2026', tr_tmp, va_tmp))
    print(f"  Cluster-clean splits ready | Temporal train={len(tr_tmp):,} (0 cluster leaks!), test_2026={len(va_tmp):,}", flush=True)

    bundles = Parallel(n_jobs=6, backend='loky')(
        delayed(_build_one_split)(
            sid, tr_i, va_i, df, texts, e5l, e5s, qwen3, deflator, cids, anomaly_flags, vis_bundle
        )
        for sid, tr_i, va_i in split_specs
    )

    drop_meta = set(META_COLUMNS) | set(NON_STATIONARY_MACRO) | {'immo_office_name', 'agency_norm'}
    all_cols = bundles[0]['tr_df'].columns.tolist()
    feat_cols_stationary = [c for c in all_cols if c not in drop_meta]
    feat_cols_with_agency = [c for c in all_cols if c not in (set(META_COLUMNS) | set(NON_STATIONARY_MACRO) | {'immo_office_name'})]

    # Normal Ghent home mask (€150k–€800k, excl APARTMENT_BLOCK/MANSION/SERVICE_FLAT)
    y_price = df['price'].to_numpy(float)
    normal_mask = (
        (y_price >= 150_000) & (y_price <= 800_000)
        & (~df['property_subtype'].isin(['APARTMENT_BLOCK', 'MANSION', 'SERVICE_FLAT']).to_numpy())
    )

    out = {
        'bundles': bundles,
        'y_price': y_price,
        'y_area': df['livable_area'].to_numpy(float),
        'is_house': df['property_type_HOUSE'].to_numpy(int),
        'normal_mask': normal_mask,
        'folds': f_cl,
        'cluster_ids': cids,
        'deflator': deflator,
        'cpi_idx': cpi_idx,
        'hpi_idx': hpi_idx,
        'has_vision': vis_bundle is not None,
        'feat_cols_stationary': feat_cols_stationary,
        'feat_cols_with_agency': feat_cols_with_agency,
        'anomaly_flags': anomaly_flags,
    }
    joblib.dump(out, OUT_CACHE, compress=3)
    print(
        f"\n=== PHASE-6 CACHE SAVED ({len(feat_cols_stationary)} stationary / {len(feat_cols_with_agency)} w/ agency cols, "
        f"vision={vis_bundle is not None}) in {time.perf_counter()-t0:.1f}s -> {OUT_CACHE} ===",
        flush=True,
    )


if __name__ == '__main__':
    build_phase6_splits()
