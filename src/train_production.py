"""Issue #16 — Reproducible Production Model Training & Packaging (`src/train_production.py`).

Trains and freezes the dual-track Phase-7 Lean-5 production bundle (`outputs/models/production/`):
  - Strictly agency-blind (no `immo_office_name`, `agency_norm`, `te_agency_*`, `freq_agency_norm`)
  - No Qwen3 / external MLX dependency (TF-IDF Word+Char + `intfloat/multilingual-e5-large`)
  - Winning `0.5 * CPI + 0.5 * HPI` deflator anchored to 2025 median
  - Dual-track Lean-5 ensemble:
      1. `tabular_text` track (228 GBDT cols / 131 TabM cols) — optimal for manual entry, form edits,
         and listings without photos (avoids the +1.4pp masked-vision blowup discovered in #15!)
      2. `multimodal` track (280 GBDT cols / 167 TabM cols, adds SigLIP2-base + SO400M-384 photo features)
         — used whenever photo embeddings are present
  - Each track runs full 5-fold anti-chaining Cluster CV + cluster-clean 2026 Temporal holdout to:
      * verify exact production OOF & 2026-temporal accuracy
      * fit the `HuberRegressor` stacker/recalibrator on leak-free OOF predictions
      * fit the heteroscedastic conformal `sigma_hat(x)` LGBM model + calibrated `q_80`/`q_90`
    and then refits all 5 member models on 100% of training data (`n=14,599`) for maximum live accuracy!

Apple Silicon safety:
  - Imports `torch` BEFORE `lightgbm` / `catboost` AND isolates CPU GBDT vs MPS TabM stages via
    subprocess workers so macOS OpenMP exit-139 segfaults and MPS allocator fragmentation never occur.

Usage:
    .venv-api/bin/python -u src/train_production.py
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'src'))

WORK_DIR = os.path.join(BASE_DIR, 'outputs', 'models', 'workbench')
PROD_DIR = os.path.join(BASE_DIR, 'outputs', 'models', 'production')
STAGE_DIR = os.path.join(WORK_DIR, 'prod_train_staging')

DATASET_PATH = os.path.join(BASE_DIR, 'outputs', 'dataset', 'training_dataset.parquet')
RAW_PARQUET = os.path.join(BASE_DIR, 'outputs', 'property_data_full.parquet')
CLUSTER_PATH = os.path.join(BASE_DIR, 'outputs', 'dataset', 'cluster_folds.parquet')
PREPROC_PATH = os.path.join(BASE_DIR, 'outputs', 'dataset', 'preprocessing_artifacts.joblib')
E5_LARGE_PATH = os.path.join(WORK_DIR, 'e5_large_14599.npz')
VIS_FULL_PATH = os.path.join(BASE_DIR, 'external_data', 'vision_full', 'vision_full_embeddings.npz')
P7_FSETS_PATH = os.path.join(WORK_DIR, 'phase7_feature_sets.json')
P7_HPO_PATH = os.path.join(WORK_DIR, 'phase7_gbdt_hpo.json')

MEMBER_NAMES = [
    'lgbmxt_price_hub',
    'spec_lgbm_price_hub',
    'cb_tuned_price_l2',
    'tabm_perm170_price_4s',
    'tabm_perm170_psqm_4s',
]


def _target_fwd(y_price: np.ndarray, area: np.ndarray, defl: np.ndarray, mode: str) -> np.ndarray:
    return np.log((y_price / defl) / area) if mode == 'log_real_psqm' else np.log(y_price / defl)


def _target_inv(pred_log: np.ndarray, area: np.ndarray, defl: np.ndarray, mode: str) -> np.ndarray:
    p = np.clip(np.asarray(pred_log, dtype=np.float64), -20.0, 20.0)
    return np.clip(np.exp(p) * (area * defl if mode == 'log_real_psqm' else defl), 10_000.0, 1e7)


def _load_training_inputs():
    """Load training_dataset + raw texts/streets + cluster folds + E5-large + vision embeddings."""
    from production_pipeline import prepare_base_dataframe

    preproc = joblib.load(PREPROC_PATH)
    df = pd.read_parquet(DATASET_PATH)
    df['property_id'] = df['property_id'].astype(str)
    cl = pd.read_parquet(CLUSTER_PATH)
    assert (df['property_id'].values == cl['property_id'].astype(str).values).all()

    raw = pd.read_parquet(
        RAW_PARQUET,
        columns=['property_id', 'date_obtained', 'title', 'description', 'description_nl', 'street'],
    )
    raw['property_id'] = raw['property_id'].astype(str)
    raw['date_obtained'] = pd.to_datetime(raw['date_obtained'], format='%d%m%Y', errors='coerce')
    raw = raw.drop_duplicates(subset=['property_id', 'date_obtained'], keep='last')
    df = df.merge(raw, on=['property_id', 'date_obtained'], how='left')

    base_df, texts = prepare_base_dataframe(df, preproc, already_preprocessed=True)
    e5l = np.load(E5_LARGE_PATH)['embeddings'].astype(np.float32)
    assert len(e5l) == len(base_df)

    # Extract vision arrays ONCE outside loop (avoid lazy NpzFile re-decompress!)
    vis_bundle = None
    if os.path.exists(VIS_FULL_PATH):
        vz = np.load(VIS_FULL_PATH, allow_pickle=True)
        v_pids = [str(p) for p in vz['property_id']]
        vz_sig = vz['sig_real'].astype(np.float32)
        vz_so = vz['so_sheet'].astype(np.float32)
        vz_ren = vz['render_feats'].astype(np.float32)
        vz_q = vz['q_feats'].astype(np.float32)
        vz_soq = vz['so_q_feats'].astype(np.float32)
        p2i = {p: idx for idx, p in enumerate(v_pids)}
        N = len(base_df)
        sig_real = np.full((N, 768), np.nan, dtype=np.float32)
        so_sheet = np.full((N, 1152), np.nan, dtype=np.float32)
        render_f = np.full((N, 8), np.nan, dtype=np.float32)
        q_f = np.full((N, 30), np.nan, dtype=np.float32)
        so_q_f = np.full((N, 10), np.nan, dtype=np.float32)
        has_vis = np.zeros(N, dtype=bool)
        for row_i, pid in enumerate(base_df['property_id'].values):
            vi = p2i.get(pid)
            if vi is not None and not np.isnan(vz_sig[vi, 0]):
                sig_real[row_i] = vz_sig[vi]
                so_sheet[row_i] = vz_so[vi]
                render_f[row_i] = vz_ren[vi]
                q_f[row_i] = vz_q[vi]
                so_q_f[row_i] = vz_soq[vi]
                has_vis[row_i] = True
        vis_bundle = {
            'sig_real': sig_real, 'so_sheet': so_sheet,
            'render_feats': render_f, 'q_feats': q_f, 'so_q_feats': so_q_f,
            'has_vis': has_vis,
        }

    cids = cl['cluster_id'].to_numpy(int)
    f_cl = cl['fold_cluster'].to_numpy(int)
    y_price = base_df['price'].to_numpy(float)
    y_area = base_df['livable_area'].to_numpy(float)
    normal_mask = (
        (y_price >= 150_000) & (y_price <= 800_000)
        & (~base_df['property_subtype'].isin(['APARTMENT_BLOCK', 'MANSION', 'SERVICE_FLAT']).to_numpy())
    )
    return preproc, base_df, texts, e5l, vis_bundle, cids, f_cl, y_price, y_area, normal_mask


def _slice_vis(vis_bundle: dict | None, idx: np.ndarray) -> dict | None:
    if vis_bundle is None:
        return None
    return {k: v[idx] for k, v in vis_bundle.items()}


def stage_prepare_features():
    """Stage 1: Build leak-free 6-split CV matrices AND full-data ProductionFeaturePipeline."""
    from joblib import Parallel, delayed
    from production_pipeline import (
        AGENCY_COLS,
        QWEN_COLS,
        ProductionFeaturePipeline,
        compute_deflator,
    )

    os.makedirs(STAGE_DIR, exist_ok=True)
    t0 = time.perf_counter()
    print("=== STAGE 1: FIT PRODUCTION FEATURE PIPELINE & 6 CLUSTER-CLEAN CV SPLITS ===", flush=True)
    preproc, base_df, texts, e5l, vis_bundle, cids, f_cl, y_price, y_area, normal_mask = _load_training_inputs()

    # Fit full-corpus ProductionFeaturePipeline (saved into production bundle!)
    full_pipe = ProductionFeaturePipeline(preproc)
    full_feat_df = full_pipe.fit(base_df, texts, e5l, cids, vis_bundle=vis_bundle)
    full_deflator = compute_deflator(base_df, full_pipe.cpi_ref_2025, full_pipe.hpi_ref_2025, full_pipe.latest_deflator)
    print(f"  Full ProductionFeaturePipeline fitted: {full_feat_df.shape} in {time.perf_counter()-t0:.1f}s", flush=True)

    # Resolve exact agency-blind, no-Qwen3 feature column lists for both tracks
    fsets = json.load(open(P7_FSETS_PATH))
    drop_always = set(AGENCY_COLS) | set(QWEN_COLS)
    avail = set(full_feat_df.columns)

    gbdt_mm = [c for c in fsets['gbdt_feats'] if c in avail and c not in drop_always]
    gbdt_tt = [c for c in gbdt_mm if not c.startswith('vis_')]
    nn_mm = [c for c in fsets['nn_feats_perm170'] if c in avail and c not in drop_always]
    nn_tt = [c for c in nn_mm if not c.startswith('vis_')]

    full_pipe.feature_columns_tabtxt = gbdt_tt
    full_pipe.feature_columns_multimodal = gbdt_mm
    print(f"  Column counts | multimodal: GBDT={len(gbdt_mm)}, TabM={len(nn_mm)} | tabular_text: GBDT={len(gbdt_tt)}, TabM={len(nn_tt)}", flush=True)

    # Build 6 leak-free CV splits (5 cluster folds + cluster-clean 2026 temporal holdout)
    split_specs = []
    for k in range(5):
        va_idx = np.where(f_cl == k)[0]
        tr_idx = np.where(f_cl != k)[0]
        assert len(set(cids[tr_idx]) & set(cids[va_idx])) == 0
        split_specs.append((f'fold_{k}', tr_idx, va_idx))

    va_tmp = np.where((base_df['publication_year'] >= 2026).to_numpy())[0]
    va_tmp_cids = set(cids[va_tmp])
    tr_tmp = np.where((base_df['publication_year'] < 2026).to_numpy() & (~np.isin(cids, list(va_tmp_cids))))[0]
    assert len(set(cids[tr_tmp]) & set(cids[va_tmp])) == 0
    split_specs.append(('temporal_2026', tr_tmp, va_tmp))

    def _run_split(sid, tr_i, va_i):
        ts = time.perf_counter()
        sp_pipe = ProductionFeaturePipeline(preproc)
        tr_txts = [texts[i] for i in tr_i]
        va_txts = [texts[i] for i in va_i]
        tr_feat = sp_pipe.fit(base_df.iloc[tr_i], tr_txts, e5l[tr_i], cids[tr_i], vis_bundle=_slice_vis(vis_bundle, tr_i))
        va_feat = sp_pipe.transform(base_df.iloc[va_i], va_txts, e5l[va_i], vis_bundle=_slice_vis(vis_bundle, va_i))
        tr_d = compute_deflator(base_df.iloc[tr_i], sp_pipe.cpi_ref_2025, sp_pipe.hpi_ref_2025, sp_pipe.latest_deflator)
        va_d = compute_deflator(base_df.iloc[va_i], sp_pipe.cpi_ref_2025, sp_pipe.hpi_ref_2025, sp_pipe.latest_deflator)
        print(f"  [Split {sid:13s}] tr={len(tr_feat):5d} va={len(va_feat):5d} ({time.perf_counter()-ts:.1f}s)", flush=True)
        return {
            'id': sid,
            'tr_idx': tr_i, 'va_idx': va_i,
            'tr_df': tr_feat, 'va_df': va_feat,
            'tr_deflator': tr_d, 'va_deflator': va_d,
            'tr_cids': cids[tr_i],
        }

    bundles = Parallel(n_jobs=6, backend='loky')(delayed(_run_split)(sid, ti, vi) for sid, ti, vi in split_specs)

    joblib.dump(full_pipe, os.path.join(STAGE_DIR, 'full_pipeline.joblib'), compress=3)
    joblib.dump(
        {
            'full_feat_df': full_feat_df,
            'full_deflator': full_deflator,
            'bundles': bundles,
            'y_price': y_price,
            'y_area': y_area,
            'normal_mask': normal_mask,
            'folds': f_cl,
            'cluster_ids': cids,
            'cols': {
                'multimodal': {'gbdt': gbdt_mm, 'nn': nn_mm},
                'tabular_text': {'gbdt': gbdt_tt, 'nn': nn_tt},
            },
        },
        os.path.join(STAGE_DIR, 'prod_splits.joblib'),
        compress=3,
    )
    print(f"  Stage 1 saved to {STAGE_DIR} in {time.perf_counter()-t0:.1f}s", flush=True)


def _fit_one_lgbm(X_fit, y_fit, X_es, y_es, params, n_threads=4, seed=42):
    import lightgbm as lgb

    kw = dict(
        objective='huber', n_estimators=4000, learning_rate=0.020,
        subsample_freq=1, n_jobs=n_threads, random_state=seed, verbose=-1,
    )
    kw.update(params)
    m = lgb.LGBMRegressor(**kw)
    m.fit(X_fit, y_fit, eval_set=[(X_es, y_es)], callbacks=[lgb.early_stopping(80, verbose=False)])
    return m


def _fit_one_catboost(X_fit, y_fit, X_es, y_es, cat_cols, params, n_threads=4, seed=42):
    from catboost import CatBoostRegressor

    kw = dict(
        loss_function='RMSE', iterations=3800, learning_rate=0.032,
        thread_count=n_threads, random_seed=seed, verbose=0, early_stopping_rounds=110,
    )
    kw.update(params)
    m = CatBoostRegressor(**kw)
    m.fit(X_fit, y_fit, cat_features=cat_cols, eval_set=(X_es, y_es), use_best_model=True)
    return m


def _es_split_mask(cids: np.ndarray, seed: int = 55) -> np.ndarray:
    u_c = np.unique(cids)
    es_c = set(np.random.default_rng(seed).choice(u_c, size=max(1, int(len(u_c) * 0.12)), replace=False))
    return np.isin(cids, list(es_c))


def stage_gbdt(track: str):
    """Stage 2 (CPU worker, no torch): CV + full-data fit of the 3 GBDT members for `track`."""
    from joblib import Parallel, delayed
    from phase3_features import compute_metrics
    from production_pipeline import TreePreprocSpec

    t0 = time.perf_counter()
    print(f"\n=== STAGE 2 (GBDT CPU) — track={track} ===", flush=True)
    data = joblib.load(os.path.join(STAGE_DIR, 'prod_splits.joblib'))
    hpo = json.load(open(P7_HPO_PATH))
    cols = data['cols'][track]['gbdt']
    bundles = data['bundles']
    y, a, nm = data['y_price'], data['y_area'], data['normal_mask']
    tmp_idx = [b for b in bundles if str(b['id']).startswith('temporal')][0]['va_idx']

    lgb_base_p = hpo['lgbm_price_huber']['params']
    lgb_xt_p = {**lgb_base_p, 'extra_trees': True, 'n_estimators': 4000, 'learning_rate': 0.020}
    lgb_spec_p = {**lgb_base_p, 'n_estimators': 3500, 'learning_rate': 0.022}
    cb_p = hpo['cb_price_l2']['params']

    def _cv_split(b):
        spec = TreePreprocSpec.fit(b['tr_df'], cols)
        Xtr_l = spec.transform_lgbm(b['tr_df'])
        Xva_l = spec.transform_lgbm(b['va_df'])
        Xtr_c = spec.transform_catboost(b['tr_df'])
        Xva_c = spec.transform_catboost(b['va_df'])

        ytr = _target_fwd(y[b['tr_idx']], a[b['tr_idx']], b['tr_deflator'], 'log_real_price')
        va_a, va_d = a[b['va_idx']], b['va_deflator']
        is_es = _es_split_mask(b['tr_cids'], seed=55)

        # 1) lgbmxt_price_hub
        m1 = _fit_one_lgbm(Xtr_l.iloc[~is_es], ytr[~is_es], Xtr_l.iloc[is_es], ytr[is_es], lgb_xt_p, n_threads=3, seed=42)
        p1 = _target_inv(m1.predict(Xva_l), va_a, va_d, 'log_real_price')

        # 2) spec_lgbm_price_hub (separate House / Apt specialists)
        p2 = np.zeros(len(b['va_idx']), dtype=np.float64)
        tr_h = (b['tr_df']['property_type_HOUSE'].to_numpy(int) == 1)
        va_h = (b['va_df']['property_type_HOUSE'].to_numpy(int) == 1)
        for h_flag, s_off in ((True, 0), (False, 7)):
            tm = tr_h == h_flag
            vm = va_h == h_flag
            es_sub = _es_split_mask(b['tr_cids'][tm], seed=55 + s_off)
            Xh = Xtr_l.iloc[tm]
            yh = ytr[tm]
            ms = _fit_one_lgbm(Xh.iloc[~es_sub], yh[~es_sub], Xh.iloc[es_sub], yh[es_sub], lgb_spec_p, n_threads=3, seed=42 + s_off)
            p2[vm] = _target_inv(ms.predict(Xva_l.iloc[vm]), va_a[vm], va_d[vm], 'log_real_price')

        # 3) cb_tuned_price_l2
        m3 = _fit_one_catboost(Xtr_c.iloc[~is_es], ytr[~is_es], Xtr_c.iloc[is_es], ytr[is_es], spec.cat_cols, cb_p, n_threads=3, seed=42)
        p3 = _target_inv(m3.predict(Xva_c), va_a, va_d, 'log_real_price')
        return p1, p2, p3

    outs = Parallel(n_jobs=6, backend='loky')(delayed(_cv_split)(b) for b in bundles)
    oof = {k: np.zeros(len(y), dtype=np.float64) for k in MEMBER_NAMES[:3]}
    tmp = {}
    for b, (p1, p2, p3) in zip(bundles, outs):
        for k, p in zip(MEMBER_NAMES[:3], (p1, p2, p3)):
            if str(b['id']).startswith('temporal'):
                tmp[k] = p
            else:
                oof[k][b['va_idx']] = p

    for k in MEMBER_NAMES[:3]:
        mo = compute_metrics(y[nm], oof[k][nm])
        mt = compute_metrics(y[tmp_idx][nm[tmp_idx]], tmp[k][nm[tmp_idx]])
        print(f"  [CV {track:12s}] {k:24s} | OOF {mo['mdape']:5.2f}% (€{mo['mae']:6,.0f}) | TMP {mt['mdape']:5.2f}%", flush=True)

    # Full-data refit of the 3 GBDTs on all 14,599 rows!
    full_df = data['full_feat_df']
    full_d = data['full_deflator']
    full_cids = data['cluster_ids']
    tree_spec = TreePreprocSpec.fit(full_df, cols)
    Xf_l = tree_spec.transform_lgbm(full_df)
    Xf_c = tree_spec.transform_catboost(full_df)
    yf = _target_fwd(y, a, full_d, 'log_real_price')
    is_es_f = _es_split_mask(full_cids, seed=55)

    m_lgbmxt = _fit_one_lgbm(Xf_l.iloc[~is_es_f], yf[~is_es_f], Xf_l.iloc[is_es_f], yf[is_es_f], lgb_xt_p, n_threads=8, seed=42)
    fh = (full_df['property_type_HOUSE'].to_numpy(int) == 1)
    es_h = _es_split_mask(full_cids[fh], seed=55)
    m_spec_h = _fit_one_lgbm(Xf_l.iloc[fh].iloc[~es_h], yf[fh][~es_h], Xf_l.iloc[fh].iloc[es_h], yf[fh][es_h], lgb_spec_p, n_threads=8, seed=42)
    es_a = _es_split_mask(full_cids[~fh], seed=62)
    m_spec_a = _fit_one_lgbm(Xf_l.iloc[~fh].iloc[~es_a], yf[~fh][~es_a], Xf_l.iloc[~fh].iloc[es_a], yf[~fh][es_a], lgb_spec_p, n_threads=8, seed=49)
    m_cb = _fit_one_catboost(Xf_c.iloc[~is_es_f], yf[~is_es_f], Xf_c.iloc[is_es_f], yf[is_es_f], tree_spec.cat_cols, cb_p, n_threads=8, seed=42)

    joblib.dump(
        {
            'tree_spec': tree_spec,
            'lgbmxt': m_lgbmxt,
            'spec_lgbm_house': m_spec_h,
            'spec_lgbm_apt': m_spec_a,
            'catboost_model': m_cb,
            'oof': oof,
            'tmp': tmp,
        },
        os.path.join(STAGE_DIR, f'gbdt_{track}.joblib'),
        compress=3,
    )
    print(f"  Stage 2 GBDT ({track}) done in {time.perf_counter()-t0:.1f}s", flush=True)


def stage_tabm(track: str, target_mode: str, n_seeds: int = 4, n_epochs: int = 105):
    """Stage 3 (MPS PyTorch worker, isolated subprocess): CV + full-data refit of one TabM member."""
    import torch
    from pytabkit import TabM_D_Regressor
    from phase3_features import compute_metrics
    from production_pipeline import NNPreprocSpec

    member = 'tabm_perm170_price_4s' if target_mode == 'log_real_price' else 'tabm_perm170_psqm_4s'
    out_path = os.path.join(STAGE_DIR, f'{member}_{track}.joblib')
    if os.path.exists(out_path):
        print(f"  [skip] {out_path} already exists", flush=True)
        return

    t0 = time.perf_counter()
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"\n=== STAGE 3 (TabM {device}) — track={track} member={member} ({n_seeds} seeds) ===", flush=True)
    data = joblib.load(os.path.join(STAGE_DIR, 'prod_splits.joblib'))
    cols = data['cols'][track]['nn']
    bundles = data['bundles']
    y, a, nm = data['y_price'], data['y_area'], data['normal_mask']
    tmp_idx = [b for b in bundles if str(b['id']).startswith('temporal')][0]['va_idx']

    oof = np.zeros(len(y), dtype=np.float64)
    tmp_p = None

    for b in bundles:
        nn_spec = NNPreprocSpec.fit(b['tr_df'], cols)
        Xtr = nn_spec.transform(b['tr_df'])
        Xva = nn_spec.transform(b['va_df'])
        yt = _target_fwd(y[b['tr_idx']], a[b['tr_idx']], b['tr_deflator'], target_mode).astype(np.float32)
        val_idxs = np.where(_es_split_mask(b['tr_cids'], seed=77))[0]
        s_p = []
        for s in range(n_seeds):
            m = TabM_D_Regressor(
                device=device, n_epochs=n_epochs, batch_size=256, val_metric_name='mae',
                random_state=42 + s * 19, verbosity=0,
            )
            m.fit(Xtr, yt, val_idxs=val_idxs)
            s_p.append(np.asarray(m.predict(Xva), dtype=np.float64))
            del m
            gc.collect()
            if device == 'mps':
                torch.mps.empty_cache()
        p_eur = _target_inv(np.mean(s_p, axis=0), a[b['va_idx']], b['va_deflator'], target_mode)
        if str(b['id']).startswith('temporal'):
            tmp_p = p_eur
        else:
            oof[b['va_idx']] = p_eur

    mo = compute_metrics(y[nm], oof[nm])
    mt = compute_metrics(y[tmp_idx][nm[tmp_idx]], tmp_p[nm[tmp_idx]])
    print(f"  [CV {track:12s}] {member:24s} | OOF {mo['mdape']:5.2f}% (€{mo['mae']:6,.0f}) | TMP {mt['mdape']:5.2f}%", flush=True)

    # Full-data refit of the `n_seeds` TabM models!
    full_df = data['full_feat_df']
    full_d = data['full_deflator']
    nn_spec_full = NNPreprocSpec.fit(full_df, cols)
    Xf = nn_spec_full.transform(full_df)
    yf = _target_fwd(y, a, full_d, target_mode).astype(np.float32)
    val_idxs_f = np.where(_es_split_mask(data['cluster_ids'], seed=77))[0]

    fitted_models = []
    for s in range(n_seeds):
        m = TabM_D_Regressor(
            device=device, n_epochs=n_epochs, batch_size=256, val_metric_name='mae',
            random_state=42 + s * 19, verbosity=0,
        )
        m.fit(Xf, yf, val_idxs=val_idxs_f)
        fitted_models.append(m)
        gc.collect()
        if device == 'mps':
            torch.mps.empty_cache()

    joblib.dump(
        {
            'nn_spec': nn_spec_full,
            'models': fitted_models,
            'oof': oof,
            'tmp': tmp_p,
        },
        out_path,
        compress=3,
    )
    print(f"  Stage 3 TabM ({track}/{member}) saved in {time.perf_counter()-t0:.1f}s", flush=True)


def stage_assemble():
    """Stage 4: Fit Huber stackers + heteroscedastic 80%/90% conformal models and export bundle."""
    import torch  # noqa: F401 — MUST precede lightgbm on Apple Silicon!
    import lightgbm as lgb
    from phase3_features import compute_metrics
    from production_pipeline import (
        ProductionBundle,
        TrackBundle,
        build_conformal_context,
    )

    t0 = time.perf_counter()
    print("\n=== STAGE 4: ASSEMBLE DUAL-TRACK PRODUCTION BUNDLE + CONFORMAL CALIBRATION ===", flush=True)
    data = joblib.load(os.path.join(STAGE_DIR, 'prod_splits.joblib'))
    full_pipe = joblib.load(os.path.join(STAGE_DIR, 'full_pipeline.joblib'))
    full_df = data['full_feat_df']
    y, folds, nm = data['y_price'], data['folds'], data['normal_mask']
    tmp_b = [b for b in data['bundles'] if str(b['id']).startswith('temporal')][0]
    tmp_idx = tmp_b['va_idx']
    nm_tmp = nm[tmp_idx]

    tracks: dict[str, TrackBundle] = {}
    metrics_report: dict[str, dict] = {}

    for track in ('tabular_text', 'multimodal'):
        gb = joblib.load(os.path.join(STAGE_DIR, f'gbdt_{track}.joblib'))
        tm_price = joblib.load(os.path.join(STAGE_DIR, f'tabm_perm170_price_4s_{track}.joblib'))
        tm_psqm = joblib.load(os.path.join(STAGE_DIR, f'tabm_perm170_psqm_4s_{track}.joblib'))

        O_eur = np.column_stack([
            gb['oof']['lgbmxt_price_hub'],
            gb['oof']['spec_lgbm_price_hub'],
            gb['oof']['cb_tuned_price_l2'],
            tm_price['oof'],
            tm_psqm['oof'],
        ])
        T_eur = np.column_stack([
            gb['tmp']['lgbmxt_price_hub'],
            gb['tmp']['spec_lgbm_price_hub'],
            gb['tmp']['cb_tuned_price_l2'],
            tm_price['tmp'],
            tm_psqm['tmp'],
        ])
        O_log = np.log(np.clip(O_eur, 10_000.0, 1e7))
        T_log = np.log(np.clip(T_eur, 10_000.0, 1e7))

        # Leak-free CV evaluation of Huber stacker
        oof_stack = np.zeros(len(y), dtype=np.float64)
        for fk in range(5):
            tr = (folds != fk) & nm
            va = folds == fk
            hr = HuberRegressor(alpha=15.0, max_iter=500).fit(O_log[tr], np.log(y[tr]))
            oof_stack[va] = np.clip(np.exp(hr.predict(O_log[va])), 10_000.0, 1e7)

        huber_final = HuberRegressor(alpha=15.0, max_iter=500).fit(O_log[nm], np.log(y[nm]))
        tmp_stack = np.clip(np.exp(huber_final.predict(T_log)), 10_000.0, 1e7)

        mo = compute_metrics(y[nm], oof_stack[nm])
        mt = compute_metrics(y[tmp_idx][nm_tmp], tmp_stack[nm_tmp])

        # Heteroscedastic Conformal Calibration (80% & 90% intervals)
        ens_std_oof = np.std(O_log, axis=1)
        ens_std_tmp = np.std(T_log, axis=1)
        abs_err_oof = np.abs(np.log(y) - np.log(oof_stack))
        abs_err_tmp = np.abs(np.log(y[tmp_idx]) - np.log(tmp_stack))

        X_ctx = build_conformal_context(full_df, np.log(oof_stack), ens_std_oof)
        X_ctx_tmp = build_conformal_context(full_df.iloc[tmp_idx].reset_index(drop=True), np.log(tmp_stack), ens_std_tmp)

        q_het_oof = np.zeros(len(y), dtype=np.float64)
        for fk in range(5):
            tr = (folds != fk) & nm
            va = folds == fk
            sm = lgb.LGBMRegressor(
                objective='mae', n_estimators=200, learning_rate=0.03, max_depth=3,
                num_leaves=7, min_child_samples=80, reg_lambda=20.0, random_state=42, verbose=-1,
            )
            sm.fit(X_ctx.iloc[tr], abs_err_oof[tr])
            s_tr = np.clip(sm.predict(X_ctx.iloc[tr]), 0.03, 0.50)
            s_va = np.clip(sm.predict(X_ctx.iloc[va]), 0.03, 0.50)
            q_r = float(np.quantile(abs_err_oof[tr] / s_tr, 0.80))
            q_het_oof[va] = q_r * s_va

        sm_final = lgb.LGBMRegressor(
            objective='mae', n_estimators=200, learning_rate=0.03, max_depth=3,
            num_leaves=7, min_child_samples=80, reg_lambda=20.0, random_state=42, verbose=-1,
        )
        sm_final.fit(X_ctx.iloc[nm], abs_err_oof[nm])
        s_tr_f = np.clip(sm_final.predict(X_ctx.iloc[nm]), 0.03, 0.50)
        q80_final = float(np.quantile(abs_err_oof[nm] / s_tr_f, 0.80))
        q90_final = float(np.quantile(abs_err_oof[nm] / s_tr_f, 0.90))
        s_tmp = np.clip(sm_final.predict(X_ctx_tmp), 0.03, 0.50)
        q_het_tmp = q80_final * s_tmp

        cov_o = float(np.mean(abs_err_oof[nm] <= q_het_oof[nm]) * 100.0)
        hw_o = float(np.median(np.expm1(q_het_oof[nm])) * 100.0)
        cov_t = float(np.mean(abs_err_tmp[nm_tmp] <= q_het_tmp[nm_tmp]) * 100.0)
        hw_t = float(np.median(np.expm1(q_het_tmp[nm_tmp])) * 100.0)

        print(f"\n  >>> {track.upper()} LEAN-5 STACK (agency-blind, no Qwen3):", flush=True)
        print(f"      Normal OOF : MdAPE={mo['mdape']:.2f}% | Hit@10={mo['hit_10']:.1f}% | Hit@15={mo['hit_15']:.1f}% | MAE=€{mo['mae']:,.0f} | R²={mo['r2']:.4f} | bias={mo['bias_pct']:+.2f}%", flush=True)
        print(f"      Normal TMP : MdAPE={mt['mdape']:.2f}% | Hit@10={mt['hit_10']:.1f}% | Hit@15={mt['hit_15']:.1f}% | MAE=€{mt['mae']:,.0f} | R²={mt['r2']:.4f} | bias={mt['bias_pct']:+.2f}%", flush=True)
        print(f"      Conformal80: OOF cov={cov_o:.2f}% (±{hw_o:.2f}%) | TMP cov={cov_t:.2f}% (±{hw_t:.2f}%) | q80={q80_final:.3f} q90={q90_final:.3f}", flush=True)
        print(f"      Huber coefs: {dict(zip(MEMBER_NAMES, np.round(huber_final.coef_, 4)))}", flush=True)

        tracks[track] = TrackBundle(
            name=track,
            tree_spec=gb['tree_spec'],
            nn_spec=tm_price['nn_spec'],
            lgbmxt=gb['lgbmxt'],
            spec_lgbm_house=gb['spec_lgbm_house'],
            spec_lgbm_apt=gb['spec_lgbm_apt'],
            catboost_model=gb['catboost_model'],
            tabm_price_models=tm_price['models'],
            tabm_psqm_models=tm_psqm['models'],
            huber_stacker=huber_final,
            conformal_sigma_model=sm_final,
            conformal_q80=q80_final,
            conformal_q90=q90_final,
            member_names=MEMBER_NAMES,
        )

        metrics_report[track] = {
            'n_gbdt_features': len(gb['tree_spec'].cols),
            'n_nn_features': len(tm_price['nn_spec'].cols),
            'oof_normal': {k: float(v) for k, v in mo.items()},
            'temporal_2026_normal': {k: float(v) for k, v in mt.items()},
            'conformal_80': {
                'oof_cov_pct': cov_o, 'oof_median_hw_pct': hw_o,
                'tmp_cov_pct': cov_t, 'tmp_median_hw_pct': hw_t,
                'q80': q80_final, 'q90': q90_final,
            },
            'huber_coefs': {n: float(c) for n, c in zip(MEMBER_NAMES, huber_final.coef_)},
            'huber_intercept': float(huber_final.intercept_),
        }

    try:
        git_sha = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], cwd=BASE_DIR, text=True).strip()
    except Exception:
        git_sha = 'unknown'

    metadata = {
        'model_version': f'phase7-lean5-{git_sha}',
        'git_sha': git_sha,
        'trained_at_utc': datetime.now(timezone.utc).isoformat(),
        'n_train_rows': int(len(full_df)),
        'n_normal_rows': int(nm.sum()),
        'agency_blind': True,
        'uses_qwen3': False,
        'deflator_formula': '0.5 * CPI + 0.5 * HPI (2025 median = 1.0)',
        'cpi_ref_2025': float(full_pipe.cpi_ref_2025),
        'hpi_ref_2025': float(full_pipe.hpi_ref_2025),
        'latest_deflator': float(full_pipe.latest_deflator),
        'members': MEMBER_NAMES,
        'tracks': metrics_report,
    }

    bundle = ProductionBundle(
        pipeline=full_pipe,
        tabular_text=tracks['tabular_text'],
        multimodal=tracks['multimodal'],
        metadata=metadata,
    )
    bundle.save(PROD_DIR)
    print(f"\n=== PRODUCTION BUNDLE SAVED -> {PROD_DIR} ({time.perf_counter()-t0:.1f}s) ===", flush=True)


def run_all(n_seeds: int = 4, n_epochs: int = 105):
    t_all = time.perf_counter()
    py = sys.executable
    script = os.path.abspath(__file__)

    def _run(args_list: list[str]):
        cmd = [py, '-u', script] + args_list
        print(f"\n>>> Subprocess: {' '.join(args_list)}", flush=True)
        subprocess.run(cmd, check=True)

    # Stage 1: features (skip if already staged)
    if not os.path.exists(os.path.join(STAGE_DIR, 'prod_splits.joblib')):
        _run(['--stage', 'features'])

    # Stage 2: CPU GBDT workers (isolated from torch!)
    for track in ('tabular_text', 'multimodal'):
        if not os.path.exists(os.path.join(STAGE_DIR, f'gbdt_{track}.joblib')):
            _run(['--stage', 'gbdt', '--track', track])

    # Stage 3: MPS TabM workers (each isolated in its own subprocess for clean MPS allocator!)
    for track in ('tabular_text', 'multimodal'):
        for tmode in ('log_real_price', 'log_real_psqm'):
            _run(['--stage', 'tabm', '--track', track, '--target-mode', tmode,
                  '--n-seeds', str(n_seeds), '--n-epochs', str(n_epochs)])

    # Stage 4: Assemble + calibrate + export
    _run(['--stage', 'assemble'])
    print(f"\n=== ALL STAGES COMPLETE in {(time.perf_counter()-t_all)/60:.1f} min ===", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['all', 'features', 'gbdt', 'tabm', 'assemble'], default='all')
    ap.add_argument('--track', choices=['tabular_text', 'multimodal'], default='tabular_text')
    ap.add_argument('--target-mode', choices=['log_real_price', 'log_real_psqm'], default='log_real_price')
    ap.add_argument('--n-seeds', type=int, default=4)
    ap.add_argument('--n-epochs', type=int, default=105)
    args = ap.parse_args()

    if args.stage == 'all':
        run_all(n_seeds=args.n_seeds, n_epochs=args.n_epochs)
    elif args.stage == 'features':
        stage_prepare_features()
    elif args.stage == 'gbdt':
        stage_gbdt(args.track)
    elif args.stage == 'tabm':
        stage_tabm(args.track, args.target_mode, n_seeds=args.n_seeds, n_epochs=args.n_epochs)
    elif args.stage == 'assemble':
        stage_assemble()


if __name__ == '__main__':
    main()
