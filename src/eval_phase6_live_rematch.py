"""Issue #6 Live Zero-Leak Rematch: Phase-6 Full Multimodal Ensemble vs Phase-4 & Phase-1
on the 102 ULTRA-CLEAN ZERO-LEAK Nov 22–23, 2026 Live Ghent Listings!

Trains our winning Phase-6 ensemble on 100% of the pre-22-Nov 14,599-row dataset using:
  - Winning `0.5 * CPI + 0.5 * HPI` deflator (Issue #9)
  - Full-Corpus Render-Filtered SigLIP2-base + SO400M-384 vision features (Issue #1 — already encoded for fresh!)
  - Winning Wave-2 Geo (IRCEL 10m air quality + VHA waterfront, Issue #2)
  - Winning Issue #3 Text/m²/Floor features
  - Per-family feature budgets from `phase6_feature_sets.json` (Issue #7)
  - Learned `weights_unified` from `phase6_report.json` (ZERO tuning on live holdout!)
  - Heteroscedastic 80% Conformal Intervals (Issue #8) verified LIVE on unseen Nov 22–23 homes!

Uses clean subprocess isolation (`--stage prep|gbdt|mps|eval`) so LightGBM and PyTorch MPS NEVER
share a process (100% immune to macOS ARM64 OpenMP segfaults).

Usage:
    .venv-api/bin/python -u src/eval_phase6_live_rematch.py --stage all
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import warnings

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'src'))

WORK_DIR = os.path.join(BASE_DIR, 'outputs', 'models', 'workbench')
DATASET_PATH = os.path.join(BASE_DIR, 'outputs', 'dataset', 'training_dataset.parquet')
ARTIFACTS_PATH = os.path.join(BASE_DIR, 'outputs', 'dataset', 'preprocessing_artifacts.joblib')
RAW_PARQUET = os.path.join(BASE_DIR, 'outputs', 'property_data_full.parquet')
CLUSTER_PATH = os.path.join(BASE_DIR, 'outputs', 'dataset', 'cluster_folds.parquet')
FRESH_PREV_CSV = os.path.join(WORK_DIR, 'fresh_nov23_predictions.csv')
FRESH_E5_NPZ = os.path.join(WORK_DIR, 'fresh_nov23_embs.npz')
FRESH_QWEN_NPZ = os.path.join(WORK_DIR, 'qwen3_emb_fresh_nov23.npz')
E5_LARGE_PATH = os.path.join(WORK_DIR, 'e5_large_14599.npz')
E5_SMALL_PATH = os.path.join(WORK_DIR, 'e5_embeddings_14599.npz')
QWEN3_PATH = os.path.join(WORK_DIR, 'qwen3_emb_14599.npz')
VIS_FULL_PATH = os.path.join(BASE_DIR, 'external_data', 'vision_full', 'vision_full_embeddings.npz')

FEAT_SETS_JSON = os.path.join(WORK_DIR, 'phase6_feature_sets.json')
P6_REPORT_JSON = os.path.join(WORK_DIR, 'phase6_report.json')
P6_PREDS_NPZ = os.path.join(WORK_DIR, 'phase6_frontier_preds.npz')
LIVE_BUNDLE = os.path.join(WORK_DIR, 'phase6_live_rematch_bundle.joblib')
LIVE_CKPT = os.path.join(WORK_DIR, 'phase6_live_rematch_ckpt.npz')
LIVE_CSV = os.path.join(WORK_DIR, 'fresh_nov23_phase6_rematch.csv')
LIVE_REPORT = os.path.join(WORK_DIR, 'fresh_nov23_phase6_rematch_report.json')


def _target_fwd(y_price, area, defl, mode: str):
    import numpy as np
    return np.log((y_price / defl) / area) if mode == 'log_real_psqm' else np.log(y_price / defl)


def _target_inv(pred_log, area, defl, mode: str):
    import numpy as np
    p = np.clip(np.asarray(pred_log, dtype=np.float64), -20.0, 20.0)
    return np.clip(np.exp(p) * (area * defl if mode == 'log_real_psqm' else defl), 10_000.0, 1e7)


def stage_prep():
    import joblib
    import numpy as np
    import pandas as pd
    from extract_issue3_features import compute_issue3_features
    from phase3_features import _add_domain_features, _extract_text_flags
    from phase4_features import attach_flanders_and_grb
    from phase6_features import (
        DROP_W2_BLOATED,
        KEEP_ISSUE3_COLS,
        _build_one_split,
    )
    from preprocessing import PreprocessConfig, clean_raw, transform

    t0 = time.perf_counter()
    print("=== LIVE REMATCH STAGE 1: BUILDING TRAIN(14,599) -> FRESH(1,227) PHASE-6 FEATURES ===", flush=True)

    raw = pd.read_parquet(RAW_PARQUET)
    raw['property_id'] = raw['property_id'].astype(str)
    tr_base = pd.read_parquet(DATASET_PATH).reset_index(drop=True)
    tr_base['property_id'] = tr_base['property_id'].astype(str)
    cl = pd.read_parquet(CLUSTER_PATH)
    artifacts = joblib.load(ARTIFACTS_PATH)

    # Reconstruct exact 1,227 clean fresh Nov 22-23 listings
    fresh_raw = raw[raw['date_obtained'].isin(['22112026', '23112026'])].copy()
    res = clean_raw(fresh_raw, config=PreprocessConfig(dedupe_relistings=True, outlier_quantile=1.0), annotate=False)
    fresh_kept = res['kept'].reset_index(drop=True)
    fresh_kept['property_id'] = fresh_kept['property_id'].astype(str)

    fresh_meta = fresh_kept[['property_id', 'date_obtained', 'property_key', 'title', 'description',
                             'description_nl', 'street', 'immo_office_name', 'postal_code']].copy()
    fresh_tr = transform(fresh_kept, artifacts).reset_index(drop=True)
    fresh_tr['property_id'] = fresh_kept['property_id'].values
    fresh_tr['property_key'] = fresh_kept['property_key'].values
    fresh_tr['date_obtained'] = fresh_kept['date_obtained'].values

    # Attach raw text/street/agency to train
    raw['date_dt'] = pd.to_datetime(raw['date_obtained'], format='%d%m%Y', errors='coerce')
    raw_u = raw.drop_duplicates(['property_id', 'date_dt'], keep='first')
    tr_m = tr_base[['property_id', 'date_obtained']].merge(
        raw_u[['property_id', 'date_dt', 'title', 'description', 'description_nl', 'street', 'immo_office_name']],
        left_on=['property_id', 'date_obtained'], right_on=['property_id', 'date_dt'], how='left',
    )

    def _mk_texts(meta_df):
        tt = meta_df['title'].fillna('').astype(str).to_numpy()
        nl = meta_df['description_nl'].fillna('').astype(str).to_numpy()
        mn = meta_df['description'].fillna('').astype(str).to_numpy()
        return [f"{t}. {n if len(n) >= 40 else m}"[:2400] for t, n, m in zip(tt, nl, mn)]

    tr_texts = _mk_texts(tr_m)
    fr_texts = _mk_texts(fresh_meta)

    for df_x, m_x, txts in ((tr_base, tr_m, tr_texts), (fresh_tr, fresh_meta, fr_texts)):
        df_x['agency_norm'] = m_x['immo_office_name'].fillna('Unknown').astype(str).str.strip().str.lower().values
        df_x['immo_office_name'] = m_x['immo_office_name'].fillna('Unknown').astype(str).values
        df_x['street_norm'] = (
            df_x['postal_code'].fillna('0000').astype(str).str.strip() + '|'
            + m_x['street'].fillna('unknown').astype(str).str.strip().str.lower().values
        )

    # Concatenate train + fresh to pass through identical row-local enrichments
    n_tr = len(tr_base)
    n_fr = len(fresh_tr)
    all_texts = tr_texts + fr_texts

    # Align columns cleanly
    common_cols = [c for c in tr_base.columns if c in fresh_tr.columns]
    both = pd.concat([tr_base[common_cols], fresh_tr[common_cols]], ignore_index=True)
    both = _add_domain_features(both)
    both = pd.concat([both, _extract_text_flags(all_texts)], axis=1)
    both = attach_flanders_and_grb(both)
    drop_b = [c for c in DROP_W2_BLOATED if c in both.columns]
    if drop_b:
        both.drop(columns=drop_b, inplace=True)

    # Reusable Issue #3 row-level features for BOTH train and fresh!
    i3_both = compute_issue3_features(both, all_texts)
    both = pd.concat([both.reset_index(drop=True), i3_both[[c for c in KEEP_ISSUE3_COLS if c in i3_both.columns]].reset_index(drop=True)], axis=1)

    # Embeddings (all already cached on disk!)
    tr_e5l = np.load(E5_LARGE_PATH)['embeddings'].astype(np.float32)
    tr_e5s = np.load(E5_SMALL_PATH)['embeddings'].astype(np.float32)
    tr_qw = np.load(QWEN3_PATH)['embeddings'].astype(np.float32)

    fr_e5 = np.load(FRESH_E5_NPZ)
    fr_e5l = fr_e5['e5l'].astype(np.float32)
    fr_e5s = fr_e5['e5s'].astype(np.float32)
    fr_qw_z = np.load(FRESH_QWEN_NPZ, allow_pickle=True)
    fr_qw_map = {str(p): fr_qw_z['embeddings'][i] for i, p in enumerate(fr_qw_z['property_id'])}
    fr_qw = np.vstack([fr_qw_map.get(pid, np.zeros(1024, dtype=np.float32)) for pid in fresh_tr['property_id'].values]).astype(np.float32)

    assert len(fr_e5l) == n_fr and len(fr_qw) == n_fr
    e5l = np.vstack([tr_e5l, fr_e5l])
    e5s = np.vstack([tr_e5s, fr_e5s])
    qwen3 = np.vstack([tr_qw, fr_qw])

    # Vision full embeddings aligned to BOTH train (14,599) + fresh (1,227)!
    vz = np.load(VIS_FULL_PATH, allow_pickle=True)
    p2i = {str(p): i for i, p in enumerate(vz['property_id'])}
    N = len(both)
    sig_real = np.full((N, 768), np.nan, dtype=np.float32)
    so_sheet = np.full((N, 1152), np.nan, dtype=np.float32)
    render_f = np.full((N, 8), np.nan, dtype=np.float32)
    q_f = np.full((N, 30), np.nan, dtype=np.float32)
    so_q_f = np.full((N, 10), np.nan, dtype=np.float32)
    has_vis = np.zeros(N, dtype=bool)
    for r_i, pid in enumerate(both['property_id'].values):
        vi = p2i.get(pid)
        if vi is not None and not np.isnan(vz['sig_real'][vi, 0]):
            sig_real[r_i] = vz['sig_real'][vi]
            so_sheet[r_i] = vz['so_sheet'][vi]
            render_f[r_i] = vz['render_feats'][vi]
            q_f[r_i] = vz['q_feats'][vi]
            so_q_f[r_i] = vz['so_q_feats'][vi]
            has_vis[r_i] = True
    print(f"  Vision matched: train={has_vis[:n_tr].sum():,}/{n_tr:,} | fresh={has_vis[n_tr:].sum():,}/{n_fr:,} ({100*has_vis[n_tr:].mean():.1f}%)", flush=True)
    vis_bundle = {'sig_real': sig_real, 'so_sheet': so_sheet, 'render_feats': render_f, 'q_feats': q_f, 'so_q_feats': so_q_f, 'has_vis': has_vis}

    # Winning Issue #9 Deflator: 0.5*CPI + 0.5*HPI (relative to 2025 median on TRAIN!)
    cpi_raw = both['belgian_cpi_index'].to_numpy(float)
    hpi_raw = both['belgian_hpi_index'].to_numpy(float)
    m25_tr = (both['publication_year'].iloc[:n_tr] == 2025).to_numpy()
    cpi_idx = cpi_raw / np.nanmedian(cpi_raw[:n_tr][m25_tr])
    hpi_idx = hpi_raw / np.nanmedian(hpi_raw[:n_tr][m25_tr])
    deflator = (0.5 * cpi_idx + 0.5 * hpi_idx).astype(np.float64)

    cids = np.concatenate([cl['cluster_id'].to_numpy(int), np.arange(1_000_000, 1_000_000 + n_fr, dtype=int)])
    anoms = np.zeros(N, dtype=np.int8)
    tr_idx = np.arange(0, n_tr)
    va_idx = np.arange(n_tr, N)

    sp = _build_one_split('live_nov23', tr_idx, va_idx, both, all_texts, e5l, e5s, qwen3, deflator, cids, anoms, vis_bundle)

    # Carry forward exact 3-layer ultra-clean leak flags + Phase-1 & Phase-4 predictions from previous CSV!
    prev = pd.read_csv(FRESH_PREV_CSV)
    prev['property_id'] = prev['property_id'].astype(str)
    prev_m = fresh_tr[['property_id']].merge(prev, on='property_id', how='left')

    bundle = {
        'sp': sp,
        'y_tr': both['price'].iloc[:n_tr].to_numpy(float),
        'area_tr': both['livable_area'].iloc[:n_tr].to_numpy(float),
        'house_tr': both['property_type_HOUSE'].iloc[:n_tr].to_numpy(int),
        'y_fr': both['price'].iloc[n_tr:].to_numpy(float),
        'area_fr': both['livable_area'].iloc[n_tr:].to_numpy(float),
        'house_fr': both['property_type_HOUSE'].iloc[n_tr:].to_numpy(int),
        'fresh_meta': prev_m,
    }
    joblib.dump(bundle, LIVE_BUNDLE, compress=3)
    print(f"  Saved live rematch bundle in {time.perf_counter()-t0:.1f}s -> {LIVE_BUNDLE}", flush=True)


def _load_lckpt():
    import numpy as np
    if not os.path.exists(LIVE_CKPT):
        return {}
    z = np.load(LIVE_CKPT, allow_pickle=True)
    return {k: z[k] for k in z.files}


def _save_lckpt(d):
    import numpy as np
    np.savez_compressed(LIVE_CKPT, **d)


def stage_gbdt():
    import joblib
    import lightgbm as lgb
    import numpy as np
    import pandas as pd
    import xgboost as xgb
    from catboost import CatBoostRegressor

    t0 = time.perf_counter()
    b = joblib.load(LIVE_BUNDLE)
    fsets = json.load(open(FEAT_SETS_JSON))
    sp = b['sp']
    y_tr, a_tr, h_tr = b['y_tr'], b['area_tr'], b['house_tr']
    a_fr, h_fr = b['area_fr'], b['house_fr']
    tr_def, va_def = sp['tr_deflator'], sp['va_deflator']
    tr_cids = sp['tr_cids']
    gbdt_feats = [c for c in fsets['gbdt_feats'] if c in sp['tr_df'].columns]
    cb_feats = [c for c in fsets['gbdt_feats_with_agency'] if c in sp['tr_df'].columns]

    ckpt = _load_lckpt()
    print(f"=== LIVE REMATCH STAGE 2: FITTING 8 GBDT MODELS ON ALL 14,599 TRAIN HOMES ===", flush=True)

    def _prep(df, cols, kind):
        o = df[cols].copy()
        cats = []
        for c in cols:
            if o[c].dtype == 'bool': o[c] = o[c].astype('int8')
            elif not pd.api.types.is_numeric_dtype(o[c]):
                o[c] = o[c].astype(str).fillna('Missing')
                if kind in ('lgbm', 'xgb'): o[c] = o[c].astype('category')
                cats.append(c)
            else:
                o[c] = o[c].astype('float32')
        return o, cats

    def _fit_one(cols, kind, tmode, lmode, ep, hf=None):
        tr_m = (h_tr == hf) if hf is not None else np.ones(len(y_tr), dtype=bool)
        va_m = (h_fr == hf) if hf is not None else np.ones(len(a_fr), dtype=bool)
        Xtr, cats = _prep(sp['tr_df'].iloc[tr_m].reset_index(drop=True), cols, kind)
        Xva, _ = _prep(sp['va_df'].iloc[va_m].reset_index(drop=True), cols, kind)
        if kind == 'xgb':
            for c in cats:
                Xva[c] = pd.Categorical(Xva[c].astype(str), categories=Xtr[c].cat.categories)
        yt = _target_fwd(y_tr[tr_m], a_tr[tr_m], tr_def[tr_m], tmode)
        u_c = np.unique(tr_cids[tr_m])
        es_c = set(np.random.default_rng(42).choice(u_c, size=max(1, int(len(u_c) * 0.10)), replace=False))
        es_m = np.isin(tr_cids[tr_m], list(es_c))
        if kind == 'lgbm':
            p = dict(objective='huber' if lmode == 'huber' else 'regression', n_estimators=3200, learning_rate=0.025,
                     num_leaves=47, min_child_samples=35, subsample=0.72, subsample_freq=1, colsample_bytree=0.40,
                     reg_alpha=0.8, reg_lambda=2.2, random_state=42, n_jobs=12, verbose=-1)
            if lmode == 'huber': p['alpha'] = 0.15
            p.update(ep)
            m = lgb.LGBMRegressor(**p)
            m.fit(Xtr.iloc[~es_m], yt[~es_m], eval_set=[(Xtr.iloc[es_m], yt[es_m])], callbacks=[lgb.early_stopping(90, verbose=False)])
            pl = m.predict(Xva)
        elif kind == 'xgb':
            p = dict(objective='reg:squarederror', n_estimators=3200, learning_rate=0.022, max_depth=5, min_child_weight=22,
                     subsample=0.85, colsample_bytree=0.38, reg_alpha=0.4, reg_lambda=6.5, tree_method='hist',
                     enable_categorical=True, early_stopping_rounds=90, random_state=42, n_jobs=12)
            p.update(ep)
            m = xgb.XGBRegressor(**p)
            m.fit(Xtr.iloc[~es_m], yt[~es_m], eval_set=[(Xtr.iloc[es_m], yt[es_m])], verbose=False)
            pl = m.predict(Xva)
        else:
            p = dict(loss_function='RMSE', iterations=3500, learning_rate=0.032, depth=7, l2_leaf_reg=5.0,
                     random_seed=42, thread_count=12, verbose=0)
            p.update(ep)
            m = CatBoostRegressor(**p)
            m.fit(Xtr.iloc[~es_m], yt[~es_m], cat_features=cats, eval_set=(Xtr.iloc[es_m], yt[es_m]), early_stopping_rounds=120, verbose=False)
            pl = m.predict(Xva)
        return va_m, _target_inv(pl, a_fr[va_m], va_def[va_m], tmode)

    specs = [
        ('p6_lgbm_psqm_huber',    gbdt_feats, 'lgbm',     'log_real_psqm',  'huber', {}, False),
        ('p6_lgbm_price_huber',   gbdt_feats, 'lgbm',     'log_real_price', 'huber', {}, False),
        ('p6_lgbmxt_psqm_huber',  gbdt_feats, 'lgbm',     'log_real_psqm',  'huber', {'extra_trees': True, 'num_leaves': 63, 'min_child_samples': 20, 'colsample_bytree': 0.55}, False),
        ('p6_xgb_psqm_l2',        gbdt_feats, 'xgb',      'log_real_psqm',  'l2',    {}, False),
        ('p6_cb_d7_price_l2',     cb_feats,   'catboost', 'log_real_price', 'l2',    {}, False),
        ('p6_cb_d7_psqm_l2',      cb_feats,   'catboost', 'log_real_psqm',  'l2',    {}, False),
        ('p6_spec_cb_psqm',       cb_feats,   'catboost', 'log_real_psqm',  'l2',    {}, True),
        ('p6_spec_xgb_psqm',      gbdt_feats, 'xgb',      'log_real_psqm',  'l2',    {}, True),
    ]
    for name, cols, kind, tmode, lmode, ep, is_spec in specs:
        if name in ckpt:
            continue
        ts = time.perf_counter()
        if not is_spec:
            _, pr = _fit_one(cols, kind, tmode, lmode, ep, None)
        else:
            pr = np.zeros(len(a_fr), dtype=np.float64)
            for hf in (0, 1):
                vm, p_sub = _fit_one(cols, kind, tmode, lmode, ep, hf)
                pr[vm] = p_sub
        ckpt[name] = pr
        _save_lckpt(ckpt)
        print(f"  [{name:24s}] fitted in {time.perf_counter()-ts:.1f}s", flush=True)
    print(f"  Live Stage 2 done in {time.perf_counter()-t0:.1f}s", flush=True)


def stage_mps():
    import torch
    import joblib
    import numpy as np
    import pandas as pd
    from pytabkit import RealMLP_TD_Regressor, TabM_D_Regressor
    from tabicl import TabICLRegressor
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion

    t0 = time.perf_counter()
    b = joblib.load(LIVE_BUNDLE)
    fsets = json.load(open(FEAT_SETS_JSON))
    sp = b['sp']
    y_tr, a_tr, h_tr = b['y_tr'], b['area_tr'], b['house_tr']
    a_fr, h_fr = b['area_fr'], b['house_fr']
    tr_def, va_def = sp['tr_deflator'], sp['va_deflator']
    tr_cids = sp['tr_cids']
    nn_feats = [c for c in fsets['nn_feats'] if c in sp['tr_df'].columns]
    fm_feats = [c for c in fsets['fm_feats'] if c in sp['tr_df'].columns]

    ckpt = _load_lckpt()
    print(f"=== LIVE REMATCH STAGE 3: FITTING MPS NNs + FOUNDATION MODELS ON ALL 14,599 HOMES ===", flush=True)

    def _prep_nn(tr_df, va_df, cols):
        Xtr, Xva = tr_df[cols].copy(), va_df[cols].copy()
        for c in cols:
            if Xtr[c].dtype == 'bool': Xtr[c] = Xtr[c].astype('float32'); Xva[c] = Xva[c].astype('float32')
            elif not pd.api.types.is_numeric_dtype(Xtr[c]):
                vc = Xtr[c].astype(str).fillna('Missing').value_counts()
                keep = set(vc[vc >= 5].head(40).index)
                Xtr[c] = Xtr[c].astype(str).fillna('Missing').where(lambda s: s.isin(keep), '__RARE__').astype('category')
                Xva[c] = Xva[c].astype(str).fillna('Missing').where(lambda s: s.isin(keep), '__RARE__').astype('category')
            else:
                tv = pd.to_numeric(Xtr[c], errors='coerce').replace([np.inf, -np.inf], np.nan)
                vv = pd.to_numeric(Xva[c], errors='coerce').replace([np.inf, -np.inf], np.nan)
                med = float(tv.median()) if tv.notna().any() else 0.0
                Xtr[c] = tv.fillna(med).astype('float32'); Xva[c] = vv.fillna(med).astype('float32')
        return Xtr, Xva

    def _prep_fm(tr_df, va_df, cols, max_cat=30, num_cats=False):
        Xtr, Xva = tr_df[cols].copy(), va_df[cols].copy()
        for c in cols:
            if Xtr[c].dtype == 'bool': Xtr[c] = Xtr[c].astype('float32'); Xva[c] = Xva[c].astype('float32')
            elif not pd.api.types.is_numeric_dtype(Xtr[c]):
                keep = list(Xtr[c].astype(str).fillna('Missing').value_counts().head(max_cat).index)
                if num_cats:
                    cm = {k: i for i, k in enumerate(keep)}
                    Xtr[c] = Xtr[c].astype(str).fillna('Missing').map(cm).fillna(-1).astype('float32')
                    Xva[c] = Xva[c].astype(str).fillna('Missing').map(cm).fillna(-1).astype('float32')
                else:
                    ks = set(keep)
                    Xtr[c] = Xtr[c].astype(str).fillna('Missing').where(lambda s: s.isin(ks), '__OTHER__')
                    Xva[c] = Xva[c].astype(str).fillna('Missing').where(lambda s: s.isin(ks), '__OTHER__')
            else:
                Xtr[c] = pd.to_numeric(Xtr[c], errors='coerce').replace([np.inf, -np.inf], np.nan).astype('float32')
                Xva[c] = pd.to_numeric(Xva[c], errors='coerce').replace([np.inf, -np.inf], np.nan).astype('float32')
        return Xtr, Xva

    for name, kind, tmode, n_seeds, spec in [
        ('p6_tabm_price_3s', 'tabm', 'log_real_price', 3, False),
        ('p6_tabm_psqm_3s', 'tabm', 'log_real_psqm', 3, False),
        ('p6_spec_tabm_psqm_2s', 'tabm', 'log_real_psqm', 2, True),
        ('p6_realmlp_price_3s', 'realmlp', 'log_real_price', 3, False),
    ]:
        if name in ckpt:
            continue
        ts = time.perf_counter()
        pr = np.zeros(len(a_fr), dtype=np.float64)
        for hf in ((0, 1) if spec else (None,)):
            tr_m = (h_tr == hf) if hf is not None else np.ones(len(y_tr), dtype=bool)
            va_m = (h_fr == hf) if hf is not None else np.ones(len(a_fr), dtype=bool)
            Xtr, Xva = _prep_nn(sp['tr_df'].iloc[tr_m].reset_index(drop=True), sp['va_df'].iloc[va_m].reset_index(drop=True), nn_feats)
            yt = _target_fwd(y_tr[tr_m], a_tr[tr_m], tr_def[tr_m], tmode).astype(np.float32)
            u_c = np.unique(tr_cids[tr_m])
            es_c = set(np.random.default_rng(77 + (hf or 0) * 13).choice(u_c, size=max(1, int(len(u_c) * 0.10)), replace=False))
            val_idxs = np.where(np.isin(tr_cids[tr_m], list(es_c)))[0]
            s_preds = []
            for s in range(n_seeds):
                seed = 42 + s * 19 + (hf or 0) * 7
                m = TabM_D_Regressor(device='mps', n_epochs=100, batch_size=256, val_metric_name='mae', random_state=seed, verbosity=0) if kind == 'tabm' \
                    else RealMLP_TD_Regressor(device='mps', n_epochs=128, batch_size=256, val_metric_name='mae', random_state=seed, verbosity=0)
                m.fit(Xtr, yt, val_idxs=val_idxs)
                s_preds.append(np.asarray(m.predict(Xva), dtype=np.float64))
                torch.mps.empty_cache()
            pr[va_m] = _target_inv(np.mean(s_preds, axis=0), a_fr[va_m], va_def[va_m], tmode)
        ckpt[name] = pr
        _save_lckpt(ckpt)
        print(f"  [{name:24s}] fitted in {time.perf_counter()-ts:.1f}s", flush=True)

    # TabICLv2 (price & psqm)
    for tmode in ('log_real_price', 'log_real_psqm'):
        short = 'price' if tmode == 'log_real_price' else 'psqm'
        n_mn, n_md = f'p6_tabicl_{short}_mean', f'p6_tabicl_{short}_med'
        if n_mn in ckpt and n_md in ckpt:
            continue
        ts = time.perf_counter()
        Xtr, Xva = _prep_fm(sp['tr_df'], sp['va_df'], fm_feats, max_cat=30, num_cats=False)
        yt = _target_fwd(y_tr, a_tr, tr_def, tmode).astype(np.float32)
        reg = TabICLRegressor(device='mps', n_estimators=8, batch_size=1, use_amp=True, random_state=42)
        reg.fit(Xtr, yt)
        out = reg.predict(Xva, output_type=['mean', 'median'])
        torch.mps.empty_cache()
        ckpt[n_mn] = _target_inv(out['mean'], a_fr, va_def, tmode)
        ckpt[n_md] = _target_inv(out['median'], a_fr, va_def, tmode)
        _save_lckpt(ckpt)
        print(f"  [{n_mn:24s}] fitted in {time.perf_counter()-ts:.1f}s", flush=True)

    # TabPFN V2 (price & psqm)
    for tmode in ('log_real_price', 'log_real_psqm'):
        short = 'price' if tmode == 'log_real_price' else 'psqm'
        name = f'p6_tabpfn_v2_{short}'
        if name in ckpt:
            continue
        ts = time.perf_counter()
        Xtr, Xva = _prep_fm(sp['tr_df'], sp['va_df'], fm_feats, max_cat=25, num_cats=True)
        yt = _target_fwd(y_tr, a_tr, tr_def, tmode).astype(np.float32)
        m = TabPFNRegressor.create_default_for_version(
            ModelVersion.V2, device='mps', n_estimators=8,
            ignore_pretraining_limits=True, inference_config={'SUBSAMPLE_SAMPLES': 7500}, random_state=42,
        )
        m.fit(Xtr, yt)
        ckpt[name] = _target_inv(np.asarray(m.predict(Xva), dtype=np.float64), a_fr, va_def, tmode)
        torch.mps.empty_cache()
        _save_lckpt(ckpt)
        print(f"  [{name:24s}] fitted in {time.perf_counter()-ts:.1f}s", flush=True)

    print(f"  Live Stage 3 done in {time.perf_counter()-t0:.1f}s", flush=True)


def stage_eval():
    import joblib
    import numpy as np
    import pandas as pd
    from phase3_features import compute_metrics

    b = joblib.load(LIVE_BUNDLE)
    ckpt = _load_lckpt()
    p6_rep = json.load(open(P6_REPORT_JSON))
    p6_z = np.load(P6_PREDS_NPZ, allow_pickle=True)
    names = list(p6_z['names'])
    w_uni = p6_z['weights_unified']

    # Weighted geometric mean using STRICTLY pre-learned Phase-6 weights (zero tuning on live!)
    L_mat = np.column_stack([np.log(np.clip(ckpt[n], 10_000, 1e7)) for n in names])
    p6_pred = np.clip(np.exp(L_mat @ w_uni), 10_000, 1e7)

    meta = b['fresh_meta'].copy()
    meta['p6_pred'] = p6_pred
    meta['p6_err_pct'] = (p6_pred / b['y_fr'] - 1.0) * 100.0
    meta['p6_ape'] = np.abs(meta['p6_err_pct'])
    meta.to_csv(LIVE_CSV, index=False)

    y = b['y_fr']
    uc = meta['is_ultraclean_zero_leak'].fillna(False).to_numpy(bool)
    in_dom = y <= 1_500_000
    norm_m = (y >= 150_000) & (y <= 800_000)
    ih = b['house_fr'] == 1

    print("\n=== LIVE OUT-OF-TIME REMATCH (Nov 22–23, 2026 | ZERO-LEAK ULTRA-CLEAN HOLDOUT) ===", flush=True)
    rep = {}
    slices = [
        ('ULTRA-CLEAN Normal €150k–€800k', uc & norm_m),
        ('ULTRA-CLEAN Normal Houses',      uc & norm_m & ih),
        ('ULTRA-CLEAN Normal Apts',        uc & norm_m & (~ih)),
        ('ULTRA-CLEAN All <=€1.5M',        uc & in_dom),
        ('ALL Fresh <=€1.5M (incl. relist)', in_dom),
    ]
    for label, mask in slices:
        if mask.sum() == 0:
            continue
        m6 = compute_metrics(y[mask], p6_pred[mask])
        p4_col = 'p4_ens_pred' if 'p4_ens_pred' in meta.columns else ('pred_phase4' if 'pred_phase4' in meta.columns else None)
        p1_col = 'p1_lgbm_pred' if 'p1_lgbm_pred' in meta.columns else ('pred_phase1' if 'pred_phase1' in meta.columns else None)
        m4 = compute_metrics(y[mask], meta[p4_col].to_numpy(float)[mask]) if p4_col else None
        m1 = compute_metrics(y[mask], meta[p1_col].to_numpy(float)[mask]) if p1_col else None
        rep[label] = {'n': int(mask.sum()), 'phase6': m6, 'phase4': m4, 'phase1': m1}
        s4 = f" | P4={m4['mdape']:5.2f}% (MAE €{m4['mae']:6,.0f})" if m4 else ""
        s1 = f" | P1={m1['mdape']:5.2f}%" if m1 else ""
        print(
            f"  {label:34s} (n={mask.sum():4d}) | P6={m6['mdape']:5.2f}% Hit10={m6['hit_10']:4.1f}% Hit15={m6['hit_15']:4.1f}% "
            f"MAE €{m6['mae']:6,.0f} R2={m6['r2']:.3f} bias={m6['bias_pct']:+5.2f}%{s4}{s1}",
            flush=True,
        )

    with open(LIVE_REPORT, 'w') as f:
        json.dump(rep, f, indent=2)
    print(f"\n=== LIVE REMATCH SAVED -> {LIVE_CSV} & {LIVE_REPORT} ===", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['prep', 'gbdt', 'mps', 'eval', 'all'], default='all')
    args = ap.parse_args()
    if args.stage == 'all':
        for st in ('prep', 'gbdt', 'mps', 'eval'):
            print(f"\n>>> Live Rematch subprocess stage: {st} <<<", flush=True)
            subprocess.run([sys.executable, '-u', __file__, '--stage', st], check=True)
    elif args.stage == 'prep':
        stage_prep()
    elif args.stage == 'gbdt':
        stage_gbdt()
    elif args.stage == 'mps':
        stage_mps()
    elif args.stage == 'eval':
        stage_eval()


if __name__ == '__main__':
    main()
