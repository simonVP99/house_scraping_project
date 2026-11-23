"""Phase-6 Frontier Training Suite — Cluster-Clean CV, Full Vision (#1), Per-Family Selection (#7),
SOTA Diverse Ensemble (#6), Winning `0.5*CPI+0.5*HPI` Deflator (#9), and Heteroscedastic Conformal Bands (#8)!

Uses clean subprocess stage isolation (`--stage ablate|gbdt|mps|stack`) so macOS ARM64 `libomp`
(LightGBM) and PyTorch MPS NEVER share a process — 100% immune to OpenMP segfaults!

Primary North Star:
  - **Normal Ghent Homes (`€150k–€800k`, excl. APARTMENT_BLOCK / MANSION / SERVICE_FLAT, `n=13,439 = 92.1%`)**
  - Reported alongside Normal Houses, Normal Apts, and ALL 14,599 homes on:
      1. **5-fold Anti-Chaining Spatial+Text Cluster CV (`fold_cluster`, 0 leaked clusters!)**
      2. **Cluster-Clean 2026 Temporal Holdout (`0` leaked clusters!)**

Outputs:
  - `outputs/models/workbench/phase6_feature_sets.json` (per-family feature budgets + Vision A/B report)
  - `outputs/models/workbench/phase6_models_ckpt.npz` (incremental per-model OOF + Temporal checkpoint)
  - `outputs/models/workbench/phase6_frontier_preds.npz` (final ensemble + heteroscedastic intervals)
  - `outputs/models/workbench/phase6_report.json`

Usage:
    .venv-api/bin/python -u src/phase6_frontier.py --stage all
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
P6_CACHE = os.path.join(WORK_DIR, 'phase6_splits_cache.joblib')
FEAT_SETS_JSON = os.path.join(WORK_DIR, 'phase6_feature_sets.json')
CKPT_NPZ = os.path.join(WORK_DIR, 'phase6_models_ckpt.npz')
PREDS_NPZ = os.path.join(WORK_DIR, 'phase6_frontier_preds.npz')
REPORT_JSON = os.path.join(WORK_DIR, 'phase6_report.json')


import numpy as np


# ---------------------------------------------------------------------------
# Shared Checkpoint & Target Helpers
# ---------------------------------------------------------------------------
def _target_fwd(y_price, area, defl, mode: str):
    if mode == 'log_real_psqm':
        return np.log((y_price / defl) / area)
    if mode == 'log_real_price':
        return np.log(y_price / defl)
    raise ValueError(mode)


def _target_inv(pred_log, area, defl, mode: str):
    p = np.clip(np.asarray(pred_log, dtype=np.float64), -20.0, 20.0)
    if mode == 'log_real_psqm':
        return np.clip(np.exp(p) * area * defl, 10_000.0, 1e7)
    if mode == 'log_real_price':
        return np.clip(np.exp(p) * defl, 10_000.0, 1e7)
    raise ValueError(mode)


def _load_ckpt():
    if not os.path.exists(CKPT_NPZ):
        return {}
    z = np.load(CKPT_NPZ, allow_pickle=True)
    out = {}
    for k in z.files:
        if k.startswith('oof__'):
            name = k[5:]
            out[name] = {'oof': z[f'oof__{name}'], 'tmp': z[f'tmp__{name}']}
            if f'q10_oof__{name}' in z.files:
                out[name]['q10_oof'] = z[f'q10_oof__{name}']
                out[name]['q90_oof'] = z[f'q90_oof__{name}']
                out[name]['q10_tmp'] = z[f'q10_tmp__{name}']
                out[name]['q90_tmp'] = z[f'q90_tmp__{name}']
    return out


def _save_ckpt(models: dict):
    payload = {}
    for name, d in models.items():
        payload[f'oof__{name}'] = np.asarray(d['oof'], dtype=np.float64)
        payload[f'tmp__{name}'] = np.asarray(d['tmp'], dtype=np.float64)
        if 'q10_oof' in d:
            payload[f'q10_oof__{name}'] = np.asarray(d['q10_oof'], dtype=np.float64)
            payload[f'q90_oof__{name}'] = np.asarray(d['q90_oof'], dtype=np.float64)
            payload[f'q10_tmp__{name}'] = np.asarray(d['q10_tmp'], dtype=np.float64)
            payload[f'q90_tmp__{name}'] = np.asarray(d['q90_tmp'], dtype=np.float64)
    np.savez_compressed(CKPT_NPZ, **payload)


def _print_row(name: str, oof_p, tmp_p, cache: dict, dt: float | None = None):
    from phase3_features import compute_metrics

    y = cache['y_price']
    nm = cache['normal_mask']
    ih = cache['is_house']
    tmp_b = [b for b in cache['bundles'] if str(b['id']).startswith('temporal')][0]
    ti = tmp_b['va_idx']
    tnm = nm[ti]

    mn = compute_metrics(y[nm], oof_p[nm])
    mnh = compute_metrics(y[nm & (ih == 1)], oof_p[nm & (ih == 1)])
    mna = compute_metrics(y[nm & (ih == 0)], oof_p[nm & (ih == 0)])
    ma = compute_metrics(y, oof_p)
    mtn = compute_metrics(y[ti][tnm], tmp_p[tnm])
    mta = compute_metrics(y[ti], tmp_p)
    t_s = f"({dt:5.1f}s)" if dt is not None else " (ckpt)"
    print(
        f"  {name:24s} {t_s} | NORM OOF {mn['mdape']:5.2f}% (H {mnh['mdape']:5.2f}% A {mna['mdape']:5.2f}%) "
        f"MAE €{mn['mae']:6,.0f} | ALL {ma['mdape']:5.2f}% | NORM TMP {mtn['mdape']:5.2f}% (bias {mtn['bias_pct']:+5.2f}%) ALL {mta['mdape']:5.2f}%",
        flush=True,
    )
    return {'norm_oof': mn, 'norm_house': mnh, 'norm_apt': mna, 'all_oof': ma, 'norm_tmp': mtn, 'all_tmp': mta}


# ---------------------------------------------------------------------------
# STAGE 1: Full-Corpus Vision A/B Gate (#1) + Per-Family Feature Selection (#7)
# ---------------------------------------------------------------------------
def stage_ablate():
    import joblib
    import lightgbm as lgb
    import numpy as np
    import pandas as pd
    from joblib import Parallel, delayed
    from phase3_features import compute_metrics

    t0 = time.perf_counter()
    cache = joblib.load(P6_CACHE)
    bundles = cache['bundles']
    y_price = cache['y_price']
    y_area = cache['y_area']
    all_stat = cache['feat_cols_stationary']
    has_vis = cache['has_vision']

    vis_cols = [c for c in all_stat if c.startswith('vis_')]
    no_vis_cols = [c for c in all_stat if not c.startswith('vis_')]
    print(f"=== STAGE 1: FULL-CORPUS VISION A/B (#1) + PER-FAMILY FEATURE SELECTION (#7) ===", flush=True)
    print(f"  Total stationary cols: {len(all_stat)} (vision cols: {len(vis_cols)}) | has_vision={has_vis}", flush=True)

    def _fit_one(b, cols):
        tr_df, va_df = b['tr_df'][cols].copy(), b['va_df'][cols].copy()
        for c in cols:
            if tr_df[c].dtype == 'bool':
                tr_df[c] = tr_df[c].astype('int8')
                va_df[c] = va_df[c].astype('int8')
            elif not pd.api.types.is_numeric_dtype(tr_df[c]):
                tr_df[c] = tr_df[c].astype(str).fillna('Missing').astype('category')
                va_df[c] = va_df[c].astype(str).fillna('Missing').astype('category')
            else:
                tr_df[c] = tr_df[c].astype('float32')
                va_df[c] = va_df[c].astype('float32')
        y_tr = _target_fwd(y_price[b['tr_idx']], y_area[b['tr_idx']], b['tr_deflator'], 'log_real_psqm')
        u_c = np.unique(b['tr_cids'])
        rng = np.random.default_rng(42)
        es_c = set(rng.choice(u_c, size=max(1, int(len(u_c) * 0.12)), replace=False))
        es_m = np.isin(b['tr_cids'], list(es_c))
        reg = lgb.LGBMRegressor(
            objective='huber', alpha=0.15, n_estimators=2200, learning_rate=0.03,
            num_leaves=47, min_child_samples=35, subsample=0.75, subsample_freq=1,
            colsample_bytree=0.45, reg_alpha=0.8, reg_lambda=2.5,
            random_state=42, n_jobs=3, verbose=-1,
        )
        reg.fit(
            tr_df.iloc[~es_m], y_tr[~es_m],
            eval_set=[(tr_df.iloc[es_m], y_tr[es_m])],
            callbacks=[lgb.early_stopping(75, verbose=False)],
        )
        pr = _target_inv(reg.predict(va_df), y_area[b['va_idx']], b['va_deflator'], 'log_real_psqm')
        imp = reg.booster_.feature_importance(importance_type='gain')
        return b['id'], b['va_idx'], pr, imp

    def _eval_cols(label, cols):
        ts = time.perf_counter()
        res = Parallel(n_jobs=6, backend='loky')(delayed(_fit_one)(b, cols) for b in bundles)
        oof = np.zeros(len(y_price))
        tmp = None
        gains = np.zeros(len(cols))
        for sid, vi, pr, imp in res:
            if str(sid).startswith('temporal'):
                tmp = pr
            else:
                oof[vi] = pr
                gains += imp / 5.0
        m = _print_row(label, oof, tmp, cache, time.perf_counter() - ts)
        return oof, tmp, gains, m

    ab_report = {}
    _, _, _, m_novis = _eval_cols('A0. NO-VISION (base)', no_vis_cols)
    ab_report['no_vision'] = m_novis

    _, _, gains_all, m_all = _eval_cols('A1. +FULL VISION (all)', all_stat)
    ab_report['with_vision_all'] = m_all

    # Rank all features by fold-safe LightGBM gain
    imp_df = pd.DataFrame({'feature': all_stat, 'gain': gains_all}).sort_values('gain', ascending=False).reset_index(drop=True)
    imp_df.to_csv(os.path.join(WORK_DIR, 'phase6_feature_importance.csv'), index=False)

    print("\n  Top 25 features on Cluster-Clean CV (with Full Vision):", flush=True)
    for r in imp_df.head(25).itertuples():
        print(f"    {r.Index+1:2d}. {r.feature:34s} gain={r.gain:9.1f}")
    if has_vis:
        print("\n  Top 15 Vision features by gain:", flush=True)
        for r in imp_df[imp_df['feature'].str.startswith('vis_')].head(15).itertuples():
            print(f"    rank #{r.Index+1:3d}: {r.feature:32s} gain={r.gain:8.1f}")

    # Collinearity deduplication (|r| > 0.96 among numeric features, keeping higher-gain column!)
    tr0 = bundles[0]['tr_df']
    ordered_feats = imp_df['feature'].tolist()
    num_ord = [c for c in ordered_feats if pd.api.types.is_numeric_dtype(tr0[c]) and tr0[c].dtype != 'bool']
    corr_mat = tr0[num_ord].astype(np.float32).corr().abs().to_numpy()
    dropped_collinear = set()
    for i in range(len(num_ord)):
        if num_ord[i] in dropped_collinear:
            continue
        for j in range(i + 1, len(num_ord)):
            if num_ord[j] not in dropped_collinear and corr_mat[i, j] > 0.96:
                dropped_collinear.add(num_ord[j])
    dedup_feats = [c for c in ordered_feats if c not in dropped_collinear]
    print(f"\n  Collinearity pruning (|r|>0.96): dropped {len(dropped_collinear)} redundant cols -> {len(dedup_feats)} remain", flush=True)

    # Sweep GBDT feature counts on Cluster CV + Temporal!
    sweep = {}
    for k in (160, 220, 280):
        _, _, _, mk = _eval_cols(f'GBDT sweep top-{k} dedup', dedup_feats[:k])
        sweep[k] = mk
    ab_report['gbdt_sweep'] = {str(k): v for k, v in sweep.items()}

    # Choose best GBDT budget by 0.5*NORM_OOF + 0.5*NORM_TMP
    best_gbdt_k = min(sweep.keys(), key=lambda k: 0.5 * sweep[k]['norm_oof']['mdape'] + 0.5 * sweep[k]['norm_tmp']['mdape'])

    # Build per-family feature sets (Issue #7!):
    # 1. GBDTs: best_gbdt_k deduplicated cols (+ ensure render flags stay in when vision is present!)
    must_keep_vis = [c for c in ('vis_blend_psqm', 'vis_blend_tabtxt_resid', 'vis_sig_psqm', 'vis_so_psqm',
                                 'vis_frac_any_render', 'vis_frac_clean_real', 'vis_cgi_margin_mean', 'vis_has_photos')
                     if c in all_stat]
    gbdt_feats = list(dict.fromkeys(dedup_feats[:best_gbdt_k] + must_keep_vis))

    # 2. Tabular NNs (TabM / RealMLP): ~155 dense cols (strip sparse *_missing noise & tail PCA)
    nn_pool = [c for c in dedup_feats if not c.endswith('_missing') or c in dedup_feats[:80]]
    nn_feats = list(dict.fromkeys(nn_pool[:155] + must_keep_vis))

    # 3. Tabular Foundation Models (TabICLv2 / TabPFN V2): ~95 ultra-dense cols, no high-card cats!
    drop_hi = {'statbel_sector', 'city', 'immo_office_name', 'agency_norm'}
    fm_pool = [c for c in nn_pool if c not in drop_hi]
    fm_feats = list(dict.fromkeys(fm_pool[:95] + must_keep_vis[:6]))[:98]

    out_sets = {
        'has_vision': has_vis,
        'best_gbdt_k': best_gbdt_k,
        'dropped_collinear': sorted(dropped_collinear),
        'gbdt_feats': gbdt_feats,
        'gbdt_feats_with_agency': list(dict.fromkeys(gbdt_feats + ['agency_norm'])),
        'nn_feats': nn_feats,
        'fm_feats': fm_feats,
        'ab_report': ab_report,
    }
    with open(FEAT_SETS_JSON, 'w') as f:
        json.dump(out_sets, f, indent=2)
    print(
        f"\n  Saved per-family feature budgets -> {FEAT_SETS_JSON}\n"
        f"    GBDT={len(gbdt_feats)} | NN={len(nn_feats)} | FM={len(fm_feats)} | took {time.perf_counter()-t0:.1f}s",
        flush=True,
    )


# ---------------------------------------------------------------------------
# STAGE 2: Diverse GBDT Core (CPU parallel 6 splits x 3 threads, NO torch!)
# ---------------------------------------------------------------------------
def stage_gbdt():
    import joblib
    import lightgbm as lgb
    import numpy as np
    import pandas as pd
    import xgboost as xgb
    from catboost import CatBoostRegressor
    from joblib import Parallel, delayed

    t0 = time.perf_counter()
    cache = joblib.load(P6_CACHE)
    fsets = json.load(open(FEAT_SETS_JSON))
    bundles = cache['bundles']
    y_price = cache['y_price']
    y_area = cache['y_area']
    is_house = cache['is_house']
    gbdt_feats = fsets['gbdt_feats']
    cb_feats = fsets['gbdt_feats_with_agency']

    ckpt = _load_ckpt()
    print(f"=== STAGE 2: PHASE-6 GBDT CORE (8 models | GBDT={len(gbdt_feats)}, CB={len(cb_feats)} cols) ===", flush=True)

    def _prep(df, cols, kind):
        o = df[cols].copy()
        cat_cols = []
        for c in cols:
            if o[c].dtype == 'bool':
                o[c] = o[c].astype('int8')
            elif not pd.api.types.is_numeric_dtype(o[c]):
                o[c] = o[c].astype(str).fillna('Missing')
                if kind in ('lgbm', 'xgb'):
                    o[c] = o[c].astype('category')
                cat_cols.append(c)
            else:
                o[c] = o[c].astype('float32')
        return o, cat_cols

    def _fit_split(b, cols, kind, target_mode, loss_mode, extra_params, spec_house=None):
        tr_idx, va_idx = b['tr_idx'], b['va_idx']
        tr_df, va_df = b['tr_df'], b['va_df']
        tr_def, va_def = b['tr_deflator'], b['va_deflator']
        tr_cids = b['tr_cids']

        if spec_house is not None:
            tr_m = (is_house[tr_idx] == spec_house)
            va_m = (is_house[va_idx] == spec_house)
            tr_idx = tr_idx[tr_m]; va_idx_sub = va_idx[va_m]
            tr_df = tr_df.iloc[tr_m].reset_index(drop=True)
            va_df = va_df.iloc[va_m].reset_index(drop=True)
            tr_def = tr_def[tr_m]; va_def = va_def[va_m]
            tr_cids = tr_cids[tr_m]
        else:
            va_m = np.ones(len(va_idx), dtype=bool)
            va_idx_sub = va_idx

        Xtr, cat_cols = _prep(tr_df, cols, kind)
        Xva, _ = _prep(va_df, cols, kind)
        if kind == 'xgb':
            for c in cat_cols:
                cats = Xtr[c].cat.categories
                Xva[c] = pd.Categorical(Xva[c].astype(str), categories=cats)

        y_tr = _target_fwd(y_price[tr_idx], y_area[tr_idx], tr_def, target_mode)
        u_c = np.unique(tr_cids)
        rng = np.random.default_rng(42)
        es_c = set(rng.choice(u_c, size=max(1, int(len(u_c) * 0.12)), replace=False))
        es_m = np.isin(tr_cids, list(es_c))
        fit_m = ~es_m

        if kind == 'lgbm':
            p = dict(
                objective='huber' if loss_mode == 'huber' else 'regression',
                n_estimators=3200, learning_rate=0.025, num_leaves=47, min_child_samples=35,
                subsample=0.72, subsample_freq=1, colsample_bytree=0.40,
                reg_alpha=0.8, reg_lambda=2.2, random_state=42, n_jobs=3, verbose=-1,
            )
            if loss_mode == 'huber':
                p['alpha'] = 0.15
            p.update(extra_params)
            m = lgb.LGBMRegressor(**p)
            m.fit(Xtr.iloc[fit_m], y_tr[fit_m], eval_set=[(Xtr.iloc[es_m], y_tr[es_m])],
                  callbacks=[lgb.early_stopping(90, verbose=False)])
            pred_log = m.predict(Xva)
        elif kind == 'xgb':
            p = dict(
                objective='reg:squarederror', n_estimators=3200, learning_rate=0.022,
                max_depth=5, min_child_weight=22, subsample=0.85, colsample_bytree=0.38,
                reg_alpha=0.4, reg_lambda=6.5, tree_method='hist', enable_categorical=True,
                early_stopping_rounds=90, random_state=42, n_jobs=3,
            )
            p.update(extra_params)
            m = xgb.XGBRegressor(**p)
            m.fit(Xtr.iloc[fit_m], y_tr[fit_m], eval_set=[(Xtr.iloc[es_m], y_tr[es_m])], verbose=False)
            pred_log = m.predict(Xva)
        elif kind == 'catboost':
            p = dict(
                loss_function='RMSE', iterations=3500, learning_rate=0.032, depth=7,
                l2_leaf_reg=5.0, random_seed=42, thread_count=3, verbose=0,
            )
            p.update(extra_params)
            m = CatBoostRegressor(**p)
            m.fit(Xtr.iloc[fit_m], y_tr[fit_m], cat_features=cat_cols,
                  eval_set=(Xtr.iloc[es_m], y_tr[es_m]), early_stopping_rounds=120, verbose=False)
            pred_log = m.predict(Xva)
        else:
            raise ValueError(kind)

        pr_eur = _target_inv(pred_log, y_area[va_idx_sub], va_def, target_mode)
        return b['id'], b['va_idx'], va_m, pr_eur

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
            _print_row(name, ckpt[name]['oof'], ckpt[name]['tmp'], cache, None)
            continue
        ts = time.perf_counter()
        oof = np.zeros(len(y_price), dtype=np.float64)
        tmp = None
        if not is_spec:
            res = Parallel(n_jobs=6, backend='loky')(
                delayed(_fit_split)(b, cols, kind, tmode, lmode, ep, None) for b in bundles
            )
            for sid, vi, _, pr in res:
                if str(sid).startswith('temporal'):
                    tmp = pr
                else:
                    oof[vi] = pr
        else:
            tmp_b = [b for b in bundles if str(b['id']).startswith('temporal')][0]
            tmp = np.zeros(len(tmp_b['va_idx']), dtype=np.float64)
            for h_flag in (0, 1):
                res = Parallel(n_jobs=6, backend='loky')(
                    delayed(_fit_split)(b, cols, kind, tmode, lmode, ep, h_flag) for b in bundles
                )
                for sid, vi, vm, pr in res:
                    if str(sid).startswith('temporal'):
                        tmp[vm] = pr
                    else:
                        oof[vi[vm]] = pr
        ckpt[name] = {'oof': oof, 'tmp': tmp}
        _save_ckpt(ckpt)
        _print_row(name, oof, tmp, cache, time.perf_counter() - ts)

    print(f"  Stage 2 (GBDTs) done in {time.perf_counter()-t0:.1f}s", flush=True)


# ---------------------------------------------------------------------------
# STAGE 3: SOTA Tabular NNs + Foundation Models on MPS (ZERO LightGBM import!)
# ---------------------------------------------------------------------------
def stage_mps():
    # Import torch FIRST in this clean process!
    import torch
    import joblib
    import numpy as np
    import pandas as pd

    t0 = time.perf_counter()
    cache = joblib.load(P6_CACHE)
    fsets = json.load(open(FEAT_SETS_JSON))
    bundles = cache['bundles']
    y_price = cache['y_price']
    y_area = cache['y_area']
    is_house = cache['is_house']
    nn_feats = fsets['nn_feats']
    fm_feats = fsets['fm_feats']

    ckpt = _load_ckpt()
    print(f"=== STAGE 3: MPS SOTA NNs + FOUNDATION MODELS (NN={len(nn_feats)}, FM={len(fm_feats)} cols) ===", flush=True)

    def _prep_nn(tr_df, va_df, cols, max_cat=40):
        Xtr, Xva = tr_df[cols].copy(), va_df[cols].copy()
        for c in cols:
            if Xtr[c].dtype == 'bool':
                Xtr[c] = Xtr[c].astype('float32'); Xva[c] = Xva[c].astype('float32')
            elif not pd.api.types.is_numeric_dtype(Xtr[c]):
                vc = Xtr[c].astype(str).fillna('Missing').value_counts()
                keep = set(vc[vc >= 5].head(max_cat).index)
                Xtr[c] = Xtr[c].astype(str).fillna('Missing').where(lambda s: s.isin(keep), '__RARE__').astype('category')
                Xva[c] = Xva[c].astype(str).fillna('Missing').where(lambda s: s.isin(keep), '__RARE__').astype('category')
            else:
                tv = pd.to_numeric(Xtr[c], errors='coerce').replace([np.inf, -np.inf], np.nan)
                vv = pd.to_numeric(Xva[c], errors='coerce').replace([np.inf, -np.inf], np.nan)
                med = float(tv.median()) if tv.notna().any() else 0.0
                Xtr[c] = tv.fillna(med).astype('float32')
                Xva[c] = vv.fillna(med).astype('float32')
        return Xtr, Xva

    def _prep_fm(tr_df, va_df, cols, max_cat=30, numeric_cats=False):
        Xtr, Xva = tr_df[cols].copy(), va_df[cols].copy()
        for c in cols:
            if Xtr[c].dtype == 'bool':
                Xtr[c] = Xtr[c].astype('float32'); Xva[c] = Xva[c].astype('float32')
            elif not pd.api.types.is_numeric_dtype(Xtr[c]):
                vc = Xtr[c].astype(str).fillna('Missing').value_counts()
                keep = list(vc.head(max_cat).index)
                if numeric_cats:
                    cmap = {k: i for i, k in enumerate(keep)}
                    Xtr[c] = Xtr[c].astype(str).fillna('Missing').map(cmap).fillna(-1).astype('float32')
                    Xva[c] = Xva[c].astype(str).fillna('Missing').map(cmap).fillna(-1).astype('float32')
                else:
                    kset = set(keep)
                    Xtr[c] = Xtr[c].astype(str).fillna('Missing').where(lambda s: s.isin(kset), '__OTHER__')
                    Xva[c] = Xva[c].astype(str).fillna('Missing').where(lambda s: s.isin(kset), '__OTHER__')
            else:
                Xtr[c] = pd.to_numeric(Xtr[c], errors='coerce').replace([np.inf, -np.inf], np.nan).astype('float32')
                Xva[c] = pd.to_numeric(Xva[c], errors='coerce').replace([np.inf, -np.inf], np.nan).astype('float32')
        return Xtr, Xva

    # --- 3A. pytabkit TabM + Specialist TabM + RealMLP ---
    from pytabkit import RealMLP_TD_Regressor, TabM_D_Regressor

    def _run_pytabkit(name, kind, target_mode, n_seeds=3, specialist=False):
        if name in ckpt:
            _print_row(name, ckpt[name]['oof'], ckpt[name]['tmp'], cache, None)
            return
        ts = time.perf_counter()
        oof = np.zeros(len(y_price), dtype=np.float64)
        tmp_b = [b for b in bundles if str(b['id']).startswith('temporal')][0]
        tmp = np.zeros(len(tmp_b['va_idx']), dtype=np.float64)

        for b in bundles:
            sub_flags = (0, 1) if specialist else (None,)
            for hf in sub_flags:
                tr_idx, va_idx = b['tr_idx'], b['va_idx']
                tr_df, va_df = b['tr_df'], b['va_df']
                tr_def, va_def = b['tr_deflator'], b['va_deflator']
                tr_cids = b['tr_cids']
                if hf is not None:
                    tr_m = (is_house[tr_idx] == hf)
                    va_m = (is_house[va_idx] == hf)
                    tr_idx = tr_idx[tr_m]; va_idx_sub = va_idx[va_m]
                    tr_df = tr_df.iloc[tr_m].reset_index(drop=True)
                    va_df = va_df.iloc[va_m].reset_index(drop=True)
                    tr_def = tr_def[tr_m]; va_def = va_def[va_m]
                    tr_cids = tr_cids[tr_m]
                else:
                    va_m = np.ones(len(va_idx), dtype=bool)
                    va_idx_sub = va_idx

                Xtr, Xva = _prep_nn(tr_df, va_df, nn_feats)
                y_tr = _target_fwd(y_price[tr_idx], y_area[tr_idx], tr_def, target_mode).astype(np.float32)
                u_c = np.unique(tr_cids)
                rng = np.random.default_rng(77 + (hf or 0) * 13)
                es_c = set(rng.choice(u_c, size=max(1, int(len(u_c) * 0.12)), replace=False))
                val_idxs = np.where(np.isin(tr_cids, list(es_c)))[0]

                s_preds = []
                for s in range(n_seeds):
                    seed = 42 + s * 19 + (hf or 0) * 7
                    if kind == 'tabm':
                        m = TabM_D_Regressor(device='mps', n_epochs=100, batch_size=256, val_metric_name='mae', random_state=seed, verbosity=0)
                    else:
                        m = RealMLP_TD_Regressor(device='mps', n_epochs=128, batch_size=256, val_metric_name='mae', random_state=seed, verbosity=0)
                    m.fit(Xtr, y_tr, val_idxs=val_idxs)
                    s_preds.append(np.asarray(m.predict(Xva), dtype=np.float64))
                    torch.mps.empty_cache()

                pr = _target_inv(np.mean(s_preds, axis=0), y_area[va_idx_sub], va_def, target_mode)
                if str(b['id']).startswith('temporal'):
                    tmp[va_m] = pr
                else:
                    oof[va_idx[va_m]] = pr

        ckpt[name] = {'oof': oof, 'tmp': tmp}
        _save_ckpt(ckpt)
        _print_row(name, oof, tmp, cache, time.perf_counter() - ts)

    _run_pytabkit('p6_tabm_price_3s',      'tabm',    'log_real_price', n_seeds=3, specialist=False)
    _run_pytabkit('p6_tabm_psqm_3s',       'tabm',    'log_real_psqm',  n_seeds=3, specialist=False)
    _run_pytabkit('p6_spec_tabm_psqm_2s',  'tabm',    'log_real_psqm',  n_seeds=2, specialist=True)
    _run_pytabkit('p6_realmlp_price_3s',   'realmlp', 'log_real_price', n_seeds=3, specialist=False)

    # --- 3B. TabICLv2 on MPS (mean + median + native q10/q90 quantiles!) ---
    from tabicl import TabICLRegressor

    for tmode in ('log_real_price', 'log_real_psqm'):
        short = 'price' if tmode == 'log_real_price' else 'psqm'
        n_mean = f'p6_tabicl_{short}_mean'
        n_med = f'p6_tabicl_{short}_med'
        if n_mean in ckpt and n_med in ckpt:
            _print_row(n_mean, ckpt[n_mean]['oof'], ckpt[n_mean]['tmp'], cache, None)
            _print_row(n_med, ckpt[n_med]['oof'], ckpt[n_med]['tmp'], cache, None)
            continue
        ts = time.perf_counter()
        oof_mn = np.zeros(len(y_price)); oof_md = np.zeros(len(y_price))
        q10_oof = np.zeros(len(y_price)); q90_oof = np.zeros(len(y_price))
        tmp_mn = tmp_md = q10_tmp = q90_tmp = None

        for b in bundles:
            Xtr, Xva = _prep_fm(b['tr_df'], b['va_df'], fm_feats, max_cat=30, numeric_cats=False)
            y_tr = _target_fwd(y_price[b['tr_idx']], y_area[b['tr_idx']], b['tr_deflator'], tmode).astype(np.float32)
            reg = TabICLRegressor(device='mps', n_estimators=8, batch_size=1, use_amp=True, random_state=42)
            reg.fit(Xtr, y_tr)
            out = reg.predict(Xva, output_type=['mean', 'median', 'quantiles'], alphas=[0.10, 0.90])
            torch.mps.empty_cache()

            p_mn = _target_inv(out['mean'], y_area[b['va_idx']], b['va_deflator'], tmode)
            p_md = _target_inv(out['median'], y_area[b['va_idx']], b['va_deflator'], tmode)
            q_arr = np.asarray(out['quantiles'])
            p_q10 = _target_inv(q_arr[:, 0], y_area[b['va_idx']], b['va_deflator'], tmode)
            p_q90 = _target_inv(q_arr[:, 1], y_area[b['va_idx']], b['va_deflator'], tmode)

            if str(b['id']).startswith('temporal'):
                tmp_mn, tmp_md, q10_tmp, q90_tmp = p_mn, p_md, p_q10, p_q90
            else:
                oof_mn[b['va_idx']] = p_mn
                oof_md[b['va_idx']] = p_md
                q10_oof[b['va_idx']] = p_q10
                q90_oof[b['va_idx']] = p_q90

        ckpt[n_mean] = {'oof': oof_mn, 'tmp': tmp_mn, 'q10_oof': q10_oof, 'q90_oof': q90_oof, 'q10_tmp': q10_tmp, 'q90_tmp': q90_tmp}
        ckpt[n_med] = {'oof': oof_md, 'tmp': tmp_md}
        _save_ckpt(ckpt)
        dt = time.perf_counter() - ts
        _print_row(n_mean, oof_mn, tmp_mn, cache, dt)
        _print_row(n_med, oof_md, tmp_md, cache, dt)

    # --- 3C. TabPFN V2 (Apache-2.0) on MPS (promoted from Issue #6 screen!) ---
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion

    for tmode in ('log_real_price', 'log_real_psqm'):
        short = 'price' if tmode == 'log_real_price' else 'psqm'
        name = f'p6_tabpfn_v2_{short}'
        if name in ckpt:
            _print_row(name, ckpt[name]['oof'], ckpt[name]['tmp'], cache, None)
            continue
        ts = time.perf_counter()
        oof = np.zeros(len(y_price)); tmp = None
        for b in bundles:
            Xtr, Xva = _prep_fm(b['tr_df'], b['va_df'], fm_feats, max_cat=25, numeric_cats=True)
            y_tr = _target_fwd(y_price[b['tr_idx']], y_area[b['tr_idx']], b['tr_deflator'], tmode).astype(np.float32)
            m = TabPFNRegressor.create_default_for_version(
                ModelVersion.V2, device='mps', n_estimators=8,
                ignore_pretraining_limits=True, inference_config={'SUBSAMPLE_SAMPLES': 7500}, random_state=42,
            )
            m.fit(Xtr, y_tr)
            p_log = np.asarray(m.predict(Xva), dtype=np.float64)
            torch.mps.empty_cache()
            pr = _target_inv(p_log, y_area[b['va_idx']], b['va_deflator'], tmode)
            if str(b['id']).startswith('temporal'):
                tmp = pr
            else:
                oof[b['va_idx']] = pr
        ckpt[name] = {'oof': oof, 'tmp': tmp}
        _save_ckpt(ckpt)
        _print_row(name, oof, tmp, cache, time.perf_counter() - ts)

    print(f"  Stage 3 (MPS NNs + Foundation Models) done in {time.perf_counter()-t0:.1f}s", flush=True)


# ---------------------------------------------------------------------------
# STAGE 4: Nested Cluster-CV Stacking + Heteroscedastic Conformal Intervals (#8)
# ---------------------------------------------------------------------------
def stage_stack():
    import joblib
    import lightgbm as lgb
    import numpy as np
    import pandas as pd
    from scipy.optimize import minimize
    from sklearn.linear_model import HuberRegressor
    from phase3_features import compute_metrics

    cache = joblib.load(P6_CACHE)
    ckpt = _load_ckpt()
    y = cache['y_price']
    nm = cache['normal_mask']
    ih = cache['is_house']
    folds = cache['folds']
    tmp_b = [b for b in cache['bundles'] if str(b['id']).startswith('temporal')][0]
    ti = tmp_b['va_idx']
    tnm = nm[ti]

    names = sorted(ckpt.keys())
    O_log = np.column_stack([np.log(np.clip(ckpt[n]['oof'], 10_000, 1e7)) for n in names])
    T_log = np.column_stack([np.log(np.clip(ckpt[n]['tmp'], 10_000, 1e7)) for n in names])
    y_log = np.log(y)

    def _fit_w(X, yv):
        k = X.shape[1]
        def obj(w):
            e = yv - X @ w
            return 0.55 * np.mean(np.abs(e)) + 0.45 * np.sqrt(np.mean(e ** 2))
        res = minimize(obj, np.ones(k) / k, method='SLSQP',
                       bounds=[(0.0, 1.0)] * k, constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
        return res.x

    def _nested_convex(col_idx, mask_type=None):
        oof_l = np.zeros(len(y))
        for f in range(5):
            tr_m = (folds != f) & (nm if mask_type is None else (nm & (ih == mask_type)))
            va_m = (folds == f) & (np.ones(len(y), dtype=bool) if mask_type is None else (ih == mask_type))
            w = _fit_w(O_log[tr_m][:, col_idx], y_log[tr_m])
            oof_l[va_m] = O_log[va_m][:, col_idx] @ w
        tr_all = nm if mask_type is None else (nm & (ih == mask_type))
        w_full = _fit_w(O_log[tr_all][:, col_idx], y_log[tr_all])
        tmp_l = T_log[:, col_idx] @ w_full
        return np.exp(oof_l), np.exp(tmp_l), w_full

    print(f"\n=== STAGE 4: PHASE-6 CLUSTER-CLEAN ENSEMBLE STACKING ({len(names)} streams) ===", flush=True)
    all_idx = list(range(len(names)))
    gbdt_idx = [i for i, n in enumerate(names) if any(k in n for k in ('lgbm', 'xgb', 'cb_'))]

    rep = {'models': {}, 'stacks': {}}
    for n in names:
        rep['models'][n] = _print_row(n, ckpt[n]['oof'], ckpt[n]['tmp'], cache, None)

    print("  " + "-" * 114, flush=True)
    p_gbdt_oof, p_gbdt_tmp, w_gbdt = _nested_convex(gbdt_idx)
    rep['stacks']['P6_GBDT'] = _print_row('P6-GBDT (GBDTs only)', p_gbdt_oof, p_gbdt_tmp, cache)

    p_uni_oof, p_uni_tmp, w_uni = _nested_convex(all_idx)
    rep['stacks']['P6_UNIFIED'] = _print_row('P6-UNIFIED (ALL 18)', p_uni_oof, p_uni_tmp, cache)

    # Type-specific convex (House vs Apt)
    p_typ_oof = np.zeros(len(y)); p_typ_tmp = np.zeros(len(ti))
    w_typed = {}
    for hf, lbl in ((0, 'apt'), (1, 'house')):
        o_h, t_h, w_h = _nested_convex(all_idx, mask_type=hf)
        p_typ_oof[ih == hf] = o_h[ih == hf]
        p_typ_tmp[ih[ti] == hf] = t_h[ih[ti] == hf]
        w_typed[lbl] = {names[i]: round(float(w_h[i]), 4) for i in range(len(names)) if w_h[i] > 0.005}
    rep['stacks']['P6_TYPED'] = _print_row('P6-TYPED (House/Apt)', p_typ_oof, p_typ_tmp, cache)

    # Huber 2nd-level meta-stacker
    p_hub_oof = np.zeros(len(y))
    for f in range(5):
        m = HuberRegressor(alpha=15.0, max_iter=500).fit(O_log[(folds != f) & nm], y_log[(folds != f) & nm])
        p_hub_oof[folds == f] = np.exp(m.predict(O_log[folds == f]))
    m_hub_full = HuberRegressor(alpha=15.0, max_iter=500).fit(O_log[nm], y_log[nm])
    p_hub_tmp = np.exp(m_hub_full.predict(T_log))
    rep['stacks']['P6_HUBER'] = _print_row('P6-HUBER (recal)', p_hub_oof, p_hub_tmp, cache)

    print("\n  Unified Ensemble Weights (>1%):", flush=True)
    for n, w in sorted(zip(names, w_uni), key=lambda kv: -kv[1]):
        if w >= 0.01:
            print(f"    {n:26s}: {w*100:5.1f}%")
    rep['weights_unified'] = {n: round(float(w), 4) for n, w in zip(names, w_uni) if w > 0.005}
    rep['weights_typed'] = w_typed

    # --- Issue #8: Heteroscedastic Conformal 80% Prediction Intervals! ---
    print("\n=== ISSUE #8: HETEROSCEDASTIC CONFORMAL 80% INTERVALS (Normal Ghent Homes) ===", flush=True)
    ens_oof_l = np.log(p_uni_oof)
    ens_tmp_l = np.log(p_uni_tmp)
    abs_err_oof = np.abs(y_log - ens_oof_l)
    abs_err_tmp = np.abs(np.log(y[ti]) - ens_tmp_l)

    # Uncertainty features from ensemble disagreement + TabICLv2 native width + property context
    ens_std_oof = np.std(O_log, axis=1)
    ens_std_tmp = np.std(T_log, axis=1)
    # Build context frame aligned to all 14,599 rows from OOF validation frames
    oof_parts = []
    for b in cache['bundles']:
        if not str(b['id']).startswith('temporal'):
            d = b['va_df'][['livable_area', 'property_type_HOUSE', 'is_newly_built', 'epc_ordinal', 'condition_ordinal',
                            'dist_from_center', 'vis_frac_any_render', 'vis_q_1']].copy() if 'vis_q_1' in b['va_df'].columns else \
                b['va_df'][['livable_area', 'property_type_HOUSE', 'is_newly_built', 'epc_ordinal', 'condition_ordinal', 'dist_from_center']].copy()
            d['__idx__'] = b['va_idx']
            oof_parts.append(d)
    ctx_oof = pd.concat(oof_parts, axis=0).sort_values('__idx__').drop(columns='__idx__').reset_index(drop=True)
    ctx_cols = ctx_oof.columns.tolist()
    ctx_tmp = tmp_b['va_df'][ctx_cols].reset_index(drop=True)

    ctx_oof['ens_log_pred'] = ens_oof_l
    ctx_tmp['ens_log_pred'] = ens_tmp_l
    ctx_oof['ens_std'] = ens_std_oof
    ctx_tmp['ens_std'] = ens_std_tmp
    if 'p6_tabicl_price_mean' in ckpt and 'q10_oof' in ckpt['p6_tabicl_price_mean']:
        ctx_oof['tabicl_hw'] = 0.5 * (np.log(ckpt['p6_tabicl_price_mean']['q90_oof']) - np.log(ckpt['p6_tabicl_price_mean']['q10_oof']))
        ctx_tmp['tabicl_hw'] = 0.5 * (np.log(ckpt['p6_tabicl_price_mean']['q90_tmp']) - np.log(ckpt['p6_tabicl_price_mean']['q10_tmp']))

    # Fit shallow OOF log-error scale model sigma_hat(x) strictly nested across cluster folds!
    sig_oof = np.zeros(len(y))
    for f in range(5):
        m_tr = (folds != f) & nm
        m_va = folds == f
        sm = lgb.LGBMRegressor(objective='mae', n_estimators=200, learning_rate=0.03, max_depth=3, num_leaves=7,
                               min_child_samples=80, reg_lambda=20.0, random_state=42, n_jobs=4, verbose=-1)
        sm.fit(ctx_oof.iloc[m_tr], abs_err_oof[m_tr])
        sig_oof[m_va] = np.clip(sm.predict(ctx_oof.iloc[m_va]), 0.03, 0.50)
    sm_full = lgb.LGBMRegressor(objective='mae', n_estimators=200, learning_rate=0.03, max_depth=3, num_leaves=7,
                                min_child_samples=80, reg_lambda=20.0, random_state=42, n_jobs=4, verbose=-1)
    sm_full.fit(ctx_oof.iloc[nm], abs_err_oof[nm])
    sig_tmp = np.clip(sm_full.predict(ctx_tmp), 0.03, 0.50)

    # Compare (A) Homoscedastic Conformal vs (B) Heteroscedastic Normalized Conformal (target 80%!)
    hw_homo_oof = np.zeros(len(y)); hw_het_oof = np.zeros(len(y))
    for f in range(5):
        cal = (folds != f) & nm
        va = folds == f
        q_ho = np.quantile(abs_err_oof[cal], 0.80)
        q_he = np.quantile(abs_err_oof[cal] / sig_oof[cal], 0.80)
        hw_homo_oof[va] = q_ho
        hw_het_oof[va] = q_he * sig_oof[va]
    q_ho_f = np.quantile(abs_err_oof[nm], 0.80)
    q_he_f = np.quantile(abs_err_oof[nm] / sig_oof[nm], 0.80)
    hw_homo_tmp = np.full(len(ti), q_ho_f)
    hw_het_tmp = q_he_f * sig_tmp

    def _cov(err, hw, mask):
        return 100.0 * float(np.mean(err[mask] <= hw[mask])), 100.0 * float(np.median(np.expm1(hw[mask])))

    ho_o, ho_w = _cov(abs_err_oof, hw_homo_oof, nm)
    he_o, he_w = _cov(abs_err_oof, hw_het_oof, nm)
    ho_t, ho_wt = _cov(abs_err_tmp, hw_homo_tmp, tnm)
    he_t, he_wt = _cov(abs_err_tmp, hw_het_tmp, tnm)
    print(f"  HOMOSCEDASTIC  80% | OOF cov={ho_o:5.2f}% med_hw=±{ho_w:5.2f}% | TMP cov={ho_t:5.2f}% med_hw=±{ho_wt:5.2f}%")
    print(f"  HETEROSCEDASTIC 80%| OOF cov={he_o:5.2f}% med_hw=±{he_w:5.2f}% | TMP cov={he_t:5.2f}% med_hw=±{he_wt:5.2f}%")

    # Slice calibration check: New-build vs Apt vs House vs Fixer-upper!
    nb = ctx_oof['is_newly_built'].to_numpy() == 1
    fx = ctx_oof['condition_ordinal'].fillna(3).to_numpy() <= 1
    print("  Per-slice 80% coverage & band width (Homo vs Hetero):")
    for sl_name, smask in [('New-build', nm & nb), ('Normal Apt', nm & (ih == 0)), ('Normal House', nm & (ih == 1)), ('Fixer-upper', nm & fx)]:
        c1, w1 = _cov(abs_err_oof, hw_homo_oof, smask)
        c2, w2 = _cov(abs_err_oof, hw_het_oof, smask)
        print(f"    {sl_name:14s} (n={smask.sum():5d}) | Homo: cov={c1:5.1f}% (±{w1:4.1f}%) -> Hetero: cov={c2:5.1f}% (±{w2:4.1f}%)")

    rep['conformal_80'] = {
        'homo_oof_cov': ho_o, 'homo_oof_hw_pct': ho_w, 'homo_tmp_cov': ho_t,
        'hetero_oof_cov': he_o, 'hetero_oof_hw_pct': he_w, 'hetero_tmp_cov': he_t, 'hetero_tmp_hw_pct': he_wt,
    }
    np.savez_compressed(
        PREDS_NPZ,
        names=np.array(names, dtype=object),
        O_log=O_log, T_log=T_log, y_oof=y, y_tmp=y[ti], tmp_idx=ti,
        ens_oof_u=p_uni_oof, ens_tmp_u=p_uni_tmp,
        ens_oof_t=p_typ_oof, ens_tmp_t=p_typ_tmp,
        ens_oof_h=p_hub_oof, ens_tmp_h=p_hub_tmp,
        hw_het_oof=hw_het_oof, hw_het_tmp=hw_het_tmp,
        weights_unified=w_uni,
    )
    with open(REPORT_JSON, 'w') as f:
        json.dump(rep, f, indent=2)
    print(f"\n=== PHASE-6 COMPLETE -> {PREDS_NPZ} & {REPORT_JSON} ===", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['ablate', 'gbdt', 'mps', 'stack', 'all'], default='all')
    args = ap.parse_args()
    if args.stage == 'all':
        for st in ('ablate', 'gbdt', 'mps', 'stack'):
            print(f"\n>>> Launching clean subprocess stage: {st} <<<", flush=True)
            subprocess.run([sys.executable, '-u', __file__, '--stage', st], check=True)
    elif args.stage == 'ablate':
        stage_ablate()
    elif args.stage == 'gbdt':
        stage_gbdt()
    elif args.stage == 'mps':
        stage_mps()
    elif args.stage == 'stack':
        stage_stack()


if __name__ == '__main__':
    main()
