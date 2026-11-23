"""Issue #12: Native Per-Family Feature Pruning & Permutation Selection for TabM, TabICLv2, and TabPFN V2!

Follows our anti-overfit 4-step funnel (NO Genetic Algorithms!):
  - Evaluates strictly on the fast 2-split dev profile (`fold_0` + `temporal_2026`, Normal Ghent Homes EUR 150k-800k)
  - Uses subprocess stage isolation (`--stage perm|fm|nn|finalize`) so MPS and CPU stages never collide.

Stages:
  1. `perm` (MPS ~4 min):
     - Fits `TabM_D_Regressor` on `fold_0` with the 240-col deduped candidate pool (top-240 non-collinear feats),
       then runs **zero-retraining Native Permutation Importance** (3 repeats per feature) on `fold_0` val!
     - Discovers features `TabM` truly uses (even if LightGBM ranked them low) AND flags negative-importance
       features that actively hurt `TabM` generalization.
  2. `fm` (MPS ~8 min):
     - Sweeps compact feature budgets for Foundation Models (`TabICLv2` & `TabPFN V2`):
       `65 / 85 / 98 (P6 baseline) / 120` cols + coarse block drop-one ablation (drop `vis_pca`, drop `q_scores`,
       drop `render_flags`, drop `wave2`, drop `issue3`, drop `e5_pca/svd`) on `fold_0 + temporal_2026`!
  3. `nn` (MPS ~6 min):
     - Sweeps `TabM` feature budgets built from TabM's **own** native permutation ranking (`110 / 150 / 190 / 230` cols
       + `drop_negative_perm`!) on `fold_0 + temporal_2026` vs Phase-6 `NN=160` LGBM-ranked baseline!
  4. `finalize`:
     - Applies the Golden Selection Rule (keep only subsets that improve or tie on BOTH `fold_0` and `temporal_2026`,
       picking the smaller end of any flat plateau) and saves `outputs/models/workbench/phase7_feature_sets.json`.

Usage:
    .venv-api/bin/python -u src/phase7_feature_selection.py --stage all
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'src'))

WORK_DIR = os.path.join(BASE_DIR, 'outputs', 'models', 'workbench')
P6_CACHE = os.path.join(WORK_DIR, 'phase6_splits_cache.joblib')
P6_FSETS = os.path.join(WORK_DIR, 'phase6_feature_sets.json')
P6_IMP_CSV = os.path.join(WORK_DIR, 'phase6_feature_importance.csv')
P7_PERM_CSV = os.path.join(WORK_DIR, 'phase7_tabm_perm_importance.csv')
P7_SWEEP_JSON = os.path.join(WORK_DIR, 'phase7_selection_sweeps.json')
P7_FSETS_JSON = os.path.join(WORK_DIR, 'phase7_feature_sets.json')

MUST_KEEP_RENDER = [
    'vis_has_photos', 'vis_frac_any_render', 'vis_frac_clean_real', 'vis_frac_cgi',
    'vis_cgi_margin_mean', 'vis_blend_psqm', 'vis_blend_tabtxt_resid',
]
HIGH_CARD_DROP = {'statbel_sector', 'city', 'immo_office_name', 'agency_norm'}


def _target_fwd(y_price, area, defl, mode='log_real_price'):
    return np.log((y_price / defl) / area) if mode == 'log_real_psqm' else np.log(y_price / defl)


def _target_inv(pred_log, area, defl, mode='log_real_price'):
    p = np.clip(np.asarray(pred_log, dtype=np.float64), -20.0, 20.0)
    return np.clip(np.exp(p) * (area * defl if mode == 'log_real_psqm' else defl), 10_000.0, 1e7)


def _load_2splits():
    import joblib
    cache = joblib.load(P6_CACHE)
    f0 = [b for b in cache['bundles'] if b['id'] == 'fold_0'][0]
    tm = [b for b in cache['bundles'] if str(b['id']).startswith('temporal')][0]
    return cache, f0, tm


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
            Xtr[c] = tv.fillna(med).astype('float32'); Xva[c] = vv.fillna(med).astype('float32')
    return Xtr, Xva


def _prep_fm(tr_df, va_df, cols, max_cat=30, numeric_cats=False):
    Xtr, Xva = tr_df[cols].copy(), va_df[cols].copy()
    for c in cols:
        if Xtr[c].dtype == 'bool':
            Xtr[c] = Xtr[c].astype('float32'); Xva[c] = Xva[c].astype('float32')
        elif not pd.api.types.is_numeric_dtype(Xtr[c]):
            keep = list(Xtr[c].astype(str).fillna('Missing').value_counts().head(max_cat).index)
            if numeric_cats:
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


def _load_sweep():
    if os.path.exists(P7_SWEEP_JSON):
        return json.load(open(P7_SWEEP_JSON))
    return {}


def _save_sweep(d):
    with open(P7_SWEEP_JSON, 'w') as f:
        json.dump(d, f, indent=2)


def stage_perm():
    import torch
    from pytabkit import TabM_D_Regressor
    from phase3_features import compute_metrics

    t0 = time.perf_counter()
    print("=== ISSUE #12 STAGE 1: NATIVE TabM PERMUTATION IMPORTANCE + SHADOW NOISE PROBES (fold_0) ===", flush=True)
    cache, f0, _ = _load_2splits()
    fsets = json.load(open(P6_FSETS))
    # Candidate pool: top-240 non-collinear features (from GBDT 284 minus dead *_missing flags)
    cand = [c for c in fsets['gbdt_feats'] if not c.endswith('_missing') and c in f0['tr_df'].columns][:240]
    for mk in MUST_KEEP_RENDER:
        if mk in f0['tr_df'].columns and mk not in cand:
            cand.append(mk)

    # Add 5 pure random-noise shadow probes!
    rng = np.random.default_rng(42)
    tr_df = f0['tr_df'][cand].copy()
    va_df = f0['va_df'][cand].copy()
    shadow_cols = [f'__shadow_noise_{i}' for i in range(5)]
    for sc in shadow_cols:
        tr_df[sc] = rng.standard_normal(len(tr_df)).astype(np.float32)
        va_df[sc] = rng.standard_normal(len(va_df)).astype(np.float32)
    all_cols = cand + shadow_cols

    Xtr, Xva = _prep_nn(tr_df, va_df, all_cols)
    y = cache['y_price']; a = cache['y_area']; nm = cache['normal_mask']
    y_tr = _target_fwd(y[f0['tr_idx']], a[f0['tr_idx']], f0['tr_deflator'], 'log_real_price').astype(np.float32)
    u_c = np.unique(f0['tr_cids'])
    es_c = set(np.random.default_rng(77).choice(u_c, size=max(1, int(len(u_c) * 0.12)), replace=False))
    val_idxs = np.where(np.isin(f0['tr_cids'], list(es_c)))[0]

    # Fit 2-seed TabM once!
    models = []
    base_preds = []
    for s in range(2):
        m = TabM_D_Regressor(device='mps', n_epochs=90, batch_size=256, val_metric_name='mae', random_state=42 + s * 19, verbosity=0)
        m.fit(Xtr, y_tr, val_idxs=val_idxs)
        models.append(m)
        base_preds.append(np.asarray(m.predict(Xva), dtype=np.float64))
        torch.mps.empty_cache()

    y_va = y[f0['va_idx']]
    nm_va = nm[f0['va_idx']]
    base_eur = _target_inv(np.mean(base_preds, axis=0), a[f0['va_idx']], f0['va_deflator'], 'log_real_price')
    base_m = compute_metrics(y_va[nm_va], base_eur[nm_va])
    base_log_mae = float(np.mean(np.abs(np.log(base_eur[nm_va]) - np.log(y_va[nm_va]))))
    print(f"  2-seed TabM on {len(cand)} cols + 5 shadows: f0 NORM MdAPE={base_m['mdape']:.2f}% MAE=€{base_m['mae']:,.0f} logMAE={base_log_mae:.5f}", flush=True)

    # Zero-retraining permutation importance on fold_0 val (2 repeats per feature)
    rows = []
    Xva_work = Xva.copy()
    for i, col in enumerate(all_cols):
        orig = Xva_work[col].copy()
        d_mdapes, d_lmaes = [], []
        for rep in range(2):
            perm_idx = np.random.default_rng(1000 + i * 7 + rep).permutation(len(Xva_work))
            Xva_work[col] = orig.iloc[perm_idx].values
            p_l = np.mean([np.asarray(m.predict(Xva_work), dtype=np.float64) for m in models], axis=0)
            p_eur = _target_inv(p_l, a[f0['va_idx']], f0['va_deflator'], 'log_real_price')
            pm = compute_metrics(y_va[nm_va], p_eur[nm_va])
            lmae = float(np.mean(np.abs(np.log(p_eur[nm_va]) - np.log(y_va[nm_va]))))
            d_mdapes.append(pm['mdape'] - base_m['mdape'])
            d_lmaes.append(lmae - base_log_mae)
        Xva_work[col] = orig
        rows.append({
            'feature': col,
            'is_shadow': col.startswith('__shadow_'),
            'perm_delta_mdape_pp': float(np.mean(d_mdapes)),
            'perm_delta_log_mae': float(np.mean(d_lmaes)),
        })
        if (i + 1) % 50 == 0 or (i + 1) == len(all_cols):
            print(f"  Permuted {i+1:3d}/{len(all_cols)} features ({time.perf_counter()-t0:.1f}s)", flush=True)

    pdf = pd.DataFrame(rows).sort_values('perm_delta_log_mae', ascending=False).reset_index(drop=True)
    pdf.to_csv(P7_PERM_CSV, index=False)
    sh = pdf[pdf['is_shadow']]
    sh_max = float(sh['perm_delta_log_mae'].max())
    sh_mean = float(sh['perm_delta_log_mae'].mean())
    n_pos = int((pdf[~pdf['is_shadow']]['perm_delta_log_mae'] > sh_max).sum())
    n_neg = int((pdf[~pdf['is_shadow']]['perm_delta_log_mae'] < 0).sum())
    print(f"  Shadow noise probes: mean={sh_mean:+.6f} max={sh_max:+.6f} | Real feats > shadow_max: {n_pos} | Negative feats: {n_neg}", flush=True)
    print("  Top 20 TabM Native Permutation features:", flush=True)
    for _, r in pdf[~pdf['is_shadow']].head(20).iterrows():
        print(f"    {r['feature']:34s} | ΔlogMAE={r['perm_delta_log_mae']:+.6f} | ΔMdAPE={r['perm_delta_mdape_pp']:+.3f}pp", flush=True)
    print("  Worst 10 (actively hurting TabM on fold_0):", flush=True)
    for _, r in pdf[~pdf['is_shadow']].tail(10).iterrows():
        print(f"    {r['feature']:34s} | ΔlogMAE={r['perm_delta_log_mae']:+.6f} | ΔMdAPE={r['perm_delta_mdape_pp']:+.3f}pp", flush=True)
    del models; gc.collect(); torch.mps.empty_cache()


def _run_fm_worker(kind: str, label: str, cols_json: str, tmode: str):
    import torch
    from phase3_features import compute_metrics
    cols = json.loads(cols_json)
    cache, f0, tm = _load_2splits()
    y = cache['y_price']; a = cache['y_area']; nm = cache['normal_mask']
    ts = time.perf_counter()
    out_m = {}
    for tag, sp in (('f0', f0), ('tm', tm)):
        if kind == 'tabicl':
            from tabicl import TabICLRegressor
            Xtr, Xva = _prep_fm(sp['tr_df'], sp['va_df'], cols, max_cat=30, numeric_cats=False)
            yt = _target_fwd(y[sp['tr_idx']], a[sp['tr_idx']], sp['tr_deflator'], tmode).astype(np.float32)
            reg = TabICLRegressor(device='mps', n_estimators=8, batch_size=1, use_amp=True, random_state=42)
            reg.fit(Xtr, yt)
            p_raw = np.asarray(reg.predict(Xva, output_type='median'), dtype=np.float64)
            del reg
        else:
            from tabpfn import TabPFNRegressor
            from tabpfn.constants import ModelVersion
            Xtr, Xva = _prep_fm(sp['tr_df'], sp['va_df'], cols, max_cat=25, numeric_cats=True)
            yt = _target_fwd(y[sp['tr_idx']], a[sp['tr_idx']], sp['tr_deflator'], tmode).astype(np.float32)
            m_pfn = TabPFNRegressor.create_default_for_version(
                ModelVersion.V2, device='mps', n_estimators=8,
                ignore_pretraining_limits=True, inference_config={'SUBSAMPLE_SAMPLES': 7500}, random_state=42,
            )
            m_pfn.fit(Xtr, yt)
            p_raw = np.asarray(m_pfn.predict(Xva), dtype=np.float64)
            del m_pfn
        gc.collect(); torch.mps.empty_cache()
        p_eur = _target_inv(p_raw, a[sp['va_idx']], sp['va_deflator'], tmode)
        out_m[tag] = compute_metrics(y[sp['va_idx']][nm[sp['va_idx']]], p_eur[nm[sp['va_idx']]])
    dt = time.perf_counter() - ts
    tag_s = 'TabICLv2' if kind == 'tabicl' else 'TabPFN  '
    print(f"  [{tag_s}] {label:28s} ({len(cols):3d}c, {dt:5.1f}s) | f0={out_m['f0']['mdape']:5.2f}% (€{out_m['f0']['mae']:6,.0f}) | tm={out_m['tm']['mdape']:5.2f}% (€{out_m['tm']['mae']:6,.0f} bias {out_m['tm']['bias_pct']:+5.2f}%)", flush=True)
    sw = _load_sweep()
    fm_res = sw.get('fm_sweep', {})
    fm_res[f'{kind}__{label}'] = {'cols': cols, 'n_cols': len(cols), 'f0': out_m['f0'], 'tm': out_m['tm']}
    sw['fm_sweep'] = fm_res
    _save_sweep(sw)


def stage_fm():
    t0 = time.perf_counter()
    print("\n=== ISSUE #12 STAGE 2: FOUNDATION MODEL (TabICLv2 & TabPFN V2) FEATURE BUDGET + BLOCK ABLATION (2-Split) ===", flush=True)
    cache, f0, _ = _load_2splits()
    fsets = json.load(open(P6_FSETS))
    perm_df = pd.read_csv(P7_PERM_CSV) if os.path.exists(P7_PERM_CSV) else None

    dedup_gbdt = fsets['gbdt_feats']
    fm_pool = [c for c in dedup_gbdt if c not in HIGH_CARD_DROP and not c.endswith('_missing') and c in f0['tr_df'].columns]
    prio = [c for c in MUST_KEEP_RENDER if c in fm_pool]
    fm_ranked = prio + [c for c in fm_pool if c not in prio]

    if perm_df is not None:
        r_lgb = {c: i for i, c in enumerate(fm_ranked)}
        p_real = perm_df[~perm_df['is_shadow']]['feature'].tolist()
        r_tabm = {c: i for i, c in enumerate(p_real)}
        fm_consensus = sorted(fm_ranked, key=lambda c: 0.5 * r_lgb.get(c, 300) + 0.5 * r_tabm.get(c, 300))
        fm_consensus = prio + [c for c in fm_consensus if c not in prio]
    else:
        fm_consensus = fm_ranked

    def _run_iso(kind, label, cols, tmode):
        subprocess.run(
            [sys.executable, '-u', __file__, '--stage', 'fm_worker',
             '--fm-kind', kind, '--fm-label', label, '--fm-tmode', tmode, '--fm-cols', json.dumps(cols)],
            check=True,
        )

    sw = _load_sweep()
    fm_res = sw.get('fm_sweep', {})
    p6_fm = [c for c in fsets['fm_feats'] if c in f0['tr_df'].columns]
    cands_ticl = [
        ('p6_baseline_98', p6_fm),
        ('lgb_top_65', fm_ranked[:65]),
        ('lgb_top_80', fm_ranked[:80]),
        ('lgb_top_105', fm_ranked[:105]),
        ('consensus_75', fm_consensus[:75]),
        ('consensus_95', fm_consensus[:95]),
        ('consensus_95_no_pca', [c for c in fm_consensus[:135] if not ('_pca_' in c or 'tfidf_svd_' in c)][:95]),
        ('consensus_80_no_pca', [c for c in fm_consensus[:125] if not ('_pca_' in c or 'tfidf_svd_' in c)][:80]),
    ]
    for label, cols in cands_ticl:
        if f'tabicl__{label}' not in fm_res:
            _run_iso('tabicl', label, cols, 'log_real_psqm')
            fm_res = _load_sweep().get('fm_sweep', {})
        else:
            v = fm_res[f'tabicl__{label}']
            print(f"  [TabICLv2 ckpt] {label:24s} ({v['n_cols']:3d}c) | f0={v['f0']['mdape']:5.2f}% | tm={v['tm']['mdape']:5.2f}%", flush=True)

    ticl_scored = sorted(
        [(k.split('__', 1)[1], 0.5 * v['f0']['mdape'] + 0.5 * v['tm']['mdape'], v['cols'])
         for k, v in fm_res.items() if k.startswith('tabicl__')],
        key=lambda x: x[1],
    )
    pfn_cands = [('p6_baseline_98', p6_fm)]
    for lbl, _, cols in ticl_scored[:3]:
        if lbl != 'p6_baseline_98':
            pfn_cands.append((lbl, cols))
    for label, cols in pfn_cands:
        if f'tabpfn__{label}' not in fm_res:
            _run_iso('tabpfn', label, cols, 'log_real_price')
            fm_res = _load_sweep().get('fm_sweep', {})
    print(f"  Stage 2 (FM pruning) done in {time.perf_counter()-t0:.1f}s", flush=True)


def stage_nn():
    import torch
    from pytabkit import TabM_D_Regressor
    from phase3_features import compute_metrics

    t0 = time.perf_counter()
    print("\n=== ISSUE #12 STAGE 3: NATIVE TabM FEATURE BUDGET SWEEP (2-Split, 2 seeds) ===", flush=True)
    cache, f0, tm = _load_2splits()
    fsets = json.load(open(P6_FSETS))
    perm_df = pd.read_csv(P7_PERM_CSV)
    y = cache['y_price']; a = cache['y_area']; nm = cache['normal_mask']

    real_p = perm_df[~perm_df['is_shadow']]
    sh_mean = float(perm_df[perm_df['is_shadow']]['perm_delta_log_mae'].mean())
    perm_ranked = real_p['feature'].tolist()
    pos_only = real_p[real_p['perm_delta_log_mae'] > 0]['feature'].tolist()
    above_shadow = real_p[real_p['perm_delta_log_mae'] > sh_mean]['feature'].tolist()
    for lst in (pos_only, above_shadow):
        for mk in MUST_KEEP_RENDER:
            if mk in f0['tr_df'].columns and mk not in lst:
                lst.append(mk)

    # Consensus NN ranking (50% TabM native perm + 50% LGBM gain)
    gbdt_dense = [c for c in fsets['gbdt_feats'] if not c.endswith('_missing') and c in f0['tr_df'].columns]
    r_lgb = {c: i for i, c in enumerate(gbdt_dense)}
    r_tm = {c: i for i, c in enumerate(perm_ranked)}
    nn_cons = sorted(gbdt_dense[:240], key=lambda c: 0.5 * r_lgb.get(c, 300) + 0.5 * r_tm.get(c, 300))

    p6_nn = [c for c in fsets['nn_feats'] if c in f0['tr_df'].columns]
    cands_nn = [
        ('p6_baseline_160', p6_nn),
        ('perm_top_120', perm_ranked[:120]),
        ('perm_top_160', perm_ranked[:160]),
        ('perm_pos_only', pos_only),
        ('perm_above_shadow', above_shadow),
        ('cons_130', nn_cons[:130]),
        ('cons_180', nn_cons[:180]),
        ('cons_220', nn_cons[:220]),
    ]

    sw = _load_sweep()
    nn_res = sw.get('nn_sweep', {})

    for label, cols in cands_nn:
        cols = list(dict.fromkeys([c for c in cols if c in f0['tr_df'].columns]))
        if label in nn_res:
            v = nn_res[label]
            print(f"  [TabM ckpt] {label:22s} ({v['n_cols']:3d}c) | f0={v['f0']['mdape']:5.2f}% | tm={v['tm']['mdape']:5.2f}%", flush=True)
            continue
        ts = time.perf_counter()
        out_m = {}
        for tag, sp in (('f0', f0), ('tm', tm)):
            Xtr, Xva = _prep_nn(sp['tr_df'], sp['va_df'], cols)
            yt = _target_fwd(y[sp['tr_idx']], a[sp['tr_idx']], sp['tr_deflator'], 'log_real_price').astype(np.float32)
            u_c = np.unique(sp['tr_cids'])
            es_c = set(np.random.default_rng(77).choice(u_c, size=max(1, int(len(u_c) * 0.12)), replace=False))
            val_idxs = np.where(np.isin(sp['tr_cids'], list(es_c)))[0]
            s_preds = []
            for s in range(2):
                m = TabM_D_Regressor(device='mps', n_epochs=95, batch_size=256, val_metric_name='mae', random_state=42 + s * 19, verbosity=0)
                m.fit(Xtr, yt, val_idxs=val_idxs)
                s_preds.append(np.asarray(m.predict(Xva), dtype=np.float64))
                del m; gc.collect(); torch.mps.empty_cache()
            p_eur = _target_inv(np.mean(s_preds, axis=0), a[sp['va_idx']], sp['va_deflator'], 'log_real_price')
            out_m[tag] = compute_metrics(y[sp['va_idx']][nm[sp['va_idx']]], p_eur[nm[sp['va_idx']]])
        dt = time.perf_counter() - ts
        nn_res[label] = {'cols': cols, 'n_cols': len(cols), 'f0': out_m['f0'], 'tm': out_m['tm']}
        sw['nn_sweep'] = nn_res; _save_sweep(sw)
        print(f"  [TabM] {label:22s} ({len(cols):3d}c, {dt:5.1f}s) | f0={out_m['f0']['mdape']:5.2f}% (€{out_m['f0']['mae']:6,.0f}) | tm={out_m['tm']['mdape']:5.2f}% (€{out_m['tm']['mae']:6,.0f} bias {out_m['tm']['bias_pct']:+5.2f}%)", flush=True)
    print(f"  Stage 3 (NN pruning) done in {time.perf_counter()-t0:.1f}s", flush=True)


def stage_finalize():
    print("\n=== ISSUE #12 STAGE 4: APPLYING GOLDEN SELECTION RULE & SAVING phase7_feature_sets.json ===", flush=True)
    p6 = json.load(open(P6_FSETS))
    sw = _load_sweep()

    def _pick_best(entries: dict, base_key: str, parsimony_tol: float = 0.02):
        # Score = 0.5*f0 + 0.5*tm; among candidates within parsimony_tol of best AND not hurting temporal >0.03 vs base, pick smallest n_cols!
        base = entries[base_key]
        scored = []
        for k, v in entries.items():
            sc = 0.5 * v['f0']['mdape'] + 0.5 * v['tm']['mdape']
            scored.append((k, sc, v['n_cols'], v['f0']['mdape'], v['tm']['mdape'], v['cols']))
        best_sc = min(x[1] for x in scored)
        eligible = [x for x in scored if x[1] <= best_sc + parsimony_tol]
        # Prefer smallest feature set within tolerance
        eligible.sort(key=lambda x: (x[2], x[1]))
        win = eligible[0]
        print(f"  Winner vs {base_key} ({base['f0']['mdape']:.2f}/{base['tm']['mdape']:.2f}, {base['n_cols']}c): "
              f"{win[0]} ({win[3]:.2f}/{win[4]:.2f}, {win[2]}c | avg={win[1]:.3f})", flush=True)
        return win[0], win[5]

    ticl_e = {k.split('__', 1)[1]: v for k, v in sw.get('fm_sweep', {}).items() if k.startswith('tabicl__')}
    pfn_e = {k.split('__', 1)[1]: v for k, v in sw.get('fm_sweep', {}).items() if k.startswith('tabpfn__')}
    nn_e = sw.get('nn_sweep', {})

    win_ticl_k, win_ticl_cols = _pick_best(ticl_e, 'p6_baseline_98')
    win_pfn_k, win_pfn_cols = _pick_best(pfn_e, 'p6_baseline_98')
    win_nn_k, win_nn_cols = _pick_best(nn_e, 'p6_baseline_160')

    out = {
        'gbdt_feats': p6['gbdt_feats'],
        'gbdt_feats_with_agency': p6['gbdt_feats_with_agency'],
        'nn_feats': win_nn_cols,
        'nn_winner_name': win_nn_k,
        'tabicl_feats': win_ticl_cols,
        'tabicl_winner_name': win_ticl_k,
        'tabpfn_feats': win_pfn_cols,
        'tabpfn_winner_name': win_pfn_k,
    }
    with open(P7_FSETS_JSON, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"  Saved Phase-7 native per-family feature sets -> {P7_FSETS_JSON}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['perm', 'fm', 'fm_worker', 'nn', 'finalize', 'all'], default='all')
    ap.add_argument('--fm-kind', default='tabicl')
    ap.add_argument('--fm-label', default='')
    ap.add_argument('--fm-tmode', default='log_real_psqm')
    ap.add_argument('--fm-cols', default='[]')
    args = ap.parse_args()
    if args.stage == 'all':
        stages = ['fm', 'nn', 'finalize'] if os.path.exists(P7_PERM_CSV) else ['perm', 'fm', 'nn', 'finalize']
        for st in stages:
            print(f"\n>>> Issue #12 subprocess stage: {st} <<<", flush=True)
            subprocess.run([sys.executable, '-u', __file__, '--stage', st], check=True)
    elif args.stage == 'perm':
        stage_perm()
    elif args.stage == 'fm':
        stage_fm()
    elif args.stage == 'fm_worker':
        _run_fm_worker(args.fm_kind, args.fm_label, args.fm_cols, args.fm_tmode)
    elif args.stage == 'nn':
        stage_nn()
    elif args.stage == 'finalize':
        stage_finalize()


if __name__ == '__main__':
    main()
