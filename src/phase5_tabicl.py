"""Phase 5A — Tier 1 (Highest ROI, Zero-Training 2026 Foundation Model):
Run **TabICLv2** (ICML 2026 Tabular Foundation Model, `tabicl-regressor-v2-20260212.ckpt`)
across ALL 6 Phase-4 splits (5 grouped OOF folds + 2026 temporal holdout) on MPS GPU,
for BOTH target spaces (`log_real_price` and `log_real_psqm`), extracting BOTH `mean` and
`median` predictive-distribution point estimates + `q10`/`q90` native quantiles!

Also includes strict per-split checkpointing (`outputs/models/workbench/phase5_tabicl_ckpt.npz`)
so nothing is ever lost or recomputed.

Usage:
    .venv-api/bin/python -u src/phase5_tabicl.py
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'src'))

from phase3_features import compute_metrics, fmt
from phase3_modeling import _target_inverse, _target_transform
from phase4_frontier import _fit_convex

WORK_DIR = os.path.join(BASE_DIR, 'outputs', 'models', 'workbench')
CACHE_P4 = os.path.join(WORK_DIR, 'phase4_splits_cache.joblib')
ADDON_P5 = os.path.join(WORK_DIR, 'phase5_addon_features.joblib')
CKPT_PATH = os.path.join(WORK_DIR, 'phase5_tabicl_ckpt.npz')
OUT_NPZ = os.path.join(WORK_DIR, 'phase5_tabicl_preds.npz')
OUT_JSON = os.path.join(WORK_DIR, 'phase5_tabicl_report.json')

# Keep <= 115 top non-high-card features so TabICLv2 fits comfortably in MPS memory (~65s/split in High Power)
TOP_K_FEATURES = 115
DROP_HIGH_CARD = {'statbel_sector', 'city', 'immo_office_name'}
P5_PRIORITY_COLS = [
    'txt_tab_plus_text_psqm', 'txt_p5_super_tabresid', 'txt_mlp_psqm', 'txt_mlp_price',
    'txt_mlp_tabresid', 'txt_tfidf_tabresid_a6', 'txt_dense_both_tabresid', 'tab_anchor_log_psqm',
    'txt_qwen3_a5_psqm', 'txt_p5_super_psqm', 'freq_agency_norm',
]


def _prep_tabicl_df(tr_df: pd.DataFrame, va_df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    X_tr = tr_df[cols].copy()
    X_va = va_df[cols].copy()
    for c in cols:
        if X_tr[c].dtype == 'bool':
            X_tr[c] = X_tr[c].astype(np.float32)
            X_va[c] = X_va[c].astype(np.float32)
        elif X_tr[c].dtype == 'object' or str(X_tr[c].dtype) in ('string', 'category'):
            tr_s = X_tr[c].fillna('Missing').astype(str)
            va_s = X_va[c].fillna('Missing').astype(str)
            # Cap cardinality at 35 for TabICLv2 ordinal encoder stability
            top_cats = set(tr_s.value_counts().head(35).index)
            X_tr[c] = tr_s.where(tr_s.isin(top_cats), '__OTHER__')
            X_va[c] = va_s.where(va_s.isin(top_cats), '__OTHER__')
        else:
            X_tr[c] = pd.to_numeric(X_tr[c], errors='coerce').replace([np.inf, -np.inf], np.nan).astype(np.float32)
            X_va[c] = pd.to_numeric(X_va[c], errors='coerce').replace([np.inf, -np.inf], np.nan).astype(np.float32)
    return X_tr, X_va


def main():
    from tabicl import TabICLRegressor

    t0 = time.perf_counter()
    print("=== PHASE 5A (TIER 1): TabICLv2 (ICML 2026) ACROSS ALL 6 SPLITS ===", flush=True)
    data = joblib.load(CACHE_P4)
    addon = joblib.load(ADDON_P5)
    bundles = data['bundles']
    for b in bundles:
        sid = b['id']
        b['tr_df'] = pd.concat([b['tr_df'].reset_index(drop=True), addon[sid]['tr_add'].reset_index(drop=True)], axis=1)
        b['va_df'] = pd.concat([b['va_df'].reset_index(drop=True), addon[sid]['va_add'].reset_index(drop=True)], axis=1)
    y_price = data['y_price']
    y_tmp_true = data['y_tmp_true']
    folds = data['folds']
    is_house = data['is_house']
    N = len(y_price)

    with open(os.path.join(WORK_DIR, 'phase4_report.json')) as fh:
        p4_pruned = json.load(fh)['pruned_features']
    sample_cols = set(bundles[0]['tr_df'].columns)
    base_top = [c for c in p4_pruned if c in sample_cols and c not in DROP_HIGH_CARD]
    prio = [c for c in P5_PRIORITY_COLS if c in sample_cols]
    tabicl_cols = list(dict.fromkeys(prio + base_top))[:TOP_K_FEATURES]
    print(f"  Selected {len(tabicl_cols)} top Phase-5 features ({len(prio)} new Qwen3/MLP/TabResid + {len(tabicl_cols)-len(prio)} top P4) for TabICLv2", flush=True)

    # Load per-(target, split) checkpoint if resuming
    ckpt: dict[str, np.ndarray] = {}
    if os.path.exists(CKPT_PATH):
        zc = np.load(CKPT_PATH, allow_pickle=True)
        for k in zc.files:
            ckpt[k] = zc[k]
        print(f"  Resuming from checkpoint with {len(ckpt)} cached arrays.", flush=True)

    specs = [
        ('tabicl_real_price', 'log_real_price'),
        ('tabicl_real_psqm',  'log_real_psqm'),
    ]

    results_oof: dict[str, np.ndarray] = {}
    results_tmp: dict[str, np.ndarray] = {}
    quantiles_oof: dict[str, np.ndarray] = {}
    quantiles_tmp: dict[str, np.ndarray] = {}

    for base_name, target_mode in specs:
        oof_mean = np.zeros(N, dtype=float)
        oof_med = np.zeros(N, dtype=float)
        oof_q10 = np.zeros(N, dtype=float)
        oof_q90 = np.zeros(N, dtype=float)
        tmp_mean = tmp_med = tmp_q10 = tmp_q90 = None

        t_spec = time.perf_counter()
        for b in bundles:
            sid = b['id']
            ck_key = f'{base_name}__{sid}'
            if f'{ck_key}__mean' in ckpt:
                p_mean = ckpt[f'{ck_key}__mean']
                p_med = ckpt[f'{ck_key}__med']
                p_q10 = ckpt[f'{ck_key}__q10']
                p_q90 = ckpt[f'{ck_key}__q90']
                print(f"  [cached] {base_name} / {sid}", flush=True)
            else:
                ts = time.perf_counter()
                X_tr, X_va = _prep_tabicl_df(b['tr_df'], b['va_df'], tabicl_cols)
                y_tr = _target_transform(b['tr_df'], b['tr_cpi_idx'], target_mode)
                reg = TabICLRegressor(
                    device='mps',
                    n_estimators=8,
                    batch_size=1,
                    use_amp=True,
                    random_state=42,
                    verbose=False,
                )
                reg.fit(X_tr, y_tr)
                out = reg.predict(X_va, output_type=['mean', 'median', 'quantiles'], alphas=[0.10, 0.90])
                p_mean = _target_inverse(np.asarray(out['mean'], dtype=float), b['va_df'], b['va_cpi_idx'], target_mode)
                p_med = _target_inverse(np.asarray(out['median'], dtype=float), b['va_df'], b['va_cpi_idx'], target_mode)
                q_arr = np.asarray(out['quantiles'], dtype=float)  # shape (n_va, 2)
                p_q10 = _target_inverse(q_arr[:, 0], b['va_df'], b['va_cpi_idx'], target_mode)
                p_q90 = _target_inverse(q_arr[:, 1], b['va_df'], b['va_cpi_idx'], target_mode)

                ckpt[f'{ck_key}__mean'] = p_mean
                ckpt[f'{ck_key}__med'] = p_med
                ckpt[f'{ck_key}__q10'] = p_q10
                ckpt[f'{ck_key}__q90'] = p_q90
                np.savez_compressed(CKPT_PATH, **ckpt)

                y_va = b['va_df']['price'].to_numpy(dtype=float)
                ms = compute_metrics(y_va, p_med)
                print(f"  {base_name:18s} / {sid:8s} ({time.perf_counter()-ts:5.1f}s) | median: {fmt(ms)}", flush=True)
                del reg
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()

            if str(sid).startswith('temporal'):
                tmp_mean, tmp_med, tmp_q10, tmp_q90 = p_mean, p_med, p_q10, p_q90
            else:
                va_idx = b['va_idx']
                oof_mean[va_idx] = p_mean
                oof_med[va_idx] = p_med
                oof_q10[va_idx] = p_q10
                oof_q90[va_idx] = p_q90

        results_oof[f'{base_name}_mean'] = oof_mean
        results_tmp[f'{base_name}_mean'] = tmp_mean
        results_oof[f'{base_name}_med'] = oof_med
        results_tmp[f'{base_name}_med'] = tmp_med
        quantiles_oof[f'{base_name}_q10'] = oof_q10
        quantiles_oof[f'{base_name}_q90'] = oof_q90
        quantiles_tmp[f'{base_name}_q10'] = tmp_q10
        quantiles_tmp[f'{base_name}_q90'] = tmp_q90

        print(f"\n  >>> {base_name} FULL 6-SPLIT SUMMARY ({time.perf_counter()-t_spec:.1f}s):", flush=True)
        for suffix in ('mean', 'med'):
            k = f'{base_name}_{suffix}'
            print(f"      {k:24s} | OOF: {fmt(compute_metrics(y_price, results_oof[k]))}", flush=True)
            print(f"      {'':24s} | TMP: {fmt(compute_metrics(y_tmp_true, results_tmp[k]))}", flush=True)

    # Immediately benchmark adding TabICLv2 onto our Phase-4 19-model Frontier Ensemble!
    print("\n=== TIER-1 VERDICT: PHASE-4 ENSEMBLE + TabICLv2 (ICML 2026) ===", flush=True)
    z4 = np.load(os.path.join(WORK_DIR, 'phase4_frontier_preds.npz'), allow_pickle=True)
    p4_names = list(z4['names'])
    new_names = list(results_oof.keys())
    all_names = p4_names + new_names

    O_log = np.column_stack([z4['O_log']] + [np.log(np.maximum(results_oof[k], 10_000.0)) for k in new_names])
    T_log = np.column_stack([z4['T_log']] + [np.log(np.maximum(results_tmp[k], 10_000.0)) for k in new_names])
    y_log = np.log(y_price)

    # Nested-OOF unified convex blend
    ens_oof_u = np.zeros(N)
    for k in np.unique(folds):
        tr_m, va_m = folds != k, folds == k
        wk = _fit_convex(O_log[tr_m], y_log[tr_m])
        ens_oof_u[va_m] = np.exp(O_log[va_m] @ wk)
    w_full = _fit_convex(O_log, y_log)
    ens_tmp_u = np.exp(T_log @ w_full)

    # Type-specific convex blend
    ens_oof_t = np.zeros(N)
    ens_tmp_t = np.zeros(len(y_tmp_true))
    tmp_is_house = bundles[-1]['va_df']['property_type_HOUSE'].to_numpy(dtype=int)
    for h in (1, 0):
        mh = is_house == h
        for k in np.unique(folds):
            tr_m = mh & (folds != k)
            va_m = mh & (folds == k)
            wk = _fit_convex(O_log[tr_m], y_log[tr_m])
            ens_oof_t[va_m] = np.exp(O_log[va_m] @ wk)
        wf = _fit_convex(O_log[mh], y_log[mh])
        ens_tmp_t[tmp_is_house == h] = np.exp(T_log[tmp_is_house == h] @ wf)

    print("  Phase-4 STACK-1 (before TabICL) OOF:", fmt(compute_metrics(y_price, z4['ens_oof_unified'])), flush=True)
    print("  Phase-4 STACK-1 (before TabICL) TMP:", fmt(compute_metrics(y_tmp_true, z4['ens_tmp_unified'])), flush=True)
    print("  ★ P5-T1 UNIFIED (+TabICLv2)     OOF:", fmt(compute_metrics(y_price, ens_oof_u)), flush=True)
    print("  ★ P5-T1 UNIFIED (+TabICLv2)     TMP:", fmt(compute_metrics(y_tmp_true, ens_tmp_u)), flush=True)
    print("  ★ P5-T1 TYPED   (+TabICLv2)     OOF:", fmt(compute_metrics(y_price, ens_oof_t)), flush=True)
    print("  ★ P5-T1 TYPED   (+TabICLv2)     TMP:", fmt(compute_metrics(y_tmp_true, ens_tmp_t)), flush=True)

    print("\n  Unified Ensemble Weights (>=1%):", flush=True)
    for nm, w in sorted(zip(all_names, w_full), key=lambda x: -x[1]):
        if w >= 0.01:
            print(f"    {nm:30s}: {w*100:5.1f}%", flush=True)

    # Check native TabICLv2 80% quantile coverage!
    q10_oof = quantiles_oof['tabicl_real_price_q10']
    q90_oof = quantiles_oof['tabicl_real_price_q90']
    cov_oof = float(((y_price >= q10_oof) & (y_price <= q90_oof)).mean() * 100.0)
    hw_oof = float(np.median((q90_oof - q10_oof) / (2.0 * results_oof['tabicl_real_price_med'])) * 100.0)
    print(f"\n  TabICLv2 Native 80% Interval (raw, zero conformal calibration): OOF coverage={cov_oof:.2f}%, median halfwidth={hw_oof:.2f}%", flush=True)

    np.savez_compressed(
        OUT_NPZ,
        names=np.array(new_names, dtype=object),
        O_log=np.column_stack([np.log(np.maximum(results_oof[k], 10_000.0)) for k in new_names]),
        T_log=np.column_stack([np.log(np.maximum(results_tmp[k], 10_000.0)) for k in new_names]),
        ens_oof_u=ens_oof_u,
        ens_tmp_u=ens_tmp_u,
        ens_oof_t=ens_oof_t,
        ens_tmp_t=ens_tmp_t,
        q10_price_oof=quantiles_oof['tabicl_real_price_q10'],
        q90_price_oof=quantiles_oof['tabicl_real_price_q90'],
        q10_psqm_oof=quantiles_oof['tabicl_real_psqm_q10'],
        q90_psqm_oof=quantiles_oof['tabicl_real_psqm_q90'],
    )
    report = {
        'total_seconds': round(time.perf_counter() - t0, 1),
        'tabicl_features': tabicl_cols,
        'individual': {
            k: {'oof': compute_metrics(y_price, results_oof[k]), 'tmp': compute_metrics(y_tmp_true, results_tmp[k])}
            for k in new_names
        },
        'p5_t1_unified': {'oof': compute_metrics(y_price, ens_oof_u), 'tmp': compute_metrics(y_tmp_true, ens_tmp_u)},
        'p5_t1_typed': {'oof': compute_metrics(y_price, ens_oof_t), 'tmp': compute_metrics(y_tmp_true, ens_tmp_t)},
        'weights_unified': {nm: round(float(w), 4) for nm, w in zip(all_names, w_full) if w >= 0.005},
        'native_interval_oof': {'coverage_pct': round(cov_oof, 2), 'median_halfwidth_pct': round(hw_oof, 2)},
    }
    with open(OUT_JSON, 'w') as fh:
        json.dump(report, fh, indent=2)
    print(f"\nSaved {OUT_NPZ} & {OUT_JSON} in {time.perf_counter()-t0:.1f}s!", flush=True)


if __name__ == '__main__':
    main()
