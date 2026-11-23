"""Phase 5 Fast Dev Screen (2-Split Profile: fold_0 + temporal_2026, 2 seeds):
Evaluates candidate architectures & feature budgets in ~1/4 the time of full 6-split runs:
  1. Compares `p5_tabm_price`, `p5_spec_tabm_price`, `p5_spec_tabm_psqm`, `p5_mlp_plr_price`,
     `p5_realmlp_price` directly against Phase-4 + TabICLv2 on `fold_0` (2,942 homes) and
     `temporal_2026` (5,365 homes).
  2. Quick feature-budget screen on `TabM` (top-120 vs top-180 vs all-313) & `TabICLv2` (top-75 vs top-115).

Promote ONLY winners to full 6-split 3-seed training!
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

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'src'))

from phase3_features import compute_metrics, fmt
from phase3_modeling import _target_inverse, _target_transform
from phase4_frontier import _prep_pytabkit_dfs

WORK_DIR = os.path.join(BASE_DIR, 'outputs', 'models', 'workbench')
CACHE_P4 = os.path.join(WORK_DIR, 'phase4_splits_cache.joblib')
ADDON_P5 = os.path.join(WORK_DIR, 'phase5_addon_features.joblib')
OUT_JSON = os.path.join(WORK_DIR, 'phase5_fast_screen.json')


def fit_predict_nn_one_split(b: dict, feat_cols: list[str], model_type: str, target_mode: str, n_seeds: int = 2, specialist: bool = False) -> np.ndarray:
    from pytabkit import MLP_PLR_D_Regressor, RealMLP_TD_Regressor, TabM_D_Regressor

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    va_df_full = b['va_df']
    out = np.zeros(len(va_df_full), dtype=float)
    groups_iter = (1, 0) if specialist else (None,)

    for h_flag in groups_iter:
        if h_flag is None:
            tr_sub = b['tr_df'].reset_index(drop=True)
            va_sub = va_df_full.reset_index(drop=True)
            tr_cpi = b['tr_cpi_idx']
            va_cpi = b['va_cpi_idx']
            va_mask = np.ones(len(va_df_full), dtype=bool)
        else:
            tr_m = b['tr_df']['property_type_HOUSE'].to_numpy() == h_flag
            va_mask = va_df_full['property_type_HOUSE'].to_numpy() == h_flag
            tr_sub = b['tr_df'].loc[tr_m].reset_index(drop=True)
            va_sub = va_df_full.loc[va_mask].reset_index(drop=True)
            tr_cpi = b['tr_cpi_idx'][tr_m]
            va_cpi = b['va_cpi_idx'][va_mask]

        X_tr, X_va, cat_cols = _prep_pytabkit_dfs(tr_sub, va_sub, feat_cols)
        y_tr = _target_transform(tr_sub, tr_cpi, target_mode).astype(np.float32)
        groups_tr = tr_sub['property_key'].astype(str).to_numpy()

        seed_preds = []
        for s in range(n_seeds):
            rng = np.random.default_rng(700 + s * 17 + (0 if h_flag is None else h_flag * 5))
            uniq_g = np.unique(groups_tr)
            es_g = set(rng.choice(uniq_g, size=max(1, int(0.12 * len(uniq_g))), replace=False))
            es_mask = np.isin(groups_tr, list(es_g))
            if model_type == 'tabm':
                m = TabM_D_Regressor(device=device, random_state=700 + s, n_epochs=100, batch_size=256, val_metric_name='mae', verbosity=0)
            elif model_type == 'realmlp':
                m = RealMLP_TD_Regressor(device=device, random_state=700 + s, n_epochs=128, batch_size=256, val_metric_name='mae', verbosity=0)
            else:
                m = MLP_PLR_D_Regressor(device=device, random_state=700 + s, max_epochs=100, batch_size=256, val_metric_name='mae', verbosity=0)
            m.fit(
                X_tr.loc[~es_mask].reset_index(drop=True), y_tr[~es_mask],
                X_tr.loc[es_mask].reset_index(drop=True), y_tr[es_mask],
                cat_col_names=cat_cols,
            )
            seed_preds.append(np.asarray(m.predict(X_va), dtype=float))
            del m
        out[va_mask] = _target_inverse(np.mean(seed_preds, axis=0), va_sub, va_cpi, target_mode)
        if device == 'mps':
            torch.mps.empty_cache()
    return out


def main():
    t0 = time.perf_counter()
    print("=== PHASE 5 FAST DEV SCREEN (fold_0 + temporal_2026, 2 seeds) ===", flush=True)
    data = joblib.load(CACHE_P4)
    addon = joblib.load(ADDON_P5)

    b_f0 = data['bundles'][0]
    b_tmp = data['bundles'][-1]
    for b in (b_f0, b_tmp):
        sid = b['id']
        b['tr_df'] = pd.concat([b['tr_df'].reset_index(drop=True), addon[sid]['tr_add'].reset_index(drop=True)], axis=1)
        b['va_df'] = pd.concat([b['va_df'].reset_index(drop=True), addon[sid]['va_add'].reset_index(drop=True)], axis=1)

    y_f0 = data['y_price'][b_f0['va_idx']]
    y_tmp = data['y_tmp_true']

    # Reference baselines on the exact same 2 splits
    z4 = np.load(os.path.join(WORK_DIR, 'phase4_frontier_preds.npz'), allow_pickle=True)
    zt = np.load(os.path.join(WORK_DIR, 'phase5_tabicl_preds.npz'), allow_pickle=True)
    p4_f0 = z4['ens_oof_unified'][b_f0['va_idx']]
    p4_tmp = z4['ens_tmp_unified']
    p5t1_f0 = zt['ens_oof_u'][b_f0['va_idx']]
    p5t1_tmp = zt['ens_tmp_u']

    print(f"  [REF] Phase-4 STACK-1      | f0: {fmt(compute_metrics(y_f0, p4_f0))}", flush=True)
    print(f"                             | tm: {fmt(compute_metrics(y_tmp, p4_tmp))}", flush=True)
    print(f"  [REF] P5-T1 (+TabICLv2)    | f0: {fmt(compute_metrics(y_f0, p5t1_f0))}", flush=True)
    print(f"                             | tm: {fmt(compute_metrics(y_tmp, p5t1_tmp))}", flush=True)

    with open(os.path.join(WORK_DIR, 'phase4_report.json')) as fh:
        p4_pruned = json.load(fh)['pruned_features']
    new_cols = list(addon[b_f0['id']]['tr_add'].columns)
    prio_new = [c for c in new_cols if not c.startswith('qwen3_pca_')]
    compact_new = [c for c in new_cols if not (c.startswith('qwen3_pca_') and int(c.split('_')[-1]) >= 12)]
    scols = set(b_f0['tr_df'].columns)

    feats_313 = list(dict.fromkeys([c for c in p4_pruned[:280] if c in scols] + [c for c in compact_new if c in scols]))
    feats_140 = list(dict.fromkeys([c for c in prio_new if c in scols] + [c for c in p4_pruned if c in scols]))[:140]

    candidates = [
        ('spec_tabm_psqm_140',  feats_140, 'tabm',    'log_real_psqm',  True),
        ('mlp_plr_psqm_140',    feats_140, 'mlp_plr', 'log_real_psqm',  False),
        ('realmlp_psqm_140',    feats_140, 'realmlp', 'log_real_psqm',  False),
    ]

    report = {
        'ref_p4': {'f0': compute_metrics(y_f0, p4_f0), 'tmp': compute_metrics(y_tmp, p4_tmp)},
        'ref_p5t1': {'f0': compute_metrics(y_f0, p5t1_f0), 'tmp': compute_metrics(y_tmp, p5t1_tmp)},
        'candidates': {},
    }
    print("\n--- Fast 2-Split Candidate Screen (Standalone + 20% Blend into P5-T1) ---", flush=True)
    for name, fcols, mtype, tmode, spec in candidates:
        ts = time.perf_counter()
        pf0 = fit_predict_nn_one_split(b_f0, fcols, mtype, tmode, n_seeds=2, specialist=spec)
        ptm = fit_predict_nn_one_split(b_tmp, fcols, mtype, tmode, n_seeds=2, specialist=spec)
        dt = time.perf_counter() - ts
        mf0 = compute_metrics(y_f0, pf0)
        mtm = compute_metrics(y_tmp, ptm)
        # Check 80/20 geometric blend with current best P5-T1!
        bf0 = compute_metrics(y_f0, np.exp(0.80 * np.log(p5t1_f0) + 0.20 * np.log(pf0)))
        btm = compute_metrics(y_tmp, np.exp(0.80 * np.log(p5t1_tmp) + 0.20 * np.log(ptm)))
        report['candidates'][name] = {'seconds': round(dt, 1), 'n_feats': len(fcols), 'f0': mf0, 'tmp': mtm, 'blend20_f0': bf0, 'blend20_tmp': btm}
        print(f"  {name:22s} ({dt:5.1f}s, d={len(fcols):3d}) | f0: {mf0['mdape']:5.2f}% (MAE €{mf0['mae']:,.0f}) | tm: {mtm['mdape']:5.2f}% (MAE €{mtm['mae']:,.0f})", flush=True)
        print(f"    -> +20% into P5-T1 ({p5t1_f0.size}/{y_tmp.size})     | f0: {bf0['mdape']:5.2f}% (Δ={bf0['mdape']-report['ref_p5t1']['f0']['mdape']:+.2f}pp) | tm: {btm['mdape']:5.2f}% (Δ={btm['mdape']-report['ref_p5t1']['tmp']['mdape']:+.2f}pp)", flush=True)

    with open(OUT_JSON, 'w') as fh:
        json.dump(report, fh, indent=2)
    print(f"\nSaved {OUT_JSON} in {time.perf_counter()-t0:.1f}s!", flush=True)


if __name__ == '__main__':
    main()
