"""Phase 5C — Tier 3 (Upgraded Model Zoo on Phase-5 Qwen3 + Tabular-Residual Features):
Trains ONLY the highest-leverage diverse models on the upgraded Phase-5 feature set
(`phase4_splits_cache.joblib` + `phase5_addon_features.joblib`), with strict per-model
checkpointing (`outputs/models/workbench/phase5_tier3_ckpt.npz`):

  1. **Specialist `TabM` (House vs Apartment)** — our #1 single Phase-4 family, NEVER tried as
     dedicated House/Apt specialists before! (both `log_real_price` and `log_real_psqm`)
  2. **Upgraded Unified `TabM` (3 seeds)** on Phase-5 features (`log_real_price` & `log_real_psqm`)
  3. **Upgraded `RealMLP_TD_S` / `RealMLP_TD` (3 seeds)** on Phase-5 features (`log_real_price`)
  4. **Upgraded `MLP_PLR_D` (Periodic Numerical Embeddings MLP, pytabkit)** — a brand-new 2025/2026
     NN architecture that uses periodic sinusoidal embeddings for continuous coordinates/areas!
  5. **Upgraded GBDT core** on Phase-5 features (`p5_cb_d7_price_l2`, `p5_cb_d7_psqm_l2`,
     `p5_lgbm_price_huber`, `p5_lgbmxt_psqm_huber`, `p5_spec_xgb_psqm`, `p5_spec_cb_psqm`)
  6. **Upgraded `TabICLv2` (ICML 2026)** on the top-115 Phase-5 features (`log_real_price` &
     `log_real_psqm`, mean + median) — testing whether `Qwen3` + `tabular-residual` text features
     lift the foundation model too!

Finally, builds the **Phase-5 Grand Stack** across ALL Phase-4 + Tier-1 `TabICLv2` + Tier-3 models
(strict nested-OOF SLSQP + type-specific convex + mega-area guard), saves everything to
`outputs/models/workbench/phase5_frontier_preds.npz` and `phase5_report.json`.

Usage:
    .venv-api/bin/python -u src/phase5_frontier.py
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
from phase3_modeling import _target_inverse, _target_transform, run_6splits, run_specialist_6splits
from phase4_frontier import _fit_convex, _prep_pytabkit_dfs, run_pytabkit_6splits

WORK_DIR = os.path.join(BASE_DIR, 'outputs', 'models', 'workbench')
CACHE_P4 = os.path.join(WORK_DIR, 'phase4_splits_cache.joblib')
ADDON_P5 = os.path.join(WORK_DIR, 'phase5_addon_features.joblib')
CKPT_PATH = os.path.join(WORK_DIR, 'phase5_tier3_ckpt.npz')
OUT_NPZ = os.path.join(WORK_DIR, 'phase5_frontier_preds.npz')
OUT_JSON = os.path.join(WORK_DIR, 'phase5_report.json')


def run_specialist_pytabkit_6splits(
    bundles: list[dict],
    feat_cols: list[str],
    model_type: str,
    target_mode: str,
    n_seeds: int = 3,
    n_epochs: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Train separate HOUSE and APARTMENT pytabkit NNs per split on MPS GPU."""
    from pytabkit import MLP_PLR_D_Regressor, RealMLP_TD_Regressor, TabM_D_Regressor

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    N = int(max(b['va_idx'].max() for b in bundles if not str(b['id']).startswith('temporal')) + 1)
    oof = np.zeros(N, dtype=float)
    tmp_pred = None

    for b in bundles:
        va_df_full = b['va_df']
        va_pred_full = np.zeros(len(va_df_full), dtype=float)
        for h_flag in (1, 0):
            tr_m = b['tr_df']['property_type_HOUSE'].to_numpy() == h_flag
            va_m = va_df_full['property_type_HOUSE'].to_numpy() == h_flag
            if va_m.sum() == 0:
                continue
            tr_sub = b['tr_df'].loc[tr_m].reset_index(drop=True)
            va_sub = va_df_full.loc[va_m].reset_index(drop=True)
            tr_cpi_sub = b['tr_cpi_idx'][tr_m]
            va_cpi_sub = b['va_cpi_idx'][va_m]

            X_tr, X_va, cat_cols = _prep_pytabkit_dfs(tr_sub, va_sub, feat_cols)
            y_tr = _target_transform(tr_sub, tr_cpi_sub, target_mode).astype(np.float32)
            groups_tr = tr_sub['property_key'].astype(str).to_numpy()

            seed_preds = []
            for s in range(n_seeds):
                rng = np.random.default_rng(500 + s * 19 + h_flag * 7)
                uniq_g = np.unique(groups_tr)
                es_g = set(rng.choice(uniq_g, size=max(1, int(0.12 * len(uniq_g))), replace=False))
                es_mask = np.isin(groups_tr, list(es_g))
                if model_type == 'tabm':
                    m = TabM_D_Regressor(device=device, random_state=500 + s, n_epochs=n_epochs, batch_size=256, val_metric_name='mae', verbosity=0)
                elif model_type == 'realmlp':
                    m = RealMLP_TD_Regressor(device=device, random_state=500 + s, n_epochs=n_epochs, batch_size=256, val_metric_name='mae', verbosity=0)
                else:
                    m = MLP_PLR_D_Regressor(device=device, random_state=500 + s, n_epochs=n_epochs, batch_size=256, val_metric_name='mae', verbosity=0)
                m.fit(
                    X_tr.loc[~es_mask].reset_index(drop=True), y_tr[~es_mask],
                    X_tr.loc[es_mask].reset_index(drop=True), y_tr[es_mask],
                    cat_col_names=cat_cols,
                )
                seed_preds.append(np.asarray(m.predict(X_va), dtype=float))
                del m
            va_pred_full[va_m] = _target_inverse(np.mean(seed_preds, axis=0), va_sub, va_cpi_sub, target_mode)
            if device == 'mps':
                torch.mps.empty_cache()

        if str(b['id']).startswith('temporal'):
            tmp_pred = va_pred_full
        else:
            oof[b['va_idx']] = va_pred_full
    return oof, tmp_pred


def run_mlp_plr_6splits(bundles: list[dict], feat_cols: list[str], target_mode: str, n_seeds: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Run pytabkit MLP_PLR_D_Regressor (Periodic Numerical Embeddings) across all 6 splits."""
    from pytabkit import MLP_PLR_D_Regressor

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    N = int(max(b['va_idx'].max() for b in bundles if not str(b['id']).startswith('temporal')) + 1)
    oof = np.zeros(N, dtype=float)
    tmp_pred = None

    for b in bundles:
        X_tr, X_va, cat_cols = _prep_pytabkit_dfs(b['tr_df'], b['va_df'], feat_cols)
        y_tr = _target_transform(b['tr_df'], b['tr_cpi_idx'], target_mode).astype(np.float32)
        groups_tr = b['tr_df']['property_key'].astype(str).to_numpy()
        seed_preds = []
        for s in range(n_seeds):
            rng = np.random.default_rng(600 + s * 17)
            uniq_g = np.unique(groups_tr)
            es_g = set(rng.choice(uniq_g, size=max(1, int(0.12 * len(uniq_g))), replace=False))
            es_mask = np.isin(groups_tr, list(es_g))
            m = MLP_PLR_D_Regressor(device=device, random_state=600 + s, n_epochs=100, batch_size=256, val_metric_name='mae', verbosity=0)
            m.fit(
                X_tr.loc[~es_mask].reset_index(drop=True), y_tr[~es_mask],
                X_tr.loc[es_mask].reset_index(drop=True), y_tr[es_mask],
                cat_col_names=cat_cols,
            )
            seed_preds.append(np.asarray(m.predict(X_va), dtype=float))
            del m
        pred = _target_inverse(np.mean(seed_preds, axis=0), b['va_df'], b['va_cpi_idx'], target_mode)
        if str(b['id']).startswith('temporal'):
            tmp_pred = pred
        else:
            oof[b['va_idx']] = pred
        if device == 'mps':
            torch.mps.empty_cache()
    return oof, tmp_pred


def main():
    t0 = time.perf_counter()
    print("=== PHASE 5C (TIER 3): UPGRADED MODELS ON QWEN3 + TABULAR-RESIDUAL FEATURES ===", flush=True)
    data = joblib.load(CACHE_P4)
    addon = joblib.load(ADDON_P5)
    bundles = data['bundles']
    y_price = data['y_price']
    y_tmp_true = data['y_tmp_true']
    folds = data['folds']
    is_house = data['is_house']
    N = len(y_price)

    # Merge Phase-5 add-on columns onto each split in memory (fast, <1s!)
    new_cols = list(addon[bundles[0]['id']]['tr_add'].columns)
    for b in bundles:
        sid = b['id']
        b['tr_df'] = pd.concat([b['tr_df'].reset_index(drop=True), addon[sid]['tr_add'].reset_index(drop=True)], axis=1)
        b['va_df'] = pd.concat([b['va_df'].reset_index(drop=True), addon[sid]['va_add'].reset_index(drop=True)], axis=1)

    with open(os.path.join(WORK_DIR, 'phase4_report.json')) as fh:
        p4_rep = json.load(fh)
    with open(os.path.join(WORK_DIR, 'phase3a_report.json')) as fh:
        p3a = json.load(fh)

    best_lgb_p = p3a['best_lgb_params']
    best_xgb_p = p3a['best_xgb_params']
    lgb_xt_p = dict(best_lgb_p, extra_trees=True, colsample_bytree=0.55, num_leaves=63, min_child_samples=20)

    # Keep ONLY the genuinely strong new Phase-5 cols (Multi-Task Text MLP + Tabular-Residual + Qwen3)
    # and skip noisy raw 32-d Qwen3 PCA dims (keep compact!)
    drop_noisy_p5 = {c for c in new_cols if c.startswith('qwen3_pca_') and int(c.split('_')[-1]) >= 12}
    useful_new = [c for c in new_cols if c not in drop_noisy_p5]
    sample_cols = set(bundles[0]['tr_df'].columns)
    p5_feats = list(dict.fromkeys([c for c in p4_rep['pruned_features'][:280] if c in sample_cols] + [c for c in useful_new if c in sample_cols]))
    cb_feats = list(dict.fromkeys(p5_feats + ['immo_office_name']))
    print(f"  Phase-5 unified feature count: {len(p5_feats)} (280 top Phase-4 + {len(useful_new)} compact Qwen3/TabResid/MLP cols)", flush=True)

    # Load checkpoint if resuming
    preds_oof: dict[str, np.ndarray] = {}
    preds_tmp: dict[str, np.ndarray] = {}
    if os.path.exists(CKPT_PATH):
        zc = np.load(CKPT_PATH, allow_pickle=True)
        for nm in list(zc['names']):
            preds_oof[str(nm)] = zc[f'oof_{nm}']
            preds_tmp[str(nm)] = zc[f'tmp_{nm}']
        print(f"  Loaded {len(preds_oof)} cached Phase-5 Tier-3 models from checkpoint.", flush=True)

    def save_ckpt():
        payload = {'names': np.array(list(preds_oof.keys()), dtype=object)}
        for k in preds_oof:
            payload[f'oof_{k}'] = preds_oof[k]
            payload[f'tmp_{k}'] = preds_tmp[k]
        np.savez_compressed(CKPT_PATH, **payload)

    records = {}
    def rec(name: str, o: np.ndarray, t: np.ndarray, dt: float):
        mo = compute_metrics(y_price, o)
        mt = compute_metrics(y_tmp_true, t)
        records[name] = {'seconds': round(dt, 1), 'oof': mo, 'tmp': mt}
        print(f"  {name:30s} ({dt:5.1f}s) | OOF: {fmt(mo)}", flush=True)
        print(f"  {'':38s} | TMP: {fmt(mt)}", flush=True)

    # 1. Upgraded Phase-5 GBDT core (fast, ~30-90s each across 6 parallel folds)
    print("\n--- 1. Phase-5 Upgraded GBDT Core ---", flush=True)
    gbdt_specs = [
        ('p5_lgbm_price_huber',  p5_feats, 'lgbm',     'log_real_price', 'huber', best_lgb_p),
        ('p5_lgbmxt_psqm_huber', p5_feats, 'lgbm',     'log_real_psqm',  'huber', lgb_xt_p),
        ('p5_cb_d7_price_l2',    cb_feats, 'catboost', 'log_real_price', 'l2',    {'depth': 7, 'l2_leaf_reg': 5.0, 'learning_rate': 0.03, 'iterations': 4500}),
        ('p5_cb_d7_psqm_l2',     cb_feats, 'catboost', 'log_real_psqm',  'l2',    {'depth': 7, 'l2_leaf_reg': 5.0, 'learning_rate': 0.03, 'iterations': 4500}),
    ]
    for nm, fset, kind, tmode, lmode, params in gbdt_specs:
        if nm in preds_oof:
            rec(nm, preds_oof[nm], preds_tmp[nm], 0.0)
            continue
        ts = time.perf_counter()
        o, t, _ = run_6splits(bundles, fset, kind, tmode, lmode, params)
        preds_oof[nm], preds_tmp[nm] = o, t
        save_ckpt()
        rec(nm, o, t, time.perf_counter() - ts)

    spec_gbdt = [
        ('p5_spec_xgb_psqm', p5_feats, 'xgb',      'log_real_psqm', 'l2', best_xgb_p),
        ('p5_spec_cb_psqm',  cb_feats, 'catboost', 'log_real_psqm', 'l2', {'depth': 7, 'l2_leaf_reg': 5.0, 'learning_rate': 0.03, 'iterations': 4000}),
    ]
    for nm, fset, kind, tmode, lmode, params in spec_gbdt:
        if nm in preds_oof:
            rec(nm, preds_oof[nm], preds_tmp[nm], 0.0)
            continue
        ts = time.perf_counter()
        o, t = run_specialist_6splits(bundles, fset, kind, tmode, lmode, params)
        preds_oof[nm], preds_tmp[nm] = o, t
        save_ckpt()
        rec(nm, o, t, time.perf_counter() - ts)

    # 2. Upgraded SOTA Tabular Neural Nets on MPS GPU (TabM unified + House/Apt Specialist TabM + RealMLP + MLP-PLR!)
    print("\n--- 2. Phase-5 SOTA Tabular Neural Nets on MPS (TabM, Specialist TabM, RealMLP, MLP-PLR) ---", flush=True)
    nn_jobs = [
        ('p5_tabm_price_3seed',      'unified',    'tabm',    'log_real_price', 3, 100),
        ('p5_tabm_psqm_3seed',       'unified',    'tabm',    'log_real_psqm',  3, 100),
        ('p5_spec_tabm_price_3seed', 'specialist', 'tabm',    'log_real_price', 3, 100),
        ('p5_spec_tabm_psqm_3seed',  'specialist', 'tabm',    'log_real_psqm',  3, 100),
        ('p5_realmlp_price_3seed',   'unified',    'realmlp', 'log_real_price', 3, 128),
        ('p5_mlp_plr_price_3seed',   'mlp_plr',    'mlp_plr', 'log_real_price', 3, 100),
    ]
    for nm, mode, mtype, tmode, ns, nep in nn_jobs:
        if nm in preds_oof:
            rec(nm, preds_oof[nm], preds_tmp[nm], 0.0)
            continue
        ts = time.perf_counter()
        if mode == 'unified':
            o, t = run_pytabkit_6splits(bundles, p5_feats, mtype, tmode, n_seeds=ns)
        elif mode == 'specialist':
            o, t = run_specialist_pytabkit_6splits(bundles, p5_feats, mtype, tmode, n_seeds=ns, n_epochs=nep)
        else:
            o, t = run_mlp_plr_6splits(bundles, p5_feats, tmode, n_seeds=ns)
        preds_oof[nm], preds_tmp[nm] = o, t
        save_ckpt()
        rec(nm, o, t, time.perf_counter() - ts)

    # 3. GRAND PHASE-5 META-ENSEMBLE (combining Phase-5 Zoo + Tier-1 TabICLv2 + Phase-4 Zoo!)
    print("\n=== 3. GRAND PHASE-5 ENSEMBLE COMPARISON ===", flush=True)
    z4 = np.load(os.path.join(WORK_DIR, 'phase4_frontier_preds.npz'), allow_pickle=True)
    zt = np.load(os.path.join(WORK_DIR, 'phase5_tabicl_preds.npz'), allow_pickle=True)

    p5_names = list(preds_oof.keys())
    t_names = list(zt['names'])
    p4_names = list(z4['names'])

    O_p5 = np.column_stack([np.log(np.maximum(preds_oof[k], 10_000.0)) for k in p5_names])
    T_p5 = np.column_stack([np.log(np.maximum(preds_tmp[k], 10_000.0)) for k in p5_names])

    y_log = np.log(y_price)
    tmp_is_house = bundles[-1]['va_df']['property_type_HOUSE'].to_numpy(dtype=int)

    def eval_pool(label: str, pool_names: list[str], O_mat: np.ndarray, T_mat: np.ndarray):
        # Unified nested-OOF convex
        oof_u = np.zeros(N)
        for k in np.unique(folds):
            tr_m, va_m = folds != k, folds == k
            wk = _fit_convex(O_mat[tr_m], y_log[tr_m])
            oof_u[va_m] = np.exp(O_mat[va_m] @ wk)
        w_u = _fit_convex(O_mat, y_log)
        tmp_u = np.exp(T_mat @ w_u)

        # Type-specific nested-OOF convex
        oof_t = np.zeros(N)
        tmp_t = np.zeros(len(y_tmp_true))
        w_t = {}
        for h in (1, 0):
            mh = is_house == h
            for k in np.unique(folds):
                tr_m = mh & (folds != k)
                va_m = mh & (folds == k)
                wk = _fit_convex(O_mat[tr_m], y_log[tr_m])
                oof_t[va_m] = np.exp(O_mat[va_m] @ wk)
            wf = _fit_convex(O_mat[mh], y_log[mh])
            w_t[h] = wf
            tmp_t[tmp_is_house == h] = np.exp(T_mat[tmp_is_house == h] @ wf)

        print(f"\n  [{label}] ({len(pool_names)} models)", flush=True)
        print(f"    UNIFIED OOF: {fmt(compute_metrics(y_price, oof_u))}", flush=True)
        print(f"    UNIFIED TMP: {fmt(compute_metrics(y_tmp_true, tmp_u))}", flush=True)
        print(f"    TYPED   OOF: {fmt(compute_metrics(y_price, oof_t))}", flush=True)
        print(f"    TYPED   TMP: {fmt(compute_metrics(y_tmp_true, tmp_t))}", flush=True)
        return oof_u, tmp_u, oof_t, tmp_t, w_u, w_t

    print("  [Reference] Phase-4 STACK-1 OOF:", fmt(compute_metrics(y_price, z4['ens_oof_unified'])), flush=True)
    print("  [Reference] Phase-4 STACK-1 TMP:", fmt(compute_metrics(y_tmp_true, z4['ens_tmp_unified'])), flush=True)

    # Pool A: Phase-5 standalone (12 new models + TabICLv2 = 16 models, clean & production-lean!)
    names_A = p5_names + t_names
    O_A = np.column_stack([O_p5, zt['O_log']])
    T_A = np.column_stack([T_p5, zt['T_log']])
    oof_Au, tmp_Au, oof_At, tmp_At, w_Au, w_At = eval_pool('P5-CLEAN (P5 Zoo + TabICLv2)', names_A, O_A, T_A)

    # Pool B: Full Grand Zoo (Phase-5 + TabICLv2 + Phase-4 = 35 models, max cross-representation diversity)
    names_B = p5_names + t_names + p4_names
    O_B = np.column_stack([O_p5, zt['O_log'], z4['O_log']])
    T_B = np.column_stack([T_p5, zt['T_log'], z4['T_log']])
    oof_Bu, tmp_Bu, oof_Bt, tmp_Bt, w_Bu, w_Bt = eval_pool('P5-GRAND (P5 + TabICLv2 + P4)', names_B, O_B, T_B)

    print("\n  P5-CLEAN Unified Weights (>=1%):", flush=True)
    for nm, w in sorted(zip(names_A, w_Au), key=lambda x: -x[1]):
        if w >= 0.01:
            print(f"    {nm:30s}: {w*100:5.1f}%", flush=True)

    print("\n  P5-GRAND Unified Weights (>=1%):", flush=True)
    for nm, w in sorted(zip(names_B, w_Bu), key=lambda x: -x[1]):
        if w >= 0.01:
            print(f"    {nm:30s}: {w*100:5.1f}%", flush=True)

    np.savez_compressed(
        OUT_NPZ,
        names_p5=np.array(p5_names, dtype=object),
        O_p5=O_p5, T_p5=T_p5,
        names_clean=np.array(names_A, dtype=object), w_clean_u=w_Au,
        oof_clean_u=oof_Au, tmp_clean_u=tmp_Au, oof_clean_t=oof_At, tmp_clean_t=tmp_At,
        names_grand=np.array(names_B, dtype=object), w_grand_u=w_Bu,
        oof_grand_u=oof_Bu, tmp_grand_u=tmp_Bu, oof_grand_t=oof_Bt, tmp_grand_t=tmp_Bt,
        y_oof=y_price, y_tmp=y_tmp_true,
    )
    report = {
        'total_seconds': round(time.perf_counter() - t0, 1),
        'p5_features': p5_feats,
        'models': records,
        'p5_clean_unified': {'oof': compute_metrics(y_price, oof_Au), 'tmp': compute_metrics(y_tmp_true, tmp_Au)},
        'p5_clean_typed':   {'oof': compute_metrics(y_price, oof_At), 'tmp': compute_metrics(y_tmp_true, tmp_At)},
        'p5_grand_unified': {'oof': compute_metrics(y_price, oof_Bu), 'tmp': compute_metrics(y_tmp_true, tmp_Bu)},
        'p5_grand_typed':   {'oof': compute_metrics(y_price, oof_Bt), 'tmp': compute_metrics(y_tmp_true, tmp_Bt)},
        'weights_clean_u':  {nm: round(float(w), 4) for nm, w in zip(names_A, w_Au) if w >= 0.005},
        'weights_grand_u':  {nm: round(float(w), 4) for nm, w in zip(names_B, w_Bu) if w >= 0.005},
    }
    with open(OUT_JSON, 'w') as fh:
        json.dump(report, fh, indent=2)
    print(f"\nSaved {OUT_NPZ} & {OUT_JSON} in {time.perf_counter()-t0:.1f}s!", flush=True)


if __name__ == '__main__':
    main()
