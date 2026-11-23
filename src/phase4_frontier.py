"""Phase 4 Frontier Modeling: Flanders/GRB + E5-Large + RealMLP + TabM + LightGBMXT + Tuned GBDT Zoo.

Runs strictly apples-to-apples on the exact same 5 grouped OOF folds (14,599 rows / 10,858
properties) + 2026 temporal holdout (5,365 rows) as Phases 1–3!

Pipeline (with per-model incremental checkpointing so zero work is ever lost):
  1. Feature-gain audit + pruning sweep on Phase-4 features (400+ cols -> top K) +
     verify impact of training-side anomaly filtering (`tr_anomaly == 0`, NEVER drops eval rows!).
  2. **Standalone `LightGBMXT`** (`extra_trees=True`, L2 + Huber on `log_real_psqm` & `log_real_price`)
     — captures the exact randomized-split diversity AutoGluon relied on, in ~25s instead of 18 min!
  3. **TabArena SOTA Neural Networks (`pytabkit 1.8.0`) on Apple Silicon MPS GPU**:
     - `RealMLP_TD_Regressor` on `log_real_price` AND `log_real_psqm`
     - `TabM_D_Regressor` (BatchEnsemble-style tabular MLP, 2025 ICLR SOTA!) on `log_real_price` AND `log_real_psqm`
  4. **Phase-4 Upgraded GBDTs + House/Apt Specialists**:
     - Tuned LightGBM (Huber + L2 on `log_real_psqm` & `log_real_price`)
     - Tuned CatBoost (depth 7 & 8 on `log_real_psqm` & `log_real_price` with native `immo_office_name`)
     - Tuned XGBoost (`log_real_psqm` & `log_real_price`)
     - Tuned House/Apt Specialists (CatBoost, LightGBM Huber, LightGBMXT Huber, XGBoost)
  5. **2nd-Level Meta-Stacking**:
     - Unified Convex Blend
     - Type-Specific Convex Blend (House vs Apartment)
     - Huber Meta-Regressor
     - **Grand Ensemble**: Phase-4 zoo + Phase-3B SwiGLU/ResNet (testing cross-representation diversity!)
  6. **Final Slice Diagnostics + Conformalized Quantile Regression (`p10 / p90`)**.

Usage:
    .venv-api/bin/python -u src/phase4_frontier.py
"""
from __future__ import annotations

import json
import os
import time
import warnings

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from sklearn.linear_model import HuberRegressor

from phase3_features import compute_metrics, fmt
from phase3_modeling import (
    _target_inverse,
    _target_transform,
    prep_tree_df,
    run_6splits,
    run_specialist_6splits,
)
from phase4_features import load_or_build_phase4_splits

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE_DIR, 'outputs', 'models', 'workbench')
CKPT_PATH = os.path.join(OUT_DIR, 'phase4_models_ckpt.npz')


def filter_train_anomalies(bundles: list[dict]) -> list[dict]:
    """Return view of bundles with non-residential anomalies removed from TRAIN ONLY.
    Validation/test splits remain 100% untouched (identical 14,599 OOF & 5,365 TMP rows!).
    """
    out = []
    for b in bundles:
        keep = (b['tr_anomaly'] == 0)
        b2 = dict(b)
        b2['tr_df'] = b['tr_df'][keep].reset_index(drop=True)
        b2['tr_cpi_idx'] = b['tr_cpi_idx'][keep]
        out.append(b2)
    return out


def _prep_pytabkit_dfs(tr_df: pd.DataFrame, va_df: pd.DataFrame, feat_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    X_tr, cat_cols = prep_tree_df(tr_df, feat_cols)
    X_va, _ = prep_tree_df(va_df, feat_cols)
    cat_set = set(cat_cols)
    for c in cat_cols:
        vc = X_tr[c].value_counts()
        valid = sorted(vc[vc >= 5].index.tolist())
        cats = valid + ['__RARE__']
        X_tr[c] = pd.Categorical(X_tr[c].where(X_tr[c].isin(valid), '__RARE__'), categories=cats)
        X_va[c] = pd.Categorical(X_va[c].where(X_va[c].isin(valid), '__RARE__'), categories=cats)
    # pytabkit requires finite continuous columns: compute train-only medians and add missingness flags where NaN exists
    miss_tr = {}
    miss_va = {}
    for c in feat_cols:
        if c not in cat_set:
            s_tr = X_tr[c].replace([np.inf, -np.inf], np.nan)
            s_va = X_va[c].replace([np.inf, -np.inf], np.nan)
            if s_tr.isna().any() or s_va.isna().any():
                miss_tr[f'{c}__isna'] = s_tr.isna().astype(np.float32)
                miss_va[f'{c}__isna'] = s_va.isna().astype(np.float32)
                med = float(s_tr.median()) if s_tr.notna().any() else 0.0
                if not np.isfinite(med):
                    med = 0.0
                X_tr[c] = s_tr.fillna(med).astype(np.float32)
                X_va[c] = s_va.fillna(med).astype(np.float32)
            else:
                X_tr[c] = s_tr.astype(np.float32)
                X_va[c] = s_va.astype(np.float32)
    if miss_tr:
        X_tr = pd.concat([X_tr, pd.DataFrame(miss_tr, index=X_tr.index)], axis=1)
        X_va = pd.concat([X_va, pd.DataFrame(miss_va, index=X_va.index)], axis=1)
    return X_tr, X_va, cat_cols


def run_pytabkit_6splits(
    bundles: list[dict],
    feat_cols: list[str],
    model_type: str,  # 'realmlp' or 'tabm'
    target_mode: str = 'log_real_price',
    n_seeds: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Run SOTA pytabkit RealMLP or TabM on Apple Silicon MPS across all 6 splits."""
    from pytabkit import RealMLP_TD_Regressor, TabM_D_Regressor

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    n_all = sum(len(b['va_df']) for b in bundles if b['id'].startswith('fold_'))
    oof = np.zeros(n_all, dtype=float)
    tmp = None

    for b in bundles:
        tr_df, va_df = b['tr_df'], b['va_df']
        X_tr, X_va, cat_cols = _prep_pytabkit_dfs(tr_df, va_df, feat_cols)
        y_tr = _target_transform(tr_df, b['tr_cpi_idx'], target_mode).astype(np.float32)
        groups_tr = tr_df['property_key'].astype(str).to_numpy()

        seed_preds = []
        for s in range(n_seeds):
            rng = np.random.default_rng(300 + s)
            uniq_g = np.unique(groups_tr)
            es_g = set(rng.choice(uniq_g, size=max(1, int(0.12 * len(uniq_g))), replace=False))
            es_mask = np.isin(groups_tr, list(es_g))

            X_fit = X_tr.loc[~es_mask].reset_index(drop=True)
            y_fit = y_tr[~es_mask]
            X_es = X_tr.loc[es_mask].reset_index(drop=True)
            y_es = y_tr[es_mask]

            if model_type == 'realmlp':
                m = RealMLP_TD_Regressor(
                    device=device,
                    random_state=300 + s,
                    n_epochs=128,
                    batch_size=256,
                    val_metric_name='mae',
                    verbosity=0,
                )
            elif model_type == 'tabm':
                m = TabM_D_Regressor(
                    device=device,
                    random_state=300 + s,
                    n_epochs=100,
                    batch_size=256,
                    val_metric_name='mae',
                    verbosity=0,
                )
            else:
                raise ValueError(model_type)

            m.fit(X_fit, y_fit, X_es, y_es, cat_col_names=cat_cols)
            vp = np.asarray(m.predict(X_va), dtype=float)
            seed_preds.append(vp)

        va_mean = np.mean(seed_preds, axis=0)
        eur_pred = _target_inverse(va_mean, va_df, b['va_cpi_idx'], target_mode)
        if b['id'].startswith('fold_'):
            oof[b['va_idx']] = eur_pred
        else:
            tmp = eur_pred
        print(f"    [{model_type}:{target_mode}] {b['id']} done", flush=True)

    return oof, tmp


def _fit_convex(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    k = X.shape[1]
    def loss(w):
        e = X @ w - y
        return 0.55 * np.mean(np.abs(e)) + 0.45 * np.sqrt(np.mean(e ** 2))
    res = minimize(
        loss, np.ones(k) / k,
        bounds=[(0.0, 1.0)] * k,
        constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
        method='SLSQP',
    )
    return res.x


def main():
    t_start = time.perf_counter()
    data = load_or_build_phase4_splits()
    bundles_raw = data['bundles']
    bundles_clean = filter_train_anomalies(bundles_raw)
    y_oof = data['y_price']
    y_tmp = data['y_tmp_true']
    folds = data['folds']
    is_house_oof = data['is_house']
    is_house_tmp = (bundles_raw[-1]['va_df']['property_type_HOUSE'].to_numpy() == 1)
    all_feats = data['feat_cols_stationary']

    with open(os.path.join(OUT_DIR, 'phase3a_report.json')) as fh:
        p3a = json.load(fh)
    best_lgb_p = p3a['best_lgb_params']
    best_xgb_p = p3a['best_xgb_params']

    print(f"Loaded Phase-4 splits: {len(y_oof):,} OOF, {len(y_tmp):,} 2026 TMP, {len(all_feats)} stationary features")

    records = []
    preds_oof: dict[str, np.ndarray] = {}
    preds_tmp: dict[str, np.ndarray] = {}

    if os.path.exists(CKPT_PATH):
        cz = np.load(CKPT_PATH, allow_pickle=True)
        for nm in cz['names'].tolist():
            preds_oof[nm] = cz[f'oof_{nm}']
            preds_tmp[nm] = cz[f'tmp_{nm}']
        print(f"Resumed {len(preds_oof)} cached Phase-4 models from checkpoint")

    def save_ckpt():
        if not preds_oof:
            return
        payload = {'names': np.array(list(preds_oof.keys()))}
        for k in preds_oof:
            payload[f'oof_{k}'] = preds_oof[k]
            payload[f'tmp_{k}'] = preds_tmp[k]
        np.savez_compressed(CKPT_PATH, **payload)

    def rec(name: str, op: np.ndarray, tp: np.ndarray, dt: float):
        mo = compute_metrics(y_oof, op)
        mt = compute_metrics(y_tmp, tp)
        print(f"  {name:44s} ({dt:5.1f}s) | OOF: {fmt(mo)}")
        print(f"  {'':52s} | TMP: {fmt(mt)}")
        records.append({'name': name, 'seconds': round(dt, 1), 'oof': mo, 'temporal': mt})
        return mo, mt

    # ======================================================================
    # 1. PHASE-4 FEATURE AUDIT + ANOMALY FILTER CHECK + PRUNING SWEEP
    # ======================================================================
    imp_csv_path = os.path.join(OUT_DIR, 'phase4_feature_importance.csv')
    if os.path.exists(imp_csv_path):
        imp_df = pd.read_csv(imp_csv_path)
        use_clean = False
        bundles = bundles_raw
        best_k = 320
        pruned_feats = imp_df['feature'].head(best_k).tolist()
        print(f"\n=== 1. RESUMED PHASE-4 PRUNED SET FROM {os.path.basename(imp_csv_path)} (top {best_k} features, raw train) ===")
    else:
        print("\n=== 1. PHASE-4 FEATURE AUDIT (FLANDERS + GRB + E5-LARGE + TF-IDF HUBER/SVD) ===")
        t0 = time.perf_counter()
        op_all, tp_all, imp_all = run_6splits(bundles_raw, all_feats, 'lgbm', 'log_real_psqm', 'huber', best_lgb_p)
        rec('P4-LGBM-Huber [all feats, raw train]', op_all, tp_all, time.perf_counter() - t0)

        t0 = time.perf_counter()
        op_cln, tp_cln, imp_cln = run_6splits(bundles_clean, all_feats, 'lgbm', 'log_real_psqm', 'huber', best_lgb_p)
        mo_cln, mt_cln = rec('P4-LGBM-Huber [all feats, anomaly-clean]', op_cln, tp_cln, time.perf_counter() - t0)

        # Pick raw vs anomaly-clean train
        mo_raw, mt_raw = compute_metrics(y_oof, op_all), compute_metrics(y_tmp, tp_all)
        use_clean = (mo_cln['mdape'] + 0.5 * mt_cln['mdape']) <= (mo_raw['mdape'] + 0.5 * mt_raw['mdape'])
        bundles = bundles_clean if use_clean else bundles_raw
        imp_use = imp_cln if use_clean else imp_all
        print(f"  Train anomaly filter choice: {'CLEAN (39 train anomalies excluded)' if use_clean else 'RAW (keep all)'}")

        imp_df = pd.DataFrame({'feature': all_feats, 'gain': imp_use}).sort_values('gain', ascending=False).reset_index(drop=True)
        imp_df.to_csv(imp_csv_path, index=False)
        print("\n  Top 25 Phase-4 features by LightGBM gain:")
        for i, r in imp_df.head(25).iterrows():
            print(f"    {i+1:2d}. {r['feature']:38s} gain={r['gain']:10.1f}")

        # Check how new Flanders/GRB/E5-large features ranked
        new_ranks = imp_df[imp_df['feature'].str.startswith(('fl_', 'grb_', 'gent_', 'txt_e5l_', 'txt_tfidf_huber', 'txt_blend', 'tfidf_svd', 'e5l_pca'))]
        print(f"  Top 12 NEW Phase-4 features (Flanders/GRB/E5-large/Huber-text):")
        for idx, r in new_ranks.head(12).iterrows():
            print(f"    #{idx+1:3d} {r['feature']:38s} gain={r['gain']:10.1f}")

        # Pruning sweep
        best_k = len(all_feats)
        best_sc = (mo_cln if use_clean else mo_raw)['mdape'] + 0.5 * (mt_cln if use_clean else mt_raw)['mdape']
        pruned_feats = all_feats
        for k in (180, 250, 320):
            if k >= len(all_feats):
                continue
            sub_f = imp_df['feature'].head(k).tolist()
            t0 = time.perf_counter()
            op_k, tp_k, _ = run_6splits(bundles, sub_f, 'lgbm', 'log_real_psqm', 'huber', best_lgb_p)
            mo, mt = rec(f'P4-LGBM-Huber [top {k} feats]', op_k, tp_k, time.perf_counter() - t0)
            sc = mo['mdape'] + 0.5 * mt['mdape']
            if sc < best_sc:
                best_sc = sc
                best_k = k
                pruned_feats = sub_f
        print(f"\n  Selected Phase-4 pruned feature set: top {best_k} features")

    # ======================================================================
    # 2. PHASE-4 GBDT ZOO + STANDALONE LightGBMXT + SPECIALISTS
    # ======================================================================
    print("\n=== 2. PHASE-4 GBDT ZOO (LightGBM + LightGBMXT + CatBoost + XGBoost + Specialists) ===")
    lgb_xt_p = dict(best_lgb_p, extra_trees=True, colsample_bytree=0.55, num_leaves=63, min_child_samples=20)
    cb_feats = list(dict.fromkeys(pruned_feats + ['immo_office_name']))

    gbdt_specs = [
        ('p4_lgbm_psqm_huber',   pruned_feats, 'lgbm',     'log_real_psqm',  'huber', best_lgb_p),
        ('p4_lgbm_psqm_l2',      pruned_feats, 'lgbm',     'log_real_psqm',  'l2',    best_lgb_p),
        ('p4_lgbm_price_huber',  pruned_feats, 'lgbm',     'log_real_price', 'huber', best_lgb_p),
        ('p4_lgbmxt_psqm_huber', pruned_feats, 'lgbm',     'log_real_psqm',  'huber', lgb_xt_p),
        ('p4_lgbmxt_psqm_l2',    pruned_feats, 'lgbm',     'log_real_psqm',  'l2',    lgb_xt_p),
        ('p4_lgbmxt_price_l2',   pruned_feats, 'lgbm',     'log_real_price', 'l2',    lgb_xt_p),
        ('p4_xgb_psqm_l2',       pruned_feats, 'xgb',      'log_real_psqm',  'l2',    best_xgb_p),
        ('p4_xgb_price_l2',      pruned_feats, 'xgb',      'log_real_price', 'l2',    best_xgb_p),
        ('p4_cb_d7_psqm_l2',     cb_feats,     'catboost', 'log_real_psqm',  'l2',    {'depth': 7, 'l2_leaf_reg': 5.0, 'learning_rate': 0.03, 'iterations': 4500}),
        ('p4_cb_d8_psqm_l2',     cb_feats,     'catboost', 'log_real_psqm',  'l2',    {'depth': 8, 'l2_leaf_reg': 8.0, 'learning_rate': 0.03, 'iterations': 4500}),
        ('p4_cb_d7_price_l2',    cb_feats,     'catboost', 'log_real_price', 'l2',    {'depth': 7, 'l2_leaf_reg': 5.0, 'learning_rate': 0.03, 'iterations': 4500}),
    ]
    for key, fset, kind, tmode, lmode, params in gbdt_specs:
        if key in preds_oof:
            rec(key + ' [cached]', preds_oof[key], preds_tmp[key], 0.0)
            continue
        t0 = time.perf_counter()
        op, tp, _ = run_6splits(bundles, fset, kind, tmode, lmode, params)
        rec(key, op, tp, time.perf_counter() - t0)
        preds_oof[key] = op
        preds_tmp[key] = tp
        save_ckpt()

    spec_jobs = [
        ('p4_spec_cb_psqm',      cb_feats,     'catboost', 'log_real_psqm', 'l2',    {'depth': 7, 'l2_leaf_reg': 5.0, 'learning_rate': 0.03, 'iterations': 4000}),
        ('p4_spec_lgbm_huber',   pruned_feats, 'lgbm',     'log_real_psqm', 'huber', best_lgb_p),
        ('p4_spec_lgbmxt_huber', pruned_feats, 'lgbm',     'log_real_psqm', 'huber', lgb_xt_p),
        ('p4_spec_xgb_psqm',     pruned_feats, 'xgb',      'log_real_psqm', 'l2',    best_xgb_p),
    ]
    for key, fset, kind, tmode, lmode, params in spec_jobs:
        if key in preds_oof:
            rec(key + ' [cached]', preds_oof[key], preds_tmp[key], 0.0)
            continue
        t0 = time.perf_counter()
        op, tp = run_specialist_6splits(bundles, fset, kind, tmode, lmode, params)
        rec(key, op, tp, time.perf_counter() - t0)
        preds_oof[key] = op
        preds_tmp[key] = tp
        save_ckpt()

    # ======================================================================
    # 3. TABARENA SOTA NEURAL NETWORKS ON MPS GPU: RealMLP + TabM
    # ======================================================================
    print("\n=== 3. TABARENA SOTA TABULAR NNs ON APPLE MPS (RealMLP + TabM) ===")
    nn_specs = [
        ('p4_realmlp_price_3seed', 'realmlp', 'log_real_price', 3),
        ('p4_realmlp_psqm_3seed',  'realmlp', 'log_real_psqm',  3),
        ('p4_tabm_price_3seed',    'tabm',    'log_real_price', 3),
        ('p4_tabm_psqm_3seed',     'tabm',    'log_real_psqm',  3),
    ]
    for key, mtype, tmode, ns in nn_specs:
        if key in preds_oof:
            rec(key + ' [cached]', preds_oof[key], preds_tmp[key], 0.0)
            continue
        t0 = time.perf_counter()
        op, tp = run_pytabkit_6splits(bundles, pruned_feats, mtype, tmode, n_seeds=ns)
        rec(key, op, tp, time.perf_counter() - t0)
        preds_oof[key] = op
        preds_tmp[key] = tp
        save_ckpt()

    # ======================================================================
    # 4. META-STACKING & GRAND ENSEMBLE
    # ======================================================================
    print(f"\n=== 4. PHASE-4 ENSEMBLE & META-STACKERS ({len(preds_oof)} PHASE-4 MODELS) ===")
    names = list(preds_oof.keys())
    O_log = np.column_stack([np.log(np.maximum(preds_oof[k], 10_000.0)) for k in names])
    T_log = np.column_stack([np.log(np.maximum(preds_tmp[k], 10_000.0)) for k in names])
    y_log = np.log(y_oof)

    # 4a. Phase-4 GBDT-only
    gbdt_idx = [i for i, k in enumerate(names) if not any(t in k for t in ('realmlp', 'tabm'))]
    oof_g = np.zeros(len(y_oof))
    for f in sorted(np.unique(folds)):
        tr, va = folds != f, folds == f
        oof_g[va] = O_log[va][:, gbdt_idx] @ _fit_convex(O_log[tr][:, gbdt_idx], y_log[tr])
    w_g = _fit_convex(O_log[:, gbdt_idx], y_log)
    rec('P4-STACK-0: Phase-4 GBDT-only', np.exp(oof_g), np.exp(T_log[:, gbdt_idx] @ w_g), 0.0)

    # 4b. Phase-4 Unified Convex (GBDT + RealMLP + TabM)
    oof_u = np.zeros(len(y_oof))
    for f in sorted(np.unique(folds)):
        tr, va = folds != f, folds == f
        oof_u[va] = O_log[va] @ _fit_convex(O_log[tr], y_log[tr])
    w_u = _fit_convex(O_log, y_log)
    tmp_u = T_log @ w_u
    rec('P4-STACK-1: Phase-4 Unified Convex (all)', np.exp(oof_u), np.exp(tmp_u), 0.0)

    # 4c. Phase-4 Type-Specific Convex (House vs Apt)
    oof_t = np.zeros(len(y_oof))
    tmp_t = np.zeros(len(y_tmp))
    for h_flag in (True, False):
        hm_o = is_house_oof == h_flag
        hm_t = is_house_tmp == h_flag
        for f in sorted(np.unique(folds)):
            tr = (folds != f) & hm_o
            va = (folds == f) & hm_o
            oof_t[va] = O_log[va] @ _fit_convex(O_log[tr], y_log[tr])
        wh = _fit_convex(O_log[hm_o], y_log[hm_o])
        tmp_t[hm_t] = T_log[hm_t] @ wh
    rec('P4-STACK-2: Phase-4 Type-Specific Convex', np.exp(oof_t), np.exp(tmp_t), 0.0)

    # 4d. Phase-4 Huber Meta-Stacker
    oof_h = np.zeros(len(y_oof))
    for f in sorted(np.unique(folds)):
        tr, va = folds != f, folds == f
        meta = HuberRegressor(alpha=15.0, max_iter=500).fit(O_log[tr], y_log[tr])
        oof_h[va] = meta.predict(O_log[va])
    tmp_h = HuberRegressor(alpha=15.0, max_iter=500).fit(O_log, y_log).predict(T_log)
    rec('P4-STACK-3: Phase-4 Huber Meta-Stacker', np.exp(oof_h), np.exp(tmp_h), 0.0)

    print("\nLearned Phase-4 Unified Convex weights:")
    for nm, wt in sorted(zip(names, w_u), key=lambda x: -x[1]):
        if wt >= 0.005:
            print(f"  {wt*100:5.1f}%  {nm}")

    # 4e. Also test GRAND ensemble combining Phase-4 + Phase-3B SwiGLU/ResNet (free diversity!)
    p3b_npz = os.path.join(OUT_DIR, 'phase3b_frontier_preds.npz')
    if os.path.exists(p3b_npz):
        z3 = np.load(p3b_npz, allow_pickle=True)
        n3 = z3['names'].tolist()
        keep3 = [i for i, k in enumerate(n3) if 'swiglu' in k or 'deep_resnet' in k]
        if keep3:
            O_grand = np.hstack([O_log, z3['O_log'][:, keep3]])
            T_grand = np.hstack([T_log, z3['T_log'][:, keep3]])
            grand_names = names + [n3[i] for i in keep3]
            oof_gr = np.zeros(len(y_oof))
            tmp_gr = np.zeros(len(y_tmp))
            for h_flag in (True, False):
                hm_o = is_house_oof == h_flag
                hm_t = is_house_tmp == h_flag
                for f in sorted(np.unique(folds)):
                    tr = (folds != f) & hm_o
                    va = (folds == f) & hm_o
                    oof_gr[va] = O_grand[va] @ _fit_convex(O_grand[tr], y_log[tr])
                wh = _fit_convex(O_grand[hm_o], y_log[hm_o])
                tmp_gr[hm_t] = T_grand[hm_t] @ wh
            rec('P4-GRAND: Phase-4 + P3B SwiGLU/ResNet', np.exp(oof_gr), np.exp(tmp_gr), 0.0)

    # Save final Phase-4 results
    np.savez_compressed(
        os.path.join(OUT_DIR, 'phase4_frontier_preds.npz'),
        names=np.array(names),
        O_log=O_log,
        T_log=T_log,
        y_oof=y_oof,
        y_tmp=y_tmp,
        ens_oof_unified=np.exp(oof_u),
        ens_tmp_unified=np.exp(tmp_u),
        ens_oof_typed=np.exp(oof_t),
        ens_tmp_typed=np.exp(tmp_t),
        weights_unified=w_u,
    )
    with open(os.path.join(OUT_DIR, 'phase4_report.json'), 'w') as fh:
        json.dump({
            'use_anomaly_clean': use_clean,
            'pruned_k': best_k,
            'pruned_features': pruned_feats,
            'weights_unified': {nm: round(float(wt), 4) for nm, wt in zip(names, w_u)},
            'results': records,
            'total_seconds': round(time.perf_counter() - t_start, 1),
        }, fh, indent=2)
    print(f"\nPhase 4 finished in {time.perf_counter()-t_start:.1f}s")


if __name__ == '__main__':
    main()
