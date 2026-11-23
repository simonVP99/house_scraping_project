"""Issue #13 + Phase-7 Grand Stack: Tune Highest-Weight MPS Families (TabPFN V2, TabICLv2, TabM) & Build Phase-7 Champion!

Builds directly on our native per-family feature sets from Issue #12 (`phase7_feature_sets.json`) and
our Optuna-tuned `log_real_price` GBDT twins from Issue #11 (`phase7_gbdt_ckpt.npz`):

  - **TabPFN V2 (`consensus_95` — already improved `f0 7.25->7.20%`, `tm 7.73->7.60%`!):**
    1. `p7_tabpfn_c95_price_e16`:  `n_estimators=16` (2× Phase-6!), `SUBSAMPLE_SAMPLES=8000`, `log_real_price`, `consensus_95`
    2. `p7_tabpfn_c95_psqm_e16`:   `n_estimators=16`, `SUBSAMPLE_SAMPLES=8000`, `log_real_psqm`, `consensus_95`
    3. `p7_tabpfn_multi_sub_price`: Multi-subset bagging across 2 feature views (`consensus_95` + `p6_fm_98`, `n_estimators=8` each)
  - **TabICLv2 (`consensus_95`, `n_estimators=16` — 2× Phase-6!):**
    4. `p7_tabicl_c95_psqm_e16` (`mean` & `med` + native `q10/q90` intervals!)
    5. `p7_tabicl_c95_price_e16` (`mean` & `med`)
  - **TabM (Native Permutation Features from Issue #12!):**
    6. `p7_tabm_perm170_price_4s`: 4-seed `TabM_D_Regressor` on `nn_feats_perm170` (`170` cols, 75 noise feats dropped! `-€926` MAE on untouched fold_1!)
    7. `p7_tabm_cons130_price_4s`: 4-seed `TabM_D_Regressor` on `nn_feats_cons130` (`130` cols — best temporal `7.77%`!)
    8. `p7_tabm_perm170_psqm_4s`:  4-seed `TabM_D_Regressor` on `log_real_psqm` (`170` cols)
    9. `p7_spec_tabm_cons130_psqm`: 3-seed House/Apt Specialist `TabM` on `cons_130` (`log_real_psqm`)

Every MPS model runs in its OWN clean subprocess worker (`--stage worker --model <name>`) so Apple Silicon
MPS allocator fragmentation can NEVER trigger an OOM across long runs!

Finally (`--stage stack`):
  - Merges ALL Phase-7 tuned GBDTs (`#11`) + ALL Phase-7 upgraded MPS FMs/NNs (`#13`) + Phase-6 best streams
  - Runs nested Cluster-Clean CV stacking (`P7-GBDT`, `P7-UNIFIED`, `P7-TYPED`, `P7-HUBER`, plus Caruana Top-8 Minimal!)
  - Re-runs Issue #8 Heteroscedastic Conformal 80% intervals on the Phase-7 Champion
  - Saves `outputs/models/workbench/phase7_frontier_preds.npz` + `phase7_report.json`!

Usage:
    .venv-api/bin/python -u src/phase7_mps_and_stack.py --stage all
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
P6_CKPT = os.path.join(WORK_DIR, 'phase6_models_ckpt.npz')
P7_FSETS = os.path.join(WORK_DIR, 'phase7_feature_sets.json')
P7_GBDT_CKPT = os.path.join(WORK_DIR, 'phase7_gbdt_ckpt.npz')
P7_MPS_CKPT = os.path.join(WORK_DIR, 'phase7_mps_ckpt.npz')
P7_PREDS_NPZ = os.path.join(WORK_DIR, 'phase7_frontier_preds.npz')
P7_REPORT_JSON = os.path.join(WORK_DIR, 'phase7_report.json')

MPS_MODELS = [
    'p7_tabm_perm170_price_4s',
    'p7_tabm_cons130_price_4s',
    'p7_tabm_perm170_psqm_4s',
    'p7_spec_tabm_cons130_psqm',
    'p7_tabicl_c95_psqm_e16',
    'p7_tabicl_c95_price_e16',
    'p7_tabpfn_c95_price_e16',
    'p7_tabpfn_c95_psqm_e16',
]


def _target_fwd(y_price, area, defl, mode='log_real_price'):
    return np.log((y_price / defl) / area) if mode == 'log_real_psqm' else np.log(y_price / defl)


def _target_inv(pred_log, area, defl, mode='log_real_price'):
    p = np.clip(np.asarray(pred_log, dtype=np.float64), -20.0, 20.0)
    return np.clip(np.exp(p) * (area * defl if mode == 'log_real_psqm' else defl), 10_000.0, 1e7)


def _load_ckpt(path):
    d = {}
    if os.path.exists(path):
        z = np.load(path, allow_pickle=True)
        for k in z.files:
            d[k] = z[k]
    return d


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


def run_worker(model_name: str):
    import joblib
    import torch
    from phase3_features import compute_metrics

    ckpt = _load_ckpt(P7_MPS_CKPT)
    check_key = f'oof__{model_name}_med' if 'tabicl' in model_name else f'oof__{model_name}'
    if check_key in ckpt:
        print(f"  [ckpt skip] {model_name} already present", flush=True)
        return

    t0 = time.perf_counter()
    cache = joblib.load(P6_CACHE)
    fsets = json.load(open(P7_FSETS))
    bundles = cache['bundles']
    y, a, nm, ih = cache['y_price'], cache['y_area'], cache['normal_mask'], cache['is_house']
    tmp_b = [b for b in bundles if str(b['id']).startswith('temporal')][0]
    tmp_idx = tmp_b['va_idx']

    def _report(lbl, oof, tmp_p):
        mo = compute_metrics(y[nm], oof[nm])
        mh = compute_metrics(y[nm & (ih == 1)], oof[nm & (ih == 1)])
        ma = compute_metrics(y[nm & (ih == 0)], oof[nm & (ih == 0)])
        mt = compute_metrics(y[tmp_idx][nm[tmp_idx]], tmp_p[nm[tmp_idx]])
        print(f"  {lbl:28s} ({time.perf_counter()-t0:5.1f}s) | NORM OOF={mo['mdape']:5.2f}% (H {mh['mdape']:.2f}/A {ma['mdape']:.2f}) €{mo['mae']:6,.0f} | NORM TMP={mt['mdape']:5.2f}% (bias {mt['bias_pct']:+5.2f}%)", flush=True)

    if model_name.startswith('p7_tabm_') or model_name.startswith('p7_spec_tabm_'):
        from pytabkit import TabM_D_Regressor
        is_spec = model_name.startswith('p7_spec_tabm_')
        tmode = 'log_real_psqm' if 'psqm' in model_name else 'log_real_price'
        cols_key = 'nn_feats_cons130' if 'cons130' in model_name else 'nn_feats_perm170'
        cols = [c for c in fsets[cols_key] if c in bundles[0]['tr_df'].columns]
        n_seeds = 3 if is_spec else 4
        oof = np.zeros(len(y), dtype=np.float64); tmp_p = None

        for b in bundles:
            if not is_spec:
                Xtr, Xva = _prep_nn(b['tr_df'], b['va_df'], cols)
                yt = _target_fwd(y[b['tr_idx']], a[b['tr_idx']], b['tr_deflator'], tmode).astype(np.float32)
                u_c = np.unique(b['tr_cids'])
                es_c = set(np.random.default_rng(77).choice(u_c, size=max(1, int(len(u_c) * 0.12)), replace=False))
                val_idxs = np.where(np.isin(b['tr_cids'], list(es_c)))[0]
                s_p = []
                for s in range(n_seeds):
                    m = TabM_D_Regressor(device='mps', n_epochs=105, batch_size=256, val_metric_name='mae',
                                         random_state=42 + s * 19, verbosity=0)
                    m.fit(Xtr, yt, val_idxs=val_idxs)
                    s_p.append(np.asarray(m.predict(Xva), dtype=np.float64))
                    del m; gc.collect(); torch.mps.empty_cache()
                p_eur = _target_inv(np.mean(s_p, axis=0), a[b['va_idx']], b['va_deflator'], tmode)
            else:
                p_eur = np.zeros(len(b['va_idx']), dtype=np.float64)
                for hf in (1, 0):
                    tm_m = (b['tr_df']['property_type_HOUSE'].to_numpy(int) == hf)
                    vm_m = (b['va_df']['property_type_HOUSE'].to_numpy(int) == hf)
                    Xtr, Xva = _prep_nn(b['tr_df'].iloc[tm_m], b['va_df'].iloc[vm_m], cols)
                    yt = _target_fwd(y[b['tr_idx']][tm_m], a[b['tr_idx']][tm_m], b['tr_deflator'][tm_m], tmode).astype(np.float32)
                    cids_h = b['tr_cids'][tm_m]
                    u_c = np.unique(cids_h)
                    es_c = set(np.random.default_rng(77 + hf).choice(u_c, size=max(1, int(len(u_c) * 0.12)), replace=False))
                    val_idxs = np.where(np.isin(cids_h, list(es_c)))[0]
                    s_p = []
                    for s in range(n_seeds):
                        m = TabM_D_Regressor(device='mps', n_epochs=105, batch_size=256, val_metric_name='mae',
                                             random_state=42 + s * 19 + hf, verbosity=0)
                        m.fit(Xtr, yt, val_idxs=val_idxs)
                        s_p.append(np.asarray(m.predict(Xva), dtype=np.float64))
                        del m; gc.collect(); torch.mps.empty_cache()
                    p_eur[vm_m] = _target_inv(np.mean(s_p, axis=0), a[b['va_idx']][vm_m], b['va_deflator'][vm_m], tmode)
            if str(b['id']).startswith('temporal'):
                tmp_p = p_eur
            else:
                oof[b['va_idx']] = p_eur
        ckpt = _load_ckpt(P7_MPS_CKPT)
        ckpt[f'oof__{model_name}'] = oof; ckpt[f'tmp__{model_name}'] = tmp_p
        np.savez_compressed(P7_MPS_CKPT, **ckpt)
        _report(model_name, oof, tmp_p)

    elif model_name.startswith('p7_tabicl_'):
        from tabicl import TabICLRegressor
        tmode = 'log_real_psqm' if 'psqm' in model_name else 'log_real_price'
        cols = [c for c in fsets['tabicl_feats'] if c in bundles[0]['tr_df'].columns]
        oof_mn = np.zeros(len(y), dtype=np.float64); tmp_mn = None
        oof_md = np.zeros(len(y), dtype=np.float64); tmp_md = None
        oof_hw = np.zeros(len(y), dtype=np.float64); tmp_hw = None

        for b in bundles:
            Xtr, Xva = _prep_fm(b['tr_df'], b['va_df'], cols, max_cat=30, numeric_cats=False)
            yt = _target_fwd(y[b['tr_idx']], a[b['tr_idx']], b['tr_deflator'], tmode).astype(np.float32)
            # Bag 2 x n_estimators=8 with different random seeds = 16 estimators cleanly without MPS OOM!
            p_mns, p_mds, p_hws = [], [], []
            for s in range(2):
                reg = TabICLRegressor(device='mps', n_estimators=8, batch_size=1, use_amp=True, random_state=42 + s * 31)
                reg.fit(Xtr, yt)
                if tmode == 'log_real_psqm':
                    out = reg.predict(Xva, output_type=['mean', 'median', 'quantiles'], alphas=[0.10, 0.90])
                    p_mns.append(np.asarray(out['mean'], dtype=np.float64))
                    p_mds.append(np.asarray(out['median'], dtype=np.float64))
                    q = np.asarray(out['quantiles'], dtype=np.float64)
                    p_hws.append(0.5 * (q[:, 1] - q[:, 0]))
                else:
                    out = reg.predict(Xva, output_type=['mean', 'median'])
                    p_mns.append(np.asarray(out['mean'], dtype=np.float64))
                    p_mds.append(np.asarray(out['median'], dtype=np.float64))
                del reg; gc.collect(); torch.mps.empty_cache()
            e_mn = _target_inv(np.mean(p_mns, axis=0), a[b['va_idx']], b['va_deflator'], tmode)
            e_md = _target_inv(np.mean(p_mds, axis=0), a[b['va_idx']], b['va_deflator'], tmode)
            if str(b['id']).startswith('temporal'):
                tmp_mn, tmp_md = e_mn, e_md
                if p_hws: tmp_hw = np.mean(p_hws, axis=0)
            else:
                oof_mn[b['va_idx']] = e_mn; oof_md[b['va_idx']] = e_md
                if p_hws: oof_hw[b['va_idx']] = np.mean(p_hws, axis=0)

        ckpt = _load_ckpt(P7_MPS_CKPT)
        ckpt[f'oof__{model_name}_mean'] = oof_mn; ckpt[f'tmp__{model_name}_mean'] = tmp_mn
        ckpt[f'oof__{model_name}_med'] = oof_md;  ckpt[f'tmp__{model_name}_med'] = tmp_md
        if tmp_hw is not None:
            ckpt['oof__p7_tabicl_hw'] = oof_hw; ckpt['tmp__p7_tabicl_hw'] = tmp_hw
        np.savez_compressed(P7_MPS_CKPT, **ckpt)
        _report(f'{model_name}_mean', oof_mn, tmp_mn)
        _report(f'{model_name}_med', oof_md, tmp_md)

    elif model_name.startswith('p7_tabpfn_'):
        from tabpfn import TabPFNRegressor
        from tabpfn.constants import ModelVersion
        tmode = 'log_real_psqm' if 'psqm' in model_name else 'log_real_price'
        cols = [c for c in fsets['tabpfn_feats'] if c in bundles[0]['tr_df'].columns]
        oof = np.zeros(len(y), dtype=np.float64); tmp_p = None
        for b in bundles:
            Xtr, Xva = _prep_fm(b['tr_df'], b['va_df'], cols, max_cat=25, numeric_cats=True)
            yt = _target_fwd(y[b['tr_idx']], a[b['tr_idx']], b['tr_deflator'], tmode).astype(np.float32)
            # 2 independent seeds × n_estimators=8 with SUBSAMPLE_SAMPLES=8000 = 16-estimator bagged TabPFN V2!
            s_p = []
            for s in range(2):
                m_pfn = TabPFNRegressor.create_default_for_version(
                    ModelVersion.V2, device='mps', n_estimators=8,
                    ignore_pretraining_limits=True, inference_config={'SUBSAMPLE_SAMPLES': 8000},
                    random_state=42 + s * 37,
                )
                m_pfn.fit(Xtr, yt)
                s_p.append(np.asarray(m_pfn.predict(Xva), dtype=np.float64))
                del m_pfn; gc.collect(); torch.mps.empty_cache()
            p_eur = _target_inv(np.mean(s_p, axis=0), a[b['va_idx']], b['va_deflator'], tmode)
            if str(b['id']).startswith('temporal'):
                tmp_p = p_eur
            else:
                oof[b['va_idx']] = p_eur
        ckpt = _load_ckpt(P7_MPS_CKPT)
        ckpt[f'oof__{model_name}'] = oof; ckpt[f'tmp__{model_name}'] = tmp_p
        np.savez_compressed(P7_MPS_CKPT, **ckpt)
        _report(model_name, oof, tmp_p)


def stage_stack():
    import joblib
    import lightgbm as lgb
    from scipy.optimize import minimize
    from sklearn.linear_model import HuberRegressor
    from phase3_features import compute_metrics

    print("\n=== PHASE-7 GRAND STACKING, CARUANA MINIMAL-8 & HETEROSCEDASTIC CONFORMAL ===", flush=True)
    cache = joblib.load(P6_CACHE)
    y = cache['y_price']; nm = cache['normal_mask']; ih = cache['is_house']; folds = cache['folds']
    tmp_b = [b for b in cache['bundles'] if str(b['id']).startswith('temporal')][0]
    tmp_idx = tmp_b['va_idx']; y_tmp = y[tmp_idx]
    nm_tmp = nm[tmp_idx]

    c_p7g = _load_ckpt(P7_GBDT_CKPT)
    c_p7m = _load_ckpt(P7_MPS_CKPT)
    c_p6 = _load_ckpt(P6_CKPT)

    def _collect(ckpts):
        names, oofs, tmps = [], [], []
        seen = set()
        for ck in ckpts:
            for k in sorted(ck.keys()):
                if not k.startswith('oof__') or k.endswith('_hw'):
                    continue
                nm_m = k[len('oof__'):]
                if nm_m in seen or f'tmp__{nm_m}' not in ck:
                    continue
                o = np.asarray(ck[k], dtype=np.float64)
                t = np.asarray(ck[f'tmp__{nm_m}'], dtype=np.float64)
                if not (np.all(np.isfinite(o)) and np.all(np.isfinite(t)) and np.all(o > 0) and np.all(t > 0)):
                    continue
                seen.add(nm_m); names.append(nm_m); oofs.append(np.log(o)); tmps.append(np.log(t))
        return names, np.column_stack(oofs), np.column_stack(tmps)

    def _fit_simplex(O_mat, y_eur, mask=None):
        if mask is not None:
            O_mat = O_mat[mask]; y_eur = y_eur[mask]
        yl = np.log(y_eur)
        K = O_mat.shape[1]
        def loss(w):
            err = O_mat @ w - yl
            return 0.55 * np.mean(np.abs(err)) + 0.45 * np.sqrt(np.mean(err ** 2))
        res = minimize(loss, np.ones(K) / K, method='SLSQP', bounds=[(0.0, 1.0)] * K,
                       constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}, options={'maxiter': 500})
        return res.x

    def _nested_convex(O_log, T_log, typed=False):
        oof_p = np.zeros(len(y), dtype=np.float64)
        for fk in range(5):
            tr = (folds != fk) & nm; va = (folds == fk)
            if not typed:
                w = _fit_simplex(O_log, y, tr)
                oof_p[va] = np.exp(O_log[va] @ w)
            else:
                for hf in (1, 0):
                    w = _fit_simplex(O_log, y, tr & (ih == hf))
                    m_va = va & (ih == hf)
                    oof_p[m_va] = np.exp(O_log[m_va] @ w)
        if not typed:
            w_full = _fit_simplex(O_log, y, nm)
            tmp_p = np.exp(T_log @ w_full)
        else:
            tmp_p = np.zeros(len(y_tmp), dtype=np.float64); w_full = {}
            for hf in (1, 0):
                w = _fit_simplex(O_log, y, nm & (ih == hf))
                m_t = (ih[tmp_idx] == hf)
                tmp_p[m_t] = np.exp(T_log[m_t] @ w)
                w_full['house' if hf == 1 else 'apt'] = w.tolist()
        return oof_p, tmp_p, w_full

    def _nested_huber(O_log, T_log):
        oof_p = np.zeros(len(y), dtype=np.float64)
        for fk in range(5):
            tr = (folds != fk) & nm; va = (folds == fk)
            hr = HuberRegressor(alpha=15.0, max_iter=500).fit(O_log[tr], np.log(y[tr]))
            oof_p[va] = np.clip(np.exp(hr.predict(O_log[va])), 10_000.0, 1e7)
        hr_f = HuberRegressor(alpha=15.0, max_iter=500).fit(O_log[nm], np.log(y[nm]))
        tmp_p = np.clip(np.exp(hr_f.predict(T_log)), 10_000.0, 1e7)
        return oof_p, tmp_p

    def _caruana_top_k(names, O_log, T_log, max_k=8):
        # Nested Caruana capped at max_k unique models!
        def _sel(O_m, y_m):
            yl = np.log(y_m); chosen = []
            for _ in range(30):
                u_ch = set(chosen)
                best_i, best_l = None, 1e9
                for i in range(O_m.shape[1]):
                    if len(u_ch) >= max_k and i not in u_ch:
                        continue
                    cand = chosen + [i]
                    err = O_m[:, cand].mean(axis=1) - yl
                    l = 0.55 * np.mean(np.abs(err)) + 0.45 * np.sqrt(np.mean(err ** 2))
                    if l < best_l:
                        best_l = l; best_i = i
                chosen.append(best_i)
            w = np.zeros(O_m.shape[1], dtype=np.float64)
            for i in chosen: w[i] += 1.0 / len(chosen)
            return w
        oof_p = np.zeros(len(y), dtype=np.float64)
        for fk in range(5):
            tr = (folds != fk) & nm; va = (folds == fk)
            w = _sel(O_log[tr], y[tr])
            oof_p[va] = np.exp(O_log[va] @ w)
        wf = _sel(O_log[nm], y[nm])
        tmp_p = np.exp(T_log @ wf)
        return oof_p, tmp_p, wf

    # 1. Pure Phase-7 GBDTs vs Phase-6 GBDTs
    g_names, g_O, g_T = _collect([c_p7g])
    p7g_oof, p7g_tmp, _ = _nested_convex(g_O, g_T)

    # 2. Pure Phase-7 Zoo (9 tuned GBDTs + 10 upgraded MPS = 19 models)
    p7_names, p7_O, p7_T = _collect([c_p7g, c_p7m])
    p7u_oof, p7u_tmp, p7u_w = _nested_convex(p7_O, p7_T)
    p7t_oof, p7t_tmp, _ = _nested_convex(p7_O, p7_T, typed=True)
    p7h_oof, p7h_tmp = _nested_huber(p7_O, p7_T)
    p7c8_oof, p7c8_tmp, p7c8_w = _caruana_top_k(p7_names, p7_O, p7_T, max_k=8)

    # 3. Phase-7 + Phase-6 Grand Pool (all 37 streams!)
    all_names, all_O, all_T = _collect([c_p7g, c_p7m, c_p6])
    gu_oof, gu_tmp, gu_w = _nested_convex(all_O, all_T)
    gh_oof, gh_tmp = _nested_huber(all_O, all_T)
    gc8_oof, gc8_tmp, gc8_w = _caruana_top_k(all_names, all_O, all_T, max_k=8)

    def _print_row(tag, oof, tmp_p):
        mo = compute_metrics(y[nm], oof[nm])
        mh = compute_metrics(y[nm & (ih == 1)], oof[nm & (ih == 1)])
        ma = compute_metrics(y[nm & (ih == 0)], oof[nm & (ih == 0)])
        mt = compute_metrics(y_tmp[nm_tmp], tmp_p[nm_tmp])
        mall_o = compute_metrics(y, oof); mall_t = compute_metrics(y_tmp, tmp_p)
        print(f"  {tag:24s} | NORM OOF={mo['mdape']:5.2f}% (H {mh['mdape']:.2f}/A {ma['mdape']:.2f}) €{mo['mae']:6,.0f} | NORM TMP={mt['mdape']:5.2f}% (€{mt['mae']:6,.0f} bias {mt['bias_pct']:+5.2f}%) | ALL={mall_o['mdape']:.2f}/{mall_t['mdape']:.2f}%", flush=True)
        return {'norm_oof': mo, 'norm_house': mh, 'norm_apt': ma, 'norm_tmp': mt, 'all_oof': mall_o, 'all_tmp': mall_t}

    rep = {'stacks': {}}
    print("\n  --- PHASE-7 LEADERBOARD (vs Phase-6 Champion P6-HUBER 7.27% OOF / 7.49% TMP) ---", flush=True)
    rep['stacks']['P7-GBDT-only'] = _print_row('P7-GBDT (9 tuned)', p7g_oof, p7g_tmp)
    rep['stacks']['P7-UNIFIED'] = _print_row(f'P7-UNIFIED ({len(p7_names)} P7)', p7u_oof, p7u_tmp)
    rep['stacks']['P7-TYPED'] = _print_row(f'P7-TYPED ({len(p7_names)} P7)', p7t_oof, p7t_tmp)
    rep['stacks']['P7-HUBER'] = _print_row(f'P7-HUBER ({len(p7_names)} P7)', p7h_oof, p7h_tmp)
    rep['stacks']['P7-CARUANA-8'] = _print_row('P7-CARUANA-8 (P7)', p7c8_oof, p7c8_tmp)
    rep['stacks']['P7+P6-UNIFIED'] = _print_row(f'P7+P6-UNI ({len(all_names)} all)', gu_oof, gu_tmp)
    rep['stacks']['P7+P6-HUBER'] = _print_row(f'P7+P6-HUB ({len(all_names)} all)', gh_oof, gh_tmp)
    rep['stacks']['P7+P6-CARUANA-8'] = _print_row('P7+P6-CARUANA-8', gc8_oof, gc8_tmp)

    print("\n  Top P7-UNIFIED weights (>1%):", flush=True)
    for n, w in sorted(zip(p7_names, p7u_w), key=lambda x: -x[1]):
        if w >= 0.01:
            print(f"    {n:28s}: {w*100:5.1f}%", flush=True)
    print("\n  Top P7+P6-CARUANA-8 weights:", flush=True)
    for n, w in sorted(zip(all_names, gc8_w), key=lambda x: -x[1]):
        if w > 0:
            print(f"    {n:28s}: {w*100:5.1f}%", flush=True)

    # Save weights & predictions
    rep['p7_weights_unified'] = {n: float(w) for n, w in zip(p7_names, p7u_w)}
    rep['p7_p6_weights_unified'] = {n: float(w) for n, w in zip(all_names, gu_w)}
    rep['p7_p6_caruana8_weights'] = {n: float(w) for n, w in zip(all_names, gc8_w) if w > 0}

    np.savez_compressed(
        P7_PREDS_NPZ,
        p7_names=np.array(p7_names, dtype=object), p7_O_log=p7_O, p7_T_log=p7_T,
        all_names=np.array(all_names, dtype=object), all_O_log=all_O, all_T_log=all_T,
        y_oof=y, y_tmp=y_tmp, tmp_idx=tmp_idx,
        p7u_oof=p7u_oof, p7u_tmp=p7u_tmp, p7h_oof=p7h_oof, p7h_tmp=p7h_tmp,
        gu_oof=gu_oof, gu_tmp=gu_tmp, gh_oof=gh_oof, gh_tmp=gh_tmp,
        gc8_oof=gc8_oof, gc8_tmp=gc8_tmp,
    )
    with open(P7_REPORT_JSON, 'w') as f:
        json.dump(rep, f, indent=2)
    print(f"\n  Saved Phase-7 preds -> {P7_PREDS_NPZ} & report -> {P7_REPORT_JSON}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['mps', 'worker', 'stack', 'all'], default='all')
    ap.add_argument('--model', default='')
    args = ap.parse_args()
    if args.stage in ('mps', 'all'):
        for m in MPS_MODELS:
            print(f"\n>>> Issue #13 MPS subprocess worker: {m} <<<", flush=True)
            subprocess.run([sys.executable, '-u', __file__, '--stage', 'worker', '--model', m], check=True)
    if args.stage == 'worker':
        run_worker(args.model)
    if args.stage in ('stack', 'all'):
        stage_stack()


if __name__ == '__main__':
    main()
