"""Issue #14: Learning Curves (Sample Size & Temporal Staleness) + Caruana Ensemble Pruning Curve.

Runs pure CPU (~3-4 min) right after Phase-6 completes to answer three strategic Phase-7 questions
strictly on **Normal Ghent Homes (€150k–€800k)** under **Cluster-Clean CV (`fold_cluster`)** and
**Cluster-Clean 2026 Temporal Holdout**:

  1. Sample-Size Learning Curve (`10%, 20%, 40%, 60%, 80%, 100%` of training clusters):
     - Overall Normal, Apartments, Houses, Historic Core, Fixer-Uppers
     - WITH Vision vs WITHOUT Vision (checks if 768/1152-d vision embeddings are still data-hungry!)
     - Power-law asymptote fit: `MdAPE(N) = E_floor + a * N^(-b)`
  2. Temporal Staleness / Lookback Window Curve (strictly on Cluster-Clean 2026 Temporal):
     - Training lookback: `Last 6m`, `Last 12m`, `Last 18m`, `All history` (using `0.5*CPI+0.5*HPI`)
     - Tests whether 2024 listings help (more volume) or hurt (concept drift) 2026 predictions!
  3. Caruana Forward Greedy Ensemble Pruning Curve (`K = 1, 2, ..., 18` models on `phase6_frontier_preds.npz`):
     - Strictly nested inside `fold_cluster` + evaluated on 2026 Temporal Holdout
     - Identifies minimal diverse elbow (`5-7` models) capturing >=99.5% of full-stack accuracy.

Writes:
  - `outputs/models/workbench/phase6_learning_and_pruning_curves.json`

Usage:
    .venv-api/bin/python -u src/analyze_learning_and_pruning_curves.py
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings

warnings.filterwarnings('ignore')

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import curve_fit

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'src'))

from phase3_features import compute_metrics

WORK_DIR = os.path.join(BASE_DIR, 'outputs', 'models', 'workbench')
P6_CACHE = os.path.join(WORK_DIR, 'phase6_splits_cache.joblib')
FEAT_SETS_JSON = os.path.join(WORK_DIR, 'phase6_feature_sets.json')
P6_PREDS_NPZ = os.path.join(WORK_DIR, 'phase6_frontier_preds.npz')
DATASET_PATH = os.path.join(BASE_DIR, 'outputs', 'dataset', 'training_dataset.parquet')
OUT_JSON = os.path.join(WORK_DIR, 'phase6_learning_and_pruning_curves.json')

HISTORIC_CORE = {
    'Binnenstad', 'Elisabethbegijnhof_Papegaai', 'Macharius_Heirnis', 'Sluizeken_Tolhuis_Ham',
}


def _prep_lgbm(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    o = df[cols].copy()
    for c in cols:
        if o[c].dtype == 'bool':
            o[c] = o[c].astype('int8')
        elif not pd.api.types.is_numeric_dtype(o[c]):
            o[c] = o[c].astype(str).fillna('Missing').astype('category')
        else:
            o[c] = o[c].astype('float32')
    return o


def _fit_lgbm_subset(tr_df, va_df, y_tr, def_tr, def_va, cids_tr, cols, sub_mask, seed=42):
    Xtr = _prep_lgbm(tr_df.iloc[sub_mask].reset_index(drop=True), cols)
    Xva = _prep_lgbm(va_df, cols)
    yt = np.log(y_tr[sub_mask] / def_tr[sub_mask])
    c_sub = cids_tr[sub_mask]
    u_c = np.unique(c_sub)
    n_es = max(5, int(len(u_c) * 0.12))
    es_c = set(np.random.default_rng(seed).choice(u_c, size=min(n_es, len(u_c) - 1), replace=False))
    es_m = np.isin(c_sub, list(es_c))
    if es_m.sum() < 20 or (~es_m).sum() < 50:
        es_m = np.zeros(len(yt), dtype=bool)
        es_m[:max(10, len(yt) // 10)] = True

    m = lgb.LGBMRegressor(
        objective='huber', alpha=0.15, n_estimators=2200, learning_rate=0.028,
        num_leaves=47, min_child_samples=max(10, min(35, int(len(yt) * 0.004))),
        subsample=0.72, subsample_freq=1, colsample_bytree=0.40,
        reg_alpha=0.8, reg_lambda=2.2, random_state=seed, n_jobs=3, verbose=-1,
    )
    m.fit(
        Xtr.iloc[~es_m], yt[~es_m],
        eval_set=[(Xtr.iloc[es_m], yt[es_m])],
        callbacks=[lgb.early_stopping(75, verbose=False)],
    )
    pred_va = np.clip(np.exp(np.clip(m.predict(Xva), -20, 20)) * def_va, 10_000, 1e7)
    pred_tr = np.clip(np.exp(np.clip(m.predict(Xtr.iloc[~es_m]), -20, 20)) * def_tr[sub_mask][~es_m], 10_000, 1e7)
    return pred_va, pred_tr, y_tr[sub_mask][~es_m]


def run_learning_curves(cache, fsets, ds_meta):
    print("\n=== PART 1: SAMPLE-SIZE LEARNING CURVES (10% -> 100% of Training Clusters) ===", flush=True)
    y = cache['y_price']
    nm = cache['normal_mask']
    ih = cache['is_house']
    bundles = cache['bundles']
    oof_bundles = [b for b in bundles if not str(b['id']).startswith('temporal')]
    tmp_b = [b for b in bundles if str(b['id']).startswith('temporal')][0]

    gbdt_vis = [c for c in fsets['gbdt_feats'] if c in oof_bundles[0]['tr_df'].columns]
    gbdt_novis = [c for c in gbdt_vis if not c.startswith('vis_')]

    # Slice masks on full 14,599 dataset
    is_hist = ds_meta['neighbourhood'].isin(HISTORIC_CORE).to_numpy(bool)
    is_fixer = ds_meta['building_condition'].isin({'TO_RENOVATE', 'TO_RESTORE', 'TO_BE_DONE_UP'}).to_numpy(bool)

    fracs = [0.10, 0.20, 0.40, 0.60, 0.80, 1.00]
    curves = {'with_vision': [], 'without_vision': []}

    for mode_name, cols in [('with_vision', gbdt_vis), ('without_vision', gbdt_novis)]:
        print(f"\n  --- Mode: {mode_name} ({len(cols)} cols) ---", flush=True)
        for frac in fracs:
            t0 = time.perf_counter()

            def _run_sp(b):
                cids = b['tr_cids']
                u_c = np.unique(cids)
                if frac >= 0.999:
                    sub_m = np.ones(len(cids), dtype=bool)
                else:
                    n_keep = max(50, int(round(len(u_c) * frac)))
                    keep_c = set(np.random.default_rng(100 + int(frac * 100)).choice(u_c, size=n_keep, replace=False))
                    sub_m = np.isin(cids, list(keep_c))
                pva, ptr, ytr_fit = _fit_lgbm_subset(
                    b['tr_df'], b['va_df'], y[b['tr_idx']],
                    b['tr_deflator'], b['va_deflator'], cids, cols, sub_m,
                )
                tr_mdape = compute_metrics(ytr_fit, ptr)['mdape']
                return b['id'], b['va_idx'], pva, int(sub_m.sum()), int(len(np.unique(cids[sub_m]))), tr_mdape

            res = Parallel(n_jobs=6, backend='loky')(delayed(_run_sp)(b) for b in bundles)
            oof_p = np.zeros(len(y), dtype=np.float64)
            tmp_p = None
            n_rows_oof, n_cl_oof, tr_mdapes = [], [], []
            for sid, vidx, pva, nr, nc, tr_m in res:
                if str(sid).startswith('temporal'):
                    tmp_p = pva
                else:
                    oof_p[vidx] = pva
                    n_rows_oof.append(nr); n_cl_oof.append(nc); tr_mdapes.append(tr_m)

            ti = tmp_b['va_idx']
            m_norm = compute_metrics(y[nm], oof_p[nm])
            m_h = compute_metrics(y[nm & (ih == 1)], oof_p[nm & (ih == 1)])
            m_a = compute_metrics(y[nm & (ih == 0)], oof_p[nm & (ih == 0)])
            m_hc = compute_metrics(y[nm & is_hist], oof_p[nm & is_hist])
            m_fx = compute_metrics(y[nm & is_fixer], oof_p[nm & is_fixer])
            m_tmp = compute_metrics(y[ti][nm[ti]], tmp_p[nm[ti]])
            rec = {
                'frac': frac,
                'avg_train_rows': float(np.mean(n_rows_oof)),
                'avg_train_clusters': float(np.mean(n_cl_oof)),
                'train_mdape': float(np.mean(tr_mdapes)),
                'norm_oof_mdape': m_norm['mdape'],
                'norm_oof_mae': m_norm['mae'],
                'norm_houses_mdape': m_h['mdape'],
                'norm_apts_mdape': m_a['mdape'],
                'historic_core_mdape': m_hc['mdape'],
                'fixer_mdape': m_fx['mdape'],
                'norm_tmp_mdape': m_tmp['mdape'],
                'norm_tmp_bias': m_tmp['bias_pct'],
                'seconds': round(time.perf_counter() - t0, 1),
            }
            curves[mode_name].append(rec)
            print(
                f"    frac={int(frac*100):3d}% (tr~{rec['avg_train_rows']:5.0f} rows / {rec['avg_train_clusters']:4.0f} cl) | "
                f"Tr={rec['train_mdape']:4.2f}% | NORM OOF={rec['norm_oof_mdape']:5.2f}% "
                f"(H {rec['norm_houses_mdape']:5.2f}% A {rec['norm_apts_mdape']:5.2f}% HC {rec['historic_core_mdape']:5.2f}% FX {rec['fixer_mdape']:5.2f}%) "
                f"| TMP={rec['norm_tmp_mdape']:5.2f}% | {rec['seconds']:.1f}s",
                flush=True,
            )

    # Power-law asymptote fit: MdAPE(N) = E_floor + a * N^(-b)
    fits = {}
    for mode_name in ('with_vision', 'without_vision'):
        ns = np.array([r['avg_train_clusters'] for r in curves[mode_name]], dtype=float)
        ys = np.array([r['norm_oof_mdape'] for r in curves[mode_name]], dtype=float)
        try:
            popt, _ = curve_fit(lambda n, e, a, b: e + a * np.power(n, -b), ns, ys, p0=[6.0, 40.0, 0.4], bounds=([3.0, 0.1, 0.05], [10.0, 500.0, 2.0]), maxfev=10000)
            e_fl, a_c, b_c = [float(x) for x in popt]
            pred_2x = float(e_fl + a_c * np.power(ns[-1] * 2.0, -b_c))
            fits[mode_name] = {'E_floor_pct': round(e_fl, 3), 'a': round(a_c, 3), 'b': round(b_c, 3), 'proj_2x_data_mdape': round(pred_2x, 3), 'gain_at_2x_pp': round(pred_2x - ys[-1], 3)}
            print(f"  Power-law fit [{mode_name:14s}]: E_floor={e_fl:.2f}% | current={ys[-1]:.2f}% -> projected @2x data={pred_2x:.2f}% ({pred_2x - ys[-1]:+.2f} pp)", flush=True)
        except Exception as exc:
            fits[mode_name] = {'error': str(exc)}
    return {'curves': curves, 'power_law_fits': fits}


def run_temporal_staleness(cache, fsets, ds_meta):
    print("\n=== PART 2: TEMPORAL STALENESS / LOOKBACK WINDOW CURVE (Cluster-Clean 2026 Holdout) ===", flush=True)
    y = cache['y_price']
    nm = cache['normal_mask']
    ih = cache['is_house']
    tmp_b = [b for b in cache['bundles'] if str(b['id']).startswith('temporal')][0]
    gbdt_vis = [c for c in fsets['gbdt_feats'] if c in tmp_b['tr_df'].columns]

    tr_idx = tmp_b['tr_idx']
    va_idx = tmp_b['va_idx']
    # Build approximate publication month index (year * 12 + month) relative to end of 2025 (2025*12 + 12)
    pub_y = ds_meta['publication_year'].to_numpy(float)[tr_idx]
    pub_m = ds_meta['publication_month'].to_numpy(float)[tr_idx]
    months_before_2026 = (2025 * 12 + 12) - (pub_y * 12 + np.nan_to_num(pub_m, nan=6.0))

    windows = [
        ('Last 6m (Jul-Dec 2025)',    months_before_2026 <= 6),
        ('Last 12m (Jan-Dec 2025)',   months_before_2026 <= 12),
        ('Last 18m (Jul 2024-Dec 25)', months_before_2026 <= 18),
        ('All history (<=2025)',      np.ones(len(tr_idx), dtype=bool)),
    ]

    results = []
    for label, wmask in windows:
        t0 = time.perf_counter()
        pva, _, _ = _fit_lgbm_subset(
            tmp_b['tr_df'], tmp_b['va_df'], y[tr_idx],
            tmp_b['tr_deflator'], tmp_b['va_deflator'], tmp_b['tr_cids'],
            gbdt_vis, wmask, seed=42,
        )
        m_n = compute_metrics(y[va_idx][nm[va_idx]], pva[nm[va_idx]])
        m_h = compute_metrics(y[va_idx][nm[va_idx] & (ih[va_idx] == 1)], pva[nm[va_idx] & (ih[va_idx] == 1)])
        m_a = compute_metrics(y[va_idx][nm[va_idx] & (ih[va_idx] == 0)], pva[nm[va_idx] & (ih[va_idx] == 0)])
        m_all = compute_metrics(y[va_idx], pva)
        rec = {
            'window': label,
            'train_rows': int(wmask.sum()),
            'train_clusters': int(len(np.unique(tmp_b['tr_cids'][wmask]))),
            'norm_tmp_mdape': m_n['mdape'],
            'norm_tmp_mae': m_n['mae'],
            'norm_houses_mdape': m_h['mdape'],
            'norm_apts_mdape': m_a['mdape'],
            'all_tmp_mdape': m_all['mdape'],
            'norm_tmp_bias': m_n['bias_pct'],
            'seconds': round(time.perf_counter() - t0, 1),
        }
        results.append(rec)
        print(
            f"  {label:28s} (n={rec['train_rows']:4d} / {rec['train_clusters']:4d} cl) | "
            f"NORM TMP={rec['norm_tmp_mdape']:5.2f}% (H {rec['norm_houses_mdape']:5.2f}% A {rec['norm_apts_mdape']:5.2f}%) "
            f"MAE €{rec['norm_tmp_mae']:6,.0f} bias={rec['norm_tmp_bias']:+5.2f}% | ALL={rec['all_tmp_mdape']:5.2f}%",
            flush=True,
        )
    return results


def run_caruana_pruning(cache):
    print("\n=== PART 3: CARUANA FORWARD GREEDY ENSEMBLE PRUNING CURVE (K = 1..18 Models) ===", flush=True)
    z = np.load(P6_PREDS_NPZ, allow_pickle=True)
    names = list(z['names'])
    O_log = z['O_log']
    T_log = z['T_log']
    y = z['y_oof']
    ti = z['tmp_idx']
    folds = cache['folds']
    nm = cache['normal_mask']
    tnm = nm[ti]
    ly = np.log(y)

    M = len(names)
    # 1. Rank unique models by greedy forward selection with replacement (nested inside fold_cluster!)
    def _loss(pred_l, true_l):
        e = pred_l - true_l
        return 0.55 * float(np.mean(np.abs(e))) + 0.45 * float(np.sqrt(np.mean(e * e)))

    def _caruana_weights(X_tr, y_tr, n_rounds=30, max_unique=None):
        counts = np.zeros(X_tr.shape[1], dtype=np.float64)
        cur_sum = np.zeros(len(y_tr), dtype=np.float64)
        for r in range(1, n_rounds + 1):
            best_j, best_l = -1, 1e9
            active = set(np.where(counts > 0)[0])
            for j in range(X_tr.shape[1]):
                if max_unique is not None and len(active) >= max_unique and j not in active:
                    continue
                cand = (cur_sum + X_tr[:, j]) / r
                lv = _loss(cand, y_tr)
                if lv < best_l:
                    best_l, best_j = lv, j
            counts[best_j] += 1.0
            cur_sum += X_tr[:, best_j]
        return counts / counts.sum()

    k_steps = [1, 2, 3, 4, 5, 6, 8, 10, 12, M]
    curve = []
    for k_max in k_steps:
        oof_l = np.zeros(len(y))
        for f in range(5):
            tr_m = (folds != f) & nm
            va_m = folds == f
            w_f = _caruana_weights(O_log[tr_m], ly[tr_m], n_rounds=30, max_unique=k_max)
            oof_l[va_m] = O_log[va_m] @ w_f
        w_full = _caruana_weights(O_log[nm], ly[nm], n_rounds=30, max_unique=k_max)
        tmp_l = T_log @ w_full
        p_oof = np.clip(np.exp(oof_l), 10_000, 1e7)
        p_tmp = np.clip(np.exp(tmp_l), 10_000, 1e7)
        mo = compute_metrics(y[nm], p_oof[nm])
        mt = compute_metrics(y[ti][tnm], p_tmp[tnm])
        selected = {names[j]: round(float(w_full[j]), 3) for j in np.argsort(-w_full) if w_full[j] > 0}
        rec = {
            'max_models': int(k_max),
            'actual_unique': len(selected),
            'norm_oof_mdape': mo['mdape'],
            'norm_oof_mae': mo['mae'],
            'norm_tmp_mdape': mt['mdape'],
            'norm_tmp_mae': mt['mae'],
            'norm_tmp_bias': mt['bias_pct'],
            'selected_weights': selected,
        }
        curve.append(rec)
        top_str = ', '.join(f"{k}:{v*100:.0f}%" for k, v in list(selected.items())[:5])
        print(
            f"  K<={k_max:2d} (used {len(selected):2d}) | NORM OOF={mo['mdape']:5.2f}% (MAE €{mo['mae']:6,.0f}) "
            f"| NORM TMP={mt['mdape']:5.2f}% (MAE €{mt['mae']:6,.0f} bias {mt['bias_pct']:+5.2f}%) | {top_str}",
            flush=True,
        )
    return curve


def main():
    t0 = time.perf_counter()
    print("=== ISSUE #14: LEARNING CURVES, TEMPORAL STALENESS & CARUANA ENSEMBLE PRUNING ===", flush=True)
    cache = joblib.load(P6_CACHE)
    fsets = json.load(open(FEAT_SETS_JSON))
    ds_meta = pd.read_parquet(DATASET_PATH, columns=['neighbourhood', 'building_condition', 'publication_year', 'publication_month'])

    out = {}
    out['learning_curves'] = run_learning_curves(cache, fsets, ds_meta)
    out['temporal_staleness'] = run_temporal_staleness(cache, fsets, ds_meta)
    if os.path.exists(P6_PREDS_NPZ):
        out['caruana_ensemble_pruning'] = run_caruana_pruning(cache)
    else:
        print("  (Skipping Part 3: phase6_frontier_preds.npz not found yet)", flush=True)

    out['total_seconds'] = round(time.perf_counter() - t0, 1)
    with open(OUT_JSON, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n=== ISSUE #14 COMPLETE in {out['total_seconds']:.1f}s -> {OUT_JSON} ===", flush=True)


if __name__ == '__main__':
    main()
