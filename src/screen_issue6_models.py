"""Issue #6 (Fast 2-Split Dev Screen) — Screen `MLP-PLR`, `RealMLP (psqm)`, and `TabPFN V2` on Normal Ghent Homes!

Uses our fast 2-split protocol (`fold_0` + strictly cluster-clean `temporal_2026`) on MPS so we know
in ~5 minutes which of the remaining model families deserve a spot in our unified Phase-6 ensemble!

Candidates screened (all evaluated on Normal Ghent Homes €150k–€800k AND ALL homes):
  1. `RealMLP_psqm_180` (2 seeds, `log_real_psqm`, d=180)
  2. `MLP_PLR_psqm_180` (2 seeds, `log_real_psqm`, d=180 — Periodic Numerical Embeddings MLP, now unlocked via `skorch`!)
  3. `TabPFN_V2_price_95` (Apache-2.0, `n_estimators=16`, `SUBSAMPLE_SAMPLES=8000`, d=95, `log_real_price`)
  4. `TabPFN_V2_psqm_95`  (Apache-2.0, `n_estimators=16`, `SUBSAMPLE_SAMPLES=8000`, d=95, `log_real_psqm`)

Uses the NEW winning `0.5 * CPI + 0.5 * HPI` deflator (Issue #9)!
NOTE: Do NOT `import lightgbm` before `import torch` on macOS ARM64 (`libomp` segfault).

Usage:
    .venv-api/bin/python -u src/screen_issue6_models.py
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings

warnings.filterwarnings('ignore')

# Import torch FIRST before any lightgbm/sklearn OpenMP libs!
import torch
import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'src'))

from phase3_features import compute_metrics

DATASET_PATH = os.path.join(BASE_DIR, 'outputs', 'dataset', 'training_dataset.parquet')
CLUSTER_PATH = os.path.join(BASE_DIR, 'outputs', 'dataset', 'cluster_folds.parquet')
P4_CACHE = os.path.join(BASE_DIR, 'outputs', 'models', 'workbench', 'phase4_splits_cache.joblib')
P5_ADDON = os.path.join(BASE_DIR, 'outputs', 'models', 'workbench', 'phase5_addon_features.joblib')
P4_REP = os.path.join(BASE_DIR, 'outputs', 'models', 'workbench', 'phase4_report.json')
P5_T1_PREDS = os.path.join(BASE_DIR, 'outputs', 'models', 'workbench', 'phase5_tabicl_preds.npz')
OUT_REPORT = os.path.join(BASE_DIR, 'outputs', 'models', 'workbench', 'issue6_fast_screen.json')


def _prep_nn_dfs(tr_df: pd.DataFrame, va_df: pd.DataFrame, feat_cols: list[str], max_cat: int = 40):
    Xtr = tr_df[feat_cols].copy()
    Xva = va_df[feat_cols].copy()
    cat_cols = []
    num_cols = []
    for c in feat_cols:
        if Xtr[c].dtype == 'bool':
            Xtr[c] = Xtr[c].astype('float32')
            Xva[c] = Xva[c].astype('float32')
            num_cols.append(c)
        elif not pd.api.types.is_numeric_dtype(Xtr[c]):
            vc = Xtr[c].astype(str).fillna('Missing').value_counts()
            keep = set(vc[vc >= 5].head(max_cat).index)
            Xtr[c] = Xtr[c].astype(str).fillna('Missing').where(lambda s: s.isin(keep), '__RARE__').astype('category')
            Xva[c] = Xva[c].astype(str).fillna('Missing').where(lambda s: s.isin(keep), '__RARE__').astype('category')
            cat_cols.append(c)
        else:
            tr_v = pd.to_numeric(Xtr[c], errors='coerce').replace([np.inf, -np.inf], np.nan)
            va_v = pd.to_numeric(Xva[c], errors='coerce').replace([np.inf, -np.inf], np.nan)
            med = float(tr_v.median()) if tr_v.notna().any() else 0.0
            Xtr[c] = tr_v.fillna(med).astype('float32')
            Xva[c] = va_v.fillna(med).astype('float32')
            num_cols.append(c)
    return Xtr, Xva, cat_cols


def _prep_tabpfn_dfs(tr_df: pd.DataFrame, va_df: pd.DataFrame, feat_cols: list[str], max_cat: int = 25):
    Xtr = tr_df[feat_cols].copy()
    Xva = va_df[feat_cols].copy()
    for c in feat_cols:
        if Xtr[c].dtype == 'bool':
            Xtr[c] = Xtr[c].astype('float32')
            Xva[c] = Xva[c].astype('float32')
        elif not pd.api.types.is_numeric_dtype(Xtr[c]):
            vc = Xtr[c].astype(str).fillna('Missing').value_counts()
            keep = list(vc.head(max_cat).index)
            cmap = {k: i for i, k in enumerate(keep)}
            Xtr[c] = Xtr[c].astype(str).fillna('Missing').map(cmap).fillna(-1).astype('float32')
            Xva[c] = Xva[c].astype(str).fillna('Missing').map(cmap).fillna(-1).astype('float32')
        else:
            Xtr[c] = pd.to_numeric(Xtr[c], errors='coerce').replace([np.inf, -np.inf], np.nan).astype('float32')
            Xva[c] = pd.to_numeric(Xva[c], errors='coerce').replace([np.inf, -np.inf], np.nan).astype('float32')
    return Xtr, Xva


def main():
    t0 = time.perf_counter()
    tr = pd.read_parquet(DATASET_PATH)
    cl = pd.read_parquet(CLUSTER_PATH)
    p4_cache = joblib.load(P4_CACHE)
    p5_addon = joblib.load(P5_ADDON)
    p4_rep = json.load(open(P4_REP))
    p5_t1 = np.load(P5_T1_PREDS, allow_pickle=True)

    y_price = tr['price'].to_numpy(float)
    y_area = tr['livable_area'].to_numpy(float)

    # Winning Issue #9 deflator: 0.5 * CPI + 0.5 * HPI (relative to 2025 median)
    cpi_idx = p4_cache['cpi_idx']
    hpi_raw = tr['belgian_hpi_index'].to_numpy(float)
    hpi_idx = hpi_raw / np.nanmedian(hpi_raw[tr['publication_year'] == 2025])
    deflator = 0.5 * cpi_idx + 0.5 * hpi_idx

    norm_all = (
        (y_price >= 150_000) & (y_price <= 800_000)
        & (~tr['property_subtype'].isin(['APARTMENT_BLOCK', 'MANSION', 'SERVICE_FLAT']).to_numpy())
    )
    cids = cl['cluster_id'].to_numpy(int)

    # Grab fold_0 and temporal_2026 bundles
    f0_b = [b for b in p4_cache['bundles'] if b['id'] == 'fold_0'][0]
    tm_b = [b for b in p4_cache['bundles'] if str(b['id']).startswith('temporal')][0]

    p5_prio = ['txt_tab_plus_text_psqm', 'txt_p5_super_tabresid', 'txt_mlp_psqm', 'txt_mlp_price',
               'txt_mlp_tabresid', 'txt_tfidf_tabresid_a6', 'txt_dense_both_tabresid', 'tab_anchor_log_psqm',
               'txt_qwen3_a5_psqm', 'txt_p5_super_psqm', 'freq_agency_norm']
    drop_hi = {'statbel_sector', 'city', 'immo_office_name'}

    all_cols_ranked = list(dict.fromkeys(p5_prio + p4_rep['pruned_features']))

    def _build_split(b):
        bid = b['id']
        tr_idx, va_idx = b['tr_idx'], b['va_idx']
        if str(bid).startswith('temporal'):
            va_cids = set(cids[va_idx])
            km = np.array([c not in va_cids for c in cids[tr_idx]], dtype=bool)
        else:
            km = np.ones(len(tr_idx), dtype=bool)
        tr_df = pd.concat([b['tr_df'].reset_index(drop=True), p5_addon[bid]['tr_add'].reset_index(drop=True)], axis=1).loc[km].reset_index(drop=True)
        va_df = pd.concat([b['va_df'].reset_index(drop=True), p5_addon[bid]['va_add'].reset_index(drop=True)], axis=1).reset_index(drop=True)
        eff_tr = tr_idx[km]
        return {'id': bid, 'tr_idx': eff_tr, 'va_idx': va_idx, 'tr_df': tr_df, 'va_df': va_df}

    sp_f0 = _build_split(f0_b)
    sp_tm = _build_split(tm_b)

    feats_180 = [c for c in all_cols_ranked if c in sp_f0['tr_df'].columns][:180]
    feats_95 = [c for c in all_cols_ranked if c in sp_f0['tr_df'].columns and c not in drop_hi][:95]

    # Reference P5-T1 predictions on exact same rows
    p5_f0 = p5_t1['ens_oof_u'][sp_f0['va_idx']]
    p5_tm = p5_t1['ens_tmp_u']
    f0_norm = norm_all[sp_f0['va_idx']]
    tm_norm = norm_all[sp_tm['va_idx']]
    y_f0 = y_price[sp_f0['va_idx']]
    y_tm = y_price[sp_tm['va_idx']]

    ref_f0_n = compute_metrics(y_f0[f0_norm], p5_f0[f0_norm])
    ref_tm_n = compute_metrics(y_tm[tm_norm], p5_tm[tm_norm])
    ref_f0_a = compute_metrics(y_f0, p5_f0)
    ref_tm_a = compute_metrics(y_tm, p5_tm)

    print("=== ISSUE #6 FAST 2-SPLIT SCREEN (Normal Ghent Homes €150k-€800k | 0.5*CPI+0.5*HPI) ===", flush=True)
    print(f"  REF P5-T1 (23 models): f0 NORM={ref_f0_n['mdape']:.2f}% (ALL={ref_f0_a['mdape']:.2f}%) | tm NORM={ref_tm_n['mdape']:.2f}% (ALL={ref_tm_a['mdape']:.2f}%, bias={ref_tm_n['bias_pct']:+.2f}%)\n", flush=True)

    results = {
        'ref_p5_t1': {'f0_norm': ref_f0_n, 'tm_norm': ref_tm_n, 'f0_all': ref_f0_a, 'tm_all': ref_tm_a}
    }

    def _report(name: str, pr_f0: np.ndarray, pr_tm: np.ndarray, dt: float):
        m_f0_n = compute_metrics(y_f0[f0_norm], pr_f0[f0_norm])
        m_tm_n = compute_metrics(y_tm[tm_norm], pr_tm[tm_norm])
        m_f0_a = compute_metrics(y_f0, pr_f0)
        m_tm_a = compute_metrics(y_tm, pr_tm)
        # 20% geometric blend into P5-T1!
        bl_f0 = np.exp(0.80 * np.log(p5_f0) + 0.20 * np.log(pr_f0))
        bl_tm = np.exp(0.80 * np.log(p5_tm) + 0.20 * np.log(pr_tm))
        mb_f0_n = compute_metrics(y_f0[f0_norm], bl_f0[f0_norm])
        mb_tm_n = compute_metrics(y_tm[tm_norm], bl_tm[tm_norm])
        print(
            f"  [{name:22s}] ({dt:5.1f}s) | STANDALONE f0={m_f0_n['mdape']:.2f}% (ALL {m_f0_a['mdape']:.2f}%) tm={m_tm_n['mdape']:.2f}% (bias {m_tm_n['bias_pct']:+.2f}%) "
            f"| +20% into P5-T1 -> f0={mb_f0_n['mdape']:.2f}% ({mb_f0_n['mdape']-ref_f0_n['mdape']:+.2f}p) tm={mb_tm_n['mdape']:.2f}% ({mb_tm_n['mdape']-ref_tm_n['mdape']:+.2f}p)",
            flush=True,
        )
        results[name] = {
            'seconds': dt,
            'standalone_f0_norm': m_f0_n, 'standalone_tm_norm': m_tm_n,
            'standalone_f0_all': m_f0_a, 'standalone_tm_all': m_tm_a,
            'blend20_f0_norm': mb_f0_n, 'blend20_tm_norm': mb_tm_n,
        }

    # 1. pytabkit RealMLP_TD_Regressor & MLP_PLR_D_Regressor (2 seeds each on log_real_psqm, d=180!)
    from pytabkit import MLP_PLR_D_Regressor, RealMLP_TD_Regressor

    def _run_pytabkit(kind: str, n_seeds: int = 2):
        ts = time.perf_counter()
        preds_out = {}
        for sp in (sp_f0, sp_tm):
            Xtr, Xva, _ = _prep_nn_dfs(sp['tr_df'], sp['va_df'], feats_180)
            y_tr = np.log((y_price[sp['tr_idx']] / deflator[sp['tr_idx']]) / y_area[sp['tr_idx']]).astype(np.float32)
            # Group-safe 12% val split for pytabkit early stopping
            u_cl = np.unique(cids[sp['tr_idx']])
            rng = np.random.default_rng(101)
            es_cl = set(rng.choice(u_cl, size=max(1, int(len(u_cl) * 0.12)), replace=False))
            val_idxs = np.where(np.isin(cids[sp['tr_idx']], list(es_cl)))[0]

            seed_logs = []
            for s in range(n_seeds):
                if kind == 'realmlp':
                    m = RealMLP_TD_Regressor(
                        device='mps', n_epochs=128, batch_size=256, val_metric_name='mae',
                        random_state=42 + s * 17, verbosity=0,
                    )
                else:
                    m = MLP_PLR_D_Regressor(
                        device='mps', max_epochs=100, batch_size=256, val_metric_name='mae',
                        random_state=42 + s * 17, verbosity=0,
                    )
                m.fit(Xtr, y_tr, val_idxs=val_idxs)
                seed_logs.append(np.asarray(m.predict(Xva), dtype=float))
                torch.mps.empty_cache()
            avg_log = np.mean(seed_logs, axis=0)
            preds_out[sp['id']] = np.clip(np.exp(avg_log) * y_area[sp['va_idx']] * deflator[sp['va_idx']], 10_000, 1e7)
        return preds_out['fold_0'], preds_out[sp_tm['id']], time.perf_counter() - ts

    try:
        pr_f0, pr_tm, dt = _run_pytabkit('realmlp', n_seeds=2)
        _report('RealMLP_psqm_180_2s', pr_f0, pr_tm, dt)
    except Exception as e:
        print(f"  [RealMLP FAILED]: {e}", flush=True)

    try:
        pr_f0, pr_tm, dt = _run_pytabkit('mlp_plr', n_seeds=2)
        _report('MLP_PLR_psqm_180_2s', pr_f0, pr_tm, dt)
    except Exception as e:
        print(f"  [MLP_PLR FAILED]: {e}", flush=True)

    # 2. TabPFN V2 (Apache-2.0, zero-auth!) on d=95 compact features — both log_real_price and log_real_psqm!
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion

    for target_mode in ('log_real_price', 'log_real_psqm'):
        ts = time.perf_counter()
        try:
            preds_out = {}
            for sp in (sp_f0, sp_tm):
                Xtr, Xva = _prep_tabpfn_dfs(sp['tr_df'], sp['va_df'], feats_95)
                if target_mode == 'log_real_price':
                    y_tr = np.log(y_price[sp['tr_idx']] / deflator[sp['tr_idx']]).astype(np.float32)
                else:
                    y_tr = np.log((y_price[sp['tr_idx']] / deflator[sp['tr_idx']]) / y_area[sp['tr_idx']]).astype(np.float32)

                m = TabPFNRegressor.create_default_for_version(
                    ModelVersion.V2,
                    device='mps',
                    n_estimators=8,
                    ignore_pretraining_limits=True,
                    inference_config={'SUBSAMPLE_SAMPLES': 7500},
                    random_state=42,
                )
                m.fit(Xtr, y_tr)
                p_log = np.asarray(m.predict(Xva), dtype=float)
                torch.mps.empty_cache()
                if target_mode == 'log_real_price':
                    preds_out[sp['id']] = np.clip(np.exp(p_log) * deflator[sp['va_idx']], 10_000, 1e7)
                else:
                    preds_out[sp['id']] = np.clip(np.exp(p_log) * y_area[sp['va_idx']] * deflator[sp['va_idx']], 10_000, 1e7)
            _report(f'TabPFN_V2_{target_mode.split("_")[-1]}_95', preds_out['fold_0'], preds_out[sp_tm['id']], time.perf_counter() - ts)
        except Exception as e:
            print(f"  [TabPFN {target_mode} FAILED]: {e}", flush=True)

    with open(OUT_REPORT, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nDone in {time.perf_counter()-t0:.1f}s -> {OUT_REPORT}", flush=True)


if __name__ == '__main__':
    main()
