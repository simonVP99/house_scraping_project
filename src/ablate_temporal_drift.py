"""Issue #9 — Fast Temporal Drift Ablation (Normal Ghent Homes €150k–€800k).

Tests on strictly **cluster-clean 2026 temporal holdout** (train <2026 with ZERO overlapping
`cluster_id` in 2026; test = all 2026 normal homes €150k–€800k, broken down by 2026 Q1–Q4!):
  1. Damped CPI elasticity: `deflator = cpi_idx ** beta` for `beta in {0.0, 0.4, 0.6, 0.75, 0.85, 1.0}`
  2. 6-month lagged CPI (`cpi_lag6`) & Eurostat Belgian HPI (`hpi_idx`)
  3. Recency-weighted training loss: `w(t) = 0.5 ** (age_months / half_life)` for `half_life in {inf, 24m, 15m, 9m}`
  4. Best deflator + best recency decay combined!

Fast CPU-only LightGBM Huber (~25s total). Zero GPU contention with vision fetch/encode.

Usage:
    .venv-api/bin/python -u src/ablate_temporal_drift.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import joblib
import lightgbm as lgb
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
MACRO_CSV = os.path.join(BASE_DIR, 'outputs', 'dataset', 'ecb_macro_monthly.csv')


def main():
    t0 = time.perf_counter()
    tr = pd.read_parquet(DATASET_PATH)
    cl = pd.read_parquet(CLUSTER_PATH)
    p4_cache = joblib.load(P4_CACHE)
    p5_addon = joblib.load(P5_ADDON)
    p4_rep = json.load(open(P4_REP))
    macro = pd.read_csv(MACRO_CSV)

    tmp_b = [b for b in p4_cache['bundles'] if str(b['id']).startswith('temporal')][0]
    bid = tmp_b['id']
    tr_idx, va_idx = tmp_b['tr_idx'], tmp_b['va_idx']

    # Strictly cluster-clean temporal split!
    cids = cl['cluster_id'].to_numpy(int)
    va_cids = set(cids[va_idx])
    keep_m = np.array([c not in va_cids for c in cids[tr_idx]], dtype=bool)
    eff_tr_idx = tr_idx[keep_m]

    # Build base feature frames
    p5_prio = ['txt_tab_plus_text_psqm', 'txt_p5_super_tabresid', 'txt_mlp_psqm', 'txt_mlp_price',
               'txt_mlp_tabresid', 'txt_tfidf_tabresid_a6', 'txt_dense_both_tabresid', 'tab_anchor_log_psqm',
               'txt_qwen3_a5_psqm', 'txt_p5_super_psqm', 'freq_agency_norm']
    tr_df = pd.concat([tmp_b['tr_df'].reset_index(drop=True), p5_addon[bid]['tr_add'].reset_index(drop=True)], axis=1)
    va_df = pd.concat([tmp_b['va_df'].reset_index(drop=True), p5_addon[bid]['va_add'].reset_index(drop=True)], axis=1)
    cols = [c for c in dict.fromkeys(p5_prio + p4_rep['pruned_features'][:250]) if c in tr_df.columns]

    def _prep(df):
        out = df[cols].copy()
        for c in out.columns:
            if out[c].dtype == 'bool':
                out[c] = out[c].astype('int8')
            elif not pd.api.types.is_numeric_dtype(out[c]):
                out[c] = out[c].astype(str).fillna('Missing').astype('category')
            else:
                out[c] = out[c].astype('float32')
        return out

    Xtr = _prep(tr_df.loc[keep_m].reset_index(drop=True))
    Xva = _prep(va_df.reset_index(drop=True))

    # Group-safe 12% ES mask
    u_cl = np.unique(cids[eff_tr_idx])
    rng = np.random.default_rng(42)
    es_cl = set(rng.choice(u_cl, size=max(1, int(len(u_cl) * 0.12)), replace=False))
    es_m = np.isin(cids[eff_tr_idx], list(es_cl))
    fit_m = ~es_m

    y_price = tr['price'].to_numpy(float)
    y_area = tr['livable_area'].to_numpy(float)
    cpi_idx = p4_cache['cpi_idx']

    # Build 6-month lagged CPI & HPI index relative to 2025 median
    macro = macro.sort_values('year_month').reset_index(drop=True)
    macro['cpi_lag6'] = macro['belgian_cpi_index'].shift(6).bfill()
    ym_str = (
        tr['publication_year'].fillna(2025).astype(int).astype(str)
        + '-' + tr['publication_month'].fillna(6).astype(int).astype(str).str.zfill(2)
    )
    cpi_lag6_map = dict(zip(macro['year_month'], macro['cpi_lag6']))
    cpi_lag6_raw = ym_str.map(cpi_lag6_map).fillna(tr['belgian_cpi_index']).to_numpy(float)
    cpi_lag6_idx = cpi_lag6_raw / np.nanmedian(cpi_lag6_raw[tr['publication_year'] == 2025])

    hpi_raw = tr['belgian_hpi_index'].to_numpy(float)
    hpi_idx = hpi_raw / np.nanmedian(hpi_raw[tr['publication_year'] == 2025])

    # Age in months relative to end of train period (2025-12)
    ym_num = tr['publication_year'].fillna(2025).to_numpy(float) * 12.0 + tr['publication_month'].fillna(6).to_numpy(float)
    tr_age_months = np.clip((2025 * 12 + 12) - ym_num[eff_tr_idx], 0.0, 60.0)

    # Normal Ghent homes (€150k–€800k, excl block/mansion/service_flat)
    norm_all = (
        (y_price >= 150_000) & (y_price <= 800_000)
        & (~tr['property_subtype'].isin(['APARTMENT_BLOCK', 'MANSION', 'SERVICE_FLAT']).to_numpy())
    )
    va_norm = norm_all[va_idx]
    va_y = y_price[va_idx]
    va_qtr = (((tr['publication_month'].iloc[va_idx].fillna(1).to_numpy(int) - 1) // 3) + 1)

    def eval_cfg(deflator_arr: np.ndarray, half_life_m: float | None):
        tr_def = deflator_arr[eff_tr_idx]
        va_def = deflator_arr[va_idx]
        y_tr = np.log((y_price[eff_tr_idx] / tr_def) / y_area[eff_tr_idx])
        w_tr = np.ones(len(eff_tr_idx), dtype=float) if half_life_m is None else (0.5 ** (tr_age_months / half_life_m))
        reg = lgb.LGBMRegressor(
            objective='huber', alpha=0.15, n_estimators=2000, learning_rate=0.03,
            num_leaves=47, min_child_samples=35, subsample=0.75, subsample_freq=1,
            colsample_bytree=0.45, reg_alpha=0.8, reg_lambda=2.5,
            random_state=42, n_jobs=6, verbose=-1,
        )
        reg.fit(
            Xtr.iloc[fit_m], y_tr[fit_m], sample_weight=w_tr[fit_m],
            eval_set=[(Xtr.iloc[es_m], y_tr[es_m])],
            callbacks=[lgb.early_stopping(70, verbose=False)],
        )
        pred = np.clip(np.exp(reg.predict(Xva)) * y_area[va_idx] * va_def, 10_000, 1e7)
        mn = compute_metrics(va_y[va_norm], pred[va_norm])
        ma = compute_metrics(va_y, pred)
        q_bias = []
        for q in (1, 2, 3, 4):
            qm = va_norm & (va_qtr == q)
            q_bias.append(float(np.median((pred[qm] / va_y[qm] - 1.0) * 100.0)))
        return mn, ma, q_bias

    configs = [
        ('0. No deflation (beta=0.0)',            np.ones_like(cpi_idx),             None),
        ('1. CPI^0.50 (half elasticity)',         cpi_idx ** 0.50,                   None),
        ('2. CPI^0.70 (damped housing)',          cpi_idx ** 0.70,                   None),
        ('3. CPI^0.85',                           cpi_idx ** 0.85,                   None),
        ('4. CPI^1.00 (CURRENT baseline)',        cpi_idx,                           None),
        ('5. 6m-lagged CPI (1.0)',                cpi_lag6_idx,                      None),
        ('6. Eurostat HPI (1.0)',                 hpi_idx,                           None),
        ('7. Blend 0.5*CPI_lag6 + 0.5*HPI',       0.5 * cpi_lag6_idx + 0.5 * hpi_idx, None),
        ('8. CPI^0.70 + Recency HL=24m',          cpi_idx ** 0.70,                   24.0),
        ('9. CPI^0.70 + Recency HL=15m',          cpi_idx ** 0.70,                   15.0),
        ('10. CPI^0.70 + Recency HL=9m',          cpi_idx ** 0.70,                   9.0),
        ('11. CPI_lag6 + Recency HL=15m',         cpi_lag6_idx,                      15.0),
        ('12. CPI^1.00 + Recency HL=15m',         cpi_idx,                           15.0),
    ]

    print(f"=== TEMPORAL DRIFT ABLATION (Normal Ghent Homes €150k-€800k, n={va_norm.sum():,} | Cluster-Clean 2026) ===")
    print(f"  {'Config':34s} | {'NORM MdAPE':>10s} | {'NORM MAE':>9s} | {'Bias':>6s} | {'Q1':>6s} | {'Q2':>6s} | {'Q3':>6s} | {'Q4':>6s} | {'ALL MdAPE':>9s}")
    print("  " + "-" * 112)
    for name, darr, hl in configs:
        mn, ma, qb = eval_cfg(darr, hl)
        print(
            f"  {name:34s} | {mn['mdape']:9.2f}% | €{mn['mae']:8,.0f} | {mn['bias_pct']:+5.2f}% | "
            f"{qb[0]:+5.2f}% | {qb[1]:+5.2f}% | {qb[2]:+5.2f}% | {qb[3]:+5.2f}% | {ma['mdape']:8.2f}%",
            flush=True,
        )
    print(f"\nDone in {time.perf_counter()-t0:.1f}s")


if __name__ == '__main__':
    main()
