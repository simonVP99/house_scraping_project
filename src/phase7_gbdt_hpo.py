"""Issue #11: log_real_price GBDT/Specialist Twins + Optuna HPO on Phase-6 Multimodal Cluster-Clean Features.

Phase-6 Stage-2 discovered a huge systematic pattern:
  - `log_real_price` beat `log_real_psqm` by `0.21-0.29 pp` in every GBDT family (CatBoost `7.58%` vs `7.87%`,
    LightGBM `7.66%` vs `7.87%`), yet we had NO `log_real_price` twins for `XGBoost`, `LightGBMXT`,
    `Specialist CatBoost`, or `Specialist XGBoost`!
  - Furthermore, CatBoost was completely untuned (`depth=7, l2_leaf_reg=5` defaults), and LightGBM/XGBoost
    reused Phase-3 params tuned on 210 non-vision features under the old leaky CV!

Design (strictly following our Golden Rules — fast 2-split dev tuning, zero overfitting, then 6-split promotion):
  1. `--stage hpo` (~8 min CPU):
     - Runs focused Optuna TPE search (20 trials CatBoost, 25 trials LightGBM, 20 trials XGBoost) on the
       **fast 2-split dev profile (`fold_0` + `temporal_2026`, Normal Ghent Homes)** optimizing
       `0.5 * fold_0_MdAPE + 0.5 * temporal_MdAPE` (so HPO cannot overfit `fold_0` at temporal's expense!).
     - Saves `outputs/models/workbench/phase7_gbdt_hpo.json`.
  2. `--stage train` (~10 min CPU parallel 6 splits × 3 threads):
     - Trains the full upgraded Phase-7 GBDT roster across ALL 6 cluster-clean splits (5 grouped OOF + 2026 temporal):
       1. `p7_cb_tuned_price_l2`    (Optuna-tuned CatBoost `log_real_price`, L2, with `agency_norm`)
       2. `p7_cb_tuned_price_mae`   (Optuna-tuned CatBoost `log_real_price`, MAE/L1 — direct MdAPE optimizer!)
       3. `p7_lgbm_tuned_price_hub` (Optuna-tuned LightGBM `log_real_price`, Huber)
       4. `p7_lgbmxt_price_hub`     (Missing twin! LightGBMXT `extra_trees=True`, `log_real_price`, Huber)
       5. `p7_xgb_tuned_price_l2`   (Missing twin! Optuna-tuned XGBoost `log_real_price`, L2)
       6. `p7_xgb_tuned_price_hub`  (Missing twin! Optuna-tuned XGBoost `log_real_price`, Pseudo-Huber!)
       7. `p7_spec_cb_price_l2`     (Missing twin! House/Apt Specialist CatBoost `log_real_price`)
       8. `p7_spec_xgb_price_l2`    (Missing twin! House/Apt Specialist XGBoost `log_real_price`)
       9. `p7_spec_lgbm_price_hub`  (Missing twin! House/Apt Specialist LightGBM `log_real_price`, Huber)
     - Checkpoints incrementally to `outputs/models/workbench/phase7_gbdt_ckpt.npz` and compares directly
       against Phase-6 GBDTs (`P6-GBDT` = `7.39%` OOF / `7.54%` TMP).

Usage:
    .venv-api/bin/python -u src/phase7_gbdt_hpo.py --stage all
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

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'src'))

WORK_DIR = os.path.join(BASE_DIR, 'outputs', 'models', 'workbench')
P6_CACHE = os.path.join(WORK_DIR, 'phase6_splits_cache.joblib')
P7_FSETS = os.path.join(WORK_DIR, 'phase7_feature_sets.json')
P7_HPO_JSON = os.path.join(WORK_DIR, 'phase7_gbdt_hpo.json')
P7_GBDT_CKPT = os.path.join(WORK_DIR, 'phase7_gbdt_ckpt.npz')


def _target_fwd(y_price, area, defl, mode='log_real_price'):
    return np.log((y_price / defl) / area) if mode == 'log_real_psqm' else np.log(y_price / defl)


def _target_inv(pred_log, area, defl, mode='log_real_price'):
    p = np.clip(np.asarray(pred_log, dtype=np.float64), -20.0, 20.0)
    return np.clip(np.exp(p) * (area * defl if mode == 'log_real_psqm' else defl), 10_000.0, 1e7)


def _eval_split(kind, tr_df, va_df, tr_y, va_y, tr_a, va_a, tr_d, va_d, tr_cids,
                feat_cols, tmode, loss_mode, params, n_threads=6, seed=42, spec_house=None):
    from phase3_modeling import prep_tree_df
    use_cols = [c for c in feat_cols if c in tr_df.columns]
    tr_sub, va_sub = tr_df, va_df
    if spec_house is not None:
        tm_m = (tr_df['property_type_HOUSE'].to_numpy(int) == spec_house)
        vm_m = (va_df['property_type_HOUSE'].to_numpy(int) == spec_house)
        tr_sub = tr_df.iloc[tm_m]; va_sub = va_df.iloc[vm_m]
        tr_y = tr_y[tm_m]; tr_a = tr_a[tm_m]; tr_d = tr_d[tm_m]; tr_cids = tr_cids[tm_m]
        va_a = va_a[vm_m]; va_d = va_d[vm_m]
    Xtr, Xva, cat_cols = prep_tree_df(tr_sub, va_sub, use_cols)
    ytr = _target_fwd(tr_y, tr_a, tr_d, tmode)

    u_c = np.unique(tr_cids)
    es_c = set(np.random.default_rng(seed + 13).choice(u_c, size=max(1, int(len(u_c) * 0.12)), replace=False))
    is_es = np.isin(tr_cids, list(es_c))
    X_fit, y_fit = Xtr.iloc[~is_es], ytr[~is_es]
    X_es, y_es = Xtr.iloc[is_es], ytr[is_es]

    if kind == 'lgbm':
        import lightgbm as lgb
        kw = dict(
            n_estimators=3200, learning_rate=0.025, num_leaves=47, min_child_samples=35,
            subsample=0.70, subsample_freq=1, colsample_bytree=0.38,
            reg_alpha=0.8, reg_lambda=2.5, n_jobs=n_threads, random_state=seed, verbose=-1,
        )
        kw.update(params)
        if loss_mode == 'huber':
            kw['objective'] = 'huber'
            kw.setdefault('alpha', 0.15)
        elif loss_mode == 'l1':
            kw['objective'] = 'regression_l1'
        else:
            kw['objective'] = 'regression'
        for c in cat_cols:
            X_fit[c] = X_fit[c].astype('category'); X_es[c] = X_es[c].astype('category'); Xva[c] = Xva[c].astype('category')
        m = lgb.LGBMRegressor(**kw)
        m.fit(X_fit, y_fit, eval_set=[(X_es, y_es)], callbacks=[lgb.early_stopping(80, verbose=False)])
        p = m.predict(Xva)
    elif kind == 'xgb':
        import xgboost as xgb
        kw = dict(
            n_estimators=3200, learning_rate=0.025, max_depth=5, min_child_weight=22,
            subsample=0.85, colsample_bytree=0.35, reg_alpha=0.5, reg_lambda=6.0,
            tree_method='hist', enable_categorical=True, early_stopping_rounds=80,
            n_jobs=n_threads, random_state=seed, verbosity=0,
        )
        kw.update(params)
        if loss_mode == 'huber':
            kw['objective'] = 'reg:pseudohubererror'
            kw.setdefault('huber_slope', 0.15)
        elif loss_mode == 'l1':
            kw['objective'] = 'reg:absoluteerror'
        else:
            kw['objective'] = 'reg:squarederror'
        for c in cat_cols:
            cats = pd.Index(X_fit[c].unique()).union(pd.Index(['Missing']))
            X_fit[c] = pd.Categorical(X_fit[c], categories=cats)
            X_es[c] = pd.Categorical(X_es[c].where(X_es[c].isin(cats), 'Missing'), categories=cats)
            Xva[c] = pd.Categorical(Xva[c].where(Xva[c].isin(cats), 'Missing'), categories=cats)
        m = xgb.XGBRegressor(**kw)
        m.fit(X_fit, y_fit, eval_set=[(X_es, y_es)], verbose=False)
        p = m.predict(Xva)
    elif kind == 'catboost':
        from catboost import CatBoostRegressor
        kw = dict(
            iterations=3500, learning_rate=0.035, depth=7, l2_leaf_reg=5.0,
            random_strength=1.0, bagging_temperature=0.8,
            thread_count=n_threads, random_seed=seed, verbose=0, early_stopping_rounds=110,
        )
        kw.update(params)
        kw['loss_function'] = 'MAE' if loss_mode == 'l1' else 'RMSE'
        m = CatBoostRegressor(**kw)
        m.fit(X_fit, y_fit, cat_features=cat_cols, eval_set=(X_es, y_es), use_best_model=True)
        p = m.predict(Xva)
    else:
        raise ValueError(kind)

    p_eur = _target_inv(p, va_a, va_d, tmode)
    if spec_house is not None:
        return vm_m, p_eur
    return p_eur


def stage_hpo():
    import joblib
    import optuna
    from joblib import Parallel, delayed
    from phase3_features import compute_metrics

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    t0 = time.perf_counter()
    print("=== ISSUE #11 STAGE 1: OPTUNA HPO ON PHASE-6 MULTIMODAL CLUSTER-CLEAN 2-SPLIT PROFILE ===", flush=True)
    cache = joblib.load(P6_CACHE)
    fsets = json.load(open(P7_FSETS))
    f0 = [b for b in cache['bundles'] if b['id'] == 'fold_0'][0]
    tm = [b for b in cache['bundles'] if str(b['id']).startswith('temporal')][0]
    y, a, nm = cache['y_price'], cache['y_area'], cache['normal_mask']

    def _score_2splits(kind, feat_cols, loss_mode, params):
        res = Parallel(n_jobs=2, backend='loky')(
            delayed(_eval_split)(
                kind, sp['tr_df'], sp['va_df'],
                y[sp['tr_idx']], y[sp['va_idx']], a[sp['tr_idx']], a[sp['va_idx']],
                sp['tr_deflator'], sp['va_deflator'], sp['tr_cids'],
                feat_cols, 'log_real_price', loss_mode, params, n_threads=6, seed=42,
            )
            for sp in (f0, tm)
        )
        m_f0 = compute_metrics(y[f0['va_idx']][nm[f0['va_idx']]], res[0][nm[f0['va_idx']]])
        m_tm = compute_metrics(y[tm['va_idx']][nm[tm['va_idx']]], res[1][nm[tm['va_idx']]])
        # Optimize balanced Normal MdAPE across BOTH fold_0 and 2026 temporal + small MAE tie-break!
        obj = 0.5 * m_f0['mdape'] + 0.5 * m_tm['mdape'] + 0.00001 * (0.5 * m_f0['mae'] + 0.5 * m_tm['mae'])
        return obj, m_f0['mdape'], m_tm['mdape']

    hpo_out = json.load(open(P7_HPO_JSON)) if os.path.exists(P7_HPO_JSON) else {}

    # 1. LightGBM Huber on log_real_price (25 trials, ~90s!)
    if 'lgbm_price_huber' not in hpo_out:
        ts = time.perf_counter()
        def obj_lgb(trial):
            p = {
                'num_leaves': trial.suggest_int('num_leaves', 24, 96),
                'min_child_samples': trial.suggest_int('min_child_samples', 15, 70),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.25, 0.70),
                'subsample': trial.suggest_float('subsample', 0.55, 0.92),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.05, 5.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.3, 15.0, log=True),
                'alpha': trial.suggest_float('alpha', 0.08, 0.30),
                'learning_rate': 0.03,
            }
            val, f0m, tmm = _score_2splits('lgbm', fsets['gbdt_feats'], 'huber', p)
            trial.set_user_attr('f0', f0m); trial.set_user_attr('tm', tmm)
            return val
        st = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
        st.optimize(obj_lgb, n_trials=25)
        bp = dict(st.best_params); bp['learning_rate'] = 0.022
        hpo_out['lgbm_price_huber'] = {'params': bp, 'f0': st.best_trial.user_attrs['f0'], 'tm': st.best_trial.user_attrs['tm']}
        json.dump(hpo_out, open(P7_HPO_JSON, 'w'), indent=2)
        print(f"  [HPO LightGBM 25t, {time.perf_counter()-ts:.1f}s] f0={hpo_out['lgbm_price_huber']['f0']:.2f}% tm={hpo_out['lgbm_price_huber']['tm']:.2f}% | {bp}", flush=True)

    # 2. XGBoost L2 on log_real_price (20 trials, ~140s!)
    if 'xgb_price_l2' not in hpo_out:
        ts = time.perf_counter()
        def obj_xgb(trial):
            p = {
                'max_depth': trial.suggest_int('max_depth', 4, 8),
                'min_child_weight': trial.suggest_int('min_child_weight', 8, 50),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.25, 0.70),
                'subsample': trial.suggest_float('subsample', 0.60, 0.95),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.05, 5.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 20.0, log=True),
                'learning_rate': 0.03,
            }
            val, f0m, tmm = _score_2splits('xgb', fsets['gbdt_feats'], 'l2', p)
            trial.set_user_attr('f0', f0m); trial.set_user_attr('tm', tmm)
            return val
        st = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=43))
        st.optimize(obj_xgb, n_trials=20)
        bp = dict(st.best_params); bp['learning_rate'] = 0.022
        hpo_out['xgb_price_l2'] = {'params': bp, 'f0': st.best_trial.user_attrs['f0'], 'tm': st.best_trial.user_attrs['tm']}
        json.dump(hpo_out, open(P7_HPO_JSON, 'w'), indent=2)
        print(f"  [HPO XGBoost  20t, {time.perf_counter()-ts:.1f}s] f0={hpo_out['xgb_price_l2']['f0']:.2f}% tm={hpo_out['xgb_price_l2']['tm']:.2f}% | {bp}", flush=True)

    # 3. CatBoost RMSE on log_real_price WITH agency_norm (16 trials, ~260s!)
    if 'cb_price_l2' not in hpo_out:
        ts = time.perf_counter()
        def obj_cb(trial):
            p = {
                'depth': trial.suggest_int('depth', 5, 8),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.5, 20.0, log=True),
                'random_strength': trial.suggest_float('random_strength', 0.2, 3.0, log=True),
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0.1, 1.5),
                'rsm': trial.suggest_float('rsm', 0.40, 0.95),
                'learning_rate': 0.045,
                'iterations': 2600,
            }
            val, f0m, tmm = _score_2splits('catboost', fsets['gbdt_feats_with_agency'], 'l2', p)
            trial.set_user_attr('f0', f0m); trial.set_user_attr('tm', tmm)
            return val
        st = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=44))
        st.optimize(obj_cb, n_trials=16)
        bp = dict(st.best_params); bp['learning_rate'] = 0.032; bp['iterations'] = 3800
        hpo_out['cb_price_l2'] = {'params': bp, 'f0': st.best_trial.user_attrs['f0'], 'tm': st.best_trial.user_attrs['tm']}
        json.dump(hpo_out, open(P7_HPO_JSON, 'w'), indent=2)
        print(f"  [HPO CatBoost 16t, {time.perf_counter()-ts:.1f}s] f0={hpo_out['cb_price_l2']['f0']:.2f}% tm={hpo_out['cb_price_l2']['tm']:.2f}% | {bp}", flush=True)

    print(f"  Stage 1 HPO done in {time.perf_counter()-t0:.1f}s", flush=True)


def stage_train():
    import joblib
    from joblib import Parallel, delayed
    from phase3_features import compute_metrics

    t0 = time.perf_counter()
    print("\n=== ISSUE #11 STAGE 2: FULL 6-SPLIT TRAINING OF PHASE-7 TUNED GBDT + PRICE-TWIN ROSTER ===", flush=True)
    cache = joblib.load(P6_CACHE)
    fsets = json.load(open(P7_FSETS))
    hpo = json.load(open(P7_HPO_JSON))
    bundles = cache['bundles']
    y, a, nm, ih = cache['y_price'], cache['y_area'], cache['normal_mask'], cache['is_house']
    tmp_b = [b for b in bundles if str(b['id']).startswith('temporal')][0]
    tmp_idx = tmp_b['va_idx']

    ckpt = {}
    if os.path.exists(P7_GBDT_CKPT):
        z = np.load(P7_GBDT_CKPT, allow_pickle=True)
        for k in z.files:
            ckpt[k] = z[k]
        print(f"  Loaded {len(ckpt)//2} checkpointed Phase-7 GBDTs", flush=True)

    lgb_p = hpo['lgbm_price_huber']['params']
    xgb_p = hpo['xgb_price_l2']['params']
    cb_p = hpo['cb_price_l2']['params']
    xt_p = dict(lgb_p); xt_p['extra_trees'] = True; xt_p['num_leaves'] = max(48, int(lgb_p.get('num_leaves', 47)))

    roster = [
        ('p7_cb_tuned_price_l2',    'catboost', fsets['gbdt_feats_with_agency'], 'log_real_price', 'l2',    cb_p,  False),
        ('p7_cb_tuned_price_mae',   'catboost', fsets['gbdt_feats_with_agency'], 'log_real_price', 'l1',    cb_p,  False),
        ('p7_lgbm_tuned_price_hub', 'lgbm',     fsets['gbdt_feats'],             'log_real_price', 'huber', lgb_p, False),
        ('p7_lgbmxt_price_hub',     'lgbm',     fsets['gbdt_feats'],             'log_real_price', 'huber', xt_p,  False),
        ('p7_xgb_tuned_price_l2',   'xgb',      fsets['gbdt_feats'],             'log_real_price', 'l2',    xgb_p, False),
        ('p7_xgb_tuned_price_hub',  'xgb',      fsets['gbdt_feats'],             'log_real_price', 'huber', xgb_p, False),
        ('p7_spec_cb_price_l2',     'catboost', fsets['gbdt_feats_with_agency'], 'log_real_price', 'l2',    cb_p,  True),
        ('p7_spec_xgb_price_l2',    'xgb',      fsets['gbdt_feats'],             'log_real_price', 'l2',    xgb_p, True),
        ('p7_spec_lgbm_price_hub',  'lgbm',     fsets['gbdt_feats'],             'log_real_price', 'huber', lgb_p, True),
    ]

    for name, kind, fcols, tmode, lmode, params, is_spec in roster:
        if f'oof__{name}' in ckpt:
            po, pt = ckpt[f'oof__{name}'], ckpt[f'tmp__{name}']
            mo = compute_metrics(y[nm], po[nm]); mt = compute_metrics(y[tmp_idx][nm[tmp_idx]], pt[nm[tmp_idx]])
            print(f"  [ckpt] {name:26s} | NORM OOF={mo['mdape']:5.2f}% (€{mo['mae']:6,.0f}) | NORM TMP={mt['mdape']:5.2f}%", flush=True)
            continue
        ts = time.perf_counter()
        if not is_spec:
            outs = Parallel(n_jobs=6, backend='loky')(
                delayed(_eval_split)(
                    kind, b['tr_df'], b['va_df'], y[b['tr_idx']], y[b['va_idx']],
                    a[b['tr_idx']], a[b['va_idx']], b['tr_deflator'], b['va_deflator'], b['tr_cids'],
                    fcols, tmode, lmode, params, n_threads=3, seed=42, spec_house=None,
                )
                for b in bundles
            )
        else:
            outs_h = Parallel(n_jobs=6, backend='loky')(
                delayed(_eval_split)(
                    kind, b['tr_df'], b['va_df'], y[b['tr_idx']], y[b['va_idx']],
                    a[b['tr_idx']], a[b['va_idx']], b['tr_deflator'], b['va_deflator'], b['tr_cids'],
                    fcols, tmode, lmode, params, n_threads=3, seed=42, spec_house=1,
                )
                for b in bundles
            )
            outs_a = Parallel(n_jobs=6, backend='loky')(
                delayed(_eval_split)(
                    kind, b['tr_df'], b['va_df'], y[b['tr_idx']], y[b['va_idx']],
                    a[b['tr_idx']], a[b['va_idx']], b['tr_deflator'], b['va_deflator'], b['tr_cids'],
                    fcols, tmode, lmode, params, n_threads=3, seed=42, spec_house=0,
                )
                for b in bundles
            )
            outs = []
            for (mh, ph), (ma, pa), b in zip(outs_h, outs_a, bundles):
                comb = np.zeros(len(b['va_idx']), dtype=np.float64)
                comb[mh] = ph; comb[ma] = pa
                outs.append(comb)

        oof = np.zeros(len(y), dtype=np.float64); tmp_p = None
        for b, p_eur in zip(bundles, outs):
            if str(b['id']).startswith('temporal'):
                tmp_p = p_eur
            else:
                oof[b['va_idx']] = p_eur
        ckpt[f'oof__{name}'] = oof; ckpt[f'tmp__{name}'] = tmp_p
        np.savez_compressed(P7_GBDT_CKPT, **ckpt)
        mo = compute_metrics(y[nm], oof[nm])
        mh = compute_metrics(y[nm & (ih == 1)], oof[nm & (ih == 1)])
        ma = compute_metrics(y[nm & (ih == 0)], oof[nm & (ih == 0)])
        mt = compute_metrics(y[tmp_idx][nm[tmp_idx]], tmp_p[nm[tmp_idx]])
        print(f"  {name:26s} ({time.perf_counter()-ts:5.1f}s) | NORM OOF={mo['mdape']:5.2f}% (H {mh['mdape']:.2f}/A {ma['mdape']:.2f}) €{mo['mae']:6,.0f} | NORM TMP={mt['mdape']:5.2f}% (bias {mt['bias_pct']:+5.2f}%)", flush=True)
    print(f"  Stage 2 GBDT roster done in {time.perf_counter()-t0:.1f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['hpo', 'train', 'all'], default='all')
    args = ap.parse_args()
    if args.stage == 'all':
        for st in ('hpo', 'train'):
            subprocess.run([sys.executable, '-u', __file__, '--stage', st], check=True)
    elif args.stage == 'hpo':
        stage_hpo()
    elif args.stage == 'train':
        stage_train()


if __name__ == '__main__':
    main()
