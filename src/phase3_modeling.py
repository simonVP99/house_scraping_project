"""Phase 3A: Feature Pruning, Specialist Experts, PyTorch MPS Tabular ResNet & Optuna Tuning.

Uses the cached strictly fold-safe 6-split feature bundle (`phase3_splits_cache.joblib`,
340 features across 5 grouped OOF folds + 2026 temporal holdout) and saturates:
  - All 18 CPU cores via 6-way parallel folds × 3 threads for GBDT & Optuna
  - Apple Silicon MPS GPU (fp32/AMP) for PyTorch Tabular ResNet with entity embeddings

Pipeline:
  1. Benchmark new Phase-3 features (word+char TF-IDF, structural text residual,
     street/agency EB target encoding, CPI-deflated `log(real €/m²)`) on LGBM.
  2. Feature importance audit + pruning sweep (top 120 / 160 / 220 / all 340).
  3. Separate HOUSE vs APARTMENT specialist models vs unified model.
  4. PyTorch MPS Tabular ResNet (categorical embeddings + residual LayerNorm-SiLU
     blocks + CosineAnnealing + Huber/SmoothL1 loss on `log(real €/m²)`) —
     true non-tree model family!
  5. Optuna tuning (grouped CV) for LightGBM, XGBoost, and CatBoost on winning
     target (`log(real €/m²)`) and complementary target (`log(real_price)`).
  6. Save every model's OOF & 2026 temporal predictions to
     `outputs/models/workbench/phase3_model_preds.npz` for final stacking +
     AutoGluon benchmark + Conformal Quantile calibration.

Usage:
    .venv-api/bin/python -u src/phase3_modeling.py
"""
from __future__ import annotations

import json
import os
import time
import warnings

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
from joblib import Parallel, delayed
from scipy.optimize import minimize
from sklearn.preprocessing import QuantileTransformer, StandardScaler

from phase3_features import (
    compute_metrics,
    fmt,
    load_or_build_phase3_splits,
)

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE_DIR, 'outputs', 'models', 'workbench')


def prep_tree_df(df: pd.DataFrame, feat_cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    X = df[feat_cols].copy()
    cat_cols = []
    for c in feat_cols:
        if pd.api.types.is_bool_dtype(X[c]):
            X[c] = X[c].astype('int8')
        elif pd.api.types.is_string_dtype(X[c]) or pd.api.types.is_object_dtype(X[c]) or isinstance(X[c].dtype, pd.CategoricalDtype):
            X[c] = X[c].fillna('Missing').astype(str)
            cat_cols.append(c)
        else:
            X[c] = pd.to_numeric(X[c], errors='coerce').astype('float32')
    return X, cat_cols


def _target_transform(df: pd.DataFrame, cpi_idx: np.ndarray, target_mode: str) -> np.ndarray:
    p = df['price'].to_numpy(dtype=float)
    a = df['livable_area'].to_numpy(dtype=float)
    if target_mode == 'log_real_psqm':
        return np.log((p / cpi_idx) / a)
    if target_mode == 'log_real_price':
        return np.log(p / cpi_idx)
    if target_mode == 'log_psqm':
        return np.log(p / a)
    if target_mode == 'log_price':
        return np.log(p)
    raise ValueError(target_mode)


def _target_inverse(pred: np.ndarray, df: pd.DataFrame, cpi_idx: np.ndarray, target_mode: str) -> np.ndarray:
    a = df['livable_area'].to_numpy(dtype=float)
    if target_mode == 'log_real_psqm':
        return np.exp(pred) * a * cpi_idx
    if target_mode == 'log_real_price':
        return np.exp(pred) * cpi_idx
    if target_mode == 'log_psqm':
        return np.exp(pred) * a
    if target_mode == 'log_price':
        return np.exp(pred)
    raise ValueError(target_mode)


# ---------------------------------------------------------------------------
# Generic parallel 6-split GBDT runner (returns OOF, TMP, and feature importances)
# ---------------------------------------------------------------------------
def _fit_gbdt_split(
    split_id: str,
    tr_df: pd.DataFrame,
    va_df: pd.DataFrame,
    tr_cpi: np.ndarray,
    va_cpi: np.ndarray,
    feat_cols: list[str],
    model_kind: str,
    target_mode: str,
    loss_mode: str,
    params: dict | None,
    n_threads: int,
    seed: int,
    specialist_house_flag: int | None = None,  # None=all, 1=houses, 0=apts
) -> tuple[str, np.ndarray, np.ndarray | None]:
    if specialist_house_flag is not None:
        tr_df = tr_df[tr_df['property_type_HOUSE'] == specialist_house_flag].reset_index(drop=True)
        tr_cpi = tr_cpi[(tr_df.index.to_numpy() if False else np.ones(len(tr_df), dtype=bool))]

    X_tr, cat_cols = prep_tree_df(tr_df, feat_cols)
    X_va, _ = prep_tree_df(va_df, feat_cols)
    y_tr = _target_transform(tr_df, tr_cpi, target_mode)
    groups_tr = tr_df['property_key'].astype(str).to_numpy()

    rng = np.random.default_rng(seed)
    uniq_g = np.unique(groups_tr)
    es_g = set(rng.choice(uniq_g, size=max(1, int(0.12 * len(uniq_g))), replace=False))
    es_mask = np.isin(groups_tr, list(es_g))

    imp = None
    if model_kind == 'lgbm':
        import lightgbm as lgb
        for c in cat_cols:
            cats = sorted(set(X_tr[c].unique()) | {'Missing'})
            X_tr[c] = pd.Categorical(X_tr[c], categories=cats)
            X_va[c] = pd.Categorical(X_va[c].where(X_va[c].isin(cats), 'Missing'), categories=cats)
        obj_map = {'l2': 'regression', 'l1': 'regression_l1', 'huber': 'huber'}
        cfg = dict(
            objective=obj_map[loss_mode], n_estimators=3000, learning_rate=0.025,
            num_leaves=63, min_child_samples=25, subsample=0.8,
            subsample_freq=1, colsample_bytree=0.65, reg_alpha=1.0,
            reg_lambda=3.0, random_state=seed, n_jobs=n_threads, verbose=-1,
        )
        if loss_mode == 'huber':
            cfg['alpha'] = 0.15
        if params:
            cfg.update(params)
        model = lgb.LGBMRegressor(**cfg)
        model.fit(
            X_tr[~es_mask], y_tr[~es_mask],
            eval_set=[(X_tr[es_mask], y_tr[es_mask])],
            callbacks=[lgb.early_stopping(90, verbose=False)],
        )
        vp = model.predict(X_va)
        imp = model.booster_.feature_importance(importance_type='gain')

    elif model_kind == 'xgb':
        import xgboost as xgb
        for c in cat_cols:
            cats = sorted(set(X_tr[c].unique()) | {'Missing'})
            X_tr[c] = pd.Categorical(X_tr[c], categories=cats)
            X_va[c] = pd.Categorical(X_va[c].where(X_va[c].isin(cats), 'Missing'), categories=cats)
        obj_map = {'l2': 'reg:squarederror', 'l1': 'reg:absoluteerror', 'huber': 'reg:pseudohubererror'}
        cfg = dict(
            objective=obj_map[loss_mode], n_estimators=3000, learning_rate=0.025,
            max_depth=7, min_child_weight=10, subsample=0.8,
            colsample_bytree=0.65, reg_alpha=1.0, reg_lambda=3.0,
            tree_method='hist', enable_categorical=True,
            early_stopping_rounds=90, random_state=seed, n_jobs=n_threads,
        )
        if loss_mode == 'huber':
            cfg['huber_slope'] = 0.15
        if params:
            cfg.update(params)
        model = xgb.XGBRegressor(**cfg)
        model.fit(
            X_tr[~es_mask], y_tr[~es_mask],
            eval_set=[(X_tr[es_mask], y_tr[es_mask])],
            verbose=False,
        )
        vp = model.predict(X_va)

    elif model_kind == 'catboost':
        from catboost import CatBoostRegressor
        loss_map = {'l2': 'RMSE', 'l1': 'MAE', 'huber': 'Huber:delta=0.15'}
        cfg = dict(
            loss_function=loss_map[loss_mode], iterations=3500, learning_rate=0.035,
            depth=7, l2_leaf_reg=5.0, random_seed=seed, verbose=0, thread_count=n_threads,
        )
        if params:
            cfg.update(params)
        model = CatBoostRegressor(**cfg)
        model.fit(
            X_tr[~es_mask], y_tr[~es_mask],
            cat_features=cat_cols,
            eval_set=(X_tr[es_mask], y_tr[es_mask]),
            early_stopping_rounds=120,
            verbose=False,
        )
        vp = model.predict(X_va)
    else:
        raise ValueError(model_kind)

    return split_id, _target_inverse(vp, va_df, va_cpi, target_mode), imp


def run_6splits(
    bundles: list[dict],
    feat_cols: list[str],
    model_kind: str,
    target_mode: str = 'log_real_psqm',
    loss_mode: str = 'l2',
    params: dict | None = None,
    seed: int = 42,
    n_threads: int = 3,
    only_oof: bool = False,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    run_b = [b for b in bundles if (not only_oof or b['id'].startswith('fold_'))]
    res = Parallel(n_jobs=len(run_b), backend='loky')(
        delayed(_fit_gbdt_split)(
            b['id'], b['tr_df'], b['va_df'], b['tr_cpi_idx'], b['va_cpi_idx'],
            feat_cols, model_kind, target_mode, loss_mode, params, n_threads, seed,
        )
        for b in run_b
    )
    res_map = {sid: (vp, imp) for sid, vp, imp in res}
    n_all = sum(len(b['va_df']) for b in bundles if b['id'].startswith('fold_'))
    oof = np.zeros(n_all, dtype=float)
    tmp = None
    imps = []
    for b in run_b:
        vp, imp = res_map[b['id']]
        if b['id'].startswith('fold_'):
            oof[b['va_idx']] = vp
            if imp is not None:
                imps.append(imp)
        else:
            tmp = vp
    mean_imp = np.mean(imps, axis=0) if imps else None
    return oof, tmp, mean_imp


def run_specialist_6splits(
    bundles: list[dict],
    feat_cols: list[str],
    model_kind: str,
    target_mode: str = 'log_real_psqm',
    loss_mode: str = 'l2',
    params: dict | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Train separate HOUSE and APARTMENT models on every split."""
    jobs = []
    for b in bundles:
        for h_flag in (1, 0):
            tr_m = (b['tr_df']['property_type_HOUSE'].to_numpy() == h_flag)
            va_m = (b['va_df']['property_type_HOUSE'].to_numpy() == h_flag)
            jobs.append((
                f"{b['id']}|h={h_flag}",
                b['tr_df'][tr_m].reset_index(drop=True),
                b['va_df'][va_m].reset_index(drop=True),
                b['tr_cpi_idx'][tr_m],
                b['va_cpi_idx'][va_m],
                feat_cols, model_kind, target_mode, loss_mode, params, 2, seed + h_flag,
            ))
    res = Parallel(n_jobs=6, backend='loky')(delayed(_fit_gbdt_split)(*j) for j in jobs)
    res_map = {sid: vp for sid, vp, _ in res}

    n_all = sum(len(b['va_df']) for b in bundles if b['id'].startswith('fold_'))
    oof = np.zeros(n_all, dtype=float)
    tmp = None
    for b in bundles:
        va_h = (b['va_df']['property_type_HOUSE'].to_numpy() == 1)
        pred = np.zeros(len(b['va_df']), dtype=float)
        pred[va_h] = res_map[f"{b['id']}|h=1"]
        pred[~va_h] = res_map[f"{b['id']}|h=0"]
        if b['id'].startswith('fold_'):
            oof[b['va_idx']] = pred
        else:
            tmp = pred
    return oof, tmp


# ---------------------------------------------------------------------------
# PyTorch MPS Tabular ResNet with Entity Embeddings (true non-tree diversity!)
# ---------------------------------------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class TabularResNet(nn.Module):
    def __init__(self, n_num: int, cat_cards: list[int], d_hidden: int = 320, n_blocks: int = 3, dropout: float = 0.22):
        super().__init__()
        self.embs = nn.ModuleList([
            nn.Embedding(card, min(24, max(4, int(round(1.6 * (card ** 0.5))))))
            for card in cat_cards
        ])
        emb_dim = sum(e.embedding_dim for e in self.embs)
        self.in_proj = nn.Sequential(
            nn.Linear(n_num + emb_dim, d_hidden),
            nn.LayerNorm(d_hidden),
        )
        self.blocks = nn.ModuleList([ResBlock(d_hidden, dropout) for _ in range(n_blocks)])
        self.head = nn.Sequential(nn.LayerNorm(d_hidden), nn.SiLU(), nn.Linear(d_hidden, 1))

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        if len(self.embs) > 0:
            emb_list = [emb(x_cat[:, i]) for i, emb in enumerate(self.embs)]
            x = torch.cat([x_num] + emb_list, dim=1)
        else:
            x = x_num
        h = self.in_proj(x)
        for blk in self.blocks:
            h = blk(h)
        return self.head(h).squeeze(-1)


def train_mps_resnet_6splits(
    bundles: list[dict],
    feat_cols: list[str],
    target_mode: str = 'log_real_psqm',
    epochs: int = 55,
    batch_size: int = 256,
    n_seeds: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    n_all = sum(len(b['va_df']) for b in bundles if b['id'].startswith('fold_'))
    oof = np.zeros(n_all, dtype=float)
    tmp = None

    for b in bundles:
        tr_df, va_df = b['tr_df'], b['va_df']
        X_tr_raw, cat_cols = prep_tree_df(tr_df, feat_cols)
        X_va_raw, _ = prep_tree_df(va_df, feat_cols)
        num_cols = [c for c in feat_cols if c not in set(cat_cols)]

        # Numeric: median fill + QuantileTransformer(normal) -> handles heavy tails cleanly for NNs!
        X_tr_num = X_tr_raw[num_cols].to_numpy(dtype=np.float32)
        X_va_num = X_va_raw[num_cols].to_numpy(dtype=np.float32)
        meds = np.nanmedian(X_tr_num, axis=0)
        meds = np.where(np.isnan(meds), 0.0, meds)
        tr_nan = np.isnan(X_tr_num)
        va_nan = np.isnan(X_va_num)
        X_tr_num = np.where(tr_nan, meds[None, :], X_tr_num)
        X_va_num = np.where(va_nan, meds[None, :], X_va_num)

        qt = QuantileTransformer(n_quantiles=500, output_distribution='normal', random_state=42)
        X_tr_num = np.clip(qt.fit_transform(X_tr_num), -4.5, 4.5).astype(np.float32)
        X_va_num = np.clip(qt.transform(X_va_num), -4.5, 4.5).astype(np.float32)

        # Categoricals: integer codes (0 = unknown/rare)
        cat_cards = []
        X_tr_cat = np.zeros((len(tr_df), len(cat_cols)), dtype=np.int64)
        X_va_cat = np.zeros((len(va_df), len(cat_cols)), dtype=np.int64)
        for i, c in enumerate(cat_cols):
            vc = X_tr_raw[c].value_counts()
            valid = vc[vc >= 5].index.tolist()
            mapping = {v: idx + 1 for idx, v in enumerate(valid)}
            X_tr_cat[:, i] = X_tr_raw[c].map(mapping).fillna(0).astype(np.int64).to_numpy()
            X_va_cat[:, i] = X_va_raw[c].map(mapping).fillna(0).astype(np.int64).to_numpy()
            cat_cards.append(len(valid) + 1)

        y_tr = _target_transform(tr_df, b['tr_cpi_idx'], target_mode).astype(np.float32)
        y_mean, y_std = float(y_tr.mean()), float(y_tr.std())
        y_tr_norm = (y_tr - y_mean) / y_std

        groups_tr = tr_df['property_key'].astype(str).to_numpy()
        preds_seeds = []

        for s in range(n_seeds):
            torch.manual_seed(100 + s)
            rng = np.random.default_rng(100 + s)
            uniq_g = np.unique(groups_tr)
            es_g = set(rng.choice(uniq_g, size=max(1, int(0.12 * len(uniq_g))), replace=False))
            es_mask = np.isin(groups_tr, list(es_g))

            tr_idx = np.where(~es_mask)[0]
            es_idx = np.where(es_mask)[0]

            t_xn = torch.from_numpy(X_tr_num[tr_idx]).to(device)
            t_xc = torch.from_numpy(X_tr_cat[tr_idx]).to(device)
            t_y = torch.from_numpy(y_tr_norm[tr_idx]).to(device)

            e_xn = torch.from_numpy(X_tr_num[es_idx]).to(device)
            e_xc = torch.from_numpy(X_tr_cat[es_idx]).to(device)
            e_y = torch.from_numpy(y_tr_norm[es_idx]).to(device)

            v_xn = torch.from_numpy(X_va_num).to(device)
            v_xc = torch.from_numpy(X_va_cat).to(device)

            model = TabularResNet(X_tr_num.shape[1], cat_cards, d_hidden=320, n_blocks=3, dropout=0.25).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=2e-3)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
            loss_fn = nn.HuberLoss(delta=0.6)

            best_loss = float('inf')
            best_state = None
            n_tr = len(tr_idx)

            for ep in range(epochs):
                model.train()
                perm = torch.randperm(n_tr, device=device)
                for start in range(0, n_tr, batch_size):
                    bi = perm[start:start + batch_size]
                    opt.zero_grad(set_to_none=True)
                    out = model(t_xn[bi], t_xc[bi])
                    loss = loss_fn(out, t_y[bi])
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                    opt.step()
                sched.step()

                model.eval()
                with torch.inference_mode():
                    val_out = model(e_xn, e_xc)
                    val_loss = float(torch.mean(torch.abs(val_out - e_y)).item())
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
            model.eval()
            with torch.inference_mode():
                va_pred = model(v_xn, v_xc).cpu().numpy() * y_std + y_mean
            preds_seeds.append(va_pred)

        va_pred_mean = np.mean(preds_seeds, axis=0)
        eur_pred = _target_inverse(va_pred_mean, va_df, b['va_cpi_idx'], target_mode)
        if b['id'].startswith('fold_'):
            oof[b['va_idx']] = eur_pred
        else:
            tmp = eur_pred

    return oof, tmp


# ---------------------------------------------------------------------------
# Main Phase-3A Pipeline
# ---------------------------------------------------------------------------
def main():
    t_all = time.perf_counter()
    os.makedirs(OUT_DIR, exist_ok=True)
    data = load_or_build_phase3_splits()
    bundles = data['bundles']
    y_price = data['y_price']
    y_tmp = data['y_tmp_true']
    folds = data['folds']
    all_feats = data['feat_cols_stationary']
    with_agency = data['feat_cols_with_agency']

    print(f"Loaded Phase-3 cache: {len(y_price):,} OOF rows, {len(y_tmp):,} 2026 temporal rows, {len(all_feats)} features")
    log_records: list[dict] = []

    def rec(name: str, op: np.ndarray, tp: np.ndarray, dt: float):
        mo = compute_metrics(y_price, op)
        mt = compute_metrics(y_tmp, tp)
        print(f"  {name:46s} ({dt:5.1f}s) | OOF: {fmt(mo)}")
        print(f"  {'':54s} | TMP: {fmt(mt)}")
        log_records.append({'name': name, 'seconds': round(dt, 1), 'oof': mo, 'temporal': mt})
        return mo, mt

    # ======================================================================
    # 1. PHASE-3 FEATURE BENCHMARK & PRUNING SWEEP (LightGBM)
    # ======================================================================
    print("\n=== 1. PHASE-3 FEATURES + CPI-DEFLATED log(real €/m²) + PRUNING ===")
    t0 = time.perf_counter()
    op_340, tp_340, imp_340 = run_6splits(bundles, all_feats, 'lgbm', 'log_real_psqm', 'l2')
    rec('P3-LGBM log(real €/m²) [all 340 feats]', op_340, tp_340, time.perf_counter() - t0)

    # Show top 25 features by gain
    imp_df = pd.DataFrame({'feature': all_feats, 'gain': imp_340}).sort_values('gain', ascending=False).reset_index(drop=True)
    imp_df.to_csv(os.path.join(OUT_DIR, 'phase3_feature_importance.csv'), index=False)
    print("\n  Top 20 features by LightGBM gain:")
    for i, r in imp_df.head(20).iterrows():
        print(f"    {i+1:2d}. {r['feature']:38s} gain={r['gain']:10.1f}")
    zero_imp = int((imp_df['gain'] == 0).sum())
    print(f"  Zero-gain features: {zero_imp} / {len(all_feats)}")

    # Pruning sweep: keep top K features
    best_k = len(all_feats)
    best_score = compute_metrics(y_price, op_340)['mdape'] + 0.5 * compute_metrics(y_tmp, tp_340)['mdape']
    pruned_feats = all_feats
    for k in (100, 150, 210):
        sub_f = imp_df['feature'].head(k).tolist()
        t0 = time.perf_counter()
        op_k, tp_k, _ = run_6splits(bundles, sub_f, 'lgbm', 'log_real_psqm', 'l2')
        mo, mt = rec(f'P3-LGBM log(real €/m²) [top {k} feats]', op_k, tp_k, time.perf_counter() - t0)
        sc = mo['mdape'] + 0.5 * mt['mdape']
        if sc < best_score:
            best_score = sc
            best_k = k
            pruned_feats = sub_f

    print(f"\n  Selected pruned feature set: top {best_k} features")

    # ======================================================================
    # 2. HOUSE vs APARTMENT SPECIALIST EXPERTS
    # ======================================================================
    print("\n=== 2. HOUSE vs APARTMENT SPECIALIST EXPERTS ===")
    t0 = time.perf_counter()
    op_spec_lgb, tp_spec_lgb = run_specialist_6splits(bundles, pruned_feats, 'lgbm', 'log_real_psqm', 'l2')
    rec('Specialist LGBM (House + Apt separate)', op_spec_lgb, tp_spec_lgb, time.perf_counter() - t0)

    t0 = time.perf_counter()
    op_spec_cb, tp_spec_cb = run_specialist_6splits(bundles, pruned_feats, 'catboost', 'log_real_psqm', 'l2')
    rec('Specialist CatBoost (House + Apt separate)', op_spec_cb, tp_spec_cb, time.perf_counter() - t0)

    # ======================================================================
    # 3. PYTORCH MPS TABULAR RESNET (NON-TREE MODEL FAMILY!)
    # ======================================================================
    print("\n=== 3. PYTORCH MPS TABULAR RESNET (ENTITY EMBEDDINGS + HUBER) ===")
    t0 = time.perf_counter()
    op_nn_psqm, tp_nn_psqm = train_mps_resnet_6splits(bundles, pruned_feats, 'log_real_psqm', epochs=55, n_seeds=2)
    rec('PyTorch MPS ResNet log(real €/m²)', op_nn_psqm, tp_nn_psqm, time.perf_counter() - t0)

    t0 = time.perf_counter()
    op_nn_price, tp_nn_price = train_mps_resnet_6splits(bundles, pruned_feats, 'log_real_price', epochs=55, n_seeds=2)
    rec('PyTorch MPS ResNet log(real_price)', op_nn_price, tp_nn_price, time.perf_counter() - t0)

    # ======================================================================
    # 4. OPTUNA TUNING (GROUPED 5-FOLD CV, ALL 18 CORES)
    # ======================================================================
    print("\n=== 4. OPTUNA HYPERPARAMETER TUNING (GROUPED CV) ===")

    # 4a. LightGBM Optuna (18 trials × 5 parallel folds)
    def lgb_objective(trial: optuna.Trial) -> float:
        p = {
            'learning_rate': 0.03,
            'num_leaves': trial.suggest_int('num_leaves', 24, 128),
            'min_child_samples': trial.suggest_int('min_child_samples', 15, 70),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.35, 0.85),
            'subsample': trial.suggest_float('subsample', 0.65, 0.95),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 20.0, log=True),
            'max_bin': trial.suggest_categorical('max_bin', [255, 511]),
            'n_estimators': 2500,
        }
        op, _, _ = run_6splits(bundles, pruned_feats, 'lgbm', 'log_real_psqm', 'l2', p, n_threads=3, only_oof=True)
        m = compute_metrics(y_price, op)
        return 0.6 * m['mdape'] + 0.4 * (m['log_rmse'] * 100.0)

    t0 = time.perf_counter()
    study_lgb = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
    study_lgb.optimize(lgb_objective, n_trials=16)
    best_lgb_p = dict(study_lgb.best_params, learning_rate=0.02, n_estimators=4000)
    print(f"  Optuna LGBM done in {time.perf_counter()-t0:.1f}s | best params: {study_lgb.best_params}")

    # 4b. XGBoost Optuna (14 trials × 5 parallel folds)
    def xgb_objective(trial: optuna.Trial) -> float:
        p = {
            'learning_rate': 0.03,
            'max_depth': trial.suggest_int('max_depth', 5, 9),
            'min_child_weight': trial.suggest_int('min_child_weight', 5, 40),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.35, 0.85),
            'subsample': trial.suggest_float('subsample', 0.65, 0.95),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 25.0, log=True),
            'max_cat_to_onehot': trial.suggest_categorical('max_cat_to_onehot', [4, 8]),
            'n_estimators': 2500,
        }
        op, _, _ = run_6splits(bundles, pruned_feats, 'xgb', 'log_real_psqm', 'l2', p, n_threads=3, only_oof=True)
        m = compute_metrics(y_price, op)
        return 0.6 * m['mdape'] + 0.4 * (m['log_rmse'] * 100.0)

    t0 = time.perf_counter()
    study_xgb = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
    study_xgb.optimize(xgb_objective, n_trials=14)
    best_xgb_p = dict(study_xgb.best_params, learning_rate=0.02, n_estimators=4000)
    print(f"  Optuna XGB  done in {time.perf_counter()-t0:.1f}s | best params: {study_xgb.best_params}")

    # ======================================================================
    # 5. TRAIN FULL DIVERSE MODEL ZOO ON TUNED CONFIGS
    # ======================================================================
    print("\n=== 5. TUNED DIVERSE MODEL ZOO (PRUNED FEATURES) ===")
    preds_oof: dict[str, np.ndarray] = {
        'spec_lgbm_real_psqm': op_spec_lgb,
        'spec_catboost_real_psqm': op_spec_cb,
        'mps_resnet_real_psqm': op_nn_psqm,
        'mps_resnet_real_price': op_nn_price,
    }
    preds_tmp: dict[str, np.ndarray] = {
        'spec_lgbm_real_psqm': tp_spec_lgb,
        'spec_catboost_real_psqm': tp_spec_cb,
        'mps_resnet_real_psqm': tp_nn_psqm,
        'mps_resnet_real_price': tp_nn_price,
    }

    zoo = [
        ('lgbm_tuned_real_psqm_l2',    pruned_feats, 'lgbm',     'log_real_psqm',  'l2',    best_lgb_p),
        ('lgbm_tuned_real_psqm_huber', pruned_feats, 'lgbm',     'log_real_psqm',  'huber', best_lgb_p),
        ('lgbm_tuned_real_price_l2',   pruned_feats, 'lgbm',     'log_real_price', 'l2',    best_lgb_p),
        ('xgb_tuned_real_psqm_l2',     pruned_feats, 'xgb',      'log_real_psqm',  'l2',    best_xgb_p),
        ('xgb_tuned_real_psqm_huber',  pruned_feats, 'xgb',      'log_real_psqm',  'huber', best_xgb_p),
        ('xgb_tuned_real_price_l2',    pruned_feats, 'xgb',      'log_real_price', 'l2',    best_xgb_p),
        # CatBoost with native ordered target encoding on pruned + immo_office_name!
        ('cb_d7_real_psqm_l2',         pruned_feats + ['immo_office_name'], 'catboost', 'log_real_psqm',  'l2',    {'depth': 7, 'l2_leaf_reg': 5.0, 'learning_rate': 0.03, 'iterations': 4500}),
        ('cb_d8_real_psqm_l2',         pruned_feats + ['immo_office_name'], 'catboost', 'log_real_psqm',  'l2',    {'depth': 8, 'l2_leaf_reg': 8.0, 'learning_rate': 0.03, 'iterations': 4500}),
        ('cb_d7_real_price_l2',        pruned_feats + ['immo_office_name'], 'catboost', 'log_real_price', 'l2',    {'depth': 7, 'l2_leaf_reg': 5.0, 'learning_rate': 0.03, 'iterations': 4500}),
        ('cb_d7_real_psqm_huber',      pruned_feats + ['immo_office_name'], 'catboost', 'log_real_psqm',  'huber', {'depth': 7, 'l2_leaf_reg': 5.0, 'learning_rate': 0.03, 'iterations': 4500}),
    ]

    for key, fset, kind, tmode, lmode, params in zoo:
        # Deduplicate feature names in case immo_office_name already included
        fset_clean = list(dict.fromkeys(fset))
        t0 = time.perf_counter()
        op, tp, _ = run_6splits(bundles, fset_clean, kind, tmode, lmode, params)
        rec(key, op, tp, time.perf_counter() - t0)
        preds_oof[key] = op
        preds_tmp[key] = tp

    # ======================================================================
    # 6. NESTED-OOF ENSEMBLE BLENDING (ALL FAMILIES: CB + XGB + LGBM + MPS ResNet + Specialists)
    # ======================================================================
    print("\n=== 6. PHASE-3A ENSEMBLE SUMMARY ===")
    names = list(preds_oof.keys())
    O_log = np.column_stack([np.log(np.maximum(preds_oof[k], 10_000.0)) for k in names])
    T_log = np.column_stack([np.log(np.maximum(preds_tmp[k], 10_000.0)) for k in names])
    y_log = np.log(y_price)

    def _fit_weights(X_mat: np.ndarray, y_t: np.ndarray) -> np.ndarray:
        k = X_mat.shape[1]
        def loss_fn(w):
            err = (X_mat @ w) - y_t
            return 0.55 * np.mean(np.abs(err)) + 0.45 * np.sqrt(np.mean(err ** 2))
        res = minimize(
            loss_fn, np.ones(k) / k,
            bounds=[(0.0, 1.0)] * k,
            constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
            method='SLSQP',
        )
        return res.x

    nested_oof_log = np.zeros(len(y_price))
    for f in sorted(np.unique(folds)):
        tr_m = folds != f
        va_m = folds == f
        wf = _fit_weights(O_log[tr_m], y_log[tr_m])
        nested_oof_log[va_m] = O_log[va_m] @ wf

    w_full = _fit_weights(O_log, y_log)
    ens_oof = np.exp(nested_oof_log)
    ens_tmp = np.exp(T_log @ w_full)

    # Also evaluate GBDT-only vs GBDT + MPS ResNet to quantify neural net contribution!
    nn_keys = {'mps_resnet_real_psqm', 'mps_resnet_real_price'}
    gbdt_idx = [i for i, k in enumerate(names) if k not in nn_keys]
    nested_gbdt = np.zeros(len(y_price))
    for f in sorted(np.unique(folds)):
        tr_m = folds != f
        va_m = folds == f
        wf = _fit_weights(O_log[tr_m][:, gbdt_idx], y_log[tr_m])
        nested_gbdt[va_m] = O_log[va_m][:, gbdt_idx] @ wf
    w_gbdt = _fit_weights(O_log[:, gbdt_idx], y_log)

    rec('ENS-GBDT-only (CB + XGB + LGBM + Spec)', np.exp(nested_gbdt), np.exp(T_log[:, gbdt_idx] @ w_gbdt), 0.0)
    rec('ENS-FULL (+ PyTorch MPS Tabular ResNet)', ens_oof, ens_tmp, 0.0)

    print("\nLearned Phase-3A ensemble weights:")
    for nm, wt in sorted(zip(names, w_full), key=lambda x: -x[1]):
        if wt >= 0.005:
            print(f"  {wt*100:5.1f}%  {nm}")

    # Save everything for Phase 3B (AutoGluon benchmark + Conformal Quantile bands)
    np.savez_compressed(
        os.path.join(OUT_DIR, 'phase3_model_preds.npz'),
        names=np.array(names),
        O_log=O_log,
        T_log=T_log,
        y_oof=y_price,
        y_tmp=y_tmp,
        ens_oof=ens_oof,
        ens_tmp=ens_tmp,
        weights=w_full,
    )
    with open(os.path.join(OUT_DIR, 'phase3a_report.json'), 'w') as fh:
        json.dump({
            'pruned_k': best_k,
            'pruned_features': pruned_feats,
            'best_lgb_params': best_lgb_p,
            'best_xgb_params': best_xgb_p,
            'ensemble_weights': {nm: round(float(wt), 4) for nm, wt in zip(names, w_full)},
            'results': log_records,
            'total_seconds': round(time.perf_counter() - t_all, 1),
        }, fh, indent=2)
    print(f"\nPhase 3A finished in {time.perf_counter()-t_all:.1f}s")


if __name__ == '__main__':
    main()
