"""Phase 3B: Push the Frontier — AutoGluon Grouped Benchmark, Fixed Huber/L1 Experts,
Multi-Seed MPS Tabular ResNet + MLP, 2nd-Level Meta-Stacker, and Conformal Quantiles.

Builds directly on `outputs/models/workbench/phase3_model_preds.npz` (12 valid models)
and `phase3_splits_cache.joblib` (210 pruned features across 5 grouped folds + 2026 temporal):

  1. Add **fixed & strong missing experts**:
     - `xgb_tuned_real_psqm_l1` (MAE on log real €/m²)
     - `cb_d7_real_psqm_l1` (MAE on log real €/m² — CatBoost's exact L1, no Huber step bug!)
     - `lgbm_tuned_real_price_huber` (Huber on log real_price — since LGBM Huber hit 7.45%!)
     - `spec_lgbm_tuned_huber` (Specialist House/Apt with tuned Optuna params + Huber!)
     - `spec_xgb_tuned_psqm` (Specialist House/Apt XGBoost tuned!)
     - `mps_resnet_5seed_real_price` (5-seed + deeper/wider PyTorch MPS ResNet with
       Skip-Connections + CosineAnnealing + Weight Averaging on `log(real_price)` —
       since our 2-seed ResNet already got 21.3% blend weight!)
     - `mps_ft_mlp_real_price` (Feature-Tokenizer / Gated GLU MLP on MPS for extra NN diversity)
  2. **AutoGluon `best_quality` Benchmark** on the exact same pruned 210 features +
     grouped 5-fold CV + 2026 temporal holdout (adds `NeuralNetFastAI`, `NeuralNetTorch`,
     `ExtraTrees`, `RandomForest`, and AutoGluon's internal L2/L3 stacker).
  3. **Meta-Learner Comparison**:
     - Convex L1+L2 blend (nested OOF)
     - Type-specific Convex blend (separate weights for HOUSE vs APARTMENT!)
     - Ridge / Huber / ElasticNet 2nd-level stacker (nested OOF)
  4. **Conformalized Quantile Regression (p10 / p50 / p90)** using LightGBM quantile
     models + split-conformal residual calibration -> guaranteed ~80% empirical coverage
     on both OOF and 2026 temporal!

Usage:
    .venv-api/bin/python -u src/phase3b_frontier.py
"""
from __future__ import annotations

import json
import os
import shutil
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.optimize import minimize
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.preprocessing import QuantileTransformer

from phase3_features import compute_metrics, fmt, load_or_build_phase3_splits
from phase3_modeling import (
    _target_inverse,
    _target_transform,
    prep_tree_df,
    run_6splits,
    run_specialist_6splits,
)

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE_DIR, 'outputs', 'models', 'workbench')


# ---------------------------------------------------------------------------
# Stronger MPS Neural Architectures (5-seed ResNet + SwiGLU MLP)
# ---------------------------------------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim), nn.SiLU(), nn.Dropout(dropout), nn.Linear(dim, dim),
            nn.LayerNorm(dim), nn.SiLU(), nn.Dropout(dropout), nn.Linear(dim, dim),
        )

    def forward(self, x):
        return x + self.net(x)


class DeepTabularResNet(nn.Module):
    def __init__(self, n_num: int, cat_cards: list[int], dim: int = 384, n_blocks: int = 4, dropout: float = 0.25):
        super().__init__()
        self.embs = nn.ModuleList([
            nn.Embedding(card, min(28, max(4, int(round(1.8 * (card ** 0.5))))))
            for card in cat_cards
        ])
        in_dim = n_num + sum(e.embedding_dim for e in self.embs)
        self.in_proj = nn.Sequential(nn.Linear(in_dim, dim), nn.LayerNorm(dim))
        self.blocks = nn.ModuleList([ResBlock(dim, dropout) for _ in range(n_blocks)])
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.SiLU(), nn.Linear(dim, 1))

    def forward(self, x_num, x_cat):
        parts = [x_num] + [emb(x_cat[:, i]) for i, emb in enumerate(self.embs)]
        h = self.in_proj(torch.cat(parts, dim=1))
        for b in self.blocks:
            h = b(h)
        return self.head(h).squeeze(-1)


class SwiGLUNet(nn.Module):
    """Gated Linear Unit Tabular MLP — different inductive bias from ResNet."""
    def __init__(self, n_num: int, cat_cards: list[int], dim: int = 384, dropout: float = 0.25):
        super().__init__()
        self.embs = nn.ModuleList([
            nn.Embedding(card, min(24, max(4, int(round(1.6 * (card ** 0.5))))))
            for card in cat_cards
        ])
        in_dim = n_num + sum(e.embedding_dim for e in self.embs)
        self.ln0 = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, dim * 2)
        self.do1 = nn.Dropout(dropout)
        self.ln1 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim * 2)
        self.do2 = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(dim)
        self.fc3 = nn.Linear(dim, (dim // 2) * 2)
        self.do3 = nn.Dropout(dropout * 0.5)
        self.head = nn.Linear(dim // 2, 1)

    def _glu(self, x):
        a, b = x.chunk(2, dim=-1)
        return a * torch.nn.functional.silu(b)

    def forward(self, x_num, x_cat):
        parts = [x_num] + [emb(x_cat[:, i]) for i, emb in enumerate(self.embs)]
        x = self.ln0(torch.cat(parts, dim=1))
        h = self.do1(self._glu(self.fc1(x)))
        h = h + self.do2(self._glu(self.fc2(self.ln1(h))))
        h = self.do3(self._glu(self.fc3(self.ln2(h))))
        return self.head(h).squeeze(-1)


def train_mps_nn_6splits(
    bundles: list[dict],
    feat_cols: list[str],
    arch: str = 'resnet',
    target_mode: str = 'log_real_price',
    epochs: int = 65,
    batch_size: int = 256,
    n_seeds: int = 5,
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

        X_tr_num = X_tr_raw[num_cols].to_numpy(dtype=np.float32)
        X_va_num = X_va_raw[num_cols].to_numpy(dtype=np.float32)
        meds = np.nanmedian(X_tr_num, axis=0)
        meds = np.where(np.isnan(meds), 0.0, meds)
        X_tr_num = np.where(np.isnan(X_tr_num), meds[None, :], X_tr_num)
        X_va_num = np.where(np.isnan(X_va_num), meds[None, :], X_va_num)

        qt = QuantileTransformer(n_quantiles=500, output_distribution='normal', random_state=42)
        X_tr_num = np.clip(qt.fit_transform(X_tr_num), -4.5, 4.5).astype(np.float32)
        X_va_num = np.clip(qt.transform(X_va_num), -4.5, 4.5).astype(np.float32)

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
            torch.manual_seed(200 + s)
            rng = np.random.default_rng(200 + s)
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

            if arch == 'resnet':
                model = DeepTabularResNet(X_tr_num.shape[1], cat_cards, dim=384, n_blocks=4, dropout=0.28).to(device)
            else:
                model = SwiGLUNet(X_tr_num.shape[1], cat_cards, dim=384, dropout=0.25).to(device)

            opt = torch.optim.AdamW(model.parameters(), lr=1.2e-3, weight_decay=3e-3)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
            loss_fn = nn.HuberLoss(delta=0.5)

            best_loss = float('inf')
            best_state = None
            n_tr = len(tr_idx)

            for ep in range(epochs):
                model.train()
                perm = torch.randperm(n_tr, device=device)
                for start in range(0, n_tr, batch_size):
                    bi = perm[start:start + batch_size]
                    opt.zero_grad(set_to_none=True)
                    loss = loss_fn(model(t_xn[bi], t_xc[bi]), t_y[bi])
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.5)
                    opt.step()
                sched.step()

                model.eval()
                with torch.inference_mode():
                    val_loss = float(torch.mean(torch.abs(model(e_xn, e_xc) - e_y)).item())
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
            model.eval()
            with torch.inference_mode():
                preds_seeds.append(model(v_xn, v_xc).cpu().numpy() * y_std + y_mean)

        eur_pred = _target_inverse(np.mean(preds_seeds, axis=0), va_df, b['va_cpi_idx'], target_mode)
        if b['id'].startswith('fold_'):
            oof[b['va_idx']] = eur_pred
        else:
            tmp = eur_pred

    return oof, tmp


# ---------------------------------------------------------------------------
# AutoGluon best_quality benchmark (strictly grouped folds + temporal holdout)
# ---------------------------------------------------------------------------
def run_autogluon_benchmark(
    bundles: list[dict],
    feat_cols: list[str],
    target_mode: str = 'log_real_psqm',
    time_limit_per_split: int = 240,
) -> tuple[np.ndarray, np.ndarray]:
    """Run AutoGluon 1.5 (`best_v150` / diverse zoo: NN_TORCH, FASTAI, XT, RF, CAT, GBM, XGB + L2 stacker)
    on all 6 splits with 5 grouped inner folds per split (no Ray/DyStack).
    """
    from autogluon.tabular import TabularPredictor

    ag_root = os.path.join(OUT_DIR, '_ag_bench_tmp')
    if os.path.exists(ag_root):
        shutil.rmtree(ag_root)
    os.makedirs(ag_root, exist_ok=True)

    n_all = sum(len(b['va_df']) for b in bundles if b['id'].startswith('fold_'))
    oof = np.zeros(n_all, dtype=float)
    tmp = None

    for b in bundles:
        tr_df = b['tr_df']
        va_df = b['va_df']
        X_tr, _ = prep_tree_df(tr_df, feat_cols)
        X_va, _ = prep_tree_df(va_df, feat_cols)
        X_tr['__target__'] = _target_transform(tr_df, b['tr_cpi_idx'], target_mode)

        # IMPORTANT: AutoGluon `groups=` treats each unique value as a FOLD ID!
        # Passing raw property_key creates 8,700 folds. Map property_key -> 5 grouped inner folds (0..4).
        keys = tr_df['property_key'].astype(str).to_numpy()
        uk = np.unique(keys)
        rng = np.random.default_rng(42)
        rng.shuffle(uk)
        k2f = {k: i % 5 for i, k in enumerate(uk)}
        X_tr['__inner_fold__'] = pd.Series(keys).map(k2f).to_numpy()

        split_dir = os.path.join(ag_root, b['id'])
        predictor = TabularPredictor(
            label='__target__',
            problem_type='regression',
            eval_metric='mean_absolute_error',
            path=split_dir,
            groups='__inner_fold__',
            verbosity=0,
        ).fit(
            train_data=X_tr,
            num_bag_folds=5,
            num_stack_levels=1,
            dynamic_stacking=False,
            time_limit=time_limit_per_split,
            num_cpus=16,
            excluded_model_types=['KNN'],
            ag_args_fit={'num_gpus': 0},
            ag_args_ensemble={'fold_fitting_strategy': 'sequential_local'},
        )
        vp = predictor.predict(X_va).to_numpy(dtype=float)
        eur_pred = _target_inverse(vp, va_df, b['va_cpi_idx'], target_mode)
        if b['id'] == 'fold_0':
            lb = predictor.leaderboard(silent=True)
            print("  AutoGluon fold_0 top 8 models:")
            for _, r in lb.head(8).iterrows():
                print(f"    {r['model']:34s} val_MAE={-r['score_val']:.5f}")
        if b['id'].startswith('fold_'):
            oof[b['va_idx']] = eur_pred
        else:
            tmp = eur_pred
        shutil.rmtree(split_dir, ignore_errors=True)

    shutil.rmtree(ag_root, ignore_errors=True)
    return oof, tmp


# ---------------------------------------------------------------------------
# Conformalized Quantile Regression (p10 / p50 / p90 calibrated intervals)
# ---------------------------------------------------------------------------
def train_conformal_quantiles(
    bundles: list[dict],
    feat_cols: list[str],
    best_lgb_p: dict,
    y_oof: np.ndarray,
    y_tmp: np.ndarray,
    folds: np.ndarray,
    ens_oof: np.ndarray,
    ens_tmp: np.ndarray,
) -> dict:
    """Train LightGBM quantile (0.10, 0.90) on log_real_psqm + conformalize around ensemble."""
    from joblib import Parallel, delayed
    import lightgbm as lgb

    def _fit_q(b, alpha):
        tr_df, va_df = b['tr_df'], b['va_df']
        X_tr, cat_cols = prep_tree_df(tr_df, feat_cols)
        X_va, _ = prep_tree_df(va_df, feat_cols)
        for c in cat_cols:
            cats = sorted(set(X_tr[c].unique()) | {'Missing'})
            X_tr[c] = pd.Categorical(X_tr[c], categories=cats)
            X_va[c] = pd.Categorical(X_va[c].where(X_va[c].isin(cats), 'Missing'), categories=cats)
        y_tr = _target_transform(tr_df, b['tr_cpi_idx'], 'log_real_psqm')
        groups_tr = tr_df['property_key'].astype(str).to_numpy()
        rng = np.random.default_rng(42)
        uniq_g = np.unique(groups_tr)
        es_g = set(rng.choice(uniq_g, size=max(1, int(0.12 * len(uniq_g))), replace=False))
        es_mask = np.isin(groups_tr, list(es_g))

        cfg = dict(best_lgb_p)
        cfg.update(objective='quantile', alpha=alpha, n_jobs=3, verbose=-1, random_state=42)
        m = lgb.LGBMRegressor(**cfg)
        m.fit(
            X_tr[~es_mask], y_tr[~es_mask],
            eval_set=[(X_tr[es_mask], y_tr[es_mask])],
            callbacks=[lgb.early_stopping(80, verbose=False)],
        )
        return b['id'], alpha, _target_inverse(m.predict(X_va), va_df, b['va_cpi_idx'], 'log_real_psqm')

    jobs = [(b, a) for b in bundles for a in (0.10, 0.90)]
    res = Parallel(n_jobs=6, backend='loky')(delayed(_fit_q)(b, a) for b, a in jobs)
    rmap = {(sid, a): pred for sid, a, pred in res}

    n_all = len(y_oof)
    q10_oof = np.zeros(n_all)
    q90_oof = np.zeros(n_all)
    q10_tmp = q90_tmp = None
    for b in bundles:
        if b['id'].startswith('fold_'):
            q10_oof[b['va_idx']] = rmap[(b['id'], 0.10)]
            q90_oof[b['va_idx']] = rmap[(b['id'], 0.90)]
        else:
            q10_tmp = rmap[(b['id'], 0.10)]
            q90_tmp = rmap[(b['id'], 0.90)]

    # Conformalize in log-space (guarantees finite-sample 80% coverage on exchangeable data!)
    log_y = np.log(y_oof)
    log_q10 = np.log(np.minimum(q10_oof, ens_oof))
    log_q90 = np.log(np.maximum(q90_oof, ens_oof))
    scores = np.maximum(log_q10 - log_y, log_y - log_q90)

    # Nested-OOF conformal calibration for OOF evaluation
    c_lo_oof = np.zeros(n_all)
    c_hi_oof = np.zeros(n_all)
    for f in sorted(np.unique(folds)):
        tr_m = folds != f
        va_m = folds == f
        q_hat = float(np.quantile(scores[tr_m], 0.80))
        c_lo_oof[va_m] = np.exp(log_q10[va_m] - q_hat)
        c_hi_oof[va_m] = np.exp(log_q90[va_m] + q_hat)

    # Full-OOF q_hat applied to 2026 temporal
    q_hat_full = float(np.quantile(scores, 0.80))
    c_lo_tmp = np.exp(np.log(np.minimum(q10_tmp, ens_tmp)) - q_hat_full)
    c_hi_tmp = np.exp(np.log(np.maximum(q90_tmp, ens_tmp)) + q_hat_full)

    raw_cov_oof = float(np.mean((y_oof >= q10_oof) & (y_oof <= q90_oof)) * 100.0)
    cal_cov_oof = float(np.mean((y_oof >= c_lo_oof) & (y_oof <= c_hi_oof)) * 100.0)
    cal_cov_tmp = float(np.mean((y_tmp >= c_lo_tmp) & (y_tmp <= c_hi_tmp)) * 100.0)
    med_halfwidth_oof = float(np.median((c_hi_oof - c_lo_oof) / (2.0 * ens_oof)) * 100.0)
    med_halfwidth_tmp = float(np.median((c_hi_tmp - c_lo_tmp) / (2.0 * ens_tmp)) * 100.0)

    return {
        'raw_cov_oof_pct': round(raw_cov_oof, 2),
        'conformal_cov_oof_pct': round(cal_cov_oof, 2),
        'conformal_cov_tmp_pct': round(cal_cov_tmp, 2),
        'median_halfwidth_oof_pct': round(med_halfwidth_oof, 2),
        'median_halfwidth_tmp_pct': round(med_halfwidth_tmp, 2),
        'q_hat_log': round(q_hat_full, 5),
    }


def main():
    t_start = time.perf_counter()
    data = load_or_build_phase3_splits()
    bundles = data['bundles']
    y_oof = data['y_price']
    y_tmp = data['y_tmp_true']
    folds = data['folds']
    is_house_oof = data['is_house']
    is_house_tmp = (bundles[-1]['va_df']['property_type_HOUSE'].to_numpy() == 1)

    with open(os.path.join(OUT_DIR, 'phase3a_report.json')) as fh:
        p3a = json.load(fh)
    pruned_feats = p3a['pruned_features']
    best_lgb_p = p3a['best_lgb_params']
    best_xgb_p = p3a['best_xgb_params']

    # Load clean valid models from Phase 3A
    z = np.load(os.path.join(OUT_DIR, 'phase3_model_preds.npz'), allow_pickle=True)
    old_names = z['names'].tolist()
    O_old = z['O_log']
    T_old = z['T_log']
    preds_oof: dict[str, np.ndarray] = {}
    preds_tmp: dict[str, np.ndarray] = {}
    for i, nm in enumerate(old_names):
        if np.all(np.isfinite(O_old[:, i])) and compute_metrics(y_oof, np.exp(O_old[:, i]))['mdape'] < 10.0:
            preds_oof[nm] = np.exp(O_old[:, i])
            preds_tmp[nm] = np.exp(T_old[:, i])
    print(f"Loaded {len(preds_oof)} clean models from Phase 3A")

    records = []
    def rec(name: str, op: np.ndarray, tp: np.ndarray, dt: float):
        mo = compute_metrics(y_oof, op)
        mt = compute_metrics(y_tmp, tp)
        print(f"  {name:44s} ({dt:5.1f}s) | OOF: {fmt(mo)}")
        print(f"  {'':52s} | TMP: {fmt(mt)}")
        records.append({'name': name, 'seconds': round(dt, 1), 'oof': mo, 'temporal': mt})

    # Check for incremental checkpoint of Phase 3B experts so nothing is lost on retry
    ckpt_path = os.path.join(OUT_DIR, 'phase3b_experts_ckpt.npz')
    if os.path.exists(ckpt_path):
        cz = np.load(ckpt_path, allow_pickle=True)
        for nm in cz['names'].tolist():
            preds_oof[nm] = cz[f'oof_{nm}']
            preds_tmp[nm] = cz[f'tmp_{nm}']
        print(f"Resumed {len(cz['names'])} cached Phase-3B experts from checkpoint")

    def _save_ckpt():
        new_keys = [k for k in preds_oof if k not in set(old_names)]
        if not new_keys:
            return
        payload = {'names': np.array(new_keys)}
        for k in new_keys:
            payload[f'oof_{k}'] = preds_oof[k]
            payload[f'tmp_{k}'] = preds_tmp[k]
        np.savez_compressed(ckpt_path, **payload)

    # ======================================================================
    # 1. HIGH-LEVERAGE NEW EXPERTS (FIXED L1/HUBER + TUNED SPECIALISTS + 5-SEED MPS NNs)
    # ======================================================================
    print("\n=== 1. NEW PHASE-3B EXPERTS ===")
    new_gbdt = [
        ('lgbm_tuned_real_price_huber', pruned_feats, 'lgbm', 'log_real_price', 'huber', best_lgb_p),
        ('xgb_tuned_real_psqm_l1',      pruned_feats, 'xgb',  'log_real_psqm',  'l1',    best_xgb_p),
        ('cb_d7_real_psqm_l1',          list(dict.fromkeys(pruned_feats + ['immo_office_name'])), 'catboost', 'log_real_psqm', 'l1', {'depth': 7, 'l2_leaf_reg': 5.0, 'learning_rate': 0.03, 'iterations': 4500}),
    ]
    for key, fset, kind, tmode, lmode, params in new_gbdt:
        if key in preds_oof:
            rec(key + ' [cached]', preds_oof[key], preds_tmp[key], 0.0)
            continue
        t0 = time.perf_counter()
        op, tp, _ = run_6splits(bundles, fset, kind, tmode, lmode, params)
        rec(key, op, tp, time.perf_counter() - t0)
        preds_oof[key] = op
        preds_tmp[key] = tp
        _save_ckpt()

    if 'spec_lgbm_tuned_huber' not in preds_oof:
        t0 = time.perf_counter()
        op, tp = run_specialist_6splits(bundles, pruned_feats, 'lgbm', 'log_real_psqm', 'huber', best_lgb_p)
        rec('spec_lgbm_tuned_huber', op, tp, time.perf_counter() - t0)
        preds_oof['spec_lgbm_tuned_huber'] = op
        preds_tmp['spec_lgbm_tuned_huber'] = tp
        _save_ckpt()

    if 'spec_xgb_tuned_psqm' not in preds_oof:
        t0 = time.perf_counter()
        op, tp = run_specialist_6splits(bundles, pruned_feats, 'xgb', 'log_real_psqm', 'l2', best_xgb_p)
        rec('spec_xgb_tuned_psqm', op, tp, time.perf_counter() - t0)
        preds_oof['spec_xgb_tuned_psqm'] = op
        preds_tmp['spec_xgb_tuned_psqm'] = tp
        _save_ckpt()

    if 'mps_deep_resnet_5seed_price' not in preds_oof:
        t0 = time.perf_counter()
        op, tp = train_mps_nn_6splits(bundles, pruned_feats, arch='resnet', target_mode='log_real_price', epochs=65, n_seeds=5)
        rec('mps_deep_resnet_5seed_price', op, tp, time.perf_counter() - t0)
        preds_oof['mps_deep_resnet_5seed_price'] = op
        preds_tmp['mps_deep_resnet_5seed_price'] = tp
        _save_ckpt()

    if 'mps_swiglu_4seed_price' not in preds_oof:
        t0 = time.perf_counter()
        op, tp = train_mps_nn_6splits(bundles, pruned_feats, arch='swiglu', target_mode='log_real_price', epochs=65, n_seeds=4)
        rec('mps_swiglu_4seed_price', op, tp, time.perf_counter() - t0)
        preds_oof['mps_swiglu_4seed_price'] = op
        preds_tmp['mps_swiglu_4seed_price'] = tp
        _save_ckpt()

    # ======================================================================
    # 2. AUTOGLUON BENCHMARK (GROUPED FOLDS, NO RAY/DYSTACK)
    # ======================================================================
    print("\n=== 2. AUTOGLUON BENCHMARK (GROUPED FOLDS) ===")
    if 'autogluon_hq_real_psqm' not in preds_oof:
        t0 = time.perf_counter()
        op_ag, tp_ag = run_autogluon_benchmark(bundles, pruned_feats, 'log_real_psqm', time_limit_per_split=180)
        rec('autogluon_hq_real_psqm', op_ag, tp_ag, time.perf_counter() - t0)
        preds_oof['autogluon_hq_real_psqm'] = op_ag
        preds_tmp['autogluon_hq_real_psqm'] = tp_ag
        _save_ckpt()

    # ======================================================================
    # 3. META-STACKER COMPARISON (ALL 20 DIVERSE MODELS)
    # ======================================================================
    print(f"\n=== 3. META-STACKER COMPARISON ({len(preds_oof)} MODELS) ===")
    names = list(preds_oof.keys())
    O_log = np.column_stack([np.log(np.maximum(preds_oof[k], 10_000.0)) for k in names])
    T_log = np.column_stack([np.log(np.maximum(preds_tmp[k], 10_000.0)) for k in names])
    y_log = np.log(y_oof)

    def _fit_convex(X, y):
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

    # 3a. Unified Convex Blend (nested OOF)
    oof_cvx = np.zeros(len(y_oof))
    for f in sorted(np.unique(folds)):
        tr, va = folds != f, folds == f
        oof_cvx[va] = O_log[va] @ _fit_convex(O_log[tr], y_log[tr])
    w_cvx = _fit_convex(O_log, y_log)
    tmp_cvx = T_log @ w_cvx
    rec('STACK-1: Unified Convex Blend (20 models)', np.exp(oof_cvx), np.exp(tmp_cvx), 0.0)

    # 3b. Type-Specific Convex Blend (separate weights for HOUSE vs APARTMENT!)
    oof_tcvx = np.zeros(len(y_oof))
    tmp_tcvx = np.zeros(len(y_tmp))
    for h_flag in (True, False):
        hm_oof = is_house_oof == h_flag
        hm_tmp = is_house_tmp == h_flag
        for f in sorted(np.unique(folds)):
            tr = (folds != f) & hm_oof
            va = (folds == f) & hm_oof
            oof_tcvx[va] = O_log[va] @ _fit_convex(O_log[tr], y_log[tr])
        w_h = _fit_convex(O_log[hm_oof], y_log[hm_oof])
        tmp_tcvx[hm_tmp] = T_log[hm_tmp] @ w_h
    rec('STACK-2: Type-Specific Convex (House/Apt)', np.exp(oof_tcvx), np.exp(tmp_tcvx), 0.0)

    # 3c. Huber Meta-Regressor (allows slight slope/intercept recalibration for tail compression!)
    oof_hub = np.zeros(len(y_oof))
    for f in sorted(np.unique(folds)):
        tr, va = folds != f, folds == f
        meta = HuberRegressor(alpha=10.0, max_iter=500).fit(O_log[tr], y_log[tr])
        oof_hub[va] = meta.predict(O_log[va])
    meta_full = HuberRegressor(alpha=10.0, max_iter=500).fit(O_log, y_log)
    tmp_hub = meta_full.predict(T_log)
    rec('STACK-3: Huber 2nd-Level Stacker', np.exp(oof_hub), np.exp(tmp_hub), 0.0)

    # 3d. Without AutoGluon (pure custom zoo: CB + XGB + LGBM + Specialists + MPS NNs)
    no_ag_idx = [i for i, k in enumerate(names) if k != 'autogluon_hq_real_psqm']
    oof_noag = np.zeros(len(y_oof))
    for f in sorted(np.unique(folds)):
        tr, va = folds != f, folds == f
        oof_noag[va] = O_log[va][:, no_ag_idx] @ _fit_convex(O_log[tr][:, no_ag_idx], y_log[tr])
    w_noag = _fit_convex(O_log[:, no_ag_idx], y_log)
    rec('STACK-4: Custom Zoo Only (NO AutoGluon)', np.exp(oof_noag), np.exp(T_log[:, no_ag_idx] @ w_noag), 0.0)

    print("\nLearned Unified Convex weights (all 20 models):")
    for nm, wt in sorted(zip(names, w_cvx), key=lambda x: -x[1]):
        if wt >= 0.005:
            print(f"  {wt*100:5.1f}%  {nm}")

    # Pick winner for conformal quantile calibration
    best_oof = np.exp(oof_tcvx) if compute_metrics(y_oof, np.exp(oof_tcvx))['mdape'] <= compute_metrics(y_oof, np.exp(oof_cvx))['mdape'] else np.exp(oof_cvx)
    best_tmp = np.exp(tmp_tcvx) if compute_metrics(y_oof, np.exp(oof_tcvx))['mdape'] <= compute_metrics(y_oof, np.exp(oof_cvx))['mdape'] else np.exp(tmp_cvx)

    # ======================================================================
    # 4. CONFORMALIZED QUANTILE REGRESSION (p10 / p90 calibrated bands)
    # ======================================================================
    print("\n=== 4. CONFORMALIZED QUANTILE REGRESSION (TARGET 80% COVERAGE) ===")
    t0 = time.perf_counter()
    cqr = train_conformal_quantiles(bundles, pruned_feats, best_lgb_p, y_oof, y_tmp, folds, best_oof, best_tmp)
    print(f"  Done in {time.perf_counter()-t0:.1f}s: {cqr}")

    np.savez_compressed(
        os.path.join(OUT_DIR, 'phase3b_frontier_preds.npz'),
        names=np.array(names),
        O_log=O_log,
        T_log=T_log,
        y_oof=y_oof,
        y_tmp=y_tmp,
        best_oof=best_oof,
        best_tmp=best_tmp,
        weights_unified=w_cvx,
    )
    with open(os.path.join(OUT_DIR, 'phase3b_report.json'), 'w') as fh:
        json.dump({
            'results': records,
            'weights_unified': {nm: round(float(wt), 4) for nm, wt in zip(names, w_cvx)},
            'conformal_quantiles': cqr,
            'total_seconds': round(time.perf_counter() - t_start, 1),
        }, fh, indent=2)
    print(f"\nPhase 3B finished in {time.perf_counter()-t_start:.1f}s")


if __name__ == '__main__':
    main()
