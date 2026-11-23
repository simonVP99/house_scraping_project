"""Phase 5B — Tier 2 (Clever 2026 Text & Residual-Guided Feature Engineering):
Combines our brand-new MLX `Qwen3-Embedding-0.6B-4bit-DWQ` (1024-d causal LLM embeddings)
with `E5-large` (1024-d bidirectional encoder) and `Word+Char TF-IDF` using two clever
strictly nested-OOF ideas:

  1. **Multi-Backbone Semantic Signal** (`Qwen3-Emb` + `E5-large`):
     - Nested-OOF Ridge on `Qwen3-Emb` alone (alpha 5 & 25) -> `txt_qwen3_a5_psqm`, `txt_qwen3_a25_psqm`, `txt_qwen3_a5_price`, `txt_qwen3_a5_resid`
     - Nested-OOF Ridge on concatenated `[Qwen3-Emb | E5-large]` (2048-d) -> `txt_qwen_e5l_psqm`, `txt_qwen_e5l_price`
     - Fold-safe `Qwen3-Emb` PCA-32 (`qwen3_pca_0..31`)
     - Cross-backbone disagreement: `txt_qwen3_minus_e5l_psqm`, `txt_qwen3_minus_tfidf_psqm`

  2. **Stage-0 Tabular-Residual Guided Text Learning (THE CLEVER TRICK!)**:
     - Inside each split's training set, fit a fast strictly-inner-OOF structured-only LightGBM
       (NO text features!) to get `y_tab_hat`.
     - Compute the exact **tabular residual** `r_tab = log_real_psqm - y_tab_hat` — everything
       the tabular features CANNOT explain (finish quality, light, architect design, defects,
       micro-street charm, starting-price phrasing).
     - Train BOTH:
       (a) Nested-OOF Ridge on `r_tab` using `Qwen3-Emb` (`txt_qwen3_tabresid`), `E5-large`
           (`txt_e5l_tabresid`), `[Qwen3|E5l]` (`txt_dense_tabresid`), and `Word+Char TF-IDF`
           (`txt_tfidf_tabresid`)!
       (b) A fast 2-layer PyTorch MPS **Multi-Task Text MLP** on `[Qwen3-Emb | E5-large]` (2048-d)
           jointly predicting `[log_real_psqm, log_real_price, r_tab]` -> `txt_mlp_psqm`,
           `txt_mlp_price`, `txt_mlp_tabresid`!

  3. **Agency Name Casing Cleanup**:
     - Normalizes the 8 split-casing Ghent agencies (`TREVI Gent`/`Trevi Gent`, `Vicus`/`VICUS`,
       `LANDBERGH`/`Landbergh`, etc.) before target & frequency encoding.

Saves a lightweight add-on dictionary `outputs/models/workbench/phase5_addon_features.joblib`
keyed by split id (`fold_0..4`, `temporal`) so every Phase-5 model joins it onto
`phase4_splits_cache.joblib` in <1 second without duplicating 66 MB of base data!

Usage:
    .venv-api/bin/python -u src/phase5_qwen_features.py
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.sparse import hstack
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'src'))

from phase3_features import compute_metrics, fmt

WORK_DIR = os.path.join(BASE_DIR, 'outputs', 'models', 'workbench')
CACHE_P4 = os.path.join(WORK_DIR, 'phase4_splits_cache.joblib')
QWEN3_PATH = os.path.join(WORK_DIR, 'qwen3_emb_14599.npz')
E5L_PATH = os.path.join(WORK_DIR, 'e5_large_14599.npz')
RAW_PARQUET = os.path.join(BASE_DIR, 'outputs', 'property_data_full.parquet')
DATASET_PATH = os.path.join(BASE_DIR, 'outputs', 'dataset', 'training_dataset.parquet')
OUT_ADDON = os.path.join(WORK_DIR, 'phase5_addon_features.joblib')


def _inner_group_folds(groups: np.ndarray, n_splits: int = 5, seed: int = 101) -> np.ndarray:
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(uniq)
    g2f = {g: i % n_splits for i, g in enumerate(perm)}
    return np.array([g2f[g] for g in groups], dtype=int)


def _compute_tabular_anchor_oof(tr_df: pd.DataFrame, va_df: pd.DataFrame, y_psqm_tr: np.ndarray, inner_f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fast strictly-inner-OOF structured-only LightGBM anchor (NO text features)."""
    struct_cols = [
        c for c in tr_df.columns
        if c not in ('property_id', 'date_obtained', 'property_key', 'fold', 'price', 'street_norm', 'immo_office_name')
        and not c.startswith(('txt_', 'e5_', 'e5l_', 'tfidf_svd_', 'qwen3_'))
    ]
    X_tr = tr_df[struct_cols].copy()
    X_va = va_df[struct_cols].copy()
    for c in struct_cols:
        if X_tr[c].dtype == 'bool':
            X_tr[c] = X_tr[c].astype(np.int8)
            X_va[c] = X_va[c].astype(np.int8)
        elif X_tr[c].dtype == 'object' or str(X_tr[c].dtype) in ('string', 'category'):
            cats = sorted(set(X_tr[c].fillna('Missing').astype(str).unique()) | {'Missing'})
            X_tr[c] = pd.Categorical(X_tr[c].fillna('Missing').astype(str), categories=cats)
            X_va[c] = pd.Categorical(X_va[c].fillna('Missing').astype(str).where(X_va[c].fillna('Missing').astype(str).isin(cats), 'Missing'), categories=cats)
        else:
            X_tr[c] = pd.to_numeric(X_tr[c], errors='coerce').astype(np.float32)
            X_va[c] = pd.to_numeric(X_va[c], errors='coerce').astype(np.float32)

    params = dict(
        n_estimators=450, learning_rate=0.04, num_leaves=47, min_child_samples=30,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_alpha=1.0, reg_lambda=5.0,
        random_state=42, n_jobs=12, verbose=-1,
    )
    oof_hat = np.zeros(len(tr_df), dtype=float)
    va_hats = []
    for k in range(5):
        m_tr = inner_f != k
        m_va = inner_f == k
        mod = lgb.LGBMRegressor(**params)
        mod.fit(X_tr.iloc[m_tr], y_psqm_tr[m_tr])
        oof_hat[m_va] = mod.predict(X_tr.iloc[m_va])
        va_hats.append(mod.predict(X_va))
    return oof_hat, np.mean(va_hats, axis=0)


class _MultiTaskTextMLP(nn.Module):
    def __init__(self, d_in: int, d_hid: int = 384, n_out: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Dropout(0.35),
            nn.Linear(d_in, d_hid),
            nn.SiLU(),
            nn.LayerNorm(d_hid),
            nn.Dropout(0.35),
            nn.Linear(d_hid, 128),
            nn.SiLU(),
            nn.Linear(128, n_out),
        )

    def forward(self, x):
        return self.net(x)


def _fit_multitask_text_mlp(
    X_tr: np.ndarray, Y_tr: np.ndarray, X_va: np.ndarray, inner_f: np.ndarray, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Strictly inner-OOF 2-layer Multi-Task Text MLP on MPS GPU predicting [psqm, price, tab_resid]."""
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    n_tr, d_in = X_tr.shape
    n_out = Y_tr.shape[1]
    oof_pred = np.zeros((n_tr, n_out), dtype=np.float32)
    va_preds = []

    X_va_t = torch.tensor(X_va, dtype=torch.float32, device=device)

    for k in range(5):
        m_tr = inner_f != k
        m_va = inner_f == k
        mu = Y_tr[m_tr].mean(axis=0)
        sd = np.maximum(Y_tr[m_tr].std(axis=0), 1e-6)
        y_tr_z = ((Y_tr[m_tr] - mu) / sd).astype(np.float32)
        y_va_z = ((Y_tr[m_va] - mu) / sd).astype(np.float32)

        torch.manual_seed(seed + k * 17)
        mod = _MultiTaskTextMLP(d_in, d_hid=384, n_out=n_out).to(device)
        opt = torch.optim.AdamW(mod.parameters(), lr=1.5e-3, weight_decay=2e-2)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20, eta_min=2e-5)
        loss_fn = nn.HuberLoss(delta=0.8)

        xt = torch.tensor(X_tr[m_tr], dtype=torch.float32, device=device)
        yt = torch.tensor(y_tr_z, dtype=torch.float32, device=device)
        xv = torch.tensor(X_tr[m_va], dtype=torch.float32, device=device)
        yv = torch.tensor(y_va_z, dtype=torch.float32, device=device)

        best_val = 1e9
        best_oof_k = None
        best_va_k = None
        bs = 256
        n_m = len(xt)
        rng = np.random.default_rng(seed + k)

        for ep in range(20):
            mod.train()
            perm = torch.tensor(rng.permutation(n_m), dtype=torch.long, device=device)
            for s in range(0, n_m, bs):
                idx = perm[s:s + bs]
                opt.zero_grad()
                loss = loss_fn(mod(xt[idx]), yt[idx])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(mod.parameters(), 2.0)
                opt.step()
            sched.step()
            mod.eval()
            with torch.no_grad():
                p_va_z = mod(xv)
                vloss = float(loss_fn(p_va_z, yv).item())
                if vloss < best_val:
                    best_val = vloss
                    best_oof_k = p_va_z.cpu().numpy() * sd + mu
                    best_va_k = mod(X_va_t).cpu().numpy() * sd + mu

        oof_pred[m_va] = best_oof_k
        va_preds.append(best_va_k)
        del mod, xt, yt, xv, yv
    if device == 'mps':
        torch.mps.empty_cache()
    return oof_pred, np.mean(va_preds, axis=0)


def fit_phase5_text_for_split(
    tr_df: pd.DataFrame,
    va_df: pd.DataFrame,
    tr_cpi: np.ndarray,
    tr_texts: list[str],
    va_texts: list[str],
    qwen_tr: np.ndarray,
    qwen_va: np.ndarray,
    e5l_tr: np.ndarray,
    e5l_va: np.ndarray,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute all Phase-5 Qwen3 + Tabular-Residual + Multi-Task Text MLP features for one split."""
    n_tr, n_va = len(tr_df), len(va_df)
    groups_tr = tr_df['property_key'].astype(str).to_numpy()
    inner_f = _inner_group_folds(groups_tr, n_splits=5, seed=seed)

    y_psqm = np.log((tr_df['price'].to_numpy(dtype=float) / tr_cpi) / tr_df['livable_area'].to_numpy(dtype=float))
    y_price = np.log(tr_df['price'].to_numpy(dtype=float) / tr_cpi)
    y_sp_anchor = np.log(np.maximum(tr_df['sp_same_type_idw_psqm_k15'].to_numpy(dtype=float), 500.0))
    y_sp_resid = y_psqm - y_sp_anchor

    # 1. Strictly-inner-OOF structured-only tabular anchor -> exact tabular residual!
    tab_hat_tr, tab_hat_va = _compute_tabular_anchor_oof(tr_df, va_df, y_psqm, inner_f)
    y_tab_resid = y_psqm - tab_hat_tr

    # 2. Vectorize TF-IDF (word 1-2g + char_wb 3-5g) strictly fit on tr_texts
    vw = TfidfVectorizer(max_features=35000, ngram_range=(1, 2), min_df=4, max_df=0.9, sublinear_tf=True)
    vc = TfidfVectorizer(analyzer='char_wb', max_features=30000, ngram_range=(3, 5), min_df=10, max_df=0.9, sublinear_tf=True)
    X_tf_tr = hstack([vw.fit_transform(tr_texts), vc.fit_transform(tr_texts)]).tocsr()
    X_tf_va = hstack([vw.transform(va_texts), vc.transform(va_texts)]).tocsr()

    dense_both_tr = np.hstack([qwen_tr, e5l_tr]).astype(np.float32)
    dense_both_va = np.hstack([qwen_va, e5l_va]).astype(np.float32)

    def nested_ridge(Xtr, Xva, y, alpha):
        oof = np.zeros(n_tr, dtype=np.float32)
        for k in range(5):
            m = Ridge(alpha=alpha, random_state=42)
            m.fit(Xtr[inner_f != k], y[inner_f != k])
            oof[inner_f == k] = m.predict(Xtr[inner_f == k])
        mf = Ridge(alpha=alpha, random_state=42)
        mf.fit(Xtr, y)
        return oof, mf.predict(Xva).astype(np.float32)

    tr_out, va_out = {}, {}
    tr_out['tab_anchor_log_psqm'] = tab_hat_tr.astype(np.float32)
    va_out['tab_anchor_log_psqm'] = tab_hat_va.astype(np.float32)

    # Qwen3-Embedding-0.6B Ridge signals
    for nm, Xtr, Xva, y, a in [
        ('txt_qwen3_a5_psqm',       qwen_tr,       qwen_va,       y_psqm,      5.0),
        ('txt_qwen3_a25_psqm',      qwen_tr,       qwen_va,       y_psqm,      25.0),
        ('txt_qwen3_a5_price',      qwen_tr,       qwen_va,       y_price,     5.0),
        ('txt_qwen3_a5_spresid',    qwen_tr,       qwen_va,       y_sp_resid,  8.0),
        ('txt_qwen3_tabresid',      qwen_tr,       qwen_va,       y_tab_resid, 12.0),
        ('txt_e5l_tabresid',        e5l_tr,        e5l_va,        y_tab_resid, 12.0),
        ('txt_dense_both_psqm',     dense_both_tr, dense_both_va, y_psqm,      10.0),
        ('txt_dense_both_price',    dense_both_tr, dense_both_va, y_price,     10.0),
        ('txt_dense_both_tabresid', dense_both_tr, dense_both_va, y_tab_resid, 20.0),
        ('txt_tfidf_tabresid_a6',   X_tf_tr,       X_tf_va,       y_tab_resid, 6.0),
        ('txt_tfidf_tabresid_a25',  X_tf_tr,       X_tf_va,       y_tab_resid, 25.0),
    ]:
        o, v = nested_ridge(Xtr, Xva, y, a)
        tr_out[nm] = o
        va_out[nm] = v

    # Multi-Task Text MLP on [Qwen3 | E5-large] (2048-d -> [psqm, price, tab_resid])
    Y_mt = np.column_stack([y_psqm, y_price, y_tab_resid]).astype(np.float32)
    mlp_tr, mlp_va = _fit_multitask_text_mlp(dense_both_tr, Y_mt, dense_both_va, inner_f, seed=seed)
    for j, nm in enumerate(('txt_mlp_psqm', 'txt_mlp_price', 'txt_mlp_tabresid')):
        tr_out[nm] = mlp_tr[:, j]
        va_out[nm] = mlp_va[:, j]

    # Super-blends & cross-backbone disagreements
    tr_out['txt_p5_super_psqm'] = (0.40 * tr_df['txt_tfidf_a2_psqm'].to_numpy(dtype=np.float32) + 0.30 * tr_out['txt_qwen3_a5_psqm'] + 0.30 * tr_out['txt_mlp_psqm'])
    va_out['txt_p5_super_psqm'] = (0.40 * va_df['txt_tfidf_a2_psqm'].to_numpy(dtype=np.float32) + 0.30 * va_out['txt_qwen3_a5_psqm'] + 0.30 * va_out['txt_mlp_psqm'])
    tr_out['txt_p5_super_tabresid'] = (0.40 * tr_out['txt_tfidf_tabresid_a6'] + 0.30 * tr_out['txt_dense_both_tabresid'] + 0.30 * tr_out['txt_mlp_tabresid'])
    va_out['txt_p5_super_tabresid'] = (0.40 * va_out['txt_tfidf_tabresid_a6'] + 0.30 * va_out['txt_dense_both_tabresid'] + 0.30 * va_out['txt_mlp_tabresid'])
    tr_out['txt_tab_plus_text_psqm'] = tr_out['tab_anchor_log_psqm'] + tr_out['txt_p5_super_tabresid']
    va_out['txt_tab_plus_text_psqm'] = va_out['tab_anchor_log_psqm'] + va_out['txt_p5_super_tabresid']
    tr_out['txt_qwen3_minus_e5l_psqm'] = tr_out['txt_qwen3_a5_psqm'] - tr_df['txt_e5l_a5_psqm'].to_numpy(dtype=np.float32)
    va_out['txt_qwen3_minus_e5l_psqm'] = va_out['txt_qwen3_a5_psqm'] - va_df['txt_e5l_a5_psqm'].to_numpy(dtype=np.float32)
    tr_out['txt_qwen3_minus_tfidf_psqm'] = tr_out['txt_qwen3_a5_psqm'] - tr_df['txt_tfidf_a2_psqm'].to_numpy(dtype=np.float32)
    va_out['txt_qwen3_minus_tfidf_psqm'] = va_out['txt_qwen3_a5_psqm'] - va_df['txt_tfidf_a2_psqm'].to_numpy(dtype=np.float32)

    # Fold-safe Qwen3 PCA-32
    pca = PCA(n_components=32, random_state=42)
    p_tr = pca.fit_transform(qwen_tr).astype(np.float32)
    p_va = pca.transform(qwen_va).astype(np.float32)
    for j in range(32):
        tr_out[f'qwen3_pca_{j}'] = p_tr[:, j]
        va_out[f'qwen3_pca_{j}'] = p_va[:, j]

    # Normalized agency casing (fixes the 8 split agencies!)
    tr_ag = tr_df['immo_office_name'].fillna('Unknown').astype(str).str.strip().str.lower()
    va_ag = va_df['immo_office_name'].fillna('Unknown').astype(str).str.strip().str.lower()
    vc_ag = tr_ag.value_counts()
    tr_out['freq_agency_norm'] = tr_ag.map(vc_ag).fillna(0).astype(np.float32).to_numpy()
    va_out['freq_agency_norm'] = va_ag.map(vc_ag).fillna(0).astype(np.float32).to_numpy()

    return pd.DataFrame(tr_out), pd.DataFrame(va_out)


def main():
    t0 = time.perf_counter()
    print("=== PHASE 5B (TIER 2): QWEN3-EMBEDDING + TABULAR-RESIDUAL TEXT + MULTI-TASK MLP ===", flush=True)
    data = joblib.load(CACHE_P4)
    bundles = data['bundles']
    y_price = data['y_price']
    y_tmp_true = data['y_tmp_true']

    # Align texts, Qwen3-Emb, and E5-large to training_dataset row order
    ds = pd.read_parquet(DATASET_PATH)[['property_id', 'date_obtained']].reset_index(drop=True)
    ds['property_id'] = ds['property_id'].astype(str)
    ds['date_dt'] = pd.to_datetime(ds['date_obtained'])
    raw = pd.read_parquet(RAW_PARQUET)[['property_id', 'date_obtained', 'title', 'description_nl', 'description']]
    raw['property_id'] = raw['property_id'].astype(str)
    raw['date_dt'] = pd.to_datetime(raw['date_obtained'], format='%d%m%Y', errors='coerce')
    raw_u = raw.drop_duplicates(['property_id', 'date_dt'], keep='first')
    m = ds.merge(raw_u, on=['property_id', 'date_dt'], how='left')
    texts_all = [
        (f"{str(t).strip()}. {str(nl).strip() if len(str(nl).strip())>=40 else str(d).strip()}" if str(t).strip() else (str(nl).strip() if len(str(nl).strip())>=40 else str(d).strip()))[:2400]
        for t, nl, d in zip(m['title'].fillna(''), m['description_nl'].fillna(''), m['description'].fillna(''))
    ]
    qwen_all = np.load(QWEN3_PATH)['embeddings'].astype(np.float32)
    e5l_all = np.load(E5L_PATH)['embeddings'].astype(np.float32)
    assert len(qwen_all) == len(ds) == len(e5l_all)

    addon_splits: dict[str, dict[str, pd.DataFrame]] = {}
    oof_probes: dict[str, np.ndarray] = {
        k: np.zeros(len(ds)) for k in (
            'e5l_only', 'qwen3_only', 'dense_both', 'tfidf_only', 'text_mlp', 'p5_super_text', 'tab_anchor_only', 'tab_plus_text_resid'
        )
    }

    for i, b in enumerate(bundles):
        sid = b['id']
        ts = time.perf_counter()
        tr_idx = b['tr_idx']
        va_idx = b['va_idx']
        tr_texts = [texts_all[j] for j in tr_idx]
        va_texts = [texts_all[j] for j in va_idx]
        tr_add, va_add = fit_phase5_text_for_split(
            b['tr_df'], b['va_df'], b['tr_cpi_idx'],
            tr_texts, va_texts,
            qwen_all[tr_idx], qwen_all[va_idx],
            e5l_all[tr_idx], e5l_all[va_idx],
            seed=42 + i * 13,
        )
        addon_splits[sid] = {'tr_add': tr_add, 'va_add': va_add}

        va_area = b['va_df']['livable_area'].to_numpy(dtype=float)
        va_cpi = b['va_cpi_idx']
        if sid != 'temporal':
            oof_probes['e5l_only'][va_idx] = np.exp(b['va_df']['txt_e5l_a5_psqm'].to_numpy()) * va_area * va_cpi
            oof_probes['qwen3_only'][va_idx] = np.exp(va_add['txt_qwen3_a5_psqm'].to_numpy()) * va_area * va_cpi
            oof_probes['dense_both'][va_idx] = np.exp(va_add['txt_dense_both_psqm'].to_numpy()) * va_area * va_cpi
            oof_probes['tfidf_only'][va_idx] = np.exp(b['va_df']['txt_tfidf_a2_psqm'].to_numpy()) * va_area * va_cpi
            oof_probes['text_mlp'][va_idx] = np.exp(va_add['txt_mlp_psqm'].to_numpy()) * va_area * va_cpi
            oof_probes['p5_super_text'][va_idx] = np.exp(va_add['txt_p5_super_psqm'].to_numpy()) * va_area * va_cpi
            oof_probes['tab_anchor_only'][va_idx] = np.exp(va_add['tab_anchor_log_psqm'].to_numpy()) * va_area * va_cpi
            oof_probes['tab_plus_text_resid'][va_idx] = np.exp(va_add['txt_tab_plus_text_psqm'].to_numpy()) * va_area * va_cpi
        print(f"  Built Phase-5 add-on ({tr_add.shape[1]} cols) for {sid:8s} in {time.perf_counter()-ts:.1f}s", flush=True)

    print("\n=== HEAD-TO-HEAD TEXT BACKBONE & TABULAR-RESIDUAL PROBES (5-FOLD GROUPED OOF) ===", flush=True)
    for k, pred in oof_probes.items():
        print(f"  {k:24s}: {fmt(compute_metrics(y_price, pred))}", flush=True)

    joblib.dump(addon_splits, OUT_ADDON, compress=3)
    print(f"\nSaved {OUT_ADDON} ({len(addon_splits)} splits x {tr_add.shape[1]} new cols) in {time.perf_counter()-t0:.1f}s!", flush=True)


if __name__ == '__main__':
    main()
