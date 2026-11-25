"""Issue #10 Research Experiment: Cross-Modal JEPA Latent-Residual Surprise.

Evaluates whether strictly fold-safe cross-modal latent prediction errors:
    Δz_photo = z_photo_actual − ẑ_photo_expected(Tabular + Spatial + Text)
    Δz_text  = z_text_actual  − ẑ_text_expected(Tabular + Spatial + Photo)
and derived energies / semantic surprises / intra-listing cherry-picking spreads:
  1. Improve Normal Ghent Homes (€150k–€800k) MdAPE on Cluster-Clean 5F OOF + 2026 Temporal
     on top of the frozen Phase-7 Lean-5 agency-blind multimodal features,
  2. Improve / sharpen 80% Heteroscedastic Conformal Intervals (#8),
  3. Provide a validated, buyer-facing "Honesty / Hidden-Gem / Photo-vs-Specs Mismatch"
     signal correlated with real price residuals, listing duration, and price cuts.

Run:
    PYTHONPATH=src .venv-api/bin/python -u src/research_cross_modal_jepa.py
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings

# macOS Apple Silicon safety: import torch before lightgbm
try:
    import torch  # noqa: F401
except Exception:
    pass

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGING_DIR = os.path.join(PROJECT_DIR, "outputs", "models", "workbench", "prod_train_staging")
PROD_DIR = os.path.join(PROJECT_DIR, "outputs", "models", "production")
WORK_DIR = os.path.join(PROJECT_DIR, "outputs", "models", "workbench")
VISION_NPZ = os.path.join(PROJECT_DIR, "external_data", "vision_full", "vision_full_embeddings.npz")
E5_NPZ = os.path.join(WORK_DIR, "e5_large_14599.npz")
HPARAMS_JSON = os.path.join(WORK_DIR, "phase7_gbdt_hpo.json")
OUT_REPORT = os.path.join(WORK_DIR, "issue10_jepa_report.json")
OUT_OOF_PARQUET = os.path.join(WORK_DIR, "issue10_jepa_oof.parquet")

Q_NAMES = [
    "modern_finish", "fixer_shell", "natural_light", "kitchen_luxury",
    "bath_luxury", "architect_charm", "facade_prestige", "floor_quality",
    "outdoor_green", "staging_pro",
]


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray | None = None) -> dict:
    if mask is not None:
        y_true = y_true[mask]
        y_pred = y_pred[mask]
    ape = np.abs(y_pred - y_true) / np.maximum(y_true, 1.0) * 100.0
    signed = (y_pred - y_true) / np.maximum(y_true, 1.0) * 100.0
    mae = float(np.mean(np.abs(y_pred - y_true)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return {
        "n": int(len(y_true)),
        "mdape": round(float(np.median(ape)), 4),
        "hit10": round(float(np.mean(ape <= 10.0) * 100.0), 3),
        "hit15": round(float(np.mean(ape <= 15.0) * 100.0), 3),
        "mae_eur": round(mae, 1),
        "r2": round(1.0 - ss_res / max(ss_tot, 1e-9), 5),
        "bias_pct": round(float(np.median(signed)), 3),
    }


def _build_numeric_matrix(tr_df: pd.DataFrame, va_df: pd.DataFrame, cols: list[str]):
    """Strict train-only median impute + z-score for Ridge JEPA predictors."""
    X_tr = tr_df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    X_va = va_df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    med = np.nanmedian(X_tr, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    tr_nan = ~np.isfinite(X_tr)
    va_nan = ~np.isfinite(X_va)
    if tr_nan.any():
        X_tr[tr_nan] = np.take(med, np.where(tr_nan)[1])
    if va_nan.any():
        X_va[va_nan] = np.take(med, np.where(va_nan)[1])
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr).astype(np.float32)
    X_va = scaler.transform(X_va).astype(np.float32)
    # Clip extreme outliers so Ridge stays well-conditioned
    np.clip(X_tr, -8.0, 8.0, out=X_tr)
    np.clip(X_va, -8.0, 8.0, out=X_va)
    return X_tr, X_va


def _compute_jepa_for_split(
    bundle: dict,
    tab_cols: list[str],
    sig_all: np.ndarray,
    so_all: np.ndarray,
    q_all: np.ndarray,
    so_q_all: np.ndarray,
    render_all: np.ndarray,
    has_vis_all: np.ndarray,
    e5_all: np.ndarray,
    y_price: np.ndarray,
    normal_mask: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Strictly fold-safe Cross-Modal JEPA latent predictors + surprise features."""
    tr_idx = bundle["tr_idx"]
    va_idx = bundle["va_idx"]
    tr_df = bundle["tr_df"]
    va_df = bundle["va_df"]

    tr_has = has_vis_all[tr_idx]
    va_has = has_vis_all[va_idx]

    # 1. Context matrix C = Tabular+Spatial+Text (strictly NO vis_* cols)
    C_tr, C_va = _build_numeric_matrix(tr_df, va_df, tab_cols)
    # Also include train-fit E5-PCA-32 directly inside context so predictor sees rich text semantics
    e5_pca = PCA(n_components=32, random_state=42)
    E5_tr = e5_pca.fit_transform(e5_all[tr_idx]).astype(np.float32)
    E5_va = e5_pca.transform(e5_all[va_idx]).astype(np.float32)
    e5_sc = StandardScaler()
    E5_tr_s = e5_sc.fit_transform(E5_tr).astype(np.float32)
    E5_va_s = e5_sc.transform(E5_va).astype(np.float32)

    Ctx_tr = np.hstack([C_tr, E5_tr_s])
    Ctx_va = np.hstack([C_va, E5_va_s])

    # 2. Target photo latent Z_photo = [SigLIP2 PCA-32, SO400M PCA-32] fit ONLY on train-with-photos
    sig_pca = PCA(n_components=32, random_state=42)
    so_pca = PCA(n_components=32, random_state=42)
    sig_pca.fit(sig_all[tr_idx][tr_has])
    so_pca.fit(so_all[tr_idx][tr_has])

    Z_sig_tr = sig_pca.transform(sig_all[tr_idx]).astype(np.float32)
    Z_sig_va = sig_pca.transform(sig_all[va_idx]).astype(np.float32)
    Z_so_tr = so_pca.transform(so_all[tr_idx]).astype(np.float32)
    Z_so_va = so_pca.transform(so_all[va_idx]).astype(np.float32)

    Z_ph_tr = np.hstack([Z_sig_tr, Z_so_tr])
    Z_ph_va = np.hstack([Z_sig_va, Z_so_va])
    z_sc = StandardScaler()
    z_sc.fit(Z_ph_tr[tr_has])
    Z_ph_tr_s = z_sc.transform(Z_ph_tr).astype(np.float32)
    Z_ph_va_s = z_sc.transform(Z_ph_va).astype(np.float32)

    # 3. Forward JEPA predictor f: (Tab+Spatial+Text) -> Ẑ_photo (RidgeCV, fit on tr_has)
    alphas = np.logspace(1.0, 4.5, 10)
    f_fwd = RidgeCV(alphas=alphas)
    f_fwd.fit(Ctx_tr[tr_has], Z_ph_tr_s[tr_has])

    # Use inner 5-fold OOF on train so train Δz doesn't overfit to Ridge train residual!
    tr_cids = bundle["tr_cids"]
    uniq_c = np.unique(tr_cids)
    # Deterministic 5-bucket hash by cluster id
    c_to_fold = {c: i % 5 for i, c in enumerate(uniq_c)}
    inner_f = np.array([c_to_fold[c] for c in tr_cids])

    Zhat_ph_tr = np.zeros_like(Z_ph_tr_s)
    best_alpha_fwd = float(f_fwd.alpha_)
    for k in range(5):
        m_in = (inner_f != k) & tr_has
        m_out = (inner_f == k)
        rk = Ridge(alpha=best_alpha_fwd)
        rk.fit(Ctx_tr[m_in], Z_ph_tr_s[m_in])
        Zhat_ph_tr[m_out] = rk.predict(Ctx_tr[m_out]).astype(np.float32)
    Zhat_ph_va = f_fwd.predict(Ctx_va).astype(np.float32)

    dZ_ph_tr = Z_ph_tr_s - Zhat_ph_tr
    dZ_ph_va = Z_ph_va_s - Zhat_ph_va

    # Forward R2 on validation-with-photos (diagnostic: how predictable are photos from metadata?)
    ss_res = float(np.sum((Z_ph_va_s[va_has] - Zhat_ph_va[va_has]) ** 2))
    ss_tot = float(np.sum((Z_ph_va_s[va_has] - Z_ph_va_s[va_has].mean(axis=0)) ** 2))
    fwd_r2 = 1.0 - ss_res / max(ss_tot, 1e-9)

    # 4. Semantic Q-score JEPA: predict 10 mean Q-scores + 10 SO400M Q-scores from Tab+Spatial+Text
    Q_tr = np.hstack([q_all[tr_idx, :10], so_q_all[tr_idx]]).astype(np.float32)
    Q_va = np.hstack([q_all[va_idx, :10], so_q_all[va_idx]]).astype(np.float32)
    q_sc = StandardScaler()
    q_sc.fit(Q_tr[tr_has])
    Q_tr_s = q_sc.transform(Q_tr).astype(np.float32)
    Q_va_s = q_sc.transform(Q_va).astype(np.float32)

    f_q = RidgeCV(alphas=alphas)
    f_q.fit(Ctx_tr[tr_has], Q_tr_s[tr_has])
    best_alpha_q = float(f_q.alpha_)
    Qhat_tr = np.zeros_like(Q_tr_s)
    for k in range(5):
        m_in = (inner_f != k) & tr_has
        m_out = (inner_f == k)
        rk = Ridge(alpha=best_alpha_q)
        rk.fit(Ctx_tr[m_in], Q_tr_s[m_in])
        Qhat_tr[m_out] = rk.predict(Ctx_tr[m_out]).astype(np.float32)
    Qhat_va = f_q.predict(Ctx_va).astype(np.float32)
    dQ_tr = Q_tr_s - Qhat_tr
    dQ_va = Q_va_s - Qhat_va

    # 5. Reverse JEPA g: (Tab+Spatial + Photo) -> Ẑ_text (E5-PCA-32)
    # Measures whether the listing text contradicts the photos+specs
    Rev_tr = np.hstack([C_tr, Z_ph_tr_s])
    Rev_va = np.hstack([C_va, Z_ph_va_s])
    f_rev = RidgeCV(alphas=alphas)
    f_rev.fit(Rev_tr[tr_has], E5_tr_s[tr_has])
    best_alpha_rev = float(f_rev.alpha_)
    E5hat_tr = np.zeros_like(E5_tr_s)
    for k in range(5):
        m_in = (inner_f != k) & tr_has
        m_out = (inner_f == k)
        rk = Ridge(alpha=best_alpha_rev)
        rk.fit(Rev_tr[m_in], E5_tr_s[m_in])
        E5hat_tr[m_out] = rk.predict(Rev_tr[m_out]).astype(np.float32)
    E5hat_va = f_rev.predict(Rev_va).astype(np.float32)
    dE5_tr = E5_tr_s - E5hat_tr
    dE5_va = E5_va_s - E5hat_va

    # 6. Orthogonalized JEPA price-residual Ridge predictor: Δz_photo -> log_real_price residual
    # Predicts the exact price surprise implied purely by Δz_photo (zero metadata/text leakage)
    y_tr_real = np.log(np.maximum(y_price[tr_idx] / bundle["tr_deflator"], 10_000.0))
    # Subtract the already-computed fold-safe structured+text baseline (txt_tab_plus_blend or log_real_price)
    if "txt_tab_plus_blend" in tr_df.columns:
        base_tr = pd.to_numeric(tr_df["txt_tab_plus_blend"], errors="coerce").to_numpy(dtype=np.float64)
        base_tr = np.where(np.isfinite(base_tr), base_tr, np.nanmedian(y_tr_real))
    else:
        base_tr = np.full_like(y_tr_real, np.median(y_tr_real))
    resid_tr = (y_tr_real - base_tr).astype(np.float32)

    dz_all_tr = np.hstack([dZ_ph_tr, dQ_tr])
    dz_all_va = np.hstack([dZ_ph_va, dQ_va])
    tr_norm_vis = tr_has & normal_mask[tr_idx]
    f_price_dz = RidgeCV(alphas=np.logspace(1.0, 5.0, 13))
    f_price_dz.fit(dz_all_tr[tr_norm_vis], resid_tr[tr_norm_vis])
    alpha_p = float(f_price_dz.alpha_)
    jepa_price_surp_tr = np.zeros(len(tr_idx), dtype=np.float32)
    for k in range(5):
        m_in = (inner_f != k) & tr_norm_vis
        m_out = (inner_f == k)
        rk = Ridge(alpha=alpha_p)
        rk.fit(dz_all_tr[m_in], resid_tr[m_in])
        jepa_price_surp_tr[m_out] = rk.predict(dz_all_tr[m_out]).astype(np.float32)
    jepa_price_surp_va = f_price_dz.predict(dz_all_va).astype(np.float32)

    # 7. Assemble clean, compact JEPA feature frames (masked to 0 where no photos)
    def _pack(dZ_ph, Z_ph, Zhat_ph, dQ, dE5, q_raw, render_raw, price_surp, has_v, index):
        # Forward & reverse L2 energies (RMS per dim)
        e_sig = np.sqrt(np.mean(dZ_ph[:, :32] ** 2, axis=1))
        e_so = np.sqrt(np.mean(dZ_ph[:, 32:] ** 2, axis=1))
        e_ph = np.sqrt(np.mean(dZ_ph ** 2, axis=1))
        e_q = np.sqrt(np.mean(dQ ** 2, axis=1))
        e_txt = np.sqrt(np.mean(dE5 ** 2, axis=1))
        e_sym = 0.5 * (e_ph + e_txt)

        # Cosine distance between actual and expected photo embedding in PCA-64 space
        num = np.sum(Z_ph * Zhat_ph, axis=1)
        den = np.linalg.norm(Z_ph, axis=1) * np.linalg.norm(Zhat_ph, axis=1) + 1e-6
        cos_sim = np.clip(num / den, -1.0, 1.0)
        cos_dist = 1.0 - cos_sim

        # Semantic visual surprise scores (positive = photos look BETTER than specs predict)
        # Q indices: 0=modern_finish, 1=fixer_shell (invert!), 2=natural_light, 3=kitchen_luxury,
        # 4=bath_luxury, 5=architect_charm, 6=facade_prestige, 7=floor_quality
        dz_modern = 0.5 * (dQ[:, 0] + dQ[:, 10])
        dz_fixer = 0.5 * (dQ[:, 1] + dQ[:, 11])
        dz_kitchen = 0.5 * (dQ[:, 3] + dQ[:, 13])
        dz_bath = 0.5 * (dQ[:, 4] + dQ[:, 14])
        dz_light = 0.5 * (dQ[:, 2] + dQ[:, 12])
        dz_charm = 0.5 * (dQ[:, 5] + dQ[:, 15])
        dz_facade = 0.5 * (dQ[:, 6] + dQ[:, 16])
        # Composite signed "Visual Quality Surprise": high = hidden gem, low = over-claimed / dated
        dz_quality_index = (
            0.30 * dz_modern
            - 0.30 * dz_fixer
            + 0.15 * dz_kitchen
            + 0.10 * dz_bath
            + 0.08 * dz_light
            + 0.07 * dz_facade
        )

        # Idea 3: Intra-listing cherry-picking / room-coherence spread from q_feats
        # q_feats[:, 0:10] = mean across real photos, q_feats[:, 10:20] = p75 across real photos
        # q_feats[:, 26] = max fixer_shell across interior rooms, q_feats[:, 27] = min modern_finish
        p75_minus_mean_modern = q_raw[:, 10] - q_raw[:, 0]
        p75_minus_mean_fixer = q_raw[:, 11] - q_raw[:, 1]
        cherry_pick_spread = q_raw[:, 10] - q_raw[:, 27]      # best-photo finish minus worst-room finish
        worst_room_fixer_gap = q_raw[:, 26] - q_raw[:, 1]     # worst-room fixer minus mean fixer
        avg_intra_dispersion = np.mean(q_raw[:, 10:20] - q_raw[:, 0:10], axis=1)

        cols_dict = {
            "jepa_price_surprise": price_surp,
            "jepa_energy_sig": e_sig,
            "jepa_energy_so": e_so,
            "jepa_energy_photo": e_ph,
            "jepa_energy_q": e_q,
            "jepa_energy_text": e_txt,
            "jepa_energy_sym": e_sym,
            "jepa_cos_dist": cos_dist,
            "jepa_dz_quality_index": dz_quality_index,
            "jepa_dz_modern": dz_modern,
            "jepa_dz_fixer": dz_fixer,
            "jepa_dz_kitchen": dz_kitchen,
            "jepa_dz_bath": dz_bath,
            "jepa_dz_light": dz_light,
            "jepa_dz_charm": dz_charm,
            "jepa_dz_facade": dz_facade,
            "jepa_cherry_pick_spread": cherry_pick_spread,
            "jepa_worst_room_fixer_gap": worst_room_fixer_gap,
            "jepa_p75_mean_modern": p75_minus_mean_modern,
            "jepa_p75_mean_fixer": p75_minus_mean_fixer,
            "jepa_avg_intra_disp": avg_intra_dispersion,
            # Top 8 orthogonal Δz_photo directions (lowest-variance residuals carry pure aesthetics)
            **{f"jepa_dz_sig_{j}": dZ_ph[:, j] for j in range(8)},
            **{f"jepa_dz_so_{j}": dZ_ph[:, 32 + j] for j in range(4)},
        }
        out = pd.DataFrame(cols_dict, index=index, dtype=np.float32)
        if not np.all(has_v):
            out.loc[~has_v, :] = 0.0
        return out

    j_tr = _pack(dZ_ph_tr, Z_ph_tr_s, Zhat_ph_tr, dQ_tr, dE5_tr, q_all[tr_idx], render_all[tr_idx], jepa_price_surp_tr, tr_has, tr_df.index)
    j_va = _pack(dZ_ph_va, Z_ph_va_s, Zhat_ph_va, dQ_va, dE5_va, q_all[va_idx], render_all[va_idx], jepa_price_surp_va, va_has, va_df.index)

    diag = {
        "fwd_photo_r2": round(fwd_r2, 4),
        "alpha_fwd": round(best_alpha_fwd, 1),
        "alpha_q": round(best_alpha_q, 1),
        "alpha_rev": round(best_alpha_rev, 1),
        "alpha_price_dz": round(alpha_p, 1),
    }
    return j_tr, j_va, diag


def _eval_lgbm_6splits(
    bundles: list[dict],
    jepa_splits: list[tuple[pd.DataFrame, pd.DataFrame]],
    base_cols: list[str],
    extra_cols: list[str],
    y_price: np.ndarray,
    normal_mask: np.ndarray,
    lgbm_params: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Train p7_lgbmxt_price_hub across all 6 splits (5F OOF + temporal_2026)."""
    from production_pipeline import TreePreprocSpec

    n_all = len(y_price)
    oof_pred = np.full(n_all, np.nan, dtype=np.float64)
    tmp_pred = None
    tmp_idx = None

    params = dict(lgbm_params)
    params.update({
        "objective": "huber",
        "extra_trees": True,
        "n_estimators": 4000,
        "learning_rate": 0.020,
        "n_jobs": 6,
        "random_state": 42,
        "verbose": -1,
    })

    for b, (j_tr, j_va) in zip(bundles, jepa_splits):
        tr_df = pd.concat([b["tr_df"], j_tr], axis=1) if extra_cols else b["tr_df"]
        va_df = pd.concat([b["va_df"], j_va], axis=1) if extra_cols else b["va_df"]
        cols = [c for c in (base_cols + extra_cols) if c in tr_df.columns]

        spec = TreePreprocSpec.fit(tr_df, cols)
        X_tr_all = spec.transform_lgbm(tr_df)
        X_va = spec.transform_lgbm(va_df)

        y_tr_log = np.log(np.maximum(y_price[b["tr_idx"]] / b["tr_deflator"], 10_000.0))

        # 12% cluster-clean early-stopping holdout (matches train_production.py)
        cids = b["tr_cids"]
        uniq = np.unique(cids)
        rng = np.random.default_rng(55)
        es_clusters = set(rng.choice(uniq, size=max(1, int(round(0.12 * len(uniq)))), replace=False))
        is_es = np.array([c in es_clusters for c in cids])

        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_tr_all.iloc[~is_es], y_tr_log[~is_es],
            eval_set=[(X_tr_all.iloc[is_es], y_tr_log[is_es])],
            callbacks=[lgb.early_stopping(80, verbose=False)],
        )
        p_log = np.clip(model.predict(X_va), -20.0, 20.0)
        p_eur = np.clip(np.exp(p_log) * b["va_deflator"], 10_000.0, 1e7)

        if str(b["id"]).startswith("fold_"):
            oof_pred[b["va_idx"]] = p_eur
        else:
            tmp_pred = p_eur
            tmp_idx = b["va_idx"]

    return oof_pred, tmp_pred, tmp_idx


def _eval_conformal_with_jepa(
    oof_eur: np.ndarray,
    tmp_eur: np.ndarray,
    tmp_idx: np.ndarray,
    y_price: np.ndarray,
    normal_mask: np.ndarray,
    folds: np.ndarray,
    full_df: pd.DataFrame,
    jepa_oof_df: pd.DataFrame,
    jepa_tmp_df: pd.DataFrame,
    ens_std_oof: np.ndarray,
    ens_std_tmp: np.ndarray,
    add_jepa_cols: list[str],
) -> dict:
    """Compare 80% Heteroscedastic Conformal Intervals WITH vs WITHOUT JEPA energy features."""
    from production_pipeline import build_conformal_context

    base_ctx_all = build_conformal_context(full_df, np.log(np.maximum(oof_eur, 10_000.0)), ens_std_oof)
    if add_jepa_cols:
        ctx_all = pd.concat([base_ctx_all.reset_index(drop=True), jepa_oof_df[add_jepa_cols].reset_index(drop=True)], axis=1)
    else:
        ctx_all = base_ctx_all.reset_index(drop=True)

    abs_err_oof = np.abs(np.log(np.maximum(y_price, 10_000.0)) - np.log(np.maximum(oof_eur, 10_000.0)))
    sigma_oof = np.full(len(y_price), np.nan, dtype=np.float64)

    for k in range(5):
        m_tr = (folds != k) & normal_mask & np.isfinite(oof_eur)
        m_va = (folds == k)
        sm = lgb.LGBMRegressor(
            objective="mae", n_estimators=200, learning_rate=0.03,
            max_depth=3, num_leaves=7, min_child_samples=80, reg_lambda=20.0,
            random_state=42 + k, n_jobs=4, verbose=-1,
        )
        sm.fit(ctx_all.iloc[m_tr], abs_err_oof[m_tr])
        sigma_oof[m_va] = np.clip(sm.predict(ctx_all.iloc[m_va]), 0.03, 0.50)

    # Calibrate q80 out-of-fold on normal rows
    q80_oof = np.zeros(len(y_price), dtype=np.float64)
    for k in range(5):
        m_tr = (folds != k) & normal_mask & np.isfinite(sigma_oof)
        m_va = (folds == k)
        ratios = abs_err_oof[m_tr] / sigma_oof[m_tr]
        q80_oof[m_va] = float(np.quantile(ratios, 0.80))

    half_log_oof = q80_oof * sigma_oof
    nm = normal_mask & np.isfinite(oof_eur)
    oof_cov = float(np.mean(abs_err_oof[nm] <= half_log_oof[nm]) * 100.0)
    oof_hw = float(np.median(np.expm1(half_log_oof[nm])) * 100.0)

    # Temporal 2026 evaluation (fit sigma on non-2026 normal rows)
    is_tmp = np.zeros(len(y_price), dtype=bool)
    is_tmp[tmp_idx] = True
    m_tr_tmp = (~is_tmp) & normal_mask & np.isfinite(oof_eur)
    sm_tmp = lgb.LGBMRegressor(
        objective="mae", n_estimators=200, learning_rate=0.03,
        max_depth=3, num_leaves=7, min_child_samples=80, reg_lambda=20.0,
        random_state=99, n_jobs=4, verbose=-1,
    )
    sm_tmp.fit(ctx_all.iloc[m_tr_tmp], abs_err_oof[m_tr_tmp])
    q80_tmp = float(np.quantile(abs_err_oof[m_tr_tmp] / np.clip(sm_tmp.predict(ctx_all.iloc[m_tr_tmp]), 0.03, 0.50), 0.80))

    base_ctx_tmp = build_conformal_context(
        full_df.iloc[tmp_idx].reset_index(drop=True),
        np.log(np.maximum(tmp_eur, 10_000.0)),
        ens_std_tmp,
    )
    if add_jepa_cols:
        ctx_tmp = pd.concat([base_ctx_tmp.reset_index(drop=True), jepa_tmp_df[add_jepa_cols].reset_index(drop=True)], axis=1)
    else:
        ctx_tmp = base_ctx_tmp.reset_index(drop=True)
    sigma_tmp = np.clip(sm_tmp.predict(ctx_tmp), 0.03, 0.50)
    abs_err_tmp = np.abs(np.log(np.maximum(y_price[tmp_idx], 10_000.0)) - np.log(np.maximum(tmp_eur, 10_000.0)))
    nm_tmp = normal_mask[tmp_idx]
    tmp_cov = float(np.mean(abs_err_tmp[nm_tmp] <= q80_tmp * sigma_tmp[nm_tmp]) * 100.0)
    tmp_hw = float(np.median(np.expm1(q80_tmp * sigma_tmp[nm_tmp])) * 100.0)

    # Correlation between predicted sigma and actual abs error (sharpness / calibration)
    corr_oof = float(spearmanr(sigma_oof[nm], abs_err_oof[nm]).statistic)
    corr_tmp = float(spearmanr(sigma_tmp[nm_tmp], abs_err_tmp[nm_tmp]).statistic)

    return {
        "oof_cov_pct": round(oof_cov, 2),
        "oof_hw_pct": round(oof_hw, 2),
        "tmp_cov_pct": round(tmp_cov, 2),
        "tmp_hw_pct": round(tmp_hw, 2),
        "sigma_err_spearman_oof": round(corr_oof, 4),
        "sigma_err_spearman_tmp": round(corr_tmp, 4),
    }


def main():
    t0 = time.time()
    print("=== Issue #10: Cross-Modal JEPA Latent-Residual Surprise Experiment ===")

    splits_obj = joblib.load(os.path.join(STAGING_DIR, "prod_splits.joblib"))
    bundles = splits_obj["bundles"]
    full_df = splits_obj["full_feat_df"]
    y_price = splits_obj["y_price"]
    normal_mask = splits_obj["normal_mask"]
    folds = splits_obj["folds"]
    mm_gbdt_cols = list(splits_obj["cols"]["multimodal"]["gbdt"])
    tt_gbdt_cols = list(splits_obj["cols"]["tabular_text"]["gbdt"])

    # Load raw training_dataset property_ids to align vision & E5 embeddings
    train_ds = pd.read_parquet(os.path.join(PROJECT_DIR, "outputs", "dataset", "training_dataset.parquet"))
    pids = train_ds["property_id"].astype(str).to_numpy()
    n = len(pids)
    print(f"Loaded prod_splits ({n:,} rows, {len(bundles)} splits, {len(mm_gbdt_cols)} mm cols, {len(tt_gbdt_cols)} tt cols)")

    # Align vision_full_embeddings.npz (vectorized — do NOT index npz inside a Python loop!)
    vnpz = np.load(VISION_NPZ, allow_pickle=True)
    v_pids = vnpz["property_id"]
    v_sig = vnpz["sig_real"]
    v_so = vnpz["so_sheet"]
    v_q = vnpz["q_feats"]
    v_so_q = vnpz["so_q_feats"]
    v_ren = vnpz["render_feats"]
    v_pid_to_i = {str(p): i for i, p in enumerate(v_pids)}
    matched_rows = []
    matched_vi = []
    for row_i, p in enumerate(pids):
        vi = v_pid_to_i.get(p)
        if vi is not None:
            matched_rows.append(row_i)
            matched_vi.append(vi)
    matched_rows = np.asarray(matched_rows, dtype=np.int64)
    matched_vi = np.asarray(matched_vi, dtype=np.int64)

    sig_all = np.zeros((n, 768), dtype=np.float32)
    so_all = np.zeros((n, 1152), dtype=np.float32)
    q_all = np.zeros((n, 30), dtype=np.float32)
    so_q_all = np.zeros((n, 10), dtype=np.float32)
    render_all = np.zeros((n, 8), dtype=np.float32)
    has_vis_all = np.zeros(n, dtype=bool)
    sig_all[matched_rows] = v_sig[matched_vi].astype(np.float32)
    so_all[matched_rows] = v_so[matched_vi].astype(np.float32)
    q_all[matched_rows] = v_q[matched_vi].astype(np.float32)
    so_q_all[matched_rows] = v_so_q[matched_vi].astype(np.float32)
    render_all[matched_rows] = v_ren[matched_vi].astype(np.float32)
    # Only rows with finite non-zero embeddings truly have photos
    has_vis_all = (
        np.isfinite(sig_all).all(axis=1)
        & np.isfinite(so_all).all(axis=1)
        & np.isfinite(q_all).all(axis=1)
        & np.isfinite(so_q_all).all(axis=1)
        & (np.linalg.norm(sig_all, axis=1) > 1e-4)
        & (np.linalg.norm(so_all, axis=1) > 1e-4)
    )
    sig_all[~has_vis_all] = 0.0
    so_all[~has_vis_all] = 0.0
    q_all[~has_vis_all] = 0.0
    so_q_all[~has_vis_all] = 0.0
    render_all[~has_vis_all] = 0.0
    print(f"Vision aligned (finite & non-empty): {has_vis_all.sum():,}/{n:,} rows ({has_vis_all.mean()*100:.1f}%)")

    # Align E5-large embeddings
    e5_npz = np.load(E5_NPZ, allow_pickle=True)
    e5_key = "embeddings" if "embeddings" in e5_npz.files else e5_npz.files[0]
    e5_all = e5_npz[e5_key].astype(np.float32)
    assert len(e5_all) == n, f"E5 shape mismatch: {e5_all.shape} vs {n}"

    # Context columns for JEPA predictors: numeric/bool tabular+spatial+text columns (ZERO vis_*)
    sample_tr = bundles[0]["tr_df"]
    tab_num_cols = [
        c for c in tt_gbdt_cols
        if c in sample_tr.columns and (
            pd.api.types.is_numeric_dtype(sample_tr[c]) or pd.api.types.is_bool_dtype(sample_tr[c])
        )
    ]
    print(f"JEPA context matrix: {len(tab_num_cols)} tabular+spatial+text numeric cols + 32 E5 PCA dims")

    # Step 1: Compute strictly fold-safe JEPA features on all 6 splits
    t1 = time.time()
    jepa_splits = []
    split_diags = {}
    jepa_oof_df = None
    jepa_tmp_df = None
    tmp_idx = None

    for b in bundles:
        j_tr, j_va, diag = _compute_jepa_for_split(
            b, tab_num_cols, sig_all, so_all, q_all, so_q_all, render_all,
            has_vis_all, e5_all, y_price, normal_mask,
        )
        jepa_splits.append((j_tr, j_va))
        split_diags[str(b["id"])] = diag
        if jepa_oof_df is None:
            jepa_oof_df = pd.DataFrame(np.nan, index=np.arange(n), columns=j_va.columns, dtype=np.float32)
        if str(b["id"]).startswith("fold_"):
            jepa_oof_df.iloc[b["va_idx"]] = j_va.to_numpy(dtype=np.float32)
        else:
            jepa_tmp_df = j_va.reset_index(drop=True)
            tmp_idx = b["va_idx"]

    print(f"Step 1 (6-split fold-safe JEPA features) finished in {time.time() - t1:.1f}s")
    for k, d in split_diags.items():
        print(f"  {k}: fwd_photo_R2={d['fwd_photo_r2']:.4f} (alpha_fwd={d['alpha_fwd']}, alpha_q={d['alpha_q']}, alpha_rev={d['alpha_rev']})")

    # Step 2: Diagnostic correlations against real Phase-7 OOF residuals, price cuts, duration, and condition claims
    oof_ana = pd.read_parquet(os.path.join(PROD_DIR, "oof_analytics.parquet"))
    assert len(oof_ana) == n
    nm = normal_mask & has_vis_all & np.isfinite(oof_ana["pred_mm_eur"].to_numpy())
    ask_eur = oof_ana["price"].to_numpy(dtype=np.float64)
    # Guard unit just in case
    if np.nanmedian(ask_eur) < 5000:
        ask_eur = ask_eur * 1000.0
    log_y = np.log(np.maximum(ask_eur, 10_000.0))
    log_pred_tt = np.log(np.maximum(oof_ana["pred_tt_eur"].to_numpy(dtype=np.float64), 10_000.0))
    log_pred_mm = np.log(np.maximum(oof_ana["pred_mm_eur"].to_numpy(dtype=np.float64), 10_000.0))
    resid_tt = log_y - log_pred_tt           # What tabular+text MISSED (true visual signal)
    resid_mm = log_y - log_pred_mm           # What even multimodal Phase-7 MISSED
    abs_err_mm = np.abs(resid_mm)            # Uncertainty target (conformal)

    print("\n--- Step 2: Signal Diagnostics on Normal Ghent Homes (with photos, n={:,}) ---".format(int(nm.sum())))
    diag_rows = []
    for col in [
        "jepa_price_surprise", "jepa_dz_quality_index", "jepa_dz_modern", "jepa_dz_fixer",
        "jepa_dz_kitchen", "jepa_dz_bath", "jepa_dz_facade",
        "jepa_energy_photo", "jepa_energy_sym", "jepa_energy_text", "jepa_cos_dist",
        "jepa_cherry_pick_spread", "jepa_worst_room_fixer_gap", "jepa_avg_intra_disp",
    ]:
        v = jepa_oof_df[col].to_numpy()[nm]
        r_tt = float(spearmanr(v, resid_tt[nm]).statistic)
        r_mm = float(spearmanr(v, resid_mm[nm]).statistic)
        r_abs = float(spearmanr(v, abs_err_mm[nm]).statistic)
        diag_rows.append({"feature": col, "r_vs_tt_resid": round(r_tt, 4), "r_vs_mm_resid": round(r_mm, 4), "r_vs_abs_err_mm": round(r_abs, 4)})
        print(f"  {col:26s} | r(tt_resid)={r_tt:+.4f} | r(mm_resid)={r_mm:+.4f} | r(|err_mm|)={r_abs:+.4f}")

    # Step 2b: Check buyer-facing "Condition Claim Honesty" & "Hidden Gem" stratification!
    df_chk = pd.DataFrame({
        "property_id": pids,
        "asking_eur": ask_eur,
        "pred_mm_eur": oof_ana["pred_mm_eur"].to_numpy(),
        "pred_tt_eur": oof_ana["pred_tt_eur"].to_numpy(),
        "delta_pct_mm": oof_ana["delta_pct"].to_numpy(),
        "abs_err_pct_mm": np.abs(oof_ana["pred_mm_eur"].to_numpy() - ask_eur) / np.maximum(ask_eur, 1.0) * 100.0,
        "condition": oof_ana["building_condition"].astype(str).to_numpy(),
        "had_price_cut": oof_ana["had_price_cut"].to_numpy(),
        "is_under_option": oof_ana["is_under_option"].to_numpy(),
        "listing_duration_days": oof_ana["listing_duration_days"].to_numpy(),
        "normal_vis": nm,
    })
    for c in jepa_oof_df.columns:
        df_chk[c] = jepa_oof_df[c].to_numpy()

    sub_nm = df_chk[df_chk["normal_vis"]].copy()
    sub_nm["energy_q"] = pd.qcut(sub_nm["jepa_energy_sym"], 5, labels=["Q1 (Lowest)", "Q2", "Q3", "Q4", "Q5 (Highest)"])
    print("\n--- Conformal Error & Market Dynamics by Symmetric JEPA Energy Quintile ---")
    energy_quintiles = []
    for q, g in sub_nm.groupby("energy_q", observed=True):
        row = {
            "quintile": str(q),
            "n": int(len(g)),
            "mdape_mm": round(float(g["abs_err_pct_mm"].median()), 2),
            "p80_err_mm": round(float(g["abs_err_pct_mm"].quantile(0.80)), 2),
            "pct_price_cut": round(float(g["had_price_cut"].mean() * 100.0), 2),
            "pct_under_option": round(float(g["is_under_option"].mean() * 100.0), 2),
            "mean_duration_d": round(float(g["listing_duration_days"].mean()), 1),
        }
        energy_quintiles.append(row)
        print(f"  {row['quintile']:14s}: n={row['n']:,} | MdAPE={row['mdape_mm']:5.2f}% | P80_err={row['p80_err_mm']:5.2f}% | cuts={row['pct_price_cut']:4.2f}% | opt={row['pct_under_option']:4.1f}% | stay={row['mean_duration_d']:4.1f}d")

    # Condition honesty check: among listings claiming AS_NEW or JUST_RENOVATED, compare top vs bottom quintile of jepa_dz_quality_index
    sub_claimed = sub_nm[sub_nm["condition"].isin(["AS_NEW", "JUST_RENOVATED"])].copy()
    honesty_summary = {}
    if len(sub_claimed) >= 200:
        lo_thr = float(sub_claimed["jepa_dz_quality_index"].quantile(0.15))
        hi_thr = float(sub_claimed["jepa_dz_quality_index"].quantile(0.85))
        sus = sub_claimed[sub_claimed["jepa_dz_quality_index"] <= lo_thr]
        ver = sub_claimed[sub_claimed["jepa_dz_quality_index"] >= hi_thr]
        honesty_summary = {
            "n_claimed_top_cond": int(len(sub_claimed)),
            "bottom15_looks_dated": {
                "n": int(len(sus)),
                "median_ask_eur": int(round(sus["asking_eur"].median())),
                "median_tt_pred_eur": int(round(sus["pred_tt_eur"].median())),
                "median_mm_vs_tt_shift_pct": round(float(((sus["pred_mm_eur"] - sus["pred_tt_eur"]) / sus["pred_tt_eur"]).median() * 100.0), 2),
                "median_actual_vs_tt_pct": round(float(((sus["asking_eur"] - sus["pred_tt_eur"]) / sus["pred_tt_eur"]).median() * 100.0), 2),
                "pct_price_cut": round(float(sus["had_price_cut"].mean() * 100.0), 2),
                "pct_under_option": round(float(sus["is_under_option"].mean() * 100.0), 2),
            },
            "top15_looks_sharp": {
                "n": int(len(ver)),
                "median_ask_eur": int(round(ver["asking_eur"].median())),
                "median_tt_pred_eur": int(round(ver["pred_tt_eur"].median())),
                "median_mm_vs_tt_shift_pct": round(float(((ver["pred_mm_eur"] - ver["pred_tt_eur"]) / ver["pred_tt_eur"]).median() * 100.0), 2),
                "median_actual_vs_tt_pct": round(float(((ver["asking_eur"] - ver["pred_tt_eur"]) / ver["pred_tt_eur"]).median() * 100.0), 2),
                "pct_price_cut": round(float(ver["had_price_cut"].mean() * 100.0), 2),
                "pct_under_option": round(float(ver["is_under_option"].mean() * 100.0), 2),
            },
        }
        print("\n--- Condition Honesty Check (claimed AS_NEW / JUST_RENOVATED) ---")
        print("  Bottom 15% Δz_quality (looks dated):", honesty_summary["bottom15_looks_dated"])
        print("  Top 15%    Δz_quality (looks sharp):", honesty_summary["top15_looks_sharp"])

    # Step 3: Acceptance Gate #1 — 6-split LGBM-XT ablation (Baseline vs +JEPA compact vs +JEPA full)
    with open(HPARAMS_JSON, "r", encoding="utf-8") as fh:
        hpo = json.load(fh)
    lgbm_params = hpo["lgbm_price_huber"]["params"]

    jepa_compact_cols = [
        "jepa_price_surprise", "jepa_dz_quality_index",
        "jepa_energy_photo", "jepa_energy_text", "jepa_energy_sym", "jepa_cos_dist",
        "jepa_cherry_pick_spread", "jepa_worst_room_fixer_gap",
    ]
    jepa_all_cols = list(jepa_oof_df.columns)

    print("\n--- Step 3: 6-Split LGBM-XT Ablation (Agency-Blind Multimodal) ---")
    ablation = {}
    for label, extra in [
        ("baseline_mm_279", []),
        ("mm_plus_jepa_compact_8", jepa_compact_cols),
        ("mm_plus_jepa_full_33", jepa_all_cols),
    ]:
        ts = time.time()
        oof_p, tmp_p, t_idx = _eval_lgbm_6splits(
            bundles, jepa_splits, mm_gbdt_cols, extra, y_price, normal_mask, lgbm_params,
        )
        m_oof = _metrics(y_price, oof_p, normal_mask & np.isfinite(oof_p))
        m_tmp = _metrics(y_price[t_idx], tmp_p, normal_mask[t_idx])
        ablation[label] = {"n_extra": len(extra), "oof": m_oof, "tmp": m_tmp, "elapsed_s": round(time.time() - ts, 1)}
        print(
            f"  {label:26s} | OOF MdAPE={m_oof['mdape']:.3f}% (Hit10={m_oof['hit10']:.2f}%, R2={m_oof['r2']:.4f}) "
            f"| TMP MdAPE={m_tmp['mdape']:.3f}% (Hit10={m_tmp['hit10']:.2f}%) [{ablation[label]['elapsed_s']}s]"
        )

    # Step 4: Acceptance Gate #2 — 80% Heteroscedastic Conformal Intervals on Production Lean-5 OOF/TMP
    print("\n--- Step 4: 80% Heteroscedastic Conformal Interval Gate (Production Lean-5) ---")
    gbdt_mm = joblib.load(os.path.join(STAGING_DIR, "gbdt_multimodal.joblib"))
    tabm_p = joblib.load(os.path.join(STAGING_DIR, "tabm_perm170_price_4s_multimodal.joblib"))
    tabm_s = joblib.load(os.path.join(STAGING_DIR, "tabm_perm170_psqm_4s_multimodal.joblib"))

    # Reconstruct exact 5-member log matrix for ens_std & stacked OOF/TMP
    from sklearn.linear_model import HuberRegressor
    O_mat = np.column_stack([
        gbdt_mm["oof"]["lgbmxt_price_hub"],
        gbdt_mm["oof"]["spec_lgbm_price_hub"],
        gbdt_mm["oof"]["cb_tuned_price_l2"],
        tabm_p["oof"],
        tabm_s["oof"],
    ])
    T_mat = np.column_stack([
        gbdt_mm["tmp"]["lgbmxt_price_hub"],
        gbdt_mm["tmp"]["spec_lgbm_price_hub"],
        gbdt_mm["tmp"]["cb_tuned_price_l2"],
        tabm_p["tmp"],
        tabm_s["tmp"],
    ])
    O_log = np.log(np.clip(O_mat, 10_000.0, 1e7))
    T_log = np.log(np.clip(T_mat, 10_000.0, 1e7))
    ens_std_oof = np.std(O_log, axis=1)
    ens_std_tmp = np.std(T_log, axis=1)

    y_log_all = np.log(np.clip(y_price, 10_000.0, 1e7))
    oof_stack_eur = np.full(n, np.nan, dtype=np.float64)
    for k in range(5):
        m_tr = (folds != k) & normal_mask
        m_va = (folds == k)
        hr = HuberRegressor(alpha=15.0, max_iter=500).fit(O_log[m_tr], y_log_all[m_tr])
        oof_stack_eur[m_va] = np.exp(hr.predict(O_log[m_va]))
    hr_full = HuberRegressor(alpha=15.0, max_iter=500).fit(O_log[normal_mask], y_log_all[normal_mask])
    tmp_stack_eur = np.exp(hr_full.predict(T_log))

    jepa_conf_cols = [
        "jepa_energy_photo", "jepa_energy_text", "jepa_energy_sym", "jepa_cos_dist",
        "jepa_cherry_pick_spread", "jepa_worst_room_fixer_gap",
    ]
    conf_base = _eval_conformal_with_jepa(
        oof_stack_eur, tmp_stack_eur, tmp_idx, y_price, normal_mask, folds,
        full_df, jepa_oof_df, jepa_tmp_df, ens_std_oof, ens_std_tmp, [],
    )
    conf_jepa = _eval_conformal_with_jepa(
        oof_stack_eur, tmp_stack_eur, tmp_idx, y_price, normal_mask, folds,
        full_df, jepa_oof_df, jepa_tmp_df, ens_std_oof, ens_std_tmp, jepa_conf_cols,
    )
    print(f"  Conformal Baseline: OOF cov={conf_base['oof_cov_pct']}% hw=±{conf_base['oof_hw_pct']}% (r={conf_base['sigma_err_spearman_oof']}) | TMP cov={conf_base['tmp_cov_pct']}% hw=±{conf_base['tmp_hw_pct']}% (r={conf_base['sigma_err_spearman_tmp']})")
    print(f"  Conformal + JEPA  : OOF cov={conf_jepa['oof_cov_pct']}% hw=±{conf_jepa['oof_hw_pct']}% (r={conf_jepa['sigma_err_spearman_oof']}) | TMP cov={conf_jepa['tmp_cov_pct']}% hw=±{conf_jepa['tmp_hw_pct']}% (r={conf_jepa['sigma_err_spearman_tmp']})")

    # Save artifacts
    df_chk.to_parquet(OUT_OOF_PARQUET, index=False)
    report = {
        "split_diagnostics": split_diags,
        "signal_correlations": diag_rows,
        "energy_quintiles": energy_quintiles,
        "condition_honesty": honesty_summary,
        "lgbmxt_ablation": ablation,
        "conformal_gate": {"baseline": conf_base, "plus_jepa": conf_jepa},
        "total_elapsed_s": round(time.time() - t0, 1),
    }
    with open(OUT_REPORT, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nSaved report -> {OUT_REPORT} ({report['total_elapsed_s']}s total)")


if __name__ == "__main__":
    main()
