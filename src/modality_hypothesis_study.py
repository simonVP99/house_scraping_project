"""Multimodal Information Decomposition & Hypothesis Lab on the 790-800 Stratified Pilot Homes.

Answers four foundational ML questions quantitatively (all strictly 5-fold grouped OOF):
  H1 (Modality Fingerprints): What real-estate concepts does each modality actually know?
     - Standalone OOF R² / MdAPE of each modality on log(€/m²) and log(price)
     - Cross-attribute probing: how well can Vision (VLM, SigLIP2, DINOv2, Aerial) and Text predict
       building_condition, energy_label (EPC), year_built, house_vs_apt, historic_core, plot_size,
       detached/terraced (grb_adjoining), mobiscore, and renovation_need?
  H2 (Redundancy vs Unique Information):
     - Sequential & leave-one-out variance decomposition: Tabular -> +Spatial -> +Text -> +Vision
     - Exact orthogonality audit: what fraction of Vision's price prediction is ALREADY explained by
       Tabular + Spatial + Text?
  H3 (Why did global Vision Ridge fail & can Non-Linear / Conditional Vision win?):
     - H3a: Missing-metadata & short-text slice test (where Tabular/Text is blind!)
     - H3b: Luxury €650k+ sign-flip root-cause audit (real listing inspection)
     - H3c: Non-linear interaction test (segmented Ridge + shallow LGBM/CatBoost on P4 + Vision × Context)
  H4 (Disagreement & Uncertainty Signal):
     - Does Vision-vs-Tabular mismatch predict |error| (better uncertainty bands / deal-alert filtering)?

Pure CPU (~25s). No GPU contention.
"""
from __future__ import annotations

import glob
import os
import sys
import warnings

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'src'))

from phase3_features import compute_metrics, fmt
from vision_pilot_eval import Q_PAIRS, ROOM_PROMPTS
ROOM_NAMES = list(ROOM_PROMPTS.keys())

PILOT_DIR = os.path.join(BASE_DIR, 'external_data', 'vision_pilot')
PHOTO_DIR = os.path.join(PILOT_DIR, 'photos')
AERIAL_DIR = os.path.join(PILOT_DIR, 'aerial')
MANIFEST_PATH = os.path.join(PILOT_DIR, 'pilot_manifest.parquet')
VLM_PATH = os.path.join(PILOT_DIR, 'vlm_scores_qwen36_35b_splash.parquet')
EMB_PATH = os.path.join(PILOT_DIR, 'pilot_embeddings.npz')
P4_CACHE = os.path.join(BASE_DIR, 'outputs', 'models', 'workbench', 'phase4_splits_cache.joblib')
P5_ADDON = os.path.join(BASE_DIR, 'outputs', 'models', 'workbench', 'phase5_addon_features.joblib')
DATASET_PATH = os.path.join(BASE_DIR, 'outputs', 'dataset', 'training_dataset.parquet')


def oof_ridge(X: np.ndarray, y: np.ndarray, folds: np.ndarray, alphas=None) -> np.ndarray:
    if alphas is None:
        alphas = np.logspace(0, 5, 25)
    oof = np.full(len(y), np.nan)
    X = np.asarray(X, dtype=np.float64)
    for f in np.unique(folds):
        tr = (folds != f) & np.isfinite(y)
        va = folds == f
        # Median fill strictly from train
        med = np.nanmedian(X[tr], axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        Xtr = np.where(np.isfinite(X[tr]), X[tr], med)
        Xva = np.where(np.isfinite(X[va]), X[va], med)
        sc = StandardScaler()
        m = RidgeCV(alphas=alphas)
        m.fit(sc.fit_transform(Xtr), y[tr])
        oof[va] = m.predict(sc.transform(Xva))
    return oof


def oof_logreg_auc(X: np.ndarray, y_bin: np.ndarray, folds: np.ndarray) -> float:
    oof = np.full(len(y_bin), np.nan)
    X = np.asarray(X, dtype=np.float64)
    for f in np.unique(folds):
        tr = (folds != f) & np.isfinite(y_bin)
        va = (folds == f) & np.isfinite(y_bin)
        if tr.sum() < 20 or va.sum() < 5 or len(np.unique(y_bin[tr])) < 2:
            continue
        med = np.nanmedian(X[tr], axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        Xtr = np.where(np.isfinite(X[tr]), X[tr], med)
        Xva = np.where(np.isfinite(X[va]), X[va], med)
        sc = StandardScaler()
        m = LogisticRegression(C=0.1, max_iter=500)
        m.fit(sc.fit_transform(Xtr), y_bin[tr].astype(int))
        oof[va] = m.predict_proba(sc.transform(Xva))[:, 1]
    ok = np.isfinite(oof) & np.isfinite(y_bin)
    return float(roc_auc_score(y_bin[ok].astype(int), oof[ok]))


def extract_explicit_aerial_cv(pids: list[str]) -> np.ndarray:
    """Pure-NumPy CV metrics on 50m & 200m Flanders aerial orthophotos (greenness, roof, water, brightness)."""
    feats = []
    for pid in pids:
        row = []
        for scale in ('50m', '200m'):
            p = os.path.join(AERIAL_DIR, f'{pid}_{scale}.jpg')
            if os.path.exists(p):
                arr = np.asarray(Image.open(p).convert('RGB'), dtype=np.float32) / 255.0
                R, G, B = arr[..., 0], arr[..., 1], arr[..., 2]
                exg = 2 * G - R - B  # Excess Green vegetation index
                veg = (exg > 0.08) & (G > 0.18)
                roof = (R - B > 0.12) & (R > 0.25) & (G < R)  # terracotta/reddish tile roofs
                water = (B > R + 0.05) & (B > G) & (R < 0.35)
                bright = 0.299 * R + 0.587 * G + 0.114 * B
                # Center 30% patch (the actual parcel/roof) vs outer ring (neighborhood greenery)
                h, w = R.shape
                c_veg = veg[int(h*0.35):int(h*0.65), int(w*0.35):int(w*0.65)].mean()
                row.extend([
                    float(exg.mean()), float(veg.mean()), float(c_veg),
                    float(roof.mean()), float(water.mean()),
                    float(bright.mean()), float(bright.std()),
                ])
            else:
                row.extend([np.nan] * 7)
        feats.append(row)
    return np.asarray(feats, dtype=np.float32)


def main():
    print("=" * 95)
    print("MULTIMODAL INFORMATION DECOMPOSITION & HYPOTHESIS STUDY (n=790 Stratified Pilot Homes)")
    print("=" * 95)

    # 1. Load & align all modalities on the exact same 790 homes
    man = pd.read_parquet(MANIFEST_PATH)
    man['property_id'] = man['property_id'].astype(str)
    vlm = pd.read_parquet(VLM_PATH)
    vlm['property_id'] = vlm['property_id'].astype(str)

    # Reconstruct per-listing SigLIP2 & DINOv2 pooled + Q-Score matrices from pilot_embeddings.npz
    z = np.load(EMB_PATH, allow_pickle=True)
    sig_p, dino_p = z['sig_p'], z['dino_p']
    room_txt, q_pos_txt, q_neg_txt = z['room_txt'], z['q_pos_txt'], z['q_neg_txt']
    sig_a50, sig_a200 = z['sig_a50'], z['sig_a200']
    dino_a50, dino_a200 = z['dino_a50'], z['dino_a200']

    # Align photo rows to manifest index
    n_all = len(man)
    sig_mean = np.full((n_all, 768), np.nan, dtype=np.float32)
    dino_mean = np.full((n_all, 768), np.nan, dtype=np.float32)
    q_scores = np.full((n_all, len(Q_PAIRS) * 2 + len(ROOM_NAMES)), np.nan, dtype=np.float32)

    ptr = 0
    for i, pid in enumerate(man['property_id']):
        n_ph = len(glob.glob(os.path.join(PHOTO_DIR, pid, '*.jpg')))
        if n_ph > 0 and ptr + n_ph <= len(sig_p):
            sp = sig_p[ptr:ptr + n_ph]
            dp = dino_p[ptr:ptr + n_ph]
            rooms = (sp @ room_txt.T).argmax(axis=1)
            non_plan = rooms != 6
            use = non_plan if non_plan.sum() >= 1 else np.ones(n_ph, dtype=bool)
            sig_mean[i] = sp[use].mean(axis=0)
            dino_mean[i] = dp[use].mean(axis=0)
            # Contrastive Q-scores (pos - neg) mean & p75 + room fractions
            q_mat = (sp[use] @ q_pos_txt.T) - (sp[use] @ q_neg_txt.T)
            r_frac = np.bincount(rooms, minlength=len(ROOM_NAMES)) / max(n_ph, 1)
            q_scores[i] = np.concatenate([q_mat.mean(axis=0), np.percentile(q_mat, 75, axis=0), r_frac])
        ptr += n_ph

    # Filter to the 790 homes that have VLM + photo embeddings
    vlm_map = vlm.set_index('property_id')
    keep_mask = man['property_id'].isin(vlm_map.index) & np.isfinite(sig_mean[:, 0])
    idx_keep = np.where(keep_mask)[0]
    m = man.iloc[idx_keep].reset_index(drop=True)
    vlm_aligned = vlm_map.loc[m['property_id']].reset_index()

    X_sig_photo = sig_mean[idx_keep]
    X_dino_photo = dino_mean[idx_keep]
    X_qscore = q_scores[idx_keep]
    X_aerial_emb = np.hstack([sig_a50[idx_keep], sig_a200[idx_keep], dino_a50[idx_keep], dino_a200[idx_keep]])
    X_aerial_cv = extract_explicit_aerial_cv(m['property_id'].tolist())
    vlm_cols = [c for c in vlm.columns if c.startswith('vlm_')]
    X_vlm = vlm_aligned[vlm_cols].to_numpy(dtype=np.float32)

    # Pull out-of-fold Tabular, Spatial/OpenData, and Text features for these exact 790 homes from Phase-4/5 caches!
    tr_full = pd.read_parquet(DATASET_PATH)
    tr_full['property_id'] = tr_full['property_id'].astype(str)
    tr_idx_map = {pid: i for i, pid in enumerate(tr_full['property_id'])}
    ds_rows = np.array([tr_idx_map[pid] for pid in m['property_id']])

    p4_cache = joblib.load(P4_CACHE)
    p5_addon = joblib.load(P5_ADDON)
    # Assemble strict OOF validation rows across fold_0..4 for all 14,599 rows
    oof_frames = []
    for b in p4_cache['bundles']:
        if not str(b['id']).startswith('fold_'):
            continue
        va_df = b['va_df'].copy()
        va_add = p5_addon[b['id']]['va_add'].copy()
        va_df = pd.concat([va_df.reset_index(drop=True), va_add.reset_index(drop=True)], axis=1)
        va_df['__ds_idx__'] = b['va_idx']
        oof_frames.append(va_df)
    all_oof_df = pd.concat(oof_frames, axis=0).sort_values('__ds_idx__').reset_index(drop=True)
    pilot_df = all_oof_df.iloc[ds_rows].reset_index(drop=True)

    # Partition columns cleanly by modality!
    text_cols = [c for c in pilot_df.columns if c.startswith(('txt_', 'e5_', 'e5l_', 'qwen3_', 'tfidf_svd_', 'has_text_'))]
    spatial_cols = [c for c in pilot_df.columns if c.startswith((
        'sp_', 'oof_', 'fl_', 'grb_', 'gent_', 'sector_', 'dist_', 'coord_', 'latitude', 'longitude', 'distance_to_street'
    ))]
    # Pure structured property attributes (no spatial, no text)
    exclude = set(text_cols) | set(spatial_cols) | {'__ds_idx__', 'price', 'fold', 'property_id', 'property_key', 'date_obtained'}
    struct_cols = [
        c for c in pilot_df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(pilot_df[c])
        and not c.startswith(('te_', 'freq_', 'tab_anchor_'))
    ]

    # Numeric matrices per modality
    def num_mat(cols):
        sub = pilot_df[[c for c in cols if pd.api.types.is_numeric_dtype(pilot_df[c])]]
        return sub.to_numpy(dtype=np.float32)

    X_struct = num_mat(struct_cols)
    X_spatial = num_mat(spatial_cols)
    X_text_summary = num_mat([c for c in text_cols if not c.startswith(('e5_pca_', 'e5l_pca_', 'qwen3_pca_', 'tfidf_svd_'))])
    X_text_all = num_mat(text_cols)

    y_price = m['price'].to_numpy(float)
    y_area = m['livable_area'].to_numpy(float)
    y_log_price = np.log(y_price)
    y_log_psqm = np.log(y_price / y_area)
    p4_pred = m['p4_oof_pred'].to_numpy(float)
    y_resid = m['log_resid_p4'].to_numpy(float)
    folds = m['fold'].to_numpy(int)

    print(f"Aligned n={len(m)} homes | Modality dims: Struct={X_struct.shape[1]}, Spatial/Open={X_spatial.shape[1]}, "
          f"Text={X_text_all.shape[1]}, VLM={X_vlm.shape[1]}, QScore={X_qscore.shape[1]}, "
          f"SigLIP2={X_sig_photo.shape[1]}, DINOv2={X_dino_photo.shape[1]}, AerialCV={X_aerial_cv.shape[1]}")

    # =========================================================================
    # H1A: STANDALONE PREDICTIVE POWER OF EACH MODALITY ON log(€/m²)
    # =========================================================================
    print("\n" + "=" * 95)
    print("H1A. STANDALONE PREDICTIVE POWER BY MODALITY (5-fold OOF RidgeCV predicting log(€/m²) -> price)")
    print("=" * 95)
    modalities = {
        '1. Structured Tabular Only':          X_struct,
        '2. Spatial + Open Data (Statbel/GRB/Gent)': X_spatial,
        '3. Listing Text (TF-IDF+E5+Qwen3+MLP)': X_text_all,
        '4. Vision: Qwen3.6-35B VLM (10 axes)': X_vlm,
        '5. Vision: SigLIP2 Q-Scores (21d)':   X_qscore,
        '6. Vision: SigLIP2 Photos (768d)':    X_sig_photo,
        '7. Vision: DINOv2 Photos (768d)':     X_dino_photo,
        '8. Vision: Aerial CV Explicit (14d)': X_aerial_cv,
        '9. Vision: Aerial Emb (Sig+DINO)':    X_aerial_emb,
        '10. Vision ALL Combined':             np.hstack([X_vlm, X_qscore, X_sig_photo, X_dino_photo, X_aerial_cv]),
    }
    mod_preds = {}
    for name, Xm in modalities.items():
        p_psqm = oof_ridge(Xm, y_log_psqm, folds)
        mod_preds[name] = p_psqm
        r2_psqm = r2_score(y_log_psqm, p_psqm)
        mt = compute_metrics(y_price, np.exp(p_psqm) * y_area)
        r_res = float(np.corrcoef(p_psqm - np.log(p4_pred / y_area), y_resid)[0, 1])
        print(f"  {name:43s} | R²(log €/m²)={r2_psqm:+.3f} | Price MdAPE={mt['mdape']:5.2f}% | MAE=€{mt['mae']:7,.0f} | R²(€)={mt['r2']:+.3f}")

    # =========================================================================
    # H1B: WHAT CONCEPTS DOES EACH MODALITY ACTUALLY ENCODE? (Cross-Attribute Probing!)
    # =========================================================================
    print("\n" + "=" * 95)
    print("H1B. MODALITY FINGERPRINTING — WHAT REAL-ESTATE ATTRIBUTES DOES EACH MODALITY ACTUALLY 'SEE'?")
    print("     (5-fold OOF R² for continuous targets, ROC-AUC for binary targets)")
    print("=" * 95)
    tr_sub = tr_full.iloc[ds_rows].reset_index(drop=True)
    probe_targets = [
        ('House vs Apt (AUC)',      'bin', (tr_sub['property_type_HOUSE'] == 1).astype(float).to_numpy()),
        ('Needs Renovation (AUC)',  'bin', tr_sub['building_condition'].isin(['TO_RENOVATE', 'TO_RESTORE', 'TO_BE_DONE_UP', 'TO_REFURBISH']).astype(float).to_numpy()),
        ('As-New/JustReno (AUC)',   'bin', tr_sub['building_condition'].isin(['AS_NEW', 'JUST_RENOVATED']).astype(float).to_numpy()),
        ('New-Build (AUC)',         'bin', tr_sub['is_newly_built'].fillna(False).astype(float).to_numpy()),
        ('Historic Core (AUC)',     'bin', (m['pilot_segment'] == 'historic_core').astype(float).to_numpy()),
        ('EPC Ordinal (R²)',        'reg', pilot_df['epc_ordinal'].to_numpy(float)),
        ('Year Built (R²)',         'reg', tr_sub['year_built'].to_numpy(float)),
        ('Log Livable Area (R²)',   'reg', np.log(y_area)),
        ('Mobiscore (R²)',          'reg', pilot_df['fl_mobiscore'].to_numpy(float)),
        ('GRB Adjoining/Type (R²)', 'reg', pilot_df['grb_adjoining_bldgs'].to_numpy(float)),
        ('Cadastral Income (R²)',   'reg', np.log1p(tr_sub['cadastral_income'].clip(lower=0).to_numpy(float))),
        ('P4 RESIDUAL (R²)',        'reg', y_resid),
    ]
    probe_mods = [
        ('VLM-10',   X_vlm),
        ('SigLIP2',  X_sig_photo),
        ('DINOv2',   X_dino_photo),
        ('Aerial',   np.hstack([X_aerial_cv, X_aerial_emb])),
        ('Text',     X_text_summary),
        ('Spatial',  X_spatial),
    ]
    hdr = f"  {'Target Concept':26s} | " + " | ".join(f"{n:>8s}" for n, _ in probe_mods)
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for tname, ttype, y_t in probe_targets:
        vals = []
        for _, Xm in probe_mods:
            if ttype == 'bin':
                v = oof_logreg_auc(Xm, y_t, folds)
                vals.append(f"{v:8.3f}")
            else:
                ok = np.isfinite(y_t)
                p = oof_ridge(Xm, y_t, folds)
                v = r2_score(y_t[ok], p[ok])
                vals.append(f"{v:+8.3f}")
        print(f"  {tname:26s} | " + " | ".join(vals))

    # =========================================================================
    # H2: REDUNDANCY vs UNIQUE ORTHOGONAL SIGNAL (Variance Decomposition!)
    # =========================================================================
    print("\n" + "=" * 95)
    print("H2. HOW MUCH OF EACH VISION MODALITY'S PRICE SIGNAL IS ALREADY KNOWN BY TABULAR+SPATIAL+TEXT?")
    print("=" * 95)
    X_nonvis = np.hstack([X_struct, X_spatial, X_text_summary])
    p4_log_psqm = np.log(p4_pred / y_area)
    print(f"  Reference Phase-4 Ensemble (n=790): R²(log €/m²)={r2_score(y_log_psqm, p4_log_psqm):.4f} | "
          f"{fmt(compute_metrics(y_price, p4_pred))}")

    for vname, Xv in [
        ('Qwen3.6 VLM (10d)',     X_vlm),
        ('SigLIP2 Q-Scores (21d)', X_qscore),
        ('SigLIP2 Photos (768d)', X_sig_photo),
        ('DINOv2 Photos (768d)',  X_dino_photo),
        ('Aerial CV (14d)',       X_aerial_cv),
        ('Aerial Emb (3072d)',    X_aerial_emb),
    ]:
        pv = mod_preds[[k for k in mod_preds if vname.split(' ')[0] in k and (vname.split(' ')[1][:4] in k)][0]] if False else oof_ridge(Xv, y_log_psqm, folds)
        # How much of pv is predictable from Tabular+Spatial+Text?
        pv_from_nonvis = oof_ridge(X_nonvis, pv, folds)
        redundancy_r2 = r2_score(pv, pv_from_nonvis)
        unique_part = pv - pv_from_nonvis
        # Does the UNIQUE part correlate with the true residual y_resid?
        r_unique_vs_resid = float(np.corrcoef(unique_part, y_resid)[0, 1])
        r_raw_vs_resid = float(np.corrcoef(pv, y_resid)[0, 1])
        print(f"  {vname:24s} | Standalone R²={r2_score(y_log_psqm, pv):+.3f} | "
              f"Already explained by Tab+Spat+Txt={redundancy_r2*100:5.1f}% | "
              f"Unique-part corr(P4_resid)={r_unique_vs_resid:+.4f}")

    # =========================================================================
    # H3A: WHERE IS TABULAR+TEXT BLIND? (Missing Condition / Short Description / Fixers)
    # =========================================================================
    print("\n" + "=" * 95)
    print("H3A. BLIND-SPOT HYPOTHESIS — DOES VISION HELP MORE WHEN METADATA IS MISSING OR TEXT IS SHORT?")
    print("=" * 95)
    cond_missing = tr_sub['building_condition'].isna().to_numpy()
    epc_missing = tr_sub['energy_label'].isna().to_numpy()
    short_desc = (tr_sub['description_length'].fillna(0) < 650).to_numpy()
    is_fixer = tr_sub['building_condition'].isin(['TO_RENOVATE', 'TO_RESTORE', 'TO_BE_DONE_UP', 'TO_REFURBISH']).to_numpy()
    is_not_lux = (y_price < 650_000)

    # Compact 31-dim vision vector = Qwen3.6 VLM (10) + SigLIP2 Q-Scores (21)
    X_vis_compact = np.hstack([X_vlm, X_qscore])
    oof_vlm_r = oof_ridge(X_vlm, y_resid, folds)
    oof_sig_r = oof_ridge(X_sig_photo, y_resid, folds)
    oof_comb_r = 0.5 * oof_vlm_r + 0.5 * oof_sig_r

    for label, mask in [
        ('ALL homes',                  np.ones(len(m), dtype=bool)),
        ('Non-luxury (<€650k)',        is_not_lux),
        ('Luxury (>=€650k)',           ~is_not_lux),
        ('building_condition MISSING', cond_missing),
        ('energy_label MISSING',       epc_missing),
        ('Short description (<650ch)', short_desc),
        ('Fixer-upper condition',      is_fixer),
        ('Fixer OR cond_missing',      is_fixer | cond_missing),
    ]:
        if mask.sum() < 25:
            continue
        b_mt = compute_metrics(y_price[mask], p4_pred[mask])
        r_vlm = float(np.corrcoef(oof_vlm_r[mask], y_resid[mask])[0, 1])
        r_sig = float(np.corrcoef(oof_sig_r[mask], y_resid[mask])[0, 1])
        # Test centered 0.5*combined residual correction
        c_mt = compute_metrics(y_price[mask], p4_pred[mask] * np.exp(0.5 * oof_comb_r[mask]))
        print(f"  {label:28s} (n={mask.sum():3d}) | r_vlm={r_vlm:+.3f} r_sig={r_sig:+.3f} | "
              f"P4={b_mt['mdape']:5.2f}% -> +Vis={c_mt['mdape']:5.2f}% (Δ={c_mt['mdape']-b_mt['mdape']:+.2f}pp, ΔMAE=€{c_mt['mae']-b_mt['mae']:+,.0f})")

    # =========================================================================
    # H3B: WHY DID LUXURY €650k+ FLIP NEGATIVE FOR VLM? (Root-Cause Audit!)
    # =========================================================================
    print("\n" + "=" * 95)
    print("H3B. LUXURY €650k+ SIGN-FLIP AUDIT — WHY DID vlm_charm CORRELATE -0.29 WITH P4 RESIDUAL?")
    print("=" * 95)
    lux = m['pilot_segment'] == 'luxury_650k_plus'
    lux_df = m[lux].copy()
    for c in vlm_cols:
        lux_df[c] = vlm_aligned.loc[lux, c].to_numpy()
    lux_df['p4_bias_pct'] = (lux_df['p4_oof_pred'] - lux_df['price']) / lux_df['price'] * 100
    print(f"  Luxury n={len(lux_df)} | corr(vlm_charm, price)={np.corrcoef(lux_df['vlm_charm'], lux_df['price'])[0,1]:+.3f} | "
          f"corr(vlm_charm, p4_pred)={np.corrcoef(lux_df['vlm_charm'], lux_df['p4_oof_pred'])[0,1]:+.3f} | "
          f"corr(vlm_charm, log_resid)={np.corrcoef(lux_df['vlm_charm'], lux_df['log_resid_p4'])[0,1]:+.3f}")
    print("  Top 5 Luxury P4 UNDER-predictions (actual >> P4 pred) — what did VLM score them?")
    cols_s = ['property_id', 'price', 'p4_oof_pred', 'livable_area', 'vlm_finish', 'vlm_charm', 'vlm_luxury', 'vlm_staging', 'vlm_reno_keur']
    print(lux_df.sort_values('log_resid_p4', ascending=False)[cols_s].head(5).to_string(index=False))
    print("  Top 5 Luxury P4 OVER-predictions (actual << P4 pred) — what did VLM score them?")
    print(lux_df.sort_values('log_resid_p4', ascending=True)[cols_s].head(5).to_string(index=False))

    # =========================================================================
    # H3C: NON-LINEAR / SEGMENT-AWARE VISION TEST (Can Trees/Interaction Models Beat Linear Ridge?)
    # =========================================================================
    print("\n" + "=" * 95)
    print("H3C. NON-LINEAR INTERACTION TEST — CAN SEGMENT-CONDITIONED TREES / RIDGE UNLOCK VISION LIFT?")
    print("=" * 95)
    # Build compact interaction feature matrix: P4 prediction context + VLM (10) + Q-Scores (14) + SigLIP2 OOF (1) + DINOv2 OOF (1) + Aerial CV (14)
    oof_dino_r = oof_ridge(X_dino_photo, y_resid, folds)
    oof_aer_r = oof_ridge(X_aerial_cv, y_resid, folds)
    ctx = np.column_stack([
        np.log(p4_pred),
        np.log(p4_pred / y_area),
        np.log(y_area),
        tr_sub['property_type_HOUSE'].to_numpy(float),
        (m['pilot_segment'] == 'historic_core').astype(float).to_numpy(),
        (m['pilot_segment'] == 'fixer_upper').astype(float).to_numpy(),
        (m['pilot_segment'] == 'luxury_650k_plus').astype(float).to_numpy(),
        cond_missing.astype(float),
        pilot_df['epc_ordinal'].fillna(3.5).to_numpy(float),
    ])
    X_nl_vlm = np.hstack([ctx, X_vlm])
    X_nl_all = np.hstack([ctx, X_vlm, X_qscore[:, :14], X_aerial_cv, oof_sig_r[:, None], oof_dino_r[:, None]])

    # 1. Segmented Ridge (fit separate Ridge inside <650k vs >=650k, strictly inside each training fold!)
    oof_seg_vlm = np.zeros(len(m))
    oof_seg_all = np.zeros(len(m))
    # 2. Shallow LGBM (Huber loss, max_depth=2..3, min_child_samples=35) predicting y_resid from context + vision
    oof_lgb_ctx_only = np.zeros(len(m))  # CONTROL! Does context alone (tail bias fix) explain any gain?
    oof_lgb_vlm = np.zeros(len(m))
    oof_lgb_all = np.zeros(len(m))

    for f in np.unique(folds):
        tr, va = folds != f, folds == f
        # Segmented Ridge by p4_pred < 600k vs >= 600k (no target leak — uses p4_pred!)
        for X_use, out_arr in [(X_vlm, oof_seg_vlm), (np.hstack([X_vlm, oof_sig_r[:, None]]), oof_seg_all)]:
            for hi in (False, True):
                m_tr = tr & ((p4_pred >= 600_000) == hi)
                m_va = va & ((p4_pred >= 600_000) == hi)
                if m_tr.sum() < 20 or m_va.sum() == 0:
                    continue
                med = np.nanmedian(X_use[m_tr], axis=0)
                med = np.where(np.isfinite(med), med, 0.0)
                Xtr_s = np.where(np.isfinite(X_use[m_tr]), X_use[m_tr], med)
                Xva_s = np.where(np.isfinite(X_use[m_va]), X_use[m_va], med)
                sc = StandardScaler()
                mu = np.median(y_resid[m_tr])
                mod = RidgeCV(alphas=np.logspace(0, 4, 20))
                mod.fit(sc.fit_transform(Xtr_s), y_resid[m_tr] - mu)
                out_arr[m_va] = mod.predict(sc.transform(Xva_s))

        # Shallow LGBM control vs +VLM vs +AllVision (important: compare against ctx-only control!)
        for X_use, out_arr in [(ctx, oof_lgb_ctx_only), (X_nl_vlm, oof_lgb_vlm), (X_nl_all, oof_lgb_all)]:
            reg = lgb.LGBMRegressor(
                objective='huber', alpha=0.15, n_estimators=60, learning_rate=0.03,
                num_leaves=6, max_depth=3, min_child_samples=40,
                subsample=0.8, colsample_bytree=0.7, reg_lambda=10.0,
                random_state=42 + int(f), n_jobs=4, verbose=-1,
            )
            reg.fit(X_use[tr], y_resid[tr])
            out_arr[va] = reg.predict(X_use[va])

    base_mt = compute_metrics(y_price, p4_pred)
    print(f"  {'Baseline P4 (no adjustment)':38s} | {fmt(base_mt)}")
    for name, pred_r in [
        ('CONTROL: LGBM Context-Only (no vision)', oof_lgb_ctx_only),
        ('Segmented Ridge (VLM-10, <600k/600k+)',  oof_seg_vlm),
        ('Segmented Ridge (VLM-10 + SigLIP2)',     oof_seg_all),
        ('Shallow LGBM (Context + VLM-10)',        oof_lgb_vlm),
        ('Shallow LGBM (Context + ALL Vision)',    oof_lgb_all),
    ]:
        # Evaluate pure orthogonal increment for LGBM (subtract ctx_only where applicable? Show both raw & delta vs control)
        mt = compute_metrics(y_price, p4_pred * np.exp(0.75 * pred_r))
        r = float(np.corrcoef(pred_r, y_resid)[0, 1])
        print(f"  {name:38s} | r={r:+.4f} | {fmt(mt)} | ΔMdAPE={mt['mdape']-base_mt['mdape']:+.2f}pp ΔMAE=€{mt['mae']-base_mt['mae']:+,.0f}")

    # =========================================================================
    # H4: DOES VISION-vs-TABULAR DISAGREEMENT PREDICT UNCERTAINTY / BIG ERRORS?
    # =========================================================================
    print("\n" + "=" * 95)
    print("H4. UNCERTAINTY & DEAL-DETECTION HYPOTHESIS — DOES VISION PREDICT |ERROR| / UNRELIABLE LISTINGS?")
    print("=" * 95)
    abs_ape = np.abs(p4_pred - y_price) / y_price * 100
    # Visual-vs-Tabular quality mismatch: VLM expected log(€/m²) vs P4 log(€/m²)
    vlm_psqm = oof_ridge(X_vlm, y_log_psqm, folds)
    vis_tab_gap = vlm_psqm - p4_log_psqm  # positive => looks BETTER visually than P4 thinks!
    # Staging / renovation extremity
    r_gap_signed = float(np.corrcoef(vis_tab_gap, y_resid)[0, 1])
    r_gap_abs = float(np.corrcoef(np.abs(vis_tab_gap), abs_ape)[0, 1])
    r_reno_abs = float(np.corrcoef(vlm_aligned['vlm_reno_need'].to_numpy(float), abs_ape)[0, 1])
    r_stag_abs = float(np.corrcoef(vlm_aligned['vlm_staging'].to_numpy(float), abs_ape)[0, 1])
    print(f"  corr(VLM_psqm - P4_psqm, signed_resid) = {r_gap_signed:+.4f}")
    print(f"  corr(|VLM_psqm - P4_psqm|, |APE%|)     = {r_gap_abs:+.4f}")
    print(f"  corr(vlm_reno_need, |APE%|)            = {r_reno_abs:+.4f}")
    print(f"  corr(vlm_staging, |APE%|)              = {r_stag_abs:+.4f}")
    for q_lo, q_hi, lbl in [(0, 25, 'Low reno_need (1-2)'), (25, 75, 'Mid reno_need'), (75, 100, 'High reno_need (4-9)')]:
        lo_v, hi_v = np.percentile(vlm_aligned['vlm_reno_need'], [q_lo, q_hi])
        mk = (vlm_aligned['vlm_reno_need'] >= lo_v) & (vlm_aligned['vlm_reno_need'] <= hi_v)
        print(f"    {lbl:24s} (n={mk.sum():3d}): P4 MdAPE={np.median(abs_ape[mk]):5.2f}% | P90 APE={np.percentile(abs_ape[mk],90):5.1f}%")

    # =========================================================================
    # H5: ABLATION INSIDE A FULL TABULAR MODEL (WHAT IF TEXT WERE WEAK/ABSENT?)
    # =========================================================================
    print("\n" + "=" * 95)
    print("H5. FULL MODALITY LADDER (5-fold OOF RidgeCV on log €/m², n=790) — ORDER-SENSITIVE CONTRIBUTION")
    print("=" * 95)
    ladders = [
        ('Tabular only',                             X_struct),
        ('Tabular + Spatial/OpenData',               np.hstack([X_struct, X_spatial])),
        ('Tabular + Spatial + Vision (VLM+Q+AerCV)', np.hstack([X_struct, X_spatial, X_vlm, X_qscore, X_aerial_cv])),
        ('Tabular + Spatial + Text',                 np.hstack([X_struct, X_spatial, X_text_summary])),
        ('Tabular + Spatial + Text + VLM(10)',       np.hstack([X_struct, X_spatial, X_text_summary, X_vlm])),
        ('Tabular + Spatial + Text + VisionCompact', np.hstack([X_struct, X_spatial, X_text_summary, X_vlm, X_qscore, X_aerial_cv])),
    ]
    for name, Xl in ladders:
        p = oof_ridge(Xl, y_log_psqm, folds)
        mt = compute_metrics(y_price, np.exp(p) * y_area)
        print(f"  {name:44s} | R²(€/m²)={r2_score(y_log_psqm, p):.4f} | MdAPE={mt['mdape']:5.2f}% | Hit@10={mt['hit_10']:4.1f}% | MAE=€{mt['mae']:,.0f}")


if __name__ == '__main__':
    main()
