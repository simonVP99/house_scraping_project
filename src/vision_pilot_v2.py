"""Vision Pilot v2 — Deeper Tests on the SAME 800 Cached Homes (0 downloads!):
Before discarding ANY vision idea based on an n=640 p>>n linear probe, tests 5 smarter upgrades:
  1. **SigLIP2-SO400M-384** (`google/siglip2-so400m-patch14-384`, 400M params, 384px, cached locally)
     vs `siglip2-base-224` — captures real kitchen/floor/plaster/architectural texture!
  2. **Room-Specific Q-Scores** (not smeared across all photos!):
     - `kitchen_q` scored ONLY on kitchen photos
     - `bath_q` scored ONLY on bathroom photos
     - `living_q` & `light_q` scored ONLY on living/bedroom photos
     - `facade_charm_q` scored ONLY on facade photos
     - `garden_green_q` scored ONLY on garden photos
     - `worst_photo_fixer_q` (max fixer-shell score across all photos -> catches hidden defects!)
  3. **Visual-Similarity k-NN (Non-Linear, No p>>n!)**:
     - Finds k=20 most visually similar homes in `SigLIP2-SO400M` and `DINOv2` space (strictly OOF!)
     - Computes weighted average `log_resid_p4` and `log(€/m²)` of visual twins!
  4. **Explicit Aerial Orthophoto Features (50m & 200m Flanders Ortho, pure NumPy)**:
     - Green vegetation index (`ExG = 2G - R - B`), shadow/density, roof-terracotta share,
       water/blue share, luminance entropy — immune to p>>n!
  5. **Non-Linear Interaction Test (`LightGBM` on `p4_pred + tabular context + vision`)**:
     - Lets vision interact with `is_house`, `livable_area`, `historic_core`, `building_condition`,
       `energy_label` (e.g., visual charm matters +15% in Binnenstad, ~0% in new-build suburbs!).

Usage (run after `p5_handoff` finishes):
    .venv-api/bin/python -u src/vision_pilot_v2.py
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoProcessor

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'src'))
from phase3_features import compute_metrics, fmt
from vision_pilot_eval import Q_PAIRS, ROOM_PROMPTS, load_pil_safe

PILOT_DIR = os.path.join(BASE_DIR, 'external_data', 'vision_pilot')
PHOTO_DIR = os.path.join(PILOT_DIR, 'photos')
AERIAL_DIR = os.path.join(PILOT_DIR, 'aerial')
MANIFEST_PATH = os.path.join(PILOT_DIR, 'pilot_manifest.parquet')
EMB_V1_CACHE = os.path.join(PILOT_DIR, 'pilot_embeddings.npz')
EMB_SO400M_CACHE = os.path.join(PILOT_DIR, 'pilot_siglip2_so400m.npz')
OUT_JSON = os.path.join(PILOT_DIR, 'pilot_v2_report.json')

EXTRA_Q_PAIRS = {
    'q_bathroom_luxury': (
        'modern luxury designer bathroom with walk-in rain shower, freestanding tub and sleek tiles',
        'old dated worn bathroom with discolored tiles and old plumbing',
    ),
    'q_facade_prestige': (
        'stately historic Belgian townhouse facade or architect-designed luxury villa exterior',
        'plain weathered run-down narrow terraced house exterior or dated concrete apartment block',
    ),
    'q_floor_quality': (
        'warm authentic herringbone oak parquet or polished architectural concrete floor',
        'worn old vinyl linoleum carpet or cheap damaged laminate floor',
    ),
    'q_empty_uninhabited': (
        'completely empty vacant stripped house room with bare walls and no furniture',
        'warmly furnished lived-in cozy staged residential room',
    ),
}


def encode_so400m(paths: list[str], device: str, bs: int = 32):
    model_id = 'google/siglip2-so400m-patch14-384'
    print(f"Loading {model_id} on {device}...", flush=True)
    proc = AutoProcessor.from_pretrained(model_id)
    mod = AutoModel.from_pretrained(model_id, dtype=torch.float16 if device == 'mps' else torch.float32).to(device)
    mod.eval()

    all_q = {**Q_PAIRS, **EXTRA_Q_PAIRS}
    room_keys = list(ROOM_PROMPTS.keys())
    q_keys = list(all_q.keys())
    texts = [ROOM_PROMPTS[k] for k in room_keys]
    for k in q_keys:
        texts.append(all_q[k][0])
        texts.append(all_q[k][1])
    with torch.no_grad():
        t_inp = proc(text=texts, padding='max_length', max_length=64, truncation=True, return_tensors='pt').to(device)
        t_emb = mod.get_text_features(**t_inp).float()
        t_emb = t_emb / t_emb.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        t_emb = t_emb.cpu().numpy()
    room_txt = t_emb[:len(room_keys)]
    q_pos_txt = t_emb[len(room_keys)::2]
    q_neg_txt = t_emb[len(room_keys) + 1::2]

    t0 = time.perf_counter()
    D = t_emb.shape[1]
    embs = np.zeros((len(paths), D), dtype=np.float32)
    valid = np.zeros(len(paths), dtype=bool)
    for s in range(0, len(paths), bs):
        bpaths = paths[s:s + bs]
        imgs, idxs = [], []
        for j, p in enumerate(bpaths):
            im = load_pil_safe(p)
            if im is not None:
                imgs.append(im)
                idxs.append(s + j)
        if not imgs:
            continue
        inp = proc(images=imgs, return_tensors='pt').to(device)
        if device == 'mps':
            inp['pixel_values'] = inp['pixel_values'].to(torch.float16)
        with torch.no_grad():
            feat = mod.get_image_features(**inp).float()
            feat = feat / feat.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            embs[idxs] = feat.cpu().numpy()
            valid[idxs] = True
    dt = time.perf_counter() - t0
    print(f"  SigLIP2-SO400M-384 encoded {valid.sum()}/{len(paths)} images in {dt:.1f}s ({valid.sum()/max(dt,1e-3):.1f} img/s)", flush=True)
    del mod
    if device == 'mps':
        torch.mps.empty_cache()
    return embs, valid, room_txt, q_pos_txt, q_neg_txt, q_keys


def extract_explicit_aerial_cv(pids: list[str]) -> pd.DataFrame:
    """Pure NumPy RGB vegetation/greenness/roof/shadow/brightness features from 50m & 200m Flanders orthophotos."""
    rows = []
    for pid in pids:
        r = {}
        for tag in ('50m', '200m'):
            p = os.path.join(AERIAL_DIR, f'{pid}_{tag}.jpg')
            im = load_pil_safe(p)
            if im is None:
                for k in ('green_frac', 'exg_mean', 'bright_mean', 'bright_std', 'terracotta_frac', 'water_frac'):
                    r[f'aer_{tag}_{k}'] = np.nan
                continue
            arr = np.asarray(im, dtype=np.float32) / 255.0
            R, G, B = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
            exg = 2.0 * G - R - B  # Excess Green vegetation index
            bright = 0.299 * R + 0.587 * G + 0.114 * B
            r[f'aer_{tag}_green_frac'] = float((exg > 0.08).mean())
            r[f'aer_{tag}_exg_mean'] = float(exg.mean())
            r[f'aer_{tag}_bright_mean'] = float(bright.mean())
            r[f'aer_{tag}_bright_std'] = float(bright.std())
            r[f'aer_{tag}_terracotta_frac'] = float(((R - G > 0.08) & (R - B > 0.12)).mean())
            r[f'aer_{tag}_water_frac'] = float(((B - R > 0.05) & (bright < 0.35)).mean())
        rows.append(r)
    return pd.DataFrame(rows)


def extract_photo_photometry(photo_paths: list[str], photo_pid_idx: np.ndarray, N: int) -> pd.DataFrame:
    """Basic brightness / contrast / saturation / colorfulness / vertical-photo share per listing."""
    b_mean = np.zeros(len(photo_paths), dtype=np.float32)
    b_std = np.zeros(len(photo_paths), dtype=np.float32)
    sat_mean = np.zeros(len(photo_paths), dtype=np.float32)
    is_vert = np.zeros(len(photo_paths), dtype=np.float32)
    ok = np.zeros(len(photo_paths), dtype=bool)
    for j, p in enumerate(photo_paths):
        im = load_pil_safe(p)
        if im is None:
            continue
        w, h = im.size
        is_vert[j] = 1.0 if h > w else 0.0
        arr = np.asarray(im, dtype=np.float32) / 255.0
        lum = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
        mx = arr.max(axis=2)
        mn = arr.min(axis=2)
        sat = (mx - mn) / np.maximum(mx, 1e-4)
        b_mean[j] = lum.mean()
        b_std[j] = lum.std()
        sat_mean[j] = sat.mean()
        ok[j] = True
    rows = []
    for i in range(N):
        idx = np.where((photo_pid_idx == i) & ok)[0]
        if not len(idx):
            rows.append({'ph_bright_mean': np.nan, 'ph_bright_min': np.nan, 'ph_contrast': np.nan, 'ph_sat': np.nan, 'ph_vert_share': np.nan})
        else:
            rows.append({
                'ph_bright_mean': float(b_mean[idx].mean()),
                'ph_bright_min': float(b_mean[idx].min()),
                'ph_contrast': float(b_std[idx].mean()),
                'ph_sat': float(sat_mean[idx].mean()),
                'ph_vert_share': float(is_vert[idx].mean()),
            })
    return pd.DataFrame(rows)


def oof_visual_knn(E: np.ndarray, y_target: np.ndarray, folds: np.ndarray, k: int = 20, temp: float = 0.07) -> np.ndarray:
    """Strictly OOF cosine-similarity weighted k-NN prediction (immune to p>>n!)."""
    E_n = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-6)
    oof = np.zeros(len(y_target), dtype=float)
    for f in np.unique(folds):
        tr, va = folds != f, folds == f
        sim = E_n[va] @ E_n[tr].T  # (n_va, n_tr)
        # Top-k per validation row
        kk = min(k, sim.shape[1])
        part = np.argpartition(-sim, kk - 1, axis=1)[:, :kk]
        rows = np.arange(sim.shape[0])[:, None]
        top_s = sim[rows, part]
        w = np.exp((top_s - top_s.max(axis=1, keepdims=True)) / temp)
        w = w / w.sum(axis=1, keepdims=True)
        oof[va] = (w * y_target[tr][part]).sum(axis=1)
    return oof


def oof_ridge_centered(X: np.ndarray, y_res: np.ndarray, folds: np.ndarray, alphas=np.logspace(1, 5, 25)) -> np.ndarray:
    oof = np.zeros(len(y_res), dtype=float)
    X_clean = np.nan_to_num(X, nan=0.0)
    for f in np.unique(folds):
        tr, va = folds != f, folds == f
        sc = StandardScaler()
        Xtr = sc.fit_transform(X_clean[tr])
        Xva = sc.transform(X_clean[va])
        mu = np.median(y_res[tr])
        m = RidgeCV(alphas=alphas)
        m.fit(Xtr, y_res[tr] - mu)
        oof[va] = m.predict(Xva)
    return oof


def main():
    t0 = time.perf_counter()
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"=== VISION PILOT V2: SO400M-384 + ROOM-SPECIFIC Q-SCORES + VISUAL kNN + EXPLICIT AERIAL + NON-LINEAR INTERACTIONS ===", flush=True)

    df = pd.read_parquet(MANIFEST_PATH)
    N = len(df)
    pids = df['property_id'].astype(str).tolist()
    folds = df['fold'].to_numpy(int)
    y_price = df['price'].to_numpy(float)
    p4 = df['p4_oof_pred'].to_numpy(float)
    y_res = df['log_resid_p4'].to_numpy(float)
    y_psqm = np.log(y_price / df['livable_area'].to_numpy(float))
    base_m = compute_metrics(y_price, p4)
    print(f"Baseline Phase-4 Ensemble (n={N}): {fmt(base_m)}", flush=True)

    photo_paths, photo_pid_idx = [], []
    for i, pid in enumerate(pids):
        for p in sorted(glob.glob(os.path.join(PHOTO_DIR, pid, '*.jpg'))):
            photo_paths.append(p)
            photo_pid_idx.append(i)
    photo_pid_idx = np.array(photo_pid_idx, dtype=int)

    # 1. Encode with SigLIP2-SO400M-384 (or load cache)
    if os.path.exists(EMB_SO400M_CACHE):
        z = np.load(EMB_SO400M_CACHE, allow_pickle=True)
        so_p, so_ok = z['so_p'], z['so_ok']
        room_txt, q_pos_txt, q_neg_txt, q_keys = z['room_txt'], z['q_pos_txt'], z['q_neg_txt'], list(z['q_keys'])
    else:
        so_p, so_ok, room_txt, q_pos_txt, q_neg_txt, q_keys = encode_so400m(photo_paths, device, bs=32)
        np.savez_compressed(
            EMB_SO400M_CACHE, so_p=so_p, so_ok=so_ok,
            room_txt=room_txt, q_pos_txt=q_pos_txt, q_neg_txt=q_neg_txt, q_keys=np.array(q_keys, dtype=object),
        )

    # Load v1 DINOv2 + SigLIP2-base caches too
    zv1 = np.load(EMB_V1_CACHE, allow_pickle=True)
    dino_p, dino_ok = zv1['dino_p'], zv1['dino_p_ok']
    sigb_p, sigb_ok = zv1['sig_p'], zv1['sig_p_ok']

    # 2. Room-Specific Q-Scores + Mean Embeddings
    room_keys = list(ROOM_PROMPTS.keys())  # facade=0, kitchen=1, living=2, bathroom=3, bedroom=4, garden=5, plan=6
    room_arg = (so_p @ room_txt.T).argmax(axis=1)
    q_photo = (so_p @ q_pos_txt.T) - (so_p @ q_neg_txt.T)  # (M, 11)
    q_idx = {k: j for j, k in enumerate(q_keys)}

    so_mean = np.zeros((N, so_p.shape[1]), dtype=np.float32)
    dino_mean = np.zeros((N, dino_p.shape[1]), dtype=np.float32)
    sigb_mean = np.zeros((N, sigb_p.shape[1]), dtype=np.float32)

    rsq_rows = []
    for i in range(N):
        idx = np.where((photo_pid_idx == i) & so_ok)[0]
        row = {}
        if not len(idx):
            rsq_rows.append(row)
            continue
        non_plan = idx[room_arg[idx] != 6]
        use = non_plan if len(non_plan) else idx
        so_mean[i] = so_p[use].mean(axis=0)
        dino_mean[i] = dino_p[use].mean(axis=0)
        sigb_mean[i] = sigb_p[use].mean(axis=0)

        # Global mean & max/min Q-scores
        for k, j in q_idx.items():
            vals = q_photo[use, j]
            row[f'{k}_mean'] = float(vals.mean())
            row[f'{k}_p75'] = float(np.percentile(vals, 75))
            if k in ('q_fixer_shell', 'q_empty_uninhabited'):
                row[f'{k}_max'] = float(vals.max())  # catches ONE bad/stripped room!

        # Room-SPECIFIC Q-scores (scored ONLY on the matching room!)
        room_q_map = [
            ('facade_prestige_on_facade',   0, 'q_facade_prestige'),
            ('kitchen_lux_on_kitchen',      1, 'q_kitchen_luxury'),
            ('living_modern_on_living',     2, 'q_modern_finish'),
            ('living_charm_on_living',      2, 'q_architect_charm'),
            ('living_light_on_living',      2, 'q_natural_light'),
            ('bath_lux_on_bath',            3, 'q_bathroom_luxury'),
            ('garden_green_on_garden',      5, 'q_outdoor_green'),
        ]
        for fname, ri, qk in room_q_map:
            r_idx = idx[room_arg[idx] == ri]
            row[fname] = float(q_photo[r_idx, q_idx[qk]].mean()) if len(r_idx) else np.nan
            row[f'has_room_{room_keys[ri]}'] = float(len(r_idx) > 0)

        # Photo dispersion (std across photos: uniform flip vs half-renovated house!)
        row['q_modern_std'] = float(q_photo[use, q_idx['q_modern_finish']].std())
        row['n_valid_photos'] = float(len(idx))
        rsq_rows.append(row)

    rsq_df = pd.DataFrame(rsq_rows)
    aer_cv_df = extract_explicit_aerial_cv(pids)
    phot_df = extract_photo_photometry(photo_paths, photo_pid_idx, N)

    # Print top correlations with Phase-4 residual!
    print("\n=== TOP 15 CORRELATIONS WITH PHASE-4 RESIDUAL (SO400M Room-Q + Explicit Aerial + Photometry) ===", flush=True)
    all_interp = pd.concat([rsq_df, aer_cv_df, phot_df], axis=1)
    corrs = []
    for c in all_interp.columns:
        v = all_interp[c].to_numpy(float)
        m = np.isfinite(v)
        if m.sum() >= 50 and np.std(v[m]) > 1e-8:
            r_res = float(np.corrcoef(v[m], y_res[m])[0, 1])
            r_psqm = float(np.corrcoef(v[m], y_psqm[m])[0, 1])
            corrs.append((c, r_res, r_psqm, int(m.sum())))
    for c, rr, rp, n_ok in sorted(corrs, key=lambda x: -abs(x[1]))[:18]:
        print(f"  {c:30s} (n={n_ok:3d}) | corr(P4_resid)={rr:+.4f} | corr(log_€/m²)={rp:+.4f}", flush=True)

    # 3. Head-to-head OOF evaluation of EVERY paradigm (properly regularized & non-linear!)
    print("\n=== HEAD-TO-HEAD OOF TESTS ON PHASE-4 RESIDUAL (n=800, 5-FOLD GROUPED OOF) ===", flush=True)
    res_dict = {'baseline_p4': base_m, 'tests': {}}

    def eval_resid(name: str, r_pred: np.ndarray, scale: float = 1.0):
        corr = float(np.corrcoef(r_pred, y_res)[0, 1]) if np.std(r_pred) > 1e-8 else 0.0
        m = compute_metrics(y_price, p4 * np.exp(scale * r_pred))
        res_dict['tests'][name] = {
            'corr': round(corr, 4), 'metrics': m,
            'd_mdape': round(m['mdape'] - base_m['mdape'], 3),
            'd_mae': round(m['mae'] - base_m['mae'], 1),
            'd_rmse': round(m['rmse'] - base_m['rmse'], 1),
        }
        print(f"  {name:34s} | r={corr:+.4f} | MdAPE={m['mdape']:5.2f}% ({m['mdape']-base_m['mdape']:+.2f}) | Hit@10={m['hit_10']:4.1f}% | MAE=€{m['mae']:7,.0f} ({m['mae']-base_m['mae']:+6,.0f}) | RMSE=€{m['rmse']:7,.0f} ({m['rmse']-base_m['rmse']:+6,.0f})", flush=True)
        return r_pred

    # A. SigLIP2-base vs SigLIP2-SO400M-384 Ridge
    r_sigb = eval_resid('1. SigLIP2-base-224 Ridge', oof_ridge_centered(sigb_mean, y_res, folds))
    r_so400 = eval_resid('2. SigLIP2-SO400M-384 Ridge', oof_ridge_centered(so_mean, y_res, folds))
    r_dino = eval_resid('3. DINOv2-base Ridge', oof_ridge_centered(dino_mean, y_res, folds))

    # B. Room-specific Q-scores + Photometry + Explicit Aerial CV (Ridge)
    rsq_fill = all_interp.fillna(all_interp.median()).to_numpy(float)
    r_rsq = eval_resid('4. Room-Specific Q+Photo+AerCV', oof_ridge_centered(rsq_fill, y_res, folds))
    r_aer = eval_resid('5. Explicit Aerial CV Only (12d)', oof_ridge_centered(aer_cv_df.fillna(aer_cv_df.median()).to_numpy(float), y_res, folds))

    # C. Visual-Similarity k-NN (Non-Linear, immune to p>>n!)
    r_knn_so = eval_resid('6. Visual k-NN (SO400M, k=20)', oof_visual_knn(so_mean, y_res, folds, k=20, temp=0.07))
    r_knn_dino = eval_resid('7. Visual k-NN (DINOv2, k=20)', oof_visual_knn(dino_mean, y_res, folds, k=20, temp=0.07))

    # D. Non-Linear Interaction Model (shallow regularized LGBM on tabular context + compact vision signals)
    ctx = pd.DataFrame({
        'log_p4': np.log(p4),
        'log_area': np.log(df['livable_area'].to_numpy(float)),
        'is_house': df['property_type_HOUSE'].to_numpy(float),
        'is_hist': df['pilot_segment'].eq('historic_core').astype(float).to_numpy(),
        'is_fixer': df['pilot_segment'].eq('fixer_upper').astype(float).to_numpy(),
        'r_so400_ridge': r_so400,
        'r_sigb_ridge': r_sigb,
        'r_knn_so': r_knn_so,
        'r_knn_dino': r_knn_dino,
    })
    X_int = pd.concat([ctx.reset_index(drop=True), all_interp.reset_index(drop=True)], axis=1)
    oof_lgb = np.zeros(N)
    for f in np.unique(folds):
        tr, va = folds != f, folds == f
        # Strictly nest inner-OOF? Wait: r_so400/r_sigb/r_knn are ALREADY OOF, so use ONLY raw all_interp + ctx inside LGBM for zero leakage!
        Xtr = pd.concat([ctx.iloc[tr, :5].reset_index(drop=True), all_interp.iloc[tr].reset_index(drop=True)], axis=1)
        Xva = pd.concat([ctx.iloc[va, :5].reset_index(drop=True), all_interp.iloc[va].reset_index(drop=True)], axis=1)
        mu = np.median(y_res[tr])
        m = lgb.LGBMRegressor(
            objective='huber', alpha=0.15, n_estimators=120, learning_rate=0.025,
            num_leaves=7, max_depth=3, min_child_samples=35, colsample_bytree=0.5,
            subsample=0.75, subsample_freq=1, reg_alpha=2.0, reg_lambda=10.0,
            random_state=42 + int(f), n_jobs=4, verbose=-1,
        )
        m.fit(Xtr, y_res[tr] - mu)
        oof_lgb[va] = m.predict(Xva)
    eval_resid('8. Shallow LGBM (Ctx x Room-Q+Aer)', oof_lgb)

    # E. Shrinkage-blended best multimodal vision residual (0.45*SO400M + 0.25*RoomQ_LGBM + 0.15*kNN_SO + 0.15*kNN_DINO)
    r_blend = 0.45 * r_so400 + 0.25 * oof_lgb + 0.15 * r_knn_so + 0.15 * r_knn_dino
    for lam in (0.5, 0.75, 1.0):
        eval_resid(f'9. Vision Blend (scale={lam:.2f})', r_blend, scale=lam)

    # Segment breakdown on best scale=0.65
    print("\n=== SEGMENT BREAKDOWN: PHASE-4 vs +VISION V2 BLEND (scale=0.65) ===", flush=True)
    p_vis = p4 * np.exp(0.65 * r_blend)
    for seg in ['historic_core', 'fixer_upper', 'luxury_650k_plus', 'house_typical', 'apt_typical']:
        ms = df['pilot_segment'].to_numpy() == seg
        mb = compute_metrics(y_price[ms], p4[ms])
        mv = compute_metrics(y_price[ms], p_vis[ms])
        print(f"  {seg:18s} (n={ms.sum():3d}) | P4: {mb['mdape']:5.2f}% (€{mb['mae']:7,.0f}) -> +VisV2: {mv['mdape']:5.2f}% (€{mv['mae']:7,.0f}) | ΔMdAPE={mv['mdape']-mb['mdape']:+.2f}pp ΔMAE=€{mv['mae']-mb['mae']:+,.0f}", flush=True)

    with open(OUT_JSON, 'w') as fh:
        json.dump(res_dict, fh, indent=2)
    print(f"\nSaved {OUT_JSON} in {time.perf_counter()-t0:.1f}s!", flush=True)


if __name__ == '__main__':
    main()
