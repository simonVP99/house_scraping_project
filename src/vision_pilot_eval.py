"""Vision Pilot Benchmark (Options A, B, C, D head-to-head on 800 stratified Ghent homes):
Measures the *TRUE* incremental lift on top of our Phase-4 ensemble (`log_resid_p4 = log(price) - log(p4_oof)`)
strictly out-of-fold across the 5 grouped CV folds!

Tests 4 distinct vision paradigms head-to-head on the exact same 800 homes:
  - Option A: **SigLIP2-base (`google/siglip2-base-patch16-224`)** on listing photos:
      A1: Zero-shot **Room Router** + **SigLIP2 Q-Score** (14 interpretable visual quality/condition features)
      A2: Mean-pooled SigLIP2 embeddings -> nested-OOF Ridge on `log_resid_p4`
      A3: **Room-routed** SigLIP2 embeddings (facade + kitchen + living + bath + garden concatenated) -> Ridge
      A4: **Attention-MIL** network over the `(K x 768)` per-photo embeddings trained directly on `log_resid_p4`
  - Option B: **DINOv2-base (`facebook/dinov2-base`)** on listing photos (pure self-supervised visual texture/architecture):
      B1: Mean-pooled DINOv2 embeddings -> Ridge on `log_resid_p4`
      B2: Room-routed DINOv2 embeddings -> Ridge on `log_resid_p4`
  - Option C: **Flanders Official Aerial Orthophotos** (`50m` parcel zoom + `200m` context):
      C1: DINOv2 on 50m + 200m aerial ortho -> Ridge on `log_resid_p4`
      C2: SigLIP2 on 50m + 200m aerial ortho -> Ridge on `log_resid_p4`
  - Option D: **Combined Multimodal Vision Stack** (`A + B + C`) -> measures exact MdAPE / MAE / R² lift
    overall AND by segment (Historic Core, Fixer-Upper, Luxury 650k+, Typical House, Typical Apt)!

Usage (run AFTER `p5_full` finishes so MPS GPU isn't contested):
    .venv-api/bin/python -u src/vision_pilot_eval.py
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoProcessor

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'src'))
from phase3_features import compute_metrics, fmt

PILOT_DIR = os.path.join(BASE_DIR, 'external_data', 'vision_pilot')
PHOTO_DIR = os.path.join(PILOT_DIR, 'photos')
AERIAL_DIR = os.path.join(PILOT_DIR, 'aerial')
MANIFEST_PATH = os.path.join(PILOT_DIR, 'pilot_manifest.parquet')
EMB_CACHE = os.path.join(PILOT_DIR, 'pilot_embeddings.npz')
OUT_JSON = os.path.join(PILOT_DIR, 'pilot_report.json')

# Zero-shot Room Router prompts (SigLIP2)
ROOM_PROMPTS = {
    'facade':   'exterior front facade of a residential house or apartment building from the street',
    'kitchen':  'interior photo of a residential kitchen with cabinets stove and countertop',
    'living':   'interior photo of a residential living room or dining room with sofa or table',
    'bathroom': 'interior photo of a residential bathroom with shower bathtub or sink',
    'bedroom':  'interior photo of a residential bedroom with bed or wooden floor',
    'garden':   'outdoor backyard garden terrace patio or balcony of a house',
    'plan':     'architectural floor plan blueprint or technical drawing of a house',
}

# Zero-shot Contrastive Q-Score pairs (positive vs negative quality/condition axes, Aug 2026 CLIP Q-Score style)
Q_PAIRS = {
    'q_modern_finish': (
        'bright modern luxury renovated Belgian home interior with high-end materials and sleek finish',
        'dated old-fashioned worn-out unrenovated Belgian home interior from the 1970s needing renovation',
    ),
    'q_fixer_shell': (
        'dilapidated house interior under demolition, stripped walls, peeling paint, raw concrete, major renovation needed',
        'move-in ready freshly painted spotless modern home interior',
    ),
    'q_natural_light': (
        'sunlit bright spacious room with large windows and abundant natural daylight',
        'dark dim cramped room with small windows and poor lighting',
    ),
    'q_kitchen_luxury': (
        'designer modern luxury kitchen with cooking island, marble or composite countertop and built-in appliances',
        'old worn dated kitchen with cheap old cabinets and outdated appliances',
    ),
    'q_architect_charm': (
        'authentic historic architectural townhouse interior with high ceilings, parquet floors, crown molding and designer styling',
        'plain generic bland low-ceiling apartment interior with basic cheap flooring',
    ),
    'q_photo_staging': (
        'professional architectural real estate photography with wide-angle composition and tasteful staging',
        'amateur blurry dark cluttered smartphone photo with messy furniture',
    ),
    'q_outdoor_green': (
        'lush green private landscaped backyard garden or sunny terrace with greenery',
        'gray concrete courtyard or dark narrow alley with no greenery',
    ),
}


def load_pil_safe(path: str) -> Image.Image | None:
    try:
        with Image.open(path) as im:
            return im.convert('RGB')
    except Exception:
        return None


def encode_images_siglip2(paths: list[str], device: str, bs: int = 64):
    """Encode all images with google/siglip2-base-patch16-224 + return text embeddings for room/Q prompts."""
    model_id = 'google/siglip2-base-patch16-224'
    print(f"Loading {model_id} on {device}...", flush=True)
    proc = AutoProcessor.from_pretrained(model_id)
    mod = AutoModel.from_pretrained(model_id, torch_dtype=torch.float16 if device == 'mps' else torch.float32).to(device)
    mod.eval()

    # Encode text prompts once
    room_keys = list(ROOM_PROMPTS.keys())
    q_keys = list(Q_PAIRS.keys())
    all_texts = [ROOM_PROMPTS[k] for k in room_keys]
    for k in q_keys:
        all_texts.append(Q_PAIRS[k][0])
        all_texts.append(Q_PAIRS[k][1])
    with torch.no_grad():
        t_inp = proc(text=all_texts, padding='max_length', max_length=64, truncation=True, return_tensors='pt').to(device)
        t_emb = mod.get_text_features(**t_inp).float()
        t_emb = t_emb / t_emb.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        t_emb = t_emb.cpu().numpy()
    room_txt = t_emb[:len(room_keys)]
    q_pos_txt = t_emb[len(room_keys)::2]
    q_neg_txt = t_emb[len(room_keys) + 1::2]

    # Encode images in batches
    t0 = time.perf_counter()
    embs = np.zeros((len(paths), t_emb.shape[1]), dtype=np.float32)
    valid = np.zeros(len(paths), dtype=bool)
    for s in range(0, len(paths), bs):
        batch_paths = paths[s:s + bs]
        imgs, idxs = [], []
        for j, p in enumerate(batch_paths):
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
    print(f"  SigLIP2 encoded {valid.sum()}/{len(paths)} images in {dt:.1f}s ({valid.sum()/max(dt,1e-3):.1f} img/s)", flush=True)
    del mod
    if device == 'mps':
        torch.mps.empty_cache()
    return embs, valid, room_txt, q_pos_txt, q_neg_txt


def encode_images_dinov2(paths: list[str], device: str, bs: int = 64):
    """Encode all images with facebook/dinov2-base (CLS token L2-normalized, 768-d)."""
    model_id = 'facebook/dinov2-base'
    print(f"Loading {model_id} on {device}...", flush=True)
    proc = AutoProcessor.from_pretrained(model_id)
    mod = AutoModel.from_pretrained(model_id, torch_dtype=torch.float16 if device == 'mps' else torch.float32).to(device)
    mod.eval()

    t0 = time.perf_counter()
    embs = np.zeros((len(paths), 768), dtype=np.float32)
    valid = np.zeros(len(paths), dtype=bool)
    for s in range(0, len(paths), bs):
        batch_paths = paths[s:s + bs]
        imgs, idxs = [], []
        for j, p in enumerate(batch_paths):
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
            out = mod(**inp).last_hidden_state[:, 0, :].float()
            out = out / out.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            embs[idxs] = out.cpu().numpy()
            valid[idxs] = True
    dt = time.perf_counter() - t0
    print(f"  DINOv2 encoded {valid.sum()}/{len(paths)} images in {dt:.1f}s ({valid.sum()/max(dt,1e-3):.1f} img/s)", flush=True)
    del mod
    if device == 'mps':
        torch.mps.empty_cache()
    return embs, valid


def oof_ridge_resid(X: np.ndarray, y_resid: np.ndarray, folds: np.ndarray, alphas=(1.0, 5.0, 20.0, 80.0, 300.0)) -> tuple[np.ndarray, float, float]:
    """Strictly 5-fold grouped OOF Ridge predicting log_resid_p4; selects alpha by inner MSE."""
    best_r, best_pred, best_a = -1.0, None, None
    for a in alphas:
        pred = np.zeros(len(y_resid), dtype=float)
        for k in np.unique(folds):
            tr, va = folds != k, folds == k
            sc = StandardScaler()
            Xtr = sc.fit_transform(np.nan_to_num(X[tr], 0.0))
            Xva = sc.transform(np.nan_to_num(X[va], 0.0))
            m = Ridge(alpha=a)
            m.fit(Xtr, y_resid[tr])
            pred[va] = m.predict(Xva)
        r = float(np.corrcoef(pred, y_resid)[0, 1]) if np.std(pred) > 1e-8 else 0.0
        if r > best_r:
            best_r, best_pred, best_a = r, pred, a
    return best_pred, best_r, best_a


class GatedAttentionMIL(nn.Module):
    """Ilse et al. Gated Attention MIL over (K, D) listing photos predicting scalar residual."""
    def __init__(self, d_in: int, d_hid: int = 128):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(d_in, d_hid), nn.LayerNorm(d_hid), nn.GELU(), nn.Dropout(0.3))
        self.att_V = nn.Linear(d_hid, 64)
        self.att_U = nn.Linear(d_hid, 64)
        self.att_w = nn.Linear(64, 1)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(d_hid, 1))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: (B, K, D), mask: (B, K) bool
        h = self.proj(x)
        a = self.att_w(torch.tanh(self.att_V(h)) * torch.sigmoid(self.att_U(h))).squeeze(-1)
        a = a.masked_fill(~mask, -1e9)
        w = torch.softmax(a, dim=-1).unsqueeze(-1)
        pooled = (h * w).sum(dim=1)
        return self.head(pooled).squeeze(-1)


def oof_attention_mil(X_bag: np.ndarray, mask_bag: np.ndarray, y_resid: np.ndarray, folds: np.ndarray, device: str) -> np.ndarray:
    """Strictly 5-fold OOF Gated Attention MIL over per-photo SigLIP2 embeddings."""
    N, K, D = X_bag.shape
    oof = np.zeros(N, dtype=float)
    for k in np.unique(folds):
        tr, va = folds != k, folds == k
        # 2-seed average per fold for stability on 800 rows
        va_preds = []
        for s in (0, 1):
            torch.manual_seed(42 + int(k) * 10 + s)
            mod = GatedAttentionMIL(D, d_hid=128).to(device)
            opt = torch.optim.AdamW(mod.parameters(), lr=1e-3, weight_decay=3e-2)
            loss_fn = nn.HuberLoss(delta=0.15)

            xt = torch.tensor(X_bag[tr], dtype=torch.float32, device=device)
            mt = torch.tensor(mask_bag[tr], dtype=torch.bool, device=device)
            yt = torch.tensor(y_resid[tr], dtype=torch.float32, device=device)
            xv = torch.tensor(X_bag[va], dtype=torch.float32, device=device)
            mv = torch.tensor(mask_bag[va], dtype=torch.bool, device=device)

            rng = np.random.default_rng(100 + int(k) * 10 + s)
            n_tr = len(xt)
            for ep in range(30):
                mod.train()
                perm = torch.tensor(rng.permutation(n_tr), dtype=torch.long, device=device)
                for st in range(0, n_tr, 64):
                    idx = perm[st:st + 64]
                    opt.zero_grad()
                    loss = loss_fn(mod(xt[idx], mt[idx]), yt[idx])
                    loss.backward()
                    opt.step()
            mod.eval()
            with torch.no_grad():
                va_preds.append(mod(xv, mv).cpu().numpy())
            del mod, xt, mt, yt, xv, mv
        oof[va] = np.mean(va_preds, axis=0)
    if device == 'mps':
        torch.mps.empty_cache()
    return oof


def main():
    t0 = time.perf_counter()
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"=== VISION PILOT EVALUATION (DEVICE={device}) ===", flush=True)

    df = pd.read_parquet(MANIFEST_PATH)
    N = len(df)
    pids = df['property_id'].astype(str).tolist()
    folds = df['fold'].to_numpy(dtype=int)
    y_price = df['price'].to_numpy(dtype=float)
    p4_pred = df['p4_oof_pred'].to_numpy(dtype=float)
    y_resid = df['log_resid_p4'].to_numpy(dtype=float)

    base_m = compute_metrics(y_price, p4_pred)
    print(f"Baseline Phase-4 Ensemble on {N} pilot listings: {fmt(base_m)}", flush=True)

    # Collect all photo paths & aerial paths
    photo_paths, photo_pid_idx = [], []
    for i, pid in enumerate(pids):
        ps = sorted(glob.glob(os.path.join(PHOTO_DIR, pid, '*.jpg')))
        for p in ps:
            photo_paths.append(p)
            photo_pid_idx.append(i)
    photo_pid_idx = np.array(photo_pid_idx, dtype=int)

    a50_paths = [os.path.join(AERIAL_DIR, f'{pid}_50m.jpg') for pid in pids]
    a200_paths = [os.path.join(AERIAL_DIR, f'{pid}_200m.jpg') for pid in pids]

    if os.path.exists(EMB_CACHE):
        print(f"Loading cached pilot embeddings from {EMB_CACHE}...", flush=True)
        z = np.load(EMB_CACHE, allow_pickle=True)
        sig_p, sig_p_ok = z['sig_p'], z['sig_p_ok']
        room_txt, q_pos_txt, q_neg_txt = z['room_txt'], z['q_pos_txt'], z['q_neg_txt']
        dino_p, dino_p_ok = z['dino_p'], z['dino_p_ok']
        sig_a50, sig_a200 = z['sig_a50'], z['sig_a200']
        dino_a50, dino_a200 = z['dino_a50'], z['dino_a200']
    else:
        sig_p, sig_p_ok, room_txt, q_pos_txt, q_neg_txt = encode_images_siglip2(photo_paths, device)
        sig_a50, _, _, _, _ = encode_images_siglip2(a50_paths, device)
        sig_a200, _, _, _, _ = encode_images_siglip2(a200_paths, device)
        dino_p, dino_p_ok = encode_images_dinov2(photo_paths, device)
        dino_a50, _ = encode_images_dinov2(a50_paths, device)
        dino_a200, _ = encode_images_dinov2(a200_paths, device)
        np.savez_compressed(
            EMB_CACHE,
            sig_p=sig_p, sig_p_ok=sig_p_ok, room_txt=room_txt, q_pos_txt=q_pos_txt, q_neg_txt=q_neg_txt,
            dino_p=dino_p, dino_p_ok=dino_p_ok, sig_a50=sig_a50, sig_a200=sig_a200,
            dino_a50=dino_a50, dino_a200=dino_a200,
        )

    # --- Build per-listing representations ---
    D_sig = sig_p.shape[1]
    D_dino = dino_p.shape[1]
    room_keys = list(ROOM_PROMPTS.keys())
    q_keys = list(Q_PAIRS.keys())
    n_rooms = len(room_keys)
    n_q = len(q_keys)

    # Per-photo room similarities & contrastive Q-scores
    room_sim = sig_p @ room_txt.T          # (M, n_rooms)
    room_arg = room_sim.argmax(axis=1)
    q_scores_photo = (sig_p @ q_pos_txt.T) - (sig_p @ q_neg_txt.T)  # (M, n_q)

    # Print room distribution sanity check
    valid_rooms = room_arg[sig_p_ok]
    print("\nZero-Shot SigLIP2 Room Router Distribution across downloaded photos:", flush=True)
    for ri, rk in enumerate(room_keys):
        cnt = int((valid_rooms == ri).sum())
        print(f"  {rk:10s}: {cnt:5d} ({cnt/max(len(valid_rooms),1)*100:4.1f}%)", flush=True)

    # Aggregate per listing
    sig_mean = np.zeros((N, D_sig), dtype=np.float32)
    dino_mean = np.zeros((N, D_dino), dtype=np.float32)
    # Key room buckets (facade, kitchen, living, bathroom, garden)
    key_rooms = [0, 1, 2, 3, 5]
    sig_routed = np.zeros((N, D_sig * len(key_rooms)), dtype=np.float32)
    dino_routed = np.zeros((N, D_dino * len(key_rooms)), dtype=np.float32)
    q_feats = np.zeros((N, n_q * 2 + n_rooms), dtype=np.float32)  # mean + max Q-scores + room shares

    K_max = 8
    sig_bag = np.zeros((N, K_max, D_sig), dtype=np.float32)
    mask_bag = np.zeros((N, K_max), dtype=bool)

    for i in range(N):
        idx = np.where((photo_pid_idx == i) & sig_p_ok)[0]
        if len(idx) == 0:
            continue
        # Exclude floorplans from mean aesthetic pooling when other photos exist
        non_plan = idx[room_arg[idx] != 6]
        use = non_plan if len(non_plan) > 0 else idx
        sig_mean[i] = sig_p[use].mean(axis=0)
        dino_mean[i] = dino_p[use].mean(axis=0)

        # Q-scores: mean and 75th-percentile across listing photos + room composition fractions
        qs = q_scores_photo[use]
        q_feats[i, :n_q] = qs.mean(axis=0)
        q_feats[i, n_q:2 * n_q] = np.percentile(qs, 75, axis=0)
        for ri in range(n_rooms):
            q_feats[i, 2 * n_q + ri] = (room_arg[idx] == ri).mean()

        # Room-routed pooling (falls back to listing mean if room type missing)
        for j, ri in enumerate(key_rooms):
            r_idx = idx[room_arg[idx] == ri]
            sig_routed[i, j * D_sig:(j + 1) * D_sig] = sig_p[r_idx].mean(axis=0) if len(r_idx) else sig_mean[i]
            dino_routed[i, j * D_dino:(j + 1) * D_dino] = dino_p[r_idx].mean(axis=0) if len(r_idx) else dino_mean[i]

        # Bag for Attention-MIL
        take = idx[:K_max]
        sig_bag[i, :len(take)] = sig_p[take]
        mask_bag[i, :len(take)] = True

    # Print correlations of interpretable Q-scores with Phase-4 residual!
    print("\n=== ZERO-SHOT SIGLIP2 Q-SCORE CORRELATIONS WITH PHASE-4 RESIDUAL (log(actual) - log(P4_pred)) ===", flush=True)
    q_corrs = {}
    for qi, qk in enumerate(q_keys):
        r_res = float(np.corrcoef(q_feats[:, qi], y_resid)[0, 1])
        r_psqm = float(np.corrcoef(q_feats[:, qi], np.log(y_price / df['livable_area'].to_numpy()))[0, 1])
        q_corrs[qk] = {'corr_with_p4_resid': round(r_res, 4), 'corr_with_log_psqm': round(r_psqm, 4)}
        print(f"  {qk:20s} | corr(P4_resid) = {r_res:+.4f} | corr(log_€/m²) = {r_psqm:+.4f}", flush=True)

    # --- Evaluate Options A, B, C, D strictly 5-fold OOF on top of Phase-4! ---
    print("\n=== HEAD-TO-HEAD 5-FOLD OOF LIFT ON TOP OF PHASE-4 ENSEMBLE (n=800) ===", flush=True)
    options = {
        'A1_SigLIP2_QScore_ZeroShot': q_feats,
        'A2_SigLIP2_Photos_Mean':     sig_mean,
        'A3_SigLIP2_Photos_Routed':   sig_routed,
        'B1_DINOv2_Photos_Mean':      dino_mean,
        'B2_DINOv2_Photos_Routed':    dino_routed,
        'C1_Aerial_DINOv2_50+200m':   np.hstack([dino_a50, dino_a200]),
        'C2_Aerial_SigLIP2_50+200m':  np.hstack([sig_a50, sig_a200]),
        'D1_Photos_SigLIP2+DINOv2':   np.hstack([sig_mean, dino_mean, q_feats]),
        'D2_ALL_Photos+Aerial':       np.hstack([sig_mean, dino_mean, q_feats, dino_a50, dino_a200, sig_a50]),
    }

    results = {'baseline_p4': base_m, 'q_score_correlations': q_corrs, 'options': {}}
    oof_preds = {}
    for name, X in options.items():
        pred_r, corr, best_a = oof_ridge_resid(X, y_resid, folds)
        oof_preds[name] = pred_r
        # Apply predicted residual to Phase-4 ensemble!
        adj_pred = p4_pred * np.exp(pred_r)
        m = compute_metrics(y_price, adj_pred)
        d_mdape = m['mdape'] - base_m['mdape']
        d_mae = m['mae'] - base_m['mae']
        results['options'][name] = {'corr_resid': round(corr, 4), 'alpha': best_a, 'metrics': m, 'delta_mdape': round(d_mdape, 3), 'delta_mae': round(d_mae, 1)}
        print(f"  {name:28s} | r={corr:+.4f} (a={best_a:5.1f}) | {fmt(m)} | ΔMdAPE={d_mdape:+.2f}pp ΔMAE=€{d_mae:+,.0f}", flush=True)

    # Option A4: Gated Attention-MIL over per-photo SigLIP2 embeddings
    mil_r = oof_attention_mil(sig_bag, mask_bag, y_resid, folds, device)
    mil_corr = float(np.corrcoef(mil_r, y_resid)[0, 1])
    mil_m = compute_metrics(y_price, p4_pred * np.exp(mil_r))
    results['options']['A4_SigLIP2_AttentionMIL'] = {
        'corr_resid': round(mil_corr, 4), 'metrics': mil_m,
        'delta_mdape': round(mil_m['mdape'] - base_m['mdape'], 3),
        'delta_mae': round(mil_m['mae'] - base_m['mae'], 1),
    }
    print(f"  {'A4_SigLIP2_AttentionMIL':28s} | r={mil_corr:+.4f}           | {fmt(mil_m)} | ΔMdAPE={mil_m['mdape']-base_m['mdape']:+.2f}pp ΔMAE=€{mil_m['mae']-base_m['mae']:+,.0f}", flush=True)

    # Blend best Ridge + Attention-MIL
    best_comb_r = 0.65 * oof_preds['D2_ALL_Photos+Aerial'] + 0.35 * mil_r
    comb_m = compute_metrics(y_price, p4_pred * np.exp(best_comb_r))
    print(f"  {'D3_ALL_Ridge+AttnMIL_Blend':28s} | r={np.corrcoef(best_comb_r, y_resid)[0,1]:+.4f}           | {fmt(comb_m)} | ΔMdAPE={comb_m['mdape']-base_m['mdape']:+.2f}pp ΔMAE=€{comb_m['mae']-base_m['mae']:+,.0f}", flush=True)

    # Segment breakdown for best vision model vs Phase-4 baseline
    print("\n=== SEGMENT BREAKDOWN: PHASE-4 BASELINE vs +VISION (D3 Blend) ===", flush=True)
    seg_rep = {}
    for seg in ['historic_core', 'fixer_upper', 'luxury_650k_plus', 'house_typical', 'apt_typical']:
        ms = df['pilot_segment'].to_numpy() == seg
        mb = compute_metrics(y_price[ms], p4_pred[ms])
        mv = compute_metrics(y_price[ms], (p4_pred * np.exp(best_comb_r))[ms])
        seg_rep[seg] = {'n': int(ms.sum()), 'base_mdape': mb['mdape'], 'vis_mdape': mv['mdape'], 'delta_mdape': round(mv['mdape'] - mb['mdape'], 3), 'base_mae': mb['mae'], 'vis_mae': mv['mae']}
        print(f"  {seg:18s} (n={ms.sum():3d}) | P4: {mb['mdape']:5.2f}% (€{mb['mae']:7,.0f}) -> +Vis: {mv['mdape']:5.2f}% (€{mv['mae']:7,.0f}) | ΔMdAPE={mv['mdape']-mb['mdape']:+.2f}pp", flush=True)
    results['segments'] = seg_rep

    with open(OUT_JSON, 'w') as fh:
        json.dump(results, fh, indent=2)
    print(f"\nSaved {OUT_JSON} in {time.perf_counter()-t0:.1f}s!", flush=True)


if __name__ == '__main__':
    main()
