"""Vision Sandbox Lab — Fast Single-Fold (`n=2,500`, fold_0 val = 500 + train = 2,000) Mix-and-Match Bench!

Why this is the exact right scale for discovering the best vision modality before full-corpus scaling:
  - 2,500 stratified homes (covering ALL 5 folds, with fold_0 as clean validation = ~500 homes,
    folds 1-4 as train = ~2,000 homes, AND also evaluates 5-fold grouped OOF over all 2,500 homes!)
  - ~3.2x larger than the 800-home pilot (so joint LightGBM/CatBoost/Ridge can actually learn real
    interactions like `fixer_upper x vis_finish` or `short_desc x vis_score`!), yet downloads in ~2.5 min
    and encodes with SigLIP2-base in ~60s!
  - Reuses all 800 already-downloaded pilot photos automatically (0 wasted bandwidth!).

Step 1 (`--fetch`): downloads up to 6 `mediumUrl` photos/home (~14k photos total, ~180 MB, ~2.5 min)
Step 2 (`--encode`): encodes ALL photos on MPS with:
    (a) `SigLIP2-base-224` (fast: ~250 img/s -> ~55s!)
    (b) `DINOv2-base` (fast: ~200 img/s -> ~70s!)
    (c) `SigLIP2-SO400M-384` on the 3x2 contact sheet (1 image/home -> 2,500 imgs @ 17.7/s = ~2.3 min!)
    (d) Pure-NumPy photo photometry & composition stats (0s GPU)
Step 3 (`--vlm-splash`): optional `Qwen3.6-35B-A3B-5bit` over Splash with our **RENDER-AWARE** rubric
    prompt on the 2,500 contact sheets (~10-12 min if we want it, or --limit 2500!)
Step 4 (`--bench`): fast single-fold (`fold_0`) + 5-fold OOF **JOINT LightGBM & CatBoost** mix-and-match
    ablation comparing EVERY vision modality + render-filtering + combinations against exact Tab+Spatial+Text
    baseline on the SAME 2,500 homes!

Usage:
    .venv-api/bin/python -u src/vision_sandbox.py --fetch
    .venv-api/bin/python -u src/vision_sandbox.py --encode
    .venv-api/bin/python -u src/vision_sandbox.py --bench
"""
from __future__ import annotations

import argparse
import concurrent.futures
import glob
import json
import os
import re
import shutil
import sys
import time

import joblib
import numpy as np
import pandas as pd
import requests
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'src'))

from parser import extract_classified_json
from phase3_features import compute_metrics, fmt

SANDBOX_DIR = os.path.join(BASE_DIR, 'external_data', 'vision_sandbox')
PHOTO_DIR = os.path.join(SANDBOX_DIR, 'photos')
SHEET_DIR = os.path.join(SANDBOX_DIR, 'contact_sheets')
PILOT_PHOTO_DIR = os.path.join(BASE_DIR, 'external_data', 'vision_pilot', 'photos')
MANIFEST_PATH = os.path.join(SANDBOX_DIR, 'sandbox_manifest.parquet')
EMB_CACHE = os.path.join(SANDBOX_DIR, 'sandbox_embeddings.npz')
VLM_PATH = os.path.join(SANDBOX_DIR, 'vlm_scores_render_aware.parquet')
REPORT_PATH = os.path.join(SANDBOX_DIR, 'sandbox_bench_report.json')

DATASET_PATH = os.path.join(BASE_DIR, 'outputs', 'dataset', 'training_dataset.parquet')
DATA_HTML_DIR = os.path.join(BASE_DIR, 'data')
P4_PREDS = os.path.join(BASE_DIR, 'outputs', 'models', 'workbench', 'phase4_frontier_preds.npz')
P4_CACHE = os.path.join(BASE_DIR, 'outputs', 'models', 'workbench', 'phase4_splits_cache.joblib')
P5_ADDON = os.path.join(BASE_DIR, 'outputs', 'models', 'workbench', 'phase5_addon_features.joblib')

MAX_PHOTOS = 6
TARGET_N = 2500

# Sharp Render-vs-Real + Room + Quality prompts
MEDIA_TYPE_PROMPTS = [
    'an authentic real-world photograph taken with a camera inside or outside an existing house',
    'a 3D architectural CGI computer rendering or digital visualization illustration of a building or room',
    'a digitally staged or virtual renovation mockup rendering of a room',
    'a 2D architectural floorplan, blueprint, map, or site plan diagram',
    'a construction site, bare concrete shell, or stripped renovation interior',
]
ROOM_PROMPTS = {
    'facade':   'exterior front facade of a house or apartment building from the street',
    'kitchen':  'interior photograph of a home kitchen with cabinets and appliances',
    'living':   'interior photograph of a living room or dining area',
    'bathroom': 'interior photograph of a bathroom with shower, bathtub, or sink',
    'bedroom':  'interior photograph of a bedroom',
    'garden':   'outdoor garden, backyard, terrace, balcony, or courtyard of a home',
    'plan':     'architectural floor plan, blueprint, map, or technical drawing',
}
Q_PAIRS = [
    ('modern_finish',   'a luxury modern renovated home interior with high-end finishes and clean design',
                        'an old outdated worn home interior in need of renovation'),
    ('fixer_shell',     'a dilapidated fixer-upper house needing complete renovation with peeling walls or bare construction',
                        'a move-in ready turnkey renovated home in pristine condition'),
    ('natural_light',   'a bright airy sunlit room with large windows and abundant natural daylight',
                        'a dark cramped dim room with poor lighting'),
    ('kitchen_luxury',  'a bespoke designer luxury kitchen with marble island and modern built-in appliances',
                        'a dated old worn kitchen with old cabinets'),
    ('bath_luxury',     'a spa-like modern luxury bathroom with walk-in rain shower and sleek tiling',
                        'an outdated old bathroom with vintage tiles and worn fixtures'),
    ('architect_charm', 'an authentic historic Belgian townhouse interior with high ceilings, herringbone parquet, and architectural moldings',
                        'a plain generic bland boxy room with cheap finishes'),
    ('facade_prestige', 'a prestigious historic townhouse mansion or architectural villa exterior facade',
                        'a plain tired worn or dreary residential building exterior'),
    ('floor_quality',   'natural solid oak hardwood parquet or polished natural stone flooring',
                        'old worn linoleum, dated carpet, or damaged laminate flooring'),
    ('outdoor_green',   'a lush sunny private landscaped garden or spacious wooden terrace',
                        'a dreary concrete courtyard or no outdoor space'),
    ('staging_pro',     'professional real estate photography of a beautifully styled staged home',
                        'amateur cluttered messy snapshot taken with a phone'),
]

# Render-aware VLM rubric prompt for Splash (Qwen3.6-35B-A3B-5bit)
VLM_RENDER_AWARE_PROMPT = (
    "You are an expert Belgian real-estate appraiser evaluating a 6-photo contact sheet of a home in Ghent.\n"
    "CRITICAL RULE: Distinguish REAL camera photos from 3D CGI architectural renders / virtual-renovation mockups!\n"
    "Judge finish/kitchen/bath/reno_need from the REAL photos when present (do NOT mistake a virtual renovation render for actual renovated state). "
    "Set pct_renders (0-100) to the % of tiles that are CGI/3D renders or virtual mockups.\n"
    "Output ONLY one compact JSON object:\n"
    '{"finish":1-10,"kitchen":1-10,"bath":1-10,"charm":1-10,"light":1-10,"exterior":1-10,"reno_need":1-10,"reno_keur":0-250,"luxury":1-10,"staging":1-10,"pct_renders":0-100}\n'
    "Use null only if unseen. JSON only, no markdown, no explanation."
)


def _extract_medium_urls(pid: str, max_pics: int = MAX_PHOTOS) -> tuple[list[str], int]:
    d = os.path.join(DATA_HTML_DIR, str(pid))
    if not os.path.isdir(d):
        return [], 0
    htmls = sorted(glob.glob(os.path.join(d, '*.html')))
    if not htmls:
        return [], 0
    try:
        raw = open(htmls[0], 'r', encoding='utf-8', errors='ignore').read()
        cl = extract_classified_json(raw)
        if not cl:
            return [], 0
        pics = (cl.get('media') or {}).get('pictures') or []
        urls = [p.get('mediumUrl') or p.get('largeUrl') for p in pics if isinstance(p, dict)]
        urls = [u for u in urls if u]
        total = len(urls)
        if total <= max_pics:
            return urls, total
        # Evenly spaced across gallery to capture facade + interior + garden (not just first 6 facade shots)
        idx = np.linspace(0, total - 1, max_pics, dtype=int)
        return [urls[i] for i in idx], total
    except Exception:
        return [], 0


def build_sandbox_sample_and_fetch():
    os.makedirs(PHOTO_DIR, exist_ok=True)
    tr = pd.read_parquet(DATASET_PATH)
    tr['property_id'] = tr['property_id'].astype(str)
    p4 = np.load(P4_PREDS, allow_pickle=True)
    tr['p4_oof_pred'] = p4['ens_oof_unified']
    tr['log_resid_p4'] = np.log(tr['price'] / tr['p4_oof_pred'])

    # Keep ALL 800 pilot homes first (already downloaded!), then add 1,700 balanced random homes across all 5 folds!
    pilot_man_p = os.path.join(BASE_DIR, 'external_data', 'vision_pilot', 'pilot_manifest.parquet')
    pilot_pids = set()
    if os.path.exists(pilot_man_p):
        pm = pd.read_parquet(pilot_man_p)
        pilot_pids = set(pm['property_id'].astype(str))

    in_pilot = tr[tr['property_id'].isin(pilot_pids)].copy()
    rest = tr[~tr['property_id'].isin(pilot_pids)].copy()

    # Stratified add from rest across folds so fold_0..4 are balanced (~500 per fold total!)
    rng = np.random.default_rng(2026)
    adds = []
    for f in sorted(tr['fold'].unique()):
        n_have = int((in_pilot['fold'] == f).sum())
        n_need = max(0, (TARGET_N // 5) - n_have)
        pool = rest[rest['fold'] == f]
        take = pool.sample(n=min(n_need, len(pool)), random_state=int(rng.integers(0, 100000)))
        adds.append(take)
    sel = pd.concat([in_pilot] + adds, ignore_index=True)

    # Tag segment cleanly for slice diagnostics
    hist_w = {'Binnenstad', 'Elisabethbegijnhof_Prinsenhof_Papegaai_Sint_Michiels', 'Macharius_Heirnis', 'Sluizeken_Tolhuis_Ham'}
    fixer_c = {'TO_RENOVATE', 'TO_RESTORE', 'TO_BE_DONE_UP', 'TO_REFURBISH'}
    def _seg(r):
        if r['price'] >= 650_000: return 'luxury_650k_plus'
        if r['building_condition'] in fixer_c: return 'fixer_upper'
        if r['neighbourhood'] in hist_w: return 'historic_core'
        return 'house_typical' if r['property_type_HOUSE'] == 1 else 'apt_typical'
    sel['segment'] = sel.apply(_seg, axis=1)

    print(f"Selected {len(sel)} sandbox homes across folds: {sel['fold'].value_counts().sort_index().to_dict()}", flush=True)
    print(f"Segments: {sel['segment'].value_counts().to_dict()}", flush=True)

    # Extract photo URLs & reuse pilot photos where present!
    t0 = time.perf_counter()
    tasks = []
    total_pics_col = []
    reused = 0
    for pid in sel['property_id']:
        out_d = os.path.join(PHOTO_DIR, pid)
        os.makedirs(out_d, exist_ok=True)
        # Copy/hardlink up to MAX_PHOTOS from existing pilot dir if available
        pilot_d = os.path.join(PILOT_PHOTO_DIR, pid)
        existing = sorted(glob.glob(os.path.join(out_d, '*.jpg')))
        if not existing and os.path.isdir(pilot_d):
            pp = sorted(glob.glob(os.path.join(pilot_d, '*.jpg')))
            if pp:
                idx = np.linspace(0, len(pp) - 1, min(len(pp), MAX_PHOTOS), dtype=int)
                for k, j in enumerate(idx):
                    dst = os.path.join(out_d, f'{k:02d}.jpg')
                    if not os.path.exists(dst):
                        try:
                            os.link(pp[j], dst)
                        except OSError:
                            shutil.copy2(pp[j], dst)
                reused += 1
                existing = sorted(glob.glob(os.path.join(out_d, '*.jpg')))

        urls, tot = _extract_medium_urls(pid, MAX_PHOTOS)
        total_pics_col.append(tot)
        if len(existing) >= min(len(urls), MAX_PHOTOS) and len(existing) > 0:
            continue
        for k, u in enumerate(urls):
            dst = os.path.join(out_d, f'{k:02d}.jpg')
            if not os.path.exists(dst):
                tasks.append((u, dst))

    print(f"URL scan done in {time.perf_counter()-t0:.1f}s | reused {reused} pilot homes | {len(tasks)} new photos to download", flush=True)

    sess = requests.Session()
    sess.headers.update({'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'})
    ok_dl, fail_dl = 0, 0
    t_dl = time.perf_counter()

    def _dl(item):
        u, dst = item
        try:
            r = sess.get(u, timeout=10)
            if r.status_code == 200 and len(r.content) > 800:
                with open(dst, 'wb') as f:
                    f.write(r.content)
                return True
        except Exception:
            pass
        return False

    if tasks:
        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as ex:
            for idx_done, res in enumerate(ex.map(_dl, tasks), 1):
                if res: ok_dl += 1
                else: fail_dl += 1
                if idx_done % 1500 == 0 or idx_done == len(tasks):
                    dt = time.perf_counter() - t_dl
                    print(f"  Downloaded {idx_done:5d}/{len(tasks)} | ok={ok_dl} fail={fail_dl} | {idx_done/max(dt,1e-3):.0f} img/s | {dt:.1f}s", flush=True)

    n_dl = [len(glob.glob(os.path.join(PHOTO_DIR, pid, '*.jpg'))) for pid in sel['property_id']]
    sel['total_html_pics'] = total_pics_col
    sel['n_photos_dl'] = n_dl
    keep_cols = ['property_id', 'date_obtained', 'fold', 'property_key', 'price', 'livable_area',
                 'property_type_HOUSE', 'neighbourhood', 'building_condition', 'energy_label',
                 'is_newly_built', 'description_length', 'publication_year', 'p4_oof_pred',
                 'log_resid_p4', 'segment', 'total_html_pics', 'n_photos_dl']
    sel[keep_cols].to_parquet(MANIFEST_PATH, index=False)
    print(f"Saved {MANIFEST_PATH} ({len(sel)} homes, {(sel['n_photos_dl']>=1).sum()} with >=1 photo, median {int(np.median(n_dl))} photos/home)!", flush=True)


def encode_all_vision_modalities():
    """Encode all 2,500 sandbox homes with SigLIP2-base, DINOv2-base, SigLIP2-SO400M-384 (on contact sheet!), + Photometry."""
    import torch
    from transformers import AutoImageProcessor, AutoModel, AutoProcessor

    man = pd.read_parquet(MANIFEST_PATH)
    pids = man['property_id'].astype(str).tolist()
    n_homes = len(pids)

    # Gather all photo paths + listing offsets
    all_paths = []
    offsets = np.zeros(n_homes + 1, dtype=np.int32)
    for i, pid in enumerate(pids):
        ps = sorted(glob.glob(os.path.join(PHOTO_DIR, pid, '*.jpg')))
        all_paths.extend(ps)
        offsets[i + 1] = len(all_paths)
    n_photos = len(all_paths)
    print(f"=== ENCODING SANDBOX ({n_homes} homes, {n_photos} photos) ===", flush=True)

    # 0. Fast photometry stats per photo (brightness, contrast, saturation, aspect ratio)
    t0 = time.perf_counter()
    photo_stats = np.zeros((n_photos, 5), dtype=np.float32)  # bright_mean, bright_std, sat_mean, is_vert, edge_sharpness
    def _load_and_stat(p):
        try:
            with Image.open(p) as im:
                im = im.convert('RGB')
                w, h = im.size
                is_v = 1.0 if h > w else 0.0
                sm = np.asarray(im.resize((64, 64), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
                gray = 0.299 * sm[..., 0] + 0.587 * sm[..., 1] + 0.114 * sm[..., 2]
                mx = sm.max(axis=-1); mn = sm.min(axis=-1)
                sat = np.where(mx > 1e-4, (mx - mn) / np.maximum(mx, 1e-4), 0.0)
                # Simple Laplacian/gradient sharpness proxy
                gx = np.abs(np.diff(gray, axis=1)).mean()
                gy = np.abs(np.diff(gray, axis=0)).mean()
                return np.array([gray.mean(), gray.std(), sat.mean(), is_v, gx + gy], dtype=np.float32)
        except Exception:
            return np.full(5, np.nan, dtype=np.float32)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        for idx, st in enumerate(ex.map(_load_and_stat, all_paths)):
            photo_stats[idx] = st
    print(f"  Photometry computed for {n_photos} photos in {time.perf_counter()-t0:.1f}s", flush=True)

    def _safe_img(p):
        try:
            with Image.open(p) as im:
                return im.convert('RGB').copy()
        except Exception:
            return Image.new('RGB', (224, 224), (128, 128, 128))

    # 1. SigLIP2-base-224 (all photos + text prompts for Room, Media-Type/Render, and Quality!)
    t1 = time.perf_counter()
    sig_id = 'google/siglip2-base-patch16-224'
    sig_proc = AutoProcessor.from_pretrained(sig_id, use_fast=True)
    sig_mod = AutoModel.from_pretrained(sig_id, dtype=torch.bfloat16).to('mps').eval()
    sig_embs = np.zeros((n_photos, 768), dtype=np.float16)
    bs = 64
    with torch.no_grad():
        for i in range(0, n_photos, bs):
            imgs = [_safe_img(p) for p in all_paths[i:i + bs]]
            inp = sig_proc(images=imgs, return_tensors='pt')
            inp = {k: v.to('mps', dtype=torch.bfloat16) if v.is_floating_point() else v.to('mps') for k, v in inp.items()}
            f = sig_mod.get_image_features(**inp)
            if not isinstance(f, torch.Tensor): f = f.pooler_output
            f = f.float()
            f = f / f.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            sig_embs[i:i + len(imgs)] = f.cpu().numpy().astype(np.float16)
            if (i // bs) % 50 == 0 and i > 0:
                dt = time.perf_counter() - t1
                print(f"  [SigLIP2-base] {i:5d}/{n_photos} | {i/dt:.0f} img/s", flush=True)

        def _enc_txt(txts):
            inp = sig_proc(text=txts, padding='max_length', max_length=64, truncation=True, return_tensors='pt')
            inp = {k: v.to('mps') for k, v in inp.items()}
            tf = sig_mod.get_text_features(**inp)
            if not isinstance(tf, torch.Tensor): tf = tf.pooler_output
            tf = tf.float()
            tf = tf / tf.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            return tf.cpu().numpy().astype(np.float32)

        media_txt = _enc_txt(MEDIA_TYPE_PROMPTS)
        room_txt = _enc_txt(list(ROOM_PROMPTS.values()))
        q_pos_txt = _enc_txt([p for _, p, _ in Q_PAIRS])
        q_neg_txt = _enc_txt([n for _, _, n in Q_PAIRS])
    del sig_mod
    torch.mps.empty_cache()
    print(f"  [SigLIP2-base] Done {n_photos} photos in {time.perf_counter()-t1:.1f}s ({n_photos/(time.perf_counter()-t1):.0f} img/s)", flush=True)

    # 2. Build smart REAL-PHOTO-PRIORITIZED 3x2 contact sheets (filtering out CGI renders & floorplans when real photos exist!)
    os.makedirs(SHEET_DIR, exist_ok=True)
    t_sh = time.perf_counter()
    sig_f32 = sig_embs.astype(np.float32)
    media_sims = sig_f32 @ media_txt.T  # (n_photos, 5): 0=real, 1=cgi, 2=vstage, 3=plan, 4=shell
    room_ids = (sig_f32 @ room_txt.T).argmax(axis=1)
    is_cgi = (media_sims[:, 1] - media_sims[:, 0]) > 0.0
    is_plan = (room_ids == 6) | (media_sims.argmax(axis=1) == 3)

    sheet_paths = []
    prio_rooms = [0, 2, 1, 3, 5, 4]
    for i, pid in enumerate(pids):
        out_p = os.path.join(SHEET_DIR, f'{pid}.jpg')
        s, e = offsets[i], offsets[i + 1]
        if e > s:
            if not (os.path.exists(out_p) and os.path.getsize(out_p) > 2000):
                idxs = list(range(s, e))
                # Prefer real non-plan photos first, then fallback to non-plan, then all
                real_idxs = [j for j in idxs if (not is_plan[j]) and (not is_cgi[j])]
                pool = real_idxs if len(real_idxs) >= 3 else ([j for j in idxs if not is_plan[j]] or idxs)
                chosen, used = [], set()
                for rt in prio_rooms:
                    for j in pool:
                        if j not in used and room_ids[j] == rt:
                            chosen.append(all_paths[j]); used.add(j); break
                    if len(chosen) == 6: break
                for j in pool:
                    if len(chosen) == 6: break
                    if j not in used: chosen.append(all_paths[j]); used.add(j)
                canvas = Image.new('RGB', (600, 400), (20, 20, 20))
                for slot, p in enumerate(chosen[:6]):
                    try:
                        with Image.open(p) as im:
                            im = im.convert('RGB').resize((200, 200), Image.Resampling.BILINEAR)
                            canvas.paste(im, ((slot % 3) * 200, (slot // 3) * 200))
                    except Exception:
                        pass
                canvas.save(out_p, format='JPEG', quality=82)
            sheet_paths.append(out_p)
        else:
            sheet_paths.append('')
    print(f"  Built smart real-photo contact sheets for {sum(bool(x) for x in sheet_paths)}/{n_homes} homes in {time.perf_counter()-t_sh:.1f}s", flush=True)

    # 3. SigLIP2-SO400M-384 on the 3x2 Contact Sheet (1 high-res holistic pass per home -> 2,500 imgs @ ~17.7/s = ~2.3 min!)
    t2 = time.perf_counter()
    so_id = 'google/siglip2-so400m-patch14-384'
    so_proc = AutoProcessor.from_pretrained(so_id, use_fast=True)
    so_mod = AutoModel.from_pretrained(so_id, dtype=torch.bfloat16).to('mps').eval()
    so_sheet_embs = np.full((n_homes, 1152), np.nan, dtype=np.float32)
    valid_sh = [i for i, p in enumerate(sheet_paths) if p]
    bs_so = 12
    with torch.no_grad():
        for k in range(0, len(valid_sh), bs_so):
            batch_i = valid_sh[k:k + bs_so]
            imgs = [_safe_img(sheet_paths[i]) for i in batch_i]
            inp = so_proc(images=imgs, return_tensors='pt')
            inp = {kk: vv.to('mps', dtype=torch.bfloat16) if vv.is_floating_point() else vv.to('mps') for kk, vv in inp.items()}
            f = so_mod.get_image_features(**inp)
            if not isinstance(f, torch.Tensor): f = f.pooler_output
            f = f.float()
            f = f / f.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            so_sheet_embs[batch_i] = f.cpu().numpy()
            if (k // bs_so) % 40 == 0 and k > 0:
                dt = time.perf_counter() - t2
                print(f"  [SO400M-384 Sheet] {k:4d}/{len(valid_sh)} | {k/dt:.1f} homes/s", flush=True)

        # Also get SO400M text embeddings for the same Q_PAIRS (for sheet-level SO400M Q-scores!)
        def _enc_so_txt(txts):
            inp = so_proc(text=txts, padding='max_length', max_length=64, truncation=True, return_tensors='pt')
            inp = {kk: vv.to('mps') for kk, vv in inp.items()}
            tf = so_mod.get_text_features(**inp)
            if not isinstance(tf, torch.Tensor): tf = tf.pooler_output
            tf = tf.float()
            tf = tf / tf.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            return tf.cpu().numpy().astype(np.float32)

        so_q_pos = _enc_so_txt([p for _, p, _ in Q_PAIRS])
        so_q_neg = _enc_so_txt([n for _, _, n in Q_PAIRS])
    del so_mod
    torch.mps.empty_cache()
    print(f"  [SO400M-384 Sheet] Done {len(valid_sh)} homes in {time.perf_counter()-t2:.1f}s ({len(valid_sh)/(time.perf_counter()-t2):.1f} homes/s)", flush=True)

    # 4. DINOv2-base on all photos (~200 img/s -> ~70s) so we can test pure visual texture/architecture features too
    t3 = time.perf_counter()
    dino_id = 'facebook/dinov2-base'
    dino_proc = AutoImageProcessor.from_pretrained(dino_id, use_fast=True)
    dino_mod = AutoModel.from_pretrained(dino_id, dtype=torch.bfloat16).to('mps').eval()
    dino_embs = np.zeros((n_photos, 768), dtype=np.float16)
    with torch.no_grad():
        for i in range(0, n_photos, bs):
            imgs = [_safe_img(p) for p in all_paths[i:i + bs]]
            inp = dino_proc(images=imgs, return_tensors='pt')
            inp = {k: v.to('mps', dtype=torch.bfloat16) if v.is_floating_point() else v.to('mps') for k, v in inp.items()}
            out = dino_mod(**inp).last_hidden_state[:, 0, :].float()
            out = out / out.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            dino_embs[i:i + len(imgs)] = out.cpu().numpy().astype(np.float16)
            if (i // bs) % 50 == 0 and i > 0:
                dt = time.perf_counter() - t3
                print(f"  [DINOv2-base] {i:5d}/{n_photos} | {i/dt:.0f} img/s", flush=True)
    del dino_mod
    torch.mps.empty_cache()
    print(f"  [DINOv2-base] Done {n_photos} photos in {time.perf_counter()-t3:.1f}s", flush=True)

    np.savez_compressed(
        EMB_CACHE,
        offsets=offsets,
        sig_embs=sig_embs,
        dino_embs=dino_embs,
        so_sheet_embs=so_sheet_embs.astype(np.float16),
        photo_stats=photo_stats,
        media_txt=media_txt,
        room_txt=room_txt,
        q_pos_txt=q_pos_txt,
        q_neg_txt=q_neg_txt,
        so_q_pos=so_q_pos,
        so_q_neg=so_q_neg,
    )
    print(f"Saved {EMB_CACHE} ({os.path.getsize(EMB_CACHE)/1e6:.1f} MB)!", flush=True)


def run_mix_and_match_bench():
    """Fast single-fold (fold_0 val ~500, train ~2,000) + 5-fold OOF JOINT LightGBM & CatBoost mix-and-match benchmark!"""
    import lightgbm as lgb
    from catboost import CatBoostRegressor
    from sklearn.decomposition import PCA
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler

    man = pd.read_parquet(MANIFEST_PATH)
    man['property_id'] = man['property_id'].astype(str)
    z = np.load(EMB_CACHE, allow_pickle=True)
    offsets = z['offsets']
    sig_embs = z['sig_embs'].astype(np.float32)
    dino_embs = z['dino_embs'].astype(np.float32)
    so_sheet = z['so_sheet_embs'].astype(np.float32)
    pstats = z['photo_stats'].astype(np.float32)
    media_txt = z['media_txt'].astype(np.float32)
    room_txt = z['room_txt'].astype(np.float32)
    q_pos_txt = z['q_pos_txt'].astype(np.float32)
    q_neg_txt = z['q_neg_txt'].astype(np.float32)
    so_q_pos = z['so_q_pos'].astype(np.float32)
    so_q_neg = z['so_q_neg'].astype(np.float32)

    n = len(man)
    media_sims = sig_embs @ media_txt.T  # 0=real, 1=cgi, 2=vstage, 3=plan, 4=shell
    cgi_margin = media_sims[:, 1] - media_sims[:, 0]
    vstage_margin = media_sims[:, 2] - media_sims[:, 0]
    room_ids = (sig_embs @ room_txt.T).argmax(axis=1)
    q_all = (sig_embs @ q_pos_txt.T) - (sig_embs @ q_neg_txt.T)  # (n_photos, 10)

    # Build per-listing raw vs RENDER-FILTERED pools + room-specific Q-scores + render flags + photometry!
    sig_raw = np.full((n, 768), np.nan, dtype=np.float32)
    sig_real = np.full((n, 768), np.nan, dtype=np.float32)
    sig_std = np.full((n, 768), np.nan, dtype=np.float32)
    dino_real = np.full((n, 768), np.nan, dtype=np.float32)

    # Compact interpretable vision blocks
    render_feats = np.full((n, 8), np.nan, dtype=np.float32)
    q_feats = np.full((n, 30), np.nan, dtype=np.float32)      # 10 mean + 10 p75 + 10 room-routed/worst
    so_q_feats = (so_sheet @ so_q_pos.T) - (so_sheet @ so_q_neg.T)  # (n, 10)
    photom_feats = np.full((n, 10), np.nan, dtype=np.float32)  # mean & std of 5 photo stats

    for i in range(n):
        s, e = int(offsets[i]), int(offsets[i + 1])
        if e <= s:
            continue
        sp = sig_embs[s:e]
        dp = dino_embs[s:e]
        rm = room_ids[s:e]
        cm = cgi_margin[s:e]
        vm = vstage_margin[s:e]
        ms = media_sims[s:e]
        cls = ms.argmax(axis=1)
        qa = q_all[s:e]
        ps = pstats[s:e]

        is_plan = (rm == 6) | (cls == 3)
        is_cgi = (cm > 0.0) | (cls == 1)
        is_vstage = (cls == 2)
        non_plan = ~is_plan
        real_only = (~is_plan) & (~is_cgi)

        use_raw = non_plan if non_plan.any() else np.ones(e - s, dtype=bool)
        use_real = real_only if real_only.any() else use_raw

        sig_raw[i] = sp[use_raw].mean(axis=0)
        sig_real[i] = sp[use_real].mean(axis=0)
        sig_std[i] = sp[use_real].std(axis=0) if use_real.sum() >= 2 else 0.0
        dino_real[i] = dp[use_real].mean(axis=0)

        # Render / Media-type flags (explicitly solving the virtual-renovation trap!)
        render_feats[i] = [
            float(cm.mean()), float(cm.max()), float(is_cgi.mean()),
            float(is_vstage.mean()), float((is_cgi | is_vstage).mean()),
            float((~is_cgi & ~is_vstage & ~is_plan).mean()),
            float(is_plan.mean()), float((cls == 4).mean()),  # construction shell frac
        ]

        # Real-photo Q-scores (10 mean + 10 p75 + 10 room-specific/worst!)
        qm = qa[use_real].mean(axis=0)
        qp = np.percentile(qa[use_real], 75, axis=0)
        # Room-specific Q: kitchen_lux on kitchen, bath_lux on bath, facade_prest on facade, outdoor on garden,
        # modern/charm/light on living/bed, worst fixer_shell (max across all real photos!), min modern_finish
        def _rq(room_idx_set, q_col, agg='mean'):
            mk = use_real & np.isin(rm, room_idx_set)
            vals = qa[mk, q_col] if mk.any() else qa[use_real, q_col]
            return float(np.max(vals) if agg == 'max' else (np.min(vals) if agg == 'min' else np.mean(vals)))

        room_q = np.array([
            _rq([1], 3),       # kitchen luxury on kitchen
            _rq([3], 4),       # bath luxury on bath
            _rq([0], 6),       # facade prestige on facade
            _rq([5], 8),       # outdoor green on garden
            _rq([2, 4], 5),    # architect charm on living/bed
            _rq([2, 4], 2),    # natural light on living/bed
            _rq([0,1,2,3,4], 1, 'max'),  # WORST fixer_shell photo!
            _rq([0,1,2,3,4], 0, 'min'),  # WORST modern_finish photo!
            float((rm == 1).any()),      # has kitchen photo
            float((rm == 5).any()),      # has garden/terrace photo
        ], dtype=np.float32)
        q_feats[i] = np.concatenate([qm, qp, room_q])

        ok_ps = ps[use_real]
        if len(ok_ps) > 0:
            photom_feats[i] = np.concatenate([np.nanmean(ok_ps, axis=0), np.nanstd(ok_ps, axis=0)])

    # Load exact Phase-4/5 tabular+spatial+text OOF feature matrix for these 2,500 homes!
    tr_full = pd.read_parquet(DATASET_PATH)
    tr_full['property_id'] = tr_full['property_id'].astype(str)
    tr_map = {pid: idx for idx, pid in enumerate(tr_full['property_id'])}
    ds_rows = np.array([tr_map[pid] for pid in man['property_id']])

    p4_cache = joblib.load(P4_CACHE)
    p5_addon = joblib.load(P5_ADDON)
    oof_frames = []
    for b in p4_cache['bundles']:
        if not str(b['id']).startswith('fold_'):
            continue
        vd = pd.concat([b['va_df'].reset_index(drop=True), p5_addon[b['id']]['va_add'].reset_index(drop=True)], axis=1)
        vd['__ds_idx__'] = b['va_idx']
        oof_frames.append(vd)
    all_oof = pd.concat(oof_frames, axis=0).sort_values('__ds_idx__').reset_index(drop=True)
    base_df = all_oof.iloc[ds_rows].reset_index(drop=True)

    # Compact strong base feature set (top 120 Phase-4 + P5 priority cols — fast & clean for 2,500 rows!)
    p4_rep = json.load(open(os.path.join(BASE_DIR, 'outputs', 'models', 'workbench', 'phase4_report.json')))
    p5_prio = ['txt_tab_plus_text_psqm', 'txt_p5_super_tabresid', 'txt_mlp_psqm', 'txt_mlp_price',
               'txt_mlp_tabresid', 'txt_tfidf_tabresid_a6', 'txt_dense_both_tabresid', 'tab_anchor_log_psqm',
               'txt_qwen3_a5_psqm', 'txt_p5_super_psqm', 'freq_agency_norm']
    base_cols = [c for c in p5_prio if c in base_df.columns]
    for c in p4_rep['pruned_features']:
        if c in base_df.columns and c not in base_cols:
            base_cols.append(c)
        if len(base_cols) >= 130:
            break

    # Prep clean numeric/categorical frame for fast LightGBM
    X_base = base_df[base_cols].copy()
    cat_cols = []
    for c in X_base.columns:
        if X_base[c].dtype == 'bool':
            X_base[c] = X_base[c].astype('int8')
        elif not pd.api.types.is_numeric_dtype(X_base[c]):
            X_base[c] = X_base[c].astype(str).fillna('Missing').astype('category')
            cat_cols.append(c)
        else:
            X_base[c] = X_base[c].astype('float32')

    y_price = man['price'].to_numpy(float)
    y_area = man['livable_area'].to_numpy(float)
    cpi_idx = p4_cache['cpi_idx'][ds_rows]
    y_target = np.log((y_price / cpi_idx) / y_area)  # log_real_psqm!
    p4_pred = man['p4_oof_pred'].to_numpy(float)
    y_p4_resid = man['log_resid_p4'].to_numpy(float)
    folds = man['fold'].to_numpy(int)

    # Build STRICTLY LEAK-FREE nested-OOF vision meta-features per fold!
    # For each outer fold f: train a RidgeCV ONLY on tr_folds predicting (a) y_target and (b) tabular-anchor residual!
    # Inside tr_folds, use inner 4-fold OOF so the downstream LGBM never overfits the vision Ridge predictions!
    tab_anchor = base_df['tab_anchor_log_psqm'].to_numpy(float)
    y_tab_resid = y_target - tab_anchor

    def _fit_transform_emb_block(X_emb: np.ndarray, tr_mask: np.ndarray, va_mask: np.ndarray,
                                 f_arr: np.ndarray, n_pca: int = 8, tag: str = 'v') -> tuple[pd.DataFrame, pd.DataFrame]:
        """Strictly leak-free inner-OOF Ridge (on log_real_psqm & tab_resid) + PCA fitted ONLY on tr_mask!"""
        med = np.nanmedian(X_emb[tr_mask], axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        Xtr = np.where(np.isfinite(X_emb[tr_mask]), X_emb[tr_mask], med)
        Xva = np.where(np.isfinite(X_emb[va_mask]), X_emb[va_mask], med)

        sc = StandardScaler()
        Xtr_s = sc.fit_transform(Xtr)
        Xva_s = sc.transform(Xva)

        # Inner OOF by existing grouped folds!
        tr_folds = f_arr[tr_mask]
        y1_tr, y2_tr = y_target[tr_mask], y_tab_resid[tr_mask]
        oof_1 = np.zeros(len(Xtr_s), dtype=np.float32)
        oof_2 = np.zeros(len(Xtr_s), dtype=np.float32)
        alphas = np.logspace(1, 5, 20)
        for uf in np.unique(tr_folds):
            itr, iva = tr_folds != uf, tr_folds == uf
            m1 = RidgeCV(alphas=alphas).fit(Xtr_s[itr], y1_tr[itr])
            oof_1[iva] = m1.predict(Xtr_s[iva])
            m2 = RidgeCV(alphas=alphas).fit(Xtr_s[itr], y2_tr[itr])
            oof_2[iva] = m2.predict(Xtr_s[iva])

        full_m1 = RidgeCV(alphas=alphas).fit(Xtr_s, y1_tr)
        full_m2 = RidgeCV(alphas=alphas).fit(Xtr_s, y2_tr)
        va_1 = full_m1.predict(Xva_s).astype(np.float32)
        va_2 = full_m2.predict(Xva_s).astype(np.float32)

        # Optional compact PCA (fit strictly on train!)
        tr_d = {f'{tag}_psqm': oof_1, f'{tag}_tabresid': oof_2}
        va_d = {f'{tag}_psqm': va_1,  f'{tag}_tabresid': va_2}
        if n_pca > 0:
            pca = PCA(n_components=n_pca, random_state=42)
            Ztr = pca.fit_transform(Xtr_s).astype(np.float32)
            Zva = pca.transform(Xva_s).astype(np.float32)
            for k in range(n_pca):
                tr_d[f'{tag}_pca_{k}'] = Ztr[:, k]
                va_d[f'{tag}_pca_{k}'] = Zva[:, k]
        return pd.DataFrame(tr_d), pd.DataFrame(va_d)

    # Check if render-aware VLM parquet exists (optional bonus!)
    has_vlm = os.path.exists(VLM_PATH)
    vlm_mat = None
    if has_vlm:
        vdf = pd.read_parquet(VLM_PATH)
        vdf['property_id'] = vdf['property_id'].astype(str)
        vcols = [c for c in vdf.columns if c.startswith('vlm_')]
        vm = man[['property_id']].merge(vdf[['property_id'] + vcols], on='property_id', how='left')
        vlm_mat = vm[vcols].to_numpy(dtype=np.float32)

    # Helper: run fast 5-fold OOF LightGBM (Huber loss, ~2.5s per recipe!) + print BOTH fold_0 and 5-fold OOF!
    def eval_recipe(name: str, builder_fn):
        oof_pred = np.zeros(n, dtype=float)
        for f in range(5):
            tr_m, va_m = (folds != f), (folds == f)
            tr_vis, va_vis = builder_fn(tr_m, va_m)
            Xtr = pd.concat([X_base.iloc[tr_m].reset_index(drop=True), tr_vis.reset_index(drop=True)], axis=1) if tr_vis is not None else X_base.iloc[tr_m]
            Xva = pd.concat([X_base.iloc[va_m].reset_index(drop=True), va_vis.reset_index(drop=True)], axis=1) if va_vis is not None else X_base.iloc[va_m]

            # Group-safe 15% early stopping inside train fold
            u_keys = man.iloc[tr_m]['property_key'].unique()
            rng = np.random.default_rng(42 + f)
            es_keys = set(rng.choice(u_keys, size=max(1, int(len(u_keys) * 0.15)), replace=False))
            es_m = man.iloc[tr_m]['property_key'].isin(es_keys).to_numpy()
            fit_m = ~es_m

            reg = lgb.LGBMRegressor(
                objective='huber', alpha=0.15, n_estimators=1200, learning_rate=0.025,
                num_leaves=31, min_child_samples=25, subsample=0.8, subsample_freq=1,
                colsample_bytree=0.5, reg_alpha=1.0, reg_lambda=5.0,
                random_state=42 + f, n_jobs=6, verbose=-1,
            )
            reg.fit(
                Xtr.iloc[fit_m], y_target[tr_m][fit_m],
                eval_set=[(Xtr.iloc[es_m], y_target[tr_m][es_m])],
                callbacks=[lgb.early_stopping(60, verbose=False)],
            )
            p_log = reg.predict(Xva)
            oof_pred[va_m] = np.clip(np.exp(p_log) * y_area[va_m] * cpi_idx[va_m], 10_000, 1e7)
        return oof_pred

    # Helper to make static DataFrame slices for raw compact vision matrices
    def _static_block(mat, prefix):
        cols = [f'{prefix}_{k}' for k in range(mat.shape[1])]
        def _fn(tr_m, va_m):
            return pd.DataFrame(mat[tr_m], columns=cols), pd.DataFrame(mat[va_m], columns=cols)
        return _fn

    def _combine(*fns):
        def _fn(tr_m, va_m):
            trs, vas = [], []
            for fn in fns:
                a, b = fn(tr_m, va_m)
                if a is not None:
                    trs.append(a.reset_index(drop=True)); vas.append(b.reset_index(drop=True))
            return pd.concat(trs, axis=1), pd.concat(vas, axis=1)
        return _fn

    b_none = lambda tr_m, va_m: (None, None)
    b_render = _static_block(render_feats, 'vis_rnd')
    b_q = _static_block(q_feats, 'vis_q')
    b_so_q = _static_block(so_q_feats, 'vis_soq')
    b_phot = _static_block(photom_feats, 'vis_ph')
    b_sig_raw = lambda tr_m, va_m: _fit_transform_emb_block(sig_raw, tr_m, va_m, folds, n_pca=0, tag='sig_raw')
    b_sig_real = lambda tr_m, va_m: _fit_transform_emb_block(sig_real, tr_m, va_m, folds, n_pca=0, tag='sig_real')
    b_sig_real_p8 = lambda tr_m, va_m: _fit_transform_emb_block(sig_real, tr_m, va_m, folds, n_pca=8, tag='sig_real')
    b_so_sheet = lambda tr_m, va_m: _fit_transform_emb_block(so_sheet, tr_m, va_m, folds, n_pca=0, tag='so_sheet')
    b_dino_real = lambda tr_m, va_m: _fit_transform_emb_block(dino_real, tr_m, va_m, folds, n_pca=0, tag='dino_real')
    b_all_emb_concat = lambda tr_m, va_m: _fit_transform_emb_block(
        np.hstack([sig_real, so_sheet, dino_real]), tr_m, va_m, folds, n_pca=0, tag='all_vis'
    )

    recipes = [
        ('0. BASELINE (Tab+Spatial+Text ONLY, 0 vision)',   b_none),
        ('1. + Render/Media Flags ONLY (8d)',               b_render),
        ('2. + Photo Photometry ONLY (10d)',                b_phot),
        ('3. + Real-Photo Q-Scores + Room-Q (30d)',         b_q),
        ('4. + SigLIP2-base RAW (no render filter, 2d)',    b_sig_raw),
        ('5. + SigLIP2-base REAL-ONLY (render-filtered,2d)',b_sig_real),
        ('6. + SigLIP2-base REAL + PCA-8 (10d)',            b_sig_real_p8),
        ('7. + SigLIP2-SO400M-384 Contact Sheet (2d)',      b_so_sheet),
        ('8. + SO400M-384 Sheet + SO400M Q-Scores (12d)',   _combine(b_so_sheet, b_so_q)),
        ('9. + DINOv2-base REAL-ONLY (2d)',                 b_dino_real),
        ('10. MIX: SigLIP2-Real + SO400M-Sheet (4d)',       _combine(b_sig_real, b_so_sheet)),
        ('11. MIX: SigLIP2-Real + SO400M + RenderFlags(12d)', _combine(b_sig_real, b_so_sheet, b_render)),
        ('12. MIX: SigLIP2-Real + SO400M + DINOv2 (6d)',    _combine(b_sig_real, b_so_sheet, b_dino_real)),
        ('13. MIX: Joint Ridge [Sig+SO400M+DINO] + Rnd(10d)', _combine(b_all_emb_concat, b_render)),
        ('14. MIX: Compact Kitchen-Sink (Sig+SO+Q+Rnd+Ph)', _combine(b_sig_real, b_so_sheet, b_q, b_render, b_phot)),
    ]
    if has_vlm and vlm_mat is not None:
        b_vlm = _static_block(vlm_mat, 'vlm')
        recipes.append(('15. + Qwen3.6-35B Render-Aware VLM (11d)', b_vlm))
        recipes.append(('16. MIX: SigLIP2-Real + SO400M + VLM + Rnd', _combine(b_sig_real, b_so_sheet, b_vlm, b_render)))

    print("\n" + "=" * 112)
    print(f"VISION MODALITY MIX-AND-MATCH BENCHMARK (JOINT LightGBM Huber | n={n} homes | fold_0 val={int((folds==0).sum())})")
    print("=" * 112)
    print(f"  {'Recipe':50s} | {'f0 MdAPE':>8s} | {'5F MdAPE':>8s} | {'Δ5F':>7s} | {'Hit@10':>6s} | {'MAE (€)':>9s} | {'ΔMAE':>8s} | {'RMSE (€)':>9s}")
    print("  " + "-" * 108)

    results = {}
    base_mt = None
    f0_mask = folds == 0
    for name, fn in recipes:
        pred = eval_recipe(name, fn)
        mt_all = compute_metrics(y_price, pred)
        mt_f0 = compute_metrics(y_price[f0_mask], pred[f0_mask])
        if base_mt is None:
            base_mt = mt_all
            base_pred = pred.copy()
        d_md = mt_all['mdape'] - base_mt['mdape']
        d_mae = mt_all['mae'] - base_mt['mae']
        results[name] = {'f0': mt_f0, 'oof': mt_all, 'pred': pred}
        print(f"  {name:50s} | {mt_f0['mdape']:7.2f}% | {mt_all['mdape']:7.2f}% | {d_md:+6.2f}p | {mt_all['hit_10']:5.1f}% | €{mt_all['mae']:8,.0f} | €{d_mae:+7,.0f} | €{mt_all['rmse']:8,.0f}", flush=True)

    # Find best recipe by 5F MdAPE and show segment + blind-spot breakdown vs Baseline!
    best_name = min(results.keys(), key=lambda k: results[k]['oof']['mdape'])
    best_pred = results[best_name]['pred']
    print("\n" + "=" * 112)
    print(f"SLICE BREAKDOWN: BASELINE vs BEST VISION RECIPE ({best_name})")
    print("=" * 112)
    slices = [
        ('ALL homes',                  np.ones(n, dtype=bool)),
        ('fixer_upper',                (man['segment'] == 'fixer_upper').to_numpy()),
        ('historic_core',              (man['segment'] == 'historic_core').to_numpy()),
        ('luxury_650k_plus',           (man['segment'] == 'luxury_650k_plus').to_numpy()),
        ('house_typical',              (man['segment'] == 'house_typical').to_numpy()),
        ('apt_typical',                (man['segment'] == 'apt_typical').to_numpy()),
        ('short_desc (<650ch)',        (man['description_length'].fillna(0) < 650).to_numpy()),
        ('condition MISSING',          man['building_condition'].isna().to_numpy()),
        ('has CGI render (>=25%)',     (render_feats[:, 2] >= 0.25)),
        ('real photos ONLY (0% CGI)',  (render_feats[:, 2] == 0.0)),
    ]
    for sl_name, mk in slices:
        if mk.sum() < 20: continue
        mb = compute_metrics(y_price[mk], base_pred[mk])
        mv = compute_metrics(y_price[mk], best_pred[mk])
        print(f"  {sl_name:28s} (n={mk.sum():4d}) | Base: {mb['mdape']:5.2f}% (€{mb['mae']:7,.0f}) -> +Vis: {mv['mdape']:5.2f}% (€{mv['mae']:7,.0f}) | ΔMdAPE={mv['mdape']-mb['mdape']:+.2f}pp  ΔMAE=€{mv['mae']-mb['mae']:+,.0f}")

    # Also check orthogonal correlation of each individual vision meta-feature with TRUE Phase-4 full-ensemble residual!
    print("\n" + "=" * 112)
    print("ORTHOGONAL SIGNAL CHECK AGAINST FULL 19-MODEL PHASE-4 ENSEMBLE RESIDUAL (n=2,500)")
    print("=" * 112)
    for lbl, fn in [
        ('SigLIP2-base RAW (no render filter)', b_sig_raw),
        ('SigLIP2-base REAL-ONLY (filtered)',   b_sig_real),
        ('SigLIP2-SO400M-384 Contact Sheet',    b_so_sheet),
        ('DINOv2-base REAL-ONLY',               b_dino_real),
        ('Joint [SigLIP2+SO400M+DINOv2]',       b_all_emb_concat),
    ]:
        oof_psqm = np.zeros(n); oof_tres = np.zeros(n)
        for f in range(5):
            _, va_df = fn(folds != f, folds == f)
            oof_psqm[folds == f] = va_df.iloc[:, 0].to_numpy()
            oof_tres[folds == f] = va_df.iloc[:, 1].to_numpy()
        r_res = float(np.corrcoef(oof_tres, y_p4_resid)[0, 1])
        r_raw = float(np.corrcoef(oof_psqm - tab_anchor, y_p4_resid)[0, 1])
        print(f"  {lbl:40s} | corr(vis_tabresid, P4_resid)={r_res:+.4f} | corr(vis_psqm - tab, P4_resid)={r_raw:+.4f}")

    # Save JSON summary
    summary = {k: {'f0': v['f0'], 'oof': v['oof']} for k, v in results.items()}
    with open(REPORT_PATH, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved report -> {REPORT_PATH}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fetch', action='store_true')
    ap.add_argument('--encode', action='store_true')
    ap.add_argument('--bench', action='store_true')
    ap.add_argument('--all', action='store_true')
    args = ap.parse_args()
    if args.all or args.fetch:
        build_sandbox_sample_and_fetch()
    if args.all or args.encode:
        encode_all_vision_modalities()
    if args.all or args.bench:
        run_mix_and_match_bench()


if __name__ == '__main__':
    main()
