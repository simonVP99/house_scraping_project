"""Issue #1 — Full-Corpus Vision Pipeline (14,599 Training + 1,227 Fresh Nov 22–23 Holdout).

Scales the proven Sandbox Winner (Recipe #11 / #14: SigLIP2-base REAL-ONLY + SO400M-384 Contact Sheet
+ 8 Render/Media Flags + 30 Real-Photo Q-Scores + 10 Photometry Stats) cleanly to EVERY home!

Step 1 (`--fetch`, 0% GPU, ~12–15 min network-only):
  - Reuses all 2,500 homes already downloaded in `external_data/vision_sandbox/photos/` via hardlinks!
  - Downloads up to 6 evenly-spaced `mediumUrl` photos/home (~15 KB each) for the remaining ~12,100
    training homes + 1,227 fresh Nov 22–23 holdout homes into `external_data/vision_full/photos/<pid>/`.
  - Saves `external_data/vision_full/vision_full_manifest.parquet`.

Step 2 (`--encode`, MPS GPU, ~18 min total — NO `lightgbm` imported before `torch`!):
  - Encodes all photos with `google/siglip2-base-patch16-224` (bf16 bs=64, ~280 img/s -> ~4.5 min)
    -> 8 Render/Media flags + 30 Real-Photo Q-Scores & Room-Q + real-filtered 768-d mean embedding.
  - Builds smart real-photo-prioritized 3x2 contact sheets (`external_data/vision_full/contact_sheets/`).
  - Encodes every contact sheet with `google/siglip2-so400m-patch14-384` (bf16 bs=12, ~16.6 homes/s -> ~14 min)
    -> 1152-d holistic sheet embedding + 10 SO400M Q-scores.
  - Saves `external_data/vision_full/vision_full_embeddings.npz` (reusable across ALL future CV / temporal / live splits!).

Usage:
    .venv-api/bin/python -u src/vision_full_pipeline.py --fetch
    .venv-api/bin/python -u src/vision_full_pipeline.py --encode
"""
from __future__ import annotations

import argparse
import concurrent.futures
import glob
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd
import requests
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'src'))

from parser import extract_classified_json
from vision_sandbox import MEDIA_TYPE_PROMPTS, Q_PAIRS, ROOM_PROMPTS

FULL_DIR = os.path.join(BASE_DIR, 'external_data', 'vision_full')
PHOTO_DIR = os.path.join(FULL_DIR, 'photos')
SHEET_DIR = os.path.join(FULL_DIR, 'contact_sheets')
SANDBOX_PHOTO_DIR = os.path.join(BASE_DIR, 'external_data', 'vision_sandbox', 'photos')
PILOT_PHOTO_DIR = os.path.join(BASE_DIR, 'external_data', 'vision_pilot', 'photos')
MANIFEST_PATH = os.path.join(FULL_DIR, 'vision_full_manifest.parquet')
EMB_CACHE = os.path.join(FULL_DIR, 'vision_full_embeddings.npz')

DATASET_PATH = os.path.join(BASE_DIR, 'outputs', 'dataset', 'training_dataset.parquet')
FRESH_CSV = os.path.join(BASE_DIR, 'outputs', 'models', 'workbench', 'fresh_nov23_predictions.csv')
DATA_HTML_DIR = os.path.join(BASE_DIR, 'data')

MAX_PHOTOS = 6


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
        idx = np.linspace(0, total - 1, max_pics, dtype=int)
        return [urls[i] for i in idx], total
    except Exception:
        return [], 0


def _link_existing_photos(pid: str, out_d: str) -> int:
    for src_root in (SANDBOX_PHOTO_DIR, PILOT_PHOTO_DIR):
        sd = os.path.join(src_root, pid)
        if os.path.isdir(sd):
            pp = sorted(glob.glob(os.path.join(sd, '*.jpg')))
            if pp:
                idx = np.linspace(0, len(pp) - 1, min(len(pp), MAX_PHOTOS), dtype=int)
                cnt = 0
                for k, j in enumerate(idx):
                    dst = os.path.join(out_d, f'{k:02d}.jpg')
                    if not os.path.exists(dst):
                        try:
                            os.link(pp[j], dst)
                        except OSError:
                            shutil.copy2(pp[j], dst)
                    cnt += 1
                return cnt
    return 0


def fetch_full_corpus_photos():
    os.makedirs(PHOTO_DIR, exist_ok=True)
    tr = pd.read_parquet(DATASET_PATH, columns=['property_id'])
    pids = list(dict.fromkeys(tr['property_id'].astype(str).tolist()))
    n_train = len(pids)
    if os.path.exists(FRESH_CSV):
        fr = pd.read_csv(FRESH_CSV, usecols=['property_id'])
        for pid in fr['property_id'].astype(str):
            if pid not in set(pids):
                pids.append(pid)
    # Dedupe preserving order cleanly
    pids = list(dict.fromkeys(pids))
    print(f"=== FULL VISION FETCH: {len(pids):,} unique homes ({n_train:,} train + {len(pids)-n_train:,} fresh) ===", flush=True)

    t0 = time.perf_counter()
    reused = 0
    already_full = 0
    tasks: list[tuple[str, str]] = []
    total_pics_map: dict[str, int] = {}

    # Parallel HTML scan for photo URLs (pure disk IO, super fast with 8 threads!)
    def _prep_home(pid: str):
        out_d = os.path.join(PHOTO_DIR, pid)
        os.makedirs(out_d, exist_ok=True)
        existing = sorted(glob.glob(os.path.join(out_d, '*.jpg')))
        did_reuse = False
        if not existing:
            n_linked = _link_existing_photos(pid, out_d)
            if n_linked > 0:
                did_reuse = True
                existing = sorted(glob.glob(os.path.join(out_d, '*.jpg')))
        if len(existing) >= MAX_PHOTOS:
            return pid, len(existing), [], did_reuse, True
        urls, tot = _extract_medium_urls(pid, MAX_PHOTOS)
        if len(existing) >= min(len(urls), MAX_PHOTOS) and len(existing) > 0:
            return pid, tot, [], did_reuse, True
        home_tasks = []
        for k, u in enumerate(urls):
            dst = os.path.join(out_d, f'{k:02d}.jpg')
            if not os.path.exists(dst):
                home_tasks.append((u, dst))
        return pid, tot, home_tasks, did_reuse, False

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        for idx, (pid, tot, h_tasks, did_reuse, is_done) in enumerate(ex.map(_prep_home, pids), 1):
            total_pics_map[pid] = tot
            if did_reuse: reused += 1
            if is_done: already_full += 1
            tasks.extend(h_tasks)
            if idx % 3000 == 0 or idx == len(pids):
                print(f"  HTML scanned {idx:5d}/{len(pids)} | reused={reused} | complete={already_full} | queued_dl={len(tasks):,} ({time.perf_counter()-t0:.1f}s)", flush=True)

    # Download queued photos politely with 16 threads (CDN media-resize.immowebstatic.be)
    if tasks:
        t_dl = time.perf_counter()
        ok_dl, fail_dl = 0, 0
        # Thread-local sessions for keep-alive speed
        import threading
        _tls = threading.local()

        def _get_sess():
            s = getattr(_tls, 's', None)
            if s is None:
                s = requests.Session()
                s.headers.update({'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'})
                _tls.s = s
            return s

        def _dl_one(item):
            u, dst = item
            sess = _get_sess()
            for _ in range(2):
                try:
                    r = sess.get(u, timeout=10)
                    if r.status_code == 200 and len(r.content) > 800:
                        with open(dst, 'wb') as f:
                            f.write(r.content)
                        return True
                    if r.status_code in (403, 404, 410):
                        return False
                except Exception:
                    pass
            return False

        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
            for done, ok in enumerate(ex.map(_dl_one, tasks), 1):
                if ok: ok_dl += 1
                else: fail_dl += 1
                if done % 3000 == 0 or done == len(tasks):
                    dt = time.perf_counter() - t_dl
                    rate = done / max(dt, 1e-3)
                    eta = (len(tasks) - done) / max(rate, 1e-3)
                    print(f"  DL {done:6d}/{len(tasks):,} | ok={ok_dl:,} fail={fail_dl:,} | {rate:.0f} img/s | ETA {eta:.0f}s", flush=True)

    n_dl = [len(glob.glob(os.path.join(PHOTO_DIR, pid, '*.jpg'))) for pid in pids]
    man = pd.DataFrame({
        'property_id': pids,
        'total_html_pics': [total_pics_map[p] for p in pids],
        'n_photos_dl': n_dl,
    })
    man.to_parquet(MANIFEST_PATH, index=False)
    print(
        f"=== FULL FETCH DONE: {len(man):,} homes | {(man['n_photos_dl']>=1).sum():,} with >=1 photo "
        f"({100*(man['n_photos_dl']>=1).mean():.1f}%) | total JPEGs={sum(n_dl):,} -> {MANIFEST_PATH} ===",
        flush=True,
    )


def encode_full_corpus():
    """Encode ALL ~15,600 homes on MPS with SigLIP2-base + RenderFilter + Q-Scores + SO400M-384 Contact Sheet."""
    import torch
    from transformers import AutoModel, AutoProcessor

    man = pd.read_parquet(MANIFEST_PATH)
    pids = man['property_id'].astype(str).tolist()
    n_homes = len(pids)

    all_paths: list[str] = []
    offsets = np.zeros(n_homes + 1, dtype=np.int32)
    for i, pid in enumerate(pids):
        ps = sorted(glob.glob(os.path.join(PHOTO_DIR, pid, '*.jpg')))
        all_paths.extend(ps)
        offsets[i + 1] = len(all_paths)
    n_photos = len(all_paths)
    print(f"=== FULL VISION ENCODE ({n_homes:,} homes, {n_photos:,} photos) ===", flush=True)

    # 1. Fast photometry stats (CPU threads)
    t0 = time.perf_counter()
    photo_stats = np.zeros((n_photos, 5), dtype=np.float32)

    def _stat_one(p):
        try:
            with Image.open(p) as im:
                im = im.convert('RGB')
                w, h = im.size
                is_v = 1.0 if h > w else 0.0
                sm = np.asarray(im.resize((64, 64), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
                gray = 0.299 * sm[..., 0] + 0.587 * sm[..., 1] + 0.114 * sm[..., 2]
                mx = sm.max(axis=-1); mn = sm.min(axis=-1)
                sat = np.where(mx > 1e-4, (mx - mn) / np.maximum(mx, 1e-4), 0.0)
                gx = np.abs(np.diff(gray, axis=1)).mean()
                gy = np.abs(np.diff(gray, axis=0)).mean()
                return np.array([gray.mean(), gray.std(), sat.mean(), is_v, gx + gy], dtype=np.float32)
        except Exception:
            return np.full(5, np.nan, dtype=np.float32)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        for i, st in enumerate(ex.map(_stat_one, all_paths)):
            photo_stats[i] = st
    print(f"  Photometry done in {time.perf_counter()-t0:.1f}s", flush=True)

    def _safe_img(p):
        try:
            with Image.open(p) as im:
                return im.convert('RGB').copy()
        except Exception:
            return Image.new('RGB', (224, 224), (128, 128, 128))

    # 2. SigLIP2-base-224 on all photos (streamed in batches; pool per listing immediately to save RAM!)
    t1 = time.perf_counter()
    sig_id = 'google/siglip2-base-patch16-224'
    sig_proc = AutoProcessor.from_pretrained(sig_id, use_fast=True)
    sig_mod = AutoModel.from_pretrained(sig_id, dtype=torch.bfloat16).to('mps').eval()

    with torch.no_grad():
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

        sig_embs_all = np.zeros((n_photos, 768), dtype=np.float16)
        bs = 64
        for i in range(0, n_photos, bs):
            imgs = [_safe_img(p) for p in all_paths[i:i + bs]]
            inp = sig_proc(images=imgs, return_tensors='pt')
            inp = {k: v.to('mps', dtype=torch.bfloat16) if v.is_floating_point() else v.to('mps') for k, v in inp.items()}
            f = sig_mod.get_image_features(**inp)
            if not isinstance(f, torch.Tensor): f = f.pooler_output
            f = f.float()
            f = f / f.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            sig_embs_all[i:i + len(imgs)] = f.cpu().numpy().astype(np.float16)
            if (i // bs) % 150 == 0 and i > 0:
                dt = time.perf_counter() - t1
                eta = (n_photos - i) / max(i / dt, 1e-3)
                print(f"  [SigLIP2-base] {i:6d}/{n_photos:,} | {i/dt:.0f} img/s | ETA {eta:.0f}s", flush=True)

    del sig_mod
    torch.mps.empty_cache()
    print(f"  [SigLIP2-base] Done {n_photos:,} photos in {time.perf_counter()-t1:.1f}s", flush=True)

    # Aggregate listing-level REAL-FILTERED SigLIP2 embeddings, Render flags, Q-scores, Photometry + build 3x2 sheets!
    os.makedirs(SHEET_DIR, exist_ok=True)
    t_agg = time.perf_counter()
    sig_real = np.full((n_homes, 768), np.nan, dtype=np.float32)
    render_feats = np.full((n_homes, 8), np.nan, dtype=np.float32)
    q_feats = np.full((n_homes, 30), np.nan, dtype=np.float32)
    photom_feats = np.full((n_homes, 10), np.nan, dtype=np.float32)
    sheet_paths: list[str] = []
    prio_rooms = [0, 2, 1, 3, 5, 4]

    for i, pid in enumerate(pids):
        s, e = int(offsets[i]), int(offsets[i + 1])
        if e <= s:
            sheet_paths.append('')
            continue
        sp = sig_embs_all[s:e].astype(np.float32)
        ms = sp @ media_txt.T
        cm = ms[:, 1] - ms[:, 0]
        rm = (sp @ room_txt.T).argmax(axis=1)
        qa = (sp @ q_pos_txt.T) - (sp @ q_neg_txt.T)
        cls = ms.argmax(axis=1)
        ps = photo_stats[s:e]

        is_plan = (rm == 6) | (cls == 3)
        is_cgi = (cm > 0.0) | (cls == 1)
        is_vstage = (cls == 2)
        non_plan = ~is_plan
        real_only = (~is_plan) & (~is_cgi)

        use_raw = non_plan if non_plan.any() else np.ones(e - s, dtype=bool)
        use_real = real_only if real_only.any() else use_raw

        sig_real[i] = sp[use_real].mean(axis=0)
        render_feats[i] = [
            float(cm.mean()), float(cm.max()), float(is_cgi.mean()),
            float(is_vstage.mean()), float((is_cgi | is_vstage).mean()),
            float((~is_cgi & ~is_vstage & ~is_plan).mean()),
            float(is_plan.mean()), float((cls == 4).mean()),
        ]

        qm = qa[use_real].mean(axis=0)
        qp = np.percentile(qa[use_real], 75, axis=0)
        def _rq(room_idx_set, q_col, agg='mean'):
            mk = use_real & np.isin(rm, room_idx_set)
            vals = qa[mk, q_col] if mk.any() else qa[use_real, q_col]
            return float(np.max(vals) if agg == 'max' else (np.min(vals) if agg == 'min' else np.mean(vals)))

        room_q = np.array([
            _rq([1], 3), _rq([3], 4), _rq([0], 6), _rq([5], 8),
            _rq([2, 4], 5), _rq([2, 4], 2),
            _rq([0, 1, 2, 3, 4], 1, 'max'),
            _rq([0, 1, 2, 3, 4], 0, 'min'),
            float((rm == 1).any()), float((rm == 5).any()),
        ], dtype=np.float32)
        q_feats[i] = np.concatenate([qm, qp, room_q])

        ok_ps = ps[use_real]
        if len(ok_ps) > 0:
            photom_feats[i] = np.concatenate([np.nanmean(ok_ps, axis=0), np.nanstd(ok_ps, axis=0)])

        # Build smart real-photo-prioritized 3x2 contact sheet if not already built
        out_p = os.path.join(SHEET_DIR, f'{pid}.jpg')
        if not (os.path.exists(out_p) and os.path.getsize(out_p) > 2000):
            idxs = list(range(e - s))
            real_idxs = [j for j in idxs if (not is_plan[j]) and (not is_cgi[j])]
            pool = real_idxs if len(real_idxs) >= 3 else ([j for j in idxs if not is_plan[j]] or idxs)
            chosen, used = [], set()
            for rt in prio_rooms:
                for j in pool:
                    if j not in used and rm[j] == rt:
                        chosen.append(all_paths[s + j]); used.add(j); break
                if len(chosen) == 6: break
            for j in pool:
                if len(chosen) == 6: break
                if j not in used: chosen.append(all_paths[s + j]); used.add(j)
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

    del sig_embs_all
    print(f"  Aggregated + built {sum(bool(x) for x in sheet_paths):,} contact sheets in {time.perf_counter()-t_agg:.1f}s", flush=True)

    # 3. SigLIP2-SO400M-384 on the 3x2 Contact Sheet (1 high-res pass per home!)
    t2 = time.perf_counter()
    so_id = 'google/siglip2-so400m-patch14-384'
    so_proc = AutoProcessor.from_pretrained(so_id, use_fast=True)
    so_mod = AutoModel.from_pretrained(so_id, dtype=torch.bfloat16).to('mps').eval()
    so_sheet = np.full((n_homes, 1152), np.nan, dtype=np.float32)
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
            so_sheet[batch_i] = f.cpu().numpy()
            if (k // bs_so) % 100 == 0 and k > 0:
                dt = time.perf_counter() - t2
                eta = (len(valid_sh) - k) / max(k / dt, 1e-3)
                print(f"  [SO400M-384 Sheet] {k:5d}/{len(valid_sh):,} | {k/dt:.1f} homes/s | ETA {eta:.0f}s", flush=True)

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

    so_q_feats = (so_sheet @ so_q_pos.T) - (so_sheet @ so_q_neg.T)
    print(f"  [SO400M-384 Sheet] Done {len(valid_sh):,} homes in {time.perf_counter()-t2:.1f}s", flush=True)

    np.savez_compressed(
        EMB_CACHE,
        property_id=np.array(pids, dtype=object),
        sig_real=sig_real.astype(np.float16),
        so_sheet=so_sheet.astype(np.float16),
        render_feats=render_feats,
        q_feats=q_feats,
        so_q_feats=so_q_feats.astype(np.float32),
        photom_feats=photom_feats,
    )
    print(f"=== FULL VISION ENCODE SAVED -> {EMB_CACHE} ({os.path.getsize(EMB_CACHE)/1e6:.1f} MB) ===", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fetch', action='store_true')
    ap.add_argument('--encode', action='store_true')
    ap.add_argument('--all', action='store_true')
    args = ap.parse_args()
    if args.all or args.fetch:
        fetch_full_corpus_photos()
    if args.all or args.encode:
        encode_full_corpus()


if __name__ == '__main__':
    main()
