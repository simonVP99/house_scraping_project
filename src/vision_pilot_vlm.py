"""Vision Pilot V3 — Contact-Sheet VLM Visual Appraiser via MLX (`Qwen3.5-9B-MLX-4bit` / `Qwen3.6-35B-A3B-5bit`):
Zero-GPU-contention contact-sheet builder + fast MLX VLM appraiser scoring Ghent real-estate photos on a
strict 8-axis Belgian real-estate valuation rubric (no rambling, `enable_thinking=False`, ~35 output tokens).

Why this beats generic CLIP/SigLIP cosine similarity:
  - A native 2026 VLM (`Qwen3.5-9B` / `Qwen3.6-35B-A3B`) can actually *reason* across all 6 rooms on a single
    `600x400` 3x2 contact sheet: spotting hidden moisture/peeling plaster, IKEA vs bespoke architect kitchens,
    authentic Gent herenhuis moldings vs 1970s laminate, single-glazed wooden frames, and renovation cost!

Usage (run under mlx-lab venv AFTER MPS GPU is free):
    # Step 1: Build 3x2 contact sheets (pure CPU, ~3s, safe anytime!)
    .venv-api/bin/python -u src/vision_pilot_vlm.py --build-sheets-only
    # Step 2: Run MLX VLM scoring on 800 pilot homes (or --limit 100 for quick speed/signal check)
    /Users/svp/dev/mlx-lab/.venv/bin/python -u src/vision_pilot_vlm.py --model mlx-community/Qwen3.5-9B-MLX-4bit
    # Step 3: Evaluate VLM rubric scores OOF on top of Phase-4 ensemble
    .venv-api/bin/python -u src/vision_pilot_vlm.py --eval-only
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PILOT_DIR = os.path.join(BASE_DIR, 'external_data', 'vision_pilot')
PHOTO_DIR = os.path.join(PILOT_DIR, 'photos')
SHEET_DIR = os.path.join(PILOT_DIR, 'contact_sheets')
MANIFEST_PATH = os.path.join(PILOT_DIR, 'pilot_manifest.parquet')
EMB_V1_CACHE = os.path.join(PILOT_DIR, 'pilot_embeddings.npz')

RUBRIC_PROMPT = (
    "You are an expert Belgian real-estate appraiser evaluating a 6-photo contact sheet of a home in Ghent.\n"
    "Output ONLY one compact JSON object with integer scores 1-10 (1=worst/dilapidated, 5=average dated, 8=modern renovated, 10=architectural luxury) "
    "and estimated renovation cost in kEUR (0 if move-in ready, up to 250 for full gut renovation):\n"
    '{"finish":1-10,"kitchen":1-10,"bath":1-10,"charm":1-10,"light":1-10,"exterior":1-10,"reno_need":1-10,"reno_keur":0-250,"luxury":1-10,"staging":1-10}\n'
    "Rules: reno_need 1=brand new/turnkey, 10=uninhabitable shell. Use null only if totally unseen. JSON only, no markdown, no explanation."
)

RUBRIC_KEYS = ['finish', 'kitchen', 'bath', 'charm', 'light', 'exterior', 'reno_need', 'reno_keur', 'luxury', 'staging']


def build_contact_sheets():
    """Pure CPU (~3s): builds one 600x400 (3x2) smart room-selected contact sheet per pilot listing."""
    os.makedirs(SHEET_DIR, exist_ok=True)
    df = pd.read_parquet(MANIFEST_PATH)
    pids = df['property_id'].astype(str).tolist()

    # Use our already-computed SigLIP2 room router if present to pick the 6 most informative rooms!
    room_arg = None
    photo_paths_by_pid: dict[str, list[tuple[str, int]]] = {}
    if os.path.exists(EMB_V1_CACHE):
        z = np.load(EMB_V1_CACHE, allow_pickle=True)
        sig_p, room_txt = z['sig_p'], z['room_txt']
        room_arg = (sig_p @ room_txt.T).argmax(axis=1)

    ptr = 0
    for pid in pids:
        ps = sorted(glob.glob(os.path.join(PHOTO_DIR, pid, '*.jpg')))
        items = []
        for p in ps:
            r_id = int(room_arg[ptr]) if (room_arg is not None and ptr < len(room_arg)) else -1
            items.append((p, r_id))
            ptr += 1
        photo_paths_by_pid[pid] = items

    # Priority room order: facade(0), living(2), kitchen(1), bathroom(3), garden(5), bedroom(4)
    prio_rooms = [0, 2, 1, 3, 5, 4]
    built = 0
    t0 = time.perf_counter()
    for pid in pids:
        out_p = os.path.join(SHEET_DIR, f'{pid}.jpg')
        if os.path.exists(out_p) and os.path.getsize(out_p) > 2000:
            built += 1
            continue
        items = photo_paths_by_pid.get(pid, [])
        if not items:
            continue
        # Exclude floorplans(6) if possible
        non_plan = [it for it in items if it[1] != 6]
        pool = non_plan if len(non_plan) >= 3 else items

        chosen: list[str] = []
        used_idx: set[int] = set()
        # First pass: 1 representative per priority room
        for r_target in prio_rooms:
            for j, (p, rid) in enumerate(pool):
                if j not in used_idx and rid == r_target:
                    chosen.append(p)
                    used_idx.add(j)
                    break
            if len(chosen) == 6:
                break
        # Fill remaining slots up to 6 from unused photos
        for j, (p, _) in enumerate(pool):
            if len(chosen) == 6:
                break
            if j not in used_idx:
                chosen.append(p)
                used_idx.add(j)

        # Compose 3x2 grid (each tile 200x200 -> 600x400 JPEG)
        canvas = Image.new('RGB', (600, 400), (20, 20, 20))
        for slot, p in enumerate(chosen[:6]):
            try:
                with Image.open(p) as im:
                    im = im.convert('RGB').resize((200, 200), Image.Resampling.BILINEAR)
                    cx = (slot % 3) * 200
                    cy = (slot // 3) * 200
                    canvas.paste(im, (cx, cy))
            except Exception:
                pass
        canvas.save(out_p, format='JPEG', quality=82)
        built += 1
    print(f"Built {built}/{len(pids)} 600x400 3x2 contact sheets in {time.perf_counter()-t0:.1f}s -> {SHEET_DIR}", flush=True)


def parse_rubric_json(text: str) -> dict:
    # Strip any thinking blocks if emitted
    t = re.sub(r'<think>.*?</think>', '', text, flags=re.S).strip()
    m = re.search(r'\{[^{}]*\}', t, flags=re.S)
    if not m:
        return {}
    try:
        obj = json.loads(m.group(0))
        out = {}
        for k in RUBRIC_KEYS:
            v = obj.get(k)
            if isinstance(v, (int, float)) and np.isfinite(v):
                out[k] = float(v)
            else:
                out[k] = np.nan
        return out
    except Exception:
        return {}


def run_vlm_scoring(model_id: str, limit: int | None = None, out_tag: str = 'qwen35_9b'):
    import mlx.core as mx
    from mlx_vlm import generate, load
    from mlx_vlm.prompt_utils import apply_chat_template

    out_pq = os.path.join(PILOT_DIR, f'vlm_scores_{out_tag}.parquet')
    df = pd.read_parquet(MANIFEST_PATH)
    pids = df['property_id'].astype(str).tolist()
    if limit is not None:
        pids = pids[:limit]

    # Resume support
    done_rows: dict[str, dict] = {}
    if os.path.exists(out_pq):
        prev = pd.read_parquet(out_pq)
        for _, r in prev.iterrows():
            done_rows[str(r['property_id'])] = r.to_dict()
        print(f"Resuming {out_tag}: {len(done_rows)} already scored.", flush=True)

    todo = [p for p in pids if p not in done_rows and os.path.exists(os.path.join(SHEET_DIR, f'{p}.jpg'))]
    print(f"Loading VLM {model_id} via mlx_vlm ({len(todo)} sheets to score)...", flush=True)
    t_load = time.perf_counter()
    model, processor = load(model_id)
    config = model.config
    print(f"Loaded in {time.perf_counter()-t_load:.1f}s", flush=True)

    # Build formatted prompt once (enable_thinking=False so Qwen3.5/3.6/3.8 doesn't burn 200 thinking tokens!)
    try:
        formatted = apply_chat_template(
            processor, config, RUBRIC_PROMPT, num_images=1, enable_thinking=False
        )
    except TypeError:
        formatted = apply_chat_template(processor, config, RUBRIC_PROMPT, num_images=1)

    t0 = time.perf_counter()
    parsed_ok = 0
    for idx, pid in enumerate(todo, 1):
        sheet_p = os.path.join(SHEET_DIR, f'{pid}.jpg')
        try:
            res = generate(
                model, processor, formatted, image=[sheet_p],
                max_tokens=90, temperature=0.0, verbose=False,
            )
            txt = res.text if hasattr(res, 'text') else str(res)
            parsed = parse_rubric_json(txt)
        except Exception as e:
            txt = f'ERR: {e}'
            parsed = {}
        row = {'property_id': pid, 'raw_ok': bool(parsed)}
        for k in RUBRIC_KEYS:
            row[f'vlm_{k}'] = parsed.get(k, np.nan)
        if parsed:
            parsed_ok += 1
        done_rows[pid] = row

        if idx % 25 == 0 or idx == len(todo):
            dt = time.perf_counter() - t0
            rate = idx / max(dt, 1e-3)
            eta = (len(todo) - idx) / max(rate, 1e-3)
            print(f"  [{out_tag}] {idx:4d}/{len(todo)} | ok={parsed_ok}/{idx} ({parsed_ok/idx*100:4.1f}%) | {rate:.2f} homes/s | elapsed={dt:5.1f}s | ETA={eta:5.1f}s", flush=True)
            if idx == 25:
                print(f"    sample raw: {txt[:140]!r}", flush=True)
            pd.DataFrame(list(done_rows.values())).to_parquet(out_pq, index=False)
            mx.clear_cache()

    out_df = pd.DataFrame(list(done_rows.values()))
    out_df.to_parquet(out_pq, index=False)
    print(f"Saved {out_pq} ({len(out_df)} rows) in {time.perf_counter()-t0:.1f}s!", flush=True)


def eval_vlm_scores(out_tag: str = 'qwen35_9b'):
    sys.path.insert(0, os.path.join(BASE_DIR, 'src'))
    from phase3_features import compute_metrics, fmt
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler

    out_pq = os.path.join(PILOT_DIR, f'vlm_scores_{out_tag}.parquet')
    df = pd.read_parquet(MANIFEST_PATH)
    df['property_id'] = df['property_id'].astype(str)
    sc_df = pd.read_parquet(out_pq)
    sc_df['property_id'] = sc_df['property_id'].astype(str)
    m = df.merge(sc_df, on='property_id', how='inner')
    vcols = [f'vlm_{k}' for k in RUBRIC_KEYS]
    m = m[m['raw_ok'].fillna(False)].reset_index(drop=True)
    print(f"=== VLM RUBRIC EVAL ({out_tag}, n={len(m)} scored homes) ===", flush=True)

    y_price = m['price'].to_numpy(float)
    p4 = m['p4_oof_pred'].to_numpy(float)
    y_res = m['log_resid_p4'].to_numpy(float)
    y_psqm = np.log(y_price / m['livable_area'].to_numpy(float))
    folds = m['fold'].to_numpy(int)
    base = compute_metrics(y_price, p4)
    print(f"  Baseline P4 (n={len(m)}): {fmt(base)}", flush=True)

    print("\n  Per-Axis VLM Rubric Correlations:", flush=True)
    for c in vcols:
        v = m[c].to_numpy(float)
        ok = np.isfinite(v)
        if ok.sum() > 30 and np.std(v[ok]) > 1e-8:
            rr = float(np.corrcoef(v[ok], y_res[ok])[0, 1])
            rp = float(np.corrcoef(v[ok], y_psqm[ok])[0, 1])
            print(f"    {c:16s} (n={ok.sum():3d}, mean={np.nanmean(v):5.2f}) | corr(P4_resid)={rr:+.4f} | corr(log_€/m²)={rp:+.4f}", flush=True)

    X = m[vcols].fillna(m[vcols].median()).to_numpy(float)
    oof_r = np.zeros(len(m))
    for f in np.unique(folds):
        tr, va = folds != f, folds == f
        sc = StandardScaler()
        mu = np.median(y_res[tr])
        mod = RidgeCV(alphas=np.logspace(0, 4, 25))
        mod.fit(sc.fit_transform(X[tr]), y_res[tr] - mu)
        oof_r[va] = mod.predict(sc.transform(X[va]))
    r = float(np.corrcoef(oof_r, y_res)[0, 1])
    for lam in (0.5, 0.75, 1.0):
        mt = compute_metrics(y_price, p4 * np.exp(lam * oof_r))
        print(f"  P4 + {lam:.2f}*VLM({out_tag}) | r={r:+.4f} | {fmt(mt)} | ΔMdAPE={mt['mdape']-base['mdape']:+.2f}pp ΔMAE=€{mt['mae']-base['mae']:+,.0f} ΔRMSE=€{mt['rmse']-base['rmse']:+,.0f}", flush=True)


def run_splash_scoring(api_url: str, model_name: str, limit: int | None = None, out_tag: str = 'qwen36_35b_splash', concurrency: int = 4):
    """Score contact sheets in parallel against a local Splash OpenAI-compatible server (Qwen3.6-35B-A3B-Splash)!"""
    import base64
    import concurrent.futures
    import requests

    out_pq = os.path.join(PILOT_DIR, f'vlm_scores_{out_tag}.parquet')
    df = pd.read_parquet(MANIFEST_PATH)
    pids = df['property_id'].astype(str).tolist()
    if limit is not None:
        pids = pids[:limit]

    done_rows: dict[str, dict] = {}
    if os.path.exists(out_pq):
        prev = pd.read_parquet(out_pq)
        for _, r in prev.iterrows():
            done_rows[str(r['property_id'])] = r.to_dict()
        print(f"Resuming {out_tag}: {len(done_rows)} already scored.", flush=True)

    todo = [p for p in pids if p not in done_rows and os.path.exists(os.path.join(SHEET_DIR, f'{p}.jpg'))]
    print(f"Scoring {len(todo)} contact sheets via Splash ({api_url}, model={model_name}, concurrency={concurrency})...", flush=True)

    def score_one(pid: str):
        sheet_p = os.path.join(SHEET_DIR, f'{pid}.jpg')
        b64 = base64.b64encode(open(sheet_p, 'rb').read()).decode('ascii')
        payload = {
            'model': model_name,
            'messages': [{
                'role': 'user',
                'content': [
                    {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{b64}'}},
                    {'type': 'text', 'text': RUBRIC_PROMPT},
                ],
            }],
            'temperature': 0.0,
            'max_tokens': 90,
            'chat_template_kwargs': {'enable_thinking': False},
        }
        try:
            r = requests.post(f'{api_url.rstrip("/")}/v1/chat/completions', json=payload, timeout=60)
            r.raise_for_status()
            msg = r.json()['choices'][0]['message']
            txt = msg.get('content') or ''
            parsed = parse_rubric_json(txt)
        except Exception as e:
            txt = f'ERR: {e}'
            parsed = {}
        row = {'property_id': pid, 'raw_ok': bool(parsed)}
        for k in RUBRIC_KEYS:
            row[f'vlm_{k}'] = parsed.get(k, np.nan)
        return pid, row, txt

    t0 = time.perf_counter()
    parsed_ok = 0
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
        for pid, row, txt in ex.map(score_one, todo):
            done += 1
            if row['raw_ok']:
                parsed_ok += 1
            done_rows[pid] = row
            if done % 25 == 0 or done == len(todo):
                dt = time.perf_counter() - t0
                rate = done / max(dt, 1e-3)
                eta = (len(todo) - done) / max(rate, 1e-3)
                print(f"  [{out_tag}] {done:4d}/{len(todo)} | ok={parsed_ok}/{done} ({parsed_ok/done*100:4.1f}%) | {rate:.2f} homes/s | elapsed={dt:5.1f}s | ETA={eta:5.1f}s", flush=True)
                if done == 25:
                    print(f"    sample raw: {txt[:140]!r}", flush=True)
                pd.DataFrame(list(done_rows.values())).to_parquet(out_pq, index=False)

    out_df = pd.DataFrame(list(done_rows.values()))
    out_df.to_parquet(out_pq, index=False)
    print(f"Saved {out_pq} ({len(out_df)} rows) in {time.perf_counter()-t0:.1f}s!", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--build-sheets-only', action='store_true')
    ap.add_argument('--eval-only', action='store_true')
    ap.add_argument('--splash-url', default=None, help='e.g. http://127.0.0.1:8091')
    ap.add_argument('--concurrency', type=int, default=4)
    ap.add_argument('--model', default='mlx-community/Qwen3.5-9B-MLX-4bit')
    ap.add_argument('--tag', default='qwen35_9b')
    ap.add_argument('--limit', type=int, default=None)
    args = ap.parse_args()

    if args.build_sheets_only:
        build_contact_sheets()
    elif args.eval_only:
        eval_vlm_scores(args.tag)
    elif args.splash_url:
        run_splash_scoring(args.splash_url, args.model, limit=args.limit, out_tag=args.tag, concurrency=args.concurrency)
    else:
        run_vlm_scoring(args.model, limit=args.limit, out_tag=args.tag)


if __name__ == '__main__':
    main()
