"""Vision Pilot Data Prep (CPU/Network-only — 0% GPU, safe to run alongside MPS training):
Builds a clean, stratified 800-listing pilot from `training_dataset.parquet` (spanning Historic Core
houses, standard houses, apartments, fixer-uppers, luxury homes, and all 5 CV folds) + checks coverage:
  1. Extracts up to 8 `mediumUrl` (300x200, ~15KB) photo URLs per listing from local raw HTML
     (`data/<property_id>/full_source_*.html`).
  2. Downloads them gently (8 threads) into `external_data/vision_pilot/photos/<property_id>/`.
  3. Downloads 2-scale official Flanders Aerial Orthophotos (`50m` parcel zoom + `200m` context)
     from `geo.api.vlaanderen.be/OMWRGBMRVL/wms` into `external_data/vision_pilot/aerial/`.
  4. Saves manifest `external_data/vision_pilot/pilot_manifest.parquet`.

Total footprint: ~110 MB, ~60-90s, 0% GPU impact.
"""
from __future__ import annotations

import concurrent.futures
import glob
import os
import sys
import time

import numpy as np
import pandas as pd
import requests
from pyproj import Transformer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'src'))
from parser import extract_classified_json

OUT_DIR = os.path.join(BASE_DIR, 'external_data', 'vision_pilot')
PHOTO_DIR = os.path.join(OUT_DIR, 'photos')
AERIAL_DIR = os.path.join(OUT_DIR, 'aerial')
DATASET_PATH = os.path.join(BASE_DIR, 'outputs', 'dataset', 'training_dataset.parquet')
P4_PREDS = os.path.join(BASE_DIR, 'outputs', 'models', 'workbench', 'phase4_frontier_preds.npz')

MAX_PHOTOS_PER_LISTING = 8
PILOT_N = 800


def pick_even_indices(n: int, k: int) -> list[int]:
    """Always keep photo 0 (hero/facade), plus evenly spaced interior photos up to k."""
    if n <= k:
        return list(range(n))
    return sorted(set(np.linspace(0, n - 1, k).round().astype(int).tolist()))


def select_pilot_rows(ds: pd.DataFrame, p4_oof: np.ndarray) -> pd.DataFrame:
    """Stratified 800-row sample covering all 5 folds + high-value segments (Historic Core,
    fixer-uppers, luxury, typical houses & apartments) for honest 5-fold CV pilot testing."""
    df = ds[['property_id', 'date_obtained', 'fold', 'property_key', 'price', 'livable_area',
             'property_type_HOUSE', 'neighbourhood', 'building_condition', 'energy_label',
             'latitude', 'longitude', 'publication_year']].copy()
    df['property_id'] = df['property_id'].astype(str)
    df['p4_oof_pred'] = p4_oof
    df['log_resid_p4'] = np.log(df['price'].to_numpy(dtype=float)) - np.log(np.maximum(p4_oof, 10_000.0))

    # Assign segment bucket for balanced sampling
    hist = {'Binnenstad', 'Elisabethbegijnhof_Prinsenhof_Papegaai_Sint_Michiels', 'Macharius_Heirnis', 'Sluizeken_Tolhuis_Ham'}
    cond_fix = {'TO_RENOVATE', 'TO_RESTORE', 'TO_BE_DONE_UP'}
    seg = np.full(len(df), 'other', dtype=object)
    is_h = df['property_type_HOUSE'].to_numpy() == 1
    seg[~is_h] = 'apt_typical'
    seg[is_h] = 'house_typical'
    seg[df['building_condition'].isin(cond_fix).to_numpy()] = 'fixer_upper'
    seg[df['neighbourhood'].isin(hist).to_numpy()] = 'historic_core'
    seg[df['price'].to_numpy() >= 650_000] = 'luxury_650k_plus'
    df['pilot_segment'] = seg

    # Ensure we only pick rows that have an HTML file on disk
    has_html = [os.path.isdir(os.path.join(BASE_DIR, 'data', pid)) for pid in df['property_id']]
    df = df.loc[has_html].copy()

    # Sample evenly across segments & folds (total ~800)
    quotas = {
        'historic_core': 180,
        'fixer_upper': 160,
        'luxury_650k_plus': 120,
        'house_typical': 180,
        'apt_typical': 160,
    }
    rng = np.random.default_rng(42)
    parts = []
    for sname, q in quotas.items():
        sub = df[df['pilot_segment'] == sname]
        # Stratified across folds
        take = min(q, len(sub))
        idx = rng.choice(sub.index.to_numpy(), size=take, replace=False)
        parts.append(df.loc[idx])
    out = pd.concat(parts, ignore_index=True)
    return out


def extract_urls_for_pid(pid: str) -> tuple[str, list[str], int]:
    d = os.path.join(BASE_DIR, 'data', pid)
    files = sorted(glob.glob(os.path.join(d, '*.html')))
    if not files:
        return pid, [], 0
    try:
        with open(files[0], 'r', errors='ignore') as f:
            pl = extract_classified_json(f.read())
        pics = (pl or {}).get('media', {}).get('pictures', []) if isinstance(pl, dict) else []
        urls = [p.get('mediumUrl') or p.get('largeUrl') or p.get('smallUrl') for p in pics]
        urls = [u for u in urls if isinstance(u, str) and u.startswith('http')]
        chosen = [urls[i] for i in pick_even_indices(len(urls), MAX_PHOTOS_PER_LISTING)]
        return pid, chosen, len(urls)
    except Exception:
        return pid, [], 0


def download_one_photo(args: tuple[str, int, str]) -> tuple[str, int, bool, int]:
    pid, idx, url = args
    pdir = os.path.join(PHOTO_DIR, pid)
    os.makedirs(pdir, exist_ok=True)
    fpath = os.path.join(pdir, f'{idx:02d}.jpg')
    if os.path.exists(fpath) and os.path.getsize(fpath) > 500:
        return pid, idx, True, os.path.getsize(fpath)
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200 and len(r.content) > 500:
            with open(fpath, 'wb') as f:
                f.write(r.content)
            return pid, idx, True, len(r.content)
    except Exception:
        pass
    return pid, idx, False, 0


def download_aerial_pair(args: tuple[str, float, float]) -> tuple[str, bool, bool]:
    pid, x, y = args
    ok50 = ok200 = False
    for tag, half in (('50m', 25.0), ('200m', 100.0)):
        fpath = os.path.join(AERIAL_DIR, f'{pid}_{tag}.jpg')
        if os.path.exists(fpath) and os.path.getsize(fpath) > 1000:
            if tag == '50m':
                ok50 = True
            else:
                ok200 = True
            continue
        url = (
            'https://geo.api.vlaanderen.be/OMWRGBMRVL/wms?service=WMS&version=1.3.0&request=GetMap'
            f'&layers=Ortho&styles=&crs=EPSG:31370&bbox={x-half:.1f},{y-half:.1f},{x+half:.1f},{y+half:.1f}'
            '&width=256&height=256&format=image/jpeg'
        )
        try:
            r = requests.get(url, timeout=12)
            if r.status_code == 200 and len(r.content) > 1000 and r.headers.get('Content-Type', '').startswith('image'):
                with open(fpath, 'wb') as f:
                    f.write(r.content)
                if tag == '50m':
                    ok50 = True
                else:
                    ok200 = True
        except Exception:
            pass
    return pid, ok50, ok200


def main():
    t0 = time.perf_counter()
    os.makedirs(PHOTO_DIR, exist_ok=True)
    os.makedirs(AERIAL_DIR, exist_ok=True)

    print("=== VISION PILOT FETCH (800 STRATIFIED LISTINGS, 0% GPU) ===", flush=True)
    ds = pd.read_parquet(DATASET_PATH)
    z4 = np.load(P4_PREDS, allow_pickle=True)
    pilot = select_pilot_rows(ds, z4['ens_oof_unified'])
    print(f"Selected {len(pilot)} listings across segments:\n{pilot['pilot_segment'].value_counts().to_string()}", flush=True)

    # 1. Extract photo URLs from raw HTML (fast, 4 workers)
    t1 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        res = list(ex.map(extract_urls_for_pid, pilot['property_id'].tolist()))
    url_map = {pid: urls for pid, urls, _ in res}
    tot_pics_map = {pid: ntot for pid, _, ntot in res}
    pilot['total_html_pics'] = pilot['property_id'].map(tot_pics_map)
    tasks = [(pid, i, u) for pid, urls in url_map.items() for i, u in enumerate(urls)]
    print(f"Extracted {len(tasks)} photo URLs from HTML in {time.perf_counter()-t1:.1f}s", flush=True)

    # 2. Download photos gently (8 threads -> zero CPU/GPU impact)
    t2 = time.perf_counter()
    ok_counts: dict[str, int] = {}
    total_bytes = 0
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        for pid, idx, ok, nbytes in ex.map(download_one_photo, tasks):
            done += 1
            if ok:
                ok_counts[pid] = ok_counts.get(pid, 0) + 1
                total_bytes += nbytes
            if done % 1500 == 0 or done == len(tasks):
                print(f"  Photos: {done}/{len(tasks)} | ok={sum(ok_counts.values())} | {total_bytes/1e6:.1f} MB ({time.perf_counter()-t2:.1f}s)", flush=True)

    pilot['n_photos_dl'] = pilot['property_id'].map(lambda p: ok_counts.get(p, 0))

    # 3. Download official Flanders 50m + 200m Aerial Orthophotos (8 threads)
    t3 = time.perf_counter()
    tf = Transformer.from_crs('EPSG:4326', 'EPSG:31370', always_xy=True)
    xs, ys = tf.transform(pilot['longitude'].to_numpy(), pilot['latitude'].to_numpy())
    aerial_tasks = list(zip(pilot['property_id'].tolist(), xs.tolist(), ys.tolist()))
    a50_map, a200_map = {}, {}
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        for pid, ok50, ok200 in ex.map(download_aerial_pair, aerial_tasks):
            done += 1
            a50_map[pid] = ok50
            a200_map[pid] = ok200
            if done % 250 == 0 or done == len(aerial_tasks):
                print(f"  Aerial ortho: {done}/{len(aerial_tasks)} | ok50={sum(a50_map.values())} ok200={sum(a200_map.values())} ({time.perf_counter()-t3:.1f}s)", flush=True)

    pilot['has_aerial_50m'] = pilot['property_id'].map(a50_map)
    pilot['has_aerial_200m'] = pilot['property_id'].map(a200_map)

    out_pq = os.path.join(OUT_DIR, 'pilot_manifest.parquet')
    pilot.to_parquet(out_pq, index=False)
    print(f"\nDONE in {time.perf_counter()-t0:.1f}s -> {out_pq}", flush=True)
    print(f"  Listings with >=4 photos : {(pilot['n_photos_dl'] >= 4).sum()}/{len(pilot)} ({(pilot['n_photos_dl'] >= 4).mean()*100:.1f}%)", flush=True)
    print(f"  Listings with >=1 photo  : {(pilot['n_photos_dl'] >= 1).sum()}/{len(pilot)} ({(pilot['n_photos_dl'] >= 1).mean()*100:.1f}%)", flush=True)
    print(f"  Listings with aerial 50m : {pilot['has_aerial_50m'].sum()}/{len(pilot)} ({pilot['has_aerial_50m'].mean()*100:.1f}%)", flush=True)


if __name__ == '__main__':
    main()
