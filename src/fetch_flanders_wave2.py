"""Issue #2 — Wave-2 Ghent Micro-Location Open-Data Enrichment.

Fetches & caches 7 official Flemish / Stad Gent geospatial layers verified HTTP 200
and attaches them to every unique `(lat_r5, lon_r5)` coordinate in
`external_data/flanders/ghent_coord_enrichment.parquet` (zero leakage across CV/temporal/live!):

  1. **Official Watertoets Flood Risk A–D** (VMM `inspirepub.waterinfo.be` ArcGIS REST `/query`
     on fluvial + pluvial `overstromingsgevoelige_gebieden`, batched multipoints):
       - `fl_flood_fluvial_class` (`0=A..3=D`)
       - `fl_flood_pluvial_class` (`0=A..3=D`)
       - `fl_flood_max_class` (`max(fluvial, pluvial)`)
  2. **IRCEL ATMO-Street 2025 10m Air Quality** (`geo.irceline.be/wcs`):
       - `fl_no2_2025_ugm3` (street-canyon NO2, 10m resolution!)
       - `fl_bc_2025_ugm3` (black carbon traffic exposure, 10m!)
       - `fl_pm25_2025_ugm3` (PM2.5, 10m!)
  3. **Stad Gent Street Trees + Flanders 10m Buurtgroen & Urban Heat**:
       - `gent_trees_100m`, `gent_mature_trees_100m` (`stamomtrek >= 100cm`), `gent_tree_girth_sum_100m`
       - `fl_buurtgroen_10m` (Mercator WCS `hh__hh_buurtgroen_2019`, 10m raster)
       - `fl_urban_heat_c` (Mercator WCS `hh__hh_ges_hitte_vlabru_2018`)
  4. **VHA Navigable Waterfront (`CATC=0`, Leie/Schelde/Coupure/canals)**:
       - `fl_dist_navigable_water_m`, `fl_near_water_100m`, `fl_near_water_300m`
  5. **Tram Stops & Rail Stations** (filter already-cached `gent_bushaltes-gent.geojson` for `TRAM` + `locatie-stations`):
       - `gent_dist_tram_m`, `gent_tram_stops_400m`, `gent_dist_rail_station_m`
  6. **Wegenregister Motorways & Highway Ramps (E17/E40/R4)**:
       - `fl_dist_motorway_m` (noise/severance)
       - `fl_dist_highway_ramp_m` (commuter access!)
  7. **Stad Gent Circulatieplan (Autovrij / Pedestrian Core) & EV/Carshare**:
       - `gent_in_autovrij`, `gent_circulatie_sector`, `gent_cambio_500m`, `gent_ev_chargers_400m`

Outputs:
  `external_data/flanders/ghent_coord_enrichment_v2.parquet` (all 8,822+ coords x new columns)
  AND merges cleanly into `ghent_coord_enrichment.parquet`!

Usage:
    .venv-api/bin/python -u src/fetch_flanders_wave2.py
"""
from __future__ import annotations

import io
import json
import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from PIL import Image
from scipy.spatial import cKDTree
from shapely.geometry import Point

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE_DIR, 'external_data', 'flanders')
W2_DIR = os.path.join(OUT_DIR, 'wave2')
os.makedirs(W2_DIR, exist_ok=True)

ENR_PATH = os.path.join(OUT_DIR, 'ghent_coord_enrichment.parquet')
W2_OUT_PATH = os.path.join(OUT_DIR, 'ghent_coord_enrichment_v2.parquet')


def _dl(name: str, url: str, dest: str, timeout: int = 120) -> str:
    if os.path.exists(dest) and os.path.getsize(dest) > 200:
        print(f"  [cached] {name}: {os.path.getsize(dest)/1e6:.2f} MB", flush=True)
        return dest
    t0 = time.perf_counter()
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    with open(dest, 'wb') as f:
        f.write(r.content)
    print(f"  [fetched] {name}: {len(r.content)/1e6:.2f} MB in {time.perf_counter()-t0:.1f}s", flush=True)
    return dest


def _sample_geotiff(tif_path: str, xs: np.ndarray, ys: np.ndarray,
                    vmin: float = -1e6, vmax: float = 1e6) -> np.ndarray:
    """Sample a single-band EPSG:31370 GeoTIFF using PIL tags (handles both 34264 & 33550/33922)."""
    im = Image.open(tif_path)
    arr = np.asarray(im, dtype=np.float32)
    tags = im.tag_v2
    if 34264 in tags:
        m = tags[34264]
        sx, ox, sy, oy = float(m[0]), float(m[3]), abs(float(m[5])), float(m[7])
    else:
        scale = tags[33550]
        tie = tags[33922]
        sx, sy = float(scale[0]), float(scale[1])
        ox, oy = float(tie[3]), float(tie[4])
    cols = np.floor((xs - ox) / sx).astype(int)
    rows = np.floor((oy - ys) / sy).astype(int)
    h, w = arr.shape[:2]
    out = np.full(len(xs), np.nan, dtype=np.float32)
    ok = (cols >= 0) & (cols < w) & (rows >= 0) & (rows < h)
    vals = arr[rows[ok], cols[ok]]
    vals = np.where((vals >= vmin) & (vals <= vmax) & np.isfinite(vals), vals, np.nan)
    out[ok] = vals
    return out


# ── 1. Watertoets Fluvial & Pluvial A–D (5m/px ArcGIS REST /export PNG tiles!) ──
def sample_watertoets_flood(xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sample VMM Watertoets fluvial & pluvial A-D classes using 5m/px tiled PNG export.

    Exact official RGB legend from inspirepub.waterinfo.be:
      transparent/white -> 0 (Class A, no flood risk)
      light blue [138, 188, 249] -> 1 (Class B, small under climate change)
      medium blue [43, 118, 227] -> 2 (Class C, small probability)
      dark blue  [0, 69, 138]    -> 3 (Class D, medium probability)
    """
    cache_p = os.path.join(W2_DIR, 'watertoets_sampled.npz')
    if os.path.exists(cache_p):
        z = np.load(cache_p)
        if len(z['fluvial']) == len(xs):
            print(f"  [cached] Watertoets flood classes for {len(xs)} coords", flush=True)
            return z['fluvial'], z['pluvial']

    # Group coords into 4000m x 4000m tiles (800x800 px = 5m/px, safely below minScale=25000!)
    tile_sz = 4000.0
    tx = np.floor(xs / tile_sz).astype(int)
    ty = np.floor(ys / tile_sz).astype(int)
    tiles = sorted(set(zip(tx.tolist(), ty.tolist())))
    print(f"  [Watertoets] Sampling {len(xs)} coords across {len(tiles)} populated 4km tiles (5m/px)...", flush=True)

    endpoints = {
        'fluvial': 'https://inspirepub.waterinfo.be/arcgis/rest/services/informatieplicht/overstromingsgevoelige_gebieden_fluviaal/MapServer/export',
        'pluvial': 'https://inspirepub.waterinfo.be/arcgis/rest/services/informatieplicht/overstromingsgevoelige_gebieden_pluviaal/MapServer/export',
    }
    palette = np.array([[138, 188, 249], [43, 118, 227], [0, 69, 138]], dtype=np.float32)  # B=1, C=2, D=3

    def _fetch_tile(kind: str, tile_xy: tuple[int, int]) -> tuple[str, tuple[int, int], np.ndarray | None]:
        gx, gy = tile_xy
        x0, y0 = gx * tile_sz, gy * tile_sz
        x1, y1 = x0 + tile_sz, y0 + tile_sz
        tile_file = os.path.join(W2_DIR, f'wt_{kind}_{gx}_{gy}.png')
        if os.path.exists(tile_file) and os.path.getsize(tile_file) > 100:
            im = Image.open(tile_file).convert('RGBA')
            return kind, tile_xy, np.asarray(im)
        params = {
            'bbox': f'{x0},{y0},{x1},{y1}',
            'bboxSR': '31370',
            'imageSR': '31370',
            'size': '800,800',
            'format': 'png32',
            'transparent': 'true',
            'f': 'image',
        }
        for _ in range(3):
            try:
                r = requests.get(endpoints[kind], params=params, timeout=30)
                if r.status_code == 200 and r.content[:4] == b'\x89PNG':
                    with open(tile_file, 'wb') as f:
                        f.write(r.content)
                    im = Image.open(io.BytesIO(r.content)).convert('RGBA')
                    return kind, tile_xy, np.asarray(im)
            except Exception:
                time.sleep(0.5)
        return kind, tile_xy, None

    t0 = time.perf_counter()
    fluv = np.zeros(len(xs), dtype=np.int8)
    pluv = np.zeros(len(xs), dtype=np.int8)

    # Index points per tile
    tile_to_idx: dict[tuple[int, int], list[int]] = {}
    for i, txy in enumerate(zip(tx.tolist(), ty.tolist())):
        tile_to_idx.setdefault(txy, []).append(i)

    tasks = [(k, txy) for k in ('fluvial', 'pluvial') for txy in tiles]
    with ThreadPoolExecutor(max_workers=8) as ex:
        for kind, txy, rgba in ex.map(lambda a: _fetch_tile(*a), tasks):
            if rgba is None:
                continue
            gx, gy = txy
            x0, y0 = gx * tile_sz, gy * tile_sz
            y1 = y0 + tile_sz
            idxs = np.array(tile_to_idx[txy], dtype=int)
            cols = np.clip(np.floor((xs[idxs] - x0) / 5.0).astype(int), 0, 799)
            rows = np.clip(np.floor((y1 - ys[idxs]) / 5.0).astype(int), 0, 799)
            # Take max class in a 3x3 window (15m parcel radius, handles sidewalk-snapped coords!)
            cls_best = np.zeros(len(idxs), dtype=np.int8)
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr = np.clip(rows + dr, 0, 799)
                    cc = np.clip(cols + dc, 0, 799)
                    px = rgba[rr, cc]  # (m, 4)
                    vis = px[:, 3] > 80
                    if not vis.any():
                        continue
                    rgb = px[vis, :3].astype(np.float32)
                    dists = np.linalg.norm(rgb[:, None, :] - palette[None, :, :], axis=-1)
                    cls = (dists.argmin(axis=1) + 1).astype(np.int8)
                    # Only accept if reasonably close to one of the 3 official blue tones
                    cls = np.where(dists.min(axis=1) < 90.0, cls, 0)
                    tmp = np.zeros(len(idxs), dtype=np.int8)
                    tmp[vis] = cls
                    cls_best = np.maximum(cls_best, tmp)
            if kind == 'fluvial':
                fluv[idxs] = cls_best
            else:
                pluv[idxs] = cls_best

    print(
        f"  [Watertoets] Done in {time.perf_counter()-t0:.1f}s | "
        f"fluvial B/C/D={(fluv==1).sum()}/{(fluv==2).sum()}/{(fluv==3).sum()} | "
        f"pluvial B/C/D={(pluv==1).sum()}/{(pluv==2).sum()}/{(pluv==3).sum()}",
        flush=True,
    )
    np.savez_compressed(cache_p, fluvial=fluv, pluvial=pluv)
    return fluv, pluv


# ── 2. IRCEL ATMO-Street 2025 10m Air Quality (NO2, BC, PM2.5) ────────────────
def sample_ircel_air_quality(xs: np.ndarray, ys: np.ndarray) -> dict[str, np.ndarray]:
    """Tile Ghent into populated 4km x 4km WCS subsets for 10m street-canyon NO2, BC & PM2.5."""
    cache_p = os.path.join(W2_DIR, 'ircel_2025_sampled.npz')
    if os.path.exists(cache_p):
        z = np.load(cache_p)
        if len(z['fl_no2_2025_ugm3']) == len(xs):
            print(f"  [cached] IRCEL 2025 10m air quality for {len(xs)} coords", flush=True)
            return {k: z[k] for k in z.files}

    tile_sz = 4000.0
    tx = np.floor(xs / tile_sz).astype(int)
    ty = np.floor(ys / tile_sz).astype(int)
    tiles = sorted(set(zip(tx.tolist(), ty.tolist())))
    tile_to_idx: dict[tuple[int, int], list[int]] = {}
    for i, txy in enumerate(zip(tx.tolist(), ty.tolist())):
        tile_to_idx.setdefault(txy, []).append(i)

    covs = {
        'fl_no2_2025_ugm3': 'rioifdm__no2_anmean_2025_atmostreet_vl',
        'fl_bc_2025_ugm3':  'rioifdm__bc_anmean_2025_atmostreet_vl',
        'fl_pm25_2025_ugm3':'rioifdm__pm25_anmean_2025_atmostreet_vl',
    }
    out = {k: np.full(len(xs), np.nan, dtype=np.float32) for k in covs}
    print(f"  [IRCEL 2025 10m] Sampling 3 pollutants across {len(tiles)} populated 4km tiles...", flush=True)
    t0 = time.perf_counter()

    def _fetch_one(args):
        col, cov_id, txy = args
        gx, gy = txy
        x0, y0 = int(gx * tile_sz), int(gy * tile_sz)
        x1, y1 = int(x0 + tile_sz), int(y0 + tile_sz)
        dest = os.path.join(W2_DIR, f'ircel_{col}_{gx}_{gy}.tif')
        if not (os.path.exists(dest) and os.path.getsize(dest) > 500):
            url = (
                'https://geo.irceline.be/wcs?service=WCS&version=2.0.1&request=GetCoverage'
                f'&coverageId={cov_id}'
                f'&subset=X({x0},{x1})&subset=Y({y0},{y1})'
                '&format=image/tiff'
            )
            for _ in range(3):
                try:
                    r = requests.get(url, timeout=45)
                    if r.status_code == 200 and len(r.content) > 500 and b'ExceptionReport' not in r.content[:400]:
                        with open(dest, 'wb') as f:
                            f.write(r.content)
                        break
                except Exception:
                    time.sleep(0.5)
        if not os.path.exists(dest):
            return col, txy, None
        idxs = np.array(tile_to_idx[txy], dtype=int)
        vals = _sample_geotiff(dest, xs[idxs], ys[idxs], vmin=0.0, vmax=200.0)
        return col, txy, vals

    tasks = [(col, cov_id, txy) for col, cov_id in covs.items() for txy in tiles]
    with ThreadPoolExecutor(max_workers=6) as ex:
        for col, txy, vals in ex.map(_fetch_one, tasks):
            if vals is not None:
                out[col][tile_to_idx[txy]] = vals

    print(
        f"  [IRCEL 2025 10m] Done in {time.perf_counter()-t0:.1f}s | "
        f"NO2 cov={np.isfinite(out['fl_no2_2025_ugm3']).mean()*100:.1f}% (med {np.nanmedian(out['fl_no2_2025_ugm3']):.1f}) | "
        f"BC cov={np.isfinite(out['fl_bc_2025_ugm3']).mean()*100:.1f}% | "
        f"PM2.5 cov={np.isfinite(out['fl_pm25_2025_ugm3']).mean()*100:.1f}%",
        flush=True,
    )
    np.savez_compressed(cache_p, **out)
    return out


# ── 3. Mercator Buurtgroen (10m) & Urban Heat ─────────────────────────────────
def sample_mercator_green_heat(xs: np.ndarray, ys: np.ndarray) -> dict[str, np.ndarray]:
    res = {}
    layers = [
        ('fl_buurtgroen_10m', 'hh__hh_buurtgroen_2019', 0.0, 100.0),
        ('fl_urban_heat_c',   'hh__hh_ges_hitte_vlabru_2018', -10.0, 60.0),
    ]
    for col, cov_id, vmin, vmax in layers:
        dest = os.path.join(W2_DIR, f'{col}.tif')
        url = (
            'https://www.mercator.vlaanderen.be/raadpleegdienstenmercatorpubliek/wcs'
            f'?service=WCS&version=2.0.1&request=GetCoverage&coverageId={cov_id}'
            '&subset=X(90000,122000)&subset=Y(180000,212000)&format=image/tiff'
        )
        try:
            _dl(col, url, dest, timeout=90)
            vals = _sample_geotiff(dest, xs, ys, vmin=vmin, vmax=vmax)
            print(f"    {col}: cov={np.isfinite(vals).mean()*100:.1f}%, med={np.nanmedian(vals):.2f}", flush=True)
            res[col] = vals
        except Exception as e:
            print(f"    [WARN] {col} failed: {e}", flush=True)
            res[col] = np.full(len(xs), np.nan, dtype=np.float32)
    return res


# ── 4. Vector Layers: VHA Water, Wegenregister Ramps, Tram, Trees, Autovrij ───
def compute_vector_wave2(pts_31370: gpd.GeoDataFrame, xs: np.ndarray, ys: np.ndarray) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    coords = np.column_stack([xs, ys])

    # 4A. VHA Navigable Waterways (CATC=0: Leie, Schelde, Coupure, canals!)
    try:
        vha_p = os.path.join(W2_DIR, 'vha_navigable_ghent.geojson')
        vha_url = (
            'https://geo.api.vlaanderen.be/VHAWaterlopen/wfs?service=WFS&version=2.0.0'
            '&request=GetFeature&typeNames=VHAWaterlopen:Wlas&outputFormat=application/json'
            "&srsName=EPSG:31370&CQL_FILTER=BBOX(SHAPE,90000,180000,122000,212000,'EPSG:31370')%20AND%20CATC=0"
        )
        _dl('VHA Navigable Water', vha_url, vha_p)
        vha = gpd.read_file(vha_p).to_crs(31370)
        jn = gpd.sjoin_nearest(pts_31370[['geometry']], vha[['geometry']], how='left', distance_col='d')
        d_water = jn.groupby(level=0)['d'].min().reindex(range(len(xs))).to_numpy(dtype=np.float32)
        out['fl_dist_navigable_water_m'] = d_water
        out['fl_near_water_100m'] = (d_water <= 100.0).astype(np.int8)
        out['fl_near_water_300m'] = (d_water <= 300.0).astype(np.int8)
        print(f"    VHA Water: {len(vha)} segments | med dist={np.nanmedian(d_water):.0f}m | <=100m: {(d_water<=100).sum()}", flush=True)
    except Exception as e:
        print(f"    [WARN] VHA Water failed: {e}", flush=True)

    # 4B. Wegenregister Motorways & Highway Ramps (E17/E40/R4)
    try:
        weg_p = os.path.join(W2_DIR, 'wegen_motorway_ramps.geojson')
        weg_url = (
            'https://geo.api.vlaanderen.be/Wegenregister/wfs?service=WFS&version=2.0.0'
            '&request=GetFeature&typeNames=Wegenregister:Wegsegment&outputFormat=application/json'
            "&srsName=EPSG:31370&CQL_FILTER=BBOX(middellijnGeometrie,90000,180000,122000,212000,'EPSG:31370')"
            "%20AND%20(morfologischeWegklasse='autosnelweg'%20OR%20morfologischeWegklasse%20LIKE%20'%25op-%20of%20afrit%25')"
        )
        _dl('Wegenregister Motorway/Ramps', weg_url, weg_p)
        weg = gpd.read_file(weg_p).to_crs(31370)
        cls_col = [c for c in weg.columns if 'morf' in c.lower()][0]
        mw = weg[weg[cls_col].astype(str).str.lower() == 'autosnelweg']
        rmp = weg[weg[cls_col].astype(str).str.lower().str.contains('afrit', na=False)]
        if len(mw):
            jn = gpd.sjoin_nearest(pts_31370[['geometry']], mw[['geometry']], how='left', distance_col='d')
            out['fl_dist_motorway_m'] = jn.groupby(level=0)['d'].min().reindex(range(len(xs))).to_numpy(dtype=np.float32)
        if len(rmp):
            jn = gpd.sjoin_nearest(pts_31370[['geometry']], rmp[['geometry']], how='left', distance_col='d')
            out['fl_dist_highway_ramp_m'] = jn.groupby(level=0)['d'].min().reindex(range(len(xs))).to_numpy(dtype=np.float32)
        print(f"    Wegenregister: {len(mw)} motorway, {len(rmp)} ramp segs | med ramp={np.nanmedian(out.get('fl_dist_highway_ramp_m', [np.nan])):.0f}m", flush=True)
    except Exception as e:
        print(f"    [WARN] Wegenregister failed: {e}", flush=True)

    # 4C. Tram Stops (already cached in gent_bushaltes-gent.geojson!) & Rail Stations
    try:
        bus_p = os.path.join(OUT_DIR, 'gent_bushaltes-gent.geojson')
        bus = gpd.read_file(bus_p).to_crs(31370)
        bed_col = [c for c in bus.columns if 'bedien' in c.lower() or 'type' in c.lower()]
        tram = bus[bus[bed_col[0]].astype(str).str.upper().str.contains('TRAM', na=False)] if bed_col else bus.iloc[0:0]
        if len(tram):
            tc = np.column_stack([tram.geometry.x, tram.geometry.y])
            tree = cKDTree(tc)
            d_tr, _ = tree.query(coords, k=1)
            cnt_400 = np.array([len(tree.query_ball_point(pt, 400.0)) for pt in coords], dtype=np.int16)
            out['gent_dist_tram_m'] = d_tr.astype(np.float32)
            out['gent_tram_stops_400m'] = cnt_400
            print(f"    Tram stops: {len(tram)} | med dist={np.median(d_tr):.0f}m | within 400m: {(cnt_400>0).mean()*100:.1f}%", flush=True)
    except Exception as e:
        print(f"    [WARN] Tram stops failed: {e}", flush=True)

    try:
        st_p = os.path.join(W2_DIR, 'gent_rail_stations.geojson')
        _dl('Gent Rail Stations', 'https://data.stad.gent/api/explore/v2.1/catalog/datasets/locatie-stations/exports/geojson', st_p)
        st = gpd.read_file(st_p).to_crs(31370)
        sc = np.column_stack([st.geometry.centroid.x, st.geometry.centroid.y])
        d_st, _ = cKDTree(sc).query(coords, k=1)
        out['gent_dist_rail_station_m'] = d_st.astype(np.float32)
        print(f"    Rail stations: {len(st)} | med dist={np.median(d_st):.0f}m", flush=True)
    except Exception as e:
        print(f"    [WARN] Rail stations failed: {e}", flush=True)

    # 4D. Stad Gent Street Trees (68k trees with stamomtrek!)
    try:
        tr_p = os.path.join(W2_DIR, 'gent_bomen.csv')
        _dl('Gent Street Trees CSV', 'https://data.stad.gent/api/explore/v2.1/catalog/datasets/locaties-bomen-gent/exports/csv?delimiter=%3B', tr_p, timeout=180)
        bdf = pd.read_csv(tr_p, sep=';', low_memory=False)
        gp_col = [c for c in bdf.columns if 'geo' in c.lower() and 'point' in c.lower()]
        if gp_col:
            ll = bdf[gp_col[0]].astype(str).str.split(',', expand=True)
            blat = pd.to_numeric(ll[0], errors='coerce')
            blon = pd.to_numeric(ll[1], errors='coerce')
            ok = blat.notna() & blon.notna()
            bgdf = gpd.GeoDataFrame(
                bdf.loc[ok].reset_index(drop=True),
                geometry=gpd.points_from_xy(blon[ok], blat[ok]),
                crs=4326,
            ).to_crs(31370)
            bcoords = np.column_stack([bgdf.geometry.x, bgdf.geometry.y])
            girth_col = [c for c in bgdf.columns if 'stamomtrek' in c.lower()]
            girth = pd.to_numeric(bgdf[girth_col[0]], errors='coerce').fillna(40.0).to_numpy(dtype=np.float32) if girth_col else np.full(len(bgdf), 40.0, dtype=np.float32)
            btree = cKDTree(bcoords)
            cnt100 = np.zeros(len(coords), dtype=np.int16)
            mat100 = np.zeros(len(coords), dtype=np.int16)
            gsum100 = np.zeros(len(coords), dtype=np.float32)
            nbrs = btree.query_ball_point(coords, r=100.0)
            for i, idxs in enumerate(nbrs):
                if idxs:
                    g = girth[idxs]
                    cnt100[i] = len(idxs)
                    mat100[i] = int((g >= 100.0).sum())
                    gsum100[i] = float(g.sum())
            out['gent_trees_100m'] = cnt100
            out['gent_mature_trees_100m'] = mat100
            out['gent_tree_girth_sum_100m'] = gsum100
            print(f"    Street trees: {len(bgdf):,} | med trees/100m={np.median(cnt100):.0f} | mature(>=1m girth)={np.median(mat100):.0f}", flush=True)
    except Exception as e:
        print(f"    [WARN] Street trees failed: {e}", flush=True)

    # 4E. Circulatieplan (Autovrij / Pedestrian Core) + Cambio + EV Chargers
    try:
        cp_p = os.path.join(W2_DIR, 'gent_circulatieplan.geojson')
        _dl('Gent Circulatieplan', 'https://data.stad.gent/api/explore/v2.1/catalog/datasets/sectoren-circulatieplan-gent/exports/geojson', cp_p)
        cp = gpd.read_file(cp_p).to_crs(31370)
        jn = gpd.sjoin(pts_31370[['geometry']], cp[['zone', 'autovrijgebied', 'geometry']], how='left', predicate='within')
        sec = jn.groupby(level=0)['zone'].first().reindex(range(len(xs))).fillna('Outside').astype(str)
        av = jn.groupby(level=0)['autovrijgebied'].first().reindex(range(len(xs)))
        out['gent_circulatie_sector'] = sec.to_numpy()
        out['gent_in_autovrij'] = (
            av.notna() | sec.str.lower().str.contains('autovrij|voetganger', na=False)
        ).astype(np.int8).to_numpy()
        print(f"    Circulatieplan: {len(cp)} polys | autovrij={(out['gent_in_autovrij']==1).sum()} | sectors={sec.nunique()}", flush=True)
    except Exception as e:
        print(f"    [WARN] Circulatieplan failed: {e}", flush=True)

    for lbl, slug, r_m, out_col in [
        ('Cambio Carshare', 'cambio-standplaatsen', 500.0, 'gent_cambio_500m'),
        ('EV Chargers',     'laadpalen-gent',       400.0, 'gent_ev_chargers_400m'),
    ]:
        try:
            dest = os.path.join(W2_DIR, f'{slug}.geojson')
            _dl(lbl, f'https://data.stad.gent/api/explore/v2.1/catalog/datasets/{slug}/exports/geojson', dest)
            gdf = gpd.read_file(dest).to_crs(31370)
            gc = np.column_stack([gdf.geometry.centroid.x, gdf.geometry.centroid.y])
            tree = cKDTree(gc)
            cnt = np.array([len(x) for x in tree.query_ball_point(coords, r=r_m)], dtype=np.int16)
            out[out_col] = cnt
            print(f"    {lbl}: {len(gdf)} pts | med within {int(r_m)}m = {np.median(cnt):.0f}", flush=True)
        except Exception as e:
            print(f"    [WARN] {lbl} failed: {e}", flush=True)

    return out


def main():
    t0 = time.perf_counter()
    enr = pd.read_parquet(ENR_PATH)
    print(f"=== WAVE-2 FLANDERS/GENT ENRICHMENT ({len(enr)} unique coords) ===", flush=True)

    pts = gpd.GeoDataFrame(
        enr[['lat_r5', 'lon_r5']].copy(),
        geometry=[Point(lon, lat) for lat, lon in zip(enr['lat_r5'], enr['lon_r5'])],
        crs=4326,
    ).to_crs(31370)
    xs = pts.geometry.x.to_numpy(dtype=np.float64)
    ys = pts.geometry.y.to_numpy(dtype=np.float64)

    new_cols: dict[str, np.ndarray] = {}
    fluv, pluv = sample_watertoets_flood(xs, ys)
    new_cols['fl_flood_fluvial_class'] = fluv
    new_cols['fl_flood_pluvial_class'] = pluv
    new_cols['fl_flood_max_class'] = np.maximum(fluv, pluv)

    new_cols.update(sample_ircel_air_quality(xs, ys))
    new_cols.update(sample_mercator_green_heat(xs, ys))
    new_cols.update(compute_vector_wave2(pts, xs, ys))

    w2_df = pd.DataFrame({'lat_r5': enr['lat_r5'], 'lon_r5': enr['lon_r5'], **new_cols})
    w2_df.to_parquet(W2_OUT_PATH, index=False)

    # Also merge directly into ghent_coord_enrichment.parquet (dropping any prior w2 cols to stay idempotent)
    keep_base = [c for c in enr.columns if c not in new_cols]
    merged = enr[keep_base].merge(w2_df, on=['lat_r5', 'lon_r5'], how='left')
    merged.to_parquet(ENR_PATH, index=False)
    print(f"\n=== WAVE-2 COMPLETE in {time.perf_counter()-t0:.1f}s ===")
    print(f"  Added {len(new_cols)} new columns -> {W2_OUT_PATH} & updated {ENR_PATH} ({merged.shape})")
    print(f"  New cols: {list(new_cols.keys())}")


if __name__ == '__main__':
    main()
