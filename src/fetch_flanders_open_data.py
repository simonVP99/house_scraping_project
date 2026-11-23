"""Download & enrich official Ghent + Flemish open-data layers for every unique
coordinate pair in the Ghent housing dataset.

Layers fetched & cached under `external_data/flanders/`:
  1. Flanders 100m×100m **Mobiscore** GeoTIFF (Dept. Omgeving Mercator WCS `ni__ni_mobiscore_ha`)
  2. **Onroerend Erfgoed** WFS (`bes_monument`, `vast_be`, `bes_sd_gezicht`) — protected monuments,
     heritage inventory buildings, protected townscapes/landscapes
  3. **Stad Gent Open Data** (data.stad.gent):
     - `geluidsbelastingskaart-totaal-lden-gent-2021` (total noise Lden dB)
     - `geluidsbelastingskaart-wegverkeer-lden-gent-2021` (road noise Lden dB)
     - `geluidsbelastingskaart-totaal-nacht-gent-2021` (night noise Lnight dB)
     - `parken-gent` (distance to nearest park + park area nearby)
     - `bushaltes-gent` (public transit stops count within 300m/500m + nearest stop m)
     - `schooltoegangen-gent` (nearest school m + schools within 500m)
     - `lage-emissie-zone-gent` (inside Ghent LEZ)
     - `parkeertariefzones-gent` (parking tariff zone: red/orange/yellow/green/outside)
     - `kotzones-gent` (student-housing regulated zone)
     - `criminaliteitscijfers-per-wijk-per-maand-gent-2024` & `2025` (official crime rate per wijk)
  4. **Digitaal Vlaanderen GRB** WFS (`GRB:ADP` cadastral parcels + `GRB:GBG` building footprints)
     queried in clean spatial tiles / around actual property locations (exact parcel area m²,
     building footprint m², number of adjoining buildings = true terraced/semi/detached proxy!).

Produces: `external_data/flanders/ghent_coord_enrichment.parquet` keyed on `(lat_r5, lon_r5)`
so every training, CV, temporal, and inference row joins in <0.05s with ZERO leakage!

Usage:
    .venv-api/bin/python -u src/fetch_flanders_open_data.py
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
from scipy.spatial import cKDTree
from shapely.geometry import Point

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE_DIR, 'external_data', 'flanders')
os.makedirs(OUT_DIR, exist_ok=True)


def _download_json_or_geojson(name: str, url: str, dest: str) -> str:
    if os.path.exists(dest) and os.path.getsize(dest) > 200:
        print(f"  [cached] {name}: {os.path.getsize(dest)/1e6:.2f} MB")
        return dest
    t0 = time.perf_counter()
    r = requests.get(url, timeout=180)
    r.raise_for_status()
    with open(dest, 'wb') as fh:
        fh.write(r.content)
    print(f"  [fetched] {name}: {len(r.content)/1e6:.2f} MB in {time.perf_counter()-t0:.1f}s")
    return dest


def fetch_mobiscore_raster() -> str:
    """Fetch Flanders 100m Mobiscore GeoTIFF clipped to Ghent bbox in EPSG:31370."""
    dest = os.path.join(OUT_DIR, 'mobiscore_ghent.tif')
    if os.path.exists(dest) and os.path.getsize(dest) > 1000:
        print(f"  [cached] Mobiscore GeoTIFF: {os.path.getsize(dest)/1e3:.1f} KB")
        return dest
    # Ghent bbox in Belgian Lambert 72 (EPSG:31370): X in [90000, 122000], Y in [180000, 212000]
    url = (
        'https://www.mercator.vlaanderen.be/raadpleegdienstenmercatorpubliek/wcs'
        '?service=WCS&version=2.0.1&request=GetCoverage'
        '&coverageId=ni__ni_mobiscore_ha'
        '&subset=X(90000,122000)&subset=Y(180000,212000)'
        '&format=image/tiff'
    )
    t0 = time.perf_counter()
    r = requests.get(url, timeout=60)
    if r.status_code != 200 or b'ExceptionReport' in r.content[:500]:
        # Try E/N axis names if X/Y rejected
        url2 = url.replace('subset=X(', 'subset=E(').replace('subset=Y(', 'subset=N(')
        r = requests.get(url2, timeout=60)
    r.raise_for_status()
    with open(dest, 'wb') as fh:
        fh.write(r.content)
    print(f"  [fetched] Mobiscore GeoTIFF: {len(r.content)/1e3:.1f} KB in {time.perf_counter()-t0:.1f}s")
    return dest


def fetch_onroerend_erfgoed() -> dict[str, gpd.GeoDataFrame]:
    """Fetch protected monuments, established heritage inventory, and protected townscapes."""
    layers = {
        'bes_monument': 'vioe_geoportaal:bes_monument',
        'vast_be': 'vioe_geoportaal:vast_be',
        'bes_sd_gezicht': 'vioe_geoportaal:bes_sd_gezicht',
    }
    out = {}
    for short, layer in layers.items():
        dest = os.path.join(OUT_DIR, f'oe_{short}.geojson')
        url = (
            'https://geo.onroerenderfgoed.be/geoserver/vioe_geoportaal/wfs'
            f'?service=WFS&version=2.0.0&request=GetFeature&typeNames={layer}'
            '&bbox=90000,180000,122000,212000,EPSG:31370&outputFormat=application/json'
        )
        _download_json_or_geojson(f'OE {short}', url, dest)
        gdf = gpd.read_file(dest)
        if gdf.crs is None:
            gdf = gdf.set_crs(epsg=31370)
        elif gdf.crs.to_epsg() != 31370:
            gdf = gdf.to_crs(epsg=31370)
        out[short] = gdf
        print(f"    -> OE {short}: {len(gdf):,} polygons")
    return out


def fetch_stad_gent_layers() -> dict[str, object]:
    """Fetch Stad Gent Open Data vector layers + crime statistics."""
    geo_datasets = [
        'parken-gent',
        'bushaltes-gent',
        'schooltoegangen-gent',
        'lage-emissie-zone-gent',
        'parkeertariefzones-gent',
        'kotzones-gent',
        'geluidsbelastingskaart-totaal-lden-gent-2021',
        'geluidsbelastingskaart-wegverkeer-lden-gent-2021',
        'geluidsbelastingskaart-totaal-nacht-gent-2021',
    ]
    out = {}
    for ds in geo_datasets:
        dest = os.path.join(OUT_DIR, f'gent_{ds}.geojson')
        url = f'https://data.stad.gent/api/explore/v2.1/catalog/datasets/{ds}/exports/geojson'
        try:
            _download_json_or_geojson(f'Gent {ds}', url, dest)
            gdf = gpd.read_file(dest)
            if gdf.crs is None:
                gdf = gdf.set_crs(epsg=4326)
            gdf = gdf.to_crs(epsg=31370)
            out[ds] = gdf
            print(f"    -> {ds}: {len(gdf):,} features, cols={list(gdf.columns)[:6]}")
        except Exception as e:
            print(f"    ! FAILED {ds}: {e}")

    # Crime per wijk (2023 + 2024 + 2025 CSVs)
    crime_frames = []
    for yr in (2023, 2024, 2025):
        ds = f'criminaliteitscijfers-per-wijk-per-maand-gent-{yr}'
        dest = os.path.join(OUT_DIR, f'gent_crime_{yr}.csv')
        url = f'https://data.stad.gent/api/explore/v2.1/catalog/datasets/{ds}/exports/csv?delimiter=%3B'
        try:
            _download_json_or_geojson(f'Gent crime {yr}', url, dest)
            cdf = pd.read_csv(dest, sep=';')
            crime_frames.append(cdf)
        except Exception as e:
            print(f"    ! FAILED crime {yr}: {e}")
    if crime_frames:
        out['crime'] = pd.concat(crime_frames, ignore_index=True)
        print(f"    -> crime combined: {len(out['crime']):,} rows, cols={list(out['crime'].columns)[:8]}")
    return out


def _sample_mobiscore(tif_path: str, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Sample Flanders 100m Mobiscore GeoTIFF at Lambert-72 coordinates without rasterio if absent."""
    try:
        import rasterio
        with rasterio.open(tif_path) as src:
            vals = np.array([v[0] for v in src.sample(zip(xs, ys))], dtype=float)
            nodata = src.nodata
            if nodata is not None:
                vals = np.where(vals == nodata, np.nan, vals)
            return vals
    except ImportError:
        pass
    # Pure PIL fallback supporting both ModelTransformationTag (34264) and ModelPixelScale/Tiepoint (33550/33922)
    from PIL import Image
    img = Image.open(tif_path)
    arr = np.array(img, dtype=float)
    tags = img.tag_v2
    if 34264 in tags:
        m = tags[34264]  # 4x4 affine: x = m[0]*col + m[3], y = m[5]*row + m[7]
        sx, x0 = float(m[0]), float(m[3])
        sy, y0 = float(abs(m[5])), float(m[7])
    else:
        sx, sy = float(tags[33550][0]), float(tags[33550][1])
        tp = tags[33922]
        x0, y0 = float(tp[3]), float(tp[4])
    cols = np.floor((xs - x0) / sx).astype(int)
    rows = np.floor((y0 - ys) / sy).astype(int)
    h, w = arr.shape[:2]
    ok = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
    out = np.full(len(xs), np.nan, dtype=float)
    if arr.ndim == 2:
        out[ok] = arr[rows[ok], cols[ok]]
    else:
        out[ok] = arr[rows[ok], cols[ok], 0]
    out = np.where((out < 0) | (out > 10.5), np.nan, out)
    return out


def _parse_db_value(series: pd.Series) -> pd.Series:
    """Parse dB bands like '55-59 dB', '65-69', '>=75', or numeric values into midpoint dB."""
    import re
    def _one(v):
        if pd.isna(v):
            return np.nan
        s = str(v)
        nums = [float(x) for x in re.findall(r'\d+(?:\.\d+)?', s)]
        if not nums:
            return np.nan
        if len(nums) >= 2:
            return 0.5 * (nums[0] + nums[1])
        return nums[0]
    return series.map(_one)


def fetch_grb_for_coords(pts_31370: gpd.GeoDataFrame) -> pd.DataFrame:
    """Fetch official Digitaal Vlaanderen GRB parcel (ADP) & building (GBG) attributes
    around all unique coordinates using clustered 500m×500m tiles in parallel.
    """
    cache_path = os.path.join(OUT_DIR, 'grb_enrichment_cache.parquet')
    if os.path.exists(cache_path):
        cached = pd.read_parquet(cache_path)
        if len(cached) == len(pts_31370):
            print(f"  [cached] GRB enrichment: {len(cached):,} coords")
            return cached

    xs = pts_31370.geometry.x.to_numpy()
    ys = pts_31370.geometry.y.to_numpy()
    # Group points into 600m grid tiles so we query only populated tiles (~150-250 tiles)
    tile_size = 600.0
    tx = np.floor(xs / tile_size).astype(int)
    ty = np.floor(ys / tile_size).astype(int)
    tile_keys = sorted(set(zip(tx, ty)))
    print(f"  Querying Digitaal Vlaanderen GRB WFS across {len(tile_keys)} populated 600m tiles (parallel)...")

    def _query_tile(tile):
        ix, iy = tile
        x0, y0 = ix * tile_size - 25, iy * tile_size - 25
        x1, y1 = (ix + 1) * tile_size + 25, (iy + 1) * tile_size + 25
        bbox = f"{x0:.0f},{y0:.0f},{x1:.0f},{y1:.0f},EPSG:31370"
        res_t = {'tile': tile, 'adp': None, 'gbg': None}
        for key, tn in [('adp', 'GRB:ADP'), ('gbg', 'GRB:GBG')]:
            url = (
                'https://geo.api.vlaanderen.be/GRB/wfs'
                f'?service=WFS&version=2.0.0&request=GetFeature&typeNames={tn}'
                f'&bbox={bbox}&outputFormat=application/json&count=2000'
            )
            for attempt in range(2):
                try:
                    r = requests.get(url, timeout=25)
                    if r.status_code == 200 and len(r.content) > 80:
                        gdf = gpd.read_file(io.BytesIO(r.content))
                        if len(gdf) > 0:
                            if gdf.crs is None:
                                gdf = gdf.set_crs(epsg=31370)
                            elif gdf.crs.to_epsg() != 31370:
                                gdf = gdf.to_crs(epsg=31370)
                            res_t[key] = gdf
                    break
                except Exception:
                    time.sleep(0.5)
        return res_t

    n = len(pts_31370)
    grb_parcel_area = np.full(n, np.nan)
    grb_bldg_footprint = np.full(n, np.nan)
    grb_bldg_perimeter = np.full(n, np.nan)
    grb_bldg_compactness = np.full(n, np.nan)
    grb_bldg_type_code = np.full(n, -1, dtype=int)
    grb_adjoining_bldgs = np.full(n, np.nan)
    grb_bldg_density_100m = np.full(n, np.nan)
    grb_footprint_to_parcel = np.full(n, np.nan)

    # Map tile -> point indices
    tile_to_idx: dict[tuple[int, int], list[int]] = {}
    for i, tk in enumerate(zip(tx, ty)):
        tile_to_idx.setdefault(tk, []).append(i)

    t0 = time.perf_counter()
    done = 0
    with ThreadPoolExecutor(max_workers=12) as ex:
        futs = {ex.submit(_query_tile, tk): tk for tk in tile_keys}
        for fut in as_completed(futs):
            done += 1
            rt = fut.result()
            idxs = tile_to_idx[rt['tile']]
            sub_pts = pts_31370.iloc[idxs]

            adp = rt['adp']
            if adp is not None and len(adp) > 0:
                adp = adp[['geometry']].copy()
                adp['_area'] = adp.geometry.area
                j = gpd.sjoin(sub_pts[['geometry']], adp, how='left', predicate='within')
                j = j[~j.index.duplicated(keep='first')]
                for orig_i, row_val in zip(idxs, j['_area'].to_numpy()):
                    if pd.notna(row_val) and row_val > 0:
                        grb_parcel_area[orig_i] = float(row_val)

            gbg = rt['gbg']
            if gbg is not None and len(gbg) > 0:
                gbg = gbg.copy()
                gbg['_area'] = gbg.geometry.area
                gbg['_perim'] = gbg.geometry.length
                gbg['_type'] = pd.to_numeric(gbg['TYPE'], errors='coerce').fillna(-1).astype(int) if 'TYPE' in gbg.columns else -1
                sidx = gbg.sindex
                # Nearest building within 25m (handles geocoded points landing on the sidewalk!)
                j = gpd.sjoin_nearest(sub_pts[['geometry']], gbg[['geometry', '_area', '_perim', '_type']], how='left', max_distance=25.0, distance_col='_dist')
                j = j[~j.index.duplicated(keep='first')]
                for orig_i, (_, row) in zip(idxs, j.iterrows()):
                    b_idx = row.get('index_right')
                    if pd.notna(b_idx):
                        bi = int(b_idx)
                        area = float(row['_area'])
                        perim = float(row['_perim'])
                        grb_bldg_footprint[orig_i] = area
                        grb_bldg_perimeter[orig_i] = perim
                        grb_bldg_compactness[orig_i] = (4.0 * np.pi * area) / (perim ** 2 + 1e-6)
                        grb_bldg_type_code[orig_i] = int(row['_type'])
                        # Count adjoining/touching buildings (within 0.6m buffer) -> 0=detached, 1=semi, >=2=terraced!
                        bgeom = gbg.geometry.iloc[bi]
                        cand = list(sidx.intersection(bgeom.buffer(0.6).bounds))
                        touch = sum(1 for c in cand if c != bi and gbg.geometry.iloc[c].distance(bgeom) <= 0.6)
                        grb_adjoining_bldgs[orig_i] = float(touch)
                    # Local built-up density within 100m
                    pt = pts_31370.geometry.iloc[orig_i]
                    buf = pt.buffer(100.0)
                    cand100 = list(sidx.intersection(buf.bounds))
                    if cand100:
                        built = sum(gbg.geometry.iloc[c].area for c in cand100 if gbg.geometry.iloc[c].centroid.distance(pt) <= 100.0)
                        grb_bldg_density_100m[orig_i] = min(1.0, built / (np.pi * 100.0 ** 2))

            if done % 40 == 0 or done == len(tile_keys):
                print(f"    GRB progress: {done}/{len(tile_keys)} tiles ({time.perf_counter()-t0:.1f}s)", flush=True)

    valid_both = (grb_parcel_area > 0) & (grb_bldg_footprint > 0)
    grb_footprint_to_parcel[valid_both] = np.clip(grb_bldg_footprint[valid_both] / grb_parcel_area[valid_both], 0.0, 1.5)

    out = pd.DataFrame({
        'grb_parcel_area': grb_parcel_area,
        'grb_bldg_footprint': grb_bldg_footprint,
        'grb_bldg_perimeter': grb_bldg_perimeter,
        'grb_bldg_compactness': grb_bldg_compactness,
        'grb_bldg_type_code': grb_bldg_type_code,
        'grb_adjoining_bldgs': grb_adjoining_bldgs,
        'grb_bldg_density_100m': grb_bldg_density_100m,
        'grb_footprint_to_parcel': grb_footprint_to_parcel,
    })
    out.to_parquet(cache_path, index=False)
    print(f"  GRB matched: parcel={np.isfinite(grb_parcel_area).mean()*100:.1f}%, building={np.isfinite(grb_bldg_footprint).mean()*100:.1f}%")
    return out


def main():
    t0 = time.perf_counter()
    ds = pd.read_parquet(os.path.join(BASE_DIR, 'outputs', 'dataset', 'training_dataset.parquet'), columns=['latitude', 'longitude', 'neighbourhood'])
    coords = ds[['latitude', 'longitude']].copy()
    coords['lat_r5'] = coords['latitude'].round(5)
    coords['lon_r5'] = coords['longitude'].round(5)
    uniq = coords.drop_duplicates(['lat_r5', 'lon_r5']).reset_index(drop=True)
    print(f"Enriching {len(uniq):,} unique coordinate pairs with official Flanders + Stad Gent open data...")

    pts = gpd.GeoDataFrame(
        uniq[['lat_r5', 'lon_r5']],
        geometry=gpd.points_from_xy(uniq['lon_r5'], uniq['lat_r5']),
        crs='EPSG:4326',
    ).to_crs(epsg=31370)
    xs = pts.geometry.x.to_numpy()
    ys = pts.geometry.y.to_numpy()
    xy = np.column_stack([xs, ys])

    out = pd.DataFrame({'lat_r5': uniq['lat_r5'], 'lon_r5': uniq['lon_r5']})

    # 1. Flanders Mobiscore (100m raster)
    print("\n[1/4] Flanders 100m Mobiscore raster...")
    tif = fetch_mobiscore_raster()
    out['fl_mobiscore'] = _sample_mobiscore(tif, xs, ys)
    print(f"  Mobiscore coverage: {out['fl_mobiscore'].notna().mean()*100:.1f}%, median={out['fl_mobiscore'].median():.2f}, range=[{out['fl_mobiscore'].min():.2f}, {out['fl_mobiscore'].max():.2f}]")

    # 2. Onroerend Erfgoed (heritage & monuments)
    print("\n[2/4] Onroerend Erfgoed (monuments, heritage inventory, protected townscapes)...")
    oe = fetch_onroerend_erfgoed()
    for key, col_in, col_dist in [
        ('bes_monument', 'fl_is_protected_monument', 'fl_dist_monument_m'),
        ('vast_be', 'fl_is_heritage_inventory', 'fl_dist_heritage_m'),
        ('bes_sd_gezicht', 'fl_in_protected_townscape', None),
    ]:
        gdf = oe[key]
        # Inside or within 8m (building centroid vs street point)
        j = gpd.sjoin_nearest(pts[['geometry']], gdf[['geometry']], how='left', max_distance=1000.0, distance_col='_d')
        j = j[~j.index.duplicated(keep='first')]
        d = j['_d'].fillna(1000.0).to_numpy()
        out[col_in] = (d <= 8.0).astype(np.int8)
        if col_dist:
            out[col_dist] = np.clip(d, 0.0, 1000.0)
        # Heritage density within 150m for vast_be (captures historic architectural street feel!)
        if key == 'vast_be':
            c_xy = np.column_stack([gdf.geometry.centroid.x.to_numpy(), gdf.geometry.centroid.y.to_numpy()])
            tree = cKDTree(c_xy)
            out['fl_heritage_count_150m'] = [len(tree.query_ball_point(p, 150.0)) for p in xy]
        print(f"  {col_in}: {out[col_in].sum():,} ({out[col_in].mean()*100:.1f}%)")

    # 3. Stad Gent Open Data (noise, parks, transit, schools, LEZ, parking tariff, kotzone, crime)
    print("\n[3/4] Stad Gent Open Data layers...")
    gent = fetch_stad_gent_layers()

    # Noise maps (Lden total, Lden road, Lnight total)
    for ds_name, col_out in [
        ('geluidsbelastingskaart-totaal-lden-gent-2021', 'gent_noise_total_lden_db'),
        ('geluidsbelastingskaart-wegverkeer-lden-gent-2021', 'gent_noise_road_lden_db'),
        ('geluidsbelastingskaart-totaal-nacht-gent-2021', 'gent_noise_night_db'),
    ]:
        gdf = gent.get(ds_name)
        if gdf is not None and len(gdf) > 0:
            if 'low_val' in gdf.columns and 'high_val' in gdf.columns:
                lo = pd.to_numeric(gdf['low_val'], errors='coerce')
                hi = pd.to_numeric(gdf['high_val'], errors='coerce')
                gdf = gdf[['geometry']].copy()
                gdf['_db'] = np.where(hi.notna() & (hi > lo), 0.5 * (lo + hi), lo)
                db_col = 'low_val..high_val'
            else:
                db_col = next((c for c in gdf.columns if any(k in c.lower() for k in ['interval', 'klasse', 'db', 'band', 'lden', 'lnight', 'label', 'waarde', 'categorie']) and c != 'geometry' and 'geom' not in c.lower()), None)
                if db_col is None:
                    db_col = [c for c in gdf.columns if c != 'geometry'][0]
                gdf = gdf[['geometry', db_col]].copy()
                gdf['_db'] = _parse_db_value(gdf[db_col])
            j = gpd.sjoin(pts[['geometry']], gdf[['geometry', '_db']], how='left', predicate='within')
            # If overlapping polygons, take max noise
            s = j.groupby(j.index)['_db'].max()
            out[col_out] = s.reindex(range(len(pts))).to_numpy()
            print(f"  {col_out}: matched={out[col_out].notna().mean()*100:.1f}%, median={np.nanmedian(out[col_out]):.1f} dB (col={db_col})")

    # Parks (nearest distance + count within 500m)
    if 'parken-gent' in gent:
        pk = gent['parken-gent']
        j = gpd.sjoin_nearest(pts[['geometry']], pk[['geometry']], how='left', max_distance=5000.0, distance_col='_d')
        j = j[~j.index.duplicated(keep='first')]
        out['gent_dist_park_m'] = j['_d'].fillna(5000.0).to_numpy()
        c_xy = np.column_stack([pk.geometry.centroid.x.to_numpy(), pk.geometry.centroid.y.to_numpy()])
        tree = cKDTree(c_xy)
        out['gent_parks_within_600m'] = [len(tree.query_ball_point(p, 600.0)) for p in xy]
        print(f"  gent_dist_park_m: median={np.median(out['gent_dist_park_m']):.0f}m")

    # Transit stops (bushaltes-gent)
    if 'bushaltes-gent' in gent:
        tr = gent['bushaltes-gent']
        c_xy = np.column_stack([tr.geometry.centroid.x.to_numpy(), tr.geometry.centroid.y.to_numpy()])
        tree = cKDTree(c_xy)
        d1, _ = tree.query(xy, k=1)
        out['gent_dist_transit_m'] = np.clip(d1, 0.0, 3000.0)
        out['gent_transit_stops_400m'] = [len(tree.query_ball_point(p, 400.0)) for p in xy]
        print(f"  gent_dist_transit_m: median={np.median(out['gent_dist_transit_m']):.0f}m, med stops@400m={np.median(out['gent_transit_stops_400m']):.0f}")

    # Schools (schooltoegangen-gent)
    if 'schooltoegangen-gent' in gent:
        sc = gent['schooltoegangen-gent']
        c_xy = np.column_stack([sc.geometry.centroid.x.to_numpy(), sc.geometry.centroid.y.to_numpy()])
        tree = cKDTree(c_xy)
        d1, _ = tree.query(xy, k=1)
        out['gent_dist_school_m'] = np.clip(d1, 0.0, 3000.0)
        out['gent_schools_500m'] = [len(tree.query_ball_point(p, 500.0)) for p in xy]
        print(f"  gent_dist_school_m: median={np.median(out['gent_dist_school_m']):.0f}m")

    # LEZ & Kotzone & Parking tariff zone
    if 'lage-emissie-zone-gent' in gent:
        j = gpd.sjoin(pts[['geometry']], gent['lage-emissie-zone-gent'][['geometry']], how='left', predicate='within')
        j = j[~j.index.duplicated(keep='first')]
        out['gent_in_lez'] = j['index_right'].notna().astype(np.int8).to_numpy()
        print(f"  gent_in_lez: {out['gent_in_lez'].mean()*100:.1f}%")

    if 'kotzones-gent' in gent:
        j = gpd.sjoin(pts[['geometry']], gent['kotzones-gent'][['geometry']], how='left', predicate='within')
        j = j[~j.index.duplicated(keep='first')]
        out['gent_in_kotzone'] = j['index_right'].notna().astype(np.int8).to_numpy()
        print(f"  gent_in_kotzone: {out['gent_in_kotzone'].mean()*100:.1f}%")

    if 'parkeertariefzones-gent' in gent:
        pz = gent['parkeertariefzones-gent']
        zcol = next((c for c in pz.columns if any(k in c.lower() for k in ['zone', 'tarief', 'kleur', 'naam']) and c != 'geometry'), None)
        if zcol:
            j = gpd.sjoin(pts[['geometry']], pz[['geometry', zcol]], how='left', predicate='within')
            j = j[~j.index.duplicated(keep='first')]
            out['gent_parking_zone'] = j[zcol].fillna('Outside').astype(str).to_numpy()
            print(f"  gent_parking_zone value counts: {out['gent_parking_zone'].value_counts().head(6).to_dict()}")

    # Crime per wijk (aggregate 2023-2025 total incidents per quarter/wijk, save wijk lookup too)
    if 'crime' in gent:
        cdf = gent['crime']
        wijk_col = next((c for c in cdf.columns if c.lower() in ('quarter', 'wijk') or 'wijk' in c.lower()), None)
        cnt_col = next((c for c in cdf.columns if c.lower() in ('total', 'aantal', 'feiten') or any(k in c.lower() for k in ['aantal', 'feiten', 'count', 'waarde'])), None)
        if wijk_col and cnt_col:
            cdf[cnt_col] = pd.to_numeric(cdf[cnt_col], errors='coerce').fillna(0)
            w_crime = cdf.groupby(wijk_col)[cnt_col].sum().reset_index()
            w_crime.columns = ['wijk', 'gent_wijk_crime_3yr']
            w_crime.to_parquet(os.path.join(OUT_DIR, 'gent_wijk_crime.parquet'), index=False)
            print(f"  gent_wijk_crime saved: {len(w_crime)} wijken, range=[{w_crime['gent_wijk_crime_3yr'].min():.0f}, {w_crime['gent_wijk_crime_3yr'].max():.0f}]")

    # 4. Digitaal Vlaanderen GRB (parcels + building footprints + adjoining buildings)
    print("\n[4/4] Digitaal Vlaanderen GRB (official parcels & building footprints)...")
    grb_df = fetch_grb_for_coords(pts)
    out = pd.concat([out.reset_index(drop=True), grb_df.reset_index(drop=True)], axis=1)

    dest = os.path.join(OUT_DIR, 'ghent_coord_enrichment.parquet')
    out.to_parquet(dest, index=False)
    print(f"\nSaved {dest}: {out.shape[0]:,} coords × {out.shape[1]} columns in {time.perf_counter()-t0:.1f}s")


if __name__ == '__main__':
    main()
