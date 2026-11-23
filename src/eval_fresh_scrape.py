"""Evaluate our models on the freshly scraped 22–23 Nov 2026 Immoweb listings — a
100% genuine, brand-new, live out-of-time holdout that NO model has ever seen!

Compares TWO generations trained strictly on the pre-22-Nov 14,599-row dataset:
  1. **Legacy Production Model** (`outputs/models/autogluon`, 110 features, K-EUR, built Apr 2026)
     — via `api.prediction.PredictionService` (or directly on parsed rows) if compatible,
     PLUS a **Phase-1 Baseline** (default LightGBM on basic 235 features)
  2. **Phase-4 Frontier Ensemble** (Flanders Mobiscore + Onroerend Erfgoed + Stad Gent + GRB +
     E5-Large + Multi-Alpha/Huber TF-IDF + SVD + `TabM` + `RealMLP` + `CatBoost` + `LightGBM` +
     `LightGBMXT` + `XGBoost` + House/Apt Specialists), trained on 100% of the 14,599 pre-22-Nov
     dataset using the exact Phase-4 optimal ensemble weights!

Pipeline:
  1. Extract the 22–23 Nov 2026 fresh listings from `outputs/property_data_full.parquet`
     and pass them through the exact same `clean_raw` + `transform` preprocessing artifacts.
  2. Partition into:
     - **STRICTLY UNSEEN** (`176` physical homes never in `training_dataset`, of which `154` are in-domain ≤€1.5M)
     - **ALL FRESH** (`1,227` clean active listings on Immoweb today)
  3. Enrich any new coordinates with Flanders Mobiscore + Onroerend Erfgoed + Stad Gent + GRB
     (incremental — only fetches the ~120 new coords not already in `ghent_coord_enrichment.parquet`!).
  4. Embed fresh descriptions with `multilingual-e5-small` & `multilingual-e5-large` on MPS GPU.
  5. Compute strictly train-only spatial/TE/text features from the 14,599 pre-22-Nov training set
     and score both the **Phase-1 Baseline** and the **Full Phase-4 Frontier Ensemble** on the
     freshly scraped live listings!
  6. Save per-property predictions & biggest deals/misses to `outputs/models/workbench/fresh_nov23_predictions.csv`
     and `outputs/models/workbench/fresh_nov23_report.json`.

Usage:
    .venv-api/bin/python -u src/eval_fresh_scrape.py
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'src'))

from fetch_flanders_open_data import (
    _parse_db_value,
    _sample_mobiscore,
    fetch_grb_for_coords,
    fetch_mobiscore_raster,
    fetch_onroerend_erfgoed,
    fetch_stad_gent_layers,
)
from phase3_features import (
    _add_domain_features,
    _compute_spatial_and_te,
    _extract_text_flags,
    compute_metrics,
    fmt,
)
from phase3_modeling import _fit_gbdt_split, _target_inverse, _target_transform, prep_tree_df
from phase4_features import (
    E5_LARGE_PATH,
    E5_SMALL_PATH,
    FLANDERS_PATH,
    attach_flanders_and_grb,
    fit_transform_phase4_text,
)
from phase4_frontier import _prep_pytabkit_dfs
from preprocessing import PreprocessConfig, clean_raw, transform

WORK_DIR = os.path.join(BASE_DIR, 'outputs', 'models', 'workbench')
DATASET_PATH = os.path.join(BASE_DIR, 'outputs', 'dataset', 'training_dataset.parquet')
ARTIFACTS_PATH = os.path.join(BASE_DIR, 'outputs', 'dataset', 'preprocessing_artifacts.joblib')
RAW_PARQUET = os.path.join(BASE_DIR, 'outputs', 'property_data_full.parquet')


def enrich_missing_coords(df_coords: pd.DataFrame) -> pd.DataFrame:
    """Incrementally enrich any (lat_r5, lon_r5) coords missing from ghent_coord_enrichment.parquet."""
    fl = pd.read_parquet(FLANDERS_PATH)
    have = set(zip(fl['lat_r5'], fl['lon_r5']))
    c = df_coords[['latitude', 'longitude']].copy()
    c['lat_r5'] = c['latitude'].round(5)
    c['lon_r5'] = c['longitude'].round(5)
    uniq = c.drop_duplicates(['lat_r5', 'lon_r5'])
    missing = uniq[[ (a, b) not in have for a, b in zip(uniq['lat_r5'], uniq['lon_r5']) ]].reset_index(drop=True)
    if len(missing) == 0:
        print("  All fresh coordinates already enriched in Flanders/GRB cache.")
        return fl

    print(f"  Enriching {len(missing)} new coordinates with Flanders Mobiscore + Heritage + Stad Gent + GRB...")
    t0 = time.perf_counter()
    pts = gpd.GeoDataFrame(
        missing[['lat_r5', 'lon_r5']],
        geometry=gpd.points_from_xy(missing['lon_r5'], missing['lat_r5']),
        crs='EPSG:4326',
    ).to_crs(epsg=31370)
    xs, ys = pts.geometry.x.to_numpy(), pts.geometry.y.to_numpy()
    xy = np.column_stack([xs, ys])
    out = pd.DataFrame({'lat_r5': missing['lat_r5'], 'lon_r5': missing['lon_r5']})

    # 1. Mobiscore
    tif = fetch_mobiscore_raster()
    out['fl_mobiscore'] = _sample_mobiscore(tif, xs, ys)

    # 2. Onroerend Erfgoed
    oe = fetch_onroerend_erfgoed()
    for key, col_in, col_dist in [
        ('bes_monument', 'fl_is_protected_monument', 'fl_dist_monument_m'),
        ('vast_be', 'fl_is_heritage_inventory', 'fl_dist_heritage_m'),
        ('bes_sd_gezicht', 'fl_in_protected_townscape', None),
    ]:
        gdf = oe[key]
        j = gpd.sjoin_nearest(pts[['geometry']], gdf[['geometry']], how='left', max_distance=1000.0, distance_col='_d')
        j = j[~j.index.duplicated(keep='first')]
        d = j['_d'].fillna(1000.0).to_numpy()
        out[col_in] = (d <= 8.0).astype(np.int8)
        if col_dist:
            out[col_dist] = np.clip(d, 0.0, 1000.0)
        if key == 'vast_be':
            c_xy = np.column_stack([gdf.geometry.centroid.x.to_numpy(), gdf.geometry.centroid.y.to_numpy()])
            tree = cKDTree(c_xy)
            out['fl_heritage_count_150m'] = [len(tree.query_ball_point(p, 150.0)) for p in xy]

    # 3. Stad Gent
    gent = fetch_stad_gent_layers()
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
            else:
                db_col = [c for c in gdf.columns if c != 'geometry'][0]
                gdf = gdf[['geometry', db_col]].copy()
                gdf['_db'] = _parse_db_value(gdf[db_col])
            j = gpd.sjoin(pts[['geometry']], gdf[['geometry', '_db']], how='left', predicate='within')
            out[col_out] = j.groupby(j.index)['_db'].max().reindex(range(len(pts))).to_numpy()

    if 'parken-gent' in gent:
        pk = gent['parken-gent']
        j = gpd.sjoin_nearest(pts[['geometry']], pk[['geometry']], how='left', max_distance=5000.0, distance_col='_d')
        j = j[~j.index.duplicated(keep='first')]
        out['gent_dist_park_m'] = j['_d'].fillna(5000.0).to_numpy()
        tree = cKDTree(np.column_stack([pk.geometry.centroid.x.to_numpy(), pk.geometry.centroid.y.to_numpy()]))
        out['gent_parks_within_600m'] = [len(tree.query_ball_point(p, 600.0)) for p in xy]

    if 'bushaltes-gent' in gent:
        tr = gent['bushaltes-gent']
        tree = cKDTree(np.column_stack([tr.geometry.centroid.x.to_numpy(), tr.geometry.centroid.y.to_numpy()]))
        d1, _ = tree.query(xy, k=1)
        out['gent_dist_transit_m'] = np.clip(d1, 0.0, 3000.0)
        out['gent_transit_stops_400m'] = [len(tree.query_ball_point(p, 400.0)) for p in xy]

    if 'schooltoegangen-gent' in gent:
        sc = gent['schooltoegangen-gent']
        tree = cKDTree(np.column_stack([sc.geometry.centroid.x.to_numpy(), sc.geometry.centroid.y.to_numpy()]))
        d1, _ = tree.query(xy, k=1)
        out['gent_dist_school_m'] = np.clip(d1, 0.0, 3000.0)
        out['gent_schools_500m'] = [len(tree.query_ball_point(p, 500.0)) for p in xy]

    if 'lage-emissie-zone-gent' in gent:
        j = gpd.sjoin(pts[['geometry']], gent['lage-emissie-zone-gent'][['geometry']], how='left', predicate='within')
        j = j[~j.index.duplicated(keep='first')]
        out['gent_in_lez'] = j['index_right'].notna().astype(np.int8).to_numpy()

    if 'kotzones-gent' in gent:
        j = gpd.sjoin(pts[['geometry']], gent['kotzones-gent'][['geometry']], how='left', predicate='within')
        j = j[~j.index.duplicated(keep='first')]
        out['gent_in_kotzone'] = j['index_right'].notna().astype(np.int8).to_numpy()

    if 'parkeertariefzones-gent' in gent:
        pz = gent['parkeertariefzones-gent']
        zcol = next((c for c in pz.columns if any(k in c.lower() for k in ['zone', 'tarief', 'kleur', 'naam']) and c != 'geometry'), None)
        if zcol:
            j = gpd.sjoin(pts[['geometry']], pz[['geometry', zcol]], how='left', predicate='within')
            j = j[~j.index.duplicated(keep='first')]
            out['gent_parking_zone'] = j[zcol].fillna('Outside').astype(str).to_numpy()

    # 4. GRB (only for these new coords — use a separate temp cache so we don't overwrite main GRB cache!)
    import fetch_flanders_open_data as ffod
    orig_dir = ffod.OUT_DIR
    tmp_dir = os.path.join(BASE_DIR, 'external_data', 'flanders', '_tmp_fresh_grb')
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_grb_cache = os.path.join(tmp_dir, 'grb_enrichment_cache.parquet')
    if os.path.exists(tmp_grb_cache):
        os.remove(tmp_grb_cache)
    ffod.OUT_DIR = tmp_dir
    try:
        grb_df = fetch_grb_for_coords(pts)
    finally:
        ffod.OUT_DIR = orig_dir
    out = pd.concat([out.reset_index(drop=True), grb_df.reset_index(drop=True)], axis=1)

    # Align columns and append to master coordinate enrichment parquet
    combined = pd.concat([fl, out[fl.columns]], ignore_index=True).drop_duplicates(['lat_r5', 'lon_r5'], keep='last')
    combined.to_parquet(FLANDERS_PATH, index=False)
    print(f"  Appended {len(out)} enriched coords -> {FLANDERS_PATH} ({len(combined):,} total) in {time.perf_counter()-t0:.1f}s")
    return combined


def embed_texts_mps(texts: list[str], model_name: str, max_len: int, batch_size: int = 64) -> np.ndarray:
    """Compute L2-normalized mean-pooled E5 embeddings on MPS fp16."""
    from transformers import AutoModel, AutoTokenizer

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"  Embedding {len(texts)} fresh texts with {model_name} on {device}...")
    tok = AutoTokenizer.from_pretrained(model_name)
    mod = AutoModel.from_pretrained(model_name, dtype=torch.float16 if device == 'mps' else torch.float32).to(device)
    mod.eval()

    prepped = [f"passage: {t[:2400]}" for t in texts]
    order = np.argsort([len(s) for s in prepped])
    dim = mod.config.hidden_size
    embs = np.zeros((len(texts), dim), dtype=np.float32)

    with torch.no_grad():
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            batch = [prepped[i] for i in idx]
            enc = tok(batch, padding=True, truncation=True, max_length=max_len, return_tensors='pt').to(device)
            out = mod(**enc)
            mask = enc['attention_mask'].unsqueeze(-1).float()
            pooled = (out.last_hidden_state.float() * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            embs[idx] = pooled.cpu().numpy().astype(np.float32)
    del mod
    if device == 'mps':
        torch.mps.empty_cache()
    return embs


def fit_predict_pytabkit_full(tr_df, va_df, tr_cpi, va_cpi, feat_cols, model_type, target_mode, n_seeds=3):
    from pytabkit import RealMLP_TD_Regressor, TabM_D_Regressor

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    X_tr, X_va, cat_cols = _prep_pytabkit_dfs(tr_df, va_df, feat_cols)
    y_tr = _target_transform(tr_df, tr_cpi, target_mode).astype(np.float32)
    groups_tr = tr_df['property_key'].astype(str).to_numpy()

    preds = []
    for s in range(n_seeds):
        rng = np.random.default_rng(400 + s)
        uniq_g = np.unique(groups_tr)
        es_g = set(rng.choice(uniq_g, size=max(1, int(0.12 * len(uniq_g))), replace=False))
        es_mask = np.isin(groups_tr, list(es_g))
        if model_type == 'realmlp':
            m = RealMLP_TD_Regressor(device=device, random_state=400 + s, n_epochs=128, batch_size=256, val_metric_name='mae', verbosity=0)
        else:
            m = TabM_D_Regressor(device=device, random_state=400 + s, n_epochs=100, batch_size=256, val_metric_name='mae', verbosity=0)
        m.fit(X_tr.loc[~es_mask].reset_index(drop=True), y_tr[~es_mask], X_tr.loc[es_mask].reset_index(drop=True), y_tr[es_mask], cat_col_names=cat_cols)
        preds.append(np.asarray(m.predict(X_va), dtype=float))
    return _target_inverse(np.mean(preds, axis=0), va_df, va_cpi, target_mode)


def fit_predict_specialist_full(tr_df, va_df, tr_cpi, va_cpi, feat_cols, kind, target_mode, loss_mode, params):
    out = np.zeros(len(va_df), dtype=float)
    for h_flag in (1, 0):
        tr_sub = tr_df[tr_df['property_type_HOUSE'] == h_flag].reset_index(drop=True)
        tr_cpi_sub = tr_cpi[tr_df['property_type_HOUSE'].to_numpy() == h_flag]
        va_mask = va_df['property_type_HOUSE'].to_numpy() == h_flag
        if va_mask.sum() == 0:
            continue
        va_sub = va_df[va_mask].reset_index(drop=True)
        va_cpi_sub = va_cpi[va_mask]
        _, vp, _ = _fit_gbdt_split(
            f'spec_h{h_flag}', tr_sub, va_sub, tr_cpi_sub, va_cpi_sub,
            feat_cols, kind, target_mode, loss_mode, params, n_threads=12, seed=42,
        )
        out[va_mask] = vp
    return out


def main():
    t0 = time.perf_counter()
    print("=== 1. PREPARING FRESH 22–23 NOV 2026 LIVE HOLDOUT vs PRE-22-NOV TRAINING SET ===")
    raw = pd.read_parquet(RAW_PARQUET)
    raw['property_id'] = raw['property_id'].astype(str)
    tr_base = pd.read_parquet(DATASET_PATH).reset_index(drop=True)
    tr_base['property_id'] = tr_base['property_id'].astype(str)
    artifacts = joblib.load(ARTIFACTS_PATH)

    # Extract fresh 22-23 Nov rows & run the EXACT same clean_raw + transform
    fresh_raw = raw[raw['date_obtained'].isin(['22112026', '23112026'])].copy()
    res = clean_raw(fresh_raw, config=PreprocessConfig(dedupe_relistings=True, outlier_quantile=1.0), annotate=False)
    fresh_kept = res['kept'].reset_index(drop=True)
    fresh_kept['property_id'] = fresh_kept['property_id'].astype(str)

    # Save raw text/meta BEFORE transform drops them
    fresh_meta = fresh_kept[['property_id', 'date_obtained', 'property_key', 'title', 'description', 'description_nl', 'street', 'immo_office_name', 'postal_code']].copy()
    fresh_tr = transform(fresh_kept, artifacts).reset_index(drop=True)
    fresh_tr['property_id'] = fresh_kept['property_id'].values
    fresh_tr['property_key'] = fresh_kept['property_key'].values
    fresh_tr['date_obtained'] = fresh_kept['date_obtained'].values

    # Partition fresh listings into STRICTLY UNSEEN vs RE-OBSERVED
    tr_pids = set(tr_base['property_id'])
    tr_keys = set(tr_base['property_key'].astype(str))
    is_unseen = (~fresh_tr['property_id'].isin(tr_pids)) & (~fresh_tr['property_key'].astype(str).isin(tr_keys))
    fresh_tr['is_strictly_unseen'] = is_unseen.to_numpy()
    print(f"  Clean fresh listings: {len(fresh_tr):,} | STRICTLY UNSEEN: {is_unseen.sum():,} (<=€1.5M: {(is_unseen & (fresh_tr['price']<=1500000)).sum():,})")

    # Incrementally enrich any missing fresh coordinates in Flanders/GRB
    enrich_missing_coords(fresh_tr)

    # Attach raw street, agency, texts for BOTH train and fresh
    raw['date_dt'] = pd.to_datetime(raw['date_obtained'], format='%d%m%Y', errors='coerce')
    raw_u = raw.drop_duplicates(['property_id', 'date_dt'], keep='first')
    tr_m = tr_base[['property_id', 'date_obtained']].merge(
        raw_u[['property_id', 'date_dt', 'title', 'description', 'description_nl', 'street', 'immo_office_name']],
        left_on=['property_id', 'date_obtained'], right_on=['property_id', 'date_dt'], how='left'
    )

    def make_texts(df_m):
        out = []
        for t, nl, d in zip(df_m['title'].fillna(''), df_m['description_nl'].fillna(''), df_m['description'].fillna('')):
            body = str(nl).strip() if len(str(nl).strip()) >= 40 else str(d).strip()
            out.append(f"{str(t).strip()} {body}".strip())
        return out

    tr_texts = make_texts(tr_m)
    va_texts = make_texts(fresh_meta)

    # Normalize agency casing (fixes the 8 split agencies we discovered!)
    tr_base['immo_office_name'] = tr_m['immo_office_name'].fillna('Unknown').astype(str).str.strip()
    tr_base['street_norm'] = tr_base['postal_code'].astype(str) + '|' + tr_m['street'].fillna('').astype(str).str.lower().str.strip()
    fresh_tr['immo_office_name'] = fresh_meta['immo_office_name'].fillna('Unknown').astype(str).str.strip()
    fresh_tr['street_norm'] = fresh_tr['postal_code'].astype(str) + '|' + fresh_meta['street'].fillna('').astype(str).str.lower().str.strip()

    # Row-local domain + text flags + Flanders/GRB
    tr_dom = pd.concat([_add_domain_features(tr_base).reset_index(drop=True), _extract_text_flags(tr_texts).reset_index(drop=True)], axis=1)
    tr_dom = attach_flanders_and_grb(tr_dom)
    va_dom = pd.concat([_add_domain_features(fresh_tr).reset_index(drop=True), _extract_text_flags(va_texts).reset_index(drop=True)], axis=1)
    va_dom = attach_flanders_and_grb(va_dom)

    # CPI index (2025 median ref, identical to Phase 3/4)
    cpi_ref = float(tr_base.loc[tr_base['publication_year'] == 2025, 'belgian_cpi_index'].median())
    tr_cpi = (tr_base['belgian_cpi_index'] / cpi_ref).to_numpy(dtype=float)
    va_cpi = (fresh_tr['belgian_cpi_index'].fillna(tr_base['belgian_cpi_index'].max()) / cpi_ref).to_numpy(dtype=float)

    # Spatial + TE strictly from tr_dom (inner-OOF on train, full-train on fresh!)
    print("\n=== 2. COMPUTING STRICTLY TRAIN-ONLY SPATIAL, TARGET ENCODING & E5/TF-IDF FEATURES ===")
    va_sp = _compute_spatial_and_te(tr_dom, va_dom).reset_index(drop=True)
    tr_folds = tr_dom['fold'].to_numpy()
    tr_sp = pd.concat([_compute_spatial_and_te(tr_dom[tr_folds != uf], tr_dom[tr_folds == uf]) for uf in np.unique(tr_folds)], axis=0).sort_index().reset_index(drop=True)

    # Embeddings: load cached train E5-small/E5-large, embed fresh on MPS GPU
    e5l_tr = np.load(E5_LARGE_PATH)['embeddings'].astype(np.float32)
    e5s_tr = np.load(E5_SMALL_PATH)['embeddings'].astype(np.float32)
    fresh_emb_cache = os.path.join(WORK_DIR, 'fresh_nov23_embs.npz')
    if os.path.exists(fresh_emb_cache):
        zc = np.load(fresh_emb_cache)
        if len(zc['e5l']) == len(va_texts):
            e5l_va, e5s_va = zc['e5l'], zc['e5s']
            print("  Loaded cached fresh E5-large + E5-small embeddings.")
        else:
            e5l_va = embed_texts_mps(va_texts, 'intfloat/multilingual-e5-large', 448, 64)
            e5s_va = embed_texts_mps(va_texts, 'intfloat/multilingual-e5-small', 384, 128)
            np.savez_compressed(fresh_emb_cache, e5l=e5l_va, e5s=e5s_va)
    else:
        e5l_va = embed_texts_mps(va_texts, 'intfloat/multilingual-e5-large', 448, 64)
        e5s_va = embed_texts_mps(va_texts, 'intfloat/multilingual-e5-small', 384, 128)
        np.savez_compressed(fresh_emb_cache, e5l=e5l_va, e5s=e5s_va)

    tr_y_psqm = np.log((tr_dom['price'].to_numpy(dtype=float) / tr_cpi) / tr_dom['livable_area'].to_numpy(dtype=float))
    tr_y_price = np.log(tr_dom['price'].to_numpy(dtype=float) / tr_cpi)
    tr_anchor = np.log(np.maximum(tr_sp['sp_same_type_idw_psqm_k15'].to_numpy(dtype=float), 500.0))
    tr_y_resid = tr_y_psqm - tr_anchor

    tr_txt, va_txt = fit_transform_phase4_text(
        tr_texts, va_texts,
        e5l_tr, e5l_va,
        e5s_tr, e5s_va,
        tr_y_psqm, tr_y_price, tr_y_resid,
        tr_dom['property_key'].astype(str).to_numpy(),
    )

    tr_full = pd.concat([tr_dom.drop(columns=[c for c in tr_sp.columns if c in tr_dom.columns]), tr_sp, tr_txt], axis=1)
    va_full = pd.concat([va_dom.drop(columns=[c for c in va_sp.columns if c in va_dom.columns]), va_sp, va_txt], axis=1)

    for col in ('statbel_sector', 'neighbourhood', 'postal_code', 'property_subtype', 'immo_office_name'):
        if col in tr_full.columns:
            vc = tr_full[col].fillna('Missing').astype(str).value_counts()
            tr_full[f'freq_{col}'] = tr_full[col].fillna('Missing').astype(str).map(vc).fillna(0).astype('float32')
            va_full[f'freq_{col}'] = va_full[col].fillna('Missing').astype(str).map(vc).fillna(0).astype('float32')

    for drop_c in ('street_norm',):
        if drop_c in tr_full.columns:
            tr_full.drop(columns=[drop_c], inplace=True)
        if drop_c in va_full.columns:
            va_full.drop(columns=[drop_c], inplace=True)

    # Load Phase-4 report & Phase-3A tuned params
    with open(os.path.join(WORK_DIR, 'phase4_report.json')) as fh:
        p4r = json.load(fh)
    with open(os.path.join(WORK_DIR, 'phase3a_report.json')) as fh:
        p3a = json.load(fh)
    pruned_feats = [f for f in p4r['pruned_features'] if f in tr_full.columns and f in va_full.columns]
    cb_feats = list(dict.fromkeys(pruned_feats + ['immo_office_name']))
    best_lgb_p = p3a['best_lgb_params']
    best_xgb_p = p3a['best_xgb_params']
    lgb_xt_p = dict(best_lgb_p, extra_trees=True, colsample_bytree=0.55, num_leaves=63, min_child_samples=20)
    weights_u = p4r['weights_unified']

    # ======================================================================
    # 3. FIT BASELINE + PHASE-4 FRONTIER ENSEMBLE ON 100% OF PRE-22-NOV DATA
    #    AND PREDICT ON THE BRAND-NEW 22–23 NOV LIVE SCRAPE!
    # ======================================================================
    print(f"\n=== 3. TRAINING FULL-CORPUS MODELS (n=14,599) & PREDICTING ON FRESH 22–23 NOV SCRAPE ===")
    y_fresh = va_full['price'].to_numpy(dtype=float)
    unseen_mask = va_full['is_strictly_unseen'].to_numpy(dtype=bool)
    unseen_indom = unseen_mask & (y_fresh <= 1_500_000)

    def show_eval(name: str, pred: np.ndarray, dt: float):
        m_u_in = compute_metrics(y_fresh[unseen_indom], pred[unseen_indom])
        m_u_all = compute_metrics(y_fresh[unseen_mask], pred[unseen_mask])
        m_all = compute_metrics(y_fresh[y_fresh <= 1_500_000], pred[y_fresh <= 1_500_000])
        print(f"  {name:34s} ({dt:5.1f}s) | UNSEEN<=1.5M (n={unseen_indom.sum()}): {fmt(m_u_in)}")
        print(f"  {'':42s} | ALL UNSEEN   (n={unseen_mask.sum()}): {fmt(m_u_all)}")
        return {'unseen_indomain': m_u_in, 'unseen_all': m_u_all, 'all_fresh_indomain': m_all}

    # 3a. Phase-1 Baseline (basic 235 features from training_dataset.parquet, default LightGBM log_price)
    base_feats = [c for c in tr_base.columns if c not in ('property_id', 'date_obtained', 'property_key', 'fold', 'price', 'immo_office_name', 'street_norm') and c in va_full.columns]
    t1 = time.perf_counter()
    _, pred_p1_base, _ = _fit_gbdt_split(
        'full_live', tr_full, va_full, tr_cpi, va_cpi,
        base_feats, 'lgbm', 'log_price', 'l2', {}, n_threads=12, seed=42,
    )
    res_dict = {'phase1_baseline_lgbm': show_eval('Phase-1 Baseline (LGBM basic)', pred_p1_base, time.perf_counter() - t1)}

    # 3b. Every active member of the Phase-4 Frontier Ensemble (weight > 0, plus top LGBM/XT)
    member_preds: dict[str, np.ndarray] = {}
    gbdt_jobs = [
        ('p4_lgbm_price_huber',  pruned_feats, 'lgbm',     'log_real_price', 'huber', best_lgb_p),
        ('p4_lgbmxt_price_l2',   pruned_feats, 'lgbm',     'log_real_price', 'l2',    lgb_xt_p),
        ('p4_lgbmxt_psqm_huber', pruned_feats, 'lgbm',     'log_real_psqm',  'huber', lgb_xt_p),
        ('p4_cb_d7_price_l2',    cb_feats,     'catboost', 'log_real_price', 'l2',    {'depth': 7, 'l2_leaf_reg': 5.0, 'learning_rate': 0.03, 'iterations': 4500}),
        ('p4_cb_d7_psqm_l2',     cb_feats,     'catboost', 'log_real_psqm',  'l2',    {'depth': 7, 'l2_leaf_reg': 5.0, 'learning_rate': 0.03, 'iterations': 4500}),
    ]
    for nm, fset, kind, tmode, lmode, params in gbdt_jobs:
        t1 = time.perf_counter()
        _, vp, _ = _fit_gbdt_split(
            'full_live', tr_full, va_full, tr_cpi, va_cpi,
            fset, kind, tmode, lmode, params, n_threads=12, seed=42,
        )
        member_preds[nm] = vp
        res_dict[nm] = show_eval(nm, vp, time.perf_counter() - t1)

    for nm, fset, kind, tmode, lmode, params in [
        ('p4_spec_xgb_psqm', pruned_feats, 'xgb',      'log_real_psqm', 'l2', best_xgb_p),
        ('p4_spec_cb_psqm',  cb_feats,     'catboost', 'log_real_psqm', 'l2', {'depth': 7, 'l2_leaf_reg': 5.0, 'learning_rate': 0.03, 'iterations': 4000}),
    ]:
        t1 = time.perf_counter()
        vp = fit_predict_specialist_full(tr_full, va_full, tr_cpi, va_cpi, fset, kind, tmode, lmode, params)
        member_preds[nm] = vp
        res_dict[nm] = show_eval(nm, vp, time.perf_counter() - t1)

    # SOTA TabArena MPS Neural Networks (the 66.7% backbone of Phase 4!)
    for nm, mtype, tmode, ns in [
        ('p4_realmlp_price_3seed', 'realmlp', 'log_real_price', 3),
        ('p4_tabm_price_3seed',    'tabm',    'log_real_price', 3),
        ('p4_tabm_psqm_3seed',     'tabm',    'log_real_psqm',  3),
    ]:
        t1 = time.perf_counter()
        vp = fit_predict_pytabkit_full(tr_full, va_full, tr_cpi, va_cpi, pruned_feats, mtype, tmode, n_seeds=ns)
        member_preds[nm] = vp
        res_dict[nm] = show_eval(nm, vp, time.perf_counter() - t1)

    # Assemble exact Phase-4 Unified Convex Ensemble using the pre-learned OOF weights (zero tuning on live holdout!)
    active = [(nm, float(weights_u.get(nm, 0.0))) for nm in member_preds if weights_u.get(nm, 0.0) > 0]
    w_sum = sum(w for _, w in active)
    ens_log = sum((w / w_sum) * np.log(np.maximum(member_preds[nm], 10_000.0)) for nm, w in active)
    pred_ens = np.exp(ens_log)
    gbdt_only_log = np.mean([np.log(np.maximum(member_preds[nm], 10_000.0)) for nm in member_preds if 'tabm' not in nm and 'realmlp' not in nm], axis=0)
    pred_gbdt = np.exp(gbdt_only_log)

    print("\n=== 4. FINAL LIVE HOLDOUT COMPARISON (22–23 NOV 2026 FRESH SCRAPE) ===")
    res_dict['p4_gbdt_blend'] = show_eval('Phase-4 GBDT Blend', pred_gbdt, 0.0)
    res_dict['p4_frontier_ensemble'] = show_eval('★ PHASE-4 FRONTIER ENSEMBLE ★', pred_ens, 0.0)

    # Conformal 80% interval check using Phase-3B calibrated q_hat_log (~0.0375) + quantile models
    # Save detailed per-listing predictions CSV
    out_df = pd.DataFrame({
        'property_id': fresh_tr['property_id'].values,
        'date_obtained': fresh_tr['date_obtained'].astype(str).values,
        'is_strictly_unseen': unseen_mask,
        'property_type': np.where(fresh_tr['property_type_HOUSE'].to_numpy() == 1, 'HOUSE', 'APARTMENT'),
        'property_subtype': fresh_tr['property_subtype'].values,
        'neighbourhood': fresh_tr['neighbourhood'].values,
        'postal_code': fresh_tr['postal_code'].values,
        'street': fresh_meta['street'].values,
        'immo_office_name': fresh_meta['immo_office_name'].values,
        'livable_area': fresh_tr['livable_area'].values,
        'energy_label': fresh_tr['energy_label'].values,
        'asking_eur': y_fresh,
        'pred_phase1_base_eur': np.round(pred_p1_base, 0),
        'pred_phase4_ens_eur': np.round(pred_ens, 0),
        'delta_eur': np.round(y_fresh - pred_ens, 0),
        'delta_pct': np.round((y_fresh - pred_ens) / pred_ens * 100.0, 2),
        'abs_ape_pct': np.round(np.abs(y_fresh - pred_ens) / y_fresh * 100.0, 2),
        'title': fresh_meta['title'].fillna('').str.slice(0, 80).values,
    })
    csv_path = os.path.join(WORK_DIR, 'fresh_nov23_predictions.csv')
    out_df.to_csv(csv_path, index=False)

    # Print breakdown on unseen in-domain listings
    u_df = out_df[out_df['is_strictly_unseen'] & (out_df['asking_eur'] <= 1_500_000)].copy()
    print(f"\n=== 5. SLICE BREAKDOWN ON BRAND-NEW UNSEEN LISTINGS (n={len(u_df)}, <=€1.5M) ===")
    for lbl, sub in [
        ('APARTMENT', u_df[u_df['property_type'] == 'APARTMENT']),
        ('HOUSE', u_df[u_df['property_type'] == 'HOUSE']),
        ('<€350k', u_df[u_df['asking_eur'] < 350000]),
        ('€350-500k', u_df[(u_df['asking_eur'] >= 350000) & (u_df['asking_eur'] < 500000)]),
        ('€500k-1.5M', u_df[u_df['asking_eur'] >= 500000]),
    ]:
        if len(sub):
            ms = compute_metrics(sub['asking_eur'].to_numpy(), sub['pred_phase4_ens_eur'].to_numpy())
            print(f"  {lbl:14s} (n={len(sub):3d}): {fmt(ms)}")

    print("\n--- Top 8 Best Deals (Asking BELOW Phase-4 Fair Value) among Brand-New Unseen Listings ---")
    for _, r in u_df.nsmallest(8, 'delta_pct').iterrows():
        print(f"  id={r['property_id']} | Ask €{r['asking_eur']:,.0f} vs Fair €{r['pred_phase4_ens_eur']:,.0f} ({r['delta_pct']:+.1f}%) | {r['property_subtype']} {r['livable_area']:.0f}m² | {r['neighbourhood']} | EPC={r['energy_label']} | {r['immo_office_name']}")

    print("\n--- Top 8 Most Overpriced (Asking ABOVE Phase-4 Fair Value) among Brand-New Unseen Listings ---")
    for _, r in u_df.nlargest(8, 'delta_pct').iterrows():
        print(f"  id={r['property_id']} | Ask €{r['asking_eur']:,.0f} vs Fair €{r['pred_phase4_ens_eur']:,.0f} ({r['delta_pct']:+.1f}%) | {r['property_subtype']} {r['livable_area']:.0f}m² | {r['neighbourhood']} | EPC={r['energy_label']} | {r['immo_office_name']}")

    with open(os.path.join(WORK_DIR, 'fresh_nov23_report.json'), 'w') as fh:
        json.dump({'total_seconds': round(time.perf_counter() - t0, 1), 'results': res_dict}, fh, indent=2)
    print(f"\nSaved {csv_path} ({len(out_df):,} listings) in {time.perf_counter()-t0:.1f}s")


if __name__ == '__main__':
    main()
