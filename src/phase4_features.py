"""Phase 4 Feature Factory: Official Flanders + Stad Gent Open Data + GRB Parcels/Footprints
+ `multilingual-e5-large` (1024-d) + Multi-Model Text Meta-Features (Ridge + ElasticNet/Huber + SVD).

Strictly leak-free on the EXACT same 6 evaluation splits (5 grouped OOF folds over 10,858
properties + 2026 temporal holdout) so every metric is 100% apples-to-apples with Phases 1–3!

New signals beyond Phase 3:
  1. Official **Flanders + Stad Gent + GRB** coordinate enrichment (29 columns, 92–100% coverage):
     - Flanders 100m **Mobiscore** (`fl_mobiscore`, r=+0.3405 with log €/m²!)
     - **Onroerend Erfgoed** (`fl_is_protected_monument`, `fl_is_heritage_inventory`,
       `fl_in_protected_townscape`, `fl_heritage_count_150m` r=+0.290, distances)
     - **Stad Gent** 2021 Noise Lden/Lnight dB, Parks, Transit stops@400m, Schools@500m,
       LEZ (`gent_in_lez`), Kotzone, Parking tariff zone, and 3-year Wijk Crime rate
     - **Digitaal Vlaanderen GRB** official cadastral parcel area (`grb_parcel_area`),
       building footprint (`grb_bldg_footprint`), perimeter, compactness, adjoining building
       count (`0=detached €519.5k`, `1=semi €429.5k`, `2=terraced €399k`, `3=dense €350k`),
       100m built-up density, plus engineered ratios:
       `grb_implied_floors = livable_area / grb_bldg_footprint`,
       `grb_parcel_vs_reported_ratio`, `grb_log_parcel_area`, `grb_log_footprint`,
       `grb_open_plot_area = max(0, grb_parcel_area - grb_bldg_footprint)`!
  2. **Upgraded Text Representation** (strictly nested inner-OOF per split):
     - `intfloat/multilingual-e5-large` (1024-dim MPS embeddings): multi-alpha Ridge (`alpha=3`, `alpha=20`)
       predicting `log(real €/m²)`, `log(real_price)`, and structural residual + fold-safe PCA-48
     - Word (1–2g, 35k) + Char-wb (3–5g, 30k) TF-IDF:
       * Multi-alpha Ridge (`alpha=2.5` sharp + `alpha=12.0` smooth) on all 3 targets
       * **SGDRegressor (Huber loss)** on TF-IDF — robust to outlier-priced listings (fixes text overfit on mislabelled garageboxes/villas!)
       * **TruncatedSVD (24 dims)** on TF-IDF — gives tree models direct nonlinear access to latent text topics (renovation style, student kot, luxury loft, garden townhouse)!
  3. **Non-residential anomaly flag** (does NOT drop evaluation rows so benchmark stays 100% apples-to-apples,
     AND provides clean `is_anomaly_train` mask for training-row noise filtering experiments!).

Caches to `outputs/models/workbench/phase4_splits_cache.joblib`.

Usage:
    .venv-api/bin/python -u src/phase4_features.py
"""
from __future__ import annotations

import os
import re
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.sparse import hstack
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, SGDRegressor

from phase3_features import (
    META_COLUMNS,
    NON_STATIONARY_MACRO,
    _add_domain_features,
    _compute_spatial_and_te,
    _extract_text_flags,
    compute_metrics,
    fmt,
)

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK_DIR = os.path.join(BASE_DIR, 'outputs', 'models', 'workbench')
CACHE_PATH = os.path.join(WORK_DIR, 'phase4_splits_cache.joblib')
FLANDERS_PATH = os.path.join(BASE_DIR, 'external_data', 'flanders', 'ghent_coord_enrichment.parquet')
CRIME_PATH = os.path.join(BASE_DIR, 'external_data', 'flanders', 'gent_wijk_crime.parquet')
E5_LARGE_PATH = os.path.join(WORK_DIR, 'e5_large_14599.npz')
E5_SMALL_PATH = os.path.join(WORK_DIR, 'e5_embeddings_14599.npz')


def _normalize_wijk_key(s: str) -> str:
    return re.sub(r'[^a-z0-9]+', '', str(s).lower())


def attach_flanders_and_grb(df: pd.DataFrame) -> pd.DataFrame:
    """Join official Flanders Mobiscore, Onroerend Erfgoed, Stad Gent, Crime & GRB layers."""
    out = df.copy()
    fl = pd.read_parquet(FLANDERS_PATH)
    out['lat_r5'] = out['latitude'].round(5)
    out['lon_r5'] = out['longitude'].round(5)
    out = out.merge(fl, on=['lat_r5', 'lon_r5'], how='left')
    out.drop(columns=['lat_r5', 'lon_r5'], inplace=True)

    # Attach 3-year wijk crime rate
    if os.path.exists(CRIME_PATH) and 'neighbourhood' in out.columns:
        cr = pd.read_parquet(CRIME_PATH)
        cmap = {_normalize_wijk_key(w): float(v) for w, v in zip(cr['wijk'], cr['gent_wijk_crime_3yr'])}
        out['gent_wijk_crime_3yr'] = out['neighbourhood'].map(lambda w: cmap.get(_normalize_wijk_key(w), np.nan))
        out['gent_wijk_crime_log'] = np.log1p(out['gent_wijk_crime_3yr'].fillna(out['gent_wijk_crime_3yr'].median()))

    # Engineered GRB physical geometry features (100% row-local, zero leakage!)
    fp = pd.to_numeric(out['grb_bldg_footprint'], errors='coerce')
    pa = pd.to_numeric(out['grb_parcel_area'], errors='coerce')
    la = pd.to_numeric(out['livable_area'], errors='coerce').clip(lower=10.0)
    ps = pd.to_numeric(out['property_size'], errors='coerce')

    out['grb_log_parcel_area'] = np.log1p(pa.clip(lower=0).fillna(0.0))
    out['grb_log_footprint'] = np.log1p(fp.clip(lower=0).fillna(0.0))
    out['grb_implied_floors'] = np.where(fp > 10.0, np.clip(la / fp, 0.3, 12.0), np.nan)
    out['grb_open_plot_area'] = np.where((pa > 0) & (fp > 0), np.clip(pa - fp, 0.0, 50000.0), np.nan)
    out['grb_log_open_plot'] = np.log1p(out['grb_open_plot_area'].fillna(0.0))
    out['grb_parcel_vs_reported'] = np.where((pa > 10) & (ps > 10), np.clip(ps / pa, 0.1, 5.0), np.nan)
    # Effective plot area: use reported property_size for houses when present, else official GRB parcel area!
    is_h = (out['property_type_HOUSE'] == 1)
    eff_plot = np.where(is_h, np.where(ps > 10, ps, pa), 0.0)
    out['grb_effective_house_plot'] = np.log1p(np.where(np.isfinite(eff_plot) & (eff_plot > 0), eff_plot, 0.0))
    out['grb_is_detached'] = (out['grb_adjoining_bldgs'] == 0).astype(np.int8)
    out['grb_is_semi_detached'] = (out['grb_adjoining_bldgs'] == 1).astype(np.int8)

    # Interactions with Flanders open data
    mobi = pd.to_numeric(out['fl_mobiscore'], errors='coerce').fillna(8.5)
    out['fl_mobiscore_x_area'] = mobi * np.log(la)
    out['fl_heritage_x_epc'] = out['fl_is_heritage_inventory'].fillna(0) * pd.to_numeric(out.get('epc_ordinal', 3), errors='coerce').fillna(3)
    noise = pd.to_numeric(out['gent_noise_total_lden_db'], errors='coerce')
    out['gent_has_high_noise_65db'] = (noise >= 65.0).astype(np.int8)
    out['gent_noise_total_filled'] = noise.fillna(50.0)  # outside >55dB contours = quiet (<55dB)
    out['gent_noise_road_filled'] = pd.to_numeric(out['gent_noise_road_lden_db'], errors='coerce').fillna(50.0)
    out['gent_noise_night_filled'] = pd.to_numeric(out['gent_noise_night_db'], errors='coerce').fillna(45.0)
    return out


def compute_anomaly_flag(titles: np.ndarray, texts: np.ndarray, prices: np.ndarray) -> np.ndarray:
    """High-precision non-residential anomaly detector (houseboats, commercial-only, bare ownership)."""
    flags = np.zeros(len(titles), dtype=np.int8)
    re_boat = re.compile(r'\b(?:woonboot|woonark|rijnaak|woonschip|binnenschip|barge)\b')
    re_bare = re.compile(r'\b(?:blote eigendom|naakte eigendom|vruchtgebruik|erfpacht|recht van opstal|opstalrecht)\b')
    re_comm = re.compile(r'\b(?:handelspand|handelsruimte|kantoorruimte|magazijn|loods|polyvalente ruimte|dienstgebouw|horeca)\b')
    re_res = re.compile(r'\b(?:woning|huis|appartement|duplex|penthouse|villa|woonst|studio|loft)\b')
    re_share = re.compile(r'\b(?:aandelenoverdracht|overname van aandelen|verkoop van aandelen|overdracht van aandelen)\b')
    for i, (t, tx, p) in enumerate(zip(titles, texts, prices)):
        tl = str(t).lower()
        txl = str(tx).lower()
        if re_boat.search(txl) or re_bare.search(txl) or re_share.search(txl):
            flags[i] = 1
        elif re_comm.search(tl) and not re_res.search(tl):
            flags[i] = 1
    return flags


def fit_transform_phase4_text(
    tr_texts: list[str],
    va_texts: list[str],
    tr_e5l: np.ndarray,
    va_e5l: np.ndarray,
    tr_e5s: np.ndarray,
    va_e5s: np.ndarray,
    tr_y_psqm: np.ndarray,
    tr_y_price: np.ndarray,
    tr_y_resid: np.ndarray,
    tr_groups: np.ndarray,
    n_inner: int = 5,
    n_pca_e5: int = 40,
    n_svd_tf: int = 24,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Strictly fold-safe nested inner-OOF text meta-features:
      - Word (1-2g, 35k) + Char-wb (3-5g, 30k) TF-IDF:
          * Sharp Ridge (alpha=2.5) + Smooth Ridge (alpha=14.0) on log(real €/m²), log(real_price), struct_resid
          * Huber SGDRegressor on log(real €/m²) & struct_resid (robust to outlier listings!)
          * TruncatedSVD (24 dims)
      - E5-large (1024-d) Ridge (alpha=5.0 & 25.0) on all 3 targets + PCA-40
      - E5-small (384-d) Ridge (alpha=10.0) on log(real €/m²) & struct_resid
    """
    n_tr, n_va = len(tr_texts), len(va_texts)
    uniq_g = np.unique(tr_groups)
    rng = np.random.default_rng(42)
    rng.shuffle(uniq_g)
    g2f = {g: i % n_inner for i, g in enumerate(uniq_g)}
    inner_f = np.array([g2f[g] for g in tr_groups])

    vec_w = TfidfVectorizer(max_features=35000, ngram_range=(1, 2), min_df=4, max_df=0.90, sublinear_tf=True, token_pattern=r'(?u)\b[a-zA-Z0-9éèêëàâïîôùûç€²]{2,}\b')
    vec_c = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), max_features=30000, min_df=10, max_df=0.85, sublinear_tf=True)
    X_tr_tf = hstack([vec_w.fit_transform(tr_texts), vec_c.fit_transform(tr_texts)]).tocsr()
    X_va_tf = hstack([vec_w.transform(va_texts), vec_c.transform(va_texts)]).tocsr()

    # Concatenated dense embeddings (E5-large 1024 + E5-small 384)
    tr_emb_both = np.hstack([tr_e5l, tr_e5s])
    va_emb_both = np.hstack([va_e5l, va_e5s])

    cols = [
        'txt_tfidf_a2_psqm', 'txt_tfidf_a2_price', 'txt_tfidf_a2_resid',
        'txt_tfidf_a14_psqm', 'txt_tfidf_a14_resid',
        'txt_tfidf_huber_psqm', 'txt_tfidf_huber_resid',
        'txt_e5l_a5_psqm', 'txt_e5l_a5_price', 'txt_e5l_a5_resid',
        'txt_e5l_a25_psqm', 'txt_e5l_a25_resid',
        'txt_e5both_psqm', 'txt_e5both_resid',
    ]
    tr_meta = {c: np.zeros(n_tr, dtype=np.float32) for c in cols}
    va_meta = {}

    def _fit_pred_huber(Xtr, ytr, Xte):
        m = SGDRegressor(loss='huber', epsilon=0.15, alpha=1e-4, max_iter=60, tol=1e-4, random_state=42)
        m.fit(Xtr, ytr)
        return m.predict(Xte)

    for f in range(n_inner):
        m_tr = inner_f != f
        m_va = inner_f == f
        for name, alpha, target in [
            ('txt_tfidf_a2_psqm', 2.5, tr_y_psqm),
            ('txt_tfidf_a2_price', 2.5, tr_y_price),
            ('txt_tfidf_a2_resid', 2.5, tr_y_resid),
            ('txt_tfidf_a14_psqm', 14.0, tr_y_psqm),
            ('txt_tfidf_a14_resid', 14.0, tr_y_resid),
        ]:
            tr_meta[name][m_va] = Ridge(alpha=alpha, solver='lsqr', max_iter=200).fit(X_tr_tf[m_tr], target[m_tr]).predict(X_tr_tf[m_va])

        tr_meta['txt_tfidf_huber_psqm'][m_va] = _fit_pred_huber(X_tr_tf[m_tr], tr_y_psqm[m_tr], X_tr_tf[m_va])
        tr_meta['txt_tfidf_huber_resid'][m_va] = _fit_pred_huber(X_tr_tf[m_tr], tr_y_resid[m_tr], X_tr_tf[m_va])

        for name, alpha, target in [
            ('txt_e5l_a5_psqm', 5.0, tr_y_psqm),
            ('txt_e5l_a5_price', 5.0, tr_y_price),
            ('txt_e5l_a5_resid', 5.0, tr_y_resid),
            ('txt_e5l_a25_psqm', 25.0, tr_y_psqm),
            ('txt_e5l_a25_resid', 25.0, tr_y_resid),
        ]:
            tr_meta[name][m_va] = Ridge(alpha=alpha).fit(tr_e5l[m_tr], target[m_tr]).predict(tr_e5l[m_va])

        tr_meta['txt_e5both_psqm'][m_va] = Ridge(alpha=12.0).fit(tr_emb_both[m_tr], tr_y_psqm[m_tr]).predict(tr_emb_both[m_va])
        tr_meta['txt_e5both_resid'][m_va] = Ridge(alpha=12.0).fit(tr_emb_both[m_tr], tr_y_resid[m_tr]).predict(tr_emb_both[m_va])

    # Full train fit -> val/test predictions
    for name, alpha, target in [
        ('txt_tfidf_a2_psqm', 2.5, tr_y_psqm),
        ('txt_tfidf_a2_price', 2.5, tr_y_price),
        ('txt_tfidf_a2_resid', 2.5, tr_y_resid),
        ('txt_tfidf_a14_psqm', 14.0, tr_y_psqm),
        ('txt_tfidf_a14_resid', 14.0, tr_y_resid),
    ]:
        va_meta[name] = Ridge(alpha=alpha, solver='lsqr', max_iter=200).fit(X_tr_tf, target).predict(X_va_tf).astype(np.float32)

    va_meta['txt_tfidf_huber_psqm'] = _fit_pred_huber(X_tr_tf, tr_y_psqm, X_va_tf).astype(np.float32)
    va_meta['txt_tfidf_huber_resid'] = _fit_pred_huber(X_tr_tf, tr_y_resid, X_va_tf).astype(np.float32)

    for name, alpha, target in [
        ('txt_e5l_a5_psqm', 5.0, tr_y_psqm),
        ('txt_e5l_a5_price', 5.0, tr_y_price),
        ('txt_e5l_a5_resid', 5.0, tr_y_resid),
        ('txt_e5l_a25_psqm', 25.0, tr_y_psqm),
        ('txt_e5l_a25_resid', 25.0, tr_y_resid),
    ]:
        va_meta[name] = Ridge(alpha=alpha).fit(tr_e5l, target).predict(va_e5l).astype(np.float32)

    va_meta['txt_e5both_psqm'] = Ridge(alpha=12.0).fit(tr_emb_both, tr_y_psqm).predict(va_emb_both).astype(np.float32)
    va_meta['txt_e5both_resid'] = Ridge(alpha=12.0).fit(tr_emb_both, tr_y_resid).predict(va_emb_both).astype(np.float32)

    # Agreement / spread between TF-IDF and E5-large (when text lexical vs semantic disagree!)
    tr_meta['txt_tfidf_minus_e5l_psqm'] = (tr_meta['txt_tfidf_a2_psqm'] - tr_meta['txt_e5l_a5_psqm']).astype(np.float32)
    va_meta['txt_tfidf_minus_e5l_psqm'] = (va_meta['txt_tfidf_a2_psqm'] - va_meta['txt_e5l_a5_psqm']).astype(np.float32)
    tr_meta['txt_blend_psqm'] = (0.60 * tr_meta['txt_tfidf_a2_psqm'] + 0.40 * tr_meta['txt_e5l_a5_psqm']).astype(np.float32)
    va_meta['txt_blend_psqm'] = (0.60 * va_meta['txt_tfidf_a2_psqm'] + 0.40 * va_meta['txt_e5l_a5_psqm']).astype(np.float32)

    # Fold-safe PCA on E5-large (40 dims) + TruncatedSVD on TF-IDF (24 dims)
    pca = PCA(n_components=n_pca_e5, random_state=42)
    tr_p = pca.fit_transform(tr_e5l).astype(np.float32)
    va_p = pca.transform(va_e5l).astype(np.float32)
    for j in range(n_pca_e5):
        tr_meta[f'e5l_pca_{j}'] = tr_p[:, j]
        va_meta[f'e5l_pca_{j}'] = va_p[:, j]

    svd = TruncatedSVD(n_components=n_svd_tf, random_state=42)
    tr_s = svd.fit_transform(X_tr_tf).astype(np.float32)
    va_s = svd.transform(X_va_tf).astype(np.float32)
    for j in range(n_svd_tf):
        tr_meta[f'tfidf_svd_{j}'] = tr_s[:, j]
        va_meta[f'tfidf_svd_{j}'] = va_s[:, j]

    return pd.DataFrame(tr_meta), pd.DataFrame(va_meta)


def _build_one_split(split_id, tr_idx, va_idx, df, texts, e5l, e5s, cpi_idx, anomaly_flags):
    tr_raw = df.iloc[tr_idx].copy().reset_index(drop=True)
    va_raw = df.iloc[va_idx].copy().reset_index(drop=True)
    tr_cpi = cpi_idx[tr_idx]
    va_cpi = cpi_idx[va_idx]
    tr_folds = tr_raw['fold'].to_numpy()

    # Spatial + Target Encoding strictly inner-OOF on train, full-train on val
    va_sp = _compute_spatial_and_te(tr_raw, va_raw).reset_index(drop=True)
    sp_tr_parts = []
    for uf in np.unique(tr_folds):
        sp_tr_parts.append(_compute_spatial_and_te(tr_raw[tr_folds != uf], tr_raw[tr_folds == uf]))
    tr_sp = pd.concat(sp_tr_parts, axis=0).sort_index().reset_index(drop=True)

    tr_y_psqm = np.log((tr_raw['price'].to_numpy(dtype=float) / tr_cpi) / tr_raw['livable_area'].to_numpy(dtype=float))
    tr_y_price = np.log(tr_raw['price'].to_numpy(dtype=float) / tr_cpi)
    tr_anchor = np.log(np.maximum(tr_sp['sp_same_type_idw_psqm_k15'].to_numpy(dtype=float), 500.0))
    tr_y_resid = tr_y_psqm - tr_anchor

    tr_txt, va_txt = fit_transform_phase4_text(
        [texts[i] for i in tr_idx],
        [texts[i] for i in va_idx],
        e5l[tr_idx], e5l[va_idx],
        e5s[tr_idx], e5s[va_idx],
        tr_y_psqm, tr_y_price, tr_y_resid,
        tr_raw['property_key'].astype(str).to_numpy(),
    )

    tr_df = pd.concat([tr_raw.drop(columns=[c for c in tr_sp.columns if c in tr_raw.columns]), tr_sp, tr_txt], axis=1)
    va_df = pd.concat([va_raw.drop(columns=[c for c in va_sp.columns if c in va_raw.columns]), va_sp, va_txt], axis=1)

    for col in ('statbel_sector', 'neighbourhood', 'postal_code', 'property_subtype', 'immo_office_name'):
        if col in tr_df.columns:
            vc = tr_df[col].fillna('Missing').astype(str).value_counts()
            tr_df[f'freq_{col}'] = tr_df[col].fillna('Missing').astype(str).map(vc).fillna(0).astype('float32')
            va_df[f'freq_{col}'] = va_df[col].fillna('Missing').astype(str).map(vc).fillna(0).astype('float32')

    if 'street_norm' in tr_df.columns:
        tr_df.drop(columns=['street_norm'], inplace=True)
        va_df.drop(columns=['street_norm'], inplace=True)

    return {
        'id': split_id,
        'tr_idx': tr_idx,
        'va_idx': va_idx,
        'tr_df': tr_df,
        'va_df': va_df,
        'tr_cpi_idx': tr_cpi,
        'va_cpi_idx': va_cpi,
        'tr_anomaly': anomaly_flags[tr_idx],
        'va_anomaly': anomaly_flags[va_idx],
    }


def load_or_build_phase4_splits(force: bool = False) -> dict:
    if os.path.exists(CACHE_PATH) and not force:
        return joblib.load(CACHE_PATH)

    t0 = time.perf_counter()
    print("Building Phase-4 enriched 6-split feature cache (Flanders + GRB + E5-large + TF-IDF SVD/Huber)...")
    df = pd.read_parquet(os.path.join(BASE_DIR, 'outputs', 'dataset', 'training_dataset.parquet'))
    raw = pd.read_parquet(
        os.path.join(BASE_DIR, 'outputs', 'property_data_full.parquet'),
        columns=['property_id', 'date_obtained', 'title', 'description', 'description_nl', 'street', 'immo_office_name'],
    )
    df['property_id'] = df['property_id'].astype(str)
    raw['property_id'] = raw['property_id'].astype(str)
    raw['date_obtained'] = pd.to_datetime(raw['date_obtained'], format='%d%m%Y', errors='coerce')
    raw = raw.drop_duplicates(['property_id', 'date_obtained'], keep='first')
    m = df[['property_id', 'date_obtained']].merge(raw, on=['property_id', 'date_obtained'], how='left')

    titles = m['title'].fillna('').astype(str).to_numpy()
    texts = []
    for ttl, d_nl, d_main in zip(titles, m['description_nl'].fillna(''), m['description'].fillna('')):
        body = d_nl if len(str(d_nl)) >= 40 else d_main
        texts.append(f"{ttl} . {body}" if ttl else str(body))

    df['street_norm'] = (df['postal_code'].fillna('').astype(str) + '|' + m['street'].fillna('').astype(str).str.lower().str.strip())
    df['immo_office_name'] = m['immo_office_name'].fillna('Unknown').astype(str)

    # Row-local Phase-3 domain/text flags + Phase-4 Flanders/GRB layers
    df = _add_domain_features(df)
    txt_flags = _extract_text_flags(texts)
    df = pd.concat([df.reset_index(drop=True), txt_flags.reset_index(drop=True)], axis=1)
    df = attach_flanders_and_grb(df)
    anomaly_flags = compute_anomaly_flag(titles, texts, df['price'].to_numpy(dtype=float))
    print(f"  High-precision non-residential anomalies flagged: {int(anomaly_flags.sum())} / {len(df)}")

    cpi = df['belgian_cpi_index'].to_numpy(dtype=float)
    pub_y = df['publication_year'].fillna(2025).to_numpy()
    cpi_ref = float(np.median(cpi[pub_y == 2025]))
    cpi_idx = cpi / cpi_ref

    e5l = np.load(E5_LARGE_PATH)['embeddings'].astype(np.float32)
    e5s = np.load(E5_SMALL_PATH)['embeddings'].astype(np.float32)
    assert len(e5l) == len(df) == len(e5s)

    folds = df['fold'].to_numpy()
    test_2026 = (df['publication_year'] >= 2026).to_numpy()
    test_keys = set(df.loc[test_2026, 'property_key'].astype(str))
    train_pre26 = (~test_2026) & (~df['property_key'].astype(str).isin(test_keys).to_numpy())

    split_specs = [(f'fold_{f}', np.where(folds != f)[0], np.where(folds == f)[0]) for f in sorted(np.unique(folds))]
    split_specs.append(('temporal_2026', np.where(train_pre26)[0], np.where(test_2026)[0]))

    bundles = Parallel(n_jobs=6, backend='loky')(
        delayed(_build_one_split)(sid, tr_i, va_i, df, texts, e5l, e5s, cpi_idx, anomaly_flags)
        for sid, tr_i, va_i in split_specs
    )

    sample_cols = list(bundles[0]['tr_df'].columns)
    feat_cols_stationary = [
        c for c in sample_cols
        if c not in META_COLUMNS and c not in NON_STATIONARY_MACRO and c != 'immo_office_name'
    ]
    feat_cols_with_agency = [
        c for c in sample_cols
        if c not in META_COLUMNS and c not in NON_STATIONARY_MACRO
    ]

    payload = {
        'bundles': bundles,
        'y_price': df['price'].to_numpy(dtype=float),
        'y_tmp_true': df.loc[test_2026, 'price'].to_numpy(dtype=float),
        'folds': folds,
        'cpi_idx': cpi_idx,
        'cpi_ref': cpi_ref,
        'is_house': (df['property_type_HOUSE'] == 1).to_numpy(),
        'anomaly_flags': anomaly_flags,
        'feat_cols_stationary': feat_cols_stationary,
        'feat_cols_with_agency': feat_cols_with_agency,
    }
    joblib.dump(payload, CACHE_PATH, compress=3)
    print(f"Built & cached Phase-4 splits ({len(feat_cols_stationary)} features) in {time.perf_counter()-t0:.1f}s -> {CACHE_PATH}")
    return payload


if __name__ == '__main__':
    load_or_build_phase4_splits(force=True)
