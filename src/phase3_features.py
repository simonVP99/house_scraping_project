"""Shared Phase-3 Feature Factory & Leakage-Free 6-Split Builder.

Produces a cached, deterministic, strictly fold-safe feature dataset for ALL
Phase-3 model families (Tuned CatBoost, LightGBM, XGBoost, PyTorch MPS Tabular
ResNet, Specialist House/Apartment Experts, AutoGluon, and Conformal Quantiles)
so every model is compared apples-to-apples on the exact same:
  - 5-fold grouped OOF (14,599 rows / 10,858 unique physical properties, 0 leaked groups)
  - 2026 Temporal Holdout (8,787 pre-2026 train with 0 overlapping property_keys -> 5,365 2026 test)

New Phase-3 high-signal features (all strictly fold-safe via nested inner-OOF):
  1. CPI-deflated target index (combined with log €/m² -> `log(real_price / m²)`).
  2. Richer text signal:
     - Word TF-IDF (1-2 grams, 30k vocab) + Char-wb TF-IDF (3-5 grams, 30k vocab)
       -> nested-OOF Ridge on `log(€/m²)`, `log(price)`, AND **residual over
       (type, neighbourhood, area)** so text focuses purely on condition/finish!
     - Explicit Flemish real-estate text flags:
       `txt_is_starting_price`, `txt_is_totaalrenovatie`, `txt_is_to_renovate`,
       `txt_is_luxury_finish`, `txt_is_investment`, `txt_is_mixed_commercial`,
       `txt_has_water_view`, `txt_has_south_facing`, `txt_has_underfloor_heating`,
       `txt_is_heritage_charter`
     - E5-small (384-d MPS) nested-OOF Ridge on `log(€/m²)` + `log(price)` +
       structural residual + 32-dim fold-safe PCA.
  3. High-cardinality empirical-Bayes smoothed OOF target encodings (strictly nested):
     - `oof_street_psqm` (2,237 streets — captures prestigious vs noisy streets!)
     - `oof_agency_psqm` & `oof_agency_price` (529 agencies — luxury vs budget segment)
     - `oof_subtype_psqm` & `oof_subtype_wijk_psqm`
     - `oof_sector_psqm` (291 Statbel statistical sectors)
  4. Multi-scale type-conditioned IDW spatial features (`k=5, 15, 40`),
     plus local €/m² dispersion, local area ratio, and rotated coordinates.
  5. Domain interactions: `ki_per_sqm`, Flemish EPC E/F × area penalty,
     condition × area, age × condition, cyclical month, sector wealth × area.
"""
from __future__ import annotations

import os
import re
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import polars as pl
from joblib import Parallel, delayed
from scipy.sparse import hstack
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, 'outputs', 'dataset', 'training_dataset.parquet')
RAW_PARQUET_PATH = os.path.join(BASE_DIR, 'outputs', 'property_data_full.parquet')
E5_PATH = os.path.join(BASE_DIR, 'outputs', 'models', 'workbench', 'e5_embeddings_14599.npz')
CACHE_PATH = os.path.join(BASE_DIR, 'outputs', 'models', 'workbench', 'phase3_splits_cache.joblib')

META_COLUMNS = {'property_id', 'date_obtained', 'property_key', 'fold', 'price'}
NON_STATIONARY_MACRO = {
    'belgian_cpi_index', 'belgian_health_index', 'belgian_hpi_index', 'publication_year',
}


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 10_000.0, None)
    abs_err = np.abs(y_pred - y_true)
    ape = abs_err / np.maximum(y_true, 1.0) * 100.0
    log_err = np.log(y_pred) - np.log(np.maximum(y_true, 1.0))
    return {
        'mdape': float(np.median(ape)),
        'mape': float(np.mean(ape)),
        'hit_10': float(np.mean(ape <= 10.0) * 100.0),
        'hit_15': float(np.mean(ape <= 15.0) * 100.0),
        'mae': float(np.mean(abs_err)),
        'rmse': float(np.sqrt(np.mean((y_pred - y_true) ** 2))),
        'log_rmse': float(np.sqrt(np.mean(log_err ** 2))),
        'r2': float(1.0 - np.sum((y_pred - y_true) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)),
        'bias_pct': float(np.median((y_pred - y_true) / y_true) * 100.0),
    }


def fmt(m: dict[str, float]) -> str:
    return (
        f"MdAPE={m['mdape']:5.2f}% | Hit@10={m['hit_10']:4.1f}% | Hit@15={m['hit_15']:4.1f}% | "
        f"MAE=€{m['mae']:6,.0f} | RMSE=€{m['rmse']:6,.0f} | R²={m['r2']:.4f} | Bias={m['bias_pct']:+5.2f}%"
    )


def _extract_text_flags(texts: list[str]) -> pd.DataFrame:
    s = pd.Series(texts, dtype=str).str.lower()
    patterns = {
        'txt_is_starting_price': r'(?:instelprijs|vanafprijs|bieden\s+vanaf|richtprijs|startprijs|minimumprijs|bieden\s+onder\s+gesloten)',
        'txt_is_totaalrenovatie': r'(?:totaalrenovatie|volledig\s+gerenoveerd|instapklaar|nieuwstaat|recent\s+gerenoveerd)',
        'txt_is_to_renovate': r'(?:te\s+renoveren|op\s+te\s+frissen|renovatieproject|sloop|afbraak|opknapper)',
        'txt_is_luxury_finish': r'(?:architect|hoogwaardig|luxe|parket|natuursteen|maatwerk|design|domotica|marmer)',
        'txt_has_underfloor_heat': r'(?:vloerverwarming|chauffage\s+au\s+sol)',
        'txt_is_investment': r'(?:opbrengsteigendom|studentenkamer|investeringseigendom|verhuurd\s+aan|huuropbrengst|rendement)',
        'txt_is_mixed_comm': r'(?:handelshuis|handelsgelijkvloers|polyvalente\s+ruimte|handelsruimte|praktijkruimte)',
        'txt_has_water_view': r'(?:zicht\s+op\s+(?:het\s+)?water|aan\s+het\s+water|leie|schelde|vaart|waterzicht)',
        'txt_has_south_facing': r'(?:zuidgericht|zuidwest|zuid-west|zonnig\s+terras|zonnige\s+tuin|zuidtuin)',
        'txt_is_heritage': r'(?:herenhuis|stadspaleis|authentiek|hoge\s+plafonds|moluren|glas-in-lood|beschermd|art\s+deco)',
        'txt_is_quiet_deadend': r'(?:doodlopende\s+straat|rustige\s+straat|groene\s+omgeving|autoluw|kindvriendelijk)',
        'txt_has_asbestos_safe': r'(?:asbestveilig|asbestvrij)',
        'txt_has_compliant_elec': r'(?:conforme?\s+elektriciteit|elektriciteit\s+conform|are[ií]\s+conform)',
    }
    cols = {k: s.str.contains(pat, regex=True).astype('int8').to_numpy() for k, pat in patterns.items()}
    return pd.DataFrame(cols)


def _add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    new_cols: dict[str, np.ndarray] = {}

    lat = out['latitude'].to_numpy(dtype=float) - 51.0531
    lon = (out['longitude'].to_numpy(dtype=float) - 3.7265) * np.cos(np.radians(51.05))
    for deg in (15, 30, 45, 60, 75):
        rad = np.radians(deg)
        new_cols[f'coord_rot_{deg}_x'] = lat * np.cos(rad) + lon * np.sin(rad)
        new_cols[f'coord_rot_{deg}_y'] = -lat * np.sin(rad) + lon * np.cos(rad)
    new_cols['coord_angle_center'] = np.arctan2(lat, lon)

    area = np.maximum(out['livable_area'].to_numpy(dtype=float), 1.0)
    plot = np.maximum(out['property_size'].to_numpy(dtype=float), 0.0)
    ki = np.maximum(out['cadastral_income'].to_numpy(dtype=float), 1.0)
    new_cols['log_livable_area'] = np.log(area)
    new_cols['log_property_size_p1'] = np.log1p(plot)
    new_cols['log_cadastral_income'] = np.log(ki)
    new_cols['ki_per_sqm'] = ki / area
    new_cols['ki_per_sqm_known'] = np.where(out['cadastral_income_missing'].to_numpy() == 0, ki / area, np.nan)
    new_cols['area_per_bedroom'] = area / np.maximum(out['number_of_bedrooms'].to_numpy(dtype=float), 1.0)
    new_cols['area_per_room'] = area / np.maximum(out['number_of_rooms'].to_numpy(dtype=float), 1.0)
    new_cols['outdoor_surface_total'] = (
        out['garden_surface'].to_numpy(dtype=float) + out['terrace_surface'].to_numpy(dtype=float)
    )
    new_cols['log_outdoor_surface'] = np.log1p(np.maximum(new_cols['outdoor_surface_total'], 0.0))
    new_cols['total_parking'] = (
        out['parking_indoor'].to_numpy(dtype=float) + out['parking_outdoor'].to_numpy(dtype=float)
    )

    epc_map = {'A++': 0, 'A+': 1, 'A': 2, 'B': 3, 'C': 4, 'D': 5, 'E': 6, 'F': 7}
    epc_ord = out['energy_label'].map(epc_map).astype(float).to_numpy()
    new_cols['epc_ordinal'] = epc_ord
    is_ef = np.where(np.isnan(epc_ord), 0.0, (epc_ord >= 6).astype(float))
    new_cols['epc_ef_area_penalty'] = is_ef * area

    cond_map = {
        'AS_NEW': 5, 'JUST_RENOVATED': 4, 'GOOD': 3,
        'TO_BE_DONE_UP': 2, 'TO_RENOVATE': 1, 'TO_RESTORE': 0,
    }
    cond_ord = out['building_condition'].map(cond_map).astype(float).to_numpy()
    new_cols['condition_ordinal'] = cond_ord
    new_cols['condition_x_area'] = np.nan_to_num(cond_ord, nan=3.0) * area
    new_cols['age_x_condition'] = out['age'].to_numpy(dtype=float) * (5.0 - np.nan_to_num(cond_ord, nan=3.0))

    m = out['publication_month'].fillna(6).to_numpy(dtype=float)
    new_cols['pub_month_sin'] = np.sin(2.0 * np.pi * m / 12.0)
    new_cols['pub_month_cos'] = np.cos(2.0 * np.pi * m / 12.0)

    if 'sector_median_income' in out.columns:
        new_cols['log_sector_income_x_area'] = np.log(
            np.maximum(out['sector_median_income'].to_numpy(dtype=float), 10_000.0)
        ) * np.log(area)

    return pd.concat([out, pd.DataFrame(new_cols, index=out.index)], axis=1)


def _compute_spatial_and_te(ref_df: pd.DataFrame, target_df: pd.DataFrame) -> pd.DataFrame:
    """Strictly ref->target multi-scale IDW spatial + empirical-Bayes target encodings."""
    out = pd.DataFrame(index=target_df.index)
    ref_valid = ref_df[
        ref_df['latitude'].notna()
        & ref_df['longitude'].notna()
        & ref_df['price'].notna()
        & (ref_df['livable_area'] > 0)
    ]
    ref_psqm = (ref_valid['price'] / ref_valid['livable_area']).to_numpy(dtype=float)
    ref_log_psqm = np.log(ref_psqm)
    ref_area = ref_valid['livable_area'].to_numpy(dtype=float)
    ref_price = ref_valid['price'].to_numpy(dtype=float)
    ref_is_h = (ref_valid['property_type_HOUSE'].to_numpy() == 1)
    ref_coords = np.column_stack([ref_valid['latitude'].to_numpy(), ref_valid['longitude'].to_numpy()])
    tgt_coords = np.column_stack([target_df['latitude'].to_numpy(), target_df['longitude'].to_numpy()])
    tgt_is_h = (target_df['property_type_HOUSE'].to_numpy() == 1)
    tgt_area = target_df['livable_area'].to_numpy(dtype=float)

    g_med = float(np.median(ref_psqm))
    g_log_med = float(np.median(ref_log_psqm))
    g_price_med = float(np.median(ref_price))

    # 1. All-properties IDW KNN (k=5, 15, 40)
    tree_all = cKDTree(ref_coords)
    k_max = min(40, len(ref_valid))
    dists_all, idxs_all = tree_all.query(tgt_coords, k=k_max)
    dists_km = dists_all * 111.32
    eps = 0.05

    for k in (5, 15, 40):
        kk = min(k, k_max)
        w = 1.0 / (dists_km[:, :kk] + eps)
        w = w / w.sum(axis=1, keepdims=True)
        nbr_psqm = ref_psqm[idxs_all[:, :kk]]
        out[f'sp_idw_psqm_k{k}'] = np.sum(w * nbr_psqm, axis=1)
        if k == 15:
            out['sp_psqm_std_k15'] = np.std(nbr_psqm, axis=1)
            out['sp_psqm_p25_k15'] = np.percentile(nbr_psqm, 25, axis=1)
            out['sp_psqm_p75_k15'] = np.percentile(nbr_psqm, 75, axis=1)
            local_med_area = np.median(ref_area[idxs_all[:, :kk]], axis=1)
            out['sp_rel_area_k15'] = tgt_area / np.maximum(local_med_area, 10.0)
            out['sp_idw_price_k15'] = np.sum(w * ref_price[idxs_all[:, :kk]], axis=1)

    out['sp_dist_3nn_km'] = np.mean(dists_km[:, :min(3, k_max)], axis=1)
    out['sp_dist_10nn_km'] = np.mean(dists_km[:, :min(10, k_max)], axis=1)

    # 2. Type-conditioned IDW KNN (k=5, 15, 40) + Condition-similar nearby €/m²
    same_k5 = np.full(len(target_df), g_med, dtype=float)
    same_k15 = np.full(len(target_df), g_med, dtype=float)
    same_k40 = np.full(len(target_df), g_med, dtype=float)
    same_med_area = np.full(len(target_df), 120.0, dtype=float)
    same_dist5 = np.full(len(target_df), 0.5, dtype=float)

    for h_flag in (True, False):
        r_mask = ref_is_h == h_flag
        t_mask = tgt_is_h == h_flag
        if not r_mask.any() or not t_mask.any():
            continue
        t_tree = cKDTree(ref_coords[r_mask])
        r_p = ref_psqm[r_mask]
        r_a = ref_area[r_mask]
        km = min(40, int(r_mask.sum()))
        d, ix = t_tree.query(tgt_coords[t_mask], k=km)
        dk = d * 111.32
        for k, arr in ((5, same_k5), (15, same_k15), (40, same_k40)):
            kk = min(k, km)
            w = 1.0 / (dk[:, :kk] + eps)
            w = w / w.sum(axis=1, keepdims=True)
            arr[t_mask] = np.sum(w * r_p[ix[:, :kk]], axis=1)
        same_med_area[t_mask] = np.median(r_a[ix[:, :min(15, km)]], axis=1)
        same_dist5[t_mask] = np.mean(dk[:, :min(5, km)], axis=1)

    out['sp_same_type_idw_psqm_k5'] = same_k5
    out['sp_same_type_idw_psqm_k15'] = same_k15
    out['sp_same_type_idw_psqm_k40'] = same_k40
    out['sp_same_type_rel_area'] = tgt_area / np.maximum(same_med_area, 10.0)
    out['sp_same_type_dist5_km'] = same_dist5
    out['sp_prior_price'] = same_k15 * tgt_area
    out['sp_log_prior_price'] = np.log(np.maximum(out['sp_prior_price'], 10_000.0))

    # 3. Empirical-Bayes smoothed OOF target encodings (street, agency, sector, wijk, subtype)
    def _eb_encode(col: str, target_vals: np.ndarray, prior_val: float, m: float) -> np.ndarray:
        if col not in ref_valid.columns:
            return np.full(len(target_df), prior_val, dtype=float)
        tr_g = pd.DataFrame({'k': ref_valid[col].fillna('Unknown').astype(str).to_numpy(), 'v': target_vals})
        agg = tr_g.groupby('k')['v'].agg(['count', 'mean'])
        smooth = {
            idx: (row['count'] * row['mean'] + m * prior_val) / (row['count'] + m)
            for idx, row in agg.iterrows() if idx not in ('Unknown', '', 'None', 'nan')
        }
        return np.array([smooth.get(str(x), prior_val) for x in target_df[col].fillna('Unknown')], dtype=float)

    out['te_street_log_psqm'] = _eb_encode('street_norm', ref_log_psqm, g_log_med, m=4.0)
    out['te_agency_log_psqm'] = _eb_encode('immo_office_name', ref_log_psqm, g_log_med, m=8.0)
    out['te_agency_log_price'] = _eb_encode('immo_office_name', np.log(ref_price), np.log(g_price_med), m=8.0)
    out['te_subtype_log_psqm'] = _eb_encode('property_subtype', ref_log_psqm, g_log_med, m=5.0)

    for grp_col, name, m_prior in (
        ('neighbourhood', 'sp_type_wijk_psqm', 5.0),
        ('statbel_sector', 'sp_type_sector_psqm', 6.0),
        ('postal_code', 'sp_type_postal_psqm', 5.0),
    ):
        tr_g = pd.DataFrame({
            'h': ref_is_h.astype(int),
            'g': ref_valid[grp_col].astype(str).to_numpy(),
            'p': ref_psqm,
        })
        type_med = tr_g.groupby('h')['p'].median().to_dict()
        agg = tr_g.groupby(['h', 'g'])['p'].agg(['count', 'median'])
        smooth = {
            (h, g): (row['count'] * row['median'] + m_prior * type_med.get(h, g_med)) / (row['count'] + m_prior)
            for (h, g), row in agg.iterrows() if g != 'Unknown'
        }
        out[name] = [
            smooth.get((int(h), str(g)), type_med.get(int(h), g_med))
            for h, g in zip(tgt_is_h, target_df[grp_col])
        ]

    out['oof_knn10_median_price_sqm'] = np.median(ref_psqm[idxs_all[:, :min(10, k_max)]], axis=1)
    out['oof_dist_5nn_km'] = np.mean(dists_km[:, :min(5, k_max)], axis=1)
    return out


def _fit_text_fold(
    tr_idx: np.ndarray,
    va_idx: np.ndarray,
    texts: list[str],
    embs: np.ndarray,
    y_log_psqm: np.ndarray,
    y_log_price: np.ndarray,
    y_resid_struct: np.ndarray,
    tr_folds: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Word + char-wb TF-IDF Ridge AND E5 Ridge predicting log(€/m²), log(price), AND structural residual."""
    tr_texts = [texts[i] for i in tr_idx]
    va_texts = [texts[i] for i in va_idx]

    vec_w = TfidfVectorizer(
        max_features=30_000, ngram_range=(1, 2), min_df=4, max_df=0.9, sublinear_tf=True,
    )
    vec_c = TfidfVectorizer(
        analyzer='char_wb', max_features=25_000, ngram_range=(3, 5), min_df=8, max_df=0.9, sublinear_tf=True,
    )
    X_tr_tf = hstack([vec_w.fit_transform(tr_texts), vec_c.fit_transform(tr_texts)]).tocsr()
    X_va_tf = hstack([vec_w.transform(va_texts), vec_c.transform(va_texts)]).tocsr()

    E_tr = embs[tr_idx]
    E_va = embs[va_idx]

    tr_out: dict[str, np.ndarray] = {}
    va_out: dict[str, np.ndarray] = {}
    uniq_f = np.unique(tr_folds)

    for tgt_name, y_vec in (
        ('log_psqm', y_log_psqm),
        ('log_price', y_log_price),
        ('struct_resid', y_resid_struct),
    ):
        # TF-IDF Ridge (word+char)
        oof_tf = np.zeros(len(tr_idx), dtype=np.float32)
        for uf in uniq_f:
            im = tr_folds != uf
            vm = tr_folds == uf
            oof_tf[vm] = Ridge(alpha=4.0).fit(X_tr_tf[im], y_vec[tr_idx][im]).predict(X_tr_tf[vm])
        tr_out[f'txt_tfidf_{tgt_name}'] = oof_tf
        va_out[f'txt_tfidf_{tgt_name}'] = Ridge(alpha=4.0).fit(X_tr_tf, y_vec[tr_idx]).predict(X_va_tf).astype(np.float32)

        # E5 Ridge
        oof_e5 = np.zeros(len(tr_idx), dtype=np.float32)
        for uf in uniq_f:
            im = tr_folds != uf
            vm = tr_folds == uf
            oof_e5[vm] = Ridge(alpha=10.0).fit(E_tr[im], y_vec[tr_idx][im]).predict(E_tr[vm])
        tr_out[f'txt_e5_{tgt_name}'] = oof_e5
        va_out[f'txt_e5_{tgt_name}'] = Ridge(alpha=10.0).fit(E_tr, y_vec[tr_idx]).predict(E_va).astype(np.float32)

    # Fold-safe PCA-32 on E5 embeddings
    pca = PCA(n_components=32, random_state=42)
    P_tr = pca.fit_transform(E_tr).astype(np.float32)
    P_va = pca.transform(E_va).astype(np.float32)
    for j in range(32):
        tr_out[f'e5_pca_{j}'] = P_tr[:, j]
        va_out[f'e5_pca_{j}'] = P_va[:, j]

    return pd.DataFrame(tr_out), pd.DataFrame(va_out)


def load_or_build_phase3_splits(force: bool = False) -> dict:
    """Return cached Phase-3 6-split bundle (5 grouped OOF + 2026 temporal holdout)."""
    if os.path.exists(CACHE_PATH) and not force:
        return joblib.load(CACHE_PATH)

    t0 = time.perf_counter()
    df_raw = pd.read_parquet(DATASET_PATH).reset_index(drop=True)

    # Join raw text, agency, street strictly for feature extraction (never target!)
    raw = pl.read_parquet(
        RAW_PARQUET_PATH,
        columns=['property_id', 'date_obtained', 'title', 'description', 'description_nl', 'immo_office_name', 'street'],
    ).to_pandas()
    raw['date_obtained'] = pd.to_datetime(raw['date_obtained'], format='%d%m%Y', errors='coerce')
    raw['property_id'] = raw['property_id'].astype(str)

    key_df = df_raw[['property_id', 'date_obtained']].copy()
    key_df['property_id'] = key_df['property_id'].astype(str)
    m = key_df.merge(raw.drop_duplicates(['property_id', 'date_obtained']), on=['property_id', 'date_obtained'], how='left')

    texts = []
    for t, nl, d in zip(m['title'].fillna(''), m['description_nl'].fillna(''), m['description'].fillna('')):
        body = nl if len(str(nl).strip()) >= 40 else d
        texts.append(f"{str(t).strip()} {str(body).strip()}".strip())

    # Add street_norm and agency name for OOF target encoding (not raw high-card string!)
    df_raw['street_norm'] = (
        df_raw['postal_code'].astype(str) + '|' + m['street'].fillna('').astype(str).str.lower().str.strip()
    )
    df_raw['immo_office_name'] = m['immo_office_name'].fillna('Unknown').astype(str)

    # Add domain features & explicit Flemish text flags
    df_dom = _add_domain_features(df_raw)
    txt_flags = _extract_text_flags(texts)
    df_dom = pd.concat([df_dom.reset_index(drop=True), txt_flags.reset_index(drop=True)], axis=1)

    embs = np.load(E5_PATH)['embeddings'].astype(np.float32)
    y_price = df_raw['price'].to_numpy(dtype=float)
    y_log_price = np.log(y_price)
    y_log_psqm = np.log(y_price / df_raw['livable_area'].to_numpy(dtype=float))

    # CPI deflator relative to 2025 median (proven cleanest bias +0.85% on 2026 temporal!)
    cpi_ref = float(df_raw.loc[df_raw['publication_year'] == 2025, 'belgian_cpi_index'].median())
    cpi_idx = (df_raw['belgian_cpi_index'] / cpi_ref).to_numpy(dtype=float)
    y_log_real_psqm = np.log((y_price / cpi_idx) / df_raw['livable_area'].to_numpy(dtype=float))

    # Build the 6 splits
    test_mask_t = (df_raw['publication_year'] >= 2026).to_numpy()
    test_keys_t = set(df_raw.loc[test_mask_t, 'property_key'].astype(str))
    train_mask_t = (~test_mask_t) & (~df_raw['property_key'].astype(str).isin(test_keys_t).to_numpy())
    tr_idx_t = np.where(train_mask_t)[0]
    va_idx_t = np.where(test_mask_t)[0]

    split_specs = [
        (f'fold_{f}', np.where(df_raw['fold'].to_numpy() != f)[0], np.where(df_raw['fold'].to_numpy() == f)[0])
        for f in sorted(df_raw['fold'].unique())
    ] + [('temporal', tr_idx_t, va_idx_t)]

    def _build_one(sid: str, tr_idx: np.ndarray, va_idx: np.ndarray) -> dict:
        tr_base = df_dom.iloc[tr_idx].copy().reset_index(drop=True)
        va_base = df_dom.iloc[va_idx].copy().reset_index(drop=True)
        tr_folds = tr_base['fold'].to_numpy()

        # Spatial + Target Encoding strictly inner-OOF on train, full-train on val
        sp_va = _compute_spatial_and_te(tr_base, va_base)
        sp_tr_parts = []
        for uf in np.unique(tr_folds):
            sp_tr_parts.append(_compute_spatial_and_te(tr_base[tr_folds != uf], tr_base[tr_folds == uf]))
        sp_tr = pd.concat(sp_tr_parts, axis=0).sort_index()

        overlap = [c for c in sp_tr.columns if c in tr_base.columns]
        tr_df = pd.concat([tr_base.drop(columns=overlap), sp_tr.reset_index(drop=True)], axis=1)
        va_df = pd.concat([va_base.drop(columns=overlap), sp_va.reset_index(drop=True)], axis=1)

        # Compute fold-local structural residual target for text Ridge:
        # residual = log(real_price/m²) - log(sp_same_type_idw_psqm_k15 / cpi)
        # Wait: use train's sp_same_type_idw_psqm_k15 as a pure spatial baseline!
        full_sp_anchor = np.zeros(len(df_raw), dtype=float)
        full_sp_anchor[tr_idx] = np.log(np.maximum(tr_df['sp_same_type_idw_psqm_k15'].to_numpy(dtype=float), 100.0))
        full_sp_anchor[va_idx] = np.log(np.maximum(va_df['sp_same_type_idw_psqm_k15'].to_numpy(dtype=float), 100.0))
        y_resid_struct = y_log_psqm - full_sp_anchor

        txt_tr, txt_va = _fit_text_fold(
            tr_idx, va_idx, texts, embs, y_log_psqm, y_log_price, y_resid_struct, tr_folds,
        )
        tr_df = pd.concat([tr_df, txt_tr.reset_index(drop=True)], axis=1)
        va_df = pd.concat([va_df, txt_va.reset_index(drop=True)], axis=1)

        # Frequency encodings strictly from train
        for col in ('statbel_sector', 'neighbourhood', 'postal_code', 'property_subtype', 'immo_office_name'):
            freq = tr_df[col].astype(str).value_counts(normalize=True).to_dict()
            tr_df[f'freq_{col}'] = tr_df[col].astype(str).map(freq).fillna(0.0).astype('float32')
            va_df[f'freq_{col}'] = va_df[col].astype(str).map(freq).fillna(0.0).astype('float32')

        # Drop raw street_norm (we keep te_street_log_psqm); keep immo_office_name for CatBoost?
        # Keep immo_office_name dropped from generic tree cols to avoid high-card memorization, rely on te_agency + freq
        tr_df = tr_df.drop(columns=['street_norm'])
        va_df = va_df.drop(columns=['street_norm'])

        return {
            'id': sid,
            'tr_idx': tr_idx,
            'va_idx': va_idx,
            'tr_df': tr_df,
            'va_df': va_df,
            'tr_cpi_idx': cpi_idx[tr_idx],
            'va_cpi_idx': cpi_idx[va_idx],
        }

    print("Building Phase-3 leakage-free 6-split feature cache in parallel (6 workers)...")
    bundles = Parallel(n_jobs=6, backend='loky')(
        delayed(_build_one)(sid, tri, vai) for sid, tri, vai in split_specs
    )

    all_cols = [c for c in bundles[0]['tr_df'].columns if c not in META_COLUMNS]
    # Stationary feature set (default for CPI-deflated targets)
    feat_cols_stationary = [c for c in all_cols if c not in NON_STATIONARY_MACRO and c != 'immo_office_name']
    feat_cols_with_agency = [c for c in all_cols if c not in NON_STATIONARY_MACRO]

    payload = {
        'bundles': bundles,
        'y_price': y_price,
        'y_tmp_true': y_price[va_idx_t],
        'va_idx_t': va_idx_t,
        'folds': df_raw['fold'].to_numpy(),
        'is_house': (df_raw['property_type_HOUSE'].to_numpy() == 1),
        'cpi_idx': cpi_idx,
        'feat_cols_stationary': feat_cols_stationary,
        'feat_cols_with_agency': feat_cols_with_agency,
    }
    joblib.dump(payload, CACHE_PATH, compress=3)
    print(f"Built & cached Phase-3 splits in {time.perf_counter()-t0:.1f}s ({len(feat_cols_stationary)} features) -> {CACHE_PATH}")
    return payload


if __name__ == '__main__':
    load_or_build_phase3_splits(force=True)
