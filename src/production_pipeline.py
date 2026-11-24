"""Production Feature Pipeline & Model Bundle Inference (`src/production_pipeline.py`).

Implements the frozen Issue #15 / #16 production contract:
- Agency-blind (drops `immo_office_name`, `agency_norm`, `te_agency_*`, `freq_agency_norm`).
- No Qwen3 / external MLX dependency (uses Word+Char TF-IDF + `intfloat/multilingual-e5-large`).
- Winning `0.5 * CPI + 0.5 * HPI` temporal deflator anchored to 2025 median.
- Strictly leak-free fit/transform separation (`ProductionFeaturePipeline.fit` / `.transform`)
  that produces bit-for-bit identical features on 1-row live inputs and full batches.
- Dual-track Lean-5 ensemble bundle (`tabular_text` always active; `multimodal` activated
  automatically when SigLIP2/SO400M vision embeddings are supplied) with:
  * 5 base models (`lgbmxt_price_hub`, `spec_lgbm_price_hub`, `cb_tuned_price_l2`,
    `tabm_perm170_price_4s`, `tabm_perm170_psqm_4s`)
  * `HuberRegressor` log-EUR stacker/recalibrator
  * Heteroscedastic 80% conformal interval predictor (`sigma_hat(x)` + `q_80`)
  * Aligned TreeSHAP explainer built directly on the served `lgbmxt_price_hub` GBDT
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import HuberRegressor, Ridge, RidgeCV, SGDRegressor

from extract_issue3_features import compute_issue3_features
from phase3_features import (
    META_COLUMNS,
    NON_STATIONARY_MACRO,
    _add_domain_features,
    _extract_text_flags,
)
from phase4_features import CRIME_PATH, FLANDERS_PATH, _normalize_wijk_key
from phase6_features import (
    DROP_W2_BLOATED,
    KEEP_ISSUE3_COLS,
    Q_COMPACT_IDXS,
    Q_COMPACT_NAMES,
    RENDER_COL_NAMES,
    SO_Q_NAMES,
    _inner_cluster_folds,
)
from preprocessing import (
    DEFAULT_MACRO_CACHE_PATH,
    PreprocessingArtifacts,
    _add_geo_features,
    _add_macro_indicators,
    _clean_raw_sentinels,
    transform as preprocess_transform,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PROD_DIR = os.path.join(BASE_DIR, 'outputs', 'models', 'production')
DEFAULT_PREPROC_ARTIFACTS = os.path.join(BASE_DIR, 'outputs', 'dataset', 'preprocessing_artifacts.joblib')
E5_LARGE_MODEL_NAME = 'intfloat/multilingual-e5-large'

AGENCY_COLS = frozenset({
    'immo_office_name',
    'agency_norm',
    'te_agency_log_price',
    'te_agency_log_psqm',
    'freq_agency_norm',
})
QWEN_COLS = frozenset({
    'txt_qwen3_a8_psqm',
    'txt_dense_both_psqm',
    'txt_dense_both_tabresid',
})

# Spatial lookup caches (process-wide, loaded once)
_FLANDERS_TABLE: pd.DataFrame | None = None
_FLANDERS_TREE: cKDTree | None = None
_CRIME_MAP: dict[str, float] | None = None
_CRIME_MEDIAN: float | None = None


def _get_flanders_lookup() -> tuple[pd.DataFrame, cKDTree]:
    """Load Flanders/Wave-2 coord enrichment once and index with cKDTree for nearest-50m lookup."""
    global _FLANDERS_TABLE, _FLANDERS_TREE
    if _FLANDERS_TABLE is None:
        fl = pd.read_parquet(FLANDERS_PATH)
        drop_b = [c for c in DROP_W2_BLOATED if c in fl.columns]
        if drop_b:
            fl = fl.drop(columns=drop_b)
        fl = fl.drop_duplicates(subset=['lat_r5', 'lon_r5']).reset_index(drop=True)
        coords = np.column_stack([fl['lat_r5'].to_numpy(float), fl['lon_r5'].to_numpy(float)])
        _FLANDERS_TABLE = fl
        _FLANDERS_TREE = cKDTree(coords)
    return _FLANDERS_TABLE, _FLANDERS_TREE


def _get_crime_lookup() -> tuple[dict[str, float], float]:
    global _CRIME_MAP, _CRIME_MEDIAN
    if _CRIME_MAP is None:
        if os.path.exists(CRIME_PATH):
            cr = pd.read_parquet(CRIME_PATH)
            _CRIME_MAP = {_normalize_wijk_key(w): float(v) for w, v in zip(cr['wijk'], cr['gent_wijk_crime_3yr'])}
            _CRIME_MEDIAN = float(np.nanmedian(cr['gent_wijk_crime_3yr'].to_numpy(float)))
        else:
            _CRIME_MAP = {}
            _CRIME_MEDIAN = 0.0
    return _CRIME_MAP, float(_CRIME_MEDIAN)


def attach_flanders_and_grb_live(df: pd.DataFrame, max_nn_km: float = 0.25) -> pd.DataFrame:
    """Attach Flanders/Wave-2/GRB features with exact match first, falling back to nearest cached coord (<=250m)."""
    out = df.copy().reset_index(drop=True)
    fl, fl_tree = _get_flanders_lookup()
    fl_feat_cols = [c for c in fl.columns if c not in ('lat_r5', 'lon_r5')]
    # Drop any stale fl/gent/grb cols before re-attaching
    existing_overlap = [c for c in fl_feat_cols if c in out.columns]
    if existing_overlap:
        out = out.drop(columns=existing_overlap)

    lat = pd.to_numeric(out['latitude'], errors='coerce').to_numpy(float)
    lon = pd.to_numeric(out['longitude'], errors='coerce').to_numpy(float)
    lat_r5 = np.round(lat, 5)
    lon_r5 = np.round(lon, 5)

    out['_lat_r5'] = lat_r5
    out['_lon_r5'] = lon_r5
    merged = out.merge(fl, left_on=['_lat_r5', '_lon_r5'], right_on=['lat_r5', 'lon_r5'], how='left')
    merged.drop(columns=['_lat_r5', '_lon_r5', 'lat_r5', 'lon_r5'], inplace=True, errors='ignore')

    # For live coordinates not in the exact training coordinate grid, fill from nearest cached Ghent coord within max_nn_km
    miss_mask = merged[fl_feat_cols[0]].isna().to_numpy() & np.isfinite(lat) & np.isfinite(lon)
    if miss_mask.any():
        miss_idx = np.where(miss_mask)[0]
        q_coords = np.column_stack([lat[miss_idx], lon[miss_idx]])
        dists, nn_idx = fl_tree.query(q_coords, k=1)
        dists_km = np.asarray(dists, dtype=float) * 111.32
        ok = dists_km <= max_nn_km
        if ok.any():
            src_rows = fl.iloc[nn_idx[ok]][fl_feat_cols].reset_index(drop=True)
            dst_idx = miss_idx[ok]
            for col in fl_feat_cols:
                merged.loc[dst_idx, col] = src_rows[col].to_numpy()

    # Crime lookup
    cmap, cmed = _get_crime_lookup()
    if 'neighbourhood' in merged.columns:
        merged['gent_wijk_crime_3yr'] = merged['neighbourhood'].map(
            lambda w: cmap.get(_normalize_wijk_key(w), np.nan)
        )
        merged['gent_wijk_crime_log'] = np.log1p(merged['gent_wijk_crime_3yr'].fillna(cmed))

    # Row-local GRB geometry features (mirrors phase4_features.attach_flanders_and_grb)
    fp = pd.to_numeric(merged.get('grb_bldg_footprint'), errors='coerce')
    pa = pd.to_numeric(merged.get('grb_parcel_area'), errors='coerce')
    la = pd.to_numeric(merged['livable_area'], errors='coerce').clip(lower=10.0)
    ps = pd.to_numeric(merged.get('property_size'), errors='coerce')

    merged['grb_log_parcel_area'] = np.log1p(pa.clip(lower=0).fillna(0.0))
    merged['grb_log_footprint'] = np.log1p(fp.clip(lower=0).fillna(0.0))
    merged['grb_implied_floors'] = np.where(fp > 10.0, np.clip(la / fp, 0.3, 12.0), np.nan)
    merged['grb_open_plot_area'] = np.where((pa > 0) & (fp > 0), np.clip(pa - fp, 0.0, 50000.0), np.nan)
    merged['grb_log_open_plot'] = np.log1p(merged['grb_open_plot_area'].fillna(0.0))
    merged['grb_parcel_vs_reported'] = np.where((pa > 10) & (ps > 10), np.clip(ps / pa, 0.1, 5.0), np.nan)

    is_h = (pd.to_numeric(merged['property_type_HOUSE'], errors='coerce').fillna(0) == 1)
    eff_plot = np.where(is_h, np.where(ps > 10, ps, pa), 0.0)
    merged['grb_effective_house_plot'] = np.log1p(np.where(np.isfinite(eff_plot) & (eff_plot > 0), eff_plot, 0.0))
    merged['grb_is_detached'] = (merged.get('grb_adjoining_bldgs') == 0).astype(np.int8)
    merged['grb_is_semi_detached'] = (merged.get('grb_adjoining_bldgs') == 1).astype(np.int8)

    mobi = pd.to_numeric(merged.get('fl_mobiscore'), errors='coerce').fillna(8.5)
    merged['fl_mobiscore_x_area'] = mobi * np.log(la)
    merged['fl_heritage_x_epc'] = (
        pd.to_numeric(merged.get('fl_is_heritage_inventory'), errors='coerce').fillna(0.0)
        * pd.to_numeric(merged.get('epc_ordinal', 3), errors='coerce').fillna(3.0)
    )
    noise = pd.to_numeric(merged.get('gent_noise_total_lden_db'), errors='coerce')
    merged['gent_has_high_noise_65db'] = (noise >= 65.0).astype(np.int8)
    merged['gent_noise_total_filled'] = noise.fillna(50.0)
    merged['gent_noise_road_filled'] = pd.to_numeric(merged.get('gent_noise_road_lden_db'), errors='coerce').fillna(50.0)
    merged['gent_noise_night_filled'] = pd.to_numeric(merged.get('gent_noise_night_db'), errors='coerce').fillna(45.0)
    return merged


def build_text_strings(df: pd.DataFrame) -> list[str]:
    """Build Dutch-preferred title + description strings matching Phase-6."""
    n = len(df)
    titles = df['title'].fillna('').astype(str).to_numpy() if 'title' in df.columns else np.full(n, '', dtype=object)
    d_nl = df['description_nl'].fillna('').astype(str).to_numpy() if 'description_nl' in df.columns else np.full(n, '', dtype=object)
    d_mn = df['description'].fillna('').astype(str).to_numpy() if 'description' in df.columns else np.full(n, '', dtype=object)
    return [f"{t}. {nl if len(nl) >= 40 else mn}"[:2400] for t, nl, mn in zip(titles, d_nl, d_mn)]


def prepare_base_dataframe(
    raw_df: pd.DataFrame,
    preproc_artifacts: PreprocessingArtifacts,
    texts: list[str] | None = None,
    already_preprocessed: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """Turn parsed raw listing row(s) OR prebuilt training_dataset rows into the Phase-6 base matrix.

    Works identically on 1 row (URL / manual form) or N rows (batch / training).
    """
    df = raw_df.copy().reset_index(drop=True)
    if texts is None:
        texts = build_text_strings(df)

    # Preserve street for street_norm before preprocessing drops raw text columns
    streets = (
        df['street'].fillna('unknown').astype(str).str.strip().str.lower().to_numpy()
        if 'street' in df.columns
        else np.full(len(df), 'unknown', dtype=object)
    )

    if not already_preprocessed:
        # Ensure all raw columns expected by preprocess_transform / _add_domain_features exist
        # (sparse manual forms only supply a handful of fields).
        for col in preproc_artifacts.numeric_medians:
            if col not in df.columns:
                df[col] = np.nan
        for col in (
            'property_size', 'cadastral_income', 'garden_surface', 'terrace_surface',
            'parking_indoor', 'parking_outdoor', 'number_of_bedrooms', 'number_of_rooms',
            'number_of_bathrooms', 'number_of_toilets', 'year_built', 'floor_number',
            'total_floors', 'facade_count', 'street_facade_width', 'monthly_costs',
            'primary_energy_consumption', 'kitchen_surface', 'living_room_surface',
            'energy_label', 'building_condition', 'heating_type', 'kitchen_type',
            'flood_zone_type', 'property_subtype', 'postal_code', 'city',
            'has_garden', 'has_terrace', 'has_lift', 'has_basement', 'has_attic',
            'has_swimming_pool', 'has_double_glazing', 'has_photovoltaic_panels',
            'has_heat_pump', 'is_newly_built', 'renovation_obligation',
        ):
            if col not in df.columns:
                df[col] = np.nan

        # Ensure publication date/month/year exist for live requests
        if 'date_obtained' not in df.columns or df['date_obtained'].isna().all():
            now = pd.Timestamp.now().normalize()
            df['date_obtained'] = now
        else:
            df['date_obtained'] = pd.to_datetime(df['date_obtained'], errors='coerce').fillna(pd.Timestamp.now().normalize())

        df = _clean_raw_sentinels(df)
        df = preprocess_transform(df, preproc_artifacts)
        # Ensure geo + macro columns exist (preprocess_transform already runs _add_geo_features + _add_macro_indicators
        # inside _engineer, so these are already present, but guard for minimal manual rows)
        if 'neighbourhood' not in df.columns or 'statbel_sector' not in df.columns:
            df = _add_geo_features(df)
        if 'belgian_cpi_index' not in df.columns or df['belgian_cpi_index'].isna().any():
            df = _add_macro_indicators(df)
        # Ensure publication_month / publication_year exist (sparse forms lack publication_date)
        now = pd.Timestamp.now().normalize()
        if 'publication_month' not in df.columns:
            df['publication_month'] = float(now.month)
        else:
            df['publication_month'] = pd.to_numeric(df['publication_month'], errors='coerce').fillna(float(now.month))
        if 'publication_year' not in df.columns:
            df['publication_year'] = float(now.year)
        else:
            df['publication_year'] = pd.to_numeric(df['publication_year'], errors='coerce').fillna(float(now.year))
        # Ensure missing-indicator columns expected by _add_domain_features exist
        if 'cadastral_income_missing' not in df.columns:
            df['cadastral_income_missing'] = df['cadastral_income'].isna().astype('int8')

    # Ensure street_norm exists
    pc = df['postal_code'].fillna('0000').astype(str).str.strip() if 'postal_code' in df.columns else pd.Series('0000', index=df.index)
    df['street_norm'] = pc + '|' + pd.Series(streets, index=df.index)

    # Drop raw text columns if still lingering
    for c in ('title', 'description', 'description_nl', 'street', 'immo_office_name', 'agency_norm'):
        if c in df.columns:
            df.drop(columns=[c], inplace=True)

    # Domain features + Flemish text flags + Flanders/Wave-2/GRB + Issue #3 features
    df = _add_domain_features(df)
    txt_flags = _extract_text_flags(texts).reset_index(drop=True)
    df = pd.concat([df.drop(columns=[c for c in txt_flags.columns if c in df.columns]), txt_flags], axis=1)
    df = attach_flanders_and_grb_live(df)

    i3 = compute_issue3_features(df, texts)
    i3_keep = i3[[c for c in KEEP_ISSUE3_COLS if c in i3.columns]].reset_index(drop=True)
    df = pd.concat([df.drop(columns=[c for c in i3_keep.columns if c in df.columns]), i3_keep], axis=1)
    return df, texts


def compute_deflator(df: pd.DataFrame, cpi_ref_2025: float, hpi_ref_2025: float, fallback_deflator: float) -> np.ndarray:
    """Winning Issue #9 deflator: `0.5 * (CPI / CPI_2025) + 0.5 * (HPI / HPI_2025)`."""
    n = len(df)
    cpi = pd.to_numeric(df.get('belgian_cpi_index'), errors='coerce').to_numpy(float) if 'belgian_cpi_index' in df.columns else np.full(n, np.nan)
    hpi = pd.to_numeric(df.get('belgian_hpi_index'), errors='coerce').to_numpy(float) if 'belgian_hpi_index' in df.columns else np.full(n, np.nan)
    d = 0.5 * (cpi / cpi_ref_2025) + 0.5 * (hpi / hpi_ref_2025)
    d = np.where(np.isfinite(d) & (d > 0.5) & (d < 2.0), d, fallback_deflator)
    return d.astype(np.float64)


class SpatialAndTETransformer:
    """Agency-blind spatial IDW KNN + empirical-Bayes target encoder (fit once, transform 1..N rows)."""

    def fit(self, ref_df: pd.DataFrame) -> 'SpatialAndTETransformer':
        ref_valid = ref_df[
            ref_df['latitude'].notna()
            & ref_df['longitude'].notna()
            & ref_df['price'].notna()
            & (ref_df['livable_area'] > 0)
        ].reset_index(drop=True)

        self.ref_psqm = (ref_valid['price'] / ref_valid['livable_area']).to_numpy(dtype=np.float64)
        self.ref_log_psqm = np.log(self.ref_psqm)
        self.ref_area = ref_valid['livable_area'].to_numpy(dtype=np.float64)
        self.ref_price = ref_valid['price'].to_numpy(dtype=np.float64)
        self.ref_is_h = (ref_valid['property_type_HOUSE'].to_numpy() == 1)
        self.ref_coords = np.column_stack([ref_valid['latitude'].to_numpy(float), ref_valid['longitude'].to_numpy(float)])

        self.g_med = float(np.median(self.ref_psqm))
        self.g_log_med = float(np.median(self.ref_log_psqm))
        self.g_price_med = float(np.median(self.ref_price))

        self.tree_all = cKDTree(self.ref_coords)
        self.type_trees: dict[bool, tuple[cKDTree, np.ndarray, np.ndarray]] = {}
        for h_flag in (True, False):
            m = self.ref_is_h == h_flag
            if m.any():
                self.type_trees[h_flag] = (cKDTree(self.ref_coords[m]), self.ref_psqm[m], self.ref_area[m])

        # Empirical-Bayes smoothed maps (agency-blind: NO immo_office_name!)
        self.te_maps: dict[str, tuple[dict[str, float], float]] = {}
        for col, tgt_vals, prior_val, m_prior, out_name in (
            ('street_norm', self.ref_log_psqm, self.g_log_med, 4.0, 'te_street_log_psqm'),
            ('property_subtype', self.ref_log_psqm, self.g_log_med, 5.0, 'te_subtype_log_psqm'),
        ):
            if col in ref_valid.columns:
                tr_g = pd.DataFrame({'k': ref_valid[col].fillna('Unknown').astype(str).to_numpy(), 'v': tgt_vals})
                agg = tr_g.groupby('k')['v'].agg(['count', 'mean'])
                smooth = {
                    idx: float((row['count'] * row['mean'] + m_prior * prior_val) / (row['count'] + m_prior))
                    for idx, row in agg.iterrows() if idx not in ('Unknown', '', 'None', 'nan')
                }
            else:
                smooth = {}
            self.te_maps[out_name] = (smooth, prior_val)

        # Type x group median €/m² maps
        self.type_grp_maps: dict[str, tuple[str, dict[tuple[int, str], float], dict[int, float]]] = {}
        for grp_col, name, m_prior in (
            ('neighbourhood', 'sp_type_wijk_psqm', 5.0),
            ('statbel_sector', 'sp_type_sector_psqm', 6.0),
            ('postal_code', 'sp_type_postal_psqm', 5.0),
        ):
            tr_g = pd.DataFrame({
                'h': self.ref_is_h.astype(int),
                'g': ref_valid[grp_col].fillna('Unknown').astype(str).to_numpy() if grp_col in ref_valid.columns else 'Unknown',
                'p': self.ref_psqm,
            })
            type_med = {int(k): float(v) for k, v in tr_g.groupby('h')['p'].median().to_dict().items()}
            agg = tr_g.groupby(['h', 'g'])['p'].agg(['count', 'median'])
            smooth = {
                (int(h), str(g)): float((row['count'] * row['median'] + m_prior * type_med.get(int(h), self.g_med)) / (row['count'] + m_prior))
                for (h, g), row in agg.iterrows() if g != 'Unknown'
            }
            self.type_grp_maps[name] = (grp_col, smooth, type_med)

        # Also precompute overall neighbourhood & sector medians for `oof_neighbourhood_median_price_sqm` / `oof_sector_median_price_sqm`
        self.wijk_med_map: dict[str, float] = {}
        if 'neighbourhood' in ref_valid.columns:
            self.wijk_med_map = {
                str(k): float(v)
                for k, v in pd.DataFrame({'w': ref_valid['neighbourhood'].astype(str), 'p': self.ref_psqm}).groupby('w')['p'].median().to_dict().items()
            }
        self.sector_med_map: dict[str, float] = {}
        if 'statbel_sector' in ref_valid.columns:
            agg = pd.DataFrame({'s': ref_valid['statbel_sector'].astype(str), 'p': self.ref_psqm}).groupby('s')['p'].agg(['count', 'median'])
            self.sector_med_map = {
                str(idx): float((row['count'] * row['median'] + 5.0 * self.g_med) / (row['count'] + 5.0))
                for idx, row in agg.iterrows() if str(idx) != 'Unknown'
            }

        # Frequency maps (agency-blind)
        self.freq_maps: dict[str, dict[str, float]] = {}
        for col in ('statbel_sector', 'neighbourhood', 'postal_code', 'property_subtype'):
            if col in ref_valid.columns:
                vc = ref_valid[col].fillna('Missing').astype(str).value_counts()
                self.freq_maps[col] = {str(k): float(v) for k, v in vc.to_dict().items()}
            else:
                self.freq_maps[col] = {}

        return self

    def transform(self, target_df: pd.DataFrame) -> pd.DataFrame:
        n = len(target_df)
        out = pd.DataFrame(index=target_df.index)
        lat = pd.to_numeric(target_df['latitude'], errors='coerce').fillna(51.0531).to_numpy(float)
        lon = pd.to_numeric(target_df['longitude'], errors='coerce').fillna(3.7265).to_numpy(float)
        tgt_coords = np.column_stack([lat, lon])
        tgt_is_h = (pd.to_numeric(target_df['property_type_HOUSE'], errors='coerce').fillna(0).to_numpy() == 1)
        tgt_area = np.maximum(pd.to_numeric(target_df['livable_area'], errors='coerce').fillna(100.0).to_numpy(float), 10.0)

        k_max = min(40, len(self.ref_psqm))
        dists_all, idxs_all = self.tree_all.query(tgt_coords, k=k_max)
        if k_max == 1:
            dists_all = dists_all[:, None]
            idxs_all = idxs_all[:, None]
        dists_km = dists_all * 111.32
        eps = 0.05

        for k in (5, 15, 40):
            kk = min(k, k_max)
            w = 1.0 / (dists_km[:, :kk] + eps)
            w = w / w.sum(axis=1, keepdims=True)
            nbr_psqm = self.ref_psqm[idxs_all[:, :kk]]
            out[f'sp_idw_psqm_k{k}'] = np.sum(w * nbr_psqm, axis=1)
            if k == 15:
                out['sp_psqm_std_k15'] = np.std(nbr_psqm, axis=1)
                out['sp_psqm_p25_k15'] = np.percentile(nbr_psqm, 25, axis=1)
                out['sp_psqm_p75_k15'] = np.percentile(nbr_psqm, 75, axis=1)
                local_med_area = np.median(self.ref_area[idxs_all[:, :kk]], axis=1)
                out['sp_rel_area_k15'] = tgt_area / np.maximum(local_med_area, 10.0)
                out['sp_idw_price_k15'] = np.sum(w * self.ref_price[idxs_all[:, :kk]], axis=1)

        out['sp_dist_3nn_km'] = np.mean(dists_km[:, :min(3, k_max)], axis=1)
        out['sp_dist_10nn_km'] = np.mean(dists_km[:, :min(10, k_max)], axis=1)

        same_k5 = np.full(n, self.g_med, dtype=float)
        same_k15 = np.full(n, self.g_med, dtype=float)
        same_k40 = np.full(n, self.g_med, dtype=float)
        same_med_area = np.full(n, 120.0, dtype=float)
        same_dist5 = np.full(n, 0.5, dtype=float)

        for h_flag in (True, False):
            t_mask = tgt_is_h == h_flag
            if not t_mask.any() or h_flag not in self.type_trees:
                continue
            t_tree, r_p, r_a = self.type_trees[h_flag]
            km = min(40, len(r_p))
            d, ix = t_tree.query(tgt_coords[t_mask], k=km)
            if km == 1:
                d = d[:, None]; ix = ix[:, None]
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

        # TE maps
        for col, out_name in (('street_norm', 'te_street_log_psqm'), ('property_subtype', 'te_subtype_log_psqm')):
            smooth, prior_val = self.te_maps[out_name]
            vals = target_df[col].fillna('Unknown').astype(str) if col in target_df.columns else pd.Series('Unknown', index=target_df.index)
            out[out_name] = np.array([smooth.get(x, prior_val) for x in vals], dtype=float)

        for name, (grp_col, smooth, type_med) in self.type_grp_maps.items():
            vals = target_df[grp_col].fillna('Unknown').astype(str) if grp_col in target_df.columns else pd.Series('Unknown', index=target_df.index)
            out[name] = [smooth.get((int(h), g), type_med.get(int(h), self.g_med)) for h, g in zip(tgt_is_h, vals)]

        out['oof_knn10_median_price_sqm'] = np.median(self.ref_psqm[idxs_all[:, :min(10, k_max)]], axis=1)
        out['oof_dist_5nn_km'] = np.mean(dists_km[:, :min(5, k_max)], axis=1)

        if 'neighbourhood' in target_df.columns:
            out['oof_neighbourhood_median_price_sqm'] = [
                self.wijk_med_map.get(str(w), self.g_med) for w in target_df['neighbourhood'].fillna('Unknown')
            ]
        else:
            out['oof_neighbourhood_median_price_sqm'] = self.g_med

        if 'statbel_sector' in target_df.columns:
            out['oof_sector_median_price_sqm'] = [
                self.sector_med_map.get(str(s), self.g_med) for s in target_df['statbel_sector'].fillna('Unknown')
            ]
        else:
            out['oof_sector_median_price_sqm'] = self.g_med

        for col, fmap in self.freq_maps.items():
            vals = target_df[col].fillna('Missing').astype(str) if col in target_df.columns else pd.Series('Missing', index=target_df.index)
            out[f'freq_{col}'] = vals.map(fmap).fillna(0.0).astype(np.float32)

        return out


class ProductionFeaturePipeline:
    """Leak-free fit/transform feature factory matching Phase-6 (agency-blind, no Qwen3)."""

    def __init__(self, preproc_artifacts: PreprocessingArtifacts):
        self.preproc_artifacts = preproc_artifacts
        self.cpi_ref_2025: float = 1.0
        self.hpi_ref_2025: float = 1.0
        self.latest_deflator: float = 1.0
        self.latest_macro_row: dict[str, float] = {}
        self.spatial_te: SpatialAndTETransformer | None = None
        self.anchor_cols: list[str] = []
        self.anchor_cat_levels: dict[str, list[str]] = {}
        self.anchor_model: Any = None
        self.vec_w: TfidfVectorizer | None = None
        self.vec_c: TfidfVectorizer | None = None
        self.text_models: dict[str, Any] = {}
        self.pca_e5l: PCA | None = None
        self.svd_tfidf: TruncatedSVD | None = None
        self.vis_models: dict[str, Any] = {}
        self.pca_vis: PCA | None = None
        self.has_vision_fitted: bool = False
        self.feature_columns_tabtxt: list[str] = []
        self.feature_columns_multimodal: list[str] = []

    @staticmethod
    def _prep_anchor_df(df: pd.DataFrame, cols: list[str], cat_levels: dict[str, list[str]] | None = None) -> tuple[pd.DataFrame, dict[str, list[str]]]:
        levels = {} if cat_levels is None else cat_levels
        out_cols: dict[str, Any] = {}
        n = len(df)
        for c in cols:
            if cat_levels is not None:
                # Transform mode: strictly follow learned categorical vs numeric schema!
                if c in cat_levels:
                    cats = cat_levels[c]
                    sv = df[c].fillna('Missing').astype(str) if c in df.columns else pd.Series('Missing', index=df.index)
                    sv = sv.where(sv.isin(cats), 'Missing')
                    out_cols[c] = pd.Categorical(sv, categories=cats)
                else:
                    if c in df.columns:
                        s = df[c]
                        if pd.api.types.is_bool_dtype(s):
                            out_cols[c] = s.astype('int8').to_numpy()
                        else:
                            out_cols[c] = pd.to_numeric(s, errors='coerce').astype('float32').to_numpy()
                    else:
                        out_cols[c] = np.full(n, np.nan, dtype=np.float32)
                continue

            # Fit mode: infer categorical vs numeric schema from training frame
            if c not in df.columns:
                out_cols[c] = np.full(n, np.nan, dtype=np.float32)
                continue
            s = df[c]
            if pd.api.types.is_bool_dtype(s):
                out_cols[c] = s.astype('int8').to_numpy()
            elif not pd.api.types.is_numeric_dtype(s):
                sv = s.fillna('Missing').astype(str)
                cats = list(pd.Index(sv.unique()).union(pd.Index(['Missing'])))
                levels[c] = cats
                out_cols[c] = pd.Categorical(sv, categories=cats)
            else:
                out_cols[c] = pd.to_numeric(s, errors='coerce').astype('float32').to_numpy()
        return pd.DataFrame(out_cols, index=df.index), levels

    def fit(
        self,
        base_df: pd.DataFrame,
        texts: list[str],
        e5l: np.ndarray,
        cluster_ids: np.ndarray,
        vis_bundle: dict[str, np.ndarray] | None = None,
    ) -> pd.DataFrame:
        """Fit all transformers on training data and return strictly inner-OOF training matrix."""
        import lightgbm as lgb

        df = base_df.copy().reset_index(drop=True)
        n = len(df)
        inner_f = _inner_cluster_folds(cluster_ids, n_inner=5, seed=42)

        # 1. Deflator reference (2025 median) + latest macro defaults for live inference
        cpi_raw = pd.to_numeric(df['belgian_cpi_index'], errors='coerce').to_numpy(float)
        hpi_raw = pd.to_numeric(df['belgian_hpi_index'], errors='coerce').to_numpy(float)
        m25 = (df['publication_year'] == 2025).to_numpy()
        self.cpi_ref_2025 = float(np.nanmedian(cpi_raw[m25]))
        self.hpi_ref_2025 = float(np.nanmedian(hpi_raw[m25]))
        deflator = compute_deflator(df, self.cpi_ref_2025, self.hpi_ref_2025, 1.0)
        self.latest_deflator = float(np.nanmedian(deflator[df['publication_year'] == df['publication_year'].max()]))

        if os.path.exists(DEFAULT_MACRO_CACHE_PATH):
            ecb = pd.read_csv(DEFAULT_MACRO_CACHE_PATH).sort_values('year_month')
            last = ecb.iloc[-1].to_dict()
            self.latest_macro_row = {k: float(v) for k, v in last.items() if k != 'year_month' and pd.notna(v)}
            if 'belgian_cpi_index' in self.latest_macro_row and 'belgian_hpi_index' in self.latest_macro_row:
                self.latest_deflator = float(
                    0.5 * (self.latest_macro_row['belgian_cpi_index'] / self.cpi_ref_2025)
                    + 0.5 * (self.latest_macro_row['belgian_hpi_index'] / self.hpi_ref_2025)
                )

        # 2. Spatial + TE: fit full transformer for inference, compute strictly inner-OOF for training rows
        self.spatial_te = SpatialAndTETransformer().fit(df)
        sp_parts = []
        for uf in np.unique(inner_f):
            st_in = SpatialAndTETransformer().fit(df.iloc[inner_f != uf])
            sp_parts.append(st_in.transform(df.iloc[inner_f == uf]))
        tr_sp = pd.concat(sp_parts, axis=0).sort_index().reset_index(drop=True)
        tr_base = pd.concat([df.drop(columns=[c for c in tr_sp.columns if c in df.columns]), tr_sp], axis=1)

        # 3. Targets
        y_price = df['price'].to_numpy(float)
        area = df['livable_area'].to_numpy(float)
        tr_y_psqm = np.log((y_price / deflator) / area)
        tr_y_price = np.log(y_price / deflator)
        tr_sp_anchor = np.log(np.maximum(tr_sp['sp_same_type_idw_psqm_k15'].to_numpy(float), 500.0))
        tr_y_sp_resid = tr_y_psqm - tr_sp_anchor

        # 4. Structured-only LightGBM anchor (inner-OOF for train, full-fit for inference)
        drop_pfx = ('txt_', 'e5_', 'e5l_', 'tfidf_', 'qwen3_', 'vis_')
        drop_ex = set(META_COLUMNS) | set(NON_STATIONARY_MACRO) | AGENCY_COLS | {'cluster_id', 'fold_cluster', 'street_norm'}
        self.anchor_cols = [c for c in tr_base.columns if c not in drop_ex and not c.startswith(drop_pfx)]
        X_anc, self.anchor_cat_levels = self._prep_anchor_df(tr_base, self.anchor_cols)
        anc_params = dict(
            objective='huber', alpha=0.15, n_estimators=450, learning_rate=0.04,
            num_leaves=47, min_child_samples=30, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.5, reg_alpha=1.0, reg_lambda=3.0, random_state=42, n_jobs=4, verbose=-1,
        )
        tr_tab_hat = np.zeros(n, dtype=np.float32)
        for uf in np.unique(inner_f):
            m_tr, m_va = inner_f != uf, inner_f == uf
            tr_tab_hat[m_va] = lgb.LGBMRegressor(**anc_params).fit(X_anc.iloc[m_tr], tr_y_psqm[m_tr]).predict(X_anc.iloc[m_va])
        self.anchor_model = lgb.LGBMRegressor(**anc_params).fit(X_anc, tr_y_psqm)
        tr_r_tab = tr_y_psqm - tr_tab_hat

        # 5. Text vectorizers + Ridge/Huber/PCA/SVD (NO Qwen3!)
        self.vec_w = TfidfVectorizer(
            max_features=35000, ngram_range=(1, 2), min_df=4, max_df=0.90, sublinear_tf=True,
            token_pattern=r'(?u)\b[a-zA-Z0-9éèêëàâïîôùûç€²]{2,}\b',
        )
        self.vec_c = TfidfVectorizer(
            analyzer='char_wb', ngram_range=(3, 5), max_features=30000, min_df=10, max_df=0.85, sublinear_tf=True,
        )
        X_tf = hstack([self.vec_w.fit_transform(texts), self.vec_c.fit_transform(texts)]).tocsr()
        e5l = np.asarray(e5l, dtype=np.float32)

        t_names = [
            'txt_tfidf_a2_psqm', 'txt_tfidf_a2_price', 'txt_tfidf_a14_psqm', 'txt_tfidf_huber_psqm',
            'txt_tfidf_tabresid', 'txt_tfidf_spresid',
            'txt_e5l_a5_psqm', 'txt_e5l_a5_price', 'txt_e5l_tabresid',
        ]
        tr_meta: dict[str, np.ndarray] = {k: np.zeros(n, dtype=np.float32) for k in t_names}

        for uf in np.unique(inner_f):
            m_tr, m_va = inner_f != uf, inner_f == uf
            tr_meta['txt_tfidf_a2_psqm'][m_va] = Ridge(alpha=2.5, solver='lsqr', max_iter=200).fit(X_tf[m_tr], tr_y_psqm[m_tr]).predict(X_tf[m_va])
            tr_meta['txt_tfidf_a2_price'][m_va] = Ridge(alpha=2.5, solver='lsqr', max_iter=200).fit(X_tf[m_tr], tr_y_price[m_tr]).predict(X_tf[m_va])
            tr_meta['txt_tfidf_a14_psqm'][m_va] = Ridge(alpha=14.0, solver='lsqr', max_iter=200).fit(X_tf[m_tr], tr_y_psqm[m_tr]).predict(X_tf[m_va])
            tr_meta['txt_tfidf_huber_psqm'][m_va] = SGDRegressor(loss='huber', epsilon=0.15, alpha=1e-4, max_iter=60, tol=1e-4, random_state=42).fit(X_tf[m_tr], tr_y_psqm[m_tr]).predict(X_tf[m_va])
            tr_meta['txt_tfidf_tabresid'][m_va] = Ridge(alpha=6.0, solver='lsqr', max_iter=200).fit(X_tf[m_tr], tr_r_tab[m_tr]).predict(X_tf[m_va])
            tr_meta['txt_tfidf_spresid'][m_va] = Ridge(alpha=2.5, solver='lsqr', max_iter=200).fit(X_tf[m_tr], tr_y_sp_resid[m_tr]).predict(X_tf[m_va])
            tr_meta['txt_e5l_a5_psqm'][m_va] = Ridge(alpha=5.0).fit(e5l[m_tr], tr_y_psqm[m_tr]).predict(e5l[m_va])
            tr_meta['txt_e5l_a5_price'][m_va] = Ridge(alpha=5.0).fit(e5l[m_tr], tr_y_price[m_tr]).predict(e5l[m_va])
            tr_meta['txt_e5l_tabresid'][m_va] = Ridge(alpha=12.0).fit(e5l[m_tr], tr_r_tab[m_tr]).predict(e5l[m_va])

        self.text_models = {
            'txt_tfidf_a2_psqm': Ridge(alpha=2.5, solver='lsqr', max_iter=200).fit(X_tf, tr_y_psqm),
            'txt_tfidf_a2_price': Ridge(alpha=2.5, solver='lsqr', max_iter=200).fit(X_tf, tr_y_price),
            'txt_tfidf_a14_psqm': Ridge(alpha=14.0, solver='lsqr', max_iter=200).fit(X_tf, tr_y_psqm),
            'txt_tfidf_huber_psqm': SGDRegressor(loss='huber', epsilon=0.15, alpha=1e-4, max_iter=60, tol=1e-4, random_state=42).fit(X_tf, tr_y_psqm),
            'txt_tfidf_tabresid': Ridge(alpha=6.0, solver='lsqr', max_iter=200).fit(X_tf, tr_r_tab),
            'txt_tfidf_spresid': Ridge(alpha=2.5, solver='lsqr', max_iter=200).fit(X_tf, tr_y_sp_resid),
            'txt_e5l_a5_psqm': Ridge(alpha=5.0).fit(e5l, tr_y_psqm),
            'txt_e5l_a5_price': Ridge(alpha=5.0).fit(e5l, tr_y_price),
            'txt_e5l_tabresid': Ridge(alpha=12.0).fit(e5l, tr_r_tab),
        }

        tr_meta['tab_anchor_log_psqm'] = tr_tab_hat
        tr_meta['txt_blend_psqm'] = (0.60 * tr_meta['txt_tfidf_a2_psqm'] + 0.40 * tr_meta['txt_e5l_a5_psqm']).astype(np.float32)
        tr_meta['txt_super_tabresid'] = (0.60 * tr_meta['txt_tfidf_tabresid'] + 0.40 * tr_meta['txt_e5l_tabresid']).astype(np.float32)
        tr_meta['txt_tab_plus_text_psqm'] = (tr_meta['tab_anchor_log_psqm'] + tr_meta['txt_super_tabresid']).astype(np.float32)
        tr_meta['txt_tfidf_minus_e5l_psqm'] = (tr_meta['txt_tfidf_a2_psqm'] - tr_meta['txt_e5l_a5_psqm']).astype(np.float32)

        self.pca_e5l = PCA(n_components=24, random_state=42).fit(e5l)
        tr_pe = self.pca_e5l.transform(e5l).astype(np.float32)
        for j in range(24):
            tr_meta[f'e5l_pca_{j}'] = tr_pe[:, j]

        self.svd_tfidf = TruncatedSVD(n_components=16, random_state=42).fit(X_tf)
        tr_st = self.svd_tfidf.transform(X_tf).astype(np.float32)
        for j in range(16):
            tr_meta[f'tfidf_svd_{j}'] = tr_st[:, j]

        # 6. Optional vision meta-features (SigLIP2-base + SO400M-384)
        if vis_bundle is not None:
            sig = np.nan_to_num(vis_bundle['sig_real'].astype(np.float32), nan=0.0)
            so = np.nan_to_num(vis_bundle['so_sheet'].astype(np.float32), nan=0.0)
            ren = vis_bundle['render_feats'].astype(np.float32)
            qf = vis_bundle['q_feats'].astype(np.float32)
            soq = vis_bundle['so_q_feats'].astype(np.float32)
            hv = vis_bundle['has_vis'].astype(bool)

            tr_r_tabtxt = tr_y_psqm - tr_meta['txt_tab_plus_text_psqm']
            alphas = np.logspace(0.5, 4.5, 13)
            for vc in ('vis_sig_psqm', 'vis_sig_tabtxt_resid', 'vis_so_psqm', 'vis_so_tabtxt_resid'):
                tr_meta[vc] = np.full(n, np.nan, dtype=np.float32)

            for uf in np.unique(inner_f):
                m_tr = (inner_f != uf) & hv
                m_va = (inner_f == uf) & hv
                if m_tr.sum() > 50 and m_va.sum() > 0:
                    tr_meta['vis_sig_psqm'][m_va] = RidgeCV(alphas=alphas).fit(sig[m_tr], tr_y_psqm[m_tr]).predict(sig[m_va])
                    tr_meta['vis_sig_tabtxt_resid'][m_va] = RidgeCV(alphas=alphas).fit(sig[m_tr], tr_r_tabtxt[m_tr]).predict(sig[m_va])
                    tr_meta['vis_so_psqm'][m_va] = RidgeCV(alphas=alphas).fit(so[m_tr], tr_y_psqm[m_tr]).predict(so[m_va])
                    tr_meta['vis_so_tabtxt_resid'][m_va] = RidgeCV(alphas=alphas).fit(so[m_tr], tr_r_tabtxt[m_tr]).predict(so[m_va])

            self.vis_models = {
                'vis_sig_psqm': RidgeCV(alphas=alphas).fit(sig[hv], tr_y_psqm[hv]),
                'vis_sig_tabtxt_resid': RidgeCV(alphas=alphas).fit(sig[hv], tr_r_tabtxt[hv]),
                'vis_so_psqm': RidgeCV(alphas=alphas).fit(so[hv], tr_y_psqm[hv]),
                'vis_so_tabtxt_resid': RidgeCV(alphas=alphas).fit(so[hv], tr_r_tabtxt[hv]),
            }
            tr_meta['vis_blend_psqm'] = (0.55 * tr_meta['vis_sig_psqm'] + 0.45 * tr_meta['vis_so_psqm']).astype(np.float32)
            tr_meta['vis_blend_tabtxt_resid'] = (0.55 * tr_meta['vis_sig_tabtxt_resid'] + 0.45 * tr_meta['vis_so_tabtxt_resid']).astype(np.float32)

            self.pca_vis = PCA(n_components=8, random_state=42).fit(sig[hv])
            tr_pv = np.full((n, 8), np.nan, dtype=np.float32)
            tr_pv[hv] = self.pca_vis.transform(sig[hv]).astype(np.float32)
            for j in range(8):
                tr_meta[f'vis_sig_pca_{j}'] = tr_pv[:, j]

            for j, cname in enumerate(RENDER_COL_NAMES):
                tr_meta[cname] = ren[:, j]
            for j, cname in zip(Q_COMPACT_IDXS, Q_COMPACT_NAMES):
                tr_meta[cname] = qf[:, j]
            for j, cname in enumerate(SO_Q_NAMES):
                tr_meta[cname] = soq[:, j]
            tr_meta['vis_has_photos'] = hv.astype(np.int8)
            self.has_vision_fitted = True

        out_df = pd.concat([tr_base, pd.DataFrame(tr_meta)], axis=1)
        for drop_c in ('street_norm', 'cluster_id', 'fold_cluster'):
            if drop_c in out_df.columns:
                out_df.drop(columns=[drop_c], inplace=True)
        return out_df

    def transform(
        self,
        base_df: pd.DataFrame,
        texts: list[str],
        e5l: np.ndarray,
        vis_bundle: dict[str, np.ndarray] | None = None,
    ) -> pd.DataFrame:
        """Transform 1..N base rows deterministically using fitted transformers (zero leakage)."""
        assert self.spatial_te is not None and self.anchor_model is not None
        df = base_df.copy().reset_index(drop=True)
        n = len(df)

        # Fill latest macro defaults if any macro column is missing/NaN (e.g. live URL / manual form)
        for col, val in self.latest_macro_row.items():
            if col not in df.columns:
                df[col] = val
            elif df[col].isna().any():
                df[col] = df[col].fillna(val)

        sp = self.spatial_te.transform(df).reset_index(drop=True)
        feat_base = pd.concat([df.drop(columns=[c for c in sp.columns if c in df.columns]), sp], axis=1)

        X_anc, _ = self._prep_anchor_df(feat_base, self.anchor_cols, self.anchor_cat_levels)
        tab_hat = self.anchor_model.predict(X_anc).astype(np.float32)

        X_tf = hstack([self.vec_w.transform(texts), self.vec_c.transform(texts)]).tocsr()
        e5l = np.asarray(e5l, dtype=np.float32)
        if e5l.ndim == 1:
            e5l = e5l.reshape(1, -1)

        meta: dict[str, np.ndarray] = {
            'txt_tfidf_a2_psqm': self.text_models['txt_tfidf_a2_psqm'].predict(X_tf).astype(np.float32),
            'txt_tfidf_a2_price': self.text_models['txt_tfidf_a2_price'].predict(X_tf).astype(np.float32),
            'txt_tfidf_a14_psqm': self.text_models['txt_tfidf_a14_psqm'].predict(X_tf).astype(np.float32),
            'txt_tfidf_huber_psqm': self.text_models['txt_tfidf_huber_psqm'].predict(X_tf).astype(np.float32),
            'txt_tfidf_tabresid': self.text_models['txt_tfidf_tabresid'].predict(X_tf).astype(np.float32),
            'txt_tfidf_spresid': self.text_models['txt_tfidf_spresid'].predict(X_tf).astype(np.float32),
            'txt_e5l_a5_psqm': self.text_models['txt_e5l_a5_psqm'].predict(e5l).astype(np.float32),
            'txt_e5l_a5_price': self.text_models['txt_e5l_a5_price'].predict(e5l).astype(np.float32),
            'txt_e5l_tabresid': self.text_models['txt_e5l_tabresid'].predict(e5l).astype(np.float32),
            'tab_anchor_log_psqm': tab_hat,
        }
        meta['txt_blend_psqm'] = (0.60 * meta['txt_tfidf_a2_psqm'] + 0.40 * meta['txt_e5l_a5_psqm']).astype(np.float32)
        meta['txt_super_tabresid'] = (0.60 * meta['txt_tfidf_tabresid'] + 0.40 * meta['txt_e5l_tabresid']).astype(np.float32)
        meta['txt_tab_plus_text_psqm'] = (meta['tab_anchor_log_psqm'] + meta['txt_super_tabresid']).astype(np.float32)
        meta['txt_tfidf_minus_e5l_psqm'] = (meta['txt_tfidf_a2_psqm'] - meta['txt_e5l_a5_psqm']).astype(np.float32)

        pe = self.pca_e5l.transform(e5l).astype(np.float32)
        for j in range(24):
            meta[f'e5l_pca_{j}'] = pe[:, j]
        st = self.svd_tfidf.transform(X_tf).astype(np.float32)
        for j in range(16):
            meta[f'tfidf_svd_{j}'] = st[:, j]

        if self.has_vision_fitted and vis_bundle is not None:
            sig = np.nan_to_num(vis_bundle['sig_real'].astype(np.float32), nan=0.0)
            so = np.nan_to_num(vis_bundle['so_sheet'].astype(np.float32), nan=0.0)
            ren = vis_bundle['render_feats'].astype(np.float32)
            qf = vis_bundle['q_feats'].astype(np.float32)
            soq = vis_bundle['so_q_feats'].astype(np.float32)
            hv = vis_bundle['has_vis'].astype(bool)

            for vc in ('vis_sig_psqm', 'vis_sig_tabtxt_resid', 'vis_so_psqm', 'vis_so_tabtxt_resid'):
                arr = np.full(n, np.nan, dtype=np.float32)
                if hv.any():
                    X_in = sig[hv] if 'sig' in vc else so[hv]
                    arr[hv] = self.vis_models[vc].predict(X_in).astype(np.float32)
                meta[vc] = arr

            meta['vis_blend_psqm'] = (0.55 * meta['vis_sig_psqm'] + 0.45 * meta['vis_so_psqm']).astype(np.float32)
            meta['vis_blend_tabtxt_resid'] = (0.55 * meta['vis_sig_tabtxt_resid'] + 0.45 * meta['vis_so_tabtxt_resid']).astype(np.float32)

            pv = np.full((n, 8), np.nan, dtype=np.float32)
            if hv.any():
                pv[hv] = self.pca_vis.transform(sig[hv]).astype(np.float32)
            for j in range(8):
                meta[f'vis_sig_pca_{j}'] = pv[:, j]

            for j, cname in enumerate(RENDER_COL_NAMES):
                meta[cname] = ren[:, j]
            for j, cname in zip(Q_COMPACT_IDXS, Q_COMPACT_NAMES):
                meta[cname] = qf[:, j]
            for j, cname in enumerate(SO_Q_NAMES):
                meta[cname] = soq[:, j]
            meta['vis_has_photos'] = hv.astype(np.int8)
        elif self.has_vision_fitted:
            meta['vis_has_photos'] = np.zeros(n, dtype=np.int8)

        out_df = pd.concat([feat_base, pd.DataFrame(meta)], axis=1)
        for drop_c in ('street_norm', 'cluster_id', 'fold_cluster'):
            if drop_c in out_df.columns:
                out_df.drop(columns=[drop_c], inplace=True)
        return out_df


@dataclass
class NNPreprocSpec:
    """Deterministic TabM preprocessing learned on train and applied identically at inference."""
    cols: list[str]
    cat_keep: dict[str, list[str]]
    num_medians: dict[str, float]
    bool_cols: list[str]

    @classmethod
    def fit(cls, df: pd.DataFrame, cols: list[str], max_cat: int = 40) -> 'NNPreprocSpec':
        cat_keep: dict[str, list[str]] = {}
        num_medians: dict[str, float] = {}
        bool_cols: list[str] = []
        for c in cols:
            if c not in df.columns:
                num_medians[c] = 0.0
                continue
            s = df[c]
            if pd.api.types.is_bool_dtype(s):
                bool_cols.append(c)
            elif not pd.api.types.is_numeric_dtype(s):
                vc = s.astype(str).fillna('Missing').value_counts()
                keep = list(vc[vc >= 5].head(max_cat).index)
                if '__RARE__' not in keep:
                    keep.append('__RARE__')
                cat_keep[c] = keep
            else:
                tv = pd.to_numeric(s, errors='coerce').replace([np.inf, -np.inf], np.nan)
                num_medians[c] = float(tv.median()) if tv.notna().any() else 0.0
        return cls(cols=list(cols), cat_keep=cat_keep, num_medians=num_medians, bool_cols=bool_cols)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out_cols: dict[str, Any] = {}
        n = len(df)
        for c in self.cols:
            if c in self.bool_cols:
                vals = df[c] if c in df.columns else False
                out_cols[c] = pd.Series(vals, index=df.index).fillna(False).astype('float32').to_numpy()
            elif c in self.cat_keep:
                keep = self.cat_keep[c]
                ks = set(keep)
                sv = df[c].astype(str).fillna('Missing') if c in df.columns else pd.Series('Missing', index=df.index)
                sv = sv.where(sv.isin(ks), '__RARE__')
                out_cols[c] = pd.Categorical(sv, categories=keep)
            else:
                med = self.num_medians.get(c, 0.0)
                if c in df.columns:
                    vv = pd.to_numeric(df[c], errors='coerce').replace([np.inf, -np.inf], np.nan)
                    out_cols[c] = vv.fillna(med).astype('float32').to_numpy()
                else:
                    out_cols[c] = np.full(n, med, dtype=np.float32)
        return pd.DataFrame(out_cols, index=df.index)


@dataclass
class TreePreprocSpec:
    """Deterministic GBDT column schema & category levels."""
    cols: list[str]
    cat_cols: list[str]
    cat_levels: dict[str, list[str]]
    bool_cols: list[str]

    @classmethod
    def fit(cls, df: pd.DataFrame, cols: list[str]) -> 'TreePreprocSpec':
        cat_cols: list[str] = []
        cat_levels: dict[str, list[str]] = {}
        bool_cols: list[str] = []
        for c in cols:
            if c not in df.columns:
                continue
            s = df[c]
            if pd.api.types.is_bool_dtype(s):
                bool_cols.append(c)
            elif pd.api.types.is_string_dtype(s) or pd.api.types.is_object_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
                cat_cols.append(c)
                sv = s.fillna('Missing').astype(str)
                cats = list(pd.Index(sv.unique()).union(pd.Index(['Missing'])))
                cat_levels[c] = cats
        return cls(cols=list(cols), cat_cols=cat_cols, cat_levels=cat_levels, bool_cols=bool_cols)

    def transform_lgbm(self, df: pd.DataFrame) -> pd.DataFrame:
        out_cols: dict[str, Any] = {}
        n = len(df)
        for c in self.cols:
            if c in self.bool_cols:
                vals = df[c] if c in df.columns else False
                out_cols[c] = pd.Series(vals, index=df.index).fillna(False).astype('int8').to_numpy()
            elif c in self.cat_levels:
                cats = self.cat_levels[c]
                sv = df[c].fillna('Missing').astype(str) if c in df.columns else pd.Series('Missing', index=df.index)
                sv = sv.where(sv.isin(cats), 'Missing')
                out_cols[c] = pd.Categorical(sv, categories=cats)
            else:
                if c in df.columns:
                    out_cols[c] = pd.to_numeric(df[c], errors='coerce').astype('float32').to_numpy()
                else:
                    out_cols[c] = np.full(n, np.nan, dtype=np.float32)
        return pd.DataFrame(out_cols, index=df.index)

    def transform_catboost(self, df: pd.DataFrame) -> pd.DataFrame:
        out_cols: dict[str, Any] = {}
        n = len(df)
        for c in self.cols:
            if c in self.bool_cols:
                vals = df[c] if c in df.columns else False
                out_cols[c] = pd.Series(vals, index=df.index).fillna(False).astype('int8').to_numpy()
            elif c in self.cat_levels:
                sv = df[c].fillna('Missing').astype(str) if c in df.columns else pd.Series('Missing', index=df.index)
                out_cols[c] = sv.to_numpy()
            else:
                if c in df.columns:
                    out_cols[c] = pd.to_numeric(df[c], errors='coerce').astype('float32').to_numpy()
                else:
                    out_cols[c] = np.full(n, np.nan, dtype=np.float32)
        return pd.DataFrame(out_cols, index=df.index)


def build_conformal_context(df: pd.DataFrame, ens_log_pred: np.ndarray, ens_std: np.ndarray) -> pd.DataFrame:
    """Context features for the heteroscedastic conformal `sigma_hat(x)` predictor."""
    X_ctx = pd.DataFrame(index=range(len(df)))
    for cc in ('livable_area', 'property_type_HOUSE', 'is_newly_built', 'age', 'dist_from_center'):
        if cc in df.columns:
            X_ctx[cc] = pd.to_numeric(df[cc], errors='coerce').fillna(0.0).to_numpy(np.float32)
        else:
            X_ctx[cc] = np.zeros(len(df), dtype=np.float32)
    cond = df['building_condition'].astype(str) if 'building_condition' in df.columns else pd.Series('', index=df.index)
    X_ctx['is_fixer'] = cond.isin(['TO_RENOVATE', 'TO_RESTORE', 'TO_BE_DONE_UP']).astype(np.float32).to_numpy()
    X_ctx['ens_log_pred'] = np.asarray(ens_log_pred, dtype=np.float32)
    X_ctx['ens_std'] = np.asarray(ens_std, dtype=np.float32)
    return X_ctx


def _inv_target(pred_log: np.ndarray, area: np.ndarray, deflator: np.ndarray, mode: str) -> np.ndarray:
    p = np.clip(np.asarray(pred_log, dtype=np.float64), -20.0, 20.0)
    mult = area * deflator if mode == 'log_real_psqm' else deflator
    return np.clip(np.exp(p) * mult, 10_000.0, 1e7)


class TrackBundle:
    """One complete Lean-5 ensemble track (`tabular_text` or `multimodal`)."""

    def __init__(
        self,
        name: str,
        tree_spec: TreePreprocSpec,
        nn_spec: NNPreprocSpec,
        lgbmxt: Any,
        spec_lgbm_house: Any,
        spec_lgbm_apt: Any,
        catboost_model: Any,
        tabm_price_models: list[Any],
        tabm_psqm_models: list[Any],
        huber_stacker: HuberRegressor,
        conformal_sigma_model: Any,
        conformal_q80: float,
        conformal_q90: float,
        member_names: list[str],
    ):
        self.name = name
        self.tree_spec = tree_spec
        self.nn_spec = nn_spec
        self.lgbmxt = lgbmxt
        self.spec_lgbm_house = spec_lgbm_house
        self.spec_lgbm_apt = spec_lgbm_apt
        self.catboost_model = catboost_model
        self.tabm_price_models = tabm_price_models
        self.tabm_psqm_models = tabm_psqm_models
        self.huber_stacker = huber_stacker
        self.conformal_sigma_model = conformal_sigma_model
        self.conformal_q80 = float(conformal_q80)
        self.conformal_q90 = float(conformal_q90)
        self.member_names = member_names
        self._shap_explainer = None

    def predict_members_log_eur(self, feat_df: pd.DataFrame, deflator: np.ndarray) -> np.ndarray:
        """Return (N, 5) matrix of member predictions in log(EUR)."""
        n = len(feat_df)
        area = np.maximum(pd.to_numeric(feat_df['livable_area'], errors='coerce').fillna(100.0).to_numpy(float), 10.0)
        is_h = (pd.to_numeric(feat_df['property_type_HOUSE'], errors='coerce').fillna(0).to_numpy() == 1)

        X_lgb = self.tree_spec.transform_lgbm(feat_df)
        X_cb = self.tree_spec.transform_catboost(feat_df)
        X_nn = self.nn_spec.transform(feat_df)

        p_lgbmxt = _inv_target(self.lgbmxt.predict(X_lgb), area, deflator, 'log_real_price')

        p_spec = np.zeros(n, dtype=np.float64)
        if is_h.any():
            p_spec[is_h] = _inv_target(self.spec_lgbm_house.predict(X_lgb.iloc[is_h]), area[is_h], deflator[is_h], 'log_real_price')
        if (~is_h).any():
            p_spec[~is_h] = _inv_target(self.spec_lgbm_apt.predict(X_lgb.iloc[~is_h]), area[~is_h], deflator[~is_h], 'log_real_price')

        p_cb = _inv_target(self.catboost_model.predict(X_cb), area, deflator, 'log_real_price')

        p_tabm_price_log = np.mean([np.asarray(m.predict(X_nn), dtype=np.float64) for m in self.tabm_price_models], axis=0)
        p_tabm_price = _inv_target(p_tabm_price_log, area, deflator, 'log_real_price')

        p_tabm_psqm_log = np.mean([np.asarray(m.predict(X_nn), dtype=np.float64) for m in self.tabm_psqm_models], axis=0)
        p_tabm_psqm = _inv_target(p_tabm_psqm_log, area, deflator, 'log_real_psqm')

        M = np.column_stack([p_lgbmxt, p_spec, p_cb, p_tabm_price, p_tabm_psqm])
        return np.log(np.clip(M, 10_000.0, 1e7))

    def predict(self, feat_df: pd.DataFrame, deflator: np.ndarray) -> dict[str, np.ndarray]:
        """Predict point estimate + heteroscedastic 80%/90% conformal bands in raw EUR."""
        M_log = self.predict_members_log_eur(feat_df, deflator)
        pred_log = self.huber_stacker.predict(M_log)
        pred_eur = np.clip(np.exp(pred_log), 10_000.0, 1e7)
        ens_std = np.std(M_log, axis=1)

        X_ctx = build_conformal_context(feat_df, np.log(pred_eur), ens_std)
        sigma = np.clip(self.conformal_sigma_model.predict(X_ctx), 0.03, 0.50)
        hw_log_80 = self.conformal_q80 * sigma
        hw_log_90 = self.conformal_q90 * sigma

        low_80 = np.clip(pred_eur * np.exp(-hw_log_80), 10_000.0, pred_eur)
        high_80 = np.clip(pred_eur * np.exp(hw_log_80), pred_eur, 1e7)
        low_90 = np.clip(pred_eur * np.exp(-hw_log_90), 10_000.0, pred_eur)
        high_90 = np.clip(pred_eur * np.exp(hw_log_90), pred_eur, 1e7)

        return {
            'predicted_eur': pred_eur,
            'low_eur': low_80,
            'high_eur': high_80,
            'low_90_eur': low_90,
            'high_90_eur': high_90,
            'interval_halfwidth_pct': np.expm1(hw_log_80) * 100.0,
            'sigma_log': sigma,
            'ens_std_log': ens_std,
            'member_preds_eur': np.exp(M_log),
        }

    def explain_row_shap(self, feat_df_1row: pd.DataFrame, deflator_1: float, predicted_eur: float) -> dict[str, Any]:
        """Exact TreeSHAP on the served `lgbmxt_price_hub` model, mapped to EUR contributions."""
        import shap

        if self._shap_explainer is None:
            self._shap_explainer = shap.TreeExplainer(self.lgbmxt)

        X_lgb = self.tree_spec.transform_lgbm(feat_df_1row)
        sv = np.asarray(self._shap_explainer.shap_values(X_lgb), dtype=np.float64)[0]
        base_log = float(np.asarray(self._shap_explainer.expected_value).reshape(-1)[0])
        base_eur = float(np.exp(base_log) * deflator_1)
        lgb_pred_eur = float(np.exp(base_log + sv.sum()) * deflator_1)

        # Allocate total EUR delta (predicted_eur - base_eur) proportionally to log-SHAP values
        # so the waterfall sums cleanly to predicted_eur - base_eur!
        total_log = float(sv.sum())
        total_eur_delta = float(predicted_eur - base_eur)
        if abs(total_log) > 1e-9:
            eur_impacts = sv * (total_eur_delta / total_log)
        else:
            eur_impacts = np.zeros_like(sv)

        return {
            'feature_names': list(self.tree_spec.cols),
            'shap_log': sv,
            'shap_eur': eur_impacts,
            'base_value_eur': base_eur,
            'lgbmxt_pred_eur': lgb_pred_eur,
            'predicted_eur': float(predicted_eur),
        }


class E5LargeEncoder:
    """Lazy-loaded `intfloat/multilingual-e5-large` encoder (1024-d, normalized)."""

    def __init__(self, model_name: str = E5_LARGE_MODEL_NAME):
        self.model_name = model_name
        self._tokenizer = None
        self._model = None
        self._device = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name).to(self._device)
        self._model.eval()

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1024), dtype=np.float32)
        self._ensure_loaded()
        import torch

        prefixed = [f"passage: {t or ''}" for t in texts]
        out_chunks = []
        with torch.no_grad():
            for i in range(0, len(prefixed), batch_size):
                batch = prefixed[i:i + batch_size]
                enc = self._tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors='pt').to(self._device)
                hidden = self._model(**enc).last_hidden_state
                mask = enc['attention_mask'].unsqueeze(-1).float()
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                out_chunks.append(pooled.cpu().numpy().astype(np.float32))
        return np.vstack(out_chunks)


class ProductionBundle:
    """Top-level self-contained production bundle loaded by `api/` or batch scripts."""

    def __init__(
        self,
        pipeline: ProductionFeaturePipeline,
        tabular_text: TrackBundle,
        multimodal: TrackBundle | None,
        metadata: dict[str, Any],
        e5_encoder: E5LargeEncoder | None = None,
    ):
        self.pipeline = pipeline
        self.tabular_text = tabular_text
        self.multimodal = multimodal
        self.metadata = metadata
        self.e5_encoder = e5_encoder or E5LargeEncoder()

    def save(self, out_dir: str = DEFAULT_PROD_DIR) -> None:
        os.makedirs(out_dir, exist_ok=True)
        joblib.dump(self.pipeline, os.path.join(out_dir, 'feature_pipeline.joblib'), compress=3)
        joblib.dump(self.tabular_text, os.path.join(out_dir, 'track_tabular_text.joblib'), compress=3)
        if self.multimodal is not None:
            joblib.dump(self.multimodal, os.path.join(out_dir, 'track_multimodal.joblib'), compress=3)
        with open(os.path.join(out_dir, 'metadata.json'), 'w') as f:
            json.dump(self.metadata, f, indent=2)

    @classmethod
    def load(cls, prod_dir: str = DEFAULT_PROD_DIR, load_multimodal: bool = True) -> 'ProductionBundle':
        # Import torch FIRST on Apple Silicon to prevent OpenMP segfault with lightgbm/catboost!
        import torch  # noqa: F401

        pipeline = joblib.load(os.path.join(prod_dir, 'feature_pipeline.joblib'))
        tabtxt = joblib.load(os.path.join(prod_dir, 'track_tabular_text.joblib'))
        mm_path = os.path.join(prod_dir, 'track_multimodal.joblib')
        multimodal = joblib.load(mm_path) if (load_multimodal and os.path.exists(mm_path)) else None
        meta_path = os.path.join(prod_dir, 'metadata.json')
        metadata = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
        return cls(pipeline=pipeline, tabular_text=tabtxt, multimodal=multimodal, metadata=metadata)

    def predict_raw(
        self,
        raw_df: pd.DataFrame,
        e5l: np.ndarray | None = None,
        vis_bundle: dict[str, np.ndarray] | None = None,
        already_preprocessed: bool = False,
        include_shap: bool = False,
    ) -> dict[str, Any]:
        """End-to-end inference from raw parsed/manual rows -> EUR predictions, 80% bands & optional SHAP."""
        base_df, texts = prepare_base_dataframe(
            raw_df, self.pipeline.preproc_artifacts, already_preprocessed=already_preprocessed
        )
        if e5l is None:
            # Skip heavy E5 model load when all descriptions are empty (e.g. minimal manual form)
            if any(len(t.strip('. ')) > 3 for t in texts):
                e5l = self.e5_encoder.encode(texts)
            else:
                e5l = np.zeros((len(texts), 1024), dtype=np.float32)

        feat_df = self.pipeline.transform(base_df, texts, e5l, vis_bundle=vis_bundle)
        deflator = compute_deflator(
            feat_df, self.pipeline.cpi_ref_2025, self.pipeline.hpi_ref_2025, self.pipeline.latest_deflator
        )

        # Route each row to multimodal when photos are present and multimodal track is loaded,
        # otherwise use the dedicated tabular_text track (avoids the +1.4pp masked-vision blowup!).
        has_vis = (
            feat_df['vis_has_photos'].to_numpy(int) == 1
            if ('vis_has_photos' in feat_df.columns and vis_bundle is not None and self.multimodal is not None)
            else np.zeros(len(feat_df), dtype=bool)
        )

        if has_vis.all():
            res = self.multimodal.predict(feat_df, deflator)
            track_used = np.full(len(feat_df), 'multimodal', dtype=object)
            primary_track = self.multimodal
        elif (~has_vis).all():
            res = self.tabular_text.predict(feat_df, deflator)
            track_used = np.full(len(feat_df), 'tabular_text', dtype=object)
            primary_track = self.tabular_text
        else:
            res = {
                k: np.zeros((len(feat_df), 5), dtype=np.float64) if k == 'member_preds_eur' else np.zeros(len(feat_df), dtype=np.float64)
                for k in ('predicted_eur', 'low_eur', 'high_eur', 'low_90_eur', 'high_90_eur',
                          'interval_halfwidth_pct', 'sigma_log', 'ens_std_log', 'member_preds_eur')
            }
            track_used = np.full(len(feat_df), 'tabular_text', dtype=object)
            r_mm = self.multimodal.predict(feat_df.iloc[has_vis].reset_index(drop=True), deflator[has_vis])
            r_tt = self.tabular_text.predict(feat_df.iloc[~has_vis].reset_index(drop=True), deflator[~has_vis])
            for k in res:
                res[k][has_vis] = r_mm[k]
                res[k][~has_vis] = r_tt[k]
            track_used[has_vis] = 'multimodal'
            primary_track = self.multimodal if has_vis[0] else self.tabular_text

        res['track_used'] = track_used
        res['has_photos'] = has_vis
        res['feat_df'] = feat_df
        res['deflator'] = deflator

        if include_shap and len(feat_df) == 1:
            res['shap_raw'] = primary_track.explain_row_shap(
                feat_df.iloc[:1], float(deflator[0]), float(res['predicted_eur'][0])
            )
        return res
