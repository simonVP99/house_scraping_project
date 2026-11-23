"""Issue #3 — High-Residual Text & Structured Feature Extractor (Zero GPU, ~3s).

Extracts the verified pre-listing signals mined from our 14,599-row residual audit
for EVERY training row (and exposes a pure row-level function reusable at inference!):

  1. Text-vs-structured livable m² discrepancy (catches multi-unit blocks / entry errors!):
       - `txt_extracted_m2`, `log_txt_to_struct_m2_ratio`, `txt_struct_m2_mismatch_20pct`
  2. Text-vs-structured EPC kWh/m² discrepancy:
       - `txt_epc_kwh_diff`
  3. High-residual Flemish property/use flags (surviving even after TF-IDF + E5):
       - `txt_is_handelshuis` (handelshuis/handelsruimte/loods/atelier — r_resid=+0.0480!)
       - `txt_is_projectgrond` (projectgrond/afbraak — +7.5% shift!)
       - `txt_is_casco` (casco/ruwbouw — -4.5% shift!)
       - `txt_is_herenhuis` (herenhuis/herenwoning — +2.1%)
       - `txt_has_visgraat` (visgraat/chevron parquet — +3.6%)
       - `txt_is_cohousing` (co-housing/kangoeroewoning — -1.6%)
       - `txt_is_asbestvrij` (asbestvrij/asbestveilig)
       - `txt_has_high_ceilings` (hoge plafonds)
       - `txt_is_rented` (verhuurd/opbrengsteigendom)
       - `txt_mentions_underfloor` (vloerverwarming)
  4. Apartment floor interactions (n=6,448):
       - `apt_is_ground_floor`, `apt_is_top_floor`, `apt_high_floor_no_lift` (`floor>=3 & !lift`),
         `apt_floor_ratio` (`floor_number / max(total_floors, 1)`)
  5. Compact structural anomaly flags (pre-listing only — zero target leakage!):
       - `is_tiny_area_multi_bed` (`livable_area < 45 & number_of_bedrooms >= 2`)
       - `is_huge_area_low_beds` (`livable_area > 350 & number_of_bedrooms <= 2`)

Saves `outputs/dataset/issue3_extra_features.parquet` (keyed on `property_id`, `date_obtained`, aligned row-for-row with `training_dataset.parquet`).

Usage:
    .venv-api/bin/python -u src/extract_issue3_features.py
"""
from __future__ import annotations

import os
import re
import time

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, 'outputs', 'dataset', 'training_dataset.parquet')
RAW_PARQUET = os.path.join(BASE_DIR, 'outputs', 'property_data_full.parquet')
OUT_PATH = os.path.join(BASE_DIR, 'outputs', 'dataset', 'issue3_extra_features.parquet')

# Explicit livable-area mention in Dutch/FR descriptions
_RE_M2 = re.compile(
    r'(?:bewoonbare\s+oppervlakte|woonoppervlakte|surface\s+habitable|bewoonbaar|nut\w*\s+opp\w*)\s*(?:van|de|:|ca\.?|ongeveer|±|~)?\s*(\d{2,4}(?:[.,]\d)?)\s*m[²2]',
    re.IGNORECASE,
)
_RE_EPC_KWH = re.compile(r'(\d{2,4})\s*kwh\s*/\s*m[²2]', re.IGNORECASE)

TEXT_TERM_REGEXES: dict[str, re.Pattern] = {
    'txt_is_handelshuis':     re.compile(r'\b(handelshuis|handelsruimte|handelspand|loods|atelier|magazijn|praktijkruimte|winkelruimte)\b', re.I),
    'txt_is_projectgrond':    re.compile(r'\b(projectgrond|afbraak|te\s+slopen|bouwgrond|ontwikkelingsproject)\b', re.I),
    'txt_is_casco':           re.compile(r'\b(casco|ruwbouw|wind-\s*en\s*waterdicht)\b', re.I),
    'txt_is_herenhuis':       re.compile(r'\b(herenhuis|herenwoning|burgerwoning|stadspaleis|ma[îi]tre|art\s*d[ée]co)\b', re.I),
    'txt_has_visgraat':       re.compile(r'\b(visgraat|hongaarse\s+punt|chevron\s*parket)\b', re.I),
    'txt_is_cohousing':       re.compile(r'\b(co-?housing|kangoeroe\w*|zorgwoning|meergeneratie)\b', re.I),
    'txt_is_asbestvrij':      re.compile(r'\b(asbestvrij|asbestveilig|geen\s+asbest)\b', re.I),
    'txt_has_high_ceilings':  re.compile(r'\b(hoge\s+plafonds?|plafondhoogte|hauts?\s+plafonds?)\b', re.I),
    'txt_is_rented':          re.compile(r'\b(verhuurd|opbrengsteigendom|huuropbrengst|huurcontract|lou[ée])\b', re.I),
    'txt_mentions_underfloor':re.compile(r'\b(vloerverwarming|chauffage\s+au\s+sol)\b', re.I),
}


def compute_issue3_features(df_struct: pd.DataFrame, texts: list[str]) -> pd.DataFrame:
    n = len(df_struct)
    out: dict[str, np.ndarray] = {}

    # 1. Text-extracted m² & EPC kWh
    txt_m2 = np.full(n, np.nan, dtype=np.float32)
    txt_kwh = np.full(n, np.nan, dtype=np.float32)
    for i, t in enumerate(texts):
        if not t:
            continue
        m = _RE_M2.search(t)
        if m:
            try:
                v = float(m.group(1).replace(',', '.'))
                if 15.0 <= v <= 2000.0:
                    txt_m2[i] = v
            except ValueError:
                pass
        me = _RE_EPC_KWH.search(t)
        if me:
            try:
                v = float(me.group(1))
                if 0.0 <= v <= 2500.0:
                    txt_kwh[i] = v
            except ValueError:
                pass

    struct_m2 = pd.to_numeric(df_struct['livable_area'], errors='coerce').to_numpy(dtype=np.float32)
    log_ratio = np.where(
        np.isfinite(txt_m2) & np.isfinite(struct_m2) & (struct_m2 > 0),
        np.clip(np.log(np.maximum(txt_m2, 1.0) / np.maximum(struct_m2, 1.0)), -2.5, 2.5),
        0.0,
    ).astype(np.float32)
    mismatch_20 = (np.isfinite(txt_m2) & (np.abs(txt_m2 - struct_m2) / np.maximum(struct_m2, 1.0) > 0.20)).astype(np.int8)

    struct_kwh = pd.to_numeric(df_struct.get('primary_energy_consumption'), errors='coerce').to_numpy(dtype=np.float32)
    epc_diff = np.where(
        np.isfinite(txt_kwh) & np.isfinite(struct_kwh),
        np.clip(txt_kwh - struct_kwh, -500.0, 500.0),
        0.0,
    ).astype(np.float32)

    out['txt_extracted_m2'] = txt_m2
    out['log_txt_to_struct_m2_ratio'] = log_ratio
    out['txt_struct_m2_mismatch_20pct'] = mismatch_20
    out['txt_epc_kwh_diff'] = epc_diff

    # 2. High-residual Flemish term flags
    for col, pat in TEXT_TERM_REGEXES.items():
        out[col] = np.fromiter((1 if (t and pat.search(t)) else 0 for t in texts), dtype=np.int8, count=n)

    # 3. Apartment floor interactions
    is_apt = (pd.to_numeric(df_struct['property_type_HOUSE'], errors='coerce').fillna(0).to_numpy() == 0)
    fl = pd.to_numeric(df_struct.get('floor_number'), errors='coerce').to_numpy(dtype=np.float32)
    tf = pd.to_numeric(df_struct.get('total_floors'), errors='coerce').to_numpy(dtype=np.float32)
    has_lift = df_struct.get('has_lift', pd.Series(False, index=df_struct.index)).fillna(False).astype(bool).to_numpy()

    out['apt_is_ground_floor'] = (is_apt & np.isfinite(fl) & (fl == 0)).astype(np.int8)
    out['apt_is_top_floor'] = (is_apt & np.isfinite(fl) & np.isfinite(tf) & (tf >= 2) & (fl >= tf)).astype(np.int8)
    out['apt_high_floor_no_lift'] = (is_apt & np.isfinite(fl) & (fl >= 3) & (~has_lift)).astype(np.int8)
    out['apt_floor_ratio'] = np.where(
        is_apt & np.isfinite(fl) & np.isfinite(tf) & (tf > 0),
        np.clip(fl / np.maximum(tf, 1.0), 0.0, 1.5),
        np.nan,
    ).astype(np.float32)

    # 4. Pre-listing structural sanity flags
    beds = pd.to_numeric(df_struct.get('number_of_bedrooms'), errors='coerce').to_numpy(dtype=np.float32)
    out['is_tiny_area_multi_bed'] = (np.isfinite(struct_m2) & (struct_m2 < 45) & np.isfinite(beds) & (beds >= 2)).astype(np.int8)
    out['is_huge_area_low_beds'] = (np.isfinite(struct_m2) & (struct_m2 > 350) & np.isfinite(beds) & (beds <= 2)).astype(np.int8)

    return pd.DataFrame(out)


def main():
    t0 = time.perf_counter()
    tr = pd.read_parquet(DATASET_PATH)
    tr['property_id'] = tr['property_id'].astype(str)

    raw = pd.read_parquet(RAW_PARQUET, columns=['property_id', 'date_obtained', 'title', 'description', 'description_nl'])
    raw['property_id'] = raw['property_id'].astype(str)
    raw['date_obtained_dt'] = pd.to_datetime(raw['date_obtained'], format='%d%m%Y', errors='coerce')
    raw = raw.drop_duplicates(subset=['property_id', 'date_obtained_dt'], keep='first')

    tr_dt = pd.to_datetime(tr['date_obtained'], errors='coerce')
    m = pd.DataFrame({'property_id': tr['property_id'], 'date_obtained_dt': tr_dt}).merge(
        raw[['property_id', 'date_obtained_dt', 'title', 'description', 'description_nl']],
        on=['property_id', 'date_obtained_dt'],
        how='left',
    )

    texts = []
    for ti, dnl, dmain in zip(m['title'].fillna(''), m['description_nl'].fillna(''), m['description'].fillna('')):
        body = dnl if len(str(dnl)) >= 40 else dmain
        texts.append(f"{ti}. {body}".strip())

    feats = compute_issue3_features(tr, texts)
    out = pd.concat([tr[['property_id', 'date_obtained']].reset_index(drop=True), feats], axis=1)
    out.to_parquet(OUT_PATH, index=False)

    print(f"=== ISSUE #3 EXTRA FEATURES BUILT IN {time.perf_counter()-t0:.2f}s ({feats.shape[1]} cols) ===")
    for c in feats.columns:
        s = feats[c]
        if s.dtype == np.int8:
            print(f"  {c:30s}: count={int(s.sum()):5d} ({s.mean()*100:5.2f}%)")
        else:
            print(f"  {c:30s}: non-null={int(s.notna().sum()):5d} ({s.notna().mean()*100:5.1f}%) med={s.median():.3f}")
    print(f"Saved -> {OUT_PATH}")


if __name__ == '__main__':
    main()
