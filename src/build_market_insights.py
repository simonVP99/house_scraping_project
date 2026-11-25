"""Build Phase-7 OOF market analytics & insight artifacts for the web app.

Generates two compact, gitignored artifacts under ``outputs/models/production/``:
  1. ``oof_analytics.parquet`` — the 14,599 clean corpus properties enriched with
     leak-free Phase-7 Lean-5 OOF predictions (multimodal + tabular_text),
     conformal uncertainty intervals, listing duration / price-cut / agency
     metadata from raw snapshots, and clean display neighbourhood names.
  2. ``market_insights.json`` — pre-computed research insights powered by the
     agency-blind Phase-7 model (Agency Pricing Audit, EPC Renovation Cliff,
     Price-Cut Forensics, Neighbourhood Fair-Value Index, Model Benchmarks,
     and Ghent Spatial Hex/Wijk Affordability Grid).

Run:
    PYTHONPATH=src .venv-api/bin/python -u src/build_market_insights.py
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import torch  # noqa: F401  # Must precede lightgbm/catboost on macOS (OpenMP)
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "outputs" / "dataset"
WORK_DIR = ROOT / "outputs" / "models" / "workbench"
STAGE_DIR = WORK_DIR / "prod_train_staging"
PROD_DIR = ROOT / "outputs" / "models" / "production"
RAW_PARQUET = ROOT / "outputs" / "property_data_full.parquet"
WIKEN_CSV = ROOT / "external_data" / "stadswijken-gent.csv"


def _load_wijk_display_map() -> dict[str, str]:
    """Map normalized neighbourhood keys to clean official Ghent wijk display names."""
    mapping: dict[str, str] = {}
    if WIKEN_CSV.exists():
        wdf = pd.read_csv(WIKEN_CSV, sep=";")
        for raw_name in wdf["wijk"].dropna().unique():
            norm = str(raw_name).replace(" ", "").replace("-", "_")
            pretty = re.sub(r"\s*-\s*", " – ", str(raw_name).strip()).replace("en – zone", "en-zone")
            mapping[norm] = pretty
            mapping[str(raw_name)] = pretty
    return mapping


def _prettify_neighbourhood(name: str, display_map: dict[str, str]) -> str:
    if not name or pd.isna(name):
        return "Other Ghent"
    s = str(name)
    if s in display_map:
        return display_map[s]
    if s == "Unknown":
        return "Greater Ghent / Unmapped"
    # Insert space on camel-case boundaries & replace underscores
    s = s.replace("_", " ")
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)
    return s.strip()


def _compute_oof_stack_predictions(track_name: str, splits_data: dict) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce exact leak-free 5-fold Huber stacked OOF predictions in EUR."""
    gbdt_art = joblib.load(STAGE_DIR / f"gbdt_{track_name}.joblib")
    tabm_price = joblib.load(STAGE_DIR / f"tabm_perm170_price_4s_{track_name}.joblib")
    tabm_psqm = joblib.load(STAGE_DIR / f"tabm_perm170_psqm_4s_{track_name}.joblib")

    y_price = np.asarray(splits_data["y_price"], dtype=np.float64)
    normal_mask = np.asarray(splits_data["normal_mask"], dtype=bool)
    folds = np.asarray(splits_data["folds"])

    members = [
        gbdt_art["oof"]["lgbmxt_price_hub"],
        gbdt_art["oof"]["spec_lgbm_price_hub"],
        gbdt_art["oof"]["cb_tuned_price_l2"],
        tabm_price["oof"],
        tabm_psqm["oof"],
    ]
    X_log = np.column_stack([np.log(np.clip(m, 10_000.0, 1e7)) for m in members])
    y_log = np.log(np.clip(y_price, 10_000.0, 1e7))

    oof_log = np.zeros(len(y_price), dtype=np.float64)
    for k in range(5):
        tr = (folds != k) & normal_mask
        va = folds == k
        huber = HuberRegressor(alpha=15.0, max_iter=500)
        huber.fit(X_log[tr], y_log[tr])
        oof_log[va] = huber.predict(X_log[va])

    oof_eur = np.clip(np.exp(oof_log), 10_000.0, 1e7)
    ens_std = np.std(X_log, axis=1)
    return oof_eur, ens_std


def _normalize_agency(name: str | None) -> str | None:
    if not name or pd.isna(name):
        return None
    s = str(name).strip()
    if not s or s.lower() in {"nan", "none", "unknown"}:
        return None
    s = re.sub(r"\s+", " ", s)
    return s


def _canonicalize_agencies(series: pd.Series) -> pd.Series:
    """Merge case-only agency duplicates (e.g. 'TREVI Gent' and 'Trevi Gent') using the most frequent spelling."""
    cleaned = series.map(_normalize_agency)
    non_null = cleaned.dropna()
    if non_null.empty:
        return cleaned
    lower = non_null.str.lower()
    # Pick the most common spelling for each lowercase key
    canon_map = (
        pd.DataFrame({"raw": non_null, "low": lower})
        .groupby("low")["raw"]
        .agg(lambda s: s.value_counts().index[0])
        .to_dict()
    )
    return cleaned.map(lambda x: canon_map.get(x.lower(), x) if isinstance(x, str) else None)


def main() -> None:
    print("Loading training dataset & Phase-7 staging artifacts...")
    df = pd.read_parquet(DATASET_DIR / "training_dataset.parquet")
    splits_data = joblib.load(STAGE_DIR / "prod_splits.joblib")
    meta = json.loads((PROD_DIR / "metadata.json").read_text(encoding="utf-8"))

    # 1. Leak-free Phase-7 OOF predictions for both tracks
    oof_mm, std_mm = _compute_oof_stack_predictions("multimodal", splits_data)
    oof_tt, std_tt = _compute_oof_stack_predictions("tabular_text", splits_data)

    # Approximate calibrated 80% halfwidth from track conformal metadata
    mm_hw_pct = float(meta["tracks"]["multimodal"]["conformal_80"]["oof_median_hw_pct"])
    tt_hw_pct = float(meta["tracks"]["tabular_text"]["conformal_80"]["oof_median_hw_pct"])

    df = df.copy()
    df["pred_mm_eur"] = np.round(oof_mm, 0)
    df["pred_tt_eur"] = np.round(oof_tt, 0)
    df["ens_std_mm"] = np.round(std_mm, 4)
    df["ens_std_tt"] = np.round(std_tt, 4)

    # Delta: positive = asking BELOW Phase-7 fair value (underpriced deal),
    # negative = asking ABOVE Phase-7 fair value (overpriced)
    df["delta_eur"] = np.round(df["pred_mm_eur"] - df["price"], 0)
    df["delta_pct"] = np.round((df["pred_mm_eur"] - df["price"]) / np.maximum(df["price"], 1.0) * 100.0, 2)
    # Signed premium = (asking - fair_value) / fair_value  (positive = agency/seller priced above fair value)
    df["asking_premium_pct"] = np.round((df["price"] - df["pred_mm_eur"]) / np.maximum(df["pred_mm_eur"], 1.0) * 100.0, 2)
    df["vision_delta_eur"] = np.round(df["pred_mm_eur"] - df["pred_tt_eur"], 0)
    df["vision_delta_pct"] = np.round((df["pred_mm_eur"] - df["pred_tt_eur"]) / np.maximum(df["pred_tt_eur"], 1.0) * 100.0, 2)
    df["normal_mask"] = np.asarray(splits_data["normal_mask"], dtype=bool)
    df["price_per_sqm"] = np.where(df["livable_area"] > 0, df["price"] / df["livable_area"], np.nan)
    df["pred_price_per_sqm"] = np.where(df["livable_area"] > 0, df["pred_mm_eur"] / df["livable_area"], np.nan)

    # 2. Enrich with raw snapshot dynamics (listing duration, under-option, price cuts, agency, availability)
    print("Aggregating raw snapshot dynamics (duration, price cuts, agencies, availability)...")
    raw_cols = [
        "property_id", "date_obtained", "price", "is_under_option", "is_sold_or_rented",
        "immo_office_name", "price_old_value", "street", "house_number", "title", "description_nl", "description",
    ]
    raw = pd.read_parquet(RAW_PARQUET, columns=raw_cols)
    raw["dt"] = pd.to_datetime(raw["date_obtained"], format="%d%m%Y", errors="coerce")
    raw["pid"] = raw["property_id"].astype(str)
    raw = raw.sort_values("dt")
    latest_scrape_dt = raw["dt"].max()

    grp = raw.groupby("pid").agg(
        first_seen=("dt", "min"),
        last_seen=("dt", "max"),
        n_snapshots=("dt", "size"),
        first_price=("price", "first"),
        last_price=("price", "last"),
        is_under_option=("is_under_option", lambda s: bool(s.fillna(False).any())),
        is_under_option_latest=("is_under_option", lambda s: bool(s.fillna(False).iloc[-1])),
        is_sold_or_rented=("is_sold_or_rented", lambda s: bool(s.fillna(False).any())),
        immo_office_name=("immo_office_name", "last"),
        has_old_price=("price_old_value", lambda s: bool(s.notna().any())),
        street=("street", "last"),
        house_number=("house_number", "last"),
        title=("title", "last"),
        description_nl=("description_nl", "last"),
        description=("description", "last"),
    ).reset_index()
    grp["listing_duration_days"] = (grp["last_seen"] - grp["first_seen"]).dt.days.fillna(0).astype(int)
    grp["days_since_last_seen"] = (latest_scrape_dt - grp["last_seen"]).dt.days.fillna(999).astype(int)
    grp["first_seen_str"] = grp["first_seen"].dt.strftime("%Y-%m-%d")
    grp["last_seen_str"] = grp["last_seen"].dt.strftime("%Y-%m-%d")
    grp["had_price_cut"] = (
        ((grp["last_price"] < grp["first_price"]) & (grp["first_price"] > 0))
        | grp["has_old_price"]
    )
    grp["price_cut_eur"] = np.where(
        (grp["last_price"] < grp["first_price"]) & (grp["first_price"] > 0),
        grp["first_price"] - grp["last_price"],
        np.nan,
    )
    grp["price_cut_pct"] = np.where(
        (grp["last_price"] < grp["first_price"]) & (grp["first_price"] > 0),
        (grp["first_price"] - grp["last_price"]) / grp["first_price"] * 100.0,
        np.nan,
    )

    # Compact summary snippet (prefer Dutch, clean whitespace, max 360 chars)
    def _clean_snippet(row):
        for col in ("description_nl", "description"):
            val = row[col]
            if isinstance(val, str) and len(val.strip()) >= 25:
                txt = " ".join(val.split())
                return txt[:360] + ("…" if len(txt) > 360 else "")
        return ""

    grp["description_snippet"] = grp.apply(_clean_snippet, axis=1)

    df["pid"] = df["property_id"].astype(str)
    df = df.merge(
        grp[["pid", "first_seen", "last_seen", "first_seen_str", "last_seen_str", "days_since_last_seen",
             "n_snapshots", "listing_duration_days", "is_under_option", "is_under_option_latest",
             "is_sold_or_rented", "immo_office_name", "had_price_cut", "price_cut_eur", "price_cut_pct",
             "street", "house_number", "title", "description_nl", "description", "description_snippet"]],
        on="pid",
        how="left",
        suffixes=("", "_raw"),
    )
    if "is_under_option_raw" in df.columns:
        df["is_under_option"] = df["is_under_option_raw"].fillna(False).astype(bool)
        df.drop(columns=["is_under_option_raw"], inplace=True)
    else:
        df["is_under_option"] = df["is_under_option"].fillna(False).astype(bool)
    df["is_under_option_latest"] = df["is_under_option_latest"].fillna(False).astype(bool)
    df["is_sold_or_rented"] = df["is_sold_or_rented"].fillna(False).astype(bool)
    df["listing_duration_days"] = df["listing_duration_days"].fillna(0).astype(float)
    df["days_since_last_seen"] = df["days_since_last_seen"].fillna(999).astype(int)
    df["had_price_cut"] = df["had_price_cut"].fillna(False).astype(bool)
    df["latest_scrape_date"] = latest_scrape_dt.strftime("%Y-%m-%d")

    display_map = _load_wijk_display_map()
    df["neighbourhood_display"] = df["neighbourhood"].map(lambda x: _prettify_neighbourhood(x, display_map))
    df["agency_clean"] = _canonicalize_agencies(df["immo_office_name"])
    df["property_type_label"] = np.where(df.get("property_type_HOUSE", 0) == 1, "House", "Apartment")

    # 2b. Attach Cross-Modal JEPA OOF diagnostics & mixed-use / investment regime tags
    jepa_oof_path = WORK_DIR / "issue10_jepa_oof.parquet"
    if jepa_oof_path.exists():
        j_oof = pd.read_parquet(jepa_oof_path)
        if len(j_oof) == len(df):
            df["jepa_dz_quality"] = j_oof["jepa_dz_quality_index"].to_numpy(dtype=float)
            df["jepa_energy_sym"] = j_oof["jepa_energy_sym"].to_numpy(dtype=float)
            df["jepa_cherry_spread"] = j_oof["jepa_cherry_pick_spread"].to_numpy(dtype=float)
            df["jepa_has_vis"] = j_oof["normal_vis"].to_numpy(dtype=bool)
        else:
            df["jepa_dz_quality"] = np.nan
            df["jepa_energy_sym"] = np.nan
            df["jepa_cherry_spread"] = np.nan
            df["jepa_has_vis"] = False
    else:
        df["jepa_dz_quality"] = np.nan
        df["jepa_energy_sym"] = np.nan
        df["jepa_cherry_spread"] = np.nan
        df["jepa_has_vis"] = False

    from jepa_inspector import detect_property_regime
    regimes = [
        detect_property_regime(
            property_subtype=st, title=ti, description=ds, description_nl=dnl
        )["code"]
        for st, ti, ds, dnl in zip(
            df.get("property_subtype", pd.Series("", index=df.index)),
            df["title"].fillna(""),
            df["description"].fillna(""),
            df["description_nl"].fillna(""),
        )
    ]
    df["property_regime"] = regimes

    # 3. Save compact oof_analytics.parquet for DataLoader
    keep_cols = [
        "property_id", "date_obtained", "publication_year", "publication_month",
        "first_seen_str", "last_seen_str", "days_since_last_seen", "latest_scrape_date",
        "price", "pred_mm_eur", "pred_tt_eur", "delta_eur", "delta_pct", "asking_premium_pct",
        "vision_delta_eur", "vision_delta_pct", "ens_std_mm", "normal_mask",
        "price_per_sqm", "pred_price_per_sqm",
        "livable_area", "property_size", "number_of_bedrooms", "number_of_bathrooms",
        "property_type_HOUSE", "property_type_APARTMENT", "property_type_label", "property_subtype",
        "property_regime",
        "building_condition", "energy_label", "primary_energy_consumption", "year_built", "age",
        "has_garden", "garden_surface", "has_terrace", "terrace_surface",
        "has_lift", "has_double_glazing", "has_photovoltaic_panels", "has_heat_pump",
        "indoor_parking_count", "outdoor_parking_count", "cadastral_income",
        "latitude", "longitude", "postal_code", "street", "house_number",
        "neighbourhood", "neighbourhood_display",
        "statbel_sector", "dist_from_center", "street_type",
        "listing_duration_days", "is_under_option", "is_under_option_latest", "is_sold_or_rented",
        "had_price_cut", "price_cut_eur", "price_cut_pct",
        "agency_clean", "n_snapshots",
        "title", "description_snippet",
        "jepa_dz_quality", "jepa_energy_sym", "jepa_cherry_spread",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    out_parquet = PROD_DIR / "oof_analytics.parquet"
    df[keep_cols].to_parquet(out_parquet, index=False)
    print(f"Wrote {out_parquet} ({len(df)} rows, {len(keep_cols)} cols)")

    # 4. Compute rich market_insights.json
    normal_df = df[df["normal_mask"]].copy()

    # 4a. Agency Pricing Audit (agencies with >= 40 normal listings)
    ag_df = normal_df[normal_df["agency_clean"].notna()].copy()
    ag_stats = (
        ag_df.groupby("agency_clean")
        .agg(
            listings=("price", "size"),
            median_asking_eur=("price", "median"),
            median_fair_eur=("pred_mm_eur", "median"),
            median_premium_pct=("asking_premium_pct", "median"),
            mean_premium_pct=("asking_premium_pct", "mean"),
            pct_overpriced_10=("asking_premium_pct", lambda s: float((s >= 10).mean() * 100)),
            pct_underpriced_10=("asking_premium_pct", lambda s: float((s <= -10).mean() * 100)),
            pct_price_cut=("had_price_cut", lambda s: float(s.mean() * 100)),
            median_duration_days=("listing_duration_days", "median"),
            median_jepa_dz=("jepa_dz_quality", "median"),
        )
        .reset_index()
    )
    ag_stats = ag_stats[ag_stats["listings"] >= 40].sort_values("median_premium_pct", ascending=False)
    agency_audit = [
        {
            "agency": r["agency_clean"],
            "listings": int(r["listings"]),
            "median_asking_eur": int(round(r["median_asking_eur"])),
            "median_fair_eur": int(round(r["median_fair_eur"])),
            "median_premium_pct": round(float(r["median_premium_pct"]), 2),
            "pct_overpriced_10": round(float(r["pct_overpriced_10"]), 1),
            "pct_underpriced_10": round(float(r["pct_underpriced_10"]), 1),
            "pct_price_cut": round(float(r["pct_price_cut"]), 1),
            "median_duration_days": int(round(r["median_duration_days"])),
            "median_jepa_dz": round(float(r["median_jepa_dz"]), 2) if pd.notna(r["median_jepa_dz"]) else None,
        }
        for _, r in ag_stats.iterrows()
    ]

    # 4b. EPC Renovation Cliff (size-adjusted €/m² & residual by EPC label)
    epc_order = ["A++", "A+", "A", "B", "C", "D", "E", "F"]
    epc_df = normal_df[normal_df["energy_label"].isin(epc_order)].copy()
    # Compare each EPC against C-label baseline within same property type
    epc_breakdown = []
    for ptype in ["All", "House", "Apartment"]:
        sub = epc_df if ptype == "All" else epc_df[epc_df["property_type_label"] == ptype]
        c_psqm = float(sub.loc[sub["energy_label"] == "C", "price_per_sqm"].median()) if (sub["energy_label"] == "C").any() else np.nan
        rows_epc = []
        for lbl in epc_order:
            g = sub[sub["energy_label"] == lbl]
            if len(g) < 10:
                continue
            med_psqm = float(g["price_per_sqm"].median())
            med_price = float(g["price"].median())
            med_area = float(g["livable_area"].median())
            vs_c_pct = ((med_psqm - c_psqm) / c_psqm * 100.0) if np.isfinite(c_psqm) and c_psqm > 0 else 0.0
            rows_epc.append({
                "label": lbl,
                "count": int(len(g)),
                "median_price_eur": int(round(med_price)),
                "median_price_sqm": int(round(med_psqm)),
                "median_area": int(round(med_area)),
                "vs_c_psqm_pct": round(vs_c_pct, 1),
                "mandatory_renovation": lbl in {"E", "F"},
            })
        epc_breakdown.append({"property_type": ptype, "labels": rows_epc})

    # 4c. Condition Ladder (median €/m² and price by building condition)
    cond_order = ["AS_NEW", "JUST_RENOVATED", "GOOD", "TO_BE_DONE_UP", "TO_RENOVATE", "TO_RESTORE"]
    cond_labels = {
        "AS_NEW": "As New",
        "JUST_RENOVATED": "Just Renovated",
        "GOOD": "Good",
        "TO_BE_DONE_UP": "To Refresh",
        "TO_RENOVATE": "To Renovate",
        "TO_RESTORE": "To Restore",
    }
    condition_ladder = []
    for c in cond_order:
        g = normal_df[normal_df["building_condition"] == c]
        if len(g) < 10:
            continue
        condition_ladder.append({
            "code": c,
            "label": cond_labels[c],
            "count": int(len(g)),
            "median_price_eur": int(round(float(g["price"].median()))),
            "median_price_sqm": int(round(float(g["price_per_sqm"].median()))),
            "median_area": int(round(float(g["livable_area"].median()))),
            "pct_houses": round(float((g["property_type_HOUSE"] == 1).mean() * 100), 1),
        })

    # 4d. Price-Cut & Overpricing Forensics
    # Bucket listings by initial asking_premium_pct and measure price-cut rate & duration
    bins = [-100, -10, -3, 3, 10, 20, 200]
    bin_labels = ["Underpriced >10%", "Slightly Below (3–10%)", "Fairly Priced (±3%)", "Slightly Above (3–10%)", "Overpriced (10–20%)", "Heavily Overpriced >20%"]
    normal_df["pricing_bucket"] = pd.cut(normal_df["asking_premium_pct"], bins=bins, labels=bin_labels)
    overpricing_forensics = []
    for bl in bin_labels:
        g = normal_df[normal_df["pricing_bucket"] == bl]
        if len(g) == 0:
            continue
        overpricing_forensics.append({
            "bucket": bl,
            "count": int(len(g)),
            "pct_price_cut": round(float(g["had_price_cut"].mean() * 100), 2),
            "pct_under_option": round(float(g["is_under_option"].mean() * 100), 1),
            "avg_duration_days": round(float(g["listing_duration_days"].mean()), 1),
            "median_duration_days": int(round(float(g["listing_duration_days"].median()))),
        })

    # 4e. Vision Surprise Analysis (Multimodal vs Tabular+Text gap)
    vis_gap = normal_df["vision_delta_pct"].dropna()
    vision_impact = {
        "multimodal_mdape": round(float(meta["tracks"]["multimodal"]["oof_normal"]["mdape"]), 2),
        "tabular_text_mdape": round(float(meta["tracks"]["tabular_text"]["oof_normal"]["mdape"]), 2),
        "mdape_gain_pts": round(
            float(meta["tracks"]["tabular_text"]["oof_normal"]["mdape"])
            - float(meta["tracks"]["multimodal"]["oof_normal"]["mdape"]),
            2,
        ),
        "median_abs_shift_pct": round(float(vis_gap.abs().median()), 2),
        "p90_abs_shift_pct": round(float(vis_gap.abs().quantile(0.90)), 2),
        "pct_shifted_over_5pct": round(float((vis_gap.abs() >= 5.0).mean() * 100), 1),
        "histogram": _histogram(vis_gap.clip(-20, 20), bins=20),
    }

    # 4f. Model Error & Conformal Calibration by Price Decile
    decile_df = normal_df.copy()
    decile_df["abs_pct_err"] = np.abs(decile_df["pred_mm_eur"] - decile_df["price"]) / np.maximum(decile_df["price"], 1.0) * 100.0
    decile_df["price_band"] = pd.cut(
        decile_df["price"],
        bins=[150_000, 250_000, 325_000, 400_000, 500_000, 625_000, 800_000],
        labels=["€150k–250k", "€250k–325k", "€325k–400k", "€400k–500k", "€500k–625k", "€625k–800k"],
    )
    error_by_band = []
    for band, g in decile_df.groupby("price_band", observed=False):
        if len(g) == 0:
            continue
        error_by_band.append({
            "band": str(band),
            "count": int(len(g)),
            "mdape": round(float(g["abs_pct_err"].median()), 2),
            "hit_10": round(float((g["abs_pct_err"] <= 10.0).mean() * 100), 1),
            "hit_15": round(float((g["abs_pct_err"] <= 15.0).mean() * 100), 1),
            "mae_eur": int(round(float(np.abs(g["pred_mm_eur"] - g["price"]).mean()))),
        })

    # 4g. Neighbourhood Affordability Profiles (for Reverse-Search Affordability Explorer)
    # For each neighbourhood, store empirical price distributions by type + bedrooms
    nbh_profiles = []
    for nbh_key, g in df[df["neighbourhood"] != "Unknown"].groupby("neighbourhood"):
        if len(g) < 20:
            continue
        profile = {
            "name": nbh_key,
            "display_name": _prettify_neighbourhood(nbh_key, display_map),
            "count": int(len(g)),
            "lat": round(float(g["latitude"].median()), 5),
            "lon": round(float(g["longitude"].median()), 5),
            "median_price_eur": int(round(float(g["price"].median()))),
            "median_fair_eur": int(round(float(g["pred_mm_eur"].median()))),
            "median_price_sqm": int(round(float(g["price_per_sqm"].median()))),
            "p25_price_eur": int(round(float(g["price"].quantile(0.25)))),
            "p75_price_eur": int(round(float(g["price"].quantile(0.75)))),
            "pct_houses": round(float((g["property_type_HOUSE"] == 1).mean() * 100), 1),
            "median_area": int(round(float(g["livable_area"].median()))),
            "median_dist_center_km": round(float(g["dist_from_center"].median()), 2) if "dist_from_center" in g else None,
            "by_type": {},
        }
        for ptype_label in ["House", "Apartment"]:
            sub = g[g["property_type_label"] == ptype_label]
            if len(sub) >= 8:
                profile["by_type"][ptype_label] = {
                    "count": int(len(sub)),
                    "median_price_eur": int(round(float(sub["price"].median()))),
                    "median_price_sqm": int(round(float(sub["price_per_sqm"].median()))),
                    "p20_price_eur": int(round(float(sub["price"].quantile(0.20)))),
                    "p50_price_eur": int(round(float(sub["price"].quantile(0.50)))),
                    "p80_price_eur": int(round(float(sub["price"].quantile(0.80)))),
                    "median_area": int(round(float(sub["livable_area"].median()))),
                    "median_bedrooms": round(float(sub["number_of_bedrooms"].median()), 1) if sub["number_of_bedrooms"].notna().any() else None,
                }
        nbh_profiles.append(profile)
    nbh_profiles.sort(key=lambda x: x["median_price_sqm"], reverse=True)

    # 4h. Monthly city price index (clean chronological timeline)
    dated = df[df["date_obtained"].notna()].copy()
    dated["dt"] = pd.to_datetime(dated["date_obtained"], errors="coerce")
    dated = dated[dated["dt"].notna()]
    dated["month"] = dated["dt"].dt.to_period("M").astype(str)
    monthly = (
        dated.groupby("month")
        .agg(
            listings=("price", "size"),
            median_price_eur=("price", "median"),
            median_fair_eur=("pred_mm_eur", "median"),
            median_price_sqm=("price_per_sqm", "median"),
        )
        .reset_index()
        .sort_values("month")
    )
    monthly_series = [
        {
            "month": r["month"],
            "listings": int(r["listings"]),
            "median_price_eur": int(round(r["median_price_eur"])),
            "median_fair_eur": int(round(r["median_fair_eur"])),
            "median_price_sqm": int(round(r["median_price_sqm"])),
        }
        for _, r in monthly.iterrows()
        if int(r["listings"]) >= 15
    ]

    # 4i. Cross-Modal JEPA Photo-vs-Specs Forensics & Property Regime Breakdown
    jepa_forensics = None
    j_sub = normal_df[normal_df["jepa_has_vis"] & normal_df["jepa_energy_sym"].notna()].copy()
    if len(j_sub) >= 500:
        j_sub["abs_err_pct"] = np.abs(j_sub["pred_mm_eur"] - j_sub["price"]) / np.maximum(j_sub["price"], 1.0) * 100.0
        j_sub["e_q"] = pd.qcut(
            j_sub["jepa_energy_sym"], 5,
            labels=["Q1 · Match", "Q2", "Q3", "Q4", "Q5 · Contradiction"],
        )
        energy_quintiles = []
        for q, g in j_sub.groupby("e_q", observed=True):
            energy_quintiles.append({
                "quintile": str(q),
                "count": int(len(g)),
                "mdape": round(float(g["abs_err_pct"].median()), 2),
                "p80_err": round(float(g["abs_err_pct"].quantile(0.80)), 2),
                "pct_price_cut": round(float(g["had_price_cut"].mean() * 100.0), 2),
                "pct_under_option": round(float(g["is_under_option"].mean() * 100.0), 1),
                "avg_duration_days": round(float(g["listing_duration_days"].mean()), 1),
            })

        # Controlled check on fairly priced listings (-5%..+5%)
        fair_j = j_sub[j_sub["asking_premium_pct"].between(-5.0, 5.0)]
        fq1 = fair_j[fair_j["jepa_energy_sym"] <= fair_j["jepa_energy_sym"].quantile(0.20)]
        fq5 = fair_j[fair_j["jepa_energy_sym"] >= fair_j["jepa_energy_sym"].quantile(0.80)]
        fair_controlled = {
            "count": int(len(fair_j)),
            "q1_duration_days": round(float(fq1["listing_duration_days"].mean()), 1),
            "q5_duration_days": round(float(fq5["listing_duration_days"].mean()), 1),
            "q1_price_cut_pct": round(float(fq1["had_price_cut"].mean() * 100.0), 2),
            "q5_price_cut_pct": round(float(fq5["had_price_cut"].mean() * 100.0), 2),
            "q1_under_option_pct": round(float(fq1["is_under_option"].mean() * 100.0), 1),
            "q5_under_option_pct": round(float(fq5["is_under_option"].mean() * 100.0), 1),
        }

        # Condition Claim Honesty among AS_NEW / JUST_RENOVATED
        claimed_hi = j_sub[j_sub["building_condition"].isin(["AS_NEW", "JUST_RENOVATED"])]
        lo_c = claimed_hi[claimed_hi["jepa_dz_quality"] <= claimed_hi["jepa_dz_quality"].quantile(0.15)]
        hi_c = claimed_hi[claimed_hi["jepa_dz_quality"] >= claimed_hi["jepa_dz_quality"].quantile(0.85)]
        condition_honesty = {
            "total_claimed_top": int(len(claimed_hi)),
            "bottom15_looks_dated": {
                "count": int(len(lo_c)),
                "pct_price_cut": round(float(lo_c["had_price_cut"].mean() * 100.0), 2),
                "median_vision_shift_pct": round(float(lo_c["vision_delta_pct"].median()), 2),
                "avg_duration_days": round(float(lo_c["listing_duration_days"].mean()), 1),
            },
            "top15_looks_sharp": {
                "count": int(len(hi_c)),
                "pct_price_cut": round(float(hi_c["had_price_cut"].mean() * 100.0), 2),
                "median_vision_shift_pct": round(float(hi_c["vision_delta_pct"].median()), 2),
                "avg_duration_days": round(float(hi_c["listing_duration_days"].mean()), 1),
            },
        }

        # Cherry-picking quintiles
        j_sub["cp_q"] = pd.qcut(
            j_sub["jepa_cherry_spread"], 5,
            labels=["Q1 · Consistent", "Q2", "Q3", "Q4", "Q5 · Cherry-picked"],
        )
        cherry_quintiles = [
            {
                "quintile": str(q),
                "count": int(len(g)),
                "mdape": round(float(g["abs_err_pct"].median()), 2),
                "p80_err": round(float(g["abs_err_pct"].quantile(0.80)), 2),
            }
            for q, g in j_sub.groupby("cp_q", observed=True)
        ]

        # Showcase real concrete examples (Over-claimed vs Hidden Gems)
        def _example_row(r):
            pid = str(r["property_id"])
            return {
                "property_id": pid,
                "url": f"https://www.immoweb.be/nl/zoekertje/property/te-koop/gent/{r.get('postal_code') or '9000'}/{pid}",
                "neighbourhood": str(r["neighbourhood_display"]),
                "property_type": str(r["property_type_label"]),
                "livable_area": int(round(float(r["livable_area"]))) if pd.notna(r["livable_area"]) else None,
                "energy_label": str(r["energy_label"]) if pd.notna(r["energy_label"]) else None,
                "condition": str(r["building_condition"]) if pd.notna(r["building_condition"]) else None,
                "asking_eur": int(round(float(r["price"]))),
                "fair_eur": int(round(float(r["pred_mm_eur"]))),
                "vision_shift_pct": round(float(r["vision_delta_pct"]), 1),
                "dz_quality": round(float(r["jepa_dz_quality"]), 2),
                "energy_sym": round(float(r["jepa_energy_sym"]), 2),
                "had_price_cut": bool(r["had_price_cut"]),
                "duration_days": int(round(float(r["listing_duration_days"]))),
                "title": str(r["title"])[:75] if pd.notna(r["title"]) and str(r["title"]).strip() else None,
            }

        overclaimed_ex = (
            claimed_hi.drop_duplicates(subset=["neighbourhood", "price", "livable_area"])
            .nsmallest(6, "jepa_dz_quality")
        )
        hidden_gems_ex = (
            j_sub[
                j_sub["building_condition"].isin(["GOOD", "TO_BE_DONE_UP", "TO_RENOVATE"])
                & (j_sub["delta_pct"] >= 6.0)
                & (j_sub["vision_delta_pct"] >= 3.0)
            ]
            .drop_duplicates(subset=["neighbourhood", "price", "livable_area"])
            .nlargest(6, "jepa_dz_quality")
        )

        # Property regime breakdown (Residential vs Mixed-Use Commercial vs Multi-Unit Investment)
        regime_labels = {
            "residential": "Standard Residential",
            "mixed_commercial": "Mixed-Use / Store (Handelspand)",
            "live_work": "Live-Work / Practice (Praktijk)",
            "multi_unit_investment": "Multi-Unit / Opbrengsteigendom",
            "rented_yield": "Currently Rented / Yield Pitch",
            "student_kot": "Student Kot",
            "service_flat": "Service Flat / Assisted Living",
        }
        df_err = df.copy()
        df_err["abs_err_pct"] = np.abs(df_err["pred_mm_eur"] - df_err["price"]) / np.maximum(df_err["price"], 1.0) * 100.0
        regime_breakdown = []
        for code, label in regime_labels.items():
            g = df_err[df_err["property_regime"] == code]
            if len(g) < 15:
                continue
            regime_breakdown.append({
                "code": code,
                "label": label,
                "count": int(len(g)),
                "median_asking_eur": int(round(float(g["price"].median()))),
                "mdape": round(float(g["abs_err_pct"].median()), 2),
                "p80_err": round(float(g["abs_err_pct"].quantile(0.80)), 2),
                "avg_duration_days": round(float(g["listing_duration_days"].mean()), 1),
                "median_jepa_energy": round(float(g["jepa_energy_sym"].median()), 3) if g["jepa_energy_sym"].notna().any() else None,
            })

        jepa_forensics = {
            "n_scored": int(len(j_sub)),
            "energy_quintiles": energy_quintiles,
            "fair_priced_controlled": fair_controlled,
            "condition_honesty": condition_honesty,
            "cherry_quintiles": cherry_quintiles,
            "regime_breakdown": regime_breakdown,
            "examples": {
                "overclaimed": [_example_row(r) for _, r in overclaimed_ex.iterrows()],
                "hidden_gems": [_example_row(r) for _, r in hidden_gems_ex.iterrows()],
            },
        }

    insights = {
        "generated_from": meta.get("model_version", "phase7-lean5"),
        "corpus": {
            "total_properties": int(len(df)),
            "normal_properties": int(len(normal_df)),
            "raw_snapshots": int(len(raw)),
            "neighbourhoods_mapped": int(df[df["neighbourhood"] != "Unknown"]["neighbourhood"].nunique()),
            "date_min": str(dated["dt"].min().date()) if len(dated) else None,
            "date_max": str(dated["dt"].max().date()) if len(dated) else None,
            "median_price_eur": int(round(float(df["price"].median()))),
            "median_price_sqm": int(round(float(df["price_per_sqm"].median()))),
        },
        "model_metadata": meta,
        "agency_audit": agency_audit,
        "epc_cliff": epc_breakdown,
        "condition_ladder": condition_ladder,
        "overpricing_forensics": overpricing_forensics,
        "vision_impact": vision_impact,
        "jepa_forensics": jepa_forensics,
        "error_by_band": error_by_band,
        "neighbourhood_profiles": nbh_profiles,
        "monthly_series": monthly_series,
    }

    out_json = PROD_DIR / "market_insights.json"
    out_json.write_text(json.dumps(insights, indent=2), encoding="utf-8")
    print(f"Wrote {out_json} ({out_json.stat().st_size / 1024:.1f} KB)")
    print(f"  Agencies audited (>=40 listings): {len(agency_audit)}")
    print(f"  Neighbourhood profiles: {len(nbh_profiles)}")
    print(f"  Conformal 80% HW: multimodal ±{mm_hw_pct:.1f}%, tabular_text ±{tt_hw_pct:.1f}%")


def _histogram(series: pd.Series, bins: int = 20) -> list[dict]:
    arr = series.dropna().to_numpy(dtype=float)
    if len(arr) == 0:
        return []
    counts, edges = np.histogram(arr, bins=bins)
    return [
        {
            "lo": round(float(edges[i]), 2),
            "hi": round(float(edges[i + 1]), 2),
            "mid": round(float((edges[i] + edges[i + 1]) / 2.0), 2),
            "count": int(counts[i]),
        }
        for i in range(len(counts))
    ]


if __name__ == "__main__":
    main()
