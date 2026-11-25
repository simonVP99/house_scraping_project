"""Full property dossier assembler — combines raw Immoweb snapshots, Phase-7 OOF valuation,
Flanders/Gent/GRB environmental & cadastral enrichment, Cross-Modal JEPA photo honesty,
comparable sales, and locally cached listing photos.
"""
from __future__ import annotations

import math
import os
import re
from typing import Any

import numpy as np
import pandas as pd

from api.config import PROJECT_DIR

RAW_PARQUET_PATH = os.path.join(PROJECT_DIR, "outputs", "property_data_full.parquet")
COORD_ENRICH_PATH = os.path.join(PROJECT_DIR, "external_data", "flanders", "ghent_coord_enrichment.parquet")
PHOTOS_ROOT = os.path.join(PROJECT_DIR, "external_data", "vision_full", "photos")


def _clean(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        if math.isnan(v) or math.isinf(v):
            return None
        return float(v)
    if isinstance(v, pd.Timestamp):
        return v.strftime("%Y-%m-%d")
    if isinstance(v, str):
        s = v.strip()
        return s if s and s.lower() not in ("nan", "none", "null") else None
    return v


def _clean_int(v: Any) -> int | None:
    val = _clean(v)
    if val is None:
        return None
    try:
        return int(round(float(val)))
    except Exception:
        return None


def _clean_float(v: Any, ndigits: int = 2) -> float | None:
    val = _clean(v)
    if val is None:
        return None
    try:
        return round(float(val), ndigits)
    except Exception:
        return None


def _clean_bool(v: Any) -> bool | None:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, float, np.integer, np.floating)):
        return bool(int(v))
    if isinstance(v, str):
        low = v.strip().lower()
        if low in ("true", "1", "yes", "ja"):
            return True
        if low in ("false", "0", "no", "nee"):
            return False
    return None


class PropertyDossierService:
    """Assembles the complete information overview for any Ghent property ID."""

    def __init__(self, data_loader=None, prediction_service=None):
        self.data_loader = data_loader
        self.prediction_service = prediction_service
        self._coord_df: pd.DataFrame | None = None
        self._coord_tree = None
        self._coord_rad: np.ndarray | None = None
        self._coord_by_key: dict[tuple[float, float], int] = {}

    def _ensure_coord_index(self) -> None:
        if self._coord_df is not None:
            return
        if not os.path.exists(COORD_ENRICH_PATH):
            self._coord_df = pd.DataFrame()
            return
        cdf = pd.read_parquet(COORD_ENRICH_PATH)
        self._coord_df = cdf
        keys = {}
        for i, (la, lo) in enumerate(zip(cdf["lat_r5"].to_numpy(), cdf["lon_r5"].to_numpy())):
            keys[(round(float(la), 5), round(float(lo), 5))] = i
        self._coord_by_key = keys
        try:
            from scipy.spatial import cKDTree
            coords = np.column_stack([
                cdf["lat_r5"].to_numpy(dtype=float) * 111.32,
                cdf["lon_r5"].to_numpy(dtype=float) * 111.32 * math.cos(math.radians(51.05)),
            ])
            self._coord_tree = cKDTree(coords)
        except Exception:
            self._coord_tree = None

    def _lookup_coord_enrichment(self, lat: float | None, lon: float | None) -> dict[str, Any]:
        if lat is None or lon is None:
            return {}
        self._ensure_coord_index()
        if self._coord_df is None or self._coord_df.empty:
            return {}
        key = (round(float(lat), 5), round(float(lon), 5))
        idx = self._coord_by_key.get(key)
        if idx is None and self._coord_tree is not None:
            q = [float(lat) * 111.32, float(lon) * 111.32 * math.cos(math.radians(51.05))]
            dist_km, nn_idx = self._coord_tree.query(q, k=1)
            if dist_km <= 0.25:
                idx = int(nn_idx)
        if idx is None:
            return {}
        row = self._coord_df.iloc[idx]
        return {col: _clean(row[col]) for col in self._coord_df.columns if col not in ("lat_r5", "lon_r5")}

    def list_local_photos(self, property_id: str) -> list[str]:
        pid = "".join(ch for ch in str(property_id) if ch.isdigit())
        if not pid:
            return []
        pdir = os.path.join(PHOTOS_ROOT, pid)
        if not os.path.isdir(pdir):
            return []
        files = sorted(
            f for f in os.listdir(pdir)
            if re.match(r"^[0-9]{1,3}\.(jpg|jpeg|png|webp)$", f, re.IGNORECASE)
        )
        return [f"/api/properties/{pid}/photos/{fname}" for fname in files[:8]]

    def get_photo_path(self, property_id: str, filename: str) -> str | None:
        pid = "".join(ch for ch in str(property_id) if ch.isdigit())
        if not pid or not re.match(r"^[0-9]{1,3}\.(jpg|jpeg|png|webp)$", filename, re.IGNORECASE):
            return None
        path = os.path.join(PHOTOS_ROOT, pid, filename)
        if os.path.isfile(path):
            return path
        return None

    def _load_raw_snapshots(self, pid: str) -> pd.DataFrame:
        if not os.path.exists(RAW_PARQUET_PATH):
            return pd.DataFrame()
        try:
            df = pd.read_parquet(RAW_PARQUET_PATH, filters=[("property_id", "==", str(pid))])
            if not df.empty and "date_obtained" in df.columns:
                df["_dt"] = pd.to_datetime(df["date_obtained"], format="%d%m%Y", errors="coerce")
                df = df.sort_values("_dt")
            return df
        except Exception:
            return pd.DataFrame()

    def _lookup_oof_row(self, pid: str) -> dict[str, Any]:
        adf = getattr(self.data_loader, "analytics_df", None) if self.data_loader is not None else None
        if not isinstance(adf, pd.DataFrame) or adf.empty or "property_id" not in adf.columns:
            return {}
        m = adf["property_id"].astype(str) == str(pid)
        if not m.any():
            return {}
        row = adf.loc[m].iloc[-1]
        return {col: row[col] for col in adf.columns}

    def get_dossier(self, property_id: str) -> dict[str, Any] | None:
        pid = "".join(ch for ch in str(property_id) if ch.isdigit())
        if not pid:
            return None

        oof = self._lookup_oof_row(pid)
        snaps = self._load_raw_snapshots(pid)
        if not oof and snaps.empty:
            return None

        latest = snaps.iloc[-1].to_dict() if not snaps.empty else {}

        def _pick(*keys):
            for k in keys:
                if k in latest and _clean(latest[k]) is not None:
                    return latest[k]
                if k in oof and _clean(oof[k]) is not None:
                    return oof[k]
            return None

        # Price history timeline across all scrapes
        history = []
        if not snaps.empty:
            for _, srow in snaps.iterrows():
                dt = srow.get("_dt")
                history.append({
                    "date": dt.strftime("%Y-%m-%d") if pd.notna(dt) else _clean(srow.get("date_obtained")),
                    "price_eur": _clean_int(srow.get("price")),
                    "price_old_eur": _clean_int(srow.get("price_old_value")),
                    "is_under_option": bool(_clean_bool(srow.get("is_under_option")) or False),
                    "is_sold_or_rented": bool(_clean_bool(srow.get("is_sold_or_rented")) or False),
                    "view_count": _clean_int(srow.get("view_count")),
                    "bookmark_count": _clean_int(srow.get("bookmark_count")),
                })

        # Asking & Phase-7 valuation
        asking_eur = _clean_int(oof.get("price_eur")) or _clean_int(_pick("price"))
        if asking_eur is not None and asking_eur < 5000 and "price" in oof:
            asking_eur = _clean_int(float(oof["price"]) * 1000.0)
        pred_mm = _clean_int(oof.get("pred_mm_eur"))
        pred_tt = _clean_int(oof.get("pred_tt_eur"))
        hw_frac = 0.145
        low_eur = int(round(pred_mm / (1.0 + hw_frac))) if pred_mm else None
        high_eur = int(round(pred_mm * (1.0 + hw_frac))) if pred_mm else None
        delta_eur = (pred_mm - asking_eur) if (pred_mm and asking_eur) else None
        delta_pct = round(delta_eur / asking_eur * 100.0, 1) if (delta_eur is not None and asking_eur) else None
        band_position = None
        if asking_eur and low_eur and high_eur:
            band_position = (
                "below_band" if asking_eur < low_eur
                else ("above_band" if asking_eur > high_eur else "inside_band")
            )

        area = _clean_float(_pick("livable_area"), 1)
        ask_psm = int(round(asking_eur / area)) if (asking_eur and area and area > 0) else None
        fair_psm = int(round(pred_mm / area)) if (pred_mm and area and area > 0) else None

        # Address & geo
        street = _clean(_pick("street"))
        house_num = _clean(_pick("house_number"))
        box = _clean(_pick("box"))
        postal = _clean(_pick("postal_code")) or "9000"
        city = _clean(_pick("city")) or "Gent"
        nbh_display = _clean(oof.get("neighbourhood_display")) or _clean(oof.get("neighbourhood"))
        nbh_key = _clean(oof.get("neighbourhood"))
        left = " ".join(p for p in (street, house_num, f"bus {box}" if box else None) if p)
        address = f"{left}, {postal} {city}" if left else f"{nbh_display or city} · {postal} {city}"

        lat = _clean_float(_pick("latitude"), 6)
        lon = _clean_float(_pick("longitude"), 6)
        env = self._lookup_coord_enrichment(lat, lon)

        # Availability summary
        days_ago = _clean_int(oof.get("days_since_last_seen"))
        latest_scrape = _clean(oof.get("latest_scrape_date"))
        first_seen = _clean(oof.get("first_seen_str")) or (history[0]["date"] if history else None)
        last_seen = _clean(oof.get("last_seen_str")) or (history[-1]["date"] if history else None)
        is_opt = bool(_clean_bool(oof.get("is_under_option_latest")) or _clean_bool(latest.get("is_under_option")) or False)
        is_sold = bool(_clean_bool(oof.get("is_sold_or_rented")) or _clean_bool(latest.get("is_sold_or_rented")) or False)
        if is_sold:
            avail_status = "sold_or_rented"
        elif is_opt:
            avail_status = "under_option"
        elif days_ago is not None and days_ago <= 3:
            avail_status = "active_recent"
        elif days_ago is not None and days_ago <= 14:
            avail_status = "seen_recently"
        else:
            avail_status = "likely_gone"

        # Regime detection
        title = _clean(_pick("title"))
        desc_nl = _clean(latest.get("description_nl"))
        desc_en = _clean(latest.get("description_en"))
        desc_fr = _clean(latest.get("description_fr"))
        desc_main = desc_nl or _clean(latest.get("description")) or desc_en or desc_fr or _clean(oof.get("description_snippet"))
        subtype = _clean(_pick("property_subtype"))
        ptype = _clean(oof.get("property_type_label")) or _clean(_pick("property_type"))
        if ptype:
            ptype = ptype.upper()

        regime_info = None
        try:
            from jepa_inspector import detect_property_regime
            regime_info = detect_property_regime(
                property_subtype=subtype,
                title=title,
                description=desc_main,
                description_nl=desc_nl,
            )
        except Exception:
            regime_info = {"code": _clean(oof.get("property_regime")) or "residential", "label": "Standard Residential", "is_non_standard": False, "advisory": None}

        # Live JEPA breakdown if available via inspector + offline cache
        jepa_detail = None
        jepa_dz = _clean_float(oof.get("jepa_dz_quality"), 2)
        jepa_e = _clean_float(oof.get("jepa_energy_sym"), 2)
        jepa_cherry = _clean_float(oof.get("jepa_cherry_spread"), 2)
        if self.prediction_service is not None:
            insp = getattr(self.prediction_service, "jepa_inspector", None)
            venc = getattr(self.prediction_service, "vision_encoder", None)
            if insp is not None and venc is not None:
                try:
                    vis_cached = venc._lookup_cached(pid)
                    if isinstance(vis_cached, dict):
                        feat_row = pd.DataFrame([{k: v for k, v in oof.items() if isinstance(v, (int, float, np.number, bool, np.bool_))}])
                        cand_jepa = insp.score_single(
                            feat_df=feat_row,
                            e5_vec=None,
                            vis_bundle=vis_cached,
                            claimed_condition=_clean(_pick("building_condition")),
                            delta_pct=delta_pct,
                        )
                        if isinstance(cand_jepa, dict):
                            jepa_detail = cand_jepa
                except Exception:
                    jepa_detail = None

        # Comparables from DataLoader
        comparables = None
        if self.data_loader is not None and hasattr(self.data_loader, "find_comparables"):
            try:
                comp_input = {
                    "neighbourhood": nbh_key or nbh_display,
                    "postal_code": postal,
                    "livable_area": area,
                    "bedrooms": _clean_int(_pick("number_of_bedrooms")),
                    "property_type": ptype or "HOUSE",
                    "condition": _clean(_pick("building_condition")),
                    "energy_label": _clean(_pick("energy_label")),
                    "latitude": lat,
                    "longitude": lon,
                }
                cand_comp = self.data_loader.find_comparables(comp_input)
                if isinstance(cand_comp, dict):
                    comparables = cand_comp
            except Exception:
                comparables = None

        # Amenities list (all true boolean features)
        amenity_flags = [
            ("Garden", _clean_bool(_pick("has_garden")), _clean_int(_pick("garden_surface")), "m²"),
            ("Terrace", _clean_bool(_pick("has_terrace")), _clean_int(_pick("terrace_surface")), "m²"),
            ("Solar panels", _clean_bool(_pick("has_photovoltaic_panels")), None, None),
            ("Heat pump", _clean_bool(_pick("has_heat_pump")), None, None),
            ("Double glazing", _clean_bool(_pick("has_double_glazing")), None, None),
            ("Thermal solar", _clean_bool(latest.get("has_thermic_panels")), None, None),
            ("Elevator", _clean_bool(_pick("has_lift")), None, None),
            ("Basement", _clean_bool(latest.get("has_basement")), _clean_int(latest.get("basement_surface")), "m²"),
            ("Attic", _clean_bool(latest.get("has_attic")), _clean_int(latest.get("attic_surface")), "m²"),
            ("Air conditioning", _clean_bool(latest.get("has_air_conditioning")), None, None),
            ("Dressing room", _clean_bool(latest.get("has_dressing_room")), None, None),
            ("Office", _clean_bool(latest.get("has_office")), _clean_int(latest.get("office_surface")), "m²"),
            ("Workspace", _clean_bool(latest.get("has_workspace")), _clean_int(latest.get("workspace_surface")), "m²"),
            ("Fireplace", _clean_bool(latest.get("fireplace_exists")), None, None),
            ("Swimming pool", _clean_bool(latest.get("has_swimming_pool")), None, None),
            ("Sauna", _clean_bool(latest.get("has_sauna")), None, None),
            ("Jacuzzi", _clean_bool(latest.get("has_jacuzzi")), None, None),
            ("Alarm / secure access", _clean_bool(latest.get("has_secure_access_alarm")), None, None),
            ("Videophone", _clean_bool(latest.get("has_visiophone")), None, None),
            ("Disabled access", _clean_bool(latest.get("has_disabled_access")), None, None),
            ("Furnished", _clean_bool(latest.get("is_furnished")), None, None),
        ]
        amenities = []
        for label, flag, surf, unit in amenity_flags:
            if flag:
                amenities.append(f"{label} ({surf} {unit})" if surf and surf > 0 else label)

        # Parking summary
        p_in = _clean_int(_pick("parking_indoor", "indoor_parking_count"))
        p_out = _clean_int(_pick("parking_outdoor", "outdoor_parking_count"))
        p_box = _clean_int(latest.get("parking_closed_box"))

        return {
            "property_id": pid,
            "url": f"https://www.immoweb.be/nl/zoekertje/property/te-koop/gent/{postal}/{pid}",
            "address": address,
            "street": street,
            "house_number": house_num,
            "postal_code": postal,
            "city": city,
            "neighbourhood": nbh_display,
            "neighbourhood_key": nbh_key,
            "statbel_sector": _clean(oof.get("statbel_sector")),
            "latitude": lat,
            "longitude": lon,
            "dist_from_center_km": _clean_float(oof.get("dist_from_center"), 2),
            "street_type": _clean(oof.get("street_type")),
            "title": title,
            "description": desc_main,
            "description_language": _clean(latest.get("description_language")),
            "photos": self.list_local_photos(pid),
            "picture_count": _clean_int(latest.get("picture_count")),
            "floor_plan_count": _clean_int(latest.get("floor_plan_count")),
            "has_virtual_tour": _clean_bool(latest.get("has_virtual_tour")),
            "valuation": {
                "asking_eur": asking_eur,
                "predicted_mm_eur": pred_mm,
                "predicted_tt_eur": pred_tt,
                "low_80_eur": low_eur,
                "high_80_eur": high_eur,
                "interval_halfwidth_pct": 14.5,
                "band_position": band_position,
                "delta_eur": delta_eur,
                "delta_pct": delta_pct,
                "asking_psm": ask_psm,
                "fair_psm": fair_psm,
                "vision_delta_eur": _clean_int(oof.get("vision_delta_eur")),
                "vision_delta_pct": _clean_float(oof.get("vision_delta_pct"), 1),
                "ens_std_mm": _clean_float(oof.get("ens_std_mm"), 4),
            },
            "jepa": {
                "dz_quality": jepa_dz,
                "energy_sym": jepa_e,
                "cherry_spread": jepa_cherry,
                "detail": jepa_detail,
            },
            "regime": regime_info,
            "availability": {
                "status": avail_status,
                "first_seen": first_seen,
                "last_seen": last_seen,
                "days_since_last_seen": days_ago,
                "latest_scrape_date": latest_scrape,
                "listing_duration_days": _clean_int(oof.get("listing_duration_days")),
                "n_snapshots": _clean_int(oof.get("n_snapshots")) or len(history),
                "is_under_option": is_opt,
                "is_sold_or_rented": is_sold,
                "had_price_cut": bool(_clean_bool(oof.get("had_price_cut")) or False),
                "price_cut_eur": _clean_int(oof.get("price_cut_eur")),
                "price_cut_pct": _clean_float(oof.get("price_cut_pct"), 1),
                "publication_date": _clean(latest.get("publication_date")),
                "last_modification_date": _clean(latest.get("last_modification_date")),
                "view_count": _clean_int(latest.get("view_count")),
                "bookmark_count": _clean_int(latest.get("bookmark_count")),
                "agency": _clean(oof.get("agency_clean")) or _clean(latest.get("immo_office_name")),
                "seller_type": _clean(latest.get("seller_type")),
                "price_history": history,
            },
            "structure": {
                "property_type": ptype,
                "property_subtype": subtype,
                "building_condition": _clean(_pick("building_condition")),
                "construction_type": _clean(latest.get("construction_type")),
                "year_built": _clean_int(_pick("year_built")),
                "age": _clean_int(oof.get("age")),
                "livable_area": area,
                "property_size": _clean_float(_pick("property_size"), 1),
                "living_room_surface": _clean_float(latest.get("living_room_surface"), 1),
                "kitchen_surface": _clean_float(latest.get("kitchen_surface"), 1),
                "kitchen_type": _clean(latest.get("kitchen_type")),
                "facade_count": _clean_int(latest.get("facade_count")),
                "street_facade_width": _clean_float(latest.get("street_facade_width"), 1),
                "floor_number": _clean_int(latest.get("floor_number")),
                "total_floors": _clean_int(latest.get("total_floors")),
                "bedrooms": _clean_int(_pick("number_of_bedrooms")),
                "bathrooms": _clean_int(_pick("number_of_bathrooms")),
                "toilets": _clean_int(latest.get("number_of_toilets")),
                "shower_rooms": _clean_int(latest.get("shower_room_count")),
                "rooms": _clean_int(latest.get("number_of_rooms")),
                "garden_surface": _clean_float(_pick("garden_surface"), 1),
                "garden_orientation": _clean(latest.get("garden_orientation")),
                "terrace_surface": _clean_float(_pick("terrace_surface"), 1),
                "terrace_orientation": _clean(latest.get("terrace_orientation")),
                "parking_indoor": p_in,
                "parking_outdoor": p_out,
                "parking_closed_box": p_box,
                "amenities": amenities,
            },
            "energy_compliance": {
                "energy_label": _clean(_pick("energy_label")),
                "primary_energy_consumption": _clean_int(_pick("primary_energy_consumption")),
                "carbon_emission": _clean_int(latest.get("carbon_emission")),
                "epc_cert_year": _clean_int(latest.get("epc_cert_year")),
                "heating_type": _clean(latest.get("heating_type")),
                "has_double_glazing": _clean_bool(_pick("has_double_glazing")),
                "has_photovoltaic_panels": _clean_bool(_pick("has_photovoltaic_panels")),
                "has_heat_pump": _clean_bool(_pick("has_heat_pump")),
                "renovation_obligation": _clean_bool(latest.get("renovation_obligation")),
                "electrical_cert": _clean(latest.get("cert_has_electrical_installation")),
                "asbestos_cert": _clean(latest.get("cert_has_asbestos")),
                "oil_tank_status": _clean(latest.get("cert_oil_tank_status")),
                "p_score": _clean(latest.get("p_score")),
                "g_score": _clean(latest.get("g_score")),
                "flood_zone_type": _clean(latest.get("flood_zone_type")),
                "land_use_designation": _clean(latest.get("land_use_designation")),
                "permit_obtained": _clean_bool(latest.get("permit_is_obtained")),
                "urban_planning_breaches": _clean_bool(latest.get("breaches_urban_planning")),
            },
            "financial": {
                "cadastral_income": _clean_int(_pick("cadastral_income")),
                "monthly_costs": _clean_int(latest.get("monthly_costs")),
                "is_subject_to_vat": _clean_bool(latest.get("is_subject_to_vat")),
                "has_starting_price": _clean_bool(latest.get("has_starting_price")),
                "is_public_sale": _clean_bool(latest.get("is_public_sale")),
                "is_biddit_sale": _clean_bool(latest.get("is_biddit_sale")),
                "is_newly_built": _clean_bool(latest.get("is_newly_built")),
                "is_investment_property": _clean_bool(latest.get("is_investment_property")),
                "expected_monthly_rental_income": _clean_int(latest.get("expected_monthly_rental_income")),
                "availability_period_type": _clean(latest.get("availability_period_type")),
            },
            "environment": {
                "mobiscore": _clean_float(env.get("fl_mobiscore"), 2),
                "noise_total_db": _clean_float(env.get("gent_noise_total_lden_db"), 1),
                "noise_road_db": _clean_float(env.get("gent_noise_road_lden_db"), 1),
                "noise_night_db": _clean_float(env.get("gent_noise_night_db"), 1),
                "no2_ugm3": _clean_float(env.get("fl_no2_2025_ugm3"), 1),
                "pm25_ugm3": _clean_float(env.get("fl_pm25_2025_ugm3"), 1),
                "urban_heat_c": _clean_float(env.get("fl_urban_heat_c"), 2),
                "buurtgroen_pct": _clean_float(env.get("fl_buurtgroen_10m"), 1),
                "trees_100m": _clean_int(env.get("gent_trees_100m")),
                "dist_park_m": _clean_int(env.get("gent_dist_park_m")),
                "parks_600m": _clean_int(env.get("gent_parks_within_600m")),
                "dist_transit_m": _clean_int(env.get("gent_dist_transit_m")),
                "transit_stops_400m": _clean_int(env.get("gent_transit_stops_400m")),
                "dist_tram_m": _clean_int(env.get("gent_dist_tram_m")),
                "dist_rail_station_m": _clean_int(env.get("gent_dist_rail_station_m")),
                "dist_school_m": _clean_int(env.get("gent_dist_school_m")),
                "schools_500m": _clean_int(env.get("gent_schools_500m")),
                "dist_water_m": _clean_int(env.get("fl_dist_navigable_water_m")),
                "dist_motorway_m": _clean_int(env.get("fl_dist_motorway_m")),
                "in_lez": _clean_bool(env.get("gent_in_lez")),
                "in_autovrij": _clean_bool(env.get("gent_in_autovrij")),
                "in_kotzone": _clean_bool(env.get("gent_in_kotzone")),
                "parking_zone": _clean(env.get("gent_parking_zone")),
                "circulatie_sector": _clean(env.get("gent_circulatie_sector")),
                "cambio_500m": _clean_int(env.get("gent_cambio_500m")),
                "ev_chargers_400m": _clean_int(env.get("gent_ev_chargers_400m")),
                "is_protected_monument": _clean_bool(env.get("fl_is_protected_monument")),
                "is_heritage_inventory": _clean_bool(env.get("fl_is_heritage_inventory")),
                "in_protected_townscape": _clean_bool(env.get("fl_in_protected_townscape")),
                "flood_fluvial_class": _clean(env.get("fl_flood_fluvial_class")),
                "flood_pluvial_class": _clean(env.get("fl_flood_pluvial_class")),
                "grb_parcel_area": _clean_float(env.get("grb_parcel_area"), 1),
                "grb_footprint": _clean_float(env.get("grb_bldg_footprint"), 1),
                "grb_adjoining_bldgs": _clean_int(env.get("grb_adjoining_bldgs")),
                "grb_footprint_to_parcel": _clean_float(env.get("grb_footprint_to_parcel"), 2),
            },
            "comparables": comparables,
        }
