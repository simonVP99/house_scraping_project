"""Unified Prediction Service — delegates parsing to `src/parser.py` and inference to `src/production_pipeline.py`."""
from __future__ import annotations

import os
import re
import sys
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import requests

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC_DIR = os.path.join(_PROJECT_DIR, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from parser import build_row, extract_classified_json  # noqa: E402
from live_vision import LiveVisionEncoder, extract_photo_urls_from_classified  # noqa: E402
from jepa_inspector import CrossModalJEPAInspector, DEFAULT_JEPA_PATH, detect_property_regime  # noqa: E402


SHAP_LABEL_MAP = {
    "livable_area": "Livable area",
    "number_of_bedrooms": "Bedrooms",
    "number_of_bathrooms": "Bathrooms",
    "number_of_rooms": "Rooms",
    "number_of_toilets": "Toilets",
    "age": "Age",
    "year_built": "Year built",
    "building_condition": "Condition",
    "condition_ordinal": "Condition",
    "condition_x_area": "Condition × area",
    "age_x_condition": "Age × condition",
    "energy_label": "Energy label",
    "epc_ordinal": "Energy label",
    "epc_ef_area_penalty": "Low EPC area penalty",
    "primary_energy_consumption": "Energy consumption",
    "dist_from_center": "Distance from center",
    "neighbourhood": "Neighbourhood",
    "property_type_HOUSE": "Property type",
    "property_subtype": "Subtype",
    "postal_code": "Postal code",
    "has_garden": "Garden",
    "garden_surface": "Garden area",
    "has_terrace": "Terrace",
    "terrace_surface": "Terrace area",
    "outdoor_surface_total": "Outdoor surface",
    "log_outdoor_surface": "Outdoor surface",
    "has_swimming_pool": "Swimming pool",
    "parking_indoor": "Indoor parking",
    "parking_outdoor": "Outdoor parking",
    "total_parking": "Parking spaces",
    "facade_count": "Facades",
    "total_floors": "Total floors",
    "has_lift": "Lift",
    "has_double_glazing": "Double glazing",
    "has_photovoltaic_panels": "Solar panels",
    "has_heat_pump": "Heat pump",
    "cadastral_income": "Cadastral income",
    "log_cadastral_income": "Cadastral income",
    "ki_per_sqm": "Cadastral income / m²",
    "kitchen_type": "Kitchen type",
    "kitchen_surface": "Kitchen area",
    "living_room_surface": "Living room area",
    "heating_type": "Heating type",
    "street_type": "Street type",
    "property_size": "Plot size",
    "log_property_size_p1": "Plot size",
    "street_facade_width": "Street facade width",
    "floor_number": "Floor number",
    "monthly_costs": "Monthly costs",
    "is_newly_built": "Newly built",
    "has_basement": "Basement",
    "has_attic": "Attic",
    "flood_zone_type": "Flood zone",
    "renovation_obligation": "Renovation obligation",
    "fl_mobiscore": "Mobiscore",
    "fl_mobiscore_x_area": "Mobiscore × area",
    "fl_heritage_count_150m": "Heritage surroundings",
    "fl_dist_monument_m": "Distance to monument",
    "fl_dist_navigable_water_m": "Distance to water",
    "fl_dist_park_m": "Distance to park",
    "fl_dist_primary_school_m": "Distance to school",
    "fl_no2_2025_ugm3": "Air quality (NO₂)",
    "grb_bldg_footprint": "Building footprint (GRB)",
    "grb_parcel_area": "Parcel area (GRB)",
    "grb_log_parcel_area": "Parcel area (GRB)",
    "grb_open_plot_area": "Open plot area (GRB)",
    "grb_implied_floors": "Implied floors (GRB)",
    "grb_adjoining_bldgs": "Building detachment (GRB)",
    "sp_prior_price": "Local comparable price",
    "sp_log_prior_price": "Local comparable price",
    "sp_idw_price_k15": "Nearby property prices (15-NN)",
    "sp_idw_price_k5": "Immediate property prices (5-NN)",
    "sp_idw_price_k40": "Area property prices (40-NN)",
    "sp_same_type_idw_psqm_k15": "Nearby same-type €/m²",
    "sp_same_type_idw_psqm_k5": "Immediate same-type €/m²",
    "sp_same_type_idw_psqm_k40": "Area same-type €/m²",
    "sp_idw_psqm_k15": "Local neighbourhood €/m²",
    "sp_idw_psqm_k5": "Immediate €/m²",
    "sp_idw_psqm_k40": "Wider area €/m²",
    "sp_type_wijk_psqm": "Neighbourhood €/m²",
    "sp_type_sector_psqm": "Statistical sector €/m²",
    "sp_type_postal_psqm": "Postal code €/m²",
    "te_street_log_psqm": "Street price level",
    "sector_median_income": "Sector median income",
    "log_sector_income_x_area": "Sector income × area",
    "txt_tab_plus_text_psqm": "Description signal",
    "txt_blend_psqm": "Description signal",
    "txt_super_tabresid": "Description quality signal",
    "txt_e5l_a5_price": "Description semantic signal",
    "txt_e5l_a12_psqm": "Description semantic signal",
    "txt_tfidf_a2_price": "Description keyword signal",
    "txt_tfidf_a14_psqm": "Description keyword signal",
    "txt_tfidf_spresid": "Description vs location signal",
    "txt_tfidf_tabresid": "Description quality signal",
    "txt_e5l_tabresid": "Description quality signal",
    "txt_sgd_psqm": "Description signal",
    "vis_blend_psqm": "Photo visual appeal",
    "vis_blend_tabtxt_resid": "Photo condition adjustment",
    "section_exterior": "Listing exterior photos",
    "section_interior": "Listing interior photos",
    "pictures_per_bedroom": "Photo coverage",
}


def _extract_property_id(url: str | None) -> str | None:
    if not url:
        return None
    m = re.search(r"/(\d+)(?:\?|$|#)", str(url))
    return m.group(1) if m else None


def _is_latent_feature(name: str, hide_listing_meta: bool = False) -> bool:
    if (
        name.startswith("e5l_pca_")
        or name.startswith("tfidf_svd_")
        or name.startswith("vis_sig_pca_")
        or name.startswith("vis_q_")
        or name.startswith("vis_so_q_")
        or name.startswith("coord_rot_")
        or name.startswith("belgian_")
        or name.startswith("euribor")
        or name in ("be_mortgage_rate_avg", "yield_curve_slope", "pub_month_sin", "pub_month_cos")
    ):
        return True
    if hide_listing_meta and (
        name.startswith("section_")
        or name.startswith("txt_")
        or name in ("pictures_per_bedroom", "picture_count", "description_length", "title_length")
    ):
        return True
    return False


def format_shap_payload(shap_raw: dict[str, Any] | None, *, is_manual: bool = False) -> dict[str, Any] | None:
    """Convert raw TreeSHAP EUR contributions from `TrackBundle.explain_row_shap` to API schema."""
    if not shap_raw:
        return None
    feature_names = shap_raw["feature_names"]
    eur_vals = shap_raw["shap_eur"]
    base_eur = float(shap_raw["base_value_eur"])

    pairs = []
    latent_text_sum = 0.0
    latent_geo_macro_sum = 0.0
    latent_vis_sum = 0.0

    for feat, val in zip(feature_names, eur_vals):
        v = float(val)
        if _is_latent_feature(feat, hide_listing_meta=is_manual):
            if feat.startswith("e5l_pca_") or feat.startswith("tfidf_svd_") or feat.startswith("txt_"):
                latent_text_sum += v
            elif feat.startswith("vis_") or feat.startswith("section_") or "picture" in feat:
                latent_vis_sum += v
            else:
                latent_geo_macro_sum += v
        else:
            pairs.append((feat, v))

    # Merge contributions that share the same human-readable label (e.g., energy_label + epc_ordinal)
    merged: dict[str, tuple[str, float]] = {}
    for feat, val in pairs:
        label = SHAP_LABEL_MAP.get(feat, feat.replace("_", " ").title())
        if label in merged:
            rep_key, cur_val = merged[label]
            # Keep the key with larger absolute contribution as representative
            merged[label] = (feat if abs(val) > abs(cur_val) else rep_key, cur_val + val)
        else:
            merged[label] = (feat, val)

    ranked = sorted(
        [(label, rep_key, val) for label, (rep_key, val) in merged.items()],
        key=lambda x: abs(x[2]),
        reverse=True,
    )
    shap_items = []
    for label, feat, val in ranked[:15]:
        shap_items.append({
            "feature": label,
            "feature_key": feat,
            "value_k": round(val / 1000.0, 2),
            "value_eur": round(val),
        })

    # Remaining non-top-15 + latent buckets so contributions sum cleanly
    tail_sum = sum(v for _, _, v in ranked[15:]) + latent_text_sum + latent_geo_macro_sum + latent_vis_sum
    if abs(tail_sum) >= 500:
        shap_items.append({
            "feature": "Other & latent signals",
            "feature_key": "_other",
            "value_k": round(tail_sum / 1000.0, 2),
            "value_eur": round(tail_sum),
        })

    return {
        "base_value_k": round(base_eur / 1000.0, 1),
        "base_value_eur": round(base_eur),
        "features": shap_items,
    }


def _clean_json_value(v: Any) -> Any:
    if isinstance(v, (np.floating, float)):
        return None if (np.isnan(v) or np.isinf(v)) else float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def _clean_form_dict(d: dict[str, Any]) -> dict[str, Any]:
    return {k: _clean_json_value(v) for k, v in d.items()}


class PredictionService:
    """Unified prediction service backed by the Phase-7 Lean-5 `ProductionBundle`."""

    def __init__(
        self,
        bundle=None,
        data_loader=None,
        vision_encoder: LiveVisionEncoder | None = None,
        jepa_inspector: CrossModalJEPAInspector | None = None,
    ):
        self.bundle = bundle
        self.data_loader = data_loader
        self.vision_encoder = vision_encoder if vision_encoder is not None else LiveVisionEncoder()
        if jepa_inspector is not None:
            self.jepa_inspector = jepa_inspector
        else:
            try:
                self.jepa_inspector = CrossModalJEPAInspector.load(DEFAULT_JEPA_PATH)
            except Exception:
                self.jepa_inspector = None

    @staticmethod
    def parse_immoweb_html(html: str, url: str | None = None) -> dict[str, Any]:
        """Parse Immoweb HTML using the canonical `src/parser.py` JSON decoder + row builder."""
        data = extract_classified_json(html)
        if not data:
            raise ValueError("Could not find property data on this page.")
        pid = _extract_property_id(url) or str(data.get("id") or "")
        date_str = datetime.now().strftime("%d%m%Y")
        row = build_row(data, pid, date_str)
        # Preserve listing URL, photo URLs for live multimodal scoring, & clean NaNs
        row["url"] = url
        photo_urls = extract_photo_urls_from_classified(data)
        cleaned = _clean_form_dict(row)
        cleaned["_photo_urls"] = photo_urls
        return cleaned

    # Alias for legacy callers/tests
    def _parse_immoweb_html(self, html: str, url: str | None = None) -> dict[str, Any]:
        return self.parse_immoweb_html(html, url)

    def _fetch_page(self, url: str) -> str:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            "Accept-Language": "nl-BE,nl;q=0.9,en;q=0.8",
        }
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        return resp.text

    def _normalize_form_row(self, form_data: dict[str, Any]) -> dict[str, Any]:
        """Normalize user form/manual dict into a raw row compatible with `prepare_base_dataframe`."""
        row = dict(form_data or {})
        # Harmonize optional aliases from manual forms
        if "bedrooms" in row and "number_of_bedrooms" not in row:
            row["number_of_bedrooms"] = row["bedrooms"]
        if "condition" in row and "building_condition" not in row:
            row["building_condition"] = row["condition"]
        if not row.get("property_type"):
            if row.get("property_type_HOUSE") == 1:
                row["property_type"] = "HOUSE"
            elif row.get("property_type_HOUSE") == 0:
                row["property_type"] = "APARTMENT"
            else:
                row["property_type"] = "HOUSE"
        if not row.get("property_subtype"):
            row["property_subtype"] = row["property_type"]
        if not row.get("date_obtained"):
            row["date_obtained"] = datetime.now().strftime("%d%m%Y")
        # Convert numeric strings safely
        for num_col in (
            "price", "livable_area", "property_size", "latitude", "longitude",
            "number_of_bedrooms", "number_of_bathrooms", "number_of_rooms",
            "number_of_toilets", "year_built", "cadastral_income", "garden_surface",
            "terrace_surface", "kitchen_surface", "living_room_surface",
            "primary_energy_consumption", "facade_count", "floor_number",
            "total_floors", "street_facade_width", "parking_indoor", "parking_outdoor",
            "monthly_costs", "picture_count",
        ):
            if num_col in row and isinstance(row[num_col], str):
                s = row[num_col].strip()
                row[num_col] = float(s) if s != "" else None
        return row

    def _score_single_row(
        self,
        row: dict[str, Any],
        vis_bundle: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run Phase-7 production bundle on 1 row and build the unified API response."""
        clean_row = {k: v for k, v in row.items() if not k.startswith("_")}
        asking_price = clean_row.get("price")
        if asking_price is not None:
            try:
                asking_price = float(asking_price)
                if np.isnan(asking_price) or asking_price <= 0:
                    asking_price = None
            except (TypeError, ValueError):
                asking_price = None

        raw_df = pd.DataFrame([clean_row])
        pred_out = self.bundle.predict_raw(raw_df, vis_bundle=vis_bundle, include_shap=True)

        predicted_eur = float(pred_out["predicted_eur"][0])
        low_eur = float(pred_out["low_eur"][0])
        high_eur = float(pred_out["high_eur"][0])
        low_90_eur = float(pred_out["low_90_eur"][0])
        high_90_eur = float(pred_out["high_90_eur"][0])
        hw_pct = float(pred_out["interval_halfwidth_pct"][0])
        track_used = str(pred_out["track_used"][0])
        has_photos = bool(pred_out["has_photos"][0])

        # Compute vision-vs-tabular shift when multimodal track ran so the UI can show photo impact
        vision_shift_eur = None
        vision_shift_pct = None
        if has_photos and vis_bundle is not None and getattr(self.bundle, "tabular_text", None) is not None:
            try:
                tt_out = self.bundle.tabular_text.predict(pred_out["feat_df"], pred_out["deflator"])
                tt_eur = float(tt_out["predicted_eur"][0])
                vision_shift_eur = round(predicted_eur - tt_eur)
                vision_shift_pct = round((predicted_eur - tt_eur) / max(tt_eur, 1.0) * 100.0, 1)
            except Exception:
                pass

        delta_eur = round(predicted_eur - asking_price) if asking_price else None
        delta_pct = round(float((predicted_eur - asking_price) / asking_price * 100.0), 1) if asking_price else None

        # Band position relative to asking price
        band_position = None
        if asking_price is not None:
            if asking_price < low_eur:
                band_position = "below_band"
            elif asking_price > high_eur:
                band_position = "above_band"
            else:
                band_position = "inside_band"

        feat_df = pred_out["feat_df"]
        nbh = None
        if "neighbourhood" in feat_df.columns and pd.notna(feat_df["neighbourhood"].iloc[0]):
            cand = str(feat_df["neighbourhood"].iloc[0])
            if cand and cand != "Unknown":
                nbh = cand
        if not nbh and clean_row.get("latitude") and clean_row.get("longitude") and self.data_loader:
            nbh = self.data_loader.get_neighbourhood_spatial(float(clean_row["latitude"]), float(clean_row["longitude"]))

        address_parts = " ".join(p for p in [str(clean_row.get("street") or "").strip(), str(clean_row.get("house_number") or "").strip()] if p)
        city_parts = " ".join(p for p in [str(clean_row.get("postal_code") or "").strip(), str(clean_row.get("city") or "").strip()] if p)
        address = ", ".join(p for p in [address_parts, city_parts] if p)

        has_text = bool((clean_row.get("description") or "").strip() or (clean_row.get("description_nl") or "").strip())
        shap_payload = format_shap_payload(pred_out.get("shap_raw"), is_manual=not has_text)

        # Cross-Modal JEPA Photo-vs-Specs Honesty diagnostics (when photos ran)
        jepa_diag = None
        if has_photos and vis_bundle is not None and self.jepa_inspector is not None:
            try:
                e5_vec = None
                if has_text and getattr(self.bundle, "e5_encoder", None) is not None:
                    from production_pipeline import build_text_strings
                    texts = build_text_strings(raw_df)
                    e5_vec = self.bundle.e5_encoder.encode(texts)[0]
                jepa_diag = self.jepa_inspector.score_single(
                    feat_df=feat_df,
                    e5_vec=e5_vec,
                    vis_bundle=vis_bundle,
                    claimed_condition=clean_row.get("building_condition"),
                    delta_pct=delta_pct,
                    title=clean_row.get("title"),
                    description=clean_row.get("description"),
                    description_nl=clean_row.get("description_nl"),
                    energy_label=clean_row.get("energy_label"),
                )
            except Exception:
                jepa_diag = None

        # Detect non-standard commercial / mixed-use / investment property regimes
        property_regime = detect_property_regime(
            property_subtype=clean_row.get("property_subtype"),
            title=clean_row.get("title"),
            description=clean_row.get("description"),
            description_nl=clean_row.get("description_nl"),
        )

        result = {
            "property": {
                "property_id": clean_row.get("property_id"),
                "address": address,
                "price": asking_price,
                "livable_area": clean_row.get("livable_area"),
                "bedrooms": clean_row.get("number_of_bedrooms"),
                "year_built": clean_row.get("year_built"),
                "condition": clean_row.get("building_condition"),
                "energy_label": clean_row.get("energy_label"),
                "property_type": clean_row.get("property_type"),
                "property_subtype": clean_row.get("property_subtype"),
                "neighbourhood": nbh,
                "regime": property_regime,
            },
            "prediction": {
                "predicted_price": round(predicted_eur),
                "predicted_eur": round(predicted_eur),
                "low_eur": round(low_eur),
                "high_eur": round(high_eur),
                "low_90_eur": round(low_90_eur),
                "high_90_eur": round(high_90_eur),
                "interval_halfwidth_pct": round(hw_pct, 1),
                "confidence_level": 0.80,
                "band_position": band_position,
                "delta_eur": delta_eur,
                "delta_pct": delta_pct,
                "track": track_used,
                "has_photos": has_photos,
                "n_photos_used": int(vis_bundle.get("n_photos_used", 0)) if (has_photos and vis_bundle) else 0,
                "vision_source": str(vis_bundle.get("source")) if (has_photos and vis_bundle) else None,
                "vision_shift_eur": vision_shift_eur,
                "vision_shift_pct": vision_shift_pct,
                "jepa": jepa_diag,
                "model_version": self.bundle.metadata.get("model_version", "phase7-lean5"),
            },
            "form_data": _clean_form_dict(clean_row),
            "shap": shap_payload,
        }

        self._add_explainability(result, clean_row)
        return result

    def _extract_live_vision(self, parsed: dict[str, Any]) -> dict[str, Any] | None:
        """Attempt live SigLIP2+SO400M photo encoding; return None gracefully on any error."""
        if not self.vision_encoder or not getattr(self.bundle, "multimodal", None):
            return None
        pid = parsed.get("property_id")
        photo_urls = parsed.get("_photo_urls") or []
        if not pid and not photo_urls:
            return None
        try:
            return self.vision_encoder.extract_for_listing(
                property_id=str(pid) if pid else None,
                photo_urls=photo_urls,
            )
        except Exception as e:
            print(f"Live vision fallback to tabular_text ({type(e).__name__}: {e})")
            return None

    def predict_from_url(self, url: str | None):
        """Fetch Immoweb URL, parse via `src/parser.py`, encode photos if available, predict with Phase-7 bundle."""
        if not url or "immoweb.be" not in url:
            return {"error": "Please enter a valid Immoweb URL"}, 400
        if self.bundle is None:
            return {"error": "Production model bundle not loaded"}, 503

        try:
            html = self._fetch_page(url)
            parsed = self.parse_immoweb_html(html, url)
            vis_bundle = self._extract_live_vision(parsed)
            result = self._score_single_row(parsed, vis_bundle=vis_bundle)
            return result, 200
        except requests.exceptions.HTTPError as e:
            if hasattr(e, "response") and e.response is not None and e.response.status_code == 403:
                return {"error": "Access blocked by Immoweb (403). Try again later."}, 403
            return {"error": f"HTTP error: {e}"}, 502
        except ValueError as e:
            return {"error": str(e)}, 422
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}, 500

    def predict_from_form_data(self, form_data: dict[str, Any] | None):
        """Predict from edited listing form or manual property details (reuses cached vision if property_id matches)."""
        if self.bundle is None:
            return {"error": "Production model bundle not loaded"}, 503
        if not form_data or not isinstance(form_data, dict):
            return {"error": "Invalid property payload"}, 400

        try:
            row = self._normalize_form_row(form_data)
            # Reuse cached vision embeddings (zero download) if this form came from a scored Immoweb URL
            vis_bundle = None
            pid = row.get("property_id")
            if pid and self.vision_encoder and getattr(self.bundle, "multimodal", None):
                try:
                    vis_bundle = self.vision_encoder._lookup_cached(str(pid))
                except Exception:
                    vis_bundle = None
            result = self._score_single_row(row, vis_bundle=vis_bundle)
            return result, 200
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}, 500

    def predict_manual(self, form_data: dict[str, Any] | None):
        """Manual prediction endpoint — routed through the unified Phase-7 tabular+text bundle (fixes legacy 503!)."""
        result, status = self.predict_from_form_data(form_data)
        if status == 200:
            # Keep top-level `predicted_price` for backward-compatible clients alongside full structured response
            result["predicted_price"] = result["prediction"]["predicted_price"]
        return result, status

    def predict_batch_parsed(self, parsed_rows: list[dict[str, Any]]) -> dict[str, Any]:
        """Batch prediction for `LiveListingsService` using the unified Phase-7 bundle."""
        if self.bundle is None:
            raise RuntimeError("Production model bundle not loaded")
        raw_df = pd.DataFrame([self._normalize_form_row(r) for r in parsed_rows])
        return self.bundle.predict_raw(raw_df, include_shap=False)

    def _add_explainability(self, result: dict[str, Any], parsed: dict[str, Any]) -> None:
        if not self.data_loader:
            return
        nbh = result["property"].get("neighbourhood")
        prop = {
            "neighbourhood": nbh,
            "postal_code": parsed.get("postal_code"),
            "livable_area": parsed.get("livable_area"),
            "number_of_bedrooms": parsed.get("number_of_bedrooms"),
            "property_type": parsed.get("property_type"),
            "property_type_HOUSE": 1 if parsed.get("property_type") == "HOUSE" else 0,
            "building_condition": parsed.get("building_condition"),
            "energy_label": parsed.get("energy_label"),
            "year_built": parsed.get("year_built"),
            "age": (datetime.now().year - parsed["year_built"]) if parsed.get("year_built") else None,
            "price": parsed.get("price"),
            "garden_surface": parsed.get("garden_surface"),
            "terrace_surface": parsed.get("terrace_surface"),
            "cadastral_income": parsed.get("cadastral_income"),
            "has_garden": parsed.get("has_garden"),
            "has_terrace": parsed.get("has_terrace"),
            "has_swimming_pool": parsed.get("has_swimming_pool"),
            "has_lift": parsed.get("has_lift"),
            "has_double_glazing": parsed.get("has_double_glazing"),
            "has_photovoltaic_panels": parsed.get("has_photovoltaic_panels"),
            "has_heat_pump": parsed.get("has_heat_pump"),
            "parking_indoor": parsed.get("parking_indoor"),
            "parking_outdoor": parsed.get("parking_outdoor"),
        }
        try:
            result["comparables"] = self.data_loader.find_comparables(prop)
        except Exception:
            result["comparables"] = None
        try:
            result["context"] = self.data_loader.get_feature_context(prop)
        except Exception:
            result["context"] = []
