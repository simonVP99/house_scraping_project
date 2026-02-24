"""Prediction service - handles URL fetch/parse and model inference."""
import re
import json
import math
import numpy as np
import pandas as pd
import requests
from datetime import datetime


class PredictionService:
    def __init__(self, ag_predictor=None, old_model=None, old_options=None,
                 old_features=None, e5_model=None, e5_tokenizer=None,
                 e5_device=None, pca_model=None, data_loader=None):
        self.ag_predictor = ag_predictor
        self.old_model = old_model
        self.old_options = old_options or {}
        self.old_features = old_features or []
        self.e5_model = e5_model
        self.e5_tokenizer = e5_tokenizer
        self.e5_device = e5_device
        self.pca_model = pca_model
        self.data_loader = data_loader

    def predict_from_url(self, url):
        """Fetch Immoweb URL, parse, predict. Returns (result_dict, status_code)."""
        if not url or "immoweb.be" not in url:
            return {"error": "Please enter a valid Immoweb URL"}, 400

        if self.ag_predictor is None:
            return {"error": "AutoGluon model not loaded"}, 503

        try:
            html = self._fetch_page(url)
            parsed = self._parse_immoweb_html(html, url)
            features_df, asking_price = self._prepare_features(parsed)
            predicted_k = float(self.ag_predictor.predict(features_df).values[0])
            predicted_eur = predicted_k * 1000

            result = {
                "property": {
                    "property_id": parsed.get("property_id"),
                    "address": f"{parsed.get('street', '')} {parsed.get('house_number', '')}, {parsed.get('postal_code', '')} {parsed.get('city', '')}".strip(),
                    "price": asking_price,
                    "livable_area": parsed.get("livable_area"),
                    "bedrooms": parsed.get("number_of_bedrooms"),
                    "year_built": parsed.get("year_built"),
                    "condition": parsed.get("building_condition"),
                    "energy_label": parsed.get("energy_label"),
                    "property_type": parsed.get("property_type"),
                    "property_subtype": parsed.get("property_subtype"),
                    "neighbourhood": None,
                },
                "prediction": {
                    "predicted_price": round(predicted_eur),
                    "delta_eur": round(predicted_eur - asking_price) if asking_price else None,
                    "delta_pct": round(float((predicted_eur - asking_price) / asking_price * 100), 1) if asking_price else None,
                },
                "form_data": parsed,
            }

            # Add neighbourhood from spatial lookup
            lat, lon = parsed.get("latitude"), parsed.get("longitude")
            if lat and lon and self.data_loader:
                nbh = self.data_loader.get_neighbourhood_spatial(lat, lon)
                if nbh:
                    result["property"]["neighbourhood"] = nbh

            return result, 200

        except requests.exceptions.HTTPError as e:
            if hasattr(e, "response") and e.response is not None and e.response.status_code == 403:
                return {"error": "Access blocked by Immoweb (403). Try again later."}, 403
            return {"error": f"HTTP error: {e}"}, 502
        except ValueError as e:
            return {"error": str(e)}, 422
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}, 500

    def predict_manual(self, form_data):
        """Manual form prediction using old XGBoost model. Returns (result_dict, status_code)."""
        if self.old_model is None:
            return {"error": "No model loaded for manual prediction"}, 503

        try:
            input_dict = dict(form_data)
            # Compute dist_from_center
            lat = input_dict.get("latitude", 0)
            lon = input_dict.get("longitude", 0)
            input_dict["dist_from_center"] = self._haversine_dist(lat, lon)
            input_dict["distance_to_street"] = 0

            df_input = pd.DataFrame([input_dict])
            for col in self.old_features:
                if col not in df_input.columns:
                    df_input[col] = 0

            log_pred = self.old_model.predict(df_input)[0]
            price = float(np.exp(log_pred))
            return {"predicted_price": round(price)}, 200
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}, 500

    def _fetch_page(self, url):
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9,nl;q=0.8,fr;q=0.7",
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        return response.text

    def _parse_immoweb_html(self, html, url):
        json_match = re.search(r"window\.classified\s*=\s*(\{.*?\});", html, re.DOTALL)
        if not json_match:
            raise ValueError("Could not find property data on this page.")

        data = json.loads(json_match.group(1))
        prop = data.get("property", {})
        price_data = data.get("price", {})
        txn = data.get("transaction", {})
        sale = txn.get("sale", {}) or {}
        certs = txn.get("certificates", {}) or {}
        pub = data.get("publication", {}) or {}
        flags = data.get("flags", {}) or {}
        customers = data.get("customers", [])
        loc = prop.get("location", {}) or {}
        bld = prop.get("building", {}) or {}
        energy = prop.get("energy", {}) or {}
        land = prop.get("land", {}) or {}
        cperm = prop.get("constructionPermit", {}) or {}
        kitch = prop.get("kitchen", {}) or {}
        livr = prop.get("livingRoom", {}) or {}

        url_parts = url.rstrip("/").split("/")
        property_id = url_parts[-1] if url_parts[-1].isdigit() else None

        return {
            "property_id": property_id,
            "date_obtained": datetime.now().strftime("%Y-%m-%d"),
            "price": price_data.get("mainValue", price_data.get("price")),
            "property_type": prop.get("type", data.get("type")),
            "property_subtype": prop.get("subtype"),
            "postal_code": loc.get("postalCode"),
            "city": loc.get("locality"),
            "street": loc.get("street"),
            "house_number": loc.get("number"),
            "latitude": loc.get("latitude"),
            "longitude": loc.get("longitude"),
            "location_type": loc.get("type"),
            "floor_number": loc.get("floor"),
            "year_built": bld.get("constructionYear"),
            "building_condition": bld.get("condition"),
            "total_floors": bld.get("floorCount"),
            "facade_count": bld.get("facadeCount"),
            "street_facade_width": bld.get("streetFacadeWidth"),
            "livable_area": prop.get("netHabitableSurface"),
            "property_size": land.get("surface"),
            "kitchen_surface": kitch.get("surface"),
            "living_room_surface": livr.get("surface"),
            "number_of_bedrooms": prop.get("bedroomCount"),
            "number_of_bathrooms": prop.get("bathroomCount"),
            "number_of_toilets": prop.get("toiletCount"),
            "shower_room_count": prop.get("showerRoomCount"),
            "has_living_room": prop.get("hasLivingRoom"),
            "has_attic": prop.get("hasAttic"),
            "has_basement": prop.get("hasBasement"),
            "has_terrace": prop.get("hasTerrace"),
            "terrace_surface": prop.get("terraceSurface"),
            "has_garden": prop.get("hasGarden"),
            "garden_surface": prop.get("gardenSurface"),
            "has_swimming_pool": prop.get("hasSwimmingPool"),
            "has_lift": prop.get("hasLift"),
            "has_dressing_room": prop.get("hasDressingRoom"),
            "has_air_conditioning": prop.get("hasAirConditioning"),
            "has_caretaker": prop.get("hasCaretakerOrConcierge"),
            "kitchen_type": kitch.get("type"),
            "monthly_costs": prop.get("monthlyCosts"),
            "parking_indoor": prop.get("parkingCountIndoor"),
            "parking_outdoor": prop.get("parkingCountOutdoor"),
            "energy_label": certs.get("epcScore"),
            "primary_energy_consumption": certs.get("primaryEnergyConsumptionPerSqm"),
            "heating_type": energy.get("heatingType"),
            "has_photovoltaic_panels": energy.get("hasPhotovoltaicPanels"),
            "has_double_glazing": energy.get("hasDoubleGlazing"),
            "has_heat_pump": energy.get("hasHeatPump"),
            "has_thermic_panels": energy.get("hasThermicPanels"),
            "flood_zone_type": cperm.get("floodZoneType"),
            "p_score": cperm.get("pScore"),
            "g_score": cperm.get("gScore"),
            "cadastral_income": sale.get("cadastralIncome"),
            "is_furnished": sale.get("isFurnished"),
            "renovation_obligation": certs.get("renovationObligation"),
            "publication_date": pub.get("creationDate"),
            "last_modification_date": pub.get("lastModificationDate"),
            "immo_office_name": customers[0].get("name") if customers else None,
            "is_biddit_sale": flags.get("isAnInteractiveSale"),
            "is_under_option": flags.get("isUnderOption"),
            "is_newly_built": flags.get("isNewlyBuilt"),
            "description": (
                prop.get("description")
                or (prop.get("alternativeDescriptions") or {}).get("en")
                or (prop.get("alternativeDescriptions") or {}).get("nl")
                or (prop.get("alternativeDescriptions") or {}).get("fr")
            ),
        }

    def _prepare_features(self, parsed):
        """Transform parsed property dict into an AutoGluon-ready DataFrame."""
        row = {}
        asking_price_eur = parsed.get("price")

        direct_fields = [
            "property_subtype", "latitude", "longitude",
            "floor_number", "year_built", "total_floors", "facade_count",
            "street_facade_width", "livable_area", "property_size",
            "kitchen_surface", "living_room_surface", "number_of_bedrooms",
            "number_of_bathrooms", "number_of_toilets", "shower_room_count",
            "terrace_surface", "garden_surface", "monthly_costs",
            "parking_indoor", "parking_outdoor", "primary_energy_consumption",
            "p_score", "g_score", "cadastral_income",
        ]
        for f in direct_fields:
            row[f] = parsed.get(f)

        cat_fields = {
            "building_condition": parsed.get("building_condition"),
            "kitchen_type": parsed.get("kitchen_type"),
            "energy_label": parsed.get("energy_label"),
            "heating_type": parsed.get("heating_type"),
            "flood_zone_type": parsed.get("flood_zone_type"),
            "renovation_obligation": parsed.get("renovation_obligation"),
            "location_type": parsed.get("location_type"),
        }
        for k, v in cat_fields.items():
            row[k] = v if v is not None else "Unknown"

        bool_fields = [
            "has_living_room", "has_attic", "has_basement", "has_terrace",
            "has_garden", "has_swimming_pool", "has_lift", "has_dressing_room",
            "has_air_conditioning", "has_caretaker", "has_photovoltaic_panels",
            "has_double_glazing", "has_heat_pump", "has_thermic_panels",
            "is_furnished", "is_under_option", "is_newly_built",
        ]
        for f in bool_fields:
            val = parsed.get(f)
            row[f] = 1 if (val is True or val == "True" or val == 1) else 0

        for f in ["garden_surface", "terrace_surface"]:
            if row.get(f) is None:
                row[f] = 0

        current_year = datetime.now().year
        yb = row.get("year_built")
        row["age"] = (current_year - yb) if yb else None

        row["property_type_HOUSE"] = 1 if parsed.get("property_type") == "HOUSE" else 0

        lat, lon = row.get("latitude"), row.get("longitude")
        if lat and lon:
            center_lat, center_lon = 51.053098573153505, 3.7265139580328865
            row["dist_from_center"] = np.sqrt((lat - center_lat)**2 + (lon - center_lon)**2) * 100
        else:
            row["dist_from_center"] = None

        if lat and lon and self.data_loader:
            nbh = self.data_loader.get_neighbourhood_spatial(lat, lon)
            row["neighbourhood"] = nbh.replace(" ", "").replace("-", "_") if nbh else "Unknown"
        else:
            row["neighbourhood"] = "Unknown"

        row["street_type"] = self._get_street_type(parsed.get("street", ""))
        row["distance_to_street"] = 0

        pub_date = parsed.get("publication_date")
        mod_date = parsed.get("last_modification_date")
        if pub_date and mod_date:
            try:
                row["listing_duration_days"] = max(0, (pd.to_datetime(mod_date) - pd.to_datetime(pub_date)).days)
            except Exception:
                row["listing_duration_days"] = 0
        else:
            row["listing_duration_days"] = 0

        pc = parsed.get("postal_code")
        row["postal_code"] = str(int(pc)) if pc else None

        # Description embedding
        description = parsed.get("description", "")
        emb = self._embed_description(description)
        for i in range(50):
            row[f"desc_emb_{i}"] = emb[i]
        row["has_meaningful_description"] = 1 if description and len(str(description)) > 50 else 0

        # Macro indicators (best-effort)
        for col in ["belgian_cpi_index", "belgian_10y_yield", "euribor_12m"]:
            row[col] = None

        df = pd.DataFrame([row])
        return df, asking_price_eur

    def _embed_description(self, text):
        """Embed description using E5 + PCA. Returns 50-dim array."""
        import torch

        if self.e5_model is None or self.pca_model is None:
            return np.zeros(50)

        if not text or not isinstance(text, str):
            text = ""
        text = text.strip()

        prefixed = f"passage: {text}" if text else "passage: "
        encoded = self.e5_tokenizer(
            [prefixed], max_length=512, padding=True, truncation=True,
            return_tensors="pt",
        ).to(self.e5_device)

        with torch.no_grad():
            outputs = self.e5_model(**encoded)

        attention_mask = encoded["attention_mask"]
        token_embs = outputs.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embs.size()).float()
        embedding = ((token_embs * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1e-9))
        embedding = embedding.cpu().numpy()

        return self.pca_model.transform(embedding)[0]

    @staticmethod
    def _get_street_type(address):
        if not address:
            return "residential"
        addr = address.lower()
        if "laan" in addr:
            return "tertiary"
        elif "steenweg" in addr or "steeniweg" in addr:
            return "secondary"
        elif "weg" in addr:
            return "unclassified"
        elif "straat" in addr:
            return "residential"
        return "residential"

    @staticmethod
    def _haversine_dist(lat, lon):
        center_lat, center_lon = 51.0543, 3.7174
        R = 6371
        dlat = math.radians(lat - center_lat)
        dlon = math.radians(lon - center_lon)
        a = (math.sin(dlat / 2)**2 +
             math.cos(math.radians(center_lat)) * math.cos(math.radians(lat)) *
             math.sin(dlon / 2)**2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c
