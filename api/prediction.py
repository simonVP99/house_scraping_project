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
                 e5_device=None, pca_model=None, data_loader=None,
                 imputation_models=None, street_graph_path=None,
                 shap_artifact=None):
        self.ag_predictor = ag_predictor
        self.old_model = old_model
        self.old_options = old_options or {}
        self.old_features = old_features or []
        self.e5_model = e5_model
        self.e5_tokenizer = e5_tokenizer
        self.e5_device = e5_device
        self.pca_model = pca_model
        self.data_loader = data_loader
        self.imputation_models = imputation_models or {}
        self.street_graph_path = street_graph_path
        self._street_graph = None
        self._street_edges = None
        self.shap_artifact = shap_artifact
        self._shap_explainer = None
        self._macro_cache = None
        self._macro_fetched_date = None

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

            # Add explainability
            self._add_explainability(result, parsed, features_df)

            return result, 200

        except requests.exceptions.HTTPError as e:
            if hasattr(e, "response") and e.response is not None and e.response.status_code == 403:
                return {"error": "Access blocked by Immoweb (403). Try again later."}, 403
            return {"error": f"HTTP error: {e}"}, 502
        except ValueError as e:
            return {"error": str(e)}, 422
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}, 500

    def predict_from_form_data(self, form_data):
        """Re-predict from edited form_data using AutoGluon (same path as URL prediction)."""
        if self.ag_predictor is None:
            return {"error": "AutoGluon model not loaded"}, 503

        try:
            features_df, asking_price = self._prepare_features(form_data)
            predicted_k = float(self.ag_predictor.predict(features_df).values[0])
            predicted_eur = predicted_k * 1000

            result = {
                "property": {
                    "property_id": form_data.get("property_id"),
                    "address": f"{form_data.get('street', '')} {form_data.get('house_number', '')}, {form_data.get('postal_code', '')} {form_data.get('city', '')}".strip(),
                    "price": asking_price,
                    "livable_area": form_data.get("livable_area"),
                    "bedrooms": form_data.get("number_of_bedrooms"),
                    "year_built": form_data.get("year_built"),
                    "condition": form_data.get("building_condition"),
                    "energy_label": form_data.get("energy_label"),
                    "property_type": form_data.get("property_type"),
                    "property_subtype": form_data.get("property_subtype"),
                    "neighbourhood": None,
                },
                "prediction": {
                    "predicted_price": round(predicted_eur),
                    "delta_eur": round(predicted_eur - asking_price) if asking_price else None,
                    "delta_pct": round(float((predicted_eur - asking_price) / asking_price * 100), 1) if asking_price else None,
                },
                "form_data": form_data,
            }

            lat, lon = form_data.get("latitude"), form_data.get("longitude")
            if lat and lon and self.data_loader:
                nbh = self.data_loader.get_neighbourhood_spatial(float(lat), float(lon))
                if nbh:
                    result["property"]["neighbourhood"] = nbh

            self._add_explainability(result, form_data, features_df)

            return result, 200
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

        # Street type: spatial lookup via OSMnx (falls back to heuristic)
        lat, lon = row.get("latitude"), row.get("longitude")
        if lat and lon:
            street_type, distance = self._lookup_street_type(lat, lon)
            row["street_type"] = street_type
            row["distance_to_street"] = distance
        else:
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

        # Spatial lag from training data
        lat, lon = row.get("latitude"), row.get("longitude")
        if lat and lon and self.data_loader:
            spatial_lag, avg_dist = self.data_loader.compute_spatial_lag(lat, lon)
            row["spatial_lag_10"] = spatial_lag
            row["avg_dist_5nn"] = avg_dist
        else:
            row["spatial_lag_10"] = None
            row["avg_dist_5nn"] = None

        # Macro indicators from ECB (cached at startup)
        macro = self._get_macro_indicators()
        for col in ["belgian_cpi_index", "belgian_10y_yield", "euribor_12m"]:
            row[col] = macro.get(col)

        # Apply pipeline-equivalent imputation for missing values
        if self.imputation_models:
            row = self._impute_missing(row, asking_price_eur, row.get("livable_area"))

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

    def _get_macro_indicators(self):
        """Fetch latest macro indicators from ECB, cached daily."""
        from datetime import date
        today = date.today().isoformat()
        if self._macro_cache is not None and self._macro_fetched_date == today:
            return self._macro_cache

        indicators = {
            'belgian_cpi_index': 'https://data-api.ecb.europa.eu/service/data/ICP/M.BE.N.000000.4.INX?startPeriod=2024-01&format=csvdata',
            'belgian_10y_yield': 'https://data-api.ecb.europa.eu/service/data/IRS/M.BE.L.L40.CI.0000.EUR.N.Z?startPeriod=2024-01&format=csvdata',
            'euribor_12m': 'https://data-api.ecb.europa.eu/service/data/FM/M.U2.EUR.RT.MM.EURIBOR1YD_.HSTA?startPeriod=2024-01&format=csvdata',
        }
        result = {}
        for col, url in indicators.items():
            try:
                import requests, csv, io
                # requests ships certifi's CA bundle, so this verifies correctly on
                # macOS Pythons that do not have root certs installed system-wide
                # (urllib would raise CERTIFICATE_VERIFY_FAILED here).
                resp = requests.get(url, headers={'Accept': 'text/csv'}, timeout=10)
                resp.raise_for_status()
                reader = csv.DictReader(io.StringIO(resp.text))
                values = {row['TIME_PERIOD']: float(row['OBS_VALUE']) for row in reader}
                if values:
                    latest_key = max(values.keys())
                    result[col] = values[latest_key]
                    print(f"  Macro {col}: {result[col]} ({latest_key})")
            except Exception as e:
                print(f"  Macro {col} fetch failed: {e}")
                result[col] = None

        self._macro_cache = result
        self._macro_fetched_date = today
        return result

    def _compute_shap(self, features_df):
        """Compute SHAP values using the standalone XGBoost model."""
        if not self.shap_artifact:
            return None
        try:
            import shap
            model = self.shap_artifact['model']
            feature_names = self.shap_artifact['feature_names']
            encoders = self.shap_artifact['label_encoders']
            base_value = self.shap_artifact['base_value']

            # Build a row matching the XGBoost's expected features
            row_dict = {}
            for col in feature_names:
                if col in features_df.columns:
                    row_dict[col] = features_df[col].iloc[0]
                else:
                    row_dict[col] = np.nan

            xgb_df = pd.DataFrame([row_dict])

            # Coerce numeric columns that may be object type due to None values
            for col in xgb_df.columns:
                if col not in encoders:
                    xgb_df[col] = pd.to_numeric(xgb_df[col], errors='coerce')

            # Encode categoricals the same way as training
            for col, le in encoders.items():
                if col in xgb_df.columns:
                    val = xgb_df[col].iloc[0]
                    if pd.isna(val) or val is None:
                        val = '_missing_'
                    val = str(val)
                    if val in le.classes_:
                        xgb_df[col] = le.transform([val])
                    else:
                        xgb_df[col] = le.transform(['_missing_']) if '_missing_' in le.classes_ else 0

            # Bool to int
            for col in xgb_df.columns:
                if xgb_df[col].dtype == bool:
                    xgb_df[col] = xgb_df[col].astype(int)

            # Lazy-init explainer
            if self._shap_explainer is None:
                self._shap_explainer = shap.TreeExplainer(model)

            shap_values = self._shap_explainer.shap_values(xgb_df)
            vals = shap_values[0]

            # Human-readable feature name mapping
            name_map = {
                'livable_area': 'Livable area',
                'number_of_bedrooms': 'Bedrooms',
                'number_of_bathrooms': 'Bathrooms',
                'number_of_toilets': 'Toilets',
                'age': 'Age',
                'year_built': 'Year built',
                'building_condition': 'Condition',
                'energy_label': 'Energy label',
                'primary_energy_consumption': 'Energy consumption',
                'dist_from_center': 'Distance from center',
                'neighbourhood': 'Neighbourhood',
                'property_type_HOUSE': 'Property type',
                'property_subtype': 'Subtype',
                'postal_code': 'Postal code',
                'has_garden': 'Garden',
                'garden_surface': 'Garden area',
                'has_terrace': 'Terrace',
                'terrace_surface': 'Terrace area',
                'has_swimming_pool': 'Swimming pool',
                'parking_indoor': 'Indoor parking',
                'parking_outdoor': 'Outdoor parking',
                'facade_count': 'Facades',
                'total_floors': 'Total floors',
                'has_lift': 'Lift',
                'has_double_glazing': 'Double glazing',
                'has_photovoltaic_panels': 'Solar panels',
                'has_heat_pump': 'Heat pump',
                'cadastral_income': 'Cadastral income',
                'kitchen_type': 'Kitchen type',
                'kitchen_surface': 'Kitchen area',
                'living_room_surface': 'Living room area',
                'heating_type': 'Heating type',
                'street_type': 'Street type',
                'property_size': 'Plot size',
                'street_facade_width': 'Street facade width',
                'floor_number': 'Floor number',
                'monthly_costs': 'Monthly costs',
                'is_newly_built': 'Newly built',
                'has_basement': 'Basement',
                'has_attic': 'Attic',
                'flood_zone_type': 'Flood zone',
                'renovation_obligation': 'Renovation obligation',
                'listing_duration_days': 'Listing duration',
            }

            # Sort by absolute value, take top 15 (skip desc_emb_*)
            pairs = []
            for feat, val in zip(feature_names, vals):
                if feat.startswith('desc_emb_') or feat.startswith('belgian_') or feat.startswith('euribor'):
                    continue
                pairs.append((feat, float(val)))
            pairs.sort(key=lambda x: abs(x[1]), reverse=True)

            shap_items = []
            for feat, val in pairs[:15]:
                label = name_map.get(feat, feat.replace('_', ' ').title())
                shap_items.append({
                    'feature': label,
                    'feature_key': feat,
                    'value_k': round(val, 2),
                    'value_eur': round(val * 1000),
                })

            # Also sum the skipped embedding/macro features
            other_sum = sum(v for f, v in zip(feature_names, vals)
                           if f.startswith('desc_emb_') or f.startswith('belgian_') or f.startswith('euribor'))
            if abs(other_sum) > 0.5:
                shap_items.append({
                    'feature': 'Description & macro',
                    'feature_key': '_other',
                    'value_k': round(float(other_sum), 2),
                    'value_eur': round(float(other_sum) * 1000),
                })

            return {
                'base_value_k': round(float(base_value), 1),
                'base_value_eur': round(float(base_value) * 1000),
                'features': shap_items,
            }
        except Exception as e:
            print(f"SHAP computation failed: {e}")
            return None

    def _add_explainability(self, result, parsed, features_df=None):
        """Add comparables, feature context, and SHAP values to a prediction result."""
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
        if features_df is not None:
            result["shap"] = self._compute_shap(features_df)

    def _impute_missing(self, row, asking_price, livable_area):
        """Apply pipeline-equivalent imputation using saved Poisson GLMs."""
        poisson_models = self.imputation_models.get("poisson_models", {})
        median_year_built = self.imputation_models.get("median_year_built")

        # 1. year_built / age: use median if missing
        if row.get("year_built") is None and median_year_built is not None:
            row["year_built"] = median_year_built
            row["age"] = datetime.now().year - median_year_built

        # Compute price_k and price_sq_k (pipeline uses K-EUR scale)
        price_k = asking_price / 1000 if asking_price else None
        price_sq_k = (price_k / livable_area) if (price_k and livable_area) else None

        # Helper: build predictor dict and call a Poisson model
        # Core predictors (price, livable_area, property_type_HOUSE) must be present;
        # extra predictors (floor_number, facade_count, is_newly_built) default to 0
        core_predictors = {"price", "price_sq", "livable_area", "property_type_HOUSE"}

        def _poisson_predict(target_name):
            entry = poisson_models.get(target_name)
            if entry is None:
                return None
            model = entry["model"]
            predictors = entry["predictors"]
            pred_dict = {}
            for col in predictors:
                if col == "price":
                    pred_dict[col] = price_k
                elif col == "price_sq":
                    pred_dict[col] = price_sq_k
                elif col == "property_type_HOUSE":
                    pred_dict[col] = row.get("property_type_HOUSE", 0)
                else:
                    val = row.get(col)
                    # Extra predictors default to 0 if missing
                    pred_dict[col] = val if val is not None else 0
            # Core predictors must be present
            if any(pred_dict.get(c) is None for c in core_predictors if c in predictors):
                return None
            pred_df = pd.DataFrame([pred_dict])
            result = model.predict(pred_df)
            return max(0, int(round(float(result.values[0]))))

        # 2. Count fields: impute if 0 or None (pipeline treats 0 as missing)
        # Sequential: bedrooms -> toilets -> bathrooms (chained)
        for field in ["number_of_bedrooms", "number_of_toilets", "number_of_bathrooms"]:
            val = row.get(field)
            if val is None or val == 0:
                imputed = _poisson_predict(field)
                if imputed is not None:
                    row[field] = imputed

        # 3. Cadastral income: impute if 0 or None
        ci = row.get("cadastral_income")
        if ci is None or ci == 0:
            imputed = _poisson_predict("cadastral_income")
            if imputed is not None:
                row["cadastral_income"] = imputed

        return row

    def _load_street_graph(self):
        """Lazy-load the OSMnx street graph for spatial street_type lookup."""
        if self._street_graph is not None:
            return
        if not self.street_graph_path:
            return
        import os
        if not os.path.exists(self.street_graph_path):
            print(f"Street graph not found at {self.street_graph_path}")
            return
        try:
            import osmnx as ox
            self._street_graph = ox.load_graphml(self.street_graph_path)
            _, edges = ox.graph_to_gdfs(self._street_graph)
            edges['highway'] = edges['highway'].apply(
                lambda h: h[0] if isinstance(h, list) else h)
            self._street_edges = edges
            print(f"Street graph loaded ({len(edges)} edges)")
        except Exception as e:
            print(f"Could not load street graph: {e}")

    def _lookup_street_type(self, lat, lon):
        """Spatial street_type + distance lookup using OSMnx. Falls back to heuristic."""
        self._load_street_graph()
        if self._street_graph is None:
            return self._get_street_type(""), 0

        try:
            import osmnx as ox
            import numpy as np
            edge_indices, distances = ox.nearest_edges(
                self._street_graph, X=np.array([lon]), Y=np.array([lat]),
                return_dist=True)
            street_type = self._street_edges.loc[edge_indices[0], 'highway']

            # Correct cycleways/footways like the pipeline does
            undesirable = ['cycleway', 'footway', 'path', 'pedestrian',
                           'track', 'steps', 'service', 'trunk']
            if street_type in undesirable:
                main_edges = self._street_edges[
                    ~self._street_edges['highway'].isin(undesirable)]
                if not main_edges.empty:
                    nodes, _ = ox.graph_to_gdfs(self._street_graph)
                    G_main = ox.graph_from_gdfs(nodes, main_edges)
                    main_indices, main_dists = ox.nearest_edges(
                        G_main, X=np.array([lon]), Y=np.array([lat]),
                        return_dist=True)
                    if main_dists[0] <= 20.0:
                        street_type = main_edges.loc[main_indices[0], 'highway']
                        distances = main_dists

            return str(street_type), float(distances[0])
        except Exception as e:
            print(f"Street lookup failed: {e}")
            return self._get_street_type(""), 0

    def batch_lookup_street_types(self, lats, lons):
        """Batch spatial street_type + distance lookup for multiple properties."""
        self._load_street_graph()
        n = len(lats)
        if self._street_graph is None or n == 0:
            return ["residential"] * n, [0.0] * n

        try:
            import osmnx as ox
            import numpy as np
            lats_arr = np.array(lats)
            lons_arr = np.array(lons)

            edge_indices, distances = ox.nearest_edges(
                self._street_graph, X=lons_arr, Y=lats_arr,
                return_dist=True)
            street_types = self._street_edges.loc[edge_indices, 'highway'].values.tolist()

            # Correct undesirable types
            undesirable = {'cycleway', 'footway', 'path', 'pedestrian',
                           'track', 'steps', 'service', 'trunk'}
            needs_correction = [i for i, st in enumerate(street_types) if st in undesirable]

            if needs_correction:
                main_edges = self._street_edges[
                    ~self._street_edges['highway'].isin(undesirable)]
                if not main_edges.empty:
                    nodes, _ = ox.graph_to_gdfs(self._street_graph)
                    G_main = ox.graph_from_gdfs(nodes, main_edges)
                    corr_lons = lons_arr[needs_correction]
                    corr_lats = lats_arr[needs_correction]
                    main_indices, main_dists = ox.nearest_edges(
                        G_main, X=corr_lons, Y=corr_lats, return_dist=True)
                    corrected_types = main_edges.loc[main_indices, 'highway'].values
                    for j, orig_idx in enumerate(needs_correction):
                        if main_dists[j] <= 20.0:
                            street_types[orig_idx] = corrected_types[j]
                            distances[orig_idx] = main_dists[j]

            return [str(s) for s in street_types], [float(d) for d in distances]
        except Exception as e:
            print(f"Batch street lookup failed: {e}")
            return ["residential"] * n, [0.0] * n

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
