"""Live listings service - parse today's HTML files, predict, cache to CSV."""
import os
import glob
import math
import pandas as pd
import numpy as np
from datetime import date, timedelta, datetime
from api.config import PROJECT_DIR

CACHE_CSV = os.path.join(PROJECT_DIR, "outputs", "today_predictions.csv")
DATA_DIR = os.path.join(PROJECT_DIR, "data")


class LiveListingsService:
    def __init__(self, prediction_service, data_loader):
        self.prediction_service = prediction_service
        self.data_loader = data_loader
        self.listings_df = None
        self.date_used = None
        self.load_or_generate()

    def load_or_generate(self):
        """Load from CSV cache if today's data exists, otherwise parse + predict."""
        if os.path.exists(CACHE_CSV):
            df = pd.read_csv(CACHE_CSV)
            if "date_obtained" in df.columns and len(df) > 0:
                cached_date = str(df["date_obtained"].iloc[0])
                if cached_date == str(date.today()):
                    self.listings_df = df
                    self.date_used = cached_date
                    print(f"Live listings: loaded {len(df)} from cache ({cached_date})")
                    return

        self._generate()

    def refresh(self):
        """Force re-parse + re-predict, ignoring cache."""
        self._generate()
        return {"count": len(self.listings_df), "date_used": self.date_used}

    def _generate(self):
        """Scan HTML files, parse, batch-embed, batch-predict, save to cache."""
        html_files, date_tag = self._find_html_files()
        if not html_files:
            print("Live listings: no HTML files found in last 7 days")
            self.listings_df = pd.DataFrame()
            self.date_used = None
            return

        iso_date = f"{date_tag[4:8]}-{date_tag[2:4]}-{date_tag[0:2]}"
        self.date_used = iso_date
        total = len(html_files)
        print(f"Live listings: processing {total} files from {iso_date}...")

        # Phase 1: Parse all HTML files
        parsed_items = []
        parse_errors = 0
        for i, (html_path, property_id) in enumerate(html_files):
            if (i + 1) % 100 == 0:
                print(f"  Parsing {i + 1}/{total}...")
            try:
                with open(html_path, "r", encoding="utf-8", errors="replace") as f:
                    html = f.read()
                fake_url = f"https://www.immoweb.be/nl/zoekertje/property/te-koop/gent/9000/{property_id}"
                parsed = self.prediction_service._parse_immoweb_html(html, fake_url)
                if parsed.get("price"):
                    parsed_items.append((property_id, parsed))
            except Exception:
                parse_errors += 1

        if not parsed_items:
            print(f"Live listings: no valid properties parsed ({parse_errors} errors)")
            self.listings_df = pd.DataFrame()
            return

        print(f"  Parsed {len(parsed_items)} properties ({parse_errors} errors)")

        # Phase 2: Batch embed all descriptions
        descriptions = [p.get("description", "") or "" for _, p in parsed_items]
        print(f"  Batch-embedding {len(descriptions)} descriptions...")
        embeddings = self._batch_embed(descriptions)
        print(f"  Embeddings done")

        # Phase 3: Build feature rows for ALL properties
        print(f"  Building feature matrix...")
        feature_rows = []
        meta_rows = []
        current_year = datetime.now().year

        for idx, (property_id, parsed) in enumerate(parsed_items):
            try:
                feat, meta = self._build_feature_row(property_id, parsed, embeddings[idx], current_year)
                feature_rows.append(feat)
                meta_rows.append(meta)
            except Exception:
                pass

        if not feature_rows:
            self.listings_df = pd.DataFrame()
            return

        # Phase 3b: Batch OSMnx street_type lookup
        lats = [r.get("latitude") for r in feature_rows]
        lons = [r.get("longitude") for r in feature_rows]
        valid_mask = [lat is not None and lon is not None for lat, lon in zip(lats, lons)]
        valid_lats = [lat for lat, v in zip(lats, valid_mask) if v]
        valid_lons = [lon for lon, v in zip(lons, valid_mask) if v]

        if valid_lats:
            print(f"  Batch street_type lookup for {len(valid_lats)} properties...")
            street_types, distances = self.prediction_service.batch_lookup_street_types(
                valid_lats, valid_lons)
            j = 0
            for i, is_valid in enumerate(valid_mask):
                if is_valid:
                    feature_rows[i]["street_type"] = street_types[j]
                    feature_rows[i]["distance_to_street"] = distances[j]
                    j += 1

        # Phase 3c: Apply imputation for missing values
        ps = self.prediction_service
        if ps.imputation_models:
            for i, feat in enumerate(feature_rows):
                asking = meta_rows[i].get("asking_price_eur")
                livable = feat.get("livable_area")
                feature_rows[i] = ps._impute_missing(feat, asking, livable)

        # Phase 4: Single batch predict call
        features_df = pd.DataFrame(feature_rows)
        print(f"  Predicting {len(features_df)} properties in one batch...")
        predicted_k = self.prediction_service.ag_predictor.predict(features_df)

        # Phase 5: Assemble output
        output_rows = []
        for i, meta in enumerate(meta_rows):
            pk = float(predicted_k.values[i])
            predicted_eur = pk * 1000
            asking_eur = meta["asking_price_eur"]
            delta_eur = predicted_eur - asking_eur
            delta_pct = (delta_eur / asking_eur * 100) if asking_eur else 0

            output_rows.append({
                "property_id": meta["property_id"],
                "address": meta["address"],
                "url": meta["url"],
                "asking_price": round(asking_eur),
                "predicted_price": round(predicted_eur),
                "delta_eur": round(delta_eur),
                "delta_pct": round(float(delta_pct), 1),
                "livable_area": meta["livable_area"],
                "bedrooms": meta["bedrooms"],
                "condition": meta["condition"],
                "energy_label": meta["energy_label"],
                "neighbourhood": meta["neighbourhood"],
                "date_obtained": iso_date,
            })

        df = pd.DataFrame(output_rows)
        df.to_csv(CACHE_CSV, index=False)
        self.listings_df = df
        print(f"Live listings: {len(df)} predicted, cached to {CACHE_CSV}")

    def _batch_embed(self, texts, batch_size=32):
        """Batch-embed descriptions using E5 + PCA. Returns list of 50-dim arrays."""
        import torch

        ps = self.prediction_service
        if ps.e5_model is None or ps.pca_model is None:
            return [np.zeros(50) for _ in texts]

        all_embeddings = []
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start:start + batch_size]
            prefixed = [
                f"passage: {t.strip()}" if t and isinstance(t, str) and t.strip() else "passage: "
                for t in batch_texts
            ]

            encoded = ps.e5_tokenizer(
                prefixed, max_length=512, padding=True, truncation=True,
                return_tensors="pt",
            ).to(ps.e5_device)

            with torch.no_grad():
                outputs = ps.e5_model(**encoded)

            attention_mask = encoded["attention_mask"]
            token_embs = outputs.last_hidden_state
            mask_expanded = attention_mask.unsqueeze(-1).expand(token_embs.size()).float()
            batch_emb = (token_embs * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1e-9)
            batch_emb = batch_emb.cpu().numpy()

            pca_result = ps.pca_model.transform(batch_emb)
            for row in pca_result:
                all_embeddings.append(row)

        return all_embeddings

    def _build_feature_row(self, property_id, parsed, embedding, current_year):
        """Build one feature dict (for AG predict) + one metadata dict (for output)."""
        ps = self.prediction_service

        row = {}
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

        yb = row.get("year_built")
        row["age"] = (current_year - yb) if yb else None

        row["property_type_HOUSE"] = 1 if parsed.get("property_type") == "HOUSE" else 0

        lat, lon = row.get("latitude"), row.get("longitude")
        if lat and lon:
            center_lat, center_lon = 51.053098573153505, 3.7265139580328865
            row["dist_from_center"] = np.sqrt((lat - center_lat)**2 + (lon - center_lon)**2) * 100
        else:
            row["dist_from_center"] = None

        # Neighbourhood for model feature (normalized)
        neighbourhood_display = "Unknown"
        if lat and lon and self.data_loader:
            nbh = self.data_loader.get_neighbourhood_spatial(lat, lon)
            if nbh:
                neighbourhood_display = nbh
            row["neighbourhood"] = nbh.replace(" ", "").replace("-", "_") if nbh else "Unknown"
        else:
            row["neighbourhood"] = "Unknown"

        # street_type and distance_to_street set in batch by _generate()
        row["street_type"] = ps._get_street_type(parsed.get("street", ""))
        row["distance_to_street"] = 0  # overwritten by batch OSMnx lookup

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

        for i in range(50):
            row[f"desc_emb_{i}"] = embedding[i]
        description = parsed.get("description", "")
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

        macro = ps._get_macro_indicators()
        for col in ["belgian_cpi_index", "belgian_10y_yield", "euribor_12m"]:
            row[col] = macro.get(col)

        # Metadata for output (not fed to model)
        asking_price = parsed.get("price")
        street = parsed.get("street", "") or ""
        number = parsed.get("house_number", "") or ""
        city = parsed.get("city", "") or ""
        postal = parsed.get("postal_code", "") or ""
        address = f"{street} {number}, {postal} {city}".strip().strip(",").strip()

        meta = {
            "property_id": property_id,
            "address": address,
            "url": f"https://www.immoweb.be/nl/zoekertje/property/te-koop/{city.lower() or 'gent'}/{postal or '9000'}/{property_id}",
            "asking_price_eur": asking_price,  # mainValue from Immoweb is already EUR
            "livable_area": parsed.get("livable_area"),
            "bedrooms": parsed.get("number_of_bedrooms"),
            "condition": parsed.get("building_condition"),
            "energy_label": parsed.get("energy_label"),
            "neighbourhood": neighbourhood_display,
        }

        return row, meta

    def _find_html_files(self):
        """Find most recent date with HTML files, up to 7 days back."""
        for i in range(7):
            d = date.today() - timedelta(days=i)
            tag = d.strftime("%d%m%Y")
            pattern = os.path.join(DATA_DIR, "*", f"full_source_{tag}.html")
            files = glob.glob(pattern)
            if files:
                result = []
                for f in files:
                    prop_id = os.path.basename(os.path.dirname(f))
                    result.append((f, prop_id))
                return result, tag
        return [], None

    def get_listings(self, neighbourhood=None):
        """Return listings as list of dicts, optionally filtered by neighbourhood."""
        if self.listings_df is None or len(self.listings_df) == 0:
            return []
        df = self.listings_df
        if neighbourhood:
            # Match both normalized ("BrugsePoort_Rooigem") and display ("Brugse Poort - Rooigem") names
            def _normalize(s):
                return s.replace(" ", "").replace("-", "_").lower() if isinstance(s, str) else ""
            norm_query = _normalize(neighbourhood)
            df = df[df["neighbourhood"].apply(_normalize) == norm_query]
        df = df.sort_values("delta_pct", ascending=False, na_position="last")
        records = df.to_dict(orient="records")
        for row in records:
            for k, v in row.items():
                if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    row[k] = None
        return records

    def get_count(self):
        """Return total count and date used."""
        count = len(self.listings_df) if self.listings_df is not None else 0
        return {"count": count, "date_used": self.date_used}
