"""Live listings service - parse today's HTML files, predict, cache to CSV."""
import os
import glob
import pandas as pd
import numpy as np
from datetime import date, timedelta
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
        """Scan HTML files, parse, predict, save to cache."""
        html_files, date_tag = self._find_html_files()
        if not html_files:
            print("Live listings: no HTML files found in last 7 days")
            self.listings_df = pd.DataFrame()
            self.date_used = None
            return

        iso_date = f"{date_tag[4:8]}-{date_tag[2:4]}-{date_tag[0:2]}"
        self.date_used = iso_date
        print(f"Live listings: processing {len(html_files)} files from {iso_date}...")

        rows = []
        errors = 0
        for html_path, property_id in html_files:
            try:
                row = self._process_one(html_path, property_id)
                if row:
                    row["date_obtained"] = iso_date
                    rows.append(row)
            except Exception:
                errors += 1

        if not rows:
            self.listings_df = pd.DataFrame()
            return

        df = pd.DataFrame(rows)
        df.to_csv(CACHE_CSV, index=False)
        self.listings_df = df
        print(f"Live listings: {len(df)} predicted, {errors} errors, cached to {CACHE_CSV}")

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

    def _process_one(self, html_path, property_id):
        """Parse one HTML file, predict, return row dict."""
        with open(html_path, "r", encoding="utf-8", errors="replace") as f:
            html = f.read()

        fake_url = f"https://www.immoweb.be/nl/zoekertje/property/te-koop/gent/9000/{property_id}"
        parsed = self.prediction_service._parse_immoweb_html(html, fake_url)

        asking_price = parsed.get("price")
        if not asking_price:
            return None

        features_df, _ = self.prediction_service._prepare_features(parsed)
        predicted_k = float(self.prediction_service.ag_predictor.predict(features_df).values[0])
        predicted_eur = predicted_k * 1000
        asking_eur = asking_price * 1000

        lat, lon = parsed.get("latitude"), parsed.get("longitude")
        neighbourhood = "Unknown"
        if lat and lon and self.data_loader:
            nbh = self.data_loader.get_neighbourhood_spatial(lat, lon)
            if nbh:
                neighbourhood = nbh

        street = parsed.get("street", "") or ""
        number = parsed.get("house_number", "") or ""
        city = parsed.get("city", "") or ""
        postal = parsed.get("postal_code", "") or ""
        address = f"{street} {number}, {postal} {city}".strip().strip(",").strip()

        return {
            "property_id": property_id,
            "address": address,
            "url": f"https://www.immoweb.be/nl/zoekertje/property/te-koop/{city.lower() or 'gent'}/{postal or '9000'}/{property_id}",
            "asking_price": round(asking_eur),
            "predicted_price": round(predicted_eur),
            "delta_eur": round(predicted_eur - asking_eur),
            "delta_pct": round(float((predicted_eur - asking_eur) / asking_eur * 100), 1),
            "livable_area": parsed.get("livable_area"),
            "bedrooms": parsed.get("number_of_bedrooms"),
            "condition": parsed.get("building_condition"),
            "energy_label": parsed.get("energy_label"),
            "neighbourhood": neighbourhood,
        }

    def get_listings(self, neighbourhood=None):
        """Return listings as list of dicts, optionally filtered by neighbourhood."""
        if self.listings_df is None or len(self.listings_df) == 0:
            return []
        df = self.listings_df
        if neighbourhood:
            df = df[df["neighbourhood"] == neighbourhood]
        df = df.sort_values("delta_pct", ascending=False, na_position="last")
        return df.to_dict(orient="records")

    def get_count(self):
        """Return total count and date used."""
        count = len(self.listings_df) if self.listings_df is not None else 0
        return {"count": count, "date_used": self.date_used}
