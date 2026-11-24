"""Live listings service — parse today's HTML files via `src/parser.py`, batch-score via Phase-7 bundle, cache to CSV."""
from __future__ import annotations

import glob
import math
import os
from datetime import date, timedelta

import pandas as pd

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
        """Scan HTML files, parse with canonical `src/parser.py`, batch-score with Phase-7 bundle, cache CSV."""
        html_files, date_tag = self._find_html_files()
        if not html_files:
            print("Live listings: no HTML files found in last 7 days")
            self.listings_df = pd.DataFrame()
            self.date_used = None
            return

        if self.prediction_service is None or getattr(self.prediction_service, "bundle", None) is None:
            print("Live listings: production bundle not loaded; skipping scoring")
            self.listings_df = pd.DataFrame()
            self.date_used = None
            return

        iso_date = f"{date_tag[4:8]}-{date_tag[2:4]}-{date_tag[0:2]}"
        self.date_used = iso_date
        total = len(html_files)
        print(f"Live listings: processing {total} files from {iso_date}...")

        parsed_rows = []
        meta_rows = []
        parse_errors = 0

        for i, (html_path, property_id) in enumerate(html_files):
            if (i + 1) % 100 == 0:
                print(f"  Parsing {i + 1}/{total}...")
            try:
                with open(html_path, "r", encoding="utf-8", errors="replace") as f:
                    html = f.read()
                fake_url = f"https://www.immoweb.be/nl/zoekertje/property/te-koop/gent/9000/{property_id}"
                parsed = self.prediction_service.parse_immoweb_html(html, fake_url)
                asking = parsed.get("price")
                area = parsed.get("livable_area")
                if not asking or not area or float(asking) <= 0 or float(area) <= 0:
                    continue
                parsed_rows.append(parsed)

                street = str(parsed.get("street") or "").strip()
                house_num = str(parsed.get("house_number") or "").strip()
                postal = str(parsed.get("postal_code") or "").strip()
                city = str(parsed.get("city") or "").strip()
                addr_left = f"{street} {house_num}".strip()
                addr_right = f"{postal} {city}".strip()
                address = ", ".join(p for p in (addr_left, addr_right) if p)

                meta_rows.append({
                    "property_id": str(property_id),
                    "address": address,
                    "url": fake_url,
                    "asking_price_eur": float(asking),
                    "livable_area": float(area),
                    "bedrooms": parsed.get("number_of_bedrooms"),
                    "condition": parsed.get("building_condition"),
                    "energy_label": parsed.get("energy_label"),
                    "property_type": parsed.get("property_type"),
                    "latitude": parsed.get("latitude"),
                    "longitude": parsed.get("longitude"),
                })
            except Exception:
                parse_errors += 1

        if not parsed_rows:
            print(f"Live listings: no valid properties parsed ({parse_errors} errors)")
            self.listings_df = pd.DataFrame()
            return

        print(f"  Parsed {len(parsed_rows)} properties ({parse_errors} errors). Batch scoring...")
        pred_out = self.prediction_service.predict_batch_parsed(parsed_rows)
        feat_df = pred_out["feat_df"]

        output_rows = []
        for i, meta in enumerate(meta_rows):
            predicted_eur = float(pred_out["predicted_eur"][i])
            low_eur = float(pred_out["low_eur"][i])
            high_eur = float(pred_out["high_eur"][i])
            hw_pct = float(pred_out["interval_halfwidth_pct"][i])
            asking_eur = meta["asking_price_eur"]
            delta_eur = predicted_eur - asking_eur
            delta_pct = (delta_eur / asking_eur * 100.0) if asking_eur else 0.0

            nbh = None
            if "neighbourhood" in feat_df.columns and pd.notna(feat_df["neighbourhood"].iloc[i]):
                cand = str(feat_df["neighbourhood"].iloc[i])
                if cand and cand != "Unknown":
                    nbh = cand
            if not nbh and meta["latitude"] and meta["longitude"] and self.data_loader:
                nbh = self.data_loader.get_neighbourhood_spatial(float(meta["latitude"]), float(meta["longitude"]))

            output_rows.append({
                "property_id": meta["property_id"],
                "address": meta["address"],
                "url": meta["url"],
                "asking_price": round(asking_eur),
                "predicted_price": round(predicted_eur),
                "low_eur": round(low_eur),
                "high_eur": round(high_eur),
                "interval_halfwidth_pct": round(hw_pct, 1),
                "delta_eur": round(delta_eur),
                "delta_pct": round(float(delta_pct), 1),
                "livable_area": meta["livable_area"],
                "bedrooms": meta["bedrooms"],
                "condition": meta["condition"],
                "energy_label": meta["energy_label"],
                "property_type": meta["property_type"],
                "neighbourhood": nbh,
                "date_obtained": iso_date,
            })

        self.listings_df = pd.DataFrame(output_rows)
        os.makedirs(os.path.dirname(CACHE_CSV), exist_ok=True)
        self.listings_df.to_csv(CACHE_CSV, index=False)
        print(f"Live listings: scored & cached {len(self.listings_df)} listings ({iso_date})")

    def _find_html_files(self):
        """Find HTML files from today or the most recent date within the last 7 days."""
        today = date.today()
        for days_ago in range(7):
            d = today - timedelta(days=days_ago)
            date_tag = d.strftime("%d%m%Y")
            pattern = os.path.join(DATA_DIR, "*", "*", f"*_{date_tag}.html")
            matches = glob.glob(pattern)
            if matches:
                seen = {}
                for path in matches:
                    basename = os.path.basename(path)
                    pid = basename.rsplit("_", 1)[0]
                    seen[pid] = path
                return [(path, pid) for pid, path in seen.items()], date_tag
        return [], None

    def get_listings(self, neighbourhood=None):
        """Return list of live listing dicts, optionally filtered by neighbourhood."""
        if self.listings_df is None or len(self.listings_df) == 0:
            return []

        df = self.listings_df
        if neighbourhood:
            def _normalize(s):
                return str(s).lower().replace("_", " ").replace("-", " ").strip()
            target = _normalize(neighbourhood)
            mask = df["neighbourhood"].dropna().apply(lambda x: _normalize(x) == target)
            df = df.loc[mask[mask].index]

        df = df.sort_values("delta_pct", ascending=False)
        records = df.to_dict(orient="records")
        for rec in records:
            for k, v in rec.items():
                if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    rec[k] = None
        return records

    def get_count(self):
        count = len(self.listings_df) if self.listings_df is not None else 0
        return {"count": count, "date_used": self.date_used}
