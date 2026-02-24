import os
import json
import pandas as pd
import numpy as np
from shapely.geometry import shape, Point
from api.config import (
    EMBEDDINGS_CSV_PATH, TODAY_DEALS_CSV_PATH,
    NEIGHBOURHOOD_DATA_PATH,
)


class DataLoader:
    def __init__(self):
        self.neighbourhood_polygons = []
        self.analytics_df = None
        self.neighbourhood_summary = None
        self.today_deals_df = None
        self._load_neighbourhood_polygons()
        self._load_data()

    def _load_neighbourhood_polygons(self):
        self.neighbourhood_polygons = []
        try:
            if not os.path.exists(NEIGHBOURHOOD_DATA_PATH):
                return
            df = pd.read_csv(NEIGHBOURHOOD_DATA_PATH, sep=";")
            for _, row in df.iterrows():
                try:
                    geo_json = json.loads(row["Geometry"])
                    poly = shape(geo_json)
                    name = row["wijk"]
                    self.neighbourhood_polygons.append((poly, name))
                except Exception:
                    continue
        except Exception as e:
            print(f"Error loading neighbourhood data: {e}")

    def _load_data(self):
        """Load embeddings CSV and today_deals CSV into memory."""
        if os.path.exists(EMBEDDINGS_CSV_PATH):
            df = pd.read_csv(EMBEDDINGS_CSV_PATH)
            for col in ["price", "listing_duration_days", "livable_area"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df["price_per_sqm"] = np.where(
                df["livable_area"] > 0,
                df["price"] * 1000 / df["livable_area"],
                np.nan,
            )
            df["is_under_option"] = df["is_under_option"].fillna(False).astype(bool)
            if "property_type_HOUSE" in df.columns:
                df["property_type_label"] = np.where(
                    df["property_type_HOUSE"] == 1, "House", "Apartment"
                )
            self.analytics_df = df
            self._compute_neighbourhood_summary()
        else:
            print(f"Embeddings CSV not found at {EMBEDDINGS_CSV_PATH}")

        if os.path.exists(TODAY_DEALS_CSV_PATH):
            self.today_deals_df = pd.read_csv(TODAY_DEALS_CSV_PATH)

    def _compute_neighbourhood_summary(self):
        """Aggregate per-neighbourhood statistics."""
        df = self.analytics_df
        if df is None:
            return

        grouped = df.groupby("neighbourhood")
        summary = grouped.agg(
            count=("price", "size"),
            median_price_k=("price", "median"),
            median_price_sqm=("price_per_sqm", "median"),
            avg_listing_duration=("listing_duration_days", "mean"),
            median_listing_duration=("listing_duration_days", "median"),
            pct_under_option=("is_under_option", "mean"),
        ).reset_index()

        has_delta = "delta_pct" in df.columns and df["delta_pct"].notna().any()
        if has_delta:
            delta_stats = df[df["delta_pct"].notna()].groupby("neighbourhood").agg(
                median_delta_pct=("delta_pct", "median"),
                pct_underpriced=("delta_pct", lambda x: (x > 0).mean()),
            ).reset_index()
            summary = summary.merge(delta_stats, on="neighbourhood", how="left")
        else:
            summary["median_delta_pct"] = np.nan
            summary["pct_underpriced"] = np.nan

        summary["pct_under_option"] = (summary["pct_under_option"] * 100).round(1)
        summary["pct_underpriced"] = (summary["pct_underpriced"] * 100).round(1)
        summary["avg_listing_duration"] = summary["avg_listing_duration"].round(1)
        summary["median_listing_duration"] = summary["median_listing_duration"].round(0)
        summary["median_price_sqm"] = summary["median_price_sqm"].round(0)
        summary["median_delta_pct"] = summary["median_delta_pct"].round(1)
        summary = summary.sort_values(
            "median_delta_pct", ascending=False, na_position="last"
        )

        self.neighbourhood_summary = summary

    def get_stats(self):
        """Return KPI dashboard stats."""
        df = self.analytics_df
        if df is None:
            return {}
        return {
            "total_properties": len(df),
            "num_neighbourhoods": df["neighbourhood"].nunique(),
            "median_price_k": round(float(df["price"].median()), 1),
            "avg_listing_duration": round(float(df["listing_duration_days"].mean()), 1),
            "pct_under_option": round(float(df["is_under_option"].mean() * 100), 1),
            "median_delta_pct": round(
                float(df["delta_pct"].median()) if df["delta_pct"].notna().any() else 0, 1
            ),
        }

    def get_neighbourhood_geojson(self, neighbourhood_name):
        """Return GeoJSON dict for a neighbourhood polygon."""
        for poly, name in self.neighbourhood_polygons:
            norm_name = name.replace(" ", "").replace("-", "_")
            if norm_name == neighbourhood_name or name == neighbourhood_name:
                return json.loads(json.dumps(poly.__geo_interface__))
        return None

    def get_neighbourhood_spatial(self, lat, lon):
        """Find neighbourhood for a lat/lon via point-in-polygon."""
        point = Point(lon, lat)
        for poly, name in self.neighbourhood_polygons:
            if poly.contains(point):
                return name
        return None
