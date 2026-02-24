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

    def get_neighbourhood_detail(self, name):
        """Return deep-dive data for a single neighbourhood."""
        df = self.analytics_df
        if df is None:
            return None

        nbh_df = df[df["neighbourhood"] == name]
        if len(nbh_df) == 0:
            return None

        # Stats
        stats = {
            "count": len(nbh_df),
            "median_price_k": round(float(nbh_df["price"].median()), 1),
            "min_price_k": round(float(nbh_df["price"].min()), 1),
            "max_price_k": round(float(nbh_df["price"].max()), 1),
            "avg_area": round(float(nbh_df["livable_area"].mean()), 0),
            "avg_age": round(float(nbh_df["age"].mean()), 0) if "age" in nbh_df.columns else None,
            "top_condition": nbh_df["building_condition"].mode().iloc[0]
                if "building_condition" in nbh_df.columns and len(nbh_df["building_condition"].dropna()) > 0
                else None,
            "top_energy_label": nbh_df["energy_label"].mode().iloc[0]
                if "energy_label" in nbh_df.columns and len(nbh_df["energy_label"].dropna()) > 0
                else None,
        }

        # Monthly price trend: neighbourhood vs city
        trend = []
        df_dated = df[df["date_obtained"].notna()].copy()
        if len(df_dated) > 0:
            df_dated["date_obtained"] = pd.to_datetime(df_dated["date_obtained"], errors="coerce")
            df_dated = df_dated[df_dated["date_obtained"].notna()]
            df_dated["month"] = df_dated["date_obtained"].dt.to_period("M").astype(str)

            city_trend = df_dated.groupby("month")["price"].median().reset_index()
            nbh_dated = df_dated[df_dated["neighbourhood"] == name]
            nbh_trend = nbh_dated.groupby("month")["price"].median().reset_index()

            months = sorted(set(city_trend["month"].tolist() + nbh_trend["month"].tolist()))
            city_dict = dict(zip(city_trend["month"], city_trend["price"]))
            nbh_dict = dict(zip(nbh_trend["month"], nbh_trend["price"]))

            for m in months:
                trend.append({
                    "month": m,
                    "city_median_price_k": round(float(city_dict.get(m, 0)), 1) if m in city_dict else None,
                    "neighbourhood_median_price_k": round(float(nbh_dict.get(m, 0)), 1) if m in nbh_dict else None,
                })

        # Listings from today_deals
        listings = []
        if self.today_deals_df is not None and len(self.today_deals_df) > 0:
            deals = self.today_deals_df[self.today_deals_df["neighbourhood"] == name]
            display_cols = [c for c in [
                "property_id", "price", "predicted_k", "delta_pct",
                "livable_area", "number_of_bedrooms", "building_condition", "energy_label",
            ] if c in deals.columns]
            deals = deals[display_cols].sort_values("delta_pct", ascending=False, na_position="last")
            listings = deals.to_dict(orient="records")

        return {
            "name": name,
            "stats": stats,
            "trend": trend,
            "listings": listings,
        }

    def get_today_deals(self):
        """Return today's deals sorted by delta."""
        if self.today_deals_df is None:
            return []
        display_cols = [c for c in [
            "property_id", "price", "predicted_k", "delta_pct",
            "livable_area", "number_of_bedrooms", "building_condition", "energy_label",
            "neighbourhood", "postal_code",
        ] if c in self.today_deals_df.columns]
        df = self.today_deals_df[display_cols].sort_values(
            "delta_pct", ascending=False, na_position="last"
        )
        return df.to_dict(orient="records")

    def get_neighbourhood_spatial(self, lat, lon):
        """Find neighbourhood for a lat/lon via point-in-polygon."""
        point = Point(lon, lat)
        for poly, name in self.neighbourhood_polygons:
            if poly.contains(point):
                return name
        return None
