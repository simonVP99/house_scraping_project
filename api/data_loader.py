import os
import json
import math
import re
from datetime import datetime
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import shape, Point
from api.config import (
    OOF_ANALYTICS_PATH,
    EMBEDDINGS_CSV_PATH,
    TODAY_DEALS_CSV_PATH,
    NEIGHBOURHOOD_DATA_PATH,
)


def _prettify_wijk(name: str, display_map: dict | None = None) -> str:
    if not name or pd.isna(name):
        return "Other Ghent"
    s = str(name)
    if display_map and s in display_map:
        return display_map[s]
    if s == "Unknown":
        return "Greater Ghent / Unmapped"
    s = s.replace("_", " ")
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)
    return s.strip()


def _clean_nan(records):
    """Replace NaN/inf values with None in a list of dicts (JSON-safe)."""
    for row in records:
        for k, v in row.items():
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                row[k] = None
    return records


class DataLoader:
    def __init__(self):
        self.neighbourhood_polygons = []
        self.neighbourhood_display_map = {}
        self.analytics_df = None
        self.neighbourhood_summary = None
        self.today_deals_df = None
        self._spatial_tree = None
        self._spatial_prices = None
        self._load_neighbourhood_polygons()
        self._load_data()

    def _load_neighbourhood_polygons(self):
        self.neighbourhood_polygons = []
        self.neighbourhood_display_map = {}
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
                    norm = str(name).replace(" ", "").replace("-", "_")
                    pretty = re.sub(r"\s*-\s*", " – ", str(name).strip())
                    pretty = pretty.replace("en – zone", "en-zone")
                    self.neighbourhood_display_map[norm] = pretty
                    self.neighbourhood_display_map[str(name)] = pretty
                except Exception:
                    continue
        except Exception as e:
            print(f"Error loading neighbourhood data: {e}")

    def _load_data(self):
        """Load Phase-7 OOF analytics parquet (preferred) or fall back to legacy CSV."""
        if os.path.exists(OOF_ANALYTICS_PATH):
            df = pd.read_parquet(OOF_ANALYTICS_PATH)
            # Store raw EUR in price_eur and keep price in kEUR for backwards-compatible callers/tests
            df["price_eur"] = pd.to_numeric(df["price"], errors="coerce")
            df["price"] = df["price_eur"] / 1000.0
            for col in ["listing_duration_days", "livable_area"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            if "price_per_sqm" not in df.columns:
                df["price_per_sqm"] = np.where(
                    df["livable_area"] > 0,
                    df["price_eur"] / df["livable_area"],
                    np.nan,
                )
            df["is_under_option"] = df.get("is_under_option", pd.Series(False, index=df.index)).fillna(False).astype(bool)
            if "property_type_label" not in df.columns and "property_type_HOUSE" in df.columns:
                df["property_type_label"] = np.where(df["property_type_HOUSE"] == 1, "House", "Apartment")
            if "neighbourhood_display" not in df.columns:
                df["neighbourhood_display"] = df["neighbourhood"].map(
                    lambda x: _prettify_wijk(x, self.neighbourhood_display_map)
                )
            self.analytics_df = df
            self._build_spatial_tree()
            self._compute_neighbourhood_summary()
            print(f"Loaded Phase-7 OOF analytics: {len(df)} properties")
        elif os.path.exists(EMBEDDINGS_CSV_PATH):
            df = pd.read_csv(EMBEDDINGS_CSV_PATH)
            for col in ["price", "listing_duration_days", "livable_area"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df["price_eur"] = df["price"] * 1000.0
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
            df["neighbourhood_display"] = df["neighbourhood"].map(
                lambda x: _prettify_wijk(x, self.neighbourhood_display_map)
            )
            self.analytics_df = df
            self._build_spatial_tree()
            self._compute_neighbourhood_summary()
        else:
            print(f"Analytics data not found at {OOF_ANALYTICS_PATH} or {EMBEDDINGS_CSV_PATH}")

        if os.path.exists(TODAY_DEALS_CSV_PATH):
            self.today_deals_df = pd.read_csv(TODAY_DEALS_CSV_PATH)
            # Only merge legacy today_deals delta_pct when OOF Phase-7 delta_pct isn't already present
            if (
                self.analytics_df is not None
                and "delta_pct" not in self.analytics_df.columns
                and self.today_deals_df is not None
                and "delta_pct" in self.today_deals_df.columns
                and "property_id" in self.today_deals_df.columns
                and "property_id" in self.analytics_df.columns
            ):
                delta_map = self.today_deals_df.set_index("property_id")["delta_pct"]
                self.analytics_df["delta_pct"] = self.analytics_df["property_id"].map(delta_map)
                self._compute_neighbourhood_summary()

    def _build_spatial_tree(self):
        """Build KDTree from training data for spatial lag computation at inference."""
        df = self.analytics_df
        if df is None:
            return
        valid = df.dropna(subset=["latitude", "longitude", "price"])
        coords = valid[["latitude", "longitude"]].values
        self._spatial_tree = cKDTree(coords)
        self._spatial_prices = valid["price"].values
        print(f"Spatial tree built: {len(coords)} properties")

    def compute_spatial_lag(self, lat, lon):
        """Compute spatial lag for a new property using training data."""
        if self._spatial_tree is None or lat is None or lon is None:
            return None, None
        dists, indices = self._spatial_tree.query([lat, lon], k=10)
        spatial_lag_10 = float(np.mean(self._spatial_prices[indices]))
        avg_dist_5nn = float(np.mean(dists[:5] * 111))  # approx km
        return spatial_lag_10, avg_dist_5nn

    def _compute_neighbourhood_summary(self):
        """Aggregate per-neighbourhood statistics (excluding unmapped Unknown rows)."""
        df = self.analytics_df
        if df is None:
            return

        # Exclude unmapped 'Unknown' rows from the neighbourhood comparison & choropleth
        mapped = df[df["neighbourhood"].notna() & (df["neighbourhood"] != "Unknown")].copy()
        if len(mapped) == 0:
            mapped = df.copy()

        grouped = mapped.groupby("neighbourhood")
        summary = grouped.agg(
            count=("price", "size"),
            median_price_k=("price", "median"),
            median_price_sqm=("price_per_sqm", "median"),
            avg_listing_duration=("listing_duration_days", "mean"),
            median_listing_duration=("listing_duration_days", "median"),
            pct_under_option=("is_under_option", "mean"),
        ).reset_index()

        if "had_price_cut" in mapped.columns:
            cut_stats = mapped.groupby("neighbourhood")["had_price_cut"].mean().mul(100).round(1).reset_index(name="pct_price_cut")
            summary = summary.merge(cut_stats, on="neighbourhood", how="left")
        if "pred_mm_eur" in mapped.columns:
            fair_stats = mapped.groupby("neighbourhood")["pred_mm_eur"].median().round(0).reset_index(name="median_fair_eur")
            summary = summary.merge(fair_stats, on="neighbourhood", how="left")

        has_delta = "delta_pct" in mapped.columns and mapped["delta_pct"].notna().any()
        if has_delta:
            delta_stats = mapped[mapped["delta_pct"].notna()].groupby("neighbourhood").agg(
                median_delta_pct=("delta_pct", "median"),
                pct_underpriced=("delta_pct", lambda x: (x > 0).mean()),
            ).reset_index()
            summary = summary.merge(delta_stats, on="neighbourhood", how="left")
        else:
            summary["median_delta_pct"] = np.nan
            summary["pct_underpriced"] = np.nan

        display_map = getattr(self, "neighbourhood_display_map", {})
        summary["display_name"] = summary["neighbourhood"].map(lambda x: _prettify_wijk(x, display_map))
        summary["median_price_eur"] = (summary["median_price_k"] * 1000).round(0)
        summary["pct_under_option"] = (summary["pct_under_option"] * 100).round(1)
        summary["pct_underpriced"] = (summary["pct_underpriced"] * 100).round(1)
        summary["avg_listing_duration"] = summary["avg_listing_duration"].round(1)
        summary["median_listing_duration"] = summary["median_listing_duration"].round(0)
        summary["median_price_sqm"] = summary["median_price_sqm"].round(0)
        summary["median_delta_pct"] = summary["median_delta_pct"].round(1)
        summary = summary.sort_values(
            "median_price_sqm", ascending=False, na_position="last"
        )

        self.neighbourhood_summary = summary

    def get_stats(self):
        """Return KPI dashboard stats."""
        df = self.analytics_df
        if df is None:
            return {}
        mapped_nbhs = df[df["neighbourhood"].notna() & (df["neighbourhood"] != "Unknown")]["neighbourhood"].nunique()
        med_k = float(df["price"].median())
        return {
            "total_properties": len(df),
            "num_neighbourhoods": int(mapped_nbhs or df["neighbourhood"].nunique()),
            "median_price_k": round(med_k, 1),
            "median_price_eur": int(round(med_k * 1000)),
            "median_price_sqm": int(round(float(df["price_per_sqm"].median()))) if "price_per_sqm" in df.columns and df["price_per_sqm"].notna().any() else None,
            "avg_listing_duration": round(float(df["listing_duration_days"].mean()), 1),
            "pct_under_option": round(float(df["is_under_option"].mean() * 100), 1),
            "pct_price_cut": round(float(df["had_price_cut"].mean() * 100), 1) if "had_price_cut" in df.columns else None,
            "median_delta_pct": round(
                float(df["delta_pct"].median()) if "delta_pct" in df.columns and df["delta_pct"].notna().any() else 0, 1
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
        med_k = float(nbh_df["price"].median())
        min_k = float(nbh_df["price"].min())
        max_k = float(nbh_df["price"].max())
        display_map = getattr(self, "neighbourhood_display_map", {})
        stats = {
            "count": len(nbh_df),
            "display_name": _prettify_wijk(name, display_map),
            "median_price_k": round(med_k, 1),
            "median_price_eur": int(round(med_k * 1000)),
            "median_price_sqm": int(round(float(nbh_df["price_per_sqm"].median()))) if "price_per_sqm" in nbh_df.columns and nbh_df["price_per_sqm"].notna().any() else None,
            "min_price_k": round(min_k, 1),
            "max_price_k": round(max_k, 1),
            "min_price_eur": int(round(min_k * 1000)),
            "max_price_eur": int(round(max_k * 1000)),
            "avg_area": round(float(nbh_df["livable_area"].mean()), 0),
            "median_area": round(float(nbh_df["livable_area"].median()), 0),
            "avg_age": round(float(nbh_df["age"].mean()), 0) if "age" in nbh_df.columns else None,
            "pct_houses": round(float((nbh_df["property_type_HOUSE"] == 1).mean() * 100), 1) if "property_type_HOUSE" in nbh_df.columns else None,
            "pct_price_cut": round(float(nbh_df["had_price_cut"].mean() * 100), 1) if "had_price_cut" in nbh_df.columns else None,
            "top_condition": nbh_df["building_condition"].mode().iloc[0]
                if "building_condition" in nbh_df.columns and len(nbh_df["building_condition"].dropna()) > 0
                else None,
            "top_energy_label": nbh_df["energy_label"].mode().iloc[0]
                if "energy_label" in nbh_df.columns and len(nbh_df["energy_label"].dropna()) > 0
                else None,
        }

        # Monthly price trend: neighbourhood vs city over a continuous monthly timeline
        trend = []
        df_dated = df[df["date_obtained"].notna()].copy()
        if len(df_dated) > 0:
            df_dated["date_obtained"] = pd.to_datetime(df_dated["date_obtained"], errors="coerce")
            df_dated = df_dated[df_dated["date_obtained"].notna()]
            df_dated["month"] = df_dated["date_obtained"].dt.to_period("M")

            city_counts = df_dated.groupby("month")["price"].size()
            city_trend = df_dated.groupby("month")["price"].median()
            nbh_dated = df_dated[df_dated["neighbourhood"] == name]
            nbh_counts = nbh_dated.groupby("month")["price"].size()
            nbh_trend = nbh_dated.groupby("month")["price"].median()

            min_p = df_dated["month"].min()
            max_p = df_dated["month"].max()
            all_months = pd.period_range(min_p, max_p, freq="M")

            for p in all_months:
                m_str = str(p)
                c_val = float(city_trend.loc[p]) if p in city_trend.index and int(city_counts.loc[p]) >= 10 else None
                n_val = float(nbh_trend.loc[p]) if p in nbh_trend.index and int(nbh_counts.loc[p]) >= 3 else None
                trend.append({
                    "month": m_str,
                    "city_median_price_k": round(c_val, 1) if c_val is not None else None,
                    "neighbourhood_median_price_k": round(n_val, 1) if n_val is not None else None,
                    "city_median_price_eur": int(round(c_val * 1000)) if c_val is not None else None,
                    "neighbourhood_median_price_eur": int(round(n_val * 1000)) if n_val is not None else None,
                })

        # Top Phase-7 OOF value highlights in this neighbourhood (recent listings)
        recent_scored = []
        if "pred_mm_eur" in nbh_df.columns and "delta_pct" in nbh_df.columns:
            scored_sub = nbh_df.dropna(subset=["pred_mm_eur", "price", "delta_pct"]).copy()
            scored_sub = scored_sub.sort_values("date_obtained", ascending=False).head(60)
            scored_sub = scored_sub.sort_values("delta_pct", ascending=False).head(15)
            for _, r in scored_sub.iterrows():
                recent_scored.append({
                    "property_id": str(r.get("property_id", "")),
                    "asking_eur": int(round(float(r["price"]) * 1000)),
                    "predicted_eur": int(round(float(r["pred_mm_eur"]))),
                    "delta_pct": round(float(r["delta_pct"]), 1),
                    "livable_area": round(float(r["livable_area"])) if pd.notna(r.get("livable_area")) else None,
                    "bedrooms": int(r["number_of_bedrooms"]) if pd.notna(r.get("number_of_bedrooms")) else None,
                    "property_type": str(r.get("property_type_label") or ""),
                    "condition": str(r.get("building_condition") or "") if pd.notna(r.get("building_condition")) else None,
                    "energy_label": str(r.get("energy_label") or "") if pd.notna(r.get("energy_label")) else None,
                    "date": str(r["date_obtained"])[:10] if pd.notna(r.get("date_obtained")) else None,
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
            listings = _clean_nan(deals.to_dict(orient="records"))

        return {
            "name": name,
            "display_name": _prettify_wijk(name, display_map),
            "stats": stats,
            "trend": trend,
            "listings": listings,
            "recent_scored": recent_scored,
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
        return _clean_nan(df.to_dict(orient="records"))

    def get_neighbourhood_spatial(self, lat, lon):
        """Find neighbourhood for a lat/lon via point-in-polygon."""
        point = Point(lon, lat)
        for poly, name in self.neighbourhood_polygons:
            if poly.contains(point):
                return name
        return None

    def find_comparables(self, prop):
        """Find similar properties and compute how the subject differs from them."""
        df = self.analytics_df
        if df is None or len(df) == 0:
            return None

        nbh = prop.get("neighbourhood")
        nbh_norm = nbh.replace(" ", "").replace("-", "_") if nbh else None

        # Filter to same neighbourhood, fall back to same postal code
        candidates = df[df["neighbourhood"] == nbh_norm] if nbh_norm else pd.DataFrame()
        if len(candidates) < 5:
            pc = prop.get("postal_code")
            if pc:
                candidates = df[df["postal_code"].astype(str) == str(pc)]
        if len(candidates) < 3:
            return None

        area = prop.get("livable_area")
        beds = prop.get("number_of_bedrooms")
        ptype = 1 if prop.get("property_type") == "HOUSE" else prop.get("property_type_HOUSE", 0)
        condition = prop.get("building_condition")
        energy = prop.get("energy_label")
        asking = prop.get("price")

        # Similarity scoring
        scores = pd.Series(0.0, index=candidates.index)

        if area and "livable_area" in candidates.columns:
            area_diff = (candidates["livable_area"] - area).abs() / max(area, 1)
            scores += (1 - area_diff.clip(upper=1)) * 3

        if beds is not None and "number_of_bedrooms" in candidates.columns:
            bed_diff = (candidates["number_of_bedrooms"] - beds).abs()
            scores += (bed_diff <= 1).astype(float) * 2

        if "property_type_HOUSE" in candidates.columns:
            scores += (candidates["property_type_HOUSE"].astype(int) == int(ptype)).astype(float) * 2

        if condition and "building_condition" in candidates.columns:
            scores += (candidates["building_condition"] == condition).astype(float)

        if energy and "energy_label" in candidates.columns:
            energy_order = ["A++", "A+", "A", "B", "C", "D", "E", "F", "G"]
            if energy in energy_order:
                idx = energy_order.index(energy)
                def _energy_sim(e):
                    if e in energy_order:
                        return 1.0 if abs(energy_order.index(e) - idx) <= 1 else 0.0
                    return 0.0
                scores += candidates["energy_label"].apply(_energy_sim)

        top = scores.nlargest(12).index
        result_df = candidates.loc[top]

        cols = {
            "price": "price_k", "livable_area": "livable_area",
            "number_of_bedrooms": "bedrooms", "building_condition": "condition",
            "energy_label": "energy_label", "age": "age",
            "latitude": "latitude", "longitude": "longitude",
        }
        properties = []
        for _, row in result_df.iterrows():
            p = {}
            for src, dst in cols.items():
                v = row.get(src)
                p[dst] = None if (isinstance(v, float) and (math.isnan(v) or math.isinf(v))) else v
            p["price_eur"] = round(float(row["price"]) * 1000) if pd.notna(row.get("price")) else None
            p["price_per_sqm"] = round(row["price"] * 1000 / row["livable_area"]) if row.get("livable_area") else None
            pid = row.get("property_id")
            if pd.notna(pid):
                pid = str(int(pid)) if isinstance(pid, float) else str(pid)
                p["property_id"] = pid
                p["url"] = f"https://www.immoweb.be/en/classified/{pid}"
            else:
                p["property_id"] = None
                p["url"] = None
            properties.append(p)

        prices_k = result_df["price"].dropna()
        comp_psm = result_df.apply(
            lambda r: r["price"] * 1000 / r["livable_area"] if r.get("livable_area") else np.nan, axis=1
        ).dropna()

        # Subject's price/m²
        subject_psm = (asking / area) if (asking and area) else None

        # Compute how subject differs from comparable averages
        differentiators = []
        comp_avg_age = result_df["age"].mean() if "age" in result_df.columns else None
        subject_age = prop.get("age") or ((datetime.now().year - prop["year_built"]) if prop.get("year_built") else None)
        if subject_age is not None and comp_avg_age is not None and not np.isnan(comp_avg_age):
            diff = subject_age - comp_avg_age
            if abs(diff) > 5:
                differentiators.append({
                    "feature": "Age",
                    "subject": round(subject_age),
                    "comparable_avg": round(comp_avg_age),
                    "direction": "older" if diff > 0 else "newer",
                    "impact": "negative" if diff > 0 else "positive",
                })

        comp_avg_beds = result_df["number_of_bedrooms"].mean() if "number_of_bedrooms" in result_df.columns else None
        if beds is not None and comp_avg_beds is not None and not np.isnan(comp_avg_beds):
            diff = beds - comp_avg_beds
            if abs(diff) >= 0.5:
                differentiators.append({
                    "feature": "Bedrooms",
                    "subject": round(beds),
                    "comparable_avg": round(comp_avg_beds, 1),
                    "direction": "more" if diff > 0 else "fewer",
                    "impact": "positive" if diff > 0 else "negative",
                })

        comp_avg_area = result_df["livable_area"].mean() if "livable_area" in result_df.columns else None
        if area and comp_avg_area and not np.isnan(comp_avg_area):
            pct_diff = (area - comp_avg_area) / comp_avg_area * 100
            if abs(pct_diff) > 10:
                differentiators.append({
                    "feature": "Livable area",
                    "subject": round(area),
                    "comparable_avg": round(comp_avg_area),
                    "direction": "larger" if pct_diff > 0 else "smaller",
                    "impact": "positive" if pct_diff > 0 else "negative",
                })

        # Energy label comparison
        energy_order = ["A++", "A+", "A", "B", "C", "D", "E", "F", "G"]
        if energy and energy in energy_order:
            comp_labels = result_df["energy_label"].dropna() if "energy_label" in result_df.columns else pd.Series()
            comp_energy_scores = comp_labels.apply(
                lambda e: energy_order.index(e) if e in energy_order else np.nan
            ).dropna()
            if len(comp_energy_scores) > 0:
                avg_score = comp_energy_scores.mean()
                subject_score = energy_order.index(energy)
                if abs(subject_score - avg_score) > 0.8:
                    avg_label = energy_order[round(avg_score)]
                    differentiators.append({
                        "feature": "Energy label",
                        "subject": energy,
                        "comparable_avg": avg_label,
                        "direction": "worse" if subject_score > avg_score else "better",
                        "impact": "negative" if subject_score > avg_score else "positive",
                    })

        # Condition comparison
        if condition and "building_condition" in result_df.columns:
            cond_order = ["TO_RESTORE", "TO_RENOVATE", "TO_BE_DONE_UP", "GOOD", "JUST_RENOVATED", "AS_NEW"]
            comp_conditions = result_df["building_condition"].dropna()
            comp_cond_scores = comp_conditions.apply(
                lambda c: cond_order.index(c) if c in cond_order else np.nan
            ).dropna()
            if condition in cond_order and len(comp_cond_scores) > 0:
                avg_cs = comp_cond_scores.mean()
                subj_cs = cond_order.index(condition)
                if abs(subj_cs - avg_cs) > 0.8:
                    avg_cond = cond_order[round(avg_cs)]
                    differentiators.append({
                        "feature": "Condition",
                        "subject": condition.replace("_", " ").title(),
                        "comparable_avg": avg_cond.replace("_", " ").title(),
                        "direction": "better" if subj_cs > avg_cs else "worse",
                        "impact": "positive" if subj_cs > avg_cs else "negative",
                    })

        # Amenity value contributions: compare neighbourhood median price
        # with vs without each amenity to estimate its contribution
        amenity_contributions = []
        nbh_all = df[df["neighbourhood"] == nbh_norm] if nbh_norm else df
        if len(nbh_all) >= 20:
            amenity_checks = [
                ("has_garden", "garden_surface", "Garden", prop.get("has_garden"), prop.get("garden_surface")),
                ("has_terrace", "terrace_surface", "Terrace", prop.get("has_terrace"), prop.get("terrace_surface")),
                ("has_swimming_pool", None, "Swimming pool", prop.get("has_swimming_pool"), None),
                ("has_lift", None, "Lift", prop.get("has_lift"), None),
                ("has_double_glazing", None, "Double glazing", prop.get("has_double_glazing"), None),
                ("has_photovoltaic_panels", None, "Solar panels", prop.get("has_photovoltaic_panels"), None),
                ("has_heat_pump", None, "Heat pump", prop.get("has_heat_pump"), None),
            ]
            for bool_col, surface_col, label, has_it, surface_val in amenity_checks:
                if bool_col not in nbh_all.columns:
                    continue
                with_amenity = nbh_all[nbh_all[bool_col] == True]["price"].dropna()
                without_amenity = nbh_all[nbh_all[bool_col] == False]["price"].dropna()
                if len(with_amenity) < 15 or len(without_amenity) < 15:
                    continue
                diff_k = float(with_amenity.median() - without_amenity.median())
                if abs(diff_k) < 1:
                    continue
                subject_has = has_it in (True, 1, "True", "true") if has_it is not None else False
                entry = {
                    "feature": label,
                    "has_it": subject_has,
                    "estimated_value_k": round(diff_k, 1),
                    "pct_in_neighbourhood": round(float(nbh_all[bool_col].mean()) * 100),
                }
                if surface_col and surface_val and surface_val > 0:
                    entry["surface"] = round(float(surface_val))
                amenity_contributions.append(entry)

            # Parking contribution (count-based, not boolean)
            for park_col, label in [("parking_indoor", "Indoor parking"), ("parking_outdoor", "Outdoor parking")]:
                if park_col not in nbh_all.columns:
                    continue
                with_park = nbh_all[nbh_all[park_col].fillna(0) > 0]["price"].dropna()
                without_park = nbh_all[nbh_all[park_col].fillna(0) == 0]["price"].dropna()
                if len(with_park) < 15 or len(without_park) < 15:
                    continue
                diff_k = float(with_park.median() - without_park.median())
                if abs(diff_k) < 1:
                    continue
                subj_count = prop.get(park_col) or 0
                amenity_contributions.append({
                    "feature": label,
                    "has_it": subj_count > 0,
                    "count": int(subj_count) if subj_count else 0,
                    "estimated_value_k": round(diff_k, 1),
                    "pct_in_neighbourhood": round(float((nbh_all[park_col].fillna(0) > 0).mean()) * 100),
                })

        # Price/m² distribution for scale visualization
        psm_distribution = sorted(comp_psm.tolist())

        return {
            "properties": properties,
            "median_price_k": round(float(prices_k.median()), 1),
            "min_price_k": round(float(prices_k.min()), 1),
            "max_price_k": round(float(prices_k.max()), 1),
            "median_price_per_sqm": round(float(comp_psm.median())) if len(comp_psm) > 0 else None,
            "subject_price_per_sqm": round(subject_psm) if subject_psm else None,
            "price_per_sqm_distribution": [round(v) for v in psm_distribution],
            "differentiators": differentiators,
            "amenity_contributions": amenity_contributions,
            "count": len(properties),
            "neighbourhood": nbh or nbh_norm,
        }

    def get_feature_context(self, prop):
        """Compare property features against neighbourhood medians with percentile ranks."""
        df = self.analytics_df
        if df is None or len(df) == 0:
            return []

        nbh = prop.get("neighbourhood")
        nbh_norm = nbh.replace(" ", "").replace("-", "_") if nbh else None
        nbh_df = df[df["neighbourhood"] == nbh_norm] if nbh_norm else df

        if len(nbh_df) < 5:
            return []

        asking = prop.get("price")
        area = prop.get("livable_area")
        price_per_sqm = (asking / area) if (asking and area) else None

        features = [
            ("Price/m\u00B2", price_per_sqm, "price_per_sqm"),
            ("Livable Area (m\u00B2)", area, "livable_area"),
            ("Bedrooms", prop.get("number_of_bedrooms"), "number_of_bedrooms"),
            ("Age (years)", prop.get("age") or (
                (datetime.now().year - prop["year_built"]) if prop.get("year_built") else None
            ), "age"),
            ("Garden (m\u00B2)", prop.get("garden_surface"), "garden_surface"),
            ("Cadastral Income", prop.get("cadastral_income"), "cadastral_income"),
        ]

        result = []
        for label, value, col in features:
            if value is None or col not in nbh_df.columns:
                continue
            series = nbh_df[col].dropna()
            if len(series) < 5:
                continue
            median = float(series.median())
            percentile = float((series < value).mean() * 100)
            direction = "above" if value > median else ("below" if value < median else "equal")
            result.append({
                "feature": label,
                "value": round(float(value), 1),
                "neighbourhood_median": round(median, 1),
                "percentile": round(percentile),
                "direction": direction,
            })

        return result
