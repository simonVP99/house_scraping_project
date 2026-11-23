# React + Flask Frontend Rebuild Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the Gradio frontend with a React + TypeScript SPA backed by Flask REST APIs, starting with the core features (URL price predictor + neighbourhood dashboard).

**Architecture:** Two-process setup: Flask API (port 5000) serves JSON endpoints and loads existing models/CSVs from `outputs/`. React frontend (Vite dev server port 5173, production static build) consumes these APIs. Heavy operations (scraping, training) remain CLI-only.

**Tech Stack:** React 18, TypeScript (strict), Vite, Tailwind CSS, shadcn/ui, react-leaflet, Recharts, TanStack React Query, Flask, Flask-RESTful, Flask-CORS

---

### Task 1: Flask API Skeleton

**Files:**
- Create: `api/__init__.py`
- Create: `api/app.py`
- Create: `api/config.py`
- Create: `api/requirements.txt`
- Test: `api/tests/__init__.py`
- Test: `api/tests/test_health.py`

**Step 1: Write the failing test**

```python
# api/tests/test_health.py
import pytest
from api.app import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_health_endpoint(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "ok"
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/simon/Desktop/personal/immo_datacollection/house_scraping_project && python -m pytest api/tests/test_health.py -v`
Expected: FAIL (module not found)

**Step 3: Write minimal implementation**

```python
# api/config.py
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

EMBEDDINGS_CSV_PATH = os.path.join(PROJECT_DIR, "outputs", "property_data_with_embeddings.csv")
TODAY_DEALS_CSV_PATH = os.path.join(PROJECT_DIR, "outputs", "today_deals.csv")
AUTOGLUON_PATH = os.path.join(PROJECT_DIR, "outputs", "models", "autogluon")
PCA_PATH = os.path.join(PROJECT_DIR, "outputs", "models", "description_pca.joblib")
OLD_MODEL_PATH = os.path.join(PROJECT_DIR, "outputs", "models", "house_price_model.pkl")
OLD_OPTIONS_PATH = os.path.join(PROJECT_DIR, "outputs", "models", "feature_options.pkl")
NEIGHBOURHOOD_DATA_PATH = os.path.join(PROJECT_DIR, "external_data", "stadswijken-gent.csv")
E5_MODEL_NAME = "intfloat/multilingual-e5-small"
PCA_COMPONENTS = 50
```

```python
# api/__init__.py
```

```python
# api/app.py
from flask import Flask, jsonify
from flask_cors import CORS


def create_app():
    app = Flask(__name__)
    CORS(app, origins=["http://localhost:5173"])

    @app.route("/api/health")
    def health():
        return jsonify({"status": "ok"})

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, port=5000)
```

```
# api/requirements.txt
flask>=3.0
flask-cors>=4.0
flask-restful>=0.3.10
pandas>=2.0
numpy>=1.24
joblib>=1.3
requests>=2.31
geopy>=2.4
shapely>=2.0
scipy>=1.11
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/simon/Desktop/personal/immo_datacollection/house_scraping_project && pip install flask flask-cors flask-restful && python -m pytest api/tests/test_health.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add api/
git commit -m "feat: add Flask API skeleton with health endpoint"
```

---

### Task 2: Data Loading Module

**Files:**
- Create: `api/data_loader.py`
- Test: `api/tests/test_data_loader.py`

This module loads CSVs and neighbourhood geometry at startup and caches them in memory. All other endpoints read from this cache.

**Step 1: Write the failing test**

```python
# api/tests/test_data_loader.py
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from api.data_loader import DataLoader


def test_data_loader_init():
    """DataLoader can be instantiated."""
    loader = DataLoader.__new__(DataLoader)
    assert loader is not None


def test_load_neighbourhood_polygons_returns_list():
    """load_neighbourhood_polygons returns a list of (polygon, name) tuples."""
    loader = DataLoader.__new__(DataLoader)
    loader.neighbourhood_polygons = []
    loader._load_neighbourhood_polygons()
    # Even with missing file, should return empty list without crashing
    assert isinstance(loader.neighbourhood_polygons, list)


def test_get_stats_returns_dict():
    """get_stats returns a dict with expected keys."""
    loader = DataLoader.__new__(DataLoader)
    loader.analytics_df = pd.DataFrame({
        "price": [200, 300, 400],
        "listing_duration_days": [10, 20, 30],
        "is_under_option": [False, True, False],
        "delta_pct": [5.0, -3.0, 12.0],
        "neighbourhood": ["A", "B", "C"],
    })
    stats = loader.get_stats()
    assert "total_properties" in stats
    assert "median_price_k" in stats
    assert stats["total_properties"] == 3
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest api/tests/test_data_loader.py -v`
Expected: FAIL (module not found)

**Step 3: Write minimal implementation**

```python
# api/data_loader.py
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
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest api/tests/test_data_loader.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add api/data_loader.py api/tests/test_data_loader.py
git commit -m "feat: add DataLoader for CSV and neighbourhood polygon loading"
```

---

### Task 3: Stats & Neighbourhood List Endpoints

**Files:**
- Modify: `api/app.py`
- Test: `api/tests/test_endpoints.py`

**Step 1: Write the failing test**

```python
# api/tests/test_endpoints.py
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from api.app import create_app


@pytest.fixture
def mock_loader():
    loader = MagicMock()
    loader.get_stats.return_value = {
        "total_properties": 8000,
        "num_neighbourhoods": 25,
        "median_price_k": 345.0,
        "avg_listing_duration": 42.0,
        "pct_under_option": 18.5,
        "median_delta_pct": -2.3,
    }
    loader.neighbourhood_summary = pd.DataFrame({
        "neighbourhood": ["Centrum", "Brugse_Poort"],
        "count": [200, 150],
        "median_price_k": [380.0, 290.0],
        "median_price_sqm": [3200.0, 2800.0],
        "avg_listing_duration": [35.0, 48.0],
        "median_listing_duration": [30.0, 42.0],
        "pct_under_option": [22.0, 15.0],
        "median_delta_pct": [3.5, -5.2],
        "pct_underpriced": [55.0, 38.0],
    })
    loader.get_neighbourhood_geojson.return_value = {
        "type": "Polygon",
        "coordinates": [[[3.72, 51.05], [3.73, 51.05], [3.73, 51.06], [3.72, 51.05]]],
    }
    return loader


@pytest.fixture
def client(mock_loader):
    with patch("api.app.data_loader", mock_loader):
        app = create_app(loader=mock_loader)
        app.config["TESTING"] = True
        with app.test_client() as client:
            yield client


def test_stats_endpoint(client):
    resp = client.get("/api/stats")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["total_properties"] == 8000
    assert data["median_price_k"] == 345.0


def test_neighbourhoods_endpoint(client):
    resp = client.get("/api/neighbourhoods")
    assert resp.status_code == 200
    data = resp.get_json()
    assert len(data["neighbourhoods"]) == 2
    assert "geojson" in data["neighbourhoods"][0]
    assert data["neighbourhoods"][0]["name"] == "Centrum"
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest api/tests/test_endpoints.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

Update `api/app.py`:

```python
# api/app.py
from flask import Flask, jsonify, request
from flask_cors import CORS
from api.data_loader import DataLoader

data_loader = None


def create_app(loader=None):
    global data_loader
    app = Flask(__name__)
    CORS(app, origins=["http://localhost:5173"])

    if loader is not None:
        data_loader = loader
    elif data_loader is None:
        data_loader = DataLoader()

    @app.route("/api/health")
    def health():
        return jsonify({"status": "ok"})

    @app.route("/api/stats")
    def stats():
        return jsonify(data_loader.get_stats())

    @app.route("/api/neighbourhoods")
    def neighbourhoods():
        summary = data_loader.neighbourhood_summary
        if summary is None:
            return jsonify({"neighbourhoods": []})

        result = []
        for _, row in summary.iterrows():
            nbh = row["neighbourhood"]
            geojson = data_loader.get_neighbourhood_geojson(nbh)
            result.append({
                "name": nbh,
                "count": int(row["count"]),
                "median_price_k": float(row["median_price_k"]),
                "median_price_sqm": float(row.get("median_price_sqm", 0)),
                "avg_listing_duration": float(row.get("avg_listing_duration", 0)),
                "median_listing_duration": float(row.get("median_listing_duration", 0)),
                "pct_under_option": float(row.get("pct_under_option", 0)),
                "median_delta_pct": float(row.get("median_delta_pct", 0))
                    if not pd.isna(row.get("median_delta_pct")) else None,
                "pct_underpriced": float(row.get("pct_underpriced", 0))
                    if not pd.isna(row.get("pct_underpriced")) else None,
                "geojson": geojson,
            })

        return jsonify({"neighbourhoods": result})

    return app


import pandas as pd

if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, port=5000)
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest api/tests/test_endpoints.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add api/app.py api/tests/test_endpoints.py
git commit -m "feat: add /api/stats and /api/neighbourhoods endpoints"
```

---

### Task 4: Neighbourhood Detail & Today's Deals Endpoints

**Files:**
- Modify: `api/app.py`
- Modify: `api/data_loader.py`
- Test: `api/tests/test_neighbourhood_detail.py`

**Step 1: Write the failing test**

```python
# api/tests/test_neighbourhood_detail.py
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from api.app import create_app


@pytest.fixture
def mock_loader():
    loader = MagicMock()

    # Analytics DF with date_obtained for trend calculation
    loader.analytics_df = pd.DataFrame({
        "neighbourhood": ["Centrum"] * 6,
        "price": [300, 320, 310, 330, 340, 350],
        "livable_area": [100, 110, 105, 120, 115, 130],
        "age": [50, 30, 60, 20, 40, 10],
        "building_condition": ["GOOD"] * 6,
        "energy_label": ["C"] * 6,
        "date_obtained": pd.to_datetime([
            "2025-01-15", "2025-01-20", "2025-02-10",
            "2025-02-15", "2025-03-01", "2025-03-10"
        ]),
        "delta_pct": [5.0, -3.0, 8.0, -1.0, 12.0, 2.0],
        "listing_duration_days": [20, 30, 15, 25, 10, 35],
        "is_under_option": [False, True, False, False, True, False],
        "property_type_HOUSE": [1, 0, 1, 0, 1, 0],
    })

    loader.today_deals_df = pd.DataFrame({
        "property_id": ["123", "456"],
        "neighbourhood": ["Centrum", "Centrum"],
        "price": [300, 350],
        "predicted_k": [320, 330],
        "delta_pct": [6.7, -5.7],
        "livable_area": [100, 120],
        "number_of_bedrooms": [3, 2],
        "building_condition": ["GOOD", "AS_NEW"],
        "energy_label": ["C", "B"],
    })

    loader.neighbourhood_summary = pd.DataFrame({
        "neighbourhood": ["Centrum"],
        "count": [6],
        "median_price_k": [325.0],
        "median_price_sqm": [3000.0],
        "avg_listing_duration": [22.5],
        "median_listing_duration": [22.0],
        "pct_under_option": [33.3],
        "median_delta_pct": [3.5],
        "pct_underpriced": [66.7],
    })
    loader.get_neighbourhood_geojson.return_value = None
    loader.get_stats.return_value = {}

    return loader


@pytest.fixture
def client(mock_loader):
    app = create_app(loader=mock_loader)
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_neighbourhood_detail(client):
    resp = client.get("/api/neighbourhoods/Centrum")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["name"] == "Centrum"
    assert "stats" in data
    assert "trend" in data
    assert "listings" in data
    assert data["stats"]["count"] == 6


def test_neighbourhood_detail_not_found(client):
    resp = client.get("/api/neighbourhoods/NonExistent")
    assert resp.status_code == 404


def test_today_deals_endpoint(client):
    resp = client.get("/api/properties/today")
    assert resp.status_code == 200
    data = resp.get_json()
    assert len(data["properties"]) == 2
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest api/tests/test_neighbourhood_detail.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

Add methods to `api/data_loader.py`:

```python
    # Add to DataLoader class:

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
```

Add endpoints to `api/app.py`:

```python
    @app.route("/api/neighbourhoods/<name>")
    def neighbourhood_detail(name):
        detail = data_loader.get_neighbourhood_detail(name)
        if detail is None:
            return jsonify({"error": "Neighbourhood not found"}), 404
        return jsonify(detail)

    @app.route("/api/properties/today")
    def today_deals():
        return jsonify({"properties": data_loader.get_today_deals()})
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest api/tests/test_neighbourhood_detail.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add api/
git commit -m "feat: add neighbourhood detail and today deals endpoints"
```

---

### Task 5: URL Prediction Endpoint

**Files:**
- Create: `api/prediction.py`
- Modify: `api/app.py`
- Test: `api/tests/test_prediction.py`

This is the most complex endpoint. It reuses the `fetch_and_parse_url` and `prepare_single_property` logic from `src/app.py`.

**Step 1: Write the failing test**

```python
# api/tests/test_prediction.py
import pytest
from unittest.mock import patch, MagicMock
import numpy as np
from api.app import create_app
from api.prediction import PredictionService


def test_parse_immoweb_json_extracts_price():
    """parse_immoweb_json extracts price from window.classified JSON."""
    service = PredictionService.__new__(PredictionService)
    html = '''<script>window.classified = {"price":{"mainValue":350000},"property":{"type":"HOUSE","subtype":"HOUSE","location":{"postalCode":"9000","latitude":51.05,"longitude":3.72},"netHabitableSurface":120,"bedroomCount":3}};</script>'''
    result = service._parse_immoweb_html(html, "http://test/12345")
    assert result["price"] == 350000
    assert result["property_id"] == "12345"


def test_predict_url_invalid():
    """Invalid URL returns error."""
    loader = MagicMock()
    loader.get_neighbourhood_spatial.return_value = None
    service = PredictionService(ag_predictor=None, data_loader=loader)
    result, status = service.predict_from_url("not-a-url")
    assert status == 400
    assert "error" in result
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest api/tests/test_prediction.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# api/prediction.py
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
            predicted_k = self.ag_predictor.predict(features_df).values[0]
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
                    "delta_pct": round((predicted_eur - asking_price) / asking_price * 100, 1) if asking_price else None,
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
```

Add endpoints to `api/app.py`:

```python
    @app.route("/api/predict/url")
    def predict_url():
        url = request.args.get("url")
        result, status = prediction_service.predict_from_url(url)
        return jsonify(result), status

    @app.route("/api/predict/manual", methods=["POST"])
    def predict_manual():
        form_data = request.get_json()
        result, status = prediction_service.predict_manual(form_data)
        return jsonify(result), status
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest api/tests/test_prediction.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add api/prediction.py api/app.py api/tests/test_prediction.py
git commit -m "feat: add URL and manual prediction endpoints"
```

---

### Task 6: React Project Scaffold

**Files:**
- Create: `frontend/package.json`
- Create: `frontend/tsconfig.json`
- Create: `frontend/vite.config.ts`
- Create: `frontend/index.html`
- Create: `frontend/src/main.tsx`
- Create: `frontend/src/App.tsx`
- Create: `frontend/tailwind.config.ts`
- Create: `frontend/postcss.config.js`
- Create: `frontend/src/index.css`

**Step 1: Initialize React + Vite + TypeScript project**

```bash
cd /Users/simon/Desktop/personal/immo_datacollection/house_scraping_project
npm create vite@latest frontend -- --template react-ts
cd frontend
npm install
npm install -D tailwindcss @tailwindcss/vite
npm install @tanstack/react-query react-router-dom
npm install react-leaflet leaflet recharts
npm install -D @types/leaflet
```

**Step 2: Configure Tailwind**

Update `frontend/vite.config.ts`:
```typescript
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://localhost:5000",
        changeOrigin: true,
      },
    },
  },
});
```

Update `frontend/src/index.css`:
```css
@import "tailwindcss";
```

**Step 3: Install shadcn/ui**

```bash
cd frontend
npx shadcn@latest init
# Select: New York style, Slate base color, CSS variables
npx shadcn@latest add card table button input select badge separator
```

**Step 4: Set up App shell with router**

```typescript
// frontend/src/App.tsx
import { BrowserRouter, Routes, Route, NavLink } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import PredictPage from "./pages/PredictPage";
import NeighbourhoodsPage from "./pages/NeighbourhoodsPage";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5 * 60 * 1000,
      retry: 1,
    },
  },
});

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <div className="min-h-screen bg-background">
          <nav className="border-b bg-white">
            <div className="mx-auto max-w-7xl px-4 flex h-14 items-center gap-6">
              <span className="font-bold text-lg">Ghent RE Intelligence</span>
              <NavLink
                to="/predict"
                className={({ isActive }) =>
                  `text-sm font-medium ${isActive ? "text-blue-600 border-b-2 border-blue-600" : "text-muted-foreground"}`
                }
              >
                Price Predictor
              </NavLink>
              <NavLink
                to="/neighbourhoods"
                className={({ isActive }) =>
                  `text-sm font-medium ${isActive ? "text-blue-600 border-b-2 border-blue-600" : "text-muted-foreground"}`
                }
              >
                Neighbourhoods
              </NavLink>
            </div>
          </nav>
          <main className="mx-auto max-w-7xl px-4 py-6">
            <Routes>
              <Route path="/" element={<PredictPage />} />
              <Route path="/predict" element={<PredictPage />} />
              <Route path="/neighbourhoods" element={<NeighbourhoodsPage />} />
            </Routes>
          </main>
        </div>
      </BrowserRouter>
    </QueryClientProvider>
  );
}
```

**Step 5: Create placeholder pages**

```typescript
// frontend/src/pages/PredictPage.tsx
export default function PredictPage() {
  return <div>Price Predictor - coming soon</div>;
}
```

```typescript
// frontend/src/pages/NeighbourhoodsPage.tsx
export default function NeighbourhoodsPage() {
  return <div>Neighbourhoods - coming soon</div>;
}
```

**Step 6: Verify it builds**

```bash
cd frontend && npm run build
```
Expected: Build succeeds

**Step 7: Commit**

```bash
git add frontend/
git commit -m "feat: scaffold React + Vite + TypeScript + Tailwind + shadcn/ui frontend"
```

---

### Task 7: TypeScript Types & API Client

**Files:**
- Create: `frontend/src/types/prediction.ts`
- Create: `frontend/src/types/neighbourhood.ts`
- Create: `frontend/src/types/property.ts`
- Create: `frontend/src/services/api.ts`
- Create: `frontend/src/hooks/useStats.ts`
- Create: `frontend/src/hooks/useNeighbourhoods.ts`
- Create: `frontend/src/hooks/usePrediction.ts`

**Step 1: Define TypeScript types**

```typescript
// frontend/src/types/prediction.ts
export interface PredictionResult {
  property: {
    property_id: string | null;
    address: string;
    price: number | null;
    livable_area: number | null;
    bedrooms: number | null;
    year_built: number | null;
    condition: string | null;
    energy_label: string | null;
    property_type: string | null;
    property_subtype: string | null;
    neighbourhood: string | null;
  };
  prediction: {
    predicted_price: number;
    delta_eur: number | null;
    delta_pct: number | null;
  };
  form_data: Record<string, unknown>;
}

export interface ManualPredictionResult {
  predicted_price: number;
}
```

```typescript
// frontend/src/types/neighbourhood.ts
export interface NeighbourhoodSummary {
  name: string;
  count: number;
  median_price_k: number;
  median_price_sqm: number;
  avg_listing_duration: number;
  median_listing_duration: number;
  pct_under_option: number;
  median_delta_pct: number | null;
  pct_underpriced: number | null;
  geojson: GeoJSON.Geometry | null;
}

export interface NeighbourhoodDetail {
  name: string;
  stats: {
    count: number;
    median_price_k: number;
    min_price_k: number;
    max_price_k: number;
    avg_area: number;
    avg_age: number | null;
    top_condition: string | null;
    top_energy_label: string | null;
  };
  trend: Array<{
    month: string;
    city_median_price_k: number | null;
    neighbourhood_median_price_k: number | null;
  }>;
  listings: TodayDeal[];
}

export interface DashboardStats {
  total_properties: number;
  num_neighbourhoods: number;
  median_price_k: number;
  avg_listing_duration: number;
  pct_under_option: number;
  median_delta_pct: number;
}
```

```typescript
// frontend/src/types/property.ts
export interface TodayDeal {
  property_id: string;
  price: number;
  predicted_k: number;
  delta_pct: number;
  livable_area: number;
  number_of_bedrooms: number;
  building_condition: string;
  energy_label: string;
  neighbourhood?: string;
  postal_code?: string;
}
```

**Step 2: Write API client**

```typescript
// frontend/src/services/api.ts
import type { PredictionResult, ManualPredictionResult } from "../types/prediction";
import type { NeighbourhoodSummary, NeighbourhoodDetail, DashboardStats } from "../types/neighbourhood";
import type { TodayDeal } from "../types/property";

const BASE = "/api";

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.error || `HTTP ${res.status}`);
  }
  return res.json();
}

export const api = {
  getStats: () => fetchJson<DashboardStats>(`${BASE}/stats`),

  getNeighbourhoods: () =>
    fetchJson<{ neighbourhoods: NeighbourhoodSummary[] }>(`${BASE}/neighbourhoods`)
      .then((r) => r.neighbourhoods),

  getNeighbourhoodDetail: (name: string) =>
    fetchJson<NeighbourhoodDetail>(`${BASE}/neighbourhoods/${encodeURIComponent(name)}`),

  getTodayDeals: () =>
    fetchJson<{ properties: TodayDeal[] }>(`${BASE}/properties/today`)
      .then((r) => r.properties),

  predictFromUrl: (url: string) =>
    fetchJson<PredictionResult>(`${BASE}/predict/url?url=${encodeURIComponent(url)}`),

  predictManual: (formData: Record<string, unknown>) =>
    fetchJson<ManualPredictionResult>(`${BASE}/predict/manual`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(formData),
    }),
};
```

**Step 3: Write React Query hooks**

```typescript
// frontend/src/hooks/useStats.ts
import { useQuery } from "@tanstack/react-query";
import { api } from "../services/api";

export function useStats() {
  return useQuery({ queryKey: ["stats"], queryFn: api.getStats });
}
```

```typescript
// frontend/src/hooks/useNeighbourhoods.ts
import { useQuery } from "@tanstack/react-query";
import { api } from "../services/api";

export function useNeighbourhoods() {
  return useQuery({ queryKey: ["neighbourhoods"], queryFn: api.getNeighbourhoods });
}

export function useNeighbourhoodDetail(name: string | null) {
  return useQuery({
    queryKey: ["neighbourhood", name],
    queryFn: () => api.getNeighbourhoodDetail(name!),
    enabled: !!name,
  });
}
```

```typescript
// frontend/src/hooks/usePrediction.ts
import { useMutation } from "@tanstack/react-query";
import { api } from "../services/api";

export function useUrlPrediction() {
  return useMutation({ mutationFn: api.predictFromUrl });
}

export function useManualPrediction() {
  return useMutation({ mutationFn: api.predictManual });
}
```

**Step 4: Verify build**

```bash
cd frontend && npm run build
```
Expected: Build succeeds with no type errors

**Step 5: Commit**

```bash
git add frontend/src/types/ frontend/src/services/ frontend/src/hooks/
git commit -m "feat: add TypeScript types, API client, and React Query hooks"
```

---

### Task 8: Price Predictor Page - URL Analysis

**Files:**
- Create: `frontend/src/components/UrlAnalyzer.tsx`
- Create: `frontend/src/components/PredictionResult.tsx`
- Modify: `frontend/src/pages/PredictPage.tsx`

**Step 1: Build UrlAnalyzer component**

```typescript
// frontend/src/components/UrlAnalyzer.tsx
import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { useUrlPrediction } from "../hooks/usePrediction";
import PredictionResultCard from "./PredictionResult";

export default function UrlAnalyzer() {
  const [url, setUrl] = useState("");
  const mutation = useUrlPrediction();

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (url.trim()) mutation.mutate(url.trim());
  };

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle>Check a Property</CardTitle>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="flex gap-3">
            <Input
              type="url"
              placeholder="https://www.immoweb.be/en/classified/house/for-sale/gent/9000/..."
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              className="flex-1"
            />
            <Button type="submit" disabled={mutation.isPending}>
              {mutation.isPending ? "Analyzing..." : "Check Price"}
            </Button>
          </form>
          {mutation.isError && (
            <p className="mt-3 text-sm text-red-600">{mutation.error.message}</p>
          )}
        </CardContent>
      </Card>

      {mutation.data && <PredictionResultCard data={mutation.data} />}
    </div>
  );
}
```

**Step 2: Build PredictionResult component**

```typescript
// frontend/src/components/PredictionResult.tsx
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import type { PredictionResult } from "../types/prediction";

interface Props {
  data: PredictionResult;
}

export default function PredictionResultCard({ data }: Props) {
  const { property, prediction } = data;
  const isUnderpriced = prediction.delta_eur != null && prediction.delta_eur > 0;

  return (
    <Card>
      <CardContent className="pt-6">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* Asking Price */}
          <div className="text-center">
            <p className="text-sm text-muted-foreground">Asking Price</p>
            <p className="text-2xl font-bold">
              {property.price ? `€${property.price.toLocaleString()}` : "N/A"}
            </p>
          </div>

          {/* Predicted Price */}
          <div className="text-center">
            <p className="text-sm text-muted-foreground">Predicted Price</p>
            <p className="text-2xl font-bold">
              €{prediction.predicted_price.toLocaleString()}
            </p>
          </div>

          {/* Delta */}
          <div className="text-center">
            <p className="text-sm text-muted-foreground">Difference</p>
            {prediction.delta_eur != null && prediction.delta_pct != null ? (
              <div>
                <p className={`text-2xl font-bold ${isUnderpriced ? "text-green-600" : "text-red-600"}`}>
                  {prediction.delta_eur > 0 ? "+" : ""}€{prediction.delta_eur.toLocaleString()}
                </p>
                <Badge variant={isUnderpriced ? "default" : "destructive"} className="mt-1">
                  {prediction.delta_pct > 0 ? "+" : ""}{prediction.delta_pct.toFixed(1)}%
                </Badge>
              </div>
            ) : (
              <p className="text-muted-foreground">N/A</p>
            )}
          </div>
        </div>

        {/* Property summary */}
        <div className="mt-6 pt-4 border-t grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
          <div><span className="text-muted-foreground">Address:</span> {property.address}</div>
          <div><span className="text-muted-foreground">Type:</span> {property.property_subtype}</div>
          <div><span className="text-muted-foreground">Area:</span> {property.livable_area} m²</div>
          <div><span className="text-muted-foreground">Bedrooms:</span> {property.bedrooms}</div>
          <div><span className="text-muted-foreground">Year:</span> {property.year_built}</div>
          <div><span className="text-muted-foreground">Condition:</span> {property.condition}</div>
          <div><span className="text-muted-foreground">Energy:</span> {property.energy_label}</div>
          <div><span className="text-muted-foreground">Neighbourhood:</span> {property.neighbourhood}</div>
        </div>
      </CardContent>
    </Card>
  );
}
```

**Step 3: Wire up PredictPage**

```typescript
// frontend/src/pages/PredictPage.tsx
import UrlAnalyzer from "../components/UrlAnalyzer";

export default function PredictPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold">Price Predictor</h1>
        <p className="text-muted-foreground">
          Paste an Immoweb URL to get an instant price prediction
        </p>
      </div>
      <UrlAnalyzer />
    </div>
  );
}
```

**Step 4: Verify build**

```bash
cd frontend && npm run build
```
Expected: Build succeeds

**Step 5: Commit**

```bash
git add frontend/src/
git commit -m "feat: add Price Predictor page with URL analysis"
```

---

### Task 9: Neighbourhood Dashboard - KPI Strip & Map

**Files:**
- Create: `frontend/src/components/KpiStrip.tsx`
- Create: `frontend/src/components/NeighbourhoodMap.tsx`
- Modify: `frontend/src/pages/NeighbourhoodsPage.tsx`

**Step 1: Build KpiStrip component**

```typescript
// frontend/src/components/KpiStrip.tsx
import { Card, CardContent } from "@/components/ui/card";
import type { DashboardStats } from "../types/neighbourhood";

interface Props {
  stats: DashboardStats;
}

const items = [
  { key: "total_properties" as const, label: "Properties", format: (v: number) => v.toLocaleString() },
  { key: "num_neighbourhoods" as const, label: "Neighbourhoods", format: (v: number) => String(v) },
  { key: "median_price_k" as const, label: "Median Price", format: (v: number) => `${v.toFixed(0)}k€` },
  { key: "avg_listing_duration" as const, label: "Avg Duration", format: (v: number) => `${v.toFixed(0)}d` },
  { key: "pct_under_option" as const, label: "Under Option", format: (v: number) => `${v.toFixed(1)}%` },
  { key: "median_delta_pct" as const, label: "Median Delta", format: (v: number) => `${v > 0 ? "+" : ""}${v.toFixed(1)}%` },
];

export default function KpiStrip({ stats }: Props) {
  return (
    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
      {items.map(({ key, label, format }) => (
        <Card key={key}>
          <CardContent className="pt-4 pb-4 text-center">
            <p className="text-2xl font-bold">{format(stats[key])}</p>
            <p className="text-xs text-muted-foreground uppercase tracking-wide mt-1">{label}</p>
          </CardContent>
        </Card>
      ))}
    </div>
  );
}
```

**Step 2: Build NeighbourhoodMap component**

```typescript
// frontend/src/components/NeighbourhoodMap.tsx
import { useState, useMemo } from "react";
import { MapContainer, TileLayer, GeoJSON } from "react-leaflet";
import type { NeighbourhoodSummary } from "../types/neighbourhood";
import "leaflet/dist/leaflet.css";

interface Props {
  neighbourhoods: NeighbourhoodSummary[];
  onSelect: (name: string) => void;
}

type MetricKey = "median_price_sqm" | "avg_listing_duration" | "count" | "pct_underpriced" | "median_delta_pct";

const METRICS: { key: MetricKey; label: string; unit: string }[] = [
  { key: "median_price_sqm", label: "Median Price/m²", unit: "€/m²" },
  { key: "avg_listing_duration", label: "Avg Duration", unit: "days" },
  { key: "count", label: "Listings", unit: "" },
  { key: "pct_underpriced", label: "% Underpriced", unit: "%" },
  { key: "median_delta_pct", label: "Median Delta", unit: "%" },
];

function interpolateColor(value: number, min: number, max: number): string {
  const t = max === min ? 0.5 : (value - min) / (max - min);
  const r = Math.round(255 * (1 - t) + 59 * t);
  const g = Math.round(255 * (1 - t) + 130 * t);
  const b = Math.round(178 * (1 - t) + 246 * t);
  return `rgb(${r},${g},${b})`;
}

export default function NeighbourhoodMap({ neighbourhoods, onSelect }: Props) {
  const [metric, setMetric] = useState<MetricKey>("median_price_sqm");

  const { min, max } = useMemo(() => {
    const values = neighbourhoods
      .map((n) => n[metric])
      .filter((v): v is number => v != null);
    return {
      min: Math.min(...values),
      max: Math.max(...values),
    };
  }, [neighbourhoods, metric]);

  return (
    <div className="space-y-3">
      <div className="flex gap-2 flex-wrap">
        {METRICS.map((m) => (
          <button
            key={m.key}
            onClick={() => setMetric(m.key)}
            className={`px-3 py-1.5 text-sm rounded-md border ${
              metric === m.key
                ? "bg-blue-600 text-white border-blue-600"
                : "bg-white text-slate-700 border-slate-200 hover:border-slate-300"
            }`}
          >
            {m.label}
          </button>
        ))}
      </div>

      <div className="rounded-lg overflow-hidden border" style={{ height: 500 }}>
        <MapContainer
          center={[51.054, 3.725]}
          zoom={13}
          style={{ height: "100%", width: "100%" }}
          scrollWheelZoom={true}
        >
          <TileLayer
            attribution='&copy; <a href="https://carto.com">CARTO</a>'
            url="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
          />
          {neighbourhoods.map((nbh) => {
            if (!nbh.geojson) return null;
            const value = nbh[metric];
            const color = value != null ? interpolateColor(value, min, max) : "#ccc";

            return (
              <GeoJSON
                key={`${nbh.name}-${metric}`}
                data={nbh.geojson as GeoJSON.GeoJsonObject}
                style={() => ({
                  fillColor: color,
                  color: "#333",
                  weight: 1.5,
                  fillOpacity: 0.65,
                })}
                eventHandlers={{
                  click: () => onSelect(nbh.name),
                }}
                onEachFeature={(_feature, layer) => {
                  const unit = METRICS.find((m) => m.key === metric)?.unit ?? "";
                  layer.bindTooltip(
                    `<b>${nbh.name}</b><br/>${value?.toFixed(1) ?? "N/A"} ${unit}`
                  );
                }}
              />
            );
          })}
        </MapContainer>
      </div>
    </div>
  );
}
```

**Step 3: Wire up NeighbourhoodsPage**

```typescript
// frontend/src/pages/NeighbourhoodsPage.tsx
import { useState } from "react";
import KpiStrip from "../components/KpiStrip";
import NeighbourhoodMap from "../components/NeighbourhoodMap";
import { useStats } from "../hooks/useStats";
import { useNeighbourhoods } from "../hooks/useNeighbourhoods";

export default function NeighbourhoodsPage() {
  const [selectedNbh, setSelectedNbh] = useState<string | null>(null);
  const { data: stats, isLoading: statsLoading } = useStats();
  const { data: neighbourhoods, isLoading: nbhLoading } = useNeighbourhoods();

  if (statsLoading || nbhLoading) {
    return <div className="text-muted-foreground">Loading dashboard...</div>;
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold">Neighbourhood Dashboard</h1>
        <p className="text-muted-foreground">Market analytics across Ghent neighbourhoods</p>
      </div>

      {stats && <KpiStrip stats={stats} />}
      {neighbourhoods && (
        <NeighbourhoodMap neighbourhoods={neighbourhoods} onSelect={setSelectedNbh} />
      )}

      {/* ComparisonTable and NeighbourhoodDetail added in next tasks */}
      {selectedNbh && (
        <div className="text-sm text-muted-foreground">
          Selected: {selectedNbh} (detail view coming next)
        </div>
      )}
    </div>
  );
}
```

**Step 4: Verify build**

```bash
cd frontend && npm run build
```
Expected: Build succeeds

**Step 5: Commit**

```bash
git add frontend/src/
git commit -m "feat: add KPI strip and neighbourhood choropleth map"
```

---

### Task 10: Neighbourhood Comparison Table

**Files:**
- Create: `frontend/src/components/ComparisonTable.tsx`
- Modify: `frontend/src/pages/NeighbourhoodsPage.tsx`

**Step 1: Build ComparisonTable**

```typescript
// frontend/src/components/ComparisonTable.tsx
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from "@/components/ui/table";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { NeighbourhoodSummary } from "../types/neighbourhood";

interface Props {
  neighbourhoods: NeighbourhoodSummary[];
  onSelect: (name: string) => void;
}

export default function ComparisonTable({ neighbourhoods, onSelect }: Props) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Neighbourhood Comparison</CardTitle>
      </CardHeader>
      <CardContent>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Neighbourhood</TableHead>
              <TableHead className="text-right">Listings</TableHead>
              <TableHead className="text-right">Median Price (k€)</TableHead>
              <TableHead className="text-right">€/m²</TableHead>
              <TableHead className="text-right">Avg Duration (d)</TableHead>
              <TableHead className="text-right">% Underpriced</TableHead>
              <TableHead className="text-right">Median Delta %</TableHead>
              <TableHead className="text-right">% Under Option</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {neighbourhoods.map((nbh) => (
              <TableRow
                key={nbh.name}
                className="cursor-pointer hover:bg-slate-50"
                onClick={() => onSelect(nbh.name)}
              >
                <TableCell className="font-medium">{nbh.name.replace(/_/g, " ")}</TableCell>
                <TableCell className="text-right">{nbh.count}</TableCell>
                <TableCell className="text-right">{nbh.median_price_k.toFixed(0)}</TableCell>
                <TableCell className="text-right">{nbh.median_price_sqm.toFixed(0)}</TableCell>
                <TableCell className="text-right">{nbh.avg_listing_duration.toFixed(0)}</TableCell>
                <TableCell className="text-right">
                  {nbh.pct_underpriced != null ? `${nbh.pct_underpriced.toFixed(1)}%` : "—"}
                </TableCell>
                <TableCell className="text-right">
                  {nbh.median_delta_pct != null ? (
                    <span className={nbh.median_delta_pct > 0 ? "text-green-600" : "text-red-600"}>
                      {nbh.median_delta_pct > 0 ? "+" : ""}{nbh.median_delta_pct.toFixed(1)}%
                    </span>
                  ) : "—"}
                </TableCell>
                <TableCell className="text-right">{nbh.pct_under_option.toFixed(1)}%</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  );
}
```

**Step 2: Add to NeighbourhoodsPage**

Replace the placeholder comment in `NeighbourhoodsPage` with:

```typescript
import ComparisonTable from "../components/ComparisonTable";

// In the JSX, after the map:
{neighbourhoods && (
  <ComparisonTable neighbourhoods={neighbourhoods} onSelect={setSelectedNbh} />
)}
```

**Step 3: Verify build**

```bash
cd frontend && npm run build
```

**Step 4: Commit**

```bash
git add frontend/src/
git commit -m "feat: add neighbourhood comparison table"
```

---

### Task 11: Neighbourhood Detail Panel

**Files:**
- Create: `frontend/src/components/NeighbourhoodDetail.tsx`
- Create: `frontend/src/components/PriceTrendChart.tsx`
- Modify: `frontend/src/pages/NeighbourhoodsPage.tsx`

**Step 1: Build PriceTrendChart**

```typescript
// frontend/src/components/PriceTrendChart.tsx
import { LineChart, Line, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer } from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

interface TrendPoint {
  month: string;
  city_median_price_k: number | null;
  neighbourhood_median_price_k: number | null;
}

interface Props {
  data: TrendPoint[];
  neighbourhoodName: string;
}

export default function PriceTrendChart({ data, neighbourhoodName }: Props) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">
          Monthly Median Price: {neighbourhoodName.replace(/_/g, " ")} vs City
        </CardTitle>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={350}>
          <LineChart data={data}>
            <XAxis dataKey="month" tick={{ fontSize: 12 }} />
            <YAxis tick={{ fontSize: 12 }} />
            <Tooltip />
            <Legend />
            <Line
              type="monotone"
              dataKey="city_median_price_k"
              name="City-wide"
              stroke="#94a3b8"
              strokeWidth={2}
              dot={false}
              connectNulls
            />
            <Line
              type="monotone"
              dataKey="neighbourhood_median_price_k"
              name={neighbourhoodName.replace(/_/g, " ")}
              stroke="#3b82f6"
              strokeWidth={2.5}
              dot={false}
              connectNulls
            />
          </LineChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}
```

**Step 2: Build NeighbourhoodDetail**

```typescript
// frontend/src/components/NeighbourhoodDetail.tsx
import { Card, CardContent } from "@/components/ui/card";
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from "@/components/ui/table";
import PriceTrendChart from "./PriceTrendChart";
import { useNeighbourhoodDetail } from "../hooks/useNeighbourhoods";

interface Props {
  name: string;
}

export default function NeighbourhoodDetail({ name }: Props) {
  const { data, isLoading, isError } = useNeighbourhoodDetail(name);

  if (isLoading) return <div className="text-muted-foreground">Loading details...</div>;
  if (isError || !data) return <div className="text-red-600">Failed to load details</div>;

  const { stats, trend, listings } = data;
  const displayName = name.replace(/_/g, " ");

  return (
    <div className="space-y-4">
      {/* Stats card */}
      <Card className="bg-gradient-to-br from-slate-900 to-blue-900 text-white">
        <CardContent className="pt-6">
          <h2 className="text-xl font-bold mb-4">{displayName}</h2>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
            {[
              { value: stats.count.toLocaleString(), label: "Properties" },
              { value: `${stats.median_price_k.toFixed(0)}k`, label: "Median Price" },
              { value: `${stats.avg_area.toFixed(0)} m²`, label: "Avg Area" },
              { value: stats.avg_age != null ? `${stats.avg_age.toFixed(0)} yr` : "—", label: "Avg Age" },
              { value: stats.top_condition?.replace(/_/g, " ") ?? "—", label: "Top Condition" },
              { value: stats.top_energy_label ?? "—", label: "Top Energy" },
            ].map(({ value, label }) => (
              <div key={label} className="bg-white/15 rounded-lg p-3 text-center">
                <div className="text-lg font-bold">{value}</div>
                <div className="text-xs text-slate-300 mt-1">{label}</div>
              </div>
            ))}
          </div>
          <div className="mt-3 text-sm text-slate-400">
            Price range: {stats.min_price_k.toFixed(0)}k€ – {stats.max_price_k.toFixed(0)}k€
          </div>
        </CardContent>
      </Card>

      {/* Price trend */}
      {trend.length > 0 && <PriceTrendChart data={trend} neighbourhoodName={name} />}

      {/* Current listings */}
      {listings.length > 0 && (
        <Card>
          <CardContent className="pt-6">
            <h3 className="font-semibold mb-3">Current Listings ({listings.length})</h3>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>ID</TableHead>
                  <TableHead className="text-right">Price (k€)</TableHead>
                  <TableHead className="text-right">Predicted (k€)</TableHead>
                  <TableHead className="text-right">Delta %</TableHead>
                  <TableHead className="text-right">Area (m²)</TableHead>
                  <TableHead className="text-right">Beds</TableHead>
                  <TableHead>Condition</TableHead>
                  <TableHead>Energy</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {listings.map((l) => (
                  <TableRow key={l.property_id}>
                    <TableCell>{l.property_id}</TableCell>
                    <TableCell className="text-right">{l.price?.toFixed(0)}</TableCell>
                    <TableCell className="text-right">{l.predicted_k?.toFixed(0)}</TableCell>
                    <TableCell className="text-right">
                      {l.delta_pct != null ? (
                        <span className={l.delta_pct > 0 ? "text-green-600" : "text-red-600"}>
                          {l.delta_pct > 0 ? "+" : ""}{l.delta_pct.toFixed(1)}%
                        </span>
                      ) : "—"}
                    </TableCell>
                    <TableCell className="text-right">{l.livable_area}</TableCell>
                    <TableCell className="text-right">{l.number_of_bedrooms}</TableCell>
                    <TableCell>{l.building_condition?.replace(/_/g, " ")}</TableCell>
                    <TableCell>{l.energy_label}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
```

**Step 3: Add to NeighbourhoodsPage**

Replace the selectedNbh placeholder with:

```typescript
import NeighbourhoodDetail from "../components/NeighbourhoodDetail";

// In JSX:
{selectedNbh && <NeighbourhoodDetail name={selectedNbh} />}
```

**Step 4: Verify build**

```bash
cd frontend && npm run build
```

**Step 5: Commit**

```bash
git add frontend/src/
git commit -m "feat: add neighbourhood detail panel with price trend chart and listings"
```

---

### Task 12: End-to-End Integration Test

**Files:**
- Modify: `api/app.py` (wire up PredictionService initialization)
- Test: manual E2E test

**Step 1: Wire up full model loading in api/app.py**

Update `create_app` to optionally load models:

```python
def create_app(loader=None, prediction_service=None):
    global data_loader
    app = Flask(__name__)
    CORS(app, origins=["http://localhost:5173"])

    if loader is not None:
        data_loader = loader
    elif data_loader is None:
        print("Initializing DataLoader...")
        data_loader = DataLoader()

    # Prediction service
    if prediction_service is None:
        from api.prediction import PredictionService
        # Try loading models
        ag_predictor = None
        try:
            from autogluon.tabular import TabularPredictor
            import os
            from api.config import AUTOGLUON_PATH
            if os.path.exists(AUTOGLUON_PATH):
                ag_predictor = TabularPredictor.load(AUTOGLUON_PATH)
                print(f"AutoGluon model loaded")
        except Exception as e:
            print(f"AutoGluon not available: {e}")

        prediction_service = PredictionService(
            ag_predictor=ag_predictor,
            data_loader=data_loader,
        )

    # ... (register all endpoints as before, using prediction_service)
```

**Step 2: Run Flask API**

```bash
cd /Users/simon/Desktop/personal/immo_datacollection/house_scraping_project
python -m api.app
```
Expected: Starts on port 5000, prints model loading messages

**Step 3: Run React dev server**

```bash
cd frontend && npm run dev
```
Expected: Starts on port 5173

**Step 4: Manual verification checklist**

- [ ] Visit http://localhost:5173 — nav bar renders with two links
- [ ] `/predict` page shows URL input form
- [ ] `/neighbourhoods` shows KPI strip with real numbers
- [ ] Choropleth map renders with coloured neighbourhood polygons
- [ ] Clicking a polygon shows the detail panel below
- [ ] Comparison table is sortable by clicking rows
- [ ] Price trend chart shows lines for neighbourhood vs city

**Step 5: Commit**

```bash
git add api/
git commit -m "feat: wire up full model loading in Flask API"
```

---

## Summary of Tasks

| # | Task | What it builds |
|---|------|---------------|
| 1 | Flask API Skeleton | Health endpoint, project structure, requirements |
| 2 | Data Loading Module | CSV + polygon loading, stats computation |
| 3 | Stats & Neighbourhood List | `/api/stats`, `/api/neighbourhoods` |
| 4 | Neighbourhood Detail & Today | `/api/neighbourhoods/:name`, `/api/properties/today` |
| 5 | URL Prediction Endpoint | `/api/predict/url`, `/api/predict/manual` |
| 6 | React Project Scaffold | Vite + TS + Tailwind + shadcn + router |
| 7 | Types & API Client | TypeScript types, fetch wrapper, React Query hooks |
| 8 | Price Predictor Page | URL input + prediction result card |
| 9 | KPI Strip & Map | Dashboard KPIs, react-leaflet choropleth |
| 10 | Comparison Table | Sortable neighbourhood data table |
| 11 | Neighbourhood Detail | Stats card, price trend chart, listings table |
| 12 | E2E Integration | Wire up models, manual verification |

## Future Iterations (not in scope)

- Manual prediction form (31-field form with geocoding)
- Dark mode toggle
- WebSocket-based pipeline refresh
- Authentication layer
- Docker Compose for API + frontend
- Pricing speed analysis (scatter + bar charts)
