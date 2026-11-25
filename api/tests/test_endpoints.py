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
    assert data["neighbourhoods"][0]["median_price_eur"] == 380_000


def test_model_metadata_endpoint(client):
    resp = client.get("/api/model/metadata")
    # Either 200 with metadata when bundle is loaded, or 503 when bundle is absent
    assert resp.status_code in (200, 503)
    if resp.status_code == 200:
        data = resp.get_json()
        assert data.get("agency_blind") is True
        assert "tracks" in data


def test_insights_endpoint(client):
    fake_insights = {
        "generated_from": "phase7-lean5-test",
        "corpus": {"total_properties": 14599, "raw_snapshots": 145384},
        "agency_audit": [],
        "epc_cliff": {},
        "neighbourhood_profiles": [],
    }
    with patch("api.app._load_market_insights", return_value=fake_insights):
        resp = client.get("/api/insights")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["generated_from"] == "phase7-lean5-test"
        assert data["corpus"]["total_properties"] == 14599


def test_affordability_endpoint(mock_loader):
    fake_insights = {
        "neighbourhood_profiles": [
            {
                "name": "Centrum",
                "display_name": "Binnenstad",
                "count": 500,
                "lat": 51.054,
                "lon": 3.722,
                "median_price_sqm": 4200.0,
                "median_dist_center_km": 0.6,
            },
            {
                "name": "Brugse_Poort",
                "display_name": "Brugse Poort",
                "count": 400,
                "lat": 51.058,
                "lon": 3.702,
                "median_price_sqm": 3100.0,
                "median_dist_center_km": 1.8,
            },
        ]
    }
    fake_bundle = MagicMock()
    fake_bundle.metadata = {"model_version": "phase7-lean5-test"}
    # 2 neighbourhoods x (2 conditions [JUST_RENOVATED, AS_NEW] x 2 EPCs [B, A]) = 8 rows
    fake_bundle.predict_raw.return_value = {
        "predicted_eur": np.array([460_000, 480_000, 495_000, 515_000, 340_000, 355_000, 365_000, 380_000], dtype=float),
        "low_eur": np.array([395_000, 410_000, 425_000, 440_000, 290_000, 305_000, 315_000, 325_000], dtype=float),
        "high_eur": np.array([535_000, 560_000, 575_000, 600_000, 398_000, 415_000, 425_000, 445_000], dtype=float),
        "low_90_eur": np.array([375_000, 390_000, 405_000, 420_000, 275_000, 290_000, 298_000, 310_000], dtype=float),
        "high_90_eur": np.array([565_000, 590_000, 605_000, 635_000, 420_000, 438_000, 450_000, 470_000], dtype=float),
        "interval_halfwidth_pct": np.array([15.2] * 8, dtype=float),
        "track_used": ["tabular_text"] * 8,
        "has_photos": [False] * 8,
    }

    with (
        patch("api.app.data_loader", mock_loader),
        patch("api.app._load_market_insights", return_value=fake_insights),
    ):
        app = create_app(loader=mock_loader)
        app.config["TESTING"] = True
        from api import app as api_app_mod
        with patch.object(api_app_mod.prediction_service, "bundle", fake_bundle):
            with app.test_client() as test_client:
                resp = test_client.post(
                    "/api/affordability",
                    json={
                        "budget_eur": 400_000,
                        "property_type": "HOUSE",
                        "livable_area": 140,
                        "condition": "JUST_RENOVATED",
                        "condition_mode": "min",
                        "energy_label": "B",
                        "epc_mode": "min",
                    },
                )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["query"]["budget_eur"] == 400_000
    assert data["query"]["condition"] == "JUST_RENOVATED+"
    assert data["query"]["energy_label"] == "B+"
    assert data["query"]["condition_tiers"] == ["JUST_RENOVATED", "AS_NEW"]
    assert data["query"]["energy_tiers"] == ["B", "A", "A+", "A++"]
    assert len(data["neighbourhoods"]) == 2
    first, second = data["neighbourhoods"][0], data["neighbourhoods"][1]
    assert first["name"] == "Brugse_Poort"
    assert first["prob_affordable"] > second["prob_affordable"]
    assert first["entry_eur"] == 340_000
    assert first["tier"] in ("comfortable", "realistic")
    assert second["tier"] in ("stretch", "out_of_reach")
    assert "geojson" in first


def test_spa_static_serving_and_api_404(mock_loader, tmp_path):
    (tmp_path / "index.html").write_text("<html><body>Ghent SPA</body></html>", encoding="utf-8")
    (tmp_path / "app.js").write_text("console.log('ok');", encoding="utf-8")

    with patch("api.app.data_loader", mock_loader):
        app = create_app(loader=mock_loader)
        app.config["TESTING"] = True
        app.config["FRONTEND_DIST_DIR"] = str(tmp_path)
        with app.test_client() as test_client:
            # SPA deep routes return index.html
            for route in ("/", "/predict", "/deals", "/explore", "/neighbourhoods", "/insights"):
                r = test_client.get(route)
                assert r.status_code == 200
                assert b"Ghent SPA" in r.data
            # Built static asset served directly
            r_js = test_client.get("/app.js")
            assert r_js.status_code == 200
            assert b"console.log" in r_js.data
            # Unknown /api/* routes still return JSON 404 (not index.html)
            r_api = test_client.get("/api/does-not-exist")
            assert r_api.status_code == 404
            assert r_api.get_json()["error"] == "Not found"
