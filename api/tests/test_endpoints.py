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
