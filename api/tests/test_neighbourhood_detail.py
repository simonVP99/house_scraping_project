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

    def _get_detail(name):
        if name != "Centrum":
            return None
        return {
            "name": "Centrum",
            "stats": {
                "count": 6,
                "median_price_k": 325.0,
                "min_price_k": 300.0,
                "max_price_k": 350.0,
                "avg_area": 113.0,
                "avg_age": 35.0,
                "top_condition": "GOOD",
                "top_energy_label": "C",
            },
            "trend": [
                {"month": "2025-01", "city_median_price_k": 310.0, "neighbourhood_median_price_k": 310.0},
            ],
            "listings": [
                {"property_id": "123", "price": 300, "predicted_k": 320, "delta_pct": 6.7,
                 "livable_area": 100, "number_of_bedrooms": 3, "building_condition": "GOOD", "energy_label": "C"},
                {"property_id": "456", "price": 350, "predicted_k": 330, "delta_pct": -5.7,
                 "livable_area": 120, "number_of_bedrooms": 2, "building_condition": "AS_NEW", "energy_label": "B"},
            ],
        }

    loader.get_neighbourhood_detail.side_effect = _get_detail
    loader.get_today_deals.return_value = [
        {"property_id": "123", "price": 300, "predicted_k": 320, "delta_pct": 6.7,
         "livable_area": 100, "number_of_bedrooms": 3, "building_condition": "GOOD",
         "energy_label": "C", "neighbourhood": "Centrum"},
        {"property_id": "456", "price": 350, "predicted_k": 330, "delta_pct": -5.7,
         "livable_area": 120, "number_of_bedrooms": 2, "building_condition": "AS_NEW",
         "energy_label": "B", "neighbourhood": "Centrum"},
    ]

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
