import pytest
from unittest.mock import MagicMock
from api.app import create_app


@pytest.fixture
def mock_listings_service():
    svc = MagicMock()
    svc.date_used = "2026-02-24"
    svc.get_listings.return_value = [
        {
            "property_id": "12345",
            "address": "Korenmarkt 1, 9000 Gent",
            "url": "https://www.immoweb.be/nl/zoekertje/property/te-koop/gent/9000/12345",
            "asking_price": 300000,
            "predicted_price": 350000,
            "delta_eur": 50000,
            "delta_pct": 16.7,
            "livable_area": 120,
            "bedrooms": 3,
            "condition": "GOOD",
            "energy_label": "B",
            "neighbourhood": "Binnenstad",
        },
    ]
    svc.get_count.return_value = {"count": 1, "date_used": "2026-02-24"}
    svc.refresh.return_value = {"count": 1, "date_used": "2026-02-24"}
    return svc


@pytest.fixture
def client(mock_listings_service):
    mock_loader = MagicMock()
    mock_loader.neighbourhood_summary = None
    mock_loader.get_stats.return_value = {}

    mock_pred = MagicMock()

    app = create_app(
        loader=mock_loader,
        pred_service=mock_pred,
        listings_service=mock_listings_service,
    )
    app.config["TESTING"] = True
    return app.test_client()


def test_listings_live(client, mock_listings_service):
    resp = client.get("/api/listings/live")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "listings" in data
    assert len(data["listings"]) == 1
    assert data["listings"][0]["property_id"] == "12345"
    assert data["date_used"] == "2026-02-24"


def test_listings_live_filter(client, mock_listings_service):
    client.get("/api/listings/live?neighbourhood=Binnenstad")
    mock_listings_service.get_listings.assert_called_with("Binnenstad")


def test_listings_count(client):
    resp = client.get("/api/listings/count")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["count"] == 1
    assert data["date_used"] == "2026-02-24"


def test_listings_refresh(client, mock_listings_service):
    resp = client.post("/api/listings/refresh")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["count"] == 1
    mock_listings_service.refresh.assert_called_once()
