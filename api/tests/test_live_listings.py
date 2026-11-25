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


def test_deals_endpoint_and_filters():
    import pandas as pd
    from api.live_listings import LiveListingsService

    svc = LiveListingsService.__new__(LiveListingsService)
    svc.prediction_service = None
    svc.date_used = None
    svc.listings_df = pd.DataFrame()  # Force corpus fallback

    mock_loader = MagicMock()
    mock_loader.analytics_df = pd.DataFrame([
        {
            "property_id": "101",
            "date_obtained": "2026-11-10",
            "last_seen_str": "2026-11-23",
            "days_since_last_seen": 0,
            "latest_scrape_date": "2026-11-23",
            "street": "Veldstraat",
            "house_number": "12",
            "title": "Gezellige stadswoning met tuin",
            "description_snippet": "Lichtrijke woning in hartje Gent.",
            "price_eur": 300_000.0,
            "pred_mm_eur": 380_000.0,
            "normal_mask": True,
            "livable_area": 135.0,
            "number_of_bedrooms": 3,
            "property_type_HOUSE": 1,
            "property_type_label": "HOUSE",
            "building_condition": "GOOD",
            "energy_label": "B",
            "neighbourhood": "Binnenstad",
            "neighbourhood_display": "Binnenstad",
            "postal_code": "9000",
            "had_price_cut": True,
        },
        {
            "property_id": "102",
            "date_obtained": "2026-11-12",
            "last_seen_str": "2026-09-20",
            "days_since_last_seen": 64,
            "latest_scrape_date": "2026-11-23",
            "street": "Antwerpsesteenweg",
            "house_number": "5",
            "title": "Appartement nabij Dampoort",
            "description_snippet": "Ruim appartement.",
            "price_eur": 420_000.0,
            "pred_mm_eur": 400_000.0,
            "normal_mask": True,
            "livable_area": 95.0,
            "number_of_bedrooms": 2,
            "property_type_HOUSE": 0,
            "property_type_label": "APARTMENT",
            "building_condition": "AS_NEW",
            "energy_label": "E",
            "neighbourhood": "Dampoort",
            "neighbourhood_display": "Dampoort",
            "postal_code": "9040",
            "had_price_cut": False,
        },
    ])
    svc.data_loader = mock_loader

    # Corpus fallback with filters + availability status + overview fields
    res = svc.get_deals(property_type="HOUSE", min_epc="B", min_delta_pct=10.0, below_band_only=True, available_only=True)
    assert res["source"] == "corpus_recent"
    assert res["total"] == 1
    d0 = res["deals"][0]
    assert d0["property_id"] == "101"
    assert d0["band_position"] == "below_band"
    assert d0["had_price_cut"] is True
    assert d0["availability_status"] == "active_recent"
    assert d0["address"] == "Veldstraat 12, 9000 Gent"
    assert d0["title"] == "Gezellige stadswoning met tuin"

    # Available-only excludes older listings
    res_all = svc.get_deals(available_only=True)
    assert [d["property_id"] for d in res_all["deals"]] == ["101"]

    # Extra local filters: min_condition, min_area, min_bedrooms, has_garden, price_cut_only, search, include_archive
    res_extra = svc.get_deals(
        min_condition="GOOD",
        min_area=120,
        min_bedrooms=3,
        price_cut_only=True,
        search="veldstraat",
        include_archive=True,
    )
    assert res_extra["source"] == "corpus_full"
    assert [d["property_id"] for d in res_extra["deals"]] == ["101"]

    # Route test through Flask app (incl max_predicted + extra filters + dossier + compare)
    app = create_app(loader=mock_loader, pred_service=MagicMock(), listings_service=svc)
    app.config["TESTING"] = True
    client = app.test_client()
    resp = client.get(
        "/api/deals?property_type=HOUSE&min_epc=C&min_condition=GOOD&min_area=100"
        "&min_bedrooms=3&price_cut_only=true&below_band_only=true&available_only=true"
        "&max_predicted=400000&search=Veldstraat"
    )
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["source"] == "corpus_recent"
    assert len(body["deals"]) == 1
    assert body["deals"][0]["property_id"] == "101"
    assert body["deals"][0]["availability_status"] == "active_recent"

    # max_predicted below 380k excludes 101; min_price & min_predicted range filters work
    resp_none = client.get("/api/deals?max_predicted=350000")
    assert resp_none.status_code == 200
    assert len(resp_none.get_json()["deals"]) == 0

    resp_min_ask = client.get("/api/deals?min_price=350000")
    assert resp_min_ask.status_code == 200
    assert [d["property_id"] for d in resp_min_ask.get_json()["deals"]] == ["102"]

    resp_min_pred = client.get("/api/deals?min_predicted=390000")
    assert resp_min_pred.status_code == 200
    assert [d["property_id"] for d in resp_min_pred.get_json()["deals"]] == ["102"]

    resp_band = client.get("/api/deals?min_price=250000&max_price=350000&min_predicted=350000&max_predicted=400000")
    assert resp_band.status_code == 200
    assert [d["property_id"] for d in resp_band.get_json()["deals"]] == ["101"]

    # Dossier & side-by-side compare endpoints
    resp_dos = client.get("/api/properties/101/dossier")
    assert resp_dos.status_code == 200
    dos = resp_dos.get_json()
    assert dos["property_id"] == "101"
    assert dos["valuation"]["predicted_mm_eur"] == 380_000
    assert dos["availability"]["status"] == "active_recent"

    resp_cmp = client.post("/api/properties/compare", json={"property_ids": ["101", "102"]})
    assert resp_cmp.status_code == 200
    cmp_body = resp_cmp.get_json()
    assert cmp_body["count"] == 2
    assert [p["property_id"] for p in cmp_body["properties"]] == ["101", "102"]
