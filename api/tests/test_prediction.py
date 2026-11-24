"""Tests for the unified PredictionService, parser regression, and API contract."""
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from api.app import create_app
from api.prediction import PredictionService, format_shap_payload


def _fake_bundle(predicted_eur=380_000.0, low_eur=325_000.0, high_eur=444_000.0):
    bundle = MagicMock()
    bundle.metadata = {"model_version": "phase7-lean5-test"}

    def _predict_raw(raw_df, include_shap=False, **kwargs):
        n = len(raw_df)
        out = {
            "predicted_eur": np.full(n, predicted_eur),
            "low_eur": np.full(n, low_eur),
            "high_eur": np.full(n, high_eur),
            "low_90_eur": np.full(n, low_eur * 0.94),
            "high_90_eur": np.full(n, high_eur * 1.06),
            "interval_halfwidth_pct": np.full(n, 15.2),
            "track_used": np.full(n, "tabular_text", dtype=object),
            "has_photos": np.zeros(n, dtype=bool),
            "feat_df": pd.DataFrame({"neighbourhood": ["Binnenstad"] * n}),
            "deflator": np.ones(n),
        }
        if include_shap and n == 1:
            out["shap_raw"] = {
                "feature_names": ["livable_area", "energy_label", "e5l_pca_0"],
                "shap_log": np.array([0.12, -0.04, 0.01]),
                "shap_eur": np.array([45_000.0, -15_000.0, 3_000.0]),
                "base_value_eur": 347_000.0,
                "lgbmxt_pred_eur": predicted_eur,
                "predicted_eur": predicted_eur,
            }
        return out

    bundle.predict_raw.side_effect = _predict_raw
    return bundle


def test_parse_immoweb_json_extracts_price():
    """parse_immoweb_html delegates to src/parser.py and extracts price + property_id."""
    service = PredictionService(bundle=None, data_loader=None)
    html = (
        '<script>window.classified = {"price":{"mainValue":350000},'
        '"property":{"type":"HOUSE","subtype":"HOUSE",'
        '"location":{"postalCode":"9000","latitude":51.05,"longitude":3.72},'
        '"netHabitableSurface":120,"bedroomCount":3,'
        '"description":"Mooie woning; met tuin {zuidgericht}; top locatie!"}};</script>'
    )
    result = service.parse_immoweb_html(html, "https://www.immoweb.be/nl/zoekertje/huis/te-koop/gent/9000/12345")
    assert result["price"] == 350000
    assert result["property_id"] == "12345"
    assert result["livable_area"] == 120
    # Regression check: embedded `};` in description must NOT truncate JSON parsing (old non-greedy regex bug!)
    assert "zuidgericht" in (result.get("description") or "")


def test_predict_url_invalid():
    """Invalid URL returns 400."""
    loader = MagicMock()
    loader.get_neighbourhood_spatial.return_value = None
    service = PredictionService(bundle=None, data_loader=loader)
    result, status = service.predict_from_url("not-a-url")
    assert status == 400
    assert "error" in result


def test_predict_manual_returns_200_with_conformal_band_and_shap():
    """POST /api/predict/manual no longer 503s and returns 80% conformal band + aligned SHAP."""
    loader = MagicMock()
    loader.neighbourhood_summary = None
    loader.get_stats.return_value = {}
    loader.get_neighbourhood_spatial.return_value = "Binnenstad"
    loader.find_comparables.return_value = None
    loader.get_feature_context.return_value = []

    service = PredictionService(bundle=_fake_bundle(), data_loader=loader)
    app = create_app(loader=loader, pred_service=service, listings_service=MagicMock())
    app.config["TESTING"] = True
    client = app.test_client()

    resp = client.post(
        "/api/predict/manual",
        json={
            "property_type": "HOUSE",
            "livable_area": 140,
            "number_of_bedrooms": 3,
            "latitude": 51.053,
            "longitude": 3.722,
            "postal_code": "9000",
            "price": 310000,
        },
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["predicted_price"] == 380000
    pred = data["prediction"]
    assert pred["low_eur"] < pred["predicted_eur"] < pred["high_eur"]
    assert pred["confidence_level"] == 0.80
    assert pred["band_position"] == "below_band"
    assert pred["track"] == "tabular_text"
    assert data["shap"]["base_value_eur"] == 347000
    assert len(data["shap"]["features"]) >= 2


def test_format_shap_payload_groups_latent_features():
    payload = format_shap_payload({
        "feature_names": ["livable_area", "e5l_pca_0", "tfidf_svd_1"],
        "shap_eur": np.array([25000.0, 2000.0, 1500.0]),
        "base_value_eur": 350000.0,
    })
    assert payload["base_value_eur"] == 350000
    keys = [f["feature_key"] for f in payload["features"]]
    assert "livable_area" in keys
    assert "_other" in keys
    assert "e5l_pca_0" not in keys
