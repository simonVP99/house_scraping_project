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
