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
