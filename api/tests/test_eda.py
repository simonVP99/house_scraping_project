"""EDA endpoints: contract, section slicing, and 404 handling."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


@pytest.fixture
def eda_client():
    from api import app as app_module
    app = app_module.create_app()
    return app.test_client()


@pytest.fixture
def report_available():
    return os.path.exists(os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        'outputs', 'eda', 'eda_report.json'))


def test_eda_missing_report_returns_404(monkeypatch):
    from api import app as app_module
    monkeypatch.setattr(app_module, 'EDA_REPORT_PATH', '/nonexistent/eda_report.json')
    app_module._eda_cache.update({'path': None, 'mtime': None, 'report': None})
    client = app_module.create_app().test_client()
    res = client.get('/api/eda')
    assert res.status_code == 404
    assert 'error' in res.get_json()


def test_eda_unknown_section_returns_400(eda_client, report_available):
    if not report_available:
        pytest.skip('no EDA report generated')
    res = eda_client.get('/api/eda?section=bogus')
    assert res.status_code == 400
    body = res.get_json()
    assert 'available' in body


def test_eda_full_report_contract(eda_client, report_available):
    if not report_available:
        pytest.skip('no EDA report generated')
    res = eda_client.get('/api/eda')
    assert res.status_code == 200
    body = res.get_json()
    for key in ('dataset', 'fields', 'correlations', 'spatial',
                'temporal', 'missingness', 'duplicates', 'generated_at'):
        assert key in body, key
    assert body['dataset']['rows'] > 0
    # every field profile must carry the keys the UI renders
    for field in body['fields'][:25]:
        for key in ('name', 'kind', 'null_pct', 'non_null', 'is_leaky'):
            assert key in field, (field.get('name'), key)


def test_eda_section_slicing(eda_client, report_available):
    if not report_available:
        pytest.skip('no EDA report generated')
    res = eda_client.get('/api/eda?section=spatial')
    assert res.status_code == 200
    body = res.get_json()
    assert 'spatial' in body and 'fields' not in body
    assert 'generated_at' in body


def test_eda_field_lookup(eda_client, report_available):
    if not report_available:
        pytest.skip('no EDA report generated')
    res = eda_client.get('/api/eda/field/livable_area')
    assert res.status_code == 200
    assert res.get_json()['name'] == 'livable_area'
    assert eda_client.get('/api/eda/field/not_a_field').status_code == 404
