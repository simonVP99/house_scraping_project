"""Tests for parser extensions and the validated preprocessing/build_dataset pipeline."""

import os
import sys
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from parser import build_row, parse_epc_reference, recover_amenities_from_text
from preprocessing import (
    PreprocessConfig,
    _ensure_booleans,
    build_dataset,
    clean_raw,
    fit_preprocessing,
    transform,
)


def test_parse_epc_reference():
    assert parse_epc_reference('20230306-0002827179-RES-2') == (2023, 2)
    assert parse_epc_reference('201712120002017539-1') == (2017, 1)
    assert parse_epc_reference(None) == (None, None)
    assert parse_epc_reference('invalid') == (None, None)


def test_recover_amenities_and_metrics_from_text():
    row = {
        'has_garden': None,
        'has_lift': None,
        'has_photovoltaic_panels': None,
        'has_heat_pump': None,
        'energy_label': None,
        'primary_energy_consumption': None,
        'year_built': None,
        'cadastral_income': None,
        'description_nl': (
            'Prachtig appartement zonder lift met zonnepanelen en warmtepomp. '
            'Bouwjaar: 2019. EPC: 95 kWh/m2 (Energielabel A). '
            'Kadastraal inkomen: € 845. Geen tuin.'
        ),
        'description_fr': None,
    }
    recovered = recover_amenities_from_text(row['description_nl'], row)
    assert row['has_garden'] is False
    assert row['has_lift'] is False
    assert row['has_photovoltaic_panels'] is True
    assert row['has_heat_pump'] is True
    assert row['primary_energy_consumption'] == 95.0
    assert row['energy_label'] == 'A'
    assert row['year_built'] == 2019
    assert row['cadastral_income'] == 845
    assert 'has_photovoltaic_panels' in recovered
    assert 'cadastral_income' in recovered


def test_ensure_booleans_preserves_false_and_missing():
    df = pd.DataFrame({
        'has_swimming_pool': [True, False, None, 'False', 'True'],
        'has_meaningful_description': [1, 0, 1, 1, 0],
    })
    out, indicators = _ensure_booleans(df)
    assert out['has_swimming_pool'].tolist() == [True, False, False, False, True]
    assert 'has_swimming_pool_missing' in indicators
    assert out['has_swimming_pool_missing'].tolist() == [0, 0, 1, 0, 0]
    # Engineered 0/1 integer flag is untouched
    assert 'has_meaningful_description_missing' not in indicators
    assert out['has_meaningful_description'].tolist() == [1, 0, 1, 1, 0]


def test_conditional_surfaces_zero_filled_and_clamped():
    raw = pd.DataFrame({
        'property_id': ['1', '2', '3'],
        'date_obtained': ['01012026', '01012026', '01012026'],
        'price': [300000, 400000, 500000],
        'livable_area': [100, 140, 180],
        'property_size': [0, 200, 150],
        'latitude': [51.05, 51.06, 51.04],
        'longitude': [3.72, 3.73, 3.71],
        'property_type': ['APARTMENT', 'HOUSE', 'HOUSE'],
        'has_garden': [None, True, True],
        'garden_surface': [np.nan, 80.0, 250.0],
        'number_of_bedrooms': [3, 2, 4],
        'number_of_rooms': [1, 5, 1],
        'year_built': [2000, np.nan, 2099],
    })
    artifacts = fit_preprocessing(raw)
    transformed = transform(raw, artifacts)

    # Apartment with has_garden=None gets garden_surface=0.0, not median
    assert transformed.loc[0, 'garden_surface'] == 0.0
    assert transformed.loc[0, 'garden_surface_missing'] == 1
    # House where garden_surface (250) > property_size (150) is clamped to 150
    assert transformed.loc[2, 'garden_surface'] == 150.0
    assert transformed.loc[2, 'garden_exceeded_plot'] == 1
    # Room count placeholder 1 (< bedrooms) is never allowed to remain < bedrooms
    assert (transformed['number_of_rooms'] >= transformed['number_of_bedrooms']).all()
    # Future typo year 2099 is treated as missing and imputed to median (2000)
    assert transformed.loc[2, 'year_built'] == 2000.0


def test_cross_snapshot_static_backfill_and_oof_spatial():
    rows = []
    for i in range(10):
        pid = str(100 + i)
        # First snapshot missing cadastral_income; second snapshot has it
        rows.append({
            'property_id': pid,
            'date_obtained': '01012026',
            'publication_date': '2026-01-01T00:00:00.000+0000',
            'last_modification_date': '2026-01-05T00:00:00.000+0000',
            'price': 300000 + i * 15000,
            'price_type': 'residential_sale',
            'livable_area': 100 + i * 5,
            'property_size': 150 + i * 10,
            'latitude': 51.04 + i * 0.002,
            'longitude': 3.71 + i * 0.002,
            'property_type': 'HOUSE' if i % 2 == 0 else 'APARTMENT',
            'property_subtype': 'HOUSE' if i % 2 == 0 else 'APARTMENT',
            'postal_code': '9000',
            'street': f'Teststraat {i}',
            'house_number': str(i),
            'cadastral_income': np.nan,
            'year_built': 1990 + i,
            'number_of_bedrooms': 3,
            'number_of_rooms': 5,
            'description': 'Volledig gerenoveerde woning in Gent met tuin en terras.',
        })
        rows.append({
            **rows[-1],
            'date_obtained': '15012026',
            'cadastral_income': 750 + i * 20,
        })
    raw_df = pd.DataFrame(rows)
    res = build_dataset(raw_df, n_folds=5)
    matrix = res['train_ready']
    assert len(matrix) == 10
    assert res['report'].ok
    # Static backfill recovered cadastral_income from the second snapshot
    assert matrix['cadastral_income'].notna().all()
    assert (matrix['cadastral_income'] >= 750).all()
    # Out-of-fold spatial features exist and have no nulls
    assert 'oof_knn10_median_price_sqm' in matrix.columns
    assert matrix['oof_knn10_median_price_sqm'].notna().all()
    assert res['folds'].attrs['leaked_groups'] == 0


def test_sentinel_cleaning_and_apartment_plot_zero():
    raw = pd.DataFrame({
        'property_id': ['1', '2', '3'],
        'date_obtained': ['01012026', '01012026', '01012026'],
        'price': [250000, 380000, 520000],
        'livable_area': [45, 130, 190],
        'property_size': [np.nan, 0.0, 240.0],
        'latitude': [51.05, 51.06, 51.04],
        'longitude': [3.72, 3.73, 3.71],
        'property_type': ['APARTMENT', 'HOUSE', 'HOUSE'],
        'property_subtype': ['APARTMENT', 'HOUSE', 'HOUSE'],
        'cadastral_income': [0, 820, 450000],  # 0 placeholder & price-in-KI typo
        'year_built': [0, 1985, 2005],         # 0 placeholder
        'number_of_bathrooms': [0, 1, 2],
        'number_of_toilets': [0, 1, 2],
        'energy_label': ['X', 'B', 'A'],
        'bedroom_surface_sum': [900.0, np.nan, 48.0],
    })
    artifacts = fit_preprocessing(raw)
    transformed = transform(raw, artifacts)

    # Apartment plot is 0.0; House with 0 placeholder gets house median (240.0)
    assert transformed.loc[0, 'property_size'] == 0.0
    assert transformed.loc[1, 'property_size'] == 240.0
    # Cadastral income 0 and 450000 typo are replaced by valid median (820.0)
    assert (transformed['cadastral_income'] == 820.0).all()
    assert transformed.loc[0, 'cadastral_income_missing'] == 1
    assert transformed.loc[2, 'cadastral_income_missing'] == 1
    # year_built=0 is treated as missing and imputed to median (1995.0)
    assert transformed.loc[0, 'year_built'] == 1995.0
    assert transformed.loc[0, 'year_built_missing'] == 1
    # bathrooms=0 and toilets=0 are imputed to >=1
    assert (transformed['number_of_bathrooms'] >= 1).all()
    assert (transformed['number_of_toilets'] >= 1).all()
    # Invalid energy label 'X' is nulled
    assert pd.isna(transformed.loc[0, 'energy_label'])
    # Sub-room surface exceeding livable_area (900 > 45) is clamped to 45
    assert transformed.loc[0, 'bedroom_surface_sum'] == 45.0

