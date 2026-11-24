"""End-to-end production bundle, feature-parity, and conformal invariant tests (Issue #19)."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = os.path.join(PROJECT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

PROD_DIR = os.path.join(PROJECT_DIR, "outputs", "models", "production")
DATASET_PATH = os.path.join(PROJECT_DIR, "outputs", "dataset", "training_dataset.parquet")
RAW_PARQUET = os.path.join(PROJECT_DIR, "outputs", "property_data_full.parquet")
E5_PATH = os.path.join(PROJECT_DIR, "outputs", "models", "workbench", "e5_large_14599.npz")

BUNDLE_READY = os.path.exists(os.path.join(PROD_DIR, "track_tabular_text.joblib"))


@pytest.fixture(scope="module")
def prod_bundle():
    if not BUNDLE_READY:
        pytest.skip("Production bundle not trained yet")
    from production_pipeline import ProductionBundle
    return ProductionBundle.load(PROD_DIR)


@pytest.mark.skipif(not BUNDLE_READY, reason="Production bundle not trained yet (run src/train_production.py)")
class TestProductionBundle:
    def test_metadata_is_agency_blind_and_no_qwen3(self, prod_bundle):
        bundle = prod_bundle
        meta = bundle.metadata
        assert meta.get("agency_blind") is True
        assert meta.get("uses_qwen3") is False
        assert "tabular_text" in meta.get("tracks", {})
        assert "multimodal" in meta.get("tracks", {})
        # Verify no agency or Qwen3 columns leaked into any model spec
        for track in (bundle.tabular_text, bundle.multimodal):
            assert track is not None
            for col in track.tree_spec.cols + track.nn_spec.cols:
                assert "agency" not in col.lower() and "immo_office" not in col.lower()
                assert "qwen" not in col.lower()

    def test_single_row_vs_batch_feature_parity(self, prod_bundle):
        bundle = prod_bundle
        """1-row transform must match batch transform within 1e-4 (zero train/serve skew)."""
        from production_pipeline import prepare_base_dataframe

        df = pd.read_parquet(DATASET_PATH).iloc[:8].copy()
        df["property_id"] = df["property_id"].astype(str)
        e5l = np.load(E5_PATH)["embeddings"][:8].astype(np.float32)

        base_batch, texts_batch = prepare_base_dataframe(df, bundle.pipeline.preproc_artifacts, already_preprocessed=True)
        feat_batch = bundle.pipeline.transform(base_batch, texts_batch, e5l, vis_bundle=None)

        cols = bundle.tabular_text.tree_spec.cols
        for i in range(3):
            base_1, texts_1 = prepare_base_dataframe(
                df.iloc[i:i + 1], bundle.pipeline.preproc_artifacts, already_preprocessed=True
            )
            feat_1 = bundle.pipeline.transform(base_1, texts_1, e5l[i:i + 1], vis_bundle=None)
            for c in cols:
                v1 = feat_1[c].iloc[0]
                vb = feat_batch[c].iloc[i]
                if pd.api.types.is_numeric_dtype(feat_batch[c]):
                    if pd.isna(v1) and pd.isna(vb):
                        continue
                    assert np.isclose(float(v1), float(vb), atol=1e-4, rtol=1e-4), (
                        f"Skew on col {c} row {i}: 1-row={v1} vs batch={vb}"
                    )
                else:
                    assert str(v1) == str(vb), f"Categorical skew on col {c} row {i}: {v1} vs {vb}"

    def test_manual_sparse_entry_invariants_and_shap(self, prod_bundle):
        bundle = prod_bundle
        """Sparse manual form produces finite EUR prediction, valid 80% band, and aligned TreeSHAP."""
        manual_row = pd.DataFrame([{
            "property_type": "HOUSE",
            "property_subtype": "HOUSE",
            "livable_area": 145.0,
            "number_of_bedrooms": 3,
            "year_built": 1955,
            "building_condition": "GOOD",
            "energy_label": "C",
            "latitude": 51.045,
            "longitude": 3.725,
            "postal_code": "9000",
            "city": "Gent",
            "street": "Kortrijksesteenweg",
            "title": "Ruime gezinswoning met tuin in Gent",
            "description": "Lichtrijke rijwoning met 3 slaapkamers, zuidgerichte stadstuin en conforme elektriciteit.",
        }])
        out = bundle.predict_raw(manual_row, include_shap=True)
        pred = float(out["predicted_eur"][0])
        low = float(out["low_eur"][0])
        high = float(out["high_eur"][0])
        hw_pct = float(out["interval_halfwidth_pct"][0])

        assert out["track_used"][0] == "tabular_text"
        assert not out["has_photos"][0]
        assert 150_000 <= pred <= 900_000
        assert low < pred < high
        assert 5.0 <= hw_pct <= 45.0

        shap_raw = out["shap_raw"]
        assert shap_raw is not None
        reconstructed = shap_raw["base_value_eur"] + float(np.sum(shap_raw["shap_eur"]))
        assert np.isclose(reconstructed, pred, rtol=1e-3, atol=50.0)
