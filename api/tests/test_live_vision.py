"""Unit & parity tests for live Immoweb photo extraction and multimodal routing (Issue #22)."""
from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from PIL import Image

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = os.path.join(PROJECT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from live_vision import (
    LiveVisionEncoder,
    aggregate_siglip_base_features,
    build_contact_sheet,
    extract_photo_urls_from_classified,
)
from api.prediction import PredictionService


def test_extract_photo_urls_evenly_spaced():
    classified = {
        "media": {
            "pictures": [
                {"mediumUrl": f"https://img.example.com/{i}.jpg"} for i in range(12)
            ]
        }
    }
    urls = extract_photo_urls_from_classified(classified, max_pics=6)
    assert len(urls) == 6
    assert urls[0] == "https://img.example.com/0.jpg"
    assert urls[-1] == "https://img.example.com/11.jpg"


def test_extract_photo_urls_handles_missing_and_fallback_large():
    assert extract_photo_urls_from_classified(None) == []
    assert extract_photo_urls_from_classified({"media": {}}) == []
    urls = extract_photo_urls_from_classified({
        "media": {"pictures": [{"largeUrl": "https://img.example.com/big.jpg"}, {"bad": 1}]}
    })
    assert urls == ["https://img.example.com/big.jpg"]


def test_aggregate_siglip_base_features_shapes_and_real_filtering():
    rng = np.random.default_rng(7)
    sp = rng.normal(size=(6, 768)).astype(np.float32)
    sp /= np.linalg.norm(sp, axis=1, keepdims=True)
    media_txt = rng.normal(size=(5, 768)).astype(np.float32)
    room_txt = rng.normal(size=(7, 768)).astype(np.float32)
    q_pos = rng.normal(size=(10, 768)).astype(np.float32)
    q_neg = rng.normal(size=(10, 768)).astype(np.float32)

    sig_real, render_feats, q_feats, rm, is_plan, is_cgi = aggregate_siglip_base_features(
        sp, media_txt, room_txt, q_pos, q_neg
    )
    assert sig_real.shape == (768,)
    assert render_feats.shape == (8,)
    assert q_feats.shape == (30,)
    assert rm.shape == (6,)
    assert np.isfinite(sig_real).all()
    assert np.isfinite(render_feats).all()
    assert np.isfinite(q_feats).all()

    imgs = [Image.new("RGB", (120, 90), (i * 30, 100, 150)) for i in range(6)]
    sheet = build_contact_sheet(imgs, rm, is_plan, is_cgi)
    assert sheet.size == (600, 400)
    assert sheet.mode == "RGB"


def test_live_vision_disk_cache_roundtrip(tmp_path):
    enc = LiveVisionEncoder(cache_dir=str(tmp_path), offline_npz_path="", enabled=True)
    fake_bundle = {
        "sig_real": np.ones((1, 768), dtype=np.float32) * 0.25,
        "so_sheet": np.ones((1, 1152), dtype=np.float32) * -0.1,
        "render_feats": np.arange(8, dtype=np.float32).reshape(1, 8) / 10.0,
        "q_feats": np.arange(30, dtype=np.float32).reshape(1, 30) / 30.0,
        "so_q_feats": np.arange(10, dtype=np.float32).reshape(1, 10) / 10.0,
        "has_vis": np.array([True], dtype=bool),
        "n_photos_used": 6,
        "source": "live_encoded",
    }
    enc._save_cache("12345678", fake_bundle)
    loaded = enc._lookup_cached("12345678")
    assert loaded is not None
    assert loaded["source"] == "live_cache"
    assert loaded["n_photos_used"] == 6
    assert loaded["sig_real"].shape == (1, 768)
    assert loaded["so_sheet"].shape == (1, 1152)
    assert np.allclose(loaded["sig_real"], fake_bundle["sig_real"], atol=1e-3)
    assert np.allclose(loaded["render_feats"], fake_bundle["render_feats"], atol=1e-5)


def test_predict_from_url_uses_multimodal_and_falls_back_gracefully():
    sample_html = (
        '<script>window.classified = {"id": 99988877, "price": {"mainValue": 385000}, '
        '"property": {"type": "HOUSE", "subtype": "HOUSE", "netHabitableSurface": 140, '
        '"bedroomCount": 3, "location": {"postalCode": "9000", "locality": "Gent", '
        '"latitude": 51.05, "longitude": 3.72}}, '
        '"media": {"pictures": [{"mediumUrl": "https://img.example.com/1.jpg"}]}};</script>'
    )
    fake_vis = {
        "sig_real": np.zeros((1, 768), dtype=np.float32),
        "so_sheet": np.zeros((1, 1152), dtype=np.float32),
        "render_feats": np.zeros((1, 8), dtype=np.float32),
        "q_feats": np.zeros((1, 30), dtype=np.float32),
        "so_q_feats": np.zeros((1, 10), dtype=np.float32),
        "has_vis": np.array([True], dtype=bool),
        "n_photos_used": 1,
        "source": "live_encoded",
    }

    fake_bundle = MagicMock()
    fake_bundle.metadata = {"model_version": "phase7-lean5-test"}
    fake_bundle.multimodal = MagicMock()
    fake_bundle.tabular_text = MagicMock()
    fake_bundle.tabular_text.predict.return_value = {"predicted_eur": np.array([395_000.0])}
    fake_bundle.predict_raw.return_value = {
        "predicted_eur": np.array([410_000.0]),
        "low_eur": np.array([350_000.0]),
        "high_eur": np.array([480_000.0]),
        "low_90_eur": np.array([330_000.0]),
        "high_90_eur": np.array([505_000.0]),
        "interval_halfwidth_pct": np.array([14.5]),
        "track_used": np.array(["multimodal"], dtype=object),
        "has_photos": np.array([True]),
        "feat_df": pd.DataFrame([{"neighbourhood": "Binnenstad"}]),
        "deflator": np.array([1.037]),
        "shap_raw": None,
    }

    fake_vis_enc = MagicMock()
    fake_vis_enc.extract_for_listing.return_value = fake_vis

    svc = PredictionService(bundle=fake_bundle, data_loader=None, vision_encoder=fake_vis_enc)
    svc._fetch_page = MagicMock(return_value=sample_html)
    result, status = svc.predict_from_url("https://www.immoweb.be/en/classified/house/for-sale/gent/9000/99988877")
    assert status == 200
    assert result["prediction"]["track"] == "multimodal"
    assert result["prediction"]["has_photos"] is True
    assert result["prediction"]["n_photos_used"] == 1
    assert result["prediction"]["vision_shift_eur"] == 15_000
    assert result["prediction"]["vision_shift_pct"] == 3.8
    assert "_photo_urls" not in result["form_data"]

    # Now simulate vision encoder throwing -> must fall back cleanly to tabular_text without failing
    fake_vis_enc.extract_for_listing.side_effect = RuntimeError("CDN timeout")
    fake_bundle.predict_raw.return_value["track_used"] = np.array(["tabular_text"], dtype=object)
    fake_bundle.predict_raw.return_value["has_photos"] = np.array([False])
    result2, status2 = svc.predict_from_url("https://www.immoweb.be/en/classified/house/for-sale/gent/9000/99988877")
    assert status2 == 200
    assert result2["prediction"]["track"] == "tabular_text"
    assert result2["prediction"]["has_photos"] is False
    assert result2["prediction"]["vision_shift_eur"] is None


PROD_DIR = os.path.join(PROJECT_DIR, "outputs", "models", "production")
OFFLINE_NPZ = os.path.join(PROJECT_DIR, "external_data", "vision_full", "vision_full_embeddings.npz")
DATASET_PATH = os.path.join(PROJECT_DIR, "outputs", "dataset", "training_dataset.parquet")


@pytest.mark.skipif(
    not (os.path.exists(os.path.join(PROD_DIR, "track_multimodal.joblib")) and os.path.exists(OFFLINE_NPZ)),
    reason="Production multimodal bundle or offline vision cache not present",
)
def test_multimodal_bundle_scores_corpus_cached_listing_end_to_end():
    from production_pipeline import ProductionBundle

    bundle = ProductionBundle.load(PROD_DIR)
    enc = LiveVisionEncoder(cache_dir="", offline_npz_path=OFFLINE_NPZ, enabled=True)
    enc._ensure_offline_index()
    df = pd.read_parquet(DATASET_PATH)
    # Pick first listing that has valid offline vision embeddings
    chosen_row = None
    vis = None
    for i in range(min(50, len(df))):
        pid = str(df["property_id"].iloc[i])
        cand = enc._lookup_cached(pid)
        if cand is not None:
            chosen_row = df.iloc[i : i + 1].copy()
            vis = cand
            break
    assert chosen_row is not None and vis is not None
    out = bundle.predict_raw(chosen_row, vis_bundle=vis, already_preprocessed=True, include_shap=True)
    assert out["track_used"][0] == "multimodal"
    assert bool(out["has_photos"][0]) is True
    assert 50_000 <= float(out["predicted_eur"][0]) <= 2_500_000
    assert float(out["low_eur"][0]) < float(out["predicted_eur"][0]) < float(out["high_eur"][0])


def test_jepa_inspector_and_regime_detector():
    from jepa_inspector import (
        CrossModalJEPAInspector,
        DEFAULT_JEPA_PATH,
        detect_data_caveats,
        detect_property_regime,
    )

    r_mixed = detect_property_regime(
        property_subtype="HOUSE",
        title="Handelshuis met winkelruimte op het gelijkvloers",
    )
    assert r_mixed["code"] == "mixed_commercial"
    assert r_mixed["is_non_standard"] is True
    assert r_mixed["advisory"]

    r_inv = detect_property_regime(
        property_subtype="HOUSE",
        description_nl="Ideaal opbrengsteigendom met 4 vergunde studentenkamers en hoog rendement",
    )
    assert r_inv["code"] == "multi_unit_investment"
    assert r_inv["is_non_standard"] is True

    r_boat = detect_property_regime(
        property_subtype="HOUSE",
        description_nl="Deze ruime woonboot van ca. 150 m² op het water in Gent",
    )
    assert r_boat["code"] == "houseboat"
    assert r_boat["is_non_standard"] is True

    r_res = detect_property_regime(
        property_subtype="HOUSE",
        title="Gezellige rijwoning met zonnige stadstuin",
    )
    assert r_res["code"] == "residential"
    assert r_res["is_non_standard"] is False
    assert r_res["advisory"] is None

    # Render-on-fixer + condition-mislabel + text-under-option caveats (e.g. Maaltebruggestraat 81)
    cavs = detect_data_caveats(
        building_condition="FIRST_TIME_USED",
        energy_label="E",
        description_nl="Een te renoveren stadswoning met een vergund ontwerp door architect. OPTIE - BOD AANVAARD.",
        pct_cgi=0.167,
        pct_vstage=0.50,
        pct_any_render=0.50,
    )
    codes = {c["code"] for c in cavs}
    assert "renders_on_fixer" in codes
    assert "condition_mislabel" in codes
    assert "text_under_option" in codes

    if os.path.exists(DEFAULT_JEPA_PATH) and os.path.exists(OFFLINE_NPZ):
        insp = CrossModalJEPAInspector.load(DEFAULT_JEPA_PATH)
        assert insp is not None
        enc = LiveVisionEncoder(cache_dir="", offline_npz_path=OFFLINE_NPZ, enabled=True)
        enc._ensure_offline_index()
        df = pd.read_parquet(DATASET_PATH)
        pid = next(str(p) for p in df["property_id"].head(50) if enc._lookup_cached(str(p)) is not None)
        vis = enc._lookup_cached(pid)
        diag = insp.score_single(
            feat_df=pd.DataFrame([{"livable_area": 140.0, "epc_ordinal": 2.0}]),
            e5_vec=None,
            vis_bundle=vis,
            claimed_condition="AS_NEW",
            delta_pct=8.0,
        )
        assert diag is not None
        assert diag["verdict"] in {
            "renders_detected", "overclaimed_condition", "hidden_gem", "exceeds_specs",
            "high_contradiction", "cherry_picked", "better_than_fixer", "consistent",
        }
        assert 0.0 <= diag["quality_percentile"] <= 100.0
        assert 0.0 <= diag["energy_percentile"] <= 100.0
        assert len(diag["sub_scores"]) == 7
