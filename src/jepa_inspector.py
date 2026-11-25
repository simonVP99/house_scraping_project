"""Production Cross-Modal JEPA Photo-vs-Specs Inspector (Issue #10).

A lightweight (~1.5 MB, <5ms inference) cross-modal latent-residual predictor fit on
the Ghent corpus that scores ANY listing with photos along four interpretable buyer-facing
dimensions:

  1. Visual Quality Surprise (Δz_quality):
       Do the photos look better or worse than what a typical Ghent home with these
       exact specs (location, area, age, EPC, claimed condition, description) looks like?
       Decomposed into room/finish sub-axes: modern_finish, fixer_shell, kitchen, bath,
       natural_light, architect_charm, facade.
  2. Symmetric Cross-Modal Contradiction Energy (E_sym = 0.5*(E_photo + E_text)):
       How strongly do photos and metadata/description contradict each other?
       Empirically stratifies Ghent valuation error (6.7% -> 7.6% MdAPE), price-cut rate
       (1.8% -> 3.0%), and time on market (19.5d -> 29.4d) even within fairly-priced homes.
  3. Intra-Listing Cherry-Picking Spread:
       Gap between the best-looking photo finish (p75) and the worst room shown (min finish /
       max fixer), flagging listings where an agent leads with one renovated room while hiding
       untouched rooms.
  4. Property Regime Tag (Residential / Mixed-Use Commercial / Investment / Multi-Unit):
       Explicitly detects live-work / commercial-ground-floor (`handelspand`, `praktijkruimte`)
       and investment / multi-unit (`opbrengsteigendom`, `APARTMENT_BLOCK`, `KOT`) listings
       whose valuation error is systematically wider (~8.7%–14.4% MdAPE vs 7.2% pure residential).

Build artifact (~2s CPU):
    PYTHONPATH=src .venv-api/bin/python -u src/jepa_inspector.py
"""
from __future__ import annotations

import os
import re
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROD_DIR = os.path.join(PROJECT_DIR, "outputs", "models", "production")
STAGING_DIR = os.path.join(PROJECT_DIR, "outputs", "models", "workbench", "prod_train_staging")
WORK_DIR = os.path.join(PROJECT_DIR, "outputs", "models", "workbench")
VISION_NPZ = os.path.join(PROJECT_DIR, "external_data", "vision_full", "vision_full_embeddings.npz")
E5_NPZ = os.path.join(WORK_DIR, "e5_large_14599.npz")
JEPA_OOF_PARQUET = os.path.join(WORK_DIR, "issue10_jepa_oof.parquet")
DEFAULT_JEPA_PATH = os.path.join(PROD_DIR, "jepa_inspector.joblib")

# ---- Mixed-use / commercial / investment regime detectors ----
MIXED_COMMERCIAL_RE = re.compile(
    r"handelspand|handelsgelijkvloers|handelsruimte|winkelruimte|winkelpand|"
    r"horeca|handelshuis|zaak met woonst|commercieel gelijkvloers|"
    r"espace commercial|commercial space|fond de commerce",
    re.IGNORECASE,
)
LIVE_WORK_RE = re.compile(
    r"praktijkruimte|vrij beroep|woon[- ]?werk|live[ -]?work|kantoorruimte|atelier",
    re.IGNORECASE,
)
MULTI_UNIT_INVEST_RE = re.compile(
    r"opbrengsteigendom|investeringseigendom|investeringspand|meergezinswoning|"
    r"appartementsgebouw|studentenhuis|studentenkamers",
    re.IGNORECASE,
)
RENTED_YIELD_RE = re.compile(
    r"momenteel verhuurd|verhuurd aan|huuropbrengst|brutorendement|nettorendement",
    re.IGNORECASE,
)


def detect_property_regime(
    *,
    property_subtype: str | None = None,
    title: str | None = None,
    description: str | None = None,
    description_nl: str | None = None,
) -> dict[str, Any]:
    """Detect whether a listing is pure residential vs mixed-use commercial / investment.

    Returns a structured dict with `code`, `label`, `is_non_standard`, and `advisory`.
    """
    sub = str(property_subtype or "").upper().strip()
    text = f"{title or ''} {description_nl or ''} {description or ''}"

    flags: list[str] = []
    if sub == "MIXED_USE_BUILDING" or MIXED_COMMERCIAL_RE.search(text):
        flags.append("mixed_commercial")
    if LIVE_WORK_RE.search(text):
        flags.append("live_work")
    if sub in ("APARTMENT_BLOCK", "EXCEPTIONAL_PROPERTY") or MULTI_UNIT_INVEST_RE.search(text):
        flags.append("multi_unit_investment")
    if sub == "KOT":
        flags.append("student_kot")
    elif sub == "SERVICE_FLAT":
        flags.append("service_flat")
    elif RENTED_YIELD_RE.search(text):
        flags.append("rented_yield")

    if "mixed_commercial" in flags:
        return {
            "code": "mixed_commercial",
            "label": "Mixed-use / commercial space",
            "flags": flags,
            "is_non_standard": True,
            "expected_mdape_pct": 8.7,
            "advisory": (
                "Commercial / store space detected. Residential comps only capture part of "
                "commercial footfall & rental value — treat fair value as a residential baseline "
                "and check commercial yield separately."
            ),
        }
    if "multi_unit_investment" in flags:
        return {
            "code": "multi_unit_investment",
            "label": "Multi-unit / investment building",
            "flags": flags,
            "is_non_standard": True,
            "expected_mdape_pct": 9.5,
            "advisory": (
                "Multi-unit / opbrengsteigendom detected. Valuation depends strongly on net "
                "rental yield, unit count, and separate meters — expect wider variance (~9.5–14% MdAPE)."
            ),
        }
    if "live_work" in flags:
        return {
            "code": "live_work",
            "label": "Live-work / practice space",
            "flags": flags,
            "is_non_standard": True,
            "expected_mdape_pct": 8.5,
            "advisory": (
                "Practice / live-work space detected (praktijkruimte / vrij beroep). "
                "Value includes business-use flexibility beyond standard residential comparables."
            ),
        }
    if "student_kot" in flags or "service_flat" in flags:
        return {
            "code": "student_kot" if "student_kot" in flags else "service_flat",
            "label": "Student kot" if "student_kot" in flags else "Assisted living / service flat",
            "flags": flags,
            "is_non_standard": True,
            "expected_mdape_pct": 13.4 if "service_flat" in flags else 7.5,
            "advisory": (
                "Specialized residential segment — check monthly syndic/service fees or "
                "university-rental rules alongside the fair-value estimate."
            ),
        }
    if "rented_yield" in flags:
        return {
            "code": "rented_yield",
            "label": "Currently rented / yield listing",
            "flags": flags,
            "is_non_standard": False,
            "expected_mdape_pct": 7.9,
            "advisory": (
                "Currently rented / investor pitch detected. Verify existing lease terms, "
                "indexation, and notice period if buying for own occupancy."
            ),
        }
    return {
        "code": "residential",
        "label": "Standard residential",
        "flags": [],
        "is_non_standard": False,
        "expected_mdape_pct": 7.2,
        "advisory": None,
    }


def _pct_rank(val: float, sorted_ref: np.ndarray) -> float:
    if len(sorted_ref) == 0 or not np.isfinite(val):
        return 50.0
    idx = int(np.searchsorted(sorted_ref, val, side="right"))
    return round(float(idx) / float(len(sorted_ref)) * 100.0, 1)


class CrossModalJEPAInspector:
    """Lightweight Cross-Modal JEPA photo-vs-specs consistency & honesty scorer."""

    def __init__(
        self,
        tab_cols: list[str],
        tab_medians: np.ndarray,
        tab_scaler: StandardScaler,
        e5_pca: PCA,
        e5_scaler: StandardScaler,
        sig_pca: PCA,
        so_pca: PCA,
        z_scaler: StandardScaler,
        q_scaler: StandardScaler,
        f_fwd: RidgeCV,
        f_q: RidgeCV,
        f_rev: RidgeCV,
        ref_quantiles: dict[str, np.ndarray],
        metadata: dict[str, Any],
    ):
        self.tab_cols = tab_cols
        self.tab_medians = tab_medians
        self.tab_scaler = tab_scaler
        self.e5_pca = e5_pca
        self.e5_scaler = e5_scaler
        self.sig_pca = sig_pca
        self.so_pca = so_pca
        self.z_scaler = z_scaler
        self.q_scaler = q_scaler
        self.f_fwd = f_fwd
        self.f_q = f_q
        self.f_rev = f_rev
        self.ref_quantiles = ref_quantiles
        self.metadata = metadata

    def _transform_tab(self, feat_df: pd.DataFrame) -> np.ndarray:
        n = len(feat_df)
        mat = np.empty((n, len(self.tab_cols)), dtype=np.float32)
        for j, c in enumerate(self.tab_cols):
            if c in feat_df.columns:
                vals = pd.to_numeric(feat_df[c], errors="coerce").to_numpy(dtype=np.float32)
            else:
                vals = np.full(n, np.nan, dtype=np.float32)
            bad = ~np.isfinite(vals)
            if bad.any():
                vals[bad] = self.tab_medians[j]
            mat[:, j] = vals
        X = self.tab_scaler.transform(mat).astype(np.float32)
        np.clip(X, -8.0, 8.0, out=X)
        return X

    def score_single(
        self,
        feat_df: pd.DataFrame,
        e5_vec: np.ndarray | None,
        vis_bundle: dict[str, Any] | None,
        *,
        claimed_condition: str | None = None,
        delta_pct: float | None = None,
    ) -> dict[str, Any] | None:
        """Compute full Cross-Modal JEPA diagnostics for a single listing with photos."""
        if vis_bundle is None or feat_df is None or len(feat_df) == 0:
            return None
        sig = np.asarray(vis_bundle.get("sig_real"), dtype=np.float32).reshape(1, -1)
        so = np.asarray(vis_bundle.get("so_sheet"), dtype=np.float32).reshape(1, -1)
        q_raw = np.asarray(vis_bundle.get("q_feats"), dtype=np.float32).reshape(1, -1)
        so_q = np.asarray(vis_bundle.get("so_q_feats"), dtype=np.float32).reshape(1, -1)
        if sig.shape[1] != 768 or so.shape[1] != 1152 or not np.isfinite(sig).all() or np.linalg.norm(sig) < 1e-4:
            return None

        C = self._transform_tab(feat_df.iloc[:1])
        if e5_vec is None or not np.isfinite(e5_vec).all():
            E5_s = np.zeros((1, self.e5_pca.n_components_), dtype=np.float32)
        else:
            e5_arr = np.asarray(e5_vec, dtype=np.float32).reshape(1, -1)
            E5_s = self.e5_scaler.transform(self.e5_pca.transform(e5_arr)).astype(np.float32)
        Ctx = np.hstack([C, E5_s])

        Z_ph = np.hstack([self.sig_pca.transform(sig), self.so_pca.transform(so)]).astype(np.float32)
        Z_ph_s = self.z_scaler.transform(Z_ph).astype(np.float32)
        Zhat_ph = self.f_fwd.predict(Ctx).astype(np.float32)
        dZ_ph = Z_ph_s - Zhat_ph

        Q = np.hstack([q_raw[:, :10], so_q[:, :10]]).astype(np.float32)
        Q_s = self.q_scaler.transform(Q).astype(np.float32)
        dQ = Q_s - self.f_q.predict(Ctx).astype(np.float32)

        Rev = np.hstack([C, Z_ph_s])
        dE5 = E5_s - self.f_rev.predict(Rev).astype(np.float32)

        e_photo = float(np.sqrt(np.mean(dZ_ph ** 2)))
        e_text = float(np.sqrt(np.mean(dE5 ** 2)))
        e_sym = 0.5 * (e_photo + e_text)

        dz_modern = float(0.5 * (dQ[0, 0] + dQ[0, 10]))
        dz_fixer = float(0.5 * (dQ[0, 1] + dQ[0, 11]))
        dz_light = float(0.5 * (dQ[0, 2] + dQ[0, 12]))
        dz_kitchen = float(0.5 * (dQ[0, 3] + dQ[0, 13]))
        dz_bath = float(0.5 * (dQ[0, 4] + dQ[0, 14]))
        dz_charm = float(0.5 * (dQ[0, 5] + dQ[0, 15]))
        dz_facade = float(0.5 * (dQ[0, 6] + dQ[0, 16]))

        dz_quality = (
            0.30 * dz_modern
            - 0.30 * dz_fixer
            + 0.15 * dz_kitchen
            + 0.10 * dz_bath
            + 0.08 * dz_light
            + 0.07 * dz_facade
        )

        cherry_spread = float(q_raw[0, 10] - q_raw[0, 27]) if q_raw.shape[1] >= 28 else 0.0
        worst_fixer_gap = float(q_raw[0, 26] - q_raw[0, 1]) if q_raw.shape[1] >= 27 else 0.0

        quality_pct = _pct_rank(dz_quality, self.ref_quantiles["dz_quality"])
        energy_pct = _pct_rank(e_sym, self.ref_quantiles["energy_sym"])
        cherry_pct = _pct_rank(cherry_spread, self.ref_quantiles["cherry_spread"])

        # Clear, honest buyer verdict
        cond_upper = str(claimed_condition or "").upper()
        claimed_high = cond_upper in ("AS_NEW", "JUST_RENOVATED")
        claimed_low = cond_upper in ("TO_RENOVATE", "TO_BE_DONE_UP", "TO_RESTORE")

        if claimed_high and quality_pct <= 20.0:
            verdict = "overclaimed_condition"
            verdict_label = "Photos look dated vs claimed condition"
            tone = "negative"
            summary = (
                f"Listed as {cond_upper.replace('_', ' ').title()}, yet visual quality sits in the "
                f"bottom {max(1, int(round(quality_pct)))}% relative to Ghent peers with identical specs. "
                "Inspect finishes closely — historically 2× higher price-cut rate."
            )
        elif quality_pct >= 80.0 and (delta_pct is not None and delta_pct >= 5.0):
            verdict = "hidden_gem"
            verdict_label = "Visual hidden gem"
            tone = "positive"
            summary = (
                f"Photos outperform specs (top {max(1, int(round(100 - quality_pct)))}% visual quality surprise) "
                "while asking sits below Phase-7 fair value."
            )
        elif quality_pct >= 78.0:
            verdict = "exceeds_specs"
            verdict_label = "Photos exceed specs"
            tone = "positive"
            summary = (
                f"Interior/exterior presentation ranks in the top {max(1, int(round(100 - quality_pct)))}% "
                "compared to Ghent homes with the same age, EPC, and condition label."
            )
        elif energy_pct >= 85.0:
            verdict = "high_contradiction"
            verdict_label = "High photo-vs-specs mismatch"
            tone = "warning"
            summary = (
                f"Cross-modal contradiction sits at the {int(round(energy_pct))}th percentile — "
                "photos diverge noticeably from the metadata/text (unusual layout, staging, or mixed use). "
                "Expect wider valuation uncertainty."
            )
        elif cherry_pct >= 85.0:
            verdict = "cherry_picked"
            verdict_label = "Uneven room quality"
            tone = "warning"
            summary = (
                f"High room-to-room quality dispersion ({int(round(cherry_pct))}th pct): some photos look "
                "renovated while others appear untouched. Check unphotographed rooms during viewing."
            )
        elif claimed_low and quality_pct >= 70.0:
            verdict = "better_than_fixer"
            verdict_label = "Cleaner than typical fixer-upper"
            tone = "positive"
            summary = (
                "Photos look notably better maintained than typical Ghent listings in this condition bracket."
            )
        else:
            verdict = "consistent"
            verdict_label = "Photos match specs"
            tone = "neutral"
            summary = (
                "Visual presentation aligns well with the declared condition, EPC, age, and description."
            )

        sub_scores = [
            {"axis": "Modern finish", "z": round(dz_modern, 2)},
            {"axis": "Kitchen", "z": round(dz_kitchen, 2)},
            {"axis": "Bathroom", "z": round(dz_bath, 2)},
            {"axis": "Natural light", "z": round(dz_light, 2)},
            {"axis": "Facade", "z": round(dz_facade, 2)},
            {"axis": "Arch. charm", "z": round(dz_charm, 2)},
            {"axis": "Fixer signals", "z": round(-dz_fixer, 2)},  # flipped so + = cleaner than expected
        ]

        return {
            "verdict": verdict,
            "verdict_label": verdict_label,
            "tone": tone,
            "summary": summary,
            "dz_quality": round(dz_quality, 3),
            "quality_percentile": quality_pct,
            "energy_sym": round(e_sym, 3),
            "energy_photo": round(e_photo, 3),
            "energy_text": round(e_text, 3),
            "energy_percentile": energy_pct,
            "cherry_pick_spread": round(cherry_spread, 4),
            "cherry_percentile": cherry_pct,
            "worst_fixer_gap": round(worst_fixer_gap, 4),
            "sub_scores": sub_scores,
        }

    def save(self, path: str = DEFAULT_JEPA_PATH) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Save as a plain state dict so joblib.load works regardless of __main__ / module path
        payload = {
            "tab_cols": self.tab_cols,
            "tab_medians": self.tab_medians,
            "tab_scaler": self.tab_scaler,
            "e5_pca": self.e5_pca,
            "e5_scaler": self.e5_scaler,
            "sig_pca": self.sig_pca,
            "so_pca": self.so_pca,
            "z_scaler": self.z_scaler,
            "q_scaler": self.q_scaler,
            "f_fwd": self.f_fwd,
            "f_q": self.f_q,
            "f_rev": self.f_rev,
            "ref_quantiles": self.ref_quantiles,
            "metadata": self.metadata,
        }
        joblib.dump(payload, path, compress=3)

    @classmethod
    def load(cls, path: str = DEFAULT_JEPA_PATH) -> "CrossModalJEPAInspector | None":
        if not os.path.exists(path):
            return None
        obj = joblib.load(path)
        if isinstance(obj, cls):
            return obj
        return cls(**obj)


def fit_and_save_jepa_inspector(out_path: str = DEFAULT_JEPA_PATH) -> CrossModalJEPAInspector:
    """Fit the full-corpus Cross-Modal JEPA Inspector from staging artifacts (~2s CPU)."""
    t0 = time.time()
    splits_obj = joblib.load(os.path.join(STAGING_DIR, "prod_splits.joblib"))
    full_df = splits_obj["full_feat_df"]
    tt_cols = list(splits_obj["cols"]["tabular_text"]["gbdt"])

    train_ds = pd.read_parquet(os.path.join(PROJECT_DIR, "outputs", "dataset", "training_dataset.parquet"))
    pids = train_ds["property_id"].astype(str).to_numpy()
    n = len(pids)

    vnpz = np.load(VISION_NPZ, allow_pickle=True)
    v_pids = vnpz["property_id"]
    v_sig = vnpz["sig_real"]
    v_so = vnpz["so_sheet"]
    v_q = vnpz["q_feats"]
    v_so_q = vnpz["so_q_feats"]
    v_pid_to_i = {str(p): i for i, p in enumerate(v_pids)}

    matched_rows, matched_vi = [], []
    for row_i, p in enumerate(pids):
        vi = v_pid_to_i.get(p)
        if vi is not None:
            matched_rows.append(row_i)
            matched_vi.append(vi)
    matched_rows = np.asarray(matched_rows, dtype=np.int64)
    matched_vi = np.asarray(matched_vi, dtype=np.int64)

    sig_all = np.zeros((n, 768), dtype=np.float32)
    so_all = np.zeros((n, 1152), dtype=np.float32)
    q_all = np.zeros((n, 30), dtype=np.float32)
    so_q_all = np.zeros((n, 10), dtype=np.float32)
    sig_all[matched_rows] = v_sig[matched_vi].astype(np.float32)
    so_all[matched_rows] = v_so[matched_vi].astype(np.float32)
    q_all[matched_rows] = v_q[matched_vi].astype(np.float32)
    so_q_all[matched_rows] = v_so_q[matched_vi].astype(np.float32)

    has_vis = (
        np.isfinite(sig_all).all(axis=1)
        & np.isfinite(so_all).all(axis=1)
        & np.isfinite(q_all).all(axis=1)
        & np.isfinite(so_q_all).all(axis=1)
        & (np.linalg.norm(sig_all, axis=1) > 1e-4)
        & (np.linalg.norm(so_all, axis=1) > 1e-4)
    )

    e5_npz = np.load(E5_NPZ, allow_pickle=True)
    e5_key = "embeddings" if "embeddings" in e5_npz.files else e5_npz.files[0]
    e5_all = e5_npz[e5_key].astype(np.float32)

    tab_cols = [
        c for c in tt_cols
        if c in full_df.columns and (
            pd.api.types.is_numeric_dtype(full_df[c]) or pd.api.types.is_bool_dtype(full_df[c])
        )
    ]

    X_raw = full_df[tab_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    medians = np.nanmedian(X_raw, axis=0).astype(np.float32)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    bad = ~np.isfinite(X_raw)
    if bad.any():
        X_raw[bad] = np.take(medians, np.where(bad)[1])

    tab_scaler = StandardScaler()
    C_all = tab_scaler.fit_transform(X_raw).astype(np.float32)
    np.clip(C_all, -8.0, 8.0, out=C_all)

    e5_pca = PCA(n_components=32, random_state=42)
    E5_32 = e5_pca.fit_transform(e5_all).astype(np.float32)
    e5_scaler = StandardScaler()
    E5_s = e5_scaler.fit_transform(E5_32).astype(np.float32)
    Ctx_all = np.hstack([C_all, E5_s])

    sig_pca = PCA(n_components=32, random_state=42).fit(sig_all[has_vis])
    so_pca = PCA(n_components=32, random_state=42).fit(so_all[has_vis])
    Z_ph = np.hstack([sig_pca.transform(sig_all[has_vis]), so_pca.transform(so_all[has_vis])]).astype(np.float32)
    z_scaler = StandardScaler()
    Z_ph_s = z_scaler.fit_transform(Z_ph).astype(np.float32)

    Q = np.hstack([q_all[has_vis, :10], so_q_all[has_vis, :10]]).astype(np.float32)
    q_scaler = StandardScaler()
    Q_s = q_scaler.fit_transform(Q).astype(np.float32)

    alphas = np.logspace(1.0, 4.5, 10)
    f_fwd = RidgeCV(alphas=alphas).fit(Ctx_all[has_vis], Z_ph_s)
    f_q = RidgeCV(alphas=alphas).fit(Ctx_all[has_vis], Q_s)
    f_rev = RidgeCV(alphas=alphas).fit(np.hstack([C_all[has_vis], Z_ph_s]), E5_s[has_vis])

    # Calibrate percentile reference tables from strictly OOF JEPA scores (so live percentiles match OOF!)
    if os.path.exists(JEPA_OOF_PARQUET):
        oof_j = pd.read_parquet(JEPA_OOF_PARQUET)
        oof_j = oof_j[oof_j["normal_vis"]]
        ref_quantiles = {
            "dz_quality": np.sort(oof_j["jepa_dz_quality_index"].to_numpy(dtype=np.float32)),
            "energy_sym": np.sort(oof_j["jepa_energy_sym"].to_numpy(dtype=np.float32)),
            "cherry_spread": np.sort(oof_j["jepa_cherry_pick_spread"].to_numpy(dtype=np.float32)),
        }
    else:
        dZ_in = Z_ph_s - f_fwd.predict(Ctx_all[has_vis])
        dQ_in = Q_s - f_q.predict(Ctx_all[has_vis])
        dE5_in = E5_s[has_vis] - f_rev.predict(np.hstack([C_all[has_vis], Z_ph_s]))
        e_sym_in = 0.5 * (np.sqrt(np.mean(dZ_in ** 2, axis=1)) + np.sqrt(np.mean(dE5_in ** 2, axis=1)))
        dz_q_in = (
            0.30 * 0.5 * (dQ_in[:, 0] + dQ_in[:, 10])
            - 0.30 * 0.5 * (dQ_in[:, 1] + dQ_in[:, 11])
            + 0.15 * 0.5 * (dQ_in[:, 3] + dQ_in[:, 13])
            + 0.10 * 0.5 * (dQ_in[:, 4] + dQ_in[:, 14])
            + 0.08 * 0.5 * (dQ_in[:, 2] + dQ_in[:, 12])
            + 0.07 * 0.5 * (dQ_in[:, 6] + dQ_in[:, 16])
        )
        cherry_in = q_all[has_vis, 10] - q_all[has_vis, 27]
        ref_quantiles = {
            "dz_quality": np.sort(dz_q_in.astype(np.float32)),
            "energy_sym": np.sort(e_sym_in.astype(np.float32)),
            "cherry_spread": np.sort(cherry_in.astype(np.float32)),
        }

    inspector = CrossModalJEPAInspector(
        tab_cols=tab_cols,
        tab_medians=medians,
        tab_scaler=tab_scaler,
        e5_pca=e5_pca,
        e5_scaler=e5_scaler,
        sig_pca=sig_pca,
        so_pca=so_pca,
        z_scaler=z_scaler,
        q_scaler=q_scaler,
        f_fwd=f_fwd,
        f_q=f_q,
        f_rev=f_rev,
        ref_quantiles=ref_quantiles,
        metadata={
            "n_train_with_photos": int(has_vis.sum()),
            "n_tab_cols": len(tab_cols),
            "alpha_fwd": float(f_fwd.alpha_),
            "alpha_q": float(f_q.alpha_),
            "alpha_rev": float(f_rev.alpha_),
        },
    )
    inspector.save(out_path)
    sz_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"Saved CrossModalJEPAInspector -> {out_path} ({sz_mb:.2f} MB) in {time.time() - t0:.2f}s")
    return inspector


if __name__ == "__main__":
    fit_and_save_jepa_inspector()
