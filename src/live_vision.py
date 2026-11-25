"""Live Vision Feature Extractor for Issue #22 (SigLIP2-base + SO400M-384).

Reproduces the EXACT offline feature-extraction contract from `src/vision_full_pipeline.py`
for a single live Immoweb listing (or small batch) so `POST /api/predict/url` activates the
full Phase-7 `multimodal` Lean-5 track (7.27% MdAPE vs 7.70% tabular+text):

  - Extracts up to 6 evenly-spaced `mediumUrl` (or `largeUrl`) photo URLs from `window.classified`.
  - Downloads them concurrently with tight per-image timeouts and polite headers.
  - Encodes on MPS (or CPU fallback) with:
      1) `google/siglip2-base-patch16-224` -> real-photo-filtered 768-d mean embedding,
         8 render/media flags (`render_feats`), and 30 real-photo Q-scores (`q_feats`).
      2) Smart real-photo-prioritized 3x2 contact sheet (600x400) encoded with
         `google/siglip2-so400m-patch14-384` -> 1152-d holistic sheet embedding (`so_sheet`)
         and 10 SO400M Q-scores (`so_q_feats`).
  - Persists per-listing embeddings to a lightweight disk cache under
    `outputs/cache/live_vision/<property_id>.npz` (so repeat queries / form edits on the
    same listing are instantaneous) and also checks the offline corpus cache
    `external_data/vision_full/vision_full_embeddings.npz` when a property_id is already known.
  - Strictly imports `torch` lazily only when encoding; returns `None` gracefully on
    network failures, missing photos, or timeouts so callers fall back cleanly to `tabular_text`.
"""
from __future__ import annotations

import concurrent.futures
import io
import os
import threading
import time
from typing import Any

import numpy as np
import requests
from PIL import Image

from vision_sandbox import MEDIA_TYPE_PROMPTS, Q_PAIRS, ROOM_PROMPTS

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CACHE_DIR = os.path.join(BASE_DIR, "outputs", "cache", "live_vision")
OFFLINE_EMB_CACHE = os.path.join(BASE_DIR, "external_data", "vision_full", "vision_full_embeddings.npz")

SIGLIP_BASE_ID = "google/siglip2-base-patch16-224"
SIGLIP_SO400M_ID = "google/siglip2-so400m-patch14-384"
MAX_PHOTOS = 6
PRIO_ROOMS = [0, 2, 1, 3, 5, 4]


def extract_photo_urls_from_classified(classified: dict[str, Any] | None, max_pics: int = MAX_PHOTOS) -> list[str]:
    """Extract up to `max_pics` evenly-spaced medium/large photo URLs from `window.classified`."""
    if not isinstance(classified, dict):
        return []
    media = classified.get("media")
    if not isinstance(media, dict):
        return []
    pics = media.get("pictures") or []
    urls: list[str] = []
    for p in pics:
        if isinstance(p, dict):
            u = p.get("mediumUrl") or p.get("largeUrl")
            if isinstance(u, str) and u.startswith(("http://", "https://")):
                urls.append(u)
    total = len(urls)
    if total <= max_pics:
        return urls
    idx = np.linspace(0, total - 1, max_pics, dtype=int)
    return [urls[int(i)] for i in idx]


def download_listing_photos(
    urls: list[str],
    timeout_sec: float = 6.0,
    max_workers: int = 6,
) -> list[Image.Image]:
    """Concurrently download listing photos into PIL RGB images; skips broken/404 images."""
    if not urls:
        return []

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    }

    def _fetch_one(u: str) -> Image.Image | None:
        try:
            resp = requests.get(u, headers=headers, timeout=timeout_sec)
            if resp.status_code == 200 and len(resp.content) > 800:
                with Image.open(io.BytesIO(resp.content)) as im:
                    return im.convert("RGB").copy()
        except Exception:
            return None
        return None

    images: list[Image.Image] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(max_workers, len(urls))) as ex:
        for img in ex.map(_fetch_one, urls):
            if img is not None:
                images.append(img)
    return images


def build_contact_sheet(
    images: list[Image.Image],
    rm: np.ndarray,
    is_plan: np.ndarray,
    is_cgi: np.ndarray,
) -> Image.Image:
    """Build the exact 600x400 (3x2) real-photo-prioritized contact sheet used offline."""
    idxs = list(range(len(images)))
    real_idxs = [j for j in idxs if (not is_plan[j]) and (not is_cgi[j])]
    pool = real_idxs if len(real_idxs) >= 3 else ([j for j in idxs if not is_plan[j]] or idxs)

    chosen: list[Image.Image] = []
    used: set[int] = set()
    for rt in PRIO_ROOMS:
        for j in pool:
            if j not in used and int(rm[j]) == rt:
                chosen.append(images[j])
                used.add(j)
                break
        if len(chosen) == 6:
            break
    for j in pool:
        if len(chosen) == 6:
            break
        if j not in used:
            chosen.append(images[j])
            used.add(j)

    canvas = Image.new("RGB", (600, 400), (20, 20, 20))
    for slot, im in enumerate(chosen[:6]):
        tile = im.resize((200, 200), Image.Resampling.BILINEAR)
        canvas.paste(tile, ((slot % 3) * 200, (slot // 3) * 200))
    return canvas


def aggregate_siglip_base_features(
    sp: np.ndarray,
    media_txt: np.ndarray,
    room_txt: np.ndarray,
    q_pos_txt: np.ndarray,
    q_neg_txt: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute exact real-filtered SigLIP2-base embedding, render_feats (8,), q_feats (30,).

    Returns: (sig_real, render_feats, q_feats, rm, is_plan, is_cgi)
    Matches `src/vision_full_pipeline.py` lines 317-363 bit-for-bit.
    """
    sp = np.asarray(sp, dtype=np.float32)
    n = len(sp)
    ms = sp @ media_txt.T
    cm = ms[:, 1] - ms[:, 0]
    rm = (sp @ room_txt.T).argmax(axis=1)
    qa = (sp @ q_pos_txt.T) - (sp @ q_neg_txt.T)
    cls = ms.argmax(axis=1)

    is_plan = (rm == 6) | (cls == 3)
    is_cgi = (cm > 0.0) | (cls == 1)
    is_vstage = (cls == 2)
    non_plan = ~is_plan
    real_only = (~is_plan) & (~is_cgi)

    use_raw = non_plan if non_plan.any() else np.ones(n, dtype=bool)
    use_real = real_only if real_only.any() else use_raw

    sig_real = sp[use_real].mean(axis=0).astype(np.float32)
    render_feats = np.array([
        float(cm.mean()),
        float(cm.max()),
        float(is_cgi.mean()),
        float(is_vstage.mean()),
        float((is_cgi | is_vstage).mean()),
        float((~is_cgi & ~is_vstage & ~is_plan).mean()),
        float(is_plan.mean()),
        float((cls == 4).mean()),
    ], dtype=np.float32)

    qm = qa[use_real].mean(axis=0)
    qp = np.percentile(qa[use_real], 75, axis=0)

    def _rq(room_idx_set: list[int], q_col: int, agg: str = "mean") -> float:
        mk = use_real & np.isin(rm, room_idx_set)
        vals = qa[mk, q_col] if mk.any() else qa[use_real, q_col]
        if agg == "max":
            return float(np.max(vals))
        if agg == "min":
            return float(np.min(vals))
        return float(np.mean(vals))

    room_q = np.array([
        _rq([1], 3),
        _rq([3], 4),
        _rq([0], 6),
        _rq([5], 8),
        _rq([2, 4], 5),
        _rq([2, 4], 2),
        _rq([0, 1, 2, 3, 4], 1, "max"),
        _rq([0, 1, 2, 3, 4], 0, "min"),
        float((rm == 1).any()),
        float((rm == 5).any()),
    ], dtype=np.float32)
    q_feats = np.concatenate([qm, qp, room_q]).astype(np.float32)
    return sig_real, render_feats, q_feats, rm, is_plan, is_cgi


class LiveVisionEncoder:
    """Lazy-loaded, disk-cached SigLIP2-base + SO400M-384 vision feature extractor.

    Thread-safe lock around GPU inference so concurrent Flask requests never collide on MPS.
    """

    def __init__(
        self,
        cache_dir: str = DEFAULT_CACHE_DIR,
        offline_npz_path: str = OFFLINE_EMB_CACHE,
        enabled: bool = True,
    ):
        self.cache_dir = cache_dir
        self.offline_npz_path = offline_npz_path
        self.enabled = enabled
        self._lock = threading.Lock()
        self._device: str | None = None

        # Lazy model handles & pre-encoded prompt matrices
        self._sig_proc = None
        self._sig_mod = None
        self._media_txt: np.ndarray | None = None
        self._room_txt: np.ndarray | None = None
        self._q_pos_txt: np.ndarray | None = None
        self._q_neg_txt: np.ndarray | None = None

        self._so_proc = None
        self._so_mod = None
        self._so_q_pos: np.ndarray | None = None
        self._so_q_neg: np.ndarray | None = None

        # Lazy index of offline corpus vision embeddings (for instant lookup of known listings)
        self._offline_idx: dict[str, int] | None = None
        self._offline_npz = None

    def _ensure_offline_index(self) -> None:
        if self._offline_idx is not None:
            return
        self._offline_idx = {}
        if self.offline_npz_path and os.path.exists(self.offline_npz_path):
            try:
                vz = np.load(self.offline_npz_path, allow_pickle=True)
                pids = vz["property_id"].astype(str)
                self._offline_idx = {str(pid): int(i) for i, pid in enumerate(pids)}
                self._offline_npz = vz
            except Exception:
                self._offline_idx = {}

    def _lookup_cached(self, property_id: str | None) -> dict[str, Any] | None:
        if not property_id:
            return None
        pid = str(property_id).strip()
        if not pid:
            return None

        # 1) Check per-listing live disk cache first
        cache_path = os.path.join(self.cache_dir, f"{pid}.npz")
        if os.path.exists(cache_path):
            try:
                d = np.load(cache_path)
                return {
                    "sig_real": d["sig_real"].astype(np.float32).reshape(1, 768),
                    "so_sheet": d["so_sheet"].astype(np.float32).reshape(1, 1152),
                    "render_feats": d["render_feats"].astype(np.float32).reshape(1, 8),
                    "q_feats": d["q_feats"].astype(np.float32).reshape(1, 30),
                    "so_q_feats": d["so_q_feats"].astype(np.float32).reshape(1, 10),
                    "has_vis": np.array([True], dtype=bool),
                    "n_photos_used": int(d["n_photos_used"]) if "n_photos_used" in d else MAX_PHOTOS,
                    "source": "live_cache",
                }
            except Exception:
                pass

        # 2) Check offline corpus cache
        self._ensure_offline_index()
        if self._offline_npz is not None and pid in self._offline_idx:
            idx = self._offline_idx[pid]
            sig = self._offline_npz["sig_real"][idx].astype(np.float32)
            if np.isfinite(sig).all():
                return {
                    "sig_real": sig.reshape(1, 768),
                    "so_sheet": self._offline_npz["so_sheet"][idx].astype(np.float32).reshape(1, 1152),
                    "render_feats": self._offline_npz["render_feats"][idx].astype(np.float32).reshape(1, 8),
                    "q_feats": self._offline_npz["q_feats"][idx].astype(np.float32).reshape(1, 30),
                    "so_q_feats": self._offline_npz["so_q_feats"][idx].astype(np.float32).reshape(1, 10),
                    "has_vis": np.array([True], dtype=bool),
                    "n_photos_used": MAX_PHOTOS,
                    "source": "corpus_cache",
                }
        return None

    def _save_cache(self, property_id: str | None, bundle: dict[str, Any]) -> None:
        if not property_id or not self.cache_dir:
            return
        pid = str(property_id).strip()
        if not pid or not pid.isalnum():
            return
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            np.savez_compressed(
                os.path.join(self.cache_dir, f"{pid}.npz"),
                sig_real=bundle["sig_real"][0].astype(np.float16),
                so_sheet=bundle["so_sheet"][0].astype(np.float16),
                render_feats=bundle["render_feats"][0].astype(np.float32),
                q_feats=bundle["q_feats"][0].astype(np.float32),
                so_q_feats=bundle["so_q_feats"][0].astype(np.float32),
                n_photos_used=np.int32(bundle.get("n_photos_used", 0)),
            )
        except Exception:
            pass

    def _ensure_models_loaded(self) -> None:
        if self._sig_mod is not None and self._so_mod is not None:
            return
        import torch
        from transformers import AutoModel, AutoProcessor

        self._device = "mps" if torch.backends.mps.is_available() else "cpu"
        dtype = torch.bfloat16 if self._device == "mps" else torch.float32

        if self._sig_mod is None:
            self._sig_proc = AutoProcessor.from_pretrained(SIGLIP_BASE_ID, use_fast=True)
            self._sig_mod = AutoModel.from_pretrained(SIGLIP_BASE_ID, dtype=dtype).to(self._device).eval()

            with torch.no_grad():
                def _enc_sig_txt(txts: list[str]) -> np.ndarray:
                    inp = self._sig_proc(text=txts, padding="max_length", max_length=64, truncation=True, return_tensors="pt")
                    inp = {k: v.to(self._device) for k, v in inp.items()}
                    tf = self._sig_mod.get_text_features(**inp)
                    if not isinstance(tf, torch.Tensor):
                        tf = tf.pooler_output
                    tf = tf.float()
                    tf = tf / tf.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                    return tf.cpu().numpy().astype(np.float32)

                self._media_txt = _enc_sig_txt(MEDIA_TYPE_PROMPTS)
                self._room_txt = _enc_sig_txt(list(ROOM_PROMPTS.values()))
                self._q_pos_txt = _enc_sig_txt([p for _, p, _ in Q_PAIRS])
                self._q_neg_txt = _enc_sig_txt([n for _, _, n in Q_PAIRS])

        if self._so_mod is None:
            self._so_proc = AutoProcessor.from_pretrained(SIGLIP_SO400M_ID, use_fast=True)
            self._so_mod = AutoModel.from_pretrained(SIGLIP_SO400M_ID, dtype=dtype).to(self._device).eval()

            with torch.no_grad():
                def _enc_so_txt(txts: list[str]) -> np.ndarray:
                    inp = self._so_proc(text=txts, padding="max_length", max_length=64, truncation=True, return_tensors="pt")
                    inp = {k: v.to(self._device) for k, v in inp.items()}
                    tf = self._so_mod.get_text_features(**inp)
                    if not isinstance(tf, torch.Tensor):
                        tf = tf.pooler_output
                    tf = tf.float()
                    tf = tf / tf.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                    return tf.cpu().numpy().astype(np.float32)

                self._so_q_pos = _enc_so_txt([p for _, p, _ in Q_PAIRS])
                self._so_q_neg = _enc_so_txt([n for _, _, n in Q_PAIRS])

    def encode_images(self, images: list[Image.Image]) -> dict[str, Any] | None:
        """Encode 1..N PIL RGB images of a single listing into a `vis_bundle` dict."""
        if not images or not self.enabled:
            return None
        import torch

        with self._lock:
            self._ensure_models_loaded()
            dtype = torch.bfloat16 if self._device == "mps" else torch.float32
            with torch.no_grad():
                # 1) SigLIP2-base on individual listing photos
                inp = self._sig_proc(images=images, return_tensors="pt")
                inp = {
                    k: (v.to(self._device, dtype=dtype) if v.is_floating_point() else v.to(self._device))
                    for k, v in inp.items()
                }
                f = self._sig_mod.get_image_features(**inp)
                if not isinstance(f, torch.Tensor):
                    f = f.pooler_output
                f = f.float()
                f = f / f.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                sp = f.cpu().numpy().astype(np.float32)

                sig_real, render_feats, q_feats, rm, is_plan, is_cgi = aggregate_siglip_base_features(
                    sp, self._media_txt, self._room_txt, self._q_pos_txt, self._q_neg_txt
                )

                # 2) Smart 3x2 contact sheet -> SigLIP2-SO400M-384
                sheet = build_contact_sheet(images, rm, is_plan, is_cgi)
                inp_so = self._so_proc(images=[sheet], return_tensors="pt")
                inp_so = {
                    k: (v.to(self._device, dtype=dtype) if v.is_floating_point() else v.to(self._device))
                    for k, v in inp_so.items()
                }
                f_so = self._so_mod.get_image_features(**inp_so)
                if not isinstance(f_so, torch.Tensor):
                    f_so = f_so.pooler_output
                f_so = f_so.float()
                f_so = f_so / f_so.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                so_sheet = f_so.cpu().numpy().astype(np.float32)  # (1, 1152)

                so_q_feats = (so_sheet @ self._so_q_pos.T) - (so_sheet @ self._so_q_neg.T)  # (1, 10)

        return {
            "sig_real": sig_real.reshape(1, 768),
            "so_sheet": so_sheet.reshape(1, 1152),
            "render_feats": render_feats.reshape(1, 8),
            "q_feats": q_feats.reshape(1, 30),
            "so_q_feats": so_q_feats.astype(np.float32).reshape(1, 10),
            "has_vis": np.array([True], dtype=bool),
            "n_photos_used": len(images),
            "pct_cgi_renders": round(float(render_feats[2]) * 100.0, 1),
            "source": "live_encoded",
        }

    def extract_for_listing(
        self,
        property_id: str | None,
        photo_urls: list[str] | None,
        timeout_sec: float = 6.0,
    ) -> dict[str, Any] | None:
        """Cache-first live vision extraction for a listing. Returns None on failure/no photos."""
        if not self.enabled:
            return None
        cached = self._lookup_cached(property_id)
        if cached is not None:
            return cached

        urls = list(photo_urls or [])[:MAX_PHOTOS]
        if not urls:
            return None

        t0 = time.perf_counter()
        images = download_listing_photos(urls, timeout_sec=timeout_sec)
        if not images:
            return None
        bundle = self.encode_images(images)
        if bundle is not None:
            bundle["elapsed_ms"] = round((time.perf_counter() - t0) * 1000.0)
            self._save_cache(property_id, bundle)
        return bundle
