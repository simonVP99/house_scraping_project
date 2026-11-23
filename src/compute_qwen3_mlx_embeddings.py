"""Compute 2026 SOTA multilingual embeddings via MLX:
  `mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ` (1024-dim, causal last-token EOS pooling).

Embeds BOTH:
  1. All 14,599 training listings -> `outputs/models/workbench/qwen3_emb_14599.npz`
  2. All 1,227 fresh 22–23 Nov live listings -> `outputs/models/workbench/qwen3_emb_fresh_nov23.npz`

Run with the MLX virtualenv:
    /Users/svp/dev/mlx-lab/.venv/bin/python -u src/compute_qwen3_mlx_embeddings.py
"""
from __future__ import annotations

import os
import sys
import time

import mlx.core as mx
import numpy as np
import pandas as pd
from mlx_lm import load

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK_DIR = os.path.join(BASE_DIR, 'outputs', 'models', 'workbench')
DATASET_PATH = os.path.join(BASE_DIR, 'outputs', 'dataset', 'training_dataset.parquet')
RAW_PARQUET = os.path.join(BASE_DIR, 'outputs', 'property_data_full.parquet')
FRESH_AUDITED = os.path.join(WORK_DIR, 'fresh_nov23_predictions_audited.csv')

OUT_TRAIN = os.path.join(WORK_DIR, 'qwen3_emb_14599.npz')
OUT_FRESH = os.path.join(WORK_DIR, 'qwen3_emb_fresh_nov23.npz')
MODEL_ID = 'mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ'
MAX_LEN = 384
BATCH_SIZE = 64


def _build_texts(df_keys: pd.DataFrame, raw_u: pd.DataFrame) -> list[str]:
    m = df_keys[['property_id', 'date_dt']].merge(
        raw_u[['property_id', 'date_dt', 'title', 'description_nl', 'description']],
        on=['property_id', 'date_dt'], how='left',
    )
    texts = []
    for t, nl, d in zip(m['title'].fillna(''), m['description_nl'].fillna(''), m['description'].fillna('')):
        body = str(nl).strip() if len(str(nl).strip()) >= 40 else str(d).strip()
        s = f"{str(t).strip()}. {body}".strip() if str(t).strip() else body
        texts.append(s[:2400])
    return texts


def embed_batch_mlx(model, tok, texts: list[str], label: str) -> np.ndarray:
    t0 = time.perf_counter()
    eos = tok.eos_token_id
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else eos

    # Pre-tokenize and sort by length to minimize padding waste (dynamic batch padding!)
    encoded = []
    for s in texts:
        ids = tok.encode(s)
        if len(ids) > MAX_LEN - 1:
            ids = ids[:MAX_LEN - 1]
        ids.append(eos)
        encoded.append(ids)

    order = np.argsort([len(x) for x in encoded])
    out = np.zeros((len(texts), 1024), dtype=np.float32)

    done = 0
    for start in range(0, len(order), BATCH_SIZE):
        idx = order[start:start + BATCH_SIZE]
        seqs = [encoded[i] for i in idx]
        # Round pad length to multiple of 16 for Metal kernel efficiency
        raw_max = max(len(s) for s in seqs)
        max_l = min(MAX_LEN, ((raw_max + 15) // 16) * 16)
        arr = np.full((len(seqs), max_l), pad_id, dtype=np.int32)
        last_pos = np.empty(len(seqs), dtype=np.int32)
        for bi, s in enumerate(seqs):
            L = len(s)
            arr[bi, :L] = s
            last_pos[bi] = L - 1

        h = model.model(mx.array(arr))
        # Gather last non-pad token per row on GPU before transferring to CPU!
        row_idx = mx.arange(len(seqs), dtype=mx.int32)
        pooled = h[row_idx, mx.array(last_pos)]
        pooled = pooled.astype(mx.float32)
        norms = mx.sqrt(mx.maximum(mx.sum(pooled * pooled, axis=1, keepdims=True), 1e-12))
        pooled = pooled / norms
        mx.eval(pooled)
        out[idx] = np.array(pooled)

        done += len(seqs)
        if done % 2048 < BATCH_SIZE or done == len(order):
            el = time.perf_counter() - t0
            print(f"  [{label}] {done:5d}/{len(order)} ({done/el:5.1f} docs/s, {el:5.1f}s)", flush=True)

    return out


def main():
    t0 = time.perf_counter()
    raw = pd.read_parquet(RAW_PARQUET)[['property_id', 'date_obtained', 'title', 'description_nl', 'description']]
    raw['property_id'] = raw['property_id'].astype(str)
    raw['date_dt'] = pd.to_datetime(raw['date_obtained'], format='%d%m%Y', errors='coerce')
    raw_u = raw.drop_duplicates(['property_id', 'date_dt'], keep='first')

    print(f"Loading {MODEL_ID} via MLX...", flush=True)
    model, tok = load(MODEL_ID)

    if not os.path.exists(OUT_TRAIN):
        tr = pd.read_parquet(DATASET_PATH)[['property_id', 'date_obtained']].copy()
        tr['property_id'] = tr['property_id'].astype(str)
        tr['date_dt'] = pd.to_datetime(tr['date_obtained'])
        texts_tr = _build_texts(tr, raw_u)
        emb_tr = embed_batch_mlx(model, tok, texts_tr, 'TRAIN-14599')
        np.savez_compressed(
            OUT_TRAIN,
            embeddings=emb_tr,
            property_id=tr['property_id'].to_numpy(dtype=str),
            date_obtained=tr['date_dt'].astype(str).to_numpy(dtype=str),
        )
        print(f"Saved {OUT_TRAIN} shape={emb_tr.shape}", flush=True)
    else:
        print(f"[cached] {OUT_TRAIN}", flush=True)

    if not os.path.exists(OUT_FRESH):
        fr = pd.read_csv(FRESH_AUDITED)[['property_id', 'date_obtained']].copy()
        fr['property_id'] = fr['property_id'].astype(str)
        fr['date_dt'] = pd.to_datetime(fr['date_obtained'])
        texts_fr = _build_texts(fr, raw_u)
        emb_fr = embed_batch_mlx(model, tok, texts_fr, 'FRESH-1227')
        np.savez_compressed(
            OUT_FRESH,
            embeddings=emb_fr,
            property_id=fr['property_id'].to_numpy(dtype=str),
            date_obtained=fr['date_dt'].astype(str).to_numpy(dtype=str),
        )
        print(f"Saved {OUT_FRESH} shape={emb_fr.shape}", flush=True)
    else:
        print(f"[cached] {OUT_FRESH}", flush=True)

    print(f"Done in {time.perf_counter()-t0:.1f}s!", flush=True)


if __name__ == '__main__':
    main()
