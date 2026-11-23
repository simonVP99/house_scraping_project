"""Compute stronger multilingual dense text embeddings (`intfloat/multilingual-e5-large`,
1024-dim) on Apple Silicon MPS (fp16) for all 14,599 training listings.

Caches to `outputs/models/workbench/e5_large_14599.npz` so Phase 4 can build strictly
fold-safe nested-OOF Ridge + PCA features without recomputing embeddings.

Usage:
    .venv-api/bin/python -u src/compute_bge_e5_large.py
"""
from __future__ import annotations

import os
import time
import warnings

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PATH = os.path.join(BASE_DIR, 'outputs', 'models', 'workbench', 'e5_large_14599.npz')
MODEL_NAME = 'intfloat/multilingual-e5-large'


def main():
    if os.path.exists(OUT_PATH):
        z = np.load(OUT_PATH)
        print(f"Already cached {OUT_PATH}: shape={z['embeddings'].shape}")
        return

    t0 = time.perf_counter()
    ds = pd.read_parquet(os.path.join(BASE_DIR, 'outputs', 'dataset', 'training_dataset.parquet'), columns=['property_id', 'date_obtained'])
    raw = pd.read_parquet(
        os.path.join(BASE_DIR, 'outputs', 'property_data_full.parquet'),
        columns=['property_id', 'date_obtained', 'title', 'description', 'description_nl'],
    )
    ds['property_id'] = ds['property_id'].astype(str)
    raw['property_id'] = raw['property_id'].astype(str)
    raw['date_obtained'] = pd.to_datetime(raw['date_obtained'], format='%d%m%Y', errors='coerce')
    raw = raw.drop_duplicates(['property_id', 'date_obtained'], keep='first')

    merged = ds.merge(raw, on=['property_id', 'date_obtained'], how='left')
    texts = []
    for ttl, d_nl, d_main in zip(
        merged['title'].fillna(''),
        merged['description_nl'].fillna(''),
        merged['description'].fillna(''),
    ):
        body = d_nl if len(str(d_nl)) >= 40 else d_main
        txt = f"passage: {ttl}. {body}" if ttl else f"passage: {body}"
        texts.append(txt[:2400])

    print(f"Loading {MODEL_NAME} on Apple Silicon MPS (fp16)...")
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).to(device)
    model.eval()

    # Sort by length for minimal padding
    order = np.argsort([len(t) for t in texts])
    inv_order = np.argsort(order)
    sorted_texts = [texts[i] for i in order]

    batch_size = 64
    embs_list = []
    t_enc = time.perf_counter()
    with torch.inference_mode():
        for i in range(0, len(sorted_texts), batch_size):
            batch = sorted_texts[i:i + batch_size]
            tok = tokenizer(batch, padding=True, truncation=True, max_length=448, return_tensors='pt').to(device)
            out = model(**tok)
            mask = tok['attention_mask'].unsqueeze(-1).float()
            pooled = (out.last_hidden_state.float() * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            embs_list.append(pooled.cpu().numpy().astype(np.float32))
            if (i // batch_size) % 40 == 0 or i + batch_size >= len(sorted_texts):
                done = min(len(sorted_texts), i + batch_size)
                rate = done / max(0.1, time.perf_counter() - t_enc)
                print(f"  Encoded {done:,}/{len(sorted_texts):,} ({rate:.1f} docs/s)", flush=True)

    embs = np.vstack(embs_list)[inv_order]
    np.savez_compressed(
        OUT_PATH,
        embeddings=embs,
        property_id=ds['property_id'].to_numpy(),
        date_obtained=ds['date_obtained'].astype(str).to_numpy(),
    )
    print(f"Saved {OUT_PATH}: shape={embs.shape} in {time.perf_counter()-t0:.1f}s")


if __name__ == '__main__':
    main()
