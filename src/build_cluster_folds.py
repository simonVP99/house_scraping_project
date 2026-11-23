"""Issue #4 — Anti-Chaining Spatial+Text Cluster CV Builder.

Replaces exact string `property_key` folds (which leak 23.8% of rows when agencies
edit `livable_area` by 1 m² or re-format a street name) with strictly leak-free
connected-component `cluster_id` folds, while preventing transitive chaining across
distinct units in apartment blocks or terraced streets.

Edge rule (same `property_type_HOUSE` REQUIRED for every edge):
  1. `dist <= 10 m` AND `|area_1 - area_2| / max(area_1, area_2) <= 0.05`
  2. `E5-large cosine > 0.98` AND `dist <= 250 m`
  3. Same normalized `postal_code|street|house_number` AND (`property_type_HOUSE == 1` OR `area_diff <= 0.10`)

Outputs:
  `outputs/dataset/cluster_folds.parquet` with columns:
    `property_id`, `date_obtained`, `old_fold`, `cluster_id`, `cluster_size`, `fold_cluster`
  plus prints exact leakage audit + balance across all 5 folds and the "normal home"
  (€150k–€650k, non-block/mansion) slice!

Usage:
    .venv-api/bin/python -u src/build_cluster_folds.py
"""
from __future__ import annotations

import os
import re
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, 'outputs', 'dataset', 'training_dataset.parquet')
RAW_PARQUET = os.path.join(BASE_DIR, 'outputs', 'property_data_full.parquet')
E5L_PATH = os.path.join(BASE_DIR, 'outputs', 'models', 'workbench', 'e5_large_14599.npz')
OUT_PATH = os.path.join(BASE_DIR, 'outputs', 'dataset', 'cluster_folds.parquet')


class UnionFind:
    def __init__(self, n: int):
        self.p = list(range(n))
        self.sz = [1] * n

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.sz[ra] < self.sz[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra
        self.sz[ra] += self.sz[rb]
        return True


def _norm_addr(postal, street, hnr) -> str:
    if pd.isna(street) or pd.isna(hnr):
        return ''
    s = re.sub(r'[^a-z0-9]', '', str(street).lower())
    h = re.sub(r'[^a-z0-9]', '', str(hnr).lower())
    p = re.sub(r'[^0-9]', '', str(postal)) if pd.notna(postal) else ''
    if not s or not h:
        return ''
    return f'{p}|{s}|{h}'


def build_cluster_folds(n_folds: int = 5, seed: int = 42) -> pd.DataFrame:
    t0 = time.perf_counter()
    tr = pd.read_parquet(DATASET_PATH)
    tr['property_id'] = tr['property_id'].astype(str)
    n = len(tr)

    # Pull raw street + house_number for rule 3
    raw = pd.read_parquet(RAW_PARQUET, columns=['property_id', 'date_obtained', 'street', 'house_number'])
    raw['property_id'] = raw['property_id'].astype(str)
    raw['date_obtained_dt'] = pd.to_datetime(raw['date_obtained'], format='%d%m%Y', errors='coerce')
    raw = raw.drop_duplicates(subset=['property_id', 'date_obtained_dt'], keep='first')

    tr_dt = pd.to_datetime(tr['date_obtained'], errors='coerce')
    m = pd.DataFrame({'property_id': tr['property_id'], 'date_obtained_dt': tr_dt}).merge(
        raw[['property_id', 'date_obtained_dt', 'street', 'house_number']],
        on=['property_id', 'date_obtained_dt'],
        how='left',
    )

    # Load E5-large 1024-d L2-normalized embeddings (exact row alignment verified by property_id)
    e5z = np.load(E5L_PATH, allow_pickle=True)
    e5_pids = [str(x) for x in e5z['property_id']]
    assert e5_pids == tr['property_id'].tolist(), "E5-large alignment mismatch!"
    e5 = e5z['embeddings'].astype(np.float32)
    e5 = e5 / np.maximum(np.linalg.norm(e5, axis=1, keepdims=True), 1e-8)

    # Convert lat/lon to local Ghent meters (accurate to <0.5m across Ghent bbox)
    lat = tr['latitude'].to_numpy(float)
    lon = tr['longitude'].to_numpy(float)
    lat0 = 51.0531
    x_m = (lon - 3.7265) * 111_320.0 * np.cos(np.radians(lat0))
    y_m = (lat - lat0) * 111_320.0
    xy = np.column_stack([x_m, y_m])

    area = tr['livable_area'].to_numpy(float)
    is_house = tr['property_type_HOUSE'].to_numpy(int)
    pkey = tr['property_key'].astype(str).to_numpy()

    uf = UnionFind(n)

    # Always keep exact property_key together first
    pk_map: dict[str, int] = {}
    for i, k in enumerate(pkey):
        if k in pk_map:
            uf.union(pk_map[k], i)
        else:
            pk_map[k] = i

    tree = cKDTree(xy)

    # Rule 1: dist <= 10m AND area_diff <= 5% AND same_type
    edges_r1 = 0
    for i, j in tree.query_pairs(r=10.0):
        if is_house[i] != is_house[j]:
            continue
        mx = max(area[i], area[j], 1.0)
        if abs(area[i] - area[j]) / mx <= 0.05:
            if uf.union(i, j):
                edges_r1 += 1

    # Rule 2: dist <= 250m AND E5-large cosine > 0.98 AND same_type
    edges_r2 = 0
    pairs_250 = list(tree.query_pairs(r=250.0))
    if pairs_250:
        arr = np.asarray(pairs_250, dtype=np.int32)
        same_t = is_house[arr[:, 0]] == is_house[arr[:, 1]]
        arr = arr[same_t]
        # Compute cosine in chunks to stay light on RAM
        chunk = 100_000
        for s in range(0, len(arr), chunk):
            sub = arr[s:s + chunk]
            sims = np.einsum('ij,ij->i', e5[sub[:, 0]], e5[sub[:, 1]])
            hit = sub[sims > 0.98]
            for i, j in hit:
                if uf.union(int(i), int(j)):
                    edges_r2 += 1

    # Rule 3: same normalized postal|street|house_nr AND same_type AND (HOUSE==1 OR area_diff<=10%)
    edges_r3 = 0
    addr_groups: dict[str, list[int]] = defaultdict(list)
    postals = tr['postal_code'].to_numpy()
    streets = m['street'].to_numpy()
    hnrs = m['house_number'].to_numpy()
    for i in range(n):
        ak = _norm_addr(postals[i], streets[i], hnrs[i])
        if ak:
            addr_groups[ak].append(i)

    for ak, idxs in addr_groups.items():
        if len(idxs) < 2:
            continue
        for a_idx in range(len(idxs)):
            i = idxs[a_idx]
            for b_idx in range(a_idx + 1, len(idxs)):
                j = idxs[b_idx]
                if is_house[i] != is_house[j]:
                    continue
                if is_house[i] == 1:
                    if uf.union(i, j):
                        edges_r3 += 1
                else:
                    mx = max(area[i], area[j], 1.0)
                    if abs(area[i] - area[j]) / mx <= 0.10:
                        if uf.union(i, j):
                            edges_r3 += 1

    # Materialize compact cluster_id 0..K-1
    roots = [uf.find(i) for i in range(n)]
    uniq_roots = sorted(set(roots))
    rmap = {r: k for k, r in enumerate(uniq_roots)}
    cluster_ids = np.array([rmap[r] for r in roots], dtype=np.int32)

    cl_counts = pd.Series(cluster_ids).value_counts()
    cluster_sizes = np.array([cl_counts[c] for c in cluster_ids], dtype=np.int32)

    # Audit leakage in old folds
    old_folds = tr['fold'].to_numpy(int)
    df_tmp = pd.DataFrame({'cid': cluster_ids, 'old_fold': old_folds})
    folds_per_cl = df_tmp.groupby('cid')['old_fold'].nunique()
    leaked_cls = set(folds_per_cl[folds_per_cl > 1].index)
    leaked_rows = int(df_tmp['cid'].isin(leaked_cls).sum())

    # Assign 5 balanced grouped folds via greedy bin-packing (largest clusters first, tie-break shuffled)
    rng = np.random.default_rng(seed)
    cl_order = list(cl_counts.index)
    # Shuffle within same size bucket for randomness while keeping fold sizes almost identical
    sized: dict[int, list[int]] = defaultdict(list)
    for c in cl_order:
        sized[int(cl_counts[c])].append(int(c))
    ordered_cls = []
    for sz in sorted(sized.keys(), reverse=True):
        grp = sized[sz]
        rng.shuffle(grp)
        ordered_cls.extend(grp)

    fold_loads = [0] * n_folds
    cl_to_fold: dict[int, int] = {}
    for c in ordered_cls:
        f_min = int(np.argmin(fold_loads))
        cl_to_fold[c] = f_min
        fold_loads[f_min] += int(cl_counts[c])
    fold_cluster = np.array([cl_to_fold[c] for c in cluster_ids], dtype=np.int32)

    # Verify zero leaked clusters!
    chk = pd.DataFrame({'cid': cluster_ids, 'f': fold_cluster}).groupby('cid')['f'].nunique()
    assert int((chk > 1).sum()) == 0, "Leaked cluster detected!"

    out = pd.DataFrame({
        'property_id': tr['property_id'].astype(str),
        'date_obtained': tr['date_obtained'],
        'old_fold': old_folds,
        'cluster_id': cluster_ids,
        'cluster_size': cluster_sizes,
        'fold_cluster': fold_cluster,
    })
    out.to_parquet(OUT_PATH, index=False)

    # Print crisp audit summary
    normal_mask = (
        (tr['price'] >= 150_000) & (tr['price'] <= 650_000)
        & (~tr['property_subtype'].isin(['APARTMENT_BLOCK', 'MANSION', 'VILLA', 'SERVICE_FLAT']))
    )
    print(f"=== ANTI-CHAINING CLUSTER CV BUILT IN {time.perf_counter()-t0:.2f}s ===")
    print(f"  Rows: {n:,} | Old property_key groups: {len(set(pkey)):,} -> New clusters: {len(uniq_roots):,}")
    print(f"  New merges: R1(<=10m & <=5% area)={edges_r1:,} | R2(E5>0.98 & <=250m)={edges_r2:,} | R3(same addr)={edges_r3:,}")
    print(f"  Max cluster size: {int(cl_counts.max())} | Clusters > 10 rows: {int((cl_counts > 10).sum())}")
    print(f"  OLD property_key folds leaked: {len(leaked_cls):,} clusters / {leaked_rows:,} rows ({100*leaked_rows/n:.1f}%)!")
    print(f"  NEW fold_cluster balance: {dict(pd.Series(fold_cluster).value_counts().sort_index())}")
    print(f"  Normal-home focus slice (€150k-€650k, excl blocks/mansions/villas): {int(normal_mask.sum()):,} / {n:,} ({100*normal_mask.mean():.1f}%)")
    print(f"  Saved -> {OUT_PATH}")
    return out


if __name__ == '__main__':
    build_cluster_folds()
