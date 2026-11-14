"""Parse only today's HTML files and append to the existing parsed CSV."""

import sys
import os
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from parser import extract_info
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
PARSED_CSV = os.path.join(BASE_DIR, 'outputs/property_data_parsed.csv')
TODAY_TAG = '23022026'


def main():
    # 1. Find only today's HTML files
    today_files = []
    for dirpath, dirnames, filenames in os.walk(DATA_DIR):
        for fn in filenames:
            if fn == f'full_source_{TODAY_TAG}.html':
                today_files.append(os.path.join(dirpath, fn))

    print(f"Found {len(today_files)} HTML files from today")

    # 2. Parse them in parallel
    data = []
    print(f"Parsing {len(today_files)} files...")
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(extract_info, fp) for fp in today_files]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Parsing",
                           disable=not sys.stderr.isatty()):
            try:
                result = future.result()
                if result:
                    data.append(result)
            except Exception as e:
                pass

    df_today = pd.DataFrame(data)
    print(f"Successfully parsed: {len(df_today)} rows")

    # 3. Load existing parsed CSV and remove old partial today rows
    print(f"Loading existing parsed CSV...")
    df_existing = pd.read_csv(PARSED_CSV, low_memory=False)
    print(f"Existing rows: {len(df_existing):,}")

    old_today_mask = df_existing['date_obtained'].astype(str).str.zfill(8) == TODAY_TAG
    n_old_today = old_today_mask.sum()
    df_existing = df_existing[~old_today_mask]
    print(f"Removed {n_old_today} old partial today rows")

    # 4. Append new today rows
    df_combined = pd.concat([df_existing, df_today], ignore_index=True)
    print(f"Combined: {len(df_combined):,} rows ({len(df_existing):,} existing + {len(df_today)} new today)")

    # 5. Save
    df_combined.to_csv(PARSED_CSV, index=False)
    print(f"Saved updated parsed CSV")


if __name__ == '__main__':
    main()
