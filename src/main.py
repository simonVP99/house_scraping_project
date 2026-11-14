import argparse
import time
import os
import pandas as pd
from scraper import scrape_houses
from parser import parse_data_folder_parallel

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FOLDER = os.path.join(BASE_DIR, 'data')
OUTPUT_FILE = os.path.join(BASE_DIR, 'outputs/property_data_raw.csv')
FAILURES_FILE = os.path.join(BASE_DIR, 'outputs/parse_failures.csv')
MANIFEST_FILE = os.path.join(BASE_DIR, 'outputs/parse_manifest.json')


def merge_with_previous(new_df, output_file):
    """Union freshly parsed rows onto the existing raw CSV.

    Parsing incrementally only yields the *new* files, so the previously parsed
    rows have to be carried forward.  Keyed on (property_id, date_obtained) so
    re-parsing an already-known file replaces rather than duplicates it.
    """
    if new_df.empty:
        print("No new rows parsed; existing dataset left unchanged.")
        return None
    if not os.path.exists(output_file):
        return new_df
    previous = pd.read_csv(output_file, low_memory=False)
    if 'property_id' not in previous.columns or 'date_obtained' not in previous.columns:
        print("Existing output lacks identity columns; writing new rows only.")
        return new_df

    for frame in (previous, new_df):
        frame['property_id'] = frame['property_id'].astype(str)
        frame['date_obtained'] = frame['date_obtained'].astype(str)

    combined = pd.concat([previous, new_df], ignore_index=True)
    before = len(combined)
    combined = combined.drop_duplicates(subset=['property_id', 'date_obtained'], keep='last')
    combined = combined.sort_values(['date_obtained', 'property_id']).reset_index(drop=True)
    print(f"Merged: {len(previous)} previous + {len(new_df)} new "
          f"-> {len(combined)} rows ({before - len(combined)} duplicates replaced)")
    return combined


def main():
    parser = argparse.ArgumentParser(description='Scrape houses and extract data.')
    parser.add_argument('num_pages', type=int, help='Number of pages to scrape')
    parser.add_argument('--incremental', action='store_true',
                        help='Only parse HTML files not already in the manifest '
                             '(much faster; reuses the existing raw CSV).')
    parser.add_argument('--skip-scrape', action='store_true',
                        help='Skip scraping and only run the parse stage.')
    parser.add_argument('--full-reparse', action='store_true',
                        help='Ignore the manifest and re-parse every HTML file.')
    args = parser.parse_args()

    if not args.skip_scrape:
        scrape_houses(args.num_pages)
    else:
        print('Skipping scrape stage.')

    print('Start Parsing House Data')
    start_time = time.time()

    # Default to incremental when a manifest already exists: re-reading 50GB of
    # HTML to pick up a few hundred new files is the single largest avoidable
    # cost in this pipeline.
    incremental = args.incremental or (
        os.path.exists(MANIFEST_FILE) and not args.full_reparse)

    df = parse_data_folder_parallel(
        DATA_FOLDER,
        OUTPUT_FILE,
        failures_file=FAILURES_FILE,
        manifest_file=MANIFEST_FILE,
        incremental=incremental,
    )
    duration = time.time() - start_time
    print(f"Time taken to parse data folder: {duration:.2f} seconds")

    if incremental:
        merged = merge_with_previous(df, OUTPUT_FILE)
        if merged is None:
            return
        df = merged

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Data extraction and saving completed: {len(df)} rows -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
