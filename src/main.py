import argparse
import time
import os
import polars as pl
from parser import parse_data_folder_parallel, parse_snapshot_store

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FOLDER = os.path.join(BASE_DIR, 'data')
OUTPUT_FILE = os.path.join(BASE_DIR, 'outputs/property_data_raw.csv')
FULL_PARQUET_FILE = os.path.join(BASE_DIR, 'outputs/property_data_full.parquet')
FULL_CSV_FILE = os.path.join(BASE_DIR, 'outputs/property_data_full.csv')
FAILURES_FILE = os.path.join(BASE_DIR, 'outputs/parse_failures.csv')
MANIFEST_FILE = os.path.join(BASE_DIR, 'outputs/parse_manifest.json')


def merge_with_previous(new_df, output_file):
    """Union freshly parsed rows onto the existing raw CSV.

    Parsing incrementally only yields the *new* files, so the previously parsed
    rows have to be carried forward.  Keyed on (property_id, date_obtained) so
    re-parsing an already-known file replaces rather than duplicates it.
    """
    if new_df is None or new_df.height == 0:
        print("No new rows parsed; existing dataset left unchanged.")
        return None
    if not os.path.exists(output_file):
        return new_df

    # Read everything as String: this CSV mixes types across parse generations
    # (an int-like postal code in one run, a string in another), and Polars is
    # strict about that where pandas silently coerced.
    previous = pl.read_csv(output_file, infer_schema_length=0)
    if 'property_id' not in previous.columns or 'date_obtained' not in previous.columns:
        print("Existing output lacks identity columns; writing new rows only.")
        return new_df

    new = new_df.with_columns(
        pl.col('property_id').cast(pl.Utf8),
        pl.col('date_obtained').cast(pl.Utf8),
    )
    # UNION of the two column sets, not the intersection: the old CSV predates
    # newly extracted fields, and aligning to it would silently discard them on
    # every incremental run.
    all_columns = list(previous.columns) + [c for c in new.columns
                                            if c not in previous.columns]
    missing_in_new = [c for c in previous.columns if c not in new.columns]
    if missing_in_new:
        new = new.with_columns(
            [pl.lit(None, dtype=pl.Utf8).alias(c) for c in missing_in_new])
    missing_in_old = [c for c in new.columns if c not in previous.columns]
    if missing_in_old:
        previous = previous.with_columns(
            [pl.lit(None, dtype=pl.Utf8).alias(c) for c in missing_in_old])
    previous = previous.select(all_columns)
    new = new.select(all_columns)

    combined = pl.concat([previous, new], how='vertical_relaxed')
    before = combined.height
    combined = (combined
                .unique(subset=['property_id', 'date_obtained'], keep='last')
                .sort(['date_obtained', 'property_id']))
    print(f"Merged: {previous.height} previous + {new.height} new "
          f"-> {combined.height} rows ({before - combined.height} duplicates replaced)")
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
    parser.add_argument('--from-snapshots', action='store_true',
                        help='Parse the full corpus from data/.snapshots instead '
                             'of HTML files and write property_data_full.{parquet,csv}.')
    args = parser.parse_args()

    if not args.skip_scrape:
        from scraper import scrape_houses
        scrape_houses(args.num_pages)
    else:
        print('Skipping scrape stage.')

    print('Start Parsing House Data')
    start_time = time.time()

    if args.from_snapshots:
        df, errors = parse_snapshot_store(DATA_FOLDER)
        duration = time.time() - start_time
        print(f"Time taken to parse snapshot store: {duration:.2f} seconds")
        os.makedirs(os.path.dirname(FULL_PARQUET_FILE), exist_ok=True)
        df.write_parquet(FULL_PARQUET_FILE)
        df.write_csv(FULL_CSV_FILE)
        print(f"Snapshot extraction completed: {df.height} rows x {df.width} cols -> {FULL_PARQUET_FILE}")
        return

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
    df.write_csv(OUTPUT_FILE)
    print(f"Data extraction and saving completed: {df.height} rows -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
