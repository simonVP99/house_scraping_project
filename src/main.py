import argparse
import time
import os
from scraper import scrape_houses
from parser import parse_data_folder

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Scrape houses and extract data.')
    parser.add_argument('num_pages', type=int, help='Number of pages to scrape')
    args = parser.parse_args()

    # Define absolute paths
    base_dir = '/Users/simon/Desktop/personal/immo_datacollection/house_scraping_project'
    data_folder = os.path.join(base_dir, 'data')
    output_file = os.path.join(base_dir, 'outputs/property_data_raw.csv')

    # Start scraping process
    scrape_houses(args.num_pages)
    print('Start Parsing House Data')
    start_time = time.time()
    df = parse_data_folder(data_folder)
    end_time = time.time()
    duration = end_time - start_time
    print(f"Time taken to parse data folder: {duration:.2f} seconds")

    # Save DataFrame to CSV or perform further processing
    df.to_csv(output_file, index=False)
    print("Data extraction and saving completed.")

if __name__ == "__main__":
    main()
