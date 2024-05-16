import os
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.immoweb.be/en/search/house-and-apartment/for-sale/gent/9000?countries=BE&isNewlyBuilt=false&page="

def parse_page(html):
    soup = BeautifulSoup(html, 'html.parser')
    properties = soup.find_all('a', class_='card__title-link')
    property_urls = [property.get('href') for property in properties if property.get('href')]
    return property_urls

def save_house_data(house_url):
    headers = {'User-Agent': 'Mozilla/5.0 (compatible; immo/1.0; +http://yourwebsite.com/bot)'}
    try:
        response = requests.get(house_url, headers=headers)
        response.raise_for_status()

        # Extract property ID from URL
        property_id = house_url.split('/')[-1]

        # Create a directory for the property using its unique ID
        directory = f"data/{property_id}"
        os.makedirs(directory, exist_ok=True)

        # Save the full HTML source to a file
        file_path = os.path.join(directory, "full_source.html")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(response.text)

        print(f"Full source code saved to {file_path}")

    except requests.RequestException as e:
        print(f"Error fetching the URL {house_url}: {e}")

def scrape_houses():
    page = 1
    while page <10:
        url = BASE_URL + str(page)
        try:
            response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0 (compatible; immo/1.0; +http://yourwebsite.com/bot)'})
            response.raise_for_status()
            houses = parse_page(response.text)
            if not houses:
                break  # Exit loop if no more houses are found
            for house in houses:
                save_house_data(house)
            page += 1
            print(f'Completed Page: {page-1}, Starting Page: {page}')
        except requests.RequestException as e:
            print(f"Error during request to {url}: {e}")
            break

# Run the scraper
scrape_houses()
