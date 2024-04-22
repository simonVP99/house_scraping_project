import requests
from parser import parse_page
from storage import save_house_data
from config import BASE_URL

def scrape_houses():
    try:
        response = requests.get(BASE_URL)
        response.raise_for_status()
        houses = parse_page(response.text)
        for house in houses:
            save_house_data(house)
    except requests.RequestException as e:
        print(f"Error during requests to {BASE_URL}: {e}")