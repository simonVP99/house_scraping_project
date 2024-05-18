import os
import re
import json
import pandas as pd
from bs4 import BeautifulSoup

def extract_info(file_path):
    """
    Extract property ID, date, price, property type, livable area, energy-related fields, 
    and other room counts and features from the given file path.
    
    Args:
        file_path (str): The path to the HTML file.
        
    Returns:
        dict: A dictionary containing the property ID, date, price, property type, livable area, 
              energy-related fields, and other room counts and features.
    """
    # Extract the directory name (property ID)
    property_id = os.path.basename(os.path.dirname(file_path))
    
    # Extract the date from the file name
    file_name = os.path.basename(file_path)
    date_str = file_name.split('_')[2].split('.')[0]
    
    # Initialize variables
    price = None
    property_type = None
    livable_area = None
    heating_type = None
    has_photovoltaic_panels = None
    has_double_glazing = None
    number_of_rooms = None
    number_of_bathrooms = None
    number_of_bedrooms = None
    number_of_toilets = None
    has_living_room = None
    has_dining_room = None
    has_attic = None
    has_basement = None
    has_terrace = None
    terrace_surface = None
    has_garden = None
    garden_surface = None
    energy_label = None
    primary_energy_consumption = None
    postal_code = None
    latitude = None
    longitude = None
    year_built = None
    building_condition = None
    floor_number = None
    total_floors = None
    property_size = None
    flood_zone_type = None
    cadastral_income = None
    
    # Read and parse the HTML file
    with open(file_path, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'html.parser')
        
        # Find the script tag containing the JSON data
        script_tag = soup.find('script', string=re.compile(r'window\.classified'))
        if script_tag:
            script_content = script_tag.string
            # Extract JSON data from the script content
            json_text = re.search(r'window\.classified\s*=\s*(\{.*?\});', script_content, re.DOTALL)
            if json_text:
                classified_data = json.loads(json_text.group(1))
                # Extract the relevant information
                price_data = classified_data.get('price', {})
                price = price_data.get('mainValue', price_data.get('price'))
                
                property_details = classified_data.get('property', {})
                property_type = property_details.get('type', classified_data.get('type'))
                livable_area = property_details.get('netHabitableSurface', property_details.get('indoor_surface', None))
                
                # Extract the energy-related fields
                energy_data = property_details.get('energy', {})
                if energy_data is not None:
                    heating_type = energy_data.get('heatingType', None)
                    has_photovoltaic_panels = energy_data.get('hasPhotovoltaicPanels', None)
                    has_double_glazing = energy_data.get('hasDoubleGlazing', None)
                
                # Extract primary energy consumption and energy label from the transaction certificates
                transaction_data = classified_data.get('transaction', {})
                certificates_data = transaction_data.get('certificates', {})
                if certificates_data is not None:
                    primary_energy_consumption = certificates_data.get('primaryEnergyConsumptionPerSqm', None)
                    energy_label = certificates_data.get('epcScore', None)
                    
                # Extract cadastral income
                sale_data = transaction_data.get('sale', {})
                if sale_data is not None:
                    cadastral_income = sale_data.get('cadastralIncome', None)
                
                # Extract additional room counts and features
                number_of_rooms = property_details.get('roomCount', None)
                number_of_bathrooms = property_details.get('bathroomCount', None)
                number_of_bedrooms = property_details.get('bedroomCount', None)
                number_of_toilets = property_details.get('toiletCount', None)
                has_living_room = property_details.get('hasLivingRoom', None)
                has_dining_room = property_details.get('hasDiningRoom', None)
                has_attic = property_details.get('hasAttic', None)
                has_basement = property_details.get('hasBasement', None)
                has_terrace = property_details.get('hasTerrace', None)
                terrace_surface = property_details.get('terraceSurface', None)
                has_garden = property_details.get('hasGarden', None)
                garden_surface = property_details.get('gardenSurface', None)
                
                # Extract additional location and property features
                location_data = property_details.get('location', {})
                postal_code = location_data.get('postalCode', None)
                latitude = location_data.get('latitude', None)
                longitude = location_data.get('longitude', None)
                
                building_data = property_details.get('building', {})
                if building_data is not None:
                    year_built = building_data.get('constructionYear', None)
                    building_condition = building_data.get('condition', None)
                    total_floors = building_data.get('floorCount', None)
                    
                floor_number = property_details.get('floor', None)
                
                land_data = property_details.get('land', {})
                if land_data is not None:
                    property_size = land_data.get('surface', None)

                # Extract flood zone information from the constructionPermit section within property details
                construction_permit_data = property_details.get('constructionPermit', {})
                if construction_permit_data is not None:
                    flood_zone_type = construction_permit_data.get('floodZoneType', None)
            else:
                price = None
                property_type = None
                livable_area = None
                heating_type = None
                has_photovoltaic_panels = None
                has_double_glazing = None
                number_of_rooms = None
                number_of_bathrooms = None
                number_of_bedrooms = None
                number_of_toilets = None
                has_living_room = None
                has_dining_room = None
                has_attic = None
                has_basement = None
                has_terrace = None
                terrace_surface = None
                has_garden = None
                garden_surface = None
                energy_label = None
                primary_energy_consumption = None
                postal_code = None
                latitude = None
                longitude = None
                year_built = None
                building_condition = None
                floor_number = None
                total_floors = None
                property_size = None
                flood_zone_type = None
                cadastral_income = None
        else:
            price = None
            property_type = None
            livable_area = None
            heating_type = None
            has_photovoltaic_panels = None
            has_double_glazing = None
            number_of_rooms = None
            number_of_bathrooms = None
            number_of_bedrooms = None
            number_of_toilets = None
            has_living_room = None
            has_dining_room = None
            has_attic = None
            has_basement = None
            has_terrace = None
            terrace_surface = None
            has_garden = None
            garden_surface = None
            energy_label = None
            primary_energy_consumption = None
            postal_code = None
            latitude = None
            longitude = None
            year_built = None
            building_condition = None
            floor_number = None
            total_floors = None
            property_size = None
            flood_zone_type = None
            cadastral_income = None
    
    return {
        'property_id': property_id,
        'date_obtained': date_str,
        'price': price,
        'property_type': property_type,
        'livable_area': livable_area,
        'heating_type': heating_type,
        'has_photovoltaic_panels': has_photovoltaic_panels,
        'has_double_glazing': has_double_glazing,
        'number_of_rooms': number_of_rooms,
        'number_of_bathrooms': number_of_bathrooms,
        'number_of_bedrooms': number_of_bedrooms,
        'number_of_toilets': number_of_toilets,
        'has_living_room': has_living_room,
        'has_dining_room': has_dining_room,
        'has_attic': has_attic,
        'has_basement': has_basement,
        'has_terrace': has_terrace,
        'terrace_surface': terrace_surface,
        'has_garden': has_garden,
        'garden_surface': garden_surface,
        'energy_label': energy_label,
        'primary_energy_consumption': primary_energy_consumption,
        'postal_code': postal_code,
        'latitude': latitude,
        'longitude': longitude,
        'year_built': year_built,
        'building_condition': building_condition,
        'floor_number': floor_number,
        'total_floors': total_floors,
        'property_size': property_size,
        'flood_zone_type': flood_zone_type,
        'cadastral_income': cadastral_income
    }

def parse_data_folder(root_directory):
    """
    Parse the data folder and extract information from each HTML file.
    
    Args:
        root_directory (str): The root directory containing the data folders.
        
    Returns:
        pd.DataFrame: A DataFrame containing the extracted information.
    """
    data = []

    # Walk through the directory structure
    for dirpath, dirnames, filenames in os.walk(root_directory):
        for filename in filenames:
            if filename.endswith('.html'):
                file_path = os.path.join(dirpath, filename)

                # Extract information using the extractor function
                info = extract_info(file_path)

                # Append the extracted information to the list
                data.append(info)

    # Create a DataFrame from the extracted data
    df = pd.DataFrame(data)
    return df
