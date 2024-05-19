import json
import os

def save_house_data(house_data):
    directory = '../data/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_name = f"{house_data['title'].replace(' ', '_')}.json"
    with open(os.path.join(directory, file_name), 'w') as file:
        json.dump(house_data, file, indent=4)