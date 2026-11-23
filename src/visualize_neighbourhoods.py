import pandas as pd
import json
import folium
import os

def generate_neighbourhood_map(csv_path, output_path):
    # Read the CSV file
    try:
        df = pd.read_csv(csv_path, sep=';')
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Create a map centered on Ghent
    m = folium.Map(location=[51.05, 3.73], zoom_start=12)

    # Iterate over the rows and add polygons
    for index, row in df.iterrows():
        try:
            # Parse the geometry string
            # The CSV might have double quotes escaped as "", typical CSV behavior. 
            # Pandas usually handles the reading, but the string content might still need adjustment if it's not standard JSON.
            # Looking at the sample: "{""coordinates"": ...}" -> this is likely CSV escaping for ".
            # If pandas read it correctly as a string, it should be: {"coordinates": ...}
            
            geo_str = row['Geometry']
            
            # If pandas read it as a string with double quotes inside, it might be fine directly if parsed as JSON.
            # However, sometimes CSVs are tricky. Let's try loading it.
            try:
                geo_data = json.loads(geo_str)
            except json.JSONDecodeError:
                # Fallback: maybe it has some issue with quotes?
                # The sample showed: "{""coordinates"": ...}" in the file content view.
                # If that's the raw file content, standard CSV readers interpret "" as " inside a quoted field.
                # So df['Geometry'] should ideally contain valid JSON like {"coordinates": ...}.
                pass
            
            geo_json = json.loads(geo_str)
            
            # Extract coordinates
            # GeoJSON Polygon coordinates are usually [[[lon, lat], [lon, lat], ...]]
            # Folium expects [lat, lon]
            
            if geo_json['type'] == 'Polygon':
                coords = geo_json['coordinates'][0] # Outer ring
                # Swap from [lon, lat] to [lat, lon]
                folium_coords = [[p[1], p[0]] for p in coords]
                
                # Add polygon to map
                folium.Polygon(
                    locations=folium_coords,
                    color='blue',
                    weight=2,
                    fill=True,
                    fill_opacity=0.4,
                    tooltip=row['wijk'] # Use 'wijk' as label
                ).add_to(m)
                
                # Add a marker at the center
                # We can use geo_point_2d if available, otherwise calculate centroid
                if 'geo_point_2d' in row and isinstance(row['geo_point_2d'], str):
                    lat_lon = row['geo_point_2d'].split(',')
                    try:
                        center = [float(lat_lon[0]), float(lat_lon[1])]
                        folium.Marker(
                            location=center,
                            popup=row['wijk'],
                            tooltip=row['wijk']
                        ).add_to(m)
                    except:
                        pass

        except Exception as e:
            print(f"Error processing row {index}: {e}")
            continue

    # Save the map
    m.save(output_path)
    print(f"Map saved to {output_path}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    csv_file = os.path.join(project_root, "external_data", "stadswijken-gent.csv")
    output_file = os.path.join(project_root, "neighbourhood_map.html")
    
    generate_neighbourhood_map(csv_file, output_file)
