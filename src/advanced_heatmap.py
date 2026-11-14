
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
from sklearn.neighbors import KNeighborsRegressor
import os

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def generate_grid(df, resolution=100):
    """
    Generate a regular grid of coordinates covering the data area.
    """
    min_lat, max_lat = df['latitude'].min(), df['latitude'].max()
    min_lon, max_lon = df['longitude'].min(), df['longitude'].max()
    
    # Add small buffer
    lat_buffer = (max_lat - min_lat) * 0.05
    lon_buffer = (max_lon - min_lon) * 0.05
    
    lat_grid = np.linspace(min_lat - lat_buffer, max_lat + lat_buffer, resolution)
    lon_grid = np.linspace(min_lon - lon_buffer, max_lon + lon_buffer, resolution)
    
    xx, yy = np.meshgrid(lon_grid, lat_grid)
    grid_points = np.c_[yy.ravel(), xx.ravel()] # lat, lon
    return grid_points

def interpolate_values(df, grid_points, target_col, n_neighbors=20):
    """
    Interpolate values from df onto grid_points using KNN with distance weighting.
    """
    X = df[['latitude', 'longitude']].values
    y = df[target_col].values
    
    # Use KNN regressor with distance weights (Inverse Distance Weighting)
    knn = KNeighborsRegressor(n_neighbors=n_neighbors, weights='distance')
    knn.fit(X, y)
    
    predicted_values = knn.predict(grid_points)
    
    # We also want to mask grid points that are too far from any actual data
    # to avoid showing misleading heatmaps in empty areas (extrapolation)
    # Calculate distance to nearest data point for each grid point
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=1).fit(X)
    distances, _ = nn.kneighbors(grid_points)
    
    # Threshold: e.g., 95th percentile of nearest neighbor distances in original data
    # Or a fixed reasonable distance (e.g., 0.01 degrees approx 1km)
    # Let's estimate a reasonable threshold dynamically
    avg_dist_between_points = np.mean(distances)
    threshold = avg_dist_between_points * 5 # Allow some gap filling but not wild extrapolation
    
    mask = distances.flatten() < threshold
    
    return grid_points[mask], predicted_values[mask]

def create_robust_heatmap(df, output_dir, target_col, output_filename, title):
    print(f"Generating robust heatmap for {target_col}...")
    
    # Filter valid data
    df_clean = df.dropna(subset=['latitude', 'longitude', target_col])
    
    # 1. Generate Grid
    grid_points = generate_grid(df_clean, resolution=150) # Higher resolution for smoother look
    
    # 2. Interpolate
    valid_grid_points, interpolated_values = interpolate_values(df_clean, grid_points, target_col)
    
    # 3. Create Map
    m = folium.Map(location=[df_clean['latitude'].mean(), df_clean['longitude'].mean()], zoom_start=12)
    
    # Prepare data for HeatMap: [lat, lon, weight]
    # Normalize weights for better visualization if needed, but HeatMap handles raw values too.
    # However, extreme outliers in price can skew the scale.
    # Let's clip the values to 5th-95th percentile to make the color scale useful
    vmin = np.percentile(interpolated_values, 5)
    vmax = np.percentile(interpolated_values, 95)
    clipped_values = np.clip(interpolated_values, vmin, vmax)
    
    # Normalize to 0-1 for HeatMap intensity if we want consistent coloring?
    # Folium HeatMap usually auto-scales, but let's pass the raw (clipped) values.
    # Actually, passing raw large numbers to HeatMap can sometimes be tricky depending on the implementation.
    # Usually it expects a weight.
    
    heat_data = []
    for (lat, lon), val in zip(valid_grid_points, clipped_values):
        heat_data.append([lat, lon, val])
        
    # Add the heatmap
    # We use a smaller radius because we have a dense grid.
    HeatMap(heat_data, 
            name=title, 
            radius=15, 
            blur=20, 
            max_zoom=13,
            overlay=True,
            control=True,
            show=True).add_to(m)
            
    # Add a title to the map
    title_html = f'''
             <h3 align="center" style="font-size:16px"><b>{title}</b></h3>
             '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    output_path = os.path.join(output_dir, output_filename)
    m.save(output_path)
    print(f"Saved to {output_path}")

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_file = os.path.join(base_dir, 'outputs/property_data_preprocessed.csv')
    output_dir = os.path.join(base_dir, 'outputs/eda_plots')
    
    os.makedirs(output_dir, exist_ok=True)
    
    df = load_data(input_file)
    
    # Generate Price Heatmap
    create_robust_heatmap(df, output_dir, 'price', 'robust_heatmap_price.html', 'Interpolated Price Heatmap')
    
    # Generate Price/m² Heatmap
    create_robust_heatmap(df, output_dir, 'price_sq', 'robust_heatmap_price_sqm.html', 'Interpolated Price/m² Heatmap')

if __name__ == "__main__":
    main()
