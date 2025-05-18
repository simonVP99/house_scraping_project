from operator import add
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
import statsmodels.formula.api as smf
import geopandas as gpd
from shapely.geometry import Point, shape
import json


def perform_regression_plot(df, x_col, y_col, color = None, hover = None):# Put this in utils.tools
    #df_filtered = df[df[y_col] != 0]


    df_filtered = df.replace([np.inf, -np.inf], np.nan).dropna()

    fig = px.scatter(df_filtered, x=x_col, y=y_col, color=color, hover_data=hover)


    X = df_filtered[x_col]
    Y = df_filtered[y_col]
    X = sm.add_constant(X)  

    model = sm.OLS(Y, X).fit()
    predictions = model.predict(X)

    fig.add_traces(px.line(x=df_filtered[x_col], y=predictions).data)

    params = model.params
    r_squared = model.rsquared
    regression_info = f'y = {params[1]:.5f}x + {params[0]:.2f}\n, R² = {r_squared:.2f}'

    fig.add_annotation(
        x=df_filtered[x_col].max(),
        y=df_filtered[y_col].max(),
        text=regression_info,
        showarrow=False,
        xanchor='right',
        yanchor='top',
        bordercolor='black',
        borderwidth=1
    )
    fig.update_layout(title=f'Regression Plot of {x_col} vs {y_col}')
    fig.show()

def fill_missing_values(df_non_nan, df_nan, target_column, cols):
    X = df_non_nan[cols]
    y = df_non_nan[target_column]
    
    X = X.dropna()
    y = y[X.index]  

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    #print(f"Summary for {target_column}:")
    #print("Coefficients:", model.coef_)
    #print("Intercept:", model.intercept_)
    y_pred = np.round(model.predict(X_test)).astype('int')
    #print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
    #print("R-squared:", r2_score(y_test, y_pred))
    #print()
    
    X_nan = df_nan[cols].dropna()
    df_nan.loc[X_nan.index, target_column] = np.round(model.predict(X_nan)).astype('int')
    
    df_filled = pd.concat([df_non_nan, df_nan]).sort_index()
    
    return df_filled[target_column]


def fill_missing_values_poisson(df_non_nan, df_nan, target_column, cols):
    # Prepare the training data
    X_train = df_non_nan[cols]
    y_train = df_non_nan[target_column]
    
    # Concatenate X and y to create a DataFrame for the formula interface
    train_data = pd.concat([X_train, y_train], axis=1)
    
    # Build the formula for Poisson regression
    formula = f"{target_column} ~ " + ' + '.join(cols)
    
    # Fit the Poisson regression model
    poisson_model = smf.glm(formula=formula, data=train_data, family=sm.families.Poisson()).fit()
    
    # Prepare the data for prediction
    X_nan = df_nan[cols].dropna()
    
    # Predict the missing values
    y_pred = poisson_model.predict(X_nan)
    
    # Round predictions to the nearest integer and ensure non-negative
    y_pred_rounded = np.round(y_pred).astype(int)
    y_pred_rounded[y_pred_rounded < 0] = 0  # Ensure counts are non-negative
    
    # Fill in the missing values
    df_nan.loc[X_nan.index, target_column] = y_pred_rounded
    
    # Combine the DataFrames
    df_filled = pd.concat([df_non_nan, df_nan]).sort_index()
    
    return df_filled[target_column]

def preprocess(df, keep = 'first'):
    import pandas as pd
    import numpy as np
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    # Change format of 'date_obtained'
    df['date_obtained'] = df['date_obtained'].astype(str)
    df['date_obtained'] = df.date_obtained.str.zfill(8)
    df['date_obtained'] = pd.to_datetime(df['date_obtained'], format='%d%m%Y')
    df = df.sort_values('date_obtained')
    df.drop_duplicates(subset=["property_id"], keep=keep, inplace=True)

    # Set 'price' as K (divide by 1000)
    df['price'] = df['price'] / 1000

    # Calculate 'age' of the property
    df['age'] = 2024 - df['year_built']

    # Convert 'has' columns to boolean
    for col in [col for col in df.columns if col.startswith('has')]:
        df[col] = df[col].replace({'True': True, 'False': False})
        df[col].fillna(False, inplace=True)
        df[col] = df[col].astype(bool)

    # Fill NA's for categorical columns
    df['heating_type'].fillna('Unknown', inplace=True)
    df['building_condition'].fillna('Unknown', inplace=True)
    df['flood_zone_type'].fillna('Unknown', inplace=True)

    # Fill NA's for numerical columns
    df['garden_surface'].fillna(0, inplace=True)
    df['terrace_surface'].fillna(0, inplace=True)

    # Replace zeros with NaN for certain numerical columns (if zeros are invalid)
    df['number_of_bathrooms'].replace(0, np.nan, inplace=True)
    df['number_of_toilets'].replace(0, np.nan, inplace=True)
    df['number_of_bedrooms'].replace(0, np.nan, inplace=True)
    df['cadastral_income'].replace(0, np.nan, inplace=True)

    # Remove values where livable area is unknown
    df.dropna(subset=['livable_area', 'energy_label', 'latitude'], inplace=True)

    # Create additional features
    df['price_sq'] = df['price'] / df['livable_area']

    # Drop unnecessary columns
    df = df.drop(columns=['has_dining_room', "number_of_rooms", 'total_floors', 'floor_number'])

    # One-hot encode 'property_type'
    df = pd.get_dummies(df, columns=['property_type'], drop_first=True)

    df['postal_code'] = df['postal_code'].astype(str)

    # Fill in NaNs in 'number_of_bedrooms'
    df_non_nan_bedrooms = df[df['number_of_bedrooms'].notna()]
    df_nan_bedrooms = df[df['number_of_bedrooms'].isna()]
    cols_bedrooms = ['livable_area', 'property_type_HOUSE', 'price', 'price_sq']
    df['number_of_bedrooms'] = fill_missing_values_poisson(
        df_non_nan_bedrooms,
        df_nan_bedrooms,
        'number_of_bedrooms',
        cols_bedrooms
    )

    # Fill in NaNs in 'number_of_toilets'
    df_non_nan_toilets = df[df['number_of_toilets'].notna()]
    df_nan_toilets = df[df['number_of_toilets'].isna()]
    cols_toilets = cols_bedrooms + ['number_of_bedrooms']
    df['number_of_toilets'] = fill_missing_values_poisson(
        df_non_nan_toilets,
        df_nan_toilets,
        'number_of_toilets',
        cols_toilets
    )

    # Fill in NaNs in 'number_of_bathrooms'
    df_non_nan_bathrooms = df[df['number_of_bathrooms'].notna()]
    df_nan_bathrooms = df[df['number_of_bathrooms'].isna()]
    cols_bathrooms = cols_toilets + ['number_of_toilets']
    df['number_of_bathrooms'] = fill_missing_values_poisson(
        df_non_nan_bathrooms,
        df_nan_bathrooms,
        'number_of_bathrooms',
        cols_bathrooms
    )

    # Fill in NaNs in 'cadastral_income' (assuming it's count-like)
    df_non_nan_income = df[df['cadastral_income'].notna()]
    df_nan_income = df[df['cadastral_income'].isna()]
    cols_income = ['price', 'price_sq', 'property_type_HOUSE']
    df['cadastral_income'] = fill_missing_values_poisson(
        df_non_nan_income,
        df_nan_income,
        'cadastral_income',
        cols_income
    )

    # Fill missing 'primary_energy_consumption' based on 'energy_label' mean
    df['primary_energy_consumption'] = df.groupby('energy_label')['primary_energy_consumption'].transform(
        lambda x: x.fillna(x.mean())
    )
    df.dropna(subset=['primary_energy_consumption'], inplace=True)

    # Add neighborhood information (assuming the function is defined)
    df = add_neighbourhood(df)

    upper_limit = df['latitude'].quantile(0.99)
    df = df[df['latitude'] <= upper_limit]

    upper_limit = df['longitude'].quantile(0.99)
    df = df[df['longitude'] <= upper_limit]

    lower_limit = df['latitude'].quantile(0.01)
    df = df[df['latitude'] >= lower_limit]

    lower_limit = df['longitude'].quantile(0.01)
    df = df[df['longitude'] >= lower_limit]

    df['log_price'] = np.log(df.price)
    upper_limit = df['log_price'].quantile(0.99)
    df = df[df['log_price'] <= upper_limit]

    
    upper_limit = df['price_sq'].quantile(0.99)
    df = df[df['price_sq'] <= upper_limit]


    return df


def add_neighbourhood(df, wijken_path = 'external_data/stadswijken-gent.csv'):
    wijken = pd.read_csv(wijken_path, sep = ';')
    wijken['geometry'] = wijken['Geometry'].apply(parse_geometry)
    wijken_gdf = gpd.GeoDataFrame(wijken, geometry='geometry')
    wijken_gdf.set_crs(epsg=4326, inplace=True)

    df['neighbourhood'] = df.apply(
    lambda row: get_neighborhood(row['latitude'], row['longitude'], wijken_gdf),
    axis=1
    )
    df['neighbourhood'] = df['neighbourhood'].str.replace(' ', '', regex=False)
    df['neighbourhood'] = df['neighbourhood'].str.replace('-', '_', regex=False)
    
    return df

def parse_geometry(geometry_str):
    geometry_dict = json.loads(geometry_str)
    return shape(geometry_dict)

def get_neighborhood(lat, lon, wijken_gdf):
    """
    Given a latitude and longitude, return the neighborhood name.

    Parameters:
    - lat (float): Latitude of the point.
    - lon (float): Longitude of the point.
    - wijken_gdf (GeoDataFrame): GeoDataFrame containing neighborhood polygons.

    Returns:
    - str or None: The name of the neighborhood if found, else None.
    """
    # Create a shapely Point from the latitude and longitude
    point = Point(lon, lat)  # Note: Point takes (x, y) = (longitude, latitude)
    
    # Use spatial index to find possible matching neighborhoods
    possible_matches_index = list(wijken_gdf.sindex.intersection(point.bounds))
    possible_matches = wijken_gdf.iloc[possible_matches_index]
    
    # Iterate through possible matches and check containment
    for idx, row in possible_matches.iterrows():
        if row['geometry'].contains(point):
            return row['wijk']  # You can change 'wijk' to another column if needed
    
    # Return None if no neighborhood is found
    return 'Unknown'