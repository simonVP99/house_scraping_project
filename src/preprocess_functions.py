import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


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
    
    print(f"Summary for {target_column}:")
    print("Coefficients:", model.coef_)
    print("Intercept:", model.intercept_)
    y_pred = np.round(model.predict(X_test)).astype('int')
    print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
    print("R-squared:", r2_score(y_test, y_pred))
    print()
    
    X_nan = df_nan[cols].dropna()
    df_nan.loc[X_nan.index, target_column] = np.round(model.predict(X_nan)).astype('int')
    
    df_filled = pd.concat([df_non_nan, df_nan]).sort_index()
    
    return df_filled[target_column]

def preprocess(df):

    # Change format of Date_obtained
    df['date_obtained'] = pd.to_datetime(df['date_obtained'], format='%d%m%Y')
    df = df.sort_values('date_obtained')
    df.drop_duplicates(subset = ["property_id"], keep = 'last', inplace=True)

    # Set Price as K
    df['price'] = df['price'] / 1000

    df['age'] = 2024 - df['year_built']

    # Fill NA's
    df['heating_type'].fillna('Unknown', inplace=True)
    df['has_photovoltaic_panels'].fillna('False', inplace=True)
    df['has_double_glazing'].fillna('False', inplace=True)
    df['has_attic'].fillna('False', inplace=True)
    df['has_basement'].fillna('False', inplace=True)
    df['has_terrace'].fillna('False', inplace=True)
    df['has_garden'].fillna('False', inplace=True)

    
    df['building_condition'].fillna('Unknown', inplace=True)
    df['flood_zone_type'].fillna('Unknown', inplace=True)
    df['garden_surface'].fillna(0, inplace=True)
    df['terrace_surface'].fillna(0, inplace=True)


    df['number_of_bathrooms'].replace(0, np.nan, inplace=True)
    df['number_of_toilets'].replace(0, np.nan, inplace=True)
    df['number_of_bedrooms'].replace(0, np.nan, inplace=True)
    df['cadastral_income'].replace(0, np.nan, inplace=True)

    #Remove values where livable area is unknown
    df.dropna(subset=['livable_area', 'energy_label', 'latitude'], inplace = True)

    df['price_sq'] = df['price']/df['livable_area']

    df = df.drop(columns = ['has_dining_room', "number_of_rooms", 'total_floors', 'floor_number'])
    
    
    df = pd.get_dummies(df, columns=['property_type'], drop_first=True)

    df['postal_code'] = df['postal_code'].astype(str)

    #Fill in Nans in number_of_bathrooms/toilets/bedrooms

    df_non_nan_bedrooms = df[df['number_of_bedrooms'].notna()]
    df_nan_bedrooms = df[df['number_of_bedrooms'].isna()]

    cols = ['livable_area','property_type_HOUSE']

    df['number_of_bedrooms'] = fill_missing_values(df_non_nan_bedrooms, df_nan_bedrooms, 'number_of_bedrooms', cols)
    
    df_non_nan_toilets = df[df['number_of_toilets'].notna()]
    df_nan_toilets = df[df['number_of_toilets'].isna()]
    df['number_of_toilets'] = fill_missing_values(df_non_nan_toilets, df_nan_toilets, 'number_of_toilets', cols = cols+ ['number_of_bedrooms'])

    df_non_nan_bathrooms = df[df['number_of_bathrooms'].notna()]
    df_nan_bathrooms = df[df['number_of_bathrooms'].isna()]
    df['number_of_bathrooms'] = fill_missing_values(df_non_nan_bathrooms, df_nan_bathrooms, 'number_of_bathrooms', cols = cols + ['number_of_bedrooms', 'number_of_toilets'])
    
    df_non_nan_income = df[df['cadastral_income'].notna()]
    df_nan_income = df[df['cadastral_income'].isna()]

    df['cadastral_income'] = fill_missing_values(df_non_nan_income, df_nan_income, 'cadastral_income', cols = ['price', 'property_type_HOUSE'])

    df['primary_energy_consumption'] = df.groupby('energy_label')['primary_energy_consumption'].transform(
    lambda x: x.fillna(x.mean())
    )
    df.dropna(subset = ['primary_energy_consumption'], inplace = True)

    
    return df