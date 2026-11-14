
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Configuration
OUTPUT_DIR = 'outputs/models'
os.makedirs(OUTPUT_DIR, exist_ok=True)
MODEL_PATH = os.path.join(OUTPUT_DIR, 'house_price_model.pkl')
OPTIONS_PATH = os.path.join(OUTPUT_DIR, 'feature_options.pkl')

TARGET = 'log_price'

# Feature Definitions
NUMERIC_FEATURES = [
    'livable_area', 'number_of_bedrooms', 'number_of_bathrooms', 'number_of_toilets',
    'terrace_surface', 'garden_surface', 'parking_indoor', 'parking_outdoor', 'parking_closed_box',
    'primary_energy_consumption', 'latitude', 'longitude', 'year_built', 'cadastral_income',
    'dist_from_center', 'distance_to_street'
]

CATEGORICAL_FEATURES = [
    'property_subtype', 'building_condition', 'kitchen_type', 'energy_label',
    'heating_type', 'flood_zone_type', 'neighbourhood', 'street_type'
]

BOOLEAN_FEATURES = [
    'has_photovoltaic_panels', 'has_double_glazing', 'has_living_room',
    'has_attic', 'has_basement', 'has_terrace', 'has_garden', 'has_swimming_pool',
    'is_biddit_sale'
]

def load_and_clean_data(filepath):
    print("Loading data...")
    df = pd.read_csv(filepath, low_memory=False)
    
    # Basic Cleaning
    print("Cleaning data...")
    
    # Drop rows with missing target
    df = df.dropna(subset=[TARGET])
    
    # Fill NaNs for specific columns where NaN implies 0 or False
    fill_zero_cols = ['parking_indoor', 'parking_outdoor', 'parking_closed_box', 
                      'terrace_surface', 'garden_surface', 'cadastral_income']
    for col in fill_zero_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
            
    if 'is_biddit_sale' in df.columns:
        df['is_biddit_sale'] = df['is_biddit_sale'].fillna(False)
        
    # Fill Categorical NaNs with 'Unknown'
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')
            # Ensure string type
            df[col] = df[col].astype(str)
            
    # Fill Boolean NaNs with False (safe assumption for "has_" features usually)
    for col in BOOLEAN_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(False).astype(int) # Convert to 0/1

    return df

def save_model():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_file = os.path.join(base_dir, 'outputs/property_data_preprocessed.csv')
    
    df = load_and_clean_data(input_file)
    
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BOOLEAN_FEATURES
    feature_cols = [c for c in feature_cols if c in df.columns]
    
    X = df[feature_cols]
    y = df[TARGET]
    
    # Best Parameters from previous tuning
    best_params = {
        'n_estimators': 1500,
        'learning_rate': 0.1,
        'max_depth': 5,
        'subsample': 0.7,
        'colsample_bytree': 0.9,
        'gamma': 0,
        'n_jobs': -1,
        'random_state': 42
    }
    
    print("Training final model on full dataset...")
    
    categorical_transformer = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, [c for c in CATEGORICAL_FEATURES if c in X.columns]),
        ],
        remainder='passthrough' 
    )
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', xgb.XGBRegressor(**best_params))
    ])
    
    pipeline.fit(X, y)
    
    print(f"Saving model to {MODEL_PATH}...")
    joblib.dump(pipeline, MODEL_PATH)
    
    # Save feature options for UI
    options = {}
    for col in CATEGORICAL_FEATURES:
        if col in X.columns:
            options[col] = sorted(df[col].unique().tolist())
            
    # Also save median values for numeric features to use as defaults
    defaults = {}
    for col in NUMERIC_FEATURES:
        if col in X.columns:
            defaults[col] = df[col].median()
            
    print(f"Saving feature options to {OPTIONS_PATH}...")
    joblib.dump({'options': options, 'defaults': defaults, 'features': feature_cols}, OPTIONS_PATH)
    
    print("Done!")

if __name__ == "__main__":
    save_model()
