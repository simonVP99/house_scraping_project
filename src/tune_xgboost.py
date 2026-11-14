
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import shap
import warnings
import os
import traceback

# Suppress warnings
warnings.filterwarnings('ignore')

# Configuration
OUTPUT_DIR = 'outputs/models'
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

def evaluate_model(model, X_test, y_test, model_name):
    # Predictions are in log scale
    log_predictions = model.predict(X_test)
    
    # Convert back to original scale (Euros)
    predictions = np.exp(log_predictions)
    y_test_original = np.exp(y_test)
    
    mae = mean_absolute_error(y_test_original, predictions)
    rmse = np.sqrt(mean_squared_error(y_test_original, predictions))
    r2 = r2_score(y_test_original, predictions)
    
    print(f"--- {model_name} Results (Original Scale €) ---")
    print(f"MAE: €{mae:,.0f}")
    print(f"RMSE: €{rmse:,.0f}")
    print(f"R2: {r2:.4f}")
    
    return {'Model': model_name, 'MAE': mae, 'RMSE': rmse, 'R2': r2}

def tune_xgboost(X_train, y_train):
    print("\nTuning XGBoost Hyperparameters...")
    
    categorical_transformer = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, [c for c in CATEGORICAL_FEATURES if c in X_train.columns]),
        ],
        remainder='passthrough' 
    )
    
    # Define pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', xgb.XGBRegressor(n_jobs=-1, random_state=42))
    ])
    
    # Parameter Grid
    param_grid = {
        'regressor__n_estimators': [500, 1000, 1500],
        'regressor__learning_rate': [0.01, 0.05, 0.1],
        'regressor__max_depth': [3, 5, 7, 9],
        'regressor__subsample': [0.7, 0.9, 1.0],
        'regressor__colsample_bytree': [0.7, 0.9, 1.0],
        'regressor__gamma': [0, 0.1, 0.5]
    }
    
    # Randomized Search
    search = RandomizedSearchCV(
        pipeline, 
        param_distributions=param_grid, 
        n_iter=20, 
        scoring='neg_mean_squared_error', 
        cv=3, 
        verbose=1, 
        random_state=42,
        n_jobs=-1
    )
    
    search.fit(X_train, y_train)
    
    print(f"Best Parameters: {search.best_params_}")
    return search.best_estimator_

def analyze_shap(model, X_train, X_test):
    print("\nGenerating SHAP plots...")
    
    # Extract the regressor and preprocessor
    regressor = model.named_steps['regressor']
    preprocessor = model.named_steps['preprocessor']
    
    # Transform data
    X_test_transformed = preprocessor.transform(X_test)
    
    # Get feature names
    cat_names = [c for c in CATEGORICAL_FEATURES if c in X_train.columns]
    other_names = [c for c in X_train.columns if c not in cat_names]
    feature_names = cat_names + other_names
    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(regressor)
    shap_values = explainer.shap_values(X_test_transformed)
    
    # Summary Plot
    plt.figure()
    shap.summary_plot(shap_values, X_test_transformed, feature_names=feature_names, show=False)
    plt.title("SHAP Summary Plot")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'shap_summary.png'))
    plt.close()
    
    # Bar Plot
    plt.figure()
    shap.summary_plot(shap_values, X_test_transformed, feature_names=feature_names, plot_type="bar", show=False)
    plt.title("SHAP Feature Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'shap_importance.png'))
    plt.close()
    
    print(f"SHAP plots saved to {OUTPUT_DIR}")

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_file = os.path.join(base_dir, 'outputs/property_data_preprocessed.csv')
    
    df = load_and_clean_data(input_file)
    
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BOOLEAN_FEATURES
    feature_cols = [c for c in feature_cols if c in df.columns]
    
    X = df[feature_cols]
    y = df[TARGET]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    try:
        # Tune XGBoost
        best_model = tune_xgboost(X_train, y_train)
        
        # Evaluate
        evaluate_model(best_model, X_test, y_test, "Tuned XGBoost")
        
        # SHAP Analysis
        # Use a subset of test data for SHAP to speed it up if needed, but 3000 rows is fine for TreeExplainer
        analyze_shap(best_model, X_train, X_test)
        
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
