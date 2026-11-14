"""Train a standalone XGBoost model specifically for SHAP explanations."""
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs', 'models')


def main():
    print("Training standalone XGBoost for SHAP explanations...")

    df = pd.read_csv(os.path.join(BASE_DIR, 'outputs', 'property_data_with_embeddings.csv'))
    df['date_obtained'] = pd.to_datetime(df['date_obtained'])
    print(f"Loaded {len(df)} rows")

    # Same column drops as AutoGluon training
    drop_cols = ['property_id', 'date_obtained', 'street', 'house_number', 'box',
                 'city', 'immo_office_name', 'log_price', 'price_sq']
    drop_cols = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=drop_cols)

    # Encode categoricals with LabelEncoder (needed for XGBoost)
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = df[col].fillna('_missing_')
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    # Bool to int
    for col in df.select_dtypes(include=['bool']).columns:
        df[col] = df[col].astype(int)

    # Split
    X = df.drop(columns=['price'])
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # Train
    model = xgb.XGBRegressor(
        n_estimators=500, max_depth=8, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # Evaluate
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    print(f"RMSE: {rmse:.1f}k ({rmse*1000:.0f} EUR)")
    print(f"MAE:  {mae:.1f}k ({mae*1000:.0f} EUR)")

    # Save model, feature names, and label encoders
    artifact = {
        'model': model,
        'feature_names': X.columns.tolist(),
        'label_encoders': encoders,
        'base_value': float(y_train.mean()),
    }
    path = os.path.join(OUTPUT_DIR, 'shap_xgboost.joblib')
    joblib.dump(artifact, path)
    print(f"Saved to {path}")


if __name__ == "__main__":
    main()
