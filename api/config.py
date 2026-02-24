import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

EMBEDDINGS_CSV_PATH = os.path.join(PROJECT_DIR, "outputs", "property_data_with_embeddings.csv")
TODAY_DEALS_CSV_PATH = os.path.join(PROJECT_DIR, "outputs", "today_deals.csv")
AUTOGLUON_PATH = os.path.join(PROJECT_DIR, "outputs", "models", "autogluon")
PCA_PATH = os.path.join(PROJECT_DIR, "outputs", "models", "description_pca.joblib")
OLD_MODEL_PATH = os.path.join(PROJECT_DIR, "outputs", "models", "house_price_model.pkl")
OLD_OPTIONS_PATH = os.path.join(PROJECT_DIR, "outputs", "models", "feature_options.pkl")
NEIGHBOURHOOD_DATA_PATH = os.path.join(PROJECT_DIR, "external_data", "stadswijken-gent.csv")
E5_MODEL_NAME = "intfloat/multilingual-e5-small"
PCA_COMPONENTS = 50
