import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

PRODUCTION_BUNDLE_DIR = os.path.join(PROJECT_DIR, "outputs", "models", "production")
OOF_ANALYTICS_PATH = os.path.join(PRODUCTION_BUNDLE_DIR, "oof_analytics.parquet")
MARKET_INSIGHTS_PATH = os.path.join(PRODUCTION_BUNDLE_DIR, "market_insights.json")
EMBEDDINGS_CSV_PATH = os.path.join(PROJECT_DIR, "outputs", "property_data_with_embeddings.csv")
TODAY_DEALS_CSV_PATH = os.path.join(PROJECT_DIR, "outputs", "today_deals.csv")
NEIGHBOURHOOD_DATA_PATH = os.path.join(PROJECT_DIR, "external_data", "stadswijken-gent.csv")
STREET_GRAPH_PATH = os.path.join(PROJECT_DIR, "external_data", "ghent_region_network.graphml")
E5_MODEL_NAME = "intfloat/multilingual-e5-large"

