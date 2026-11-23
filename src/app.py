import gradio as gr
import pandas as pd
import numpy as np
import joblib
import os
import math
import re
import json
import requests
import urllib.request
import csv
import io
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from geopy.geocoders import Nominatim, GoogleV3
import difflib
from shapely.geometry import shape, Point
from shapely.geometry.polygon import Polygon
import folium
import branca.colormap as cm
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

# ============================================================================
# CONFIGURATION & PATHS
# ============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

# Ensure src/ is on the path for lazy imports of parser, preprocess_functions
import sys
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

DATA_DIR = os.path.join(PROJECT_DIR, 'data')
PARSED_CSV_PATH = os.path.join(PROJECT_DIR, 'outputs', 'property_data_parsed.csv')
AUTOGLUON_PATH = os.path.join(PROJECT_DIR, 'outputs', 'models', 'autogluon')
PCA_PATH = os.path.join(PROJECT_DIR, 'outputs', 'models', 'description_pca.joblib')
NEIGHBOURHOOD_DATA_PATH = os.path.join(PROJECT_DIR, 'external_data', 'stadswijken-gent.csv')
NEIGHBOURHOOD_MAP_PATH = os.path.join(PROJECT_DIR, 'neighbourhood_map.html')
EMBEDDINGS_CSV_PATH = os.path.join(PROJECT_DIR, 'outputs', 'property_data_with_embeddings.csv')
TODAY_DEALS_CSV_PATH = os.path.join(PROJECT_DIR, 'outputs', 'today_deals.csv')

# Old model paths (fallback for manual form)
OLD_MODEL_PATH = os.path.join(PROJECT_DIR, 'outputs', 'models', 'house_price_model.pkl')
OLD_OPTIONS_PATH = os.path.join(PROJECT_DIR, 'outputs', 'models', 'feature_options.pkl')

E5_MODEL_NAME = 'intfloat/multilingual-e5-small'
PCA_COMPONENTS = 50

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

# ============================================================================
# NEIGHBOURHOOD POLYGONS
# ============================================================================

def load_neighbourhood_polygons():
    polygons = []
    try:
        df = pd.read_csv(NEIGHBOURHOOD_DATA_PATH, sep=';')
        for _, row in df.iterrows():
            try:
                geo_json = json.loads(row['Geometry'])
                poly = shape(geo_json)
                name = row['wijk']
                polygons.append((poly, name))
            except Exception:
                continue
    except Exception as e:
        print(f"Error loading neighbourhood data: {e}")
    return polygons

print("Loading neighbourhood polygons...")
neighbourhood_polygons = load_neighbourhood_polygons()

# Build normalized neighbourhood name list for AutoGluon
ag_neighbourhood_names = sorted(set(
    name.replace(' ', '').replace('-', '_')
    for _, name in neighbourhood_polygons
)) if neighbourhood_polygons else []
if 'Unknown' not in ag_neighbourhood_names:
    ag_neighbourhood_names.append('Unknown')


def get_neighbourhood_map_html():
    if os.path.exists(NEIGHBOURHOOD_MAP_PATH):
        try:
            with open(NEIGHBOURHOOD_MAP_PATH, 'r', encoding='utf-8') as f:
                html_content = f.read()
                html_content = html_content.replace('"', '&quot;')
                return f'<iframe srcdoc="{html_content}" width="100%" height="600px" style="border:none;"></iframe>'
        except Exception as e:
            return f"Error loading map: {e}"
    return "<div style='text-align:center; padding:50px;'>Map not generated yet.</div>"


# ============================================================================
# LOAD MODELS
# ============================================================================

# --- AutoGluon model ---
ag_predictor = None
try:
    from autogluon.tabular import TabularPredictor
    if os.path.exists(AUTOGLUON_PATH):
        ag_predictor = TabularPredictor.load(AUTOGLUON_PATH)
        print(f"AutoGluon model loaded from {AUTOGLUON_PATH}")
    else:
        print(f"AutoGluon model not found at {AUTOGLUON_PATH}")
except Exception as e:
    print(f"Could not load AutoGluon model: {e}")

# --- E5 embedding model + PCA ---
e5_model = None
e5_tokenizer = None
e5_device = None
pca_model = None
try:
    import torch
    from transformers import AutoTokenizer, AutoModel

    if torch.backends.mps.is_available():
        e5_device = torch.device('mps')
    elif torch.cuda.is_available():
        e5_device = torch.device('cuda')
    else:
        e5_device = torch.device('cpu')

    print(f"Loading E5 model ({E5_MODEL_NAME}) on {e5_device}...")
    e5_tokenizer = AutoTokenizer.from_pretrained(E5_MODEL_NAME)
    e5_model = AutoModel.from_pretrained(E5_MODEL_NAME).to(e5_device)
    e5_model.eval()
    print(f"E5 model loaded")

    if os.path.exists(PCA_PATH):
        pca_model = joblib.load(PCA_PATH)
        print(f"PCA model loaded from {PCA_PATH}")
    else:
        print(f"PCA model not found at {PCA_PATH}")
except Exception as e:
    print(f"Could not load E5/PCA models: {e}")

# --- Old XGBoost model (fallback for manual form) ---
old_model = None
options = {}
defaults = {}
feature_cols = []
try:
    old_model = joblib.load(OLD_MODEL_PATH)
    data = joblib.load(OLD_OPTIONS_PATH)
    options = data['options']
    defaults = data['defaults']
    feature_cols = data['features']
    print("Old XGBoost model loaded")
except Exception as e:
    print(f"Could not load old XGBoost model: {e}")
    options = {
        'neighbourhood': ag_neighbourhood_names,
        'property_subtype': ['HOUSE', 'VILLA', 'APARTMENT', 'GROUND_FLOOR', 'PENTHOUSE',
                             'DUPLEX', 'TRIPLEX', 'FLAT_STUDIO', 'TOWN_HOUSE'],
        'building_condition': ['AS_NEW', 'GOOD', 'JUST_RENOVATED', 'TO_BE_DONE_UP',
                               'TO_RENOVATE', 'TO_RESTORE', 'Unknown'],
        'kitchen_type': ['HYPER_EQUIPPED', 'INSTALLED', 'NOT_INSTALLED',
                         'SEMI_EQUIPPED', 'USA_HYPER_EQUIPPED', 'USA_INSTALLED', 'Unknown'],
        'energy_label': ['A', 'A+', 'A++', 'B', 'C', 'D', 'E', 'F', 'G', 'Unknown'],
        'heating_type': ['ELECTRIC', 'FUELOIL', 'GAS', 'PELLET', 'WOOD', 'Unknown'],
        'flood_zone_type': ['NON_FLOOD_ZONE', 'POSSIBLE_FLOOD_ZONE',
                            'EFFECTIVE_FLOOD_ZONE', 'Unknown'],
        'street_type': ['residential', 'tertiary', 'secondary', 'primary',
                        'living_street', 'unclassified'],
    }
    defaults = {}
    feature_cols = []

# Macro indicator cache
_macro_cache = None


# ============================================================================
# SPATIAL HELPERS
# ============================================================================

def get_neighbourhood_spatial(lat, lon):
    """Finds the neighbourhood for a given lat/lon using point-in-polygon."""
    if not neighbourhood_polygons:
        return None
    point = Point(lon, lat)
    for poly, name in neighbourhood_polygons:
        if poly.contains(point):
            return name
    return None


def get_street_type(address):
    """Infers street type from address string."""
    if not address:
        return 'residential'
    address_lower = address.lower()
    if 'laan' in address_lower:
        return 'tertiary'
    elif 'steeniweg' in address_lower or 'steenweg' in address_lower:
        return 'secondary'
    elif 'plein' in address_lower or 'markt' in address_lower:
        return 'residential'
    elif 'dreef' in address_lower:
        return 'residential'
    elif 'weg' in address_lower:
        return 'unclassified'
    elif 'straat' in address_lower:
        return 'residential'
    return 'residential'


def get_location_data(address):
    """Geocode address, verify neighbourhood spatially, and infer street type."""
    if not address:
        return None, None, None, None
    try:
        if GOOGLE_API_KEY:
            geolocator = GoogleV3(api_key=GOOGLE_API_KEY)
            location = geolocator.geocode(address + ", Ghent, Belgium")
        else:
            geolocator = Nominatim(user_agent="ghent_house_price_predictor_v1")
            location = geolocator.geocode(address + ", Ghent, Belgium", addressdetails=True)

        if location:
            lat = location.latitude
            lon = location.longitude
            spatial_neighbourhood = get_neighbourhood_spatial(lat, lon)
            best_match_neighbourhood = None
            valid_neighbourhoods = options.get('neighbourhood', [])

            if spatial_neighbourhood:
                matches = difflib.get_close_matches(
                    spatial_neighbourhood, valid_neighbourhoods, n=1, cutoff=0.5)
                if matches:
                    best_match_neighbourhood = matches[0]

            if not best_match_neighbourhood and hasattr(location, 'raw') and 'address' in location.raw:
                addr = location.raw['address']
                potential_names = [
                    addr.get('neighbourhood'), addr.get('suburb'),
                    addr.get('quarter'), addr.get('city_district'),
                    addr.get('residential'),
                ]
                potential_names = [p for p in potential_names if p]
                for name in potential_names:
                    matches = difflib.get_close_matches(
                        name, valid_neighbourhoods, n=1, cutoff=0.5)
                    if matches:
                        best_match_neighbourhood = matches[0]
                        break

            raw_address = location.address if location.address else address
            inferred_street_type = get_street_type(raw_address)
            valid_street_types = options.get('street_type', ['residential'])
            best_match_street_type = 'residential'
            matches_st = difflib.get_close_matches(
                inferred_street_type, valid_street_types, n=1, cutoff=0.6)
            if matches_st:
                best_match_street_type = matches_st[0]

            return lat, lon, best_match_neighbourhood, best_match_street_type
        else:
            return None, None, gr.update(), gr.update()
    except Exception as e:
        print(f"Geocoding Error: {str(e)}")
        return None, None, gr.update(), gr.update()


def calculate_dist_from_center(lat, lon):
    """Haversine distance in km (used by old XGBoost model)."""
    center_lat = 51.0543
    center_lon = 3.7174
    R = 6371
    dlat = math.radians(lat - center_lat)
    dlon = math.radians(lon - center_lon)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(center_lat)) * math.cos(math.radians(lat)) *
         math.sin(dlon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def calculate_dist_from_center_ag(lat, lon):
    """Euclidean degree-scaled distance (matches preprocess_functions.py formula)."""
    center_lat = 51.053098573153505
    center_lon = 3.7265139580328865
    return np.sqrt((lat - center_lat) ** 2 + (lon - center_lon) ** 2) * 100


# ============================================================================
# MACRO INDICATORS (from ECB)
# ============================================================================

def _fetch_ecb_series(url, value_col_name):
    """Fetch a single monthly time series from ECB SDW API."""
    req = urllib.request.Request(url, headers={'Accept': 'text/csv'})
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = resp.read().decode()
    reader = csv.DictReader(io.StringIO(data))
    result = {}
    for row in reader:
        result[row['TIME_PERIOD']] = float(row['OBS_VALUE'])
    return pd.DataFrame([
        {'year_month': pd.Timestamp(k), value_col_name: v}
        for k, v in result.items()
    ])


def fetch_macro_indicators():
    """Fetch monthly macro indicators from ECB, cached after first call."""
    global _macro_cache
    if _macro_cache is not None:
        return _macro_cache

    indicators = {
        'belgian_cpi_index': 'https://data-api.ecb.europa.eu/service/data/ICP/M.BE.N.000000.4.INX?startPeriod=2024-01&endPeriod=2026-12&format=csvdata',
        'belgian_10y_yield': 'https://data-api.ecb.europa.eu/service/data/IRS/M.BE.L.L40.CI.0000.EUR.N.Z?startPeriod=2024-01&endPeriod=2026-12&format=csvdata',
        'euribor_12m': 'https://data-api.ecb.europa.eu/service/data/FM/M.U2.EUR.RT.MM.EURIBOR1YD_.HSTA?startPeriod=2024-01&endPeriod=2026-12&format=csvdata',
    }
    merged = None
    for col_name, url in indicators.items():
        try:
            series = _fetch_ecb_series(url, col_name)
            if merged is None:
                merged = series
            else:
                merged = merged.merge(series, on='year_month', how='outer')
            print(f"  Fetched {len(series)} months of {col_name}")
        except Exception as e:
            print(f"  Could not fetch {col_name}: {e}")

    if merged is not None:
        merged = merged.sort_values('year_month').ffill()

    _macro_cache = merged
    return _macro_cache


# ============================================================================
# DESCRIPTION EMBEDDING
# ============================================================================

def embed_single_description(text):
    """Embed a single description text using E5 + PCA → 50 dimensions."""
    import torch

    if e5_model is None or pca_model is None:
        return np.zeros(PCA_COMPONENTS)

    # Clean description
    if not text or not isinstance(text, str):
        text = ''
    text = text.strip()
    lang_markers = re.compile(
        r'^(English|Français|Frans|French|Nederlands|Dutch|Deutsch|German)$',
        re.IGNORECASE,
    )
    if lang_markers.match(text):
        text = ''

    prefixed = f"passage: {text}" if text else "passage: "

    encoded = e5_tokenizer(
        [prefixed], max_length=512, padding=True, truncation=True,
        return_tensors='pt',
    ).to(e5_device)

    with torch.no_grad():
        outputs = e5_model(**encoded)

    attention_mask = encoded['attention_mask']
    token_embs = outputs.last_hidden_state
    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embs.size()).float()
    embedding = ((token_embs * mask_expanded).sum(1) /
                 mask_expanded.sum(1).clamp(min=1e-9))
    embedding = embedding.cpu().numpy()

    reduced = pca_model.transform(embedding)
    return reduced[0]


# ============================================================================
# BATCH EMBEDDING
# ============================================================================

def embed_batch_descriptions(texts):
    """Embed a list of description texts using the already-loaded E5 + PCA models.

    Returns an (N, 50) numpy array of PCA-reduced embeddings.
    """
    import torch

    if e5_model is None or pca_model is None:
        return np.zeros((len(texts), PCA_COMPONENTS))

    lang_markers = re.compile(
        r'^(English|Français|Frans|French|Nederlands|Dutch|Deutsch|German)$',
        re.IGNORECASE,
    )

    def _clean(t):
        if not t or not isinstance(t, str):
            return ''
        t = t.strip()
        return '' if lang_markers.match(t) else t

    cleaned = [_clean(t) for t in texts]
    prefixed = [f"passage: {t}" if t else "passage: " for t in cleaned]

    BATCH_SIZE = 32
    all_embeddings = []
    for i in range(0, len(prefixed), BATCH_SIZE):
        batch = prefixed[i:i + BATCH_SIZE]
        encoded = e5_tokenizer(
            batch, max_length=512, padding=True, truncation=True,
            return_tensors='pt',
        ).to(e5_device)

        with torch.no_grad():
            outputs = e5_model(**encoded)

        attention_mask = encoded['attention_mask']
        token_embs = outputs.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embs.size()).float()
        embeddings = ((token_embs * mask_expanded).sum(1) /
                      mask_expanded.sum(1).clamp(min=1e-9))
        all_embeddings.append(embeddings.cpu().numpy())

    raw = np.vstack(all_embeddings)
    return pca_model.transform(raw)


# ============================================================================
# DATA REFRESH PIPELINE
# ============================================================================

def find_latest_scrape_date():
    """Scan data/ for HTML files and find the latest scrape date.

    Returns (date_tag, iso_date_str, file_list) or (None, None, []).
    """
    date_files = {}  # date_tag -> list of file paths
    for dirpath, _, filenames in os.walk(DATA_DIR):
        for fn in filenames:
            if fn.startswith('full_source_') and fn.endswith('.html'):
                tag = fn[12:-5]  # e.g. '24022026'
                if len(tag) == 8 and tag.isdigit():
                    date_files.setdefault(tag, []).append(
                        os.path.join(dirpath, fn))

    if not date_files:
        return None, None, []

    # Try today first
    today_tag = datetime.now().strftime('%d%m%Y')
    if today_tag in date_files:
        day, month, year = today_tag[:2], today_tag[2:4], today_tag[4:]
        iso = f"{year}-{month}-{day}"
        return today_tag, iso, date_files[today_tag]

    # Fall back to most recent date
    def tag_to_date(tag):
        try:
            return datetime.strptime(tag, '%d%m%Y')
        except ValueError:
            return datetime.min

    latest_tag = max(date_files.keys(), key=tag_to_date)
    day, month, year = latest_tag[:2], latest_tag[2:4], latest_tag[4:]
    iso = f"{year}-{month}-{day}"
    return latest_tag, iso, date_files[latest_tag]


def _lightweight_preprocess_new_rows(parsed_records):
    """Preprocess newly parsed records without heavy OSM/Poisson steps.

    Produces a DataFrame with the same schema as property_data_with_embeddings.csv
    (minus desc_emb_* columns, which are added later).
    """
    from preprocess_functions import (
        handle_dates, calculate_listing_duration, handle_duplicates,
        filter_non_modelable, filter_hard_bounds, handle_age_and_year_built,
        create_price_per_sqm, scale_price, convert_boolean_columns,
        handle_missing_values, drop_dead_columns, encode_categorical_variables,
        add_neighborhood_information, add_distance_from_center_feature,
    )

    df = pd.DataFrame(parsed_records)
    if df.empty:
        return df

    print(f"  Lightweight preprocess: {len(df)} rows")

    # --- Stage 1: Cleaning ---
    df = handle_dates(df)
    if len(df) > 1:
        df = calculate_listing_duration(df)
    else:
        df['listing_duration_days'] = 0
    if len(df) > 1:
        df = handle_duplicates(df, keep='last')
    if len(df) > 1:
        df = filter_non_modelable(df)
    df = filter_hard_bounds(df)

    if df.empty:
        return df

    # --- Stage 2: Feature engineering ---
    df = handle_age_and_year_built(df)
    df = create_price_per_sqm(df)
    df = scale_price(df)
    df = convert_boolean_columns(df)
    df = handle_missing_values(df)
    df = drop_dead_columns(df)
    df = encode_categorical_variables(df)
    df['postal_code'] = df['postal_code'].astype(str)

    # --- Stage 3: Spatial features (reuse full neighbourhood but skip OSM) ---
    if len(df) > 1:
        # Poisson imputation for count columns
        from preprocess_functions import fill_missing_values_in_numerical_columns
        try:
            df = fill_missing_values_in_numerical_columns(df)
        except Exception as e:
            print(f"  Poisson imputation failed (non-critical): {e}")

    df = add_neighborhood_information(df)

    # Lightweight street type: use address-based inference instead of OSM graph
    df['street_type'] = df.apply(
        lambda row: get_street_type(row.get('street', '')), axis=1)
    df['distance_to_street'] = 0

    df = add_distance_from_center_feature(df)

    # log_price (needed for schema match)
    df['log_price'] = np.log(df['price'].clip(lower=0.001))

    print(f"  Lightweight preprocess done: {len(df)} rows, {len(df.columns)} cols")
    return df


def refresh_pipeline(progress=gr.Progress(track_tqdm=True)):
    """Refresh analytics with the latest scraped data.

    Parses new HTML files, preprocesses, embeds, predicts, and updates caches.
    Returns a status string.
    """
    global _analytics_df, _neighbourhood_summary, _today_deals_df

    from parser import extract_info

    steps = []

    # 1. Find latest scrape date
    date_tag, iso_date, html_files = find_latest_scrape_date()
    if date_tag is None:
        return "No HTML files found in data/ directory."

    is_today = date_tag == datetime.now().strftime('%d%m%Y')
    date_label = f"today ({iso_date})" if is_today else f"{iso_date} (most recent)"
    steps.append(f"Found {len(html_files)} HTML files from {date_label}")

    # 2. Parse HTML files in parallel
    parsed_records = []
    descriptions = {}  # property_id -> description
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(extract_info, fp): fp for fp in html_files}
        for future in as_completed(futures):
            try:
                result = future.result()
                if result:
                    # Stash description before preprocessing drops it
                    pid = str(result.get('property_id', ''))
                    desc = result.get('description', '')
                    if pid:
                        descriptions[pid] = desc
                    parsed_records.append(result)
            except Exception:
                pass

    steps.append(f"Parsed {len(parsed_records)} properties")
    if not parsed_records:
        return "\n".join(steps) + "\nNo properties could be parsed."

    # 3. Preprocess new rows
    df_new = _lightweight_preprocess_new_rows(parsed_records)
    steps.append(f"Preprocessed: {len(df_new)} rows after filtering")
    if df_new.empty:
        return "\n".join(steps) + "\nNo valid properties after preprocessing."

    # 4. Embed descriptions for new rows
    df_new['property_id'] = df_new['property_id'].astype(str)
    desc_texts = [descriptions.get(pid, '') for pid in df_new['property_id']]
    embeddings = embed_batch_descriptions(desc_texts)
    for i in range(PCA_COMPONENTS):
        df_new[f'desc_emb_{i}'] = embeddings[:, i]
    df_new['has_meaningful_description'] = [
        1 if d and len(str(d)) > 50 else 0 for d in desc_texts]
    steps.append(f"Embedded {len(desc_texts)} descriptions")

    # 5. Load existing base dataset and merge
    if os.path.exists(EMBEDDINGS_CSV_PATH):
        df_base = pd.read_csv(EMBEDDINGS_CSV_PATH)
        df_base['property_id'] = df_base['property_id'].astype(str)
        # Remove properties that are in the new set (to replace them)
        new_ids = set(df_new['property_id'].tolist())
        df_base = df_base[~df_base['property_id'].isin(new_ids)]
        # Align columns: add any missing columns to df_new
        for col in df_base.columns:
            if col not in df_new.columns:
                df_new[col] = np.nan
        df_combined = pd.concat(
            [df_base, df_new[df_base.columns]], ignore_index=True)
        steps.append(
            f"Combined: {len(df_base)} existing + {len(df_new)} new = "
            f"{len(df_combined)} total")
    else:
        df_combined = df_new
        steps.append(f"No existing base — using {len(df_new)} new rows only")

    # 6. Run predictions via compute_neighbourhood_analytics() logic
    for col in ['price', 'listing_duration_days', 'livable_area']:
        df_combined[col] = pd.to_numeric(df_combined[col], errors='coerce')

    df_combined['price_per_sqm'] = np.where(
        df_combined['livable_area'] > 0,
        df_combined['price'] * 1000 / df_combined['livable_area'],
        np.nan,
    )

    if ag_predictor is not None:
        try:
            macro = fetch_macro_indicators()
            if macro is not None and len(macro) > 0:
                latest = macro.iloc[-1]
                for col in ['belgian_cpi_index', 'belgian_10y_yield', 'euribor_12m']:
                    df_combined[col] = latest.get(col)
            else:
                for col in ['belgian_cpi_index', 'belgian_10y_yield', 'euribor_12m']:
                    df_combined[col] = np.nan

            df_combined['predicted_k'] = ag_predictor.predict(df_combined)
            df_combined['delta_eur'] = (
                df_combined['predicted_k'] - df_combined['price']) * 1000
            df_combined['delta_pct'] = (
                df_combined['delta_eur'] / (df_combined['price'] * 1000) * 100)
            steps.append("Predictions complete")
        except Exception as e:
            steps.append(f"Prediction error: {e}")
            df_combined['predicted_k'] = np.nan
            df_combined['delta_eur'] = np.nan
            df_combined['delta_pct'] = np.nan
    else:
        df_combined['predicted_k'] = np.nan
        df_combined['delta_eur'] = np.nan
        df_combined['delta_pct'] = np.nan
        steps.append("AutoGluon not available — skipped predictions")

    # Pricing bucket
    df_combined['pricing_bucket'] = pd.cut(
        df_combined['delta_pct'],
        bins=[-np.inf, -10, 10, np.inf],
        labels=['Overpriced', 'Fair', 'Underpriced'],
    )
    df_combined['property_type_label'] = np.where(
        df_combined.get('property_type_HOUSE', 0) == 1, 'House', 'Apartment')
    df_combined['is_under_option'] = (
        df_combined['is_under_option'].fillna(False).astype(bool))

    # Neighbourhood summary
    grouped = df_combined.groupby('neighbourhood')
    summary = grouped.agg(
        count=('price', 'size'),
        median_price_k=('price', 'median'),
        median_price_sqm=('price_per_sqm', 'median'),
        avg_listing_duration=('listing_duration_days', 'mean'),
        median_listing_duration=('listing_duration_days', 'median'),
        pct_under_option=('is_under_option', 'mean'),
    ).reset_index()

    has_delta = df_combined['delta_pct'].notna()
    if has_delta.any():
        delta_stats = df_combined[has_delta].groupby('neighbourhood').agg(
            median_delta_pct=('delta_pct', 'median'),
            pct_underpriced=('delta_pct', lambda x: (x > 0).mean()),
        ).reset_index()
        summary = summary.merge(delta_stats, on='neighbourhood', how='left')
    else:
        summary['median_delta_pct'] = np.nan
        summary['pct_underpriced'] = np.nan

    summary['pct_under_option'] = (summary['pct_under_option'] * 100).round(1)
    summary['pct_underpriced'] = (summary['pct_underpriced'] * 100).round(1)
    summary['avg_listing_duration'] = summary['avg_listing_duration'].round(1)
    summary['median_listing_duration'] = summary['median_listing_duration'].round(0)
    summary['median_price_sqm'] = summary['median_price_sqm'].round(0)
    summary['median_delta_pct'] = summary['median_delta_pct'].round(1)
    summary = summary.sort_values(
        'median_delta_pct', ascending=False, na_position='last')

    # 7. Build today_deals from the new rows
    today_deals = df_new.copy()
    if 'predicted_k' in df_combined.columns:
        # Get predictions for just the new rows
        new_idx = df_combined['property_id'].isin(
            set(df_new['property_id'].tolist()))
        today_cols = df_combined[new_idx][[
            'property_id', 'predicted_k', 'delta_eur', 'delta_pct']].copy()
        today_deals = df_new.merge(
            today_cols, on='property_id', how='left', suffixes=('', '_pred'))

    # 8. Update module-level caches
    _analytics_df = df_combined
    _neighbourhood_summary = summary
    _today_deals_df = today_deals

    steps.append(
        f"Done: {len(df_combined)} properties, "
        f"{len(summary)} neighbourhoods")

    return "\n".join(steps)


# ============================================================================
# URL FETCH & PARSE
# ============================================================================

def fetch_and_parse_url(url):
    """Fetch an Immoweb listing page and extract property data."""
    if not url or 'immoweb.be' not in url:
        raise ValueError("Please enter a valid Immoweb URL")

    headers = {
        'User-Agent': ('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                       'AppleWebKit/537.36 (KHTML, like Gecko) '
                       'Chrome/120.0.0.0 Safari/537.36'),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9,nl;q=0.8,fr;q=0.7',
    }

    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()

    html = response.text
    json_match = re.search(
        r'window\.classified\s*=\s*(\{.*?\});', html, re.DOTALL)
    if not json_match:
        raise ValueError(
            "Could not find property data on this page. "
            "The page might be blocked or the URL might be incorrect.")

    data = json.loads(json_match.group(1))

    # Navigate JSON sections (same as parser.py)
    prop = data.get('property', {})
    price_data = data.get('price', {})
    txn = data.get('transaction', {})
    sale = txn.get('sale', {}) or {}
    certs = txn.get('certificates', {}) or {}
    pub = data.get('publication', {}) or {}
    flags = data.get('flags', {}) or {}
    customers = data.get('customers', [])

    loc = prop.get('location', {}) or {}
    bld = prop.get('building', {}) or {}
    energy = prop.get('energy', {}) or {}
    land = prop.get('land', {}) or {}
    cperm = prop.get('constructionPermit', {}) or {}
    kitch = prop.get('kitchen', {}) or {}
    livr = prop.get('livingRoom', {}) or {}

    # Extract property_id from URL
    url_parts = url.rstrip('/').split('/')
    property_id = url_parts[-1] if url_parts[-1].isdigit() else None

    return {
        'property_id': property_id,
        'date_obtained': datetime.now().strftime('%Y-%m-%d'),
        'price': price_data.get('mainValue', price_data.get('price')),
        'property_type': prop.get('type', data.get('type')),
        'property_subtype': prop.get('subtype'),
        'postal_code': loc.get('postalCode'),
        'city': loc.get('locality'),
        'street': loc.get('street'),
        'house_number': loc.get('number'),
        'box': loc.get('box'),
        'latitude': loc.get('latitude'),
        'longitude': loc.get('longitude'),
        'location_type': loc.get('type'),
        'floor_number': loc.get('floor'),
        'year_built': bld.get('constructionYear'),
        'building_condition': bld.get('condition'),
        'total_floors': bld.get('floorCount'),
        'facade_count': bld.get('facadeCount'),
        'street_facade_width': bld.get('streetFacadeWidth'),
        'livable_area': prop.get('netHabitableSurface', prop.get('indoor_surface')),
        'property_size': land.get('surface'),
        'kitchen_surface': kitch.get('surface'),
        'living_room_surface': livr.get('surface'),
        'number_of_bedrooms': prop.get('bedroomCount'),
        'number_of_bathrooms': prop.get('bathroomCount'),
        'number_of_toilets': prop.get('toiletCount'),
        'shower_room_count': prop.get('showerRoomCount'),
        'has_living_room': prop.get('hasLivingRoom'),
        'has_attic': prop.get('hasAttic'),
        'has_basement': prop.get('hasBasement'),
        'has_terrace': prop.get('hasTerrace'),
        'terrace_surface': prop.get('terraceSurface'),
        'has_garden': prop.get('hasGarden'),
        'garden_surface': prop.get('gardenSurface'),
        'has_swimming_pool': prop.get('hasSwimmingPool'),
        'has_lift': prop.get('hasLift'),
        'has_dressing_room': prop.get('hasDressingRoom'),
        'has_air_conditioning': prop.get('hasAirConditioning'),
        'has_caretaker': prop.get('hasCaretakerOrConcierge'),
        'kitchen_type': kitch.get('type'),
        'monthly_costs': prop.get('monthlyCosts'),
        'parking_indoor': prop.get('parkingCountIndoor'),
        'parking_outdoor': prop.get('parkingCountOutdoor'),
        'parking_closed_box': prop.get('parkingCountClosedBox'),
        'energy_label': certs.get('epcScore'),
        'primary_energy_consumption': certs.get('primaryEnergyConsumptionPerSqm'),
        'heating_type': energy.get('heatingType'),
        'has_photovoltaic_panels': energy.get('hasPhotovoltaicPanels'),
        'has_double_glazing': energy.get('hasDoubleGlazing'),
        'has_heat_pump': energy.get('hasHeatPump'),
        'has_thermic_panels': energy.get('hasThermicPanels'),
        'flood_zone_type': cperm.get('floodZoneType'),
        'p_score': cperm.get('pScore'),
        'g_score': cperm.get('gScore'),
        'cadastral_income': sale.get('cadastralIncome'),
        'is_furnished': sale.get('isFurnished'),
        'renovation_obligation': certs.get('renovationObligation'),
        'publication_date': pub.get('creationDate'),
        'last_modification_date': pub.get('lastModificationDate'),
        'immo_office_name': customers[0].get('name') if customers else None,
        'is_biddit_sale': flags.get('isAnInteractiveSale'),
        'is_under_option': flags.get('isUnderOption'),
        'is_newly_built': flags.get('isNewlyBuilt'),
        'description': (
            prop.get('description')
            or (prop.get('alternativeDescriptions') or {}).get('en')
            or (prop.get('alternativeDescriptions') or {}).get('nl')
            or (prop.get('alternativeDescriptions') or {}).get('fr')
        ),
    }


# ============================================================================
# SINGLE PROPERTY PREPARATION (for AutoGluon)
# ============================================================================

def prepare_single_property(parsed):
    """Transform a parsed property dict into an AutoGluon-ready feature row."""
    row = {}
    asking_price_eur = parsed.get('price')

    # --- Copy direct fields ---
    direct_fields = [
        'property_subtype', 'latitude', 'longitude',
        'floor_number', 'year_built', 'total_floors', 'facade_count',
        'street_facade_width', 'livable_area', 'property_size',
        'kitchen_surface', 'living_room_surface', 'number_of_bedrooms',
        'number_of_bathrooms', 'number_of_toilets', 'shower_room_count',
        'terrace_surface', 'garden_surface', 'monthly_costs',
        'parking_indoor', 'parking_outdoor', 'primary_energy_consumption',
        'p_score', 'g_score', 'cadastral_income',
    ]
    for f in direct_fields:
        row[f] = parsed.get(f)

    # --- Categorical fields (NaN → 'Unknown') ---
    cat_fields = {
        'building_condition': parsed.get('building_condition'),
        'kitchen_type': parsed.get('kitchen_type'),
        'energy_label': parsed.get('energy_label'),
        'heating_type': parsed.get('heating_type'),
        'flood_zone_type': parsed.get('flood_zone_type'),
        'renovation_obligation': parsed.get('renovation_obligation'),
        'location_type': parsed.get('location_type'),
    }
    for k, v in cat_fields.items():
        row[k] = v if v is not None else 'Unknown'

    # --- Boolean fields → int (NaN → 0) ---
    bool_fields = [
        'has_living_room', 'has_attic', 'has_basement', 'has_terrace',
        'has_garden', 'has_swimming_pool', 'has_lift', 'has_dressing_room',
        'has_air_conditioning', 'has_caretaker', 'has_photovoltaic_panels',
        'has_double_glazing', 'has_heat_pump', 'has_thermic_panels',
        'is_furnished', 'is_under_option', 'is_newly_built',
    ]
    for f in bool_fields:
        val = parsed.get(f)
        row[f] = 1 if (val is True or val == 'True' or val == 1) else 0

    # --- Surface NaN → 0 ---
    for f in ['garden_surface', 'terrace_surface']:
        if row.get(f) is None:
            row[f] = 0

    # --- Derived: age ---
    current_year = datetime.now().year
    yb = row.get('year_built')
    row['age'] = (current_year - yb) if yb else None

    # --- Derived: property_type_HOUSE ---
    row['property_type_HOUSE'] = 1 if parsed.get('property_type') == 'HOUSE' else 0

    # --- Derived: dist_from_center (same formula as preprocess_functions.py) ---
    lat = row.get('latitude')
    lon = row.get('longitude')
    if lat and lon:
        row['dist_from_center'] = calculate_dist_from_center_ag(lat, lon)
    else:
        row['dist_from_center'] = None

    # --- Derived: neighbourhood (spatial lookup, normalized) ---
    if lat and lon:
        nbh = get_neighbourhood_spatial(lat, lon)
        row['neighbourhood'] = (
            nbh.replace(' ', '').replace('-', '_') if nbh else 'Unknown')
    else:
        row['neighbourhood'] = 'Unknown'

    # --- Derived: street_type (address-based inference → OSM-like types) ---
    row['street_type'] = get_street_type(parsed.get('street', ''))

    # --- Derived: distance_to_street (no OSM graph available) ---
    row['distance_to_street'] = 0

    # --- Derived: listing_duration_days ---
    pub_date = parsed.get('publication_date')
    mod_date = parsed.get('last_modification_date')
    if pub_date and mod_date:
        try:
            pub_dt = pd.to_datetime(pub_date)
            mod_dt = pd.to_datetime(mod_date)
            row['listing_duration_days'] = max(0, (mod_dt - pub_dt).days)
        except Exception:
            row['listing_duration_days'] = 0
    else:
        row['listing_duration_days'] = 0

    # --- postal_code as string ---
    pc = parsed.get('postal_code')
    row['postal_code'] = str(int(pc)) if pc else None

    # --- Description embedding → 50 PCA components ---
    description = parsed.get('description', '')
    emb = embed_single_description(description)
    for i in range(PCA_COMPONENTS):
        row[f'desc_emb_{i}'] = emb[i]

    row['has_meaningful_description'] = (
        1 if description and len(str(description)) > 50 else 0)

    # --- Macro indicators ---
    try:
        macro = fetch_macro_indicators()
        if macro is not None and len(macro) > 0:
            latest = macro.iloc[-1]
            for col in ['belgian_cpi_index', 'belgian_10y_yield', 'euribor_12m']:
                row[col] = latest.get(col)
        else:
            for col in ['belgian_cpi_index', 'belgian_10y_yield', 'euribor_12m']:
                row[col] = None
    except Exception:
        for col in ['belgian_cpi_index', 'belgian_10y_yield', 'euribor_12m']:
            row[col] = None

    df = pd.DataFrame([row])
    return df, asking_price_eur


# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================

def analyze_url(url):
    """Full pipeline: fetch URL → parse → prepare → predict → auto-fill form."""
    n_form_fields = 31
    no_update = [gr.update()] * n_form_fields

    if not url or 'immoweb.be' not in url:
        return ["Please enter a valid Immoweb URL (e.g. https://www.immoweb.be/en/classified/house/for-sale/gent/9000/12345678)"] + no_update

    if ag_predictor is None:
        return ["Error: AutoGluon model not loaded. Cannot predict from URL."] + no_update

    try:
        # Step 1: Fetch and parse
        parsed = fetch_and_parse_url(url)

        # Step 2: Prepare features
        features_df, asking_price = prepare_single_property(parsed)

        # Step 3: Predict
        predicted_k = ag_predictor.predict(features_df).values[0]
        predicted_eur = predicted_k * 1000

        # Step 4: Format result
        lines = []
        if asking_price:
            delta_eur = predicted_eur - asking_price
            delta_pct = delta_eur / asking_price * 100
            lines.append(f"Asking Price:    EUR {asking_price:,.0f}")
            lines.append(f"Predicted Price: EUR {predicted_eur:,.0f}")
            lines.append(f"Difference:      EUR {delta_eur:+,.0f} ({delta_pct:+.1f}%)")
            lines.append("")
            if delta_eur > 0:
                lines.append(">> Model thinks this property is worth MORE than asking price (potential deal)")
            else:
                lines.append(">> Model thinks this property is OVERPRICED")
        else:
            lines.append(f"Predicted Price: EUR {predicted_eur:,.0f}")
            lines.append("(No asking price found on listing)")

        lines.append("")
        lines.append("--- Property Details ---")
        ptype = parsed.get('property_type', '?')
        psub = parsed.get('property_subtype', '?')
        lines.append(f"Type: {ptype} / {psub}")
        street = parsed.get('street', '?')
        hnum = parsed.get('house_number', '')
        pc = parsed.get('postal_code', '?')
        city = parsed.get('city', '?')
        lines.append(f"Location: {street} {hnum}, {pc} {city}")
        lines.append(f"Area: {parsed.get('livable_area', '?')} m2")
        lines.append(f"Bedrooms: {parsed.get('number_of_bedrooms', '?')}")
        lines.append(f"Year built: {parsed.get('year_built', '?')}")
        lines.append(f"Condition: {parsed.get('building_condition', '?')}")
        lines.append(f"Energy: {parsed.get('energy_label', '?')} ({parsed.get('primary_energy_consumption', '?')} kWh/m2)")

        # Neighbourhood from spatial lookup
        lat = parsed.get('latitude')
        lon = parsed.get('longitude')
        if lat and lon:
            nbh_raw = get_neighbourhood_spatial(lat, lon)
            if nbh_raw:
                lines.append(f"Neighbourhood: {nbh_raw}")

        desc = parsed.get('description', '')
        if desc and len(str(desc)) > 100:
            lines.append(f"Description: {str(desc)[:150]}...")
        elif desc:
            lines.append(f"Description: {desc}")

        result_text = "\n".join(lines)

        # Step 5: Auto-fill form fields
        def val_or_skip(v):
            return v if v is not None else gr.update()

        def dropdown_val(v, key):
            choices = options.get(key, [])
            if v and v in choices:
                return v
            return gr.update()

        # Normalize neighbourhood for dropdown
        nbh_normalized = None
        if lat and lon:
            nbh_raw = get_neighbourhood_spatial(lat, lon)
            if nbh_raw:
                nbh_normalized = nbh_raw.replace(' ', '').replace('-', '_')

        form_updates = [
            val_or_skip(parsed.get('latitude')),                    # lat_input
            val_or_skip(parsed.get('longitude')),                   # lon_input
            dropdown_val(nbh_normalized, 'neighbourhood'),          # nbr_input
            dropdown_val(get_street_type(parsed.get('street', '')),
                         'street_type'),                            # street_type_input
            dropdown_val(parsed.get('flood_zone_type'),
                         'flood_zone_type'),                        # flood_input
            val_or_skip(parsed.get('livable_area')),                # livable_area
            val_or_skip(parsed.get('year_built')),                  # year_built
            val_or_skip(parsed.get('number_of_bedrooms')),          # bedrooms
            val_or_skip(parsed.get('number_of_bathrooms')),         # bathrooms
            dropdown_val(parsed.get('property_subtype'),
                         'property_subtype'),                       # subtype
            dropdown_val(parsed.get('building_condition'),
                         'building_condition'),                     # condition
            val_or_skip(parsed.get('number_of_toilets')),           # toilets
            val_or_skip(parsed.get('primary_energy_consumption')),  # epc
            val_or_skip(parsed.get('cadastral_income')),            # cad_income
            dropdown_val(parsed.get('kitchen_type'), 'kitchen_type'),  # kitchen
            dropdown_val(parsed.get('energy_label'), 'energy_label'), # energy
            dropdown_val(parsed.get('heating_type'), 'heating_type'), # heating
            parsed.get('terrace_surface') or 0,                     # terrace_surf
            parsed.get('garden_surface') or 0,                      # garden_surf
            parsed.get('parking_indoor') or 0,                      # parking_in
            parsed.get('parking_outdoor') or 0,                     # parking_out
            parsed.get('parking_closed_box') or 0,                  # parking_box
            bool(parsed.get('has_photovoltaic_panels')),            # solar
            bool(parsed.get('has_double_glazing')),                 # double_glaz
            bool(parsed.get('has_living_room')),                    # living
            bool(parsed.get('has_attic')),                          # attic
            bool(parsed.get('has_basement')),                       # basement
            bool(parsed.get('has_terrace')),                        # terrace
            bool(parsed.get('has_garden')),                         # garden
            bool(parsed.get('has_swimming_pool')),                  # pool
            bool(parsed.get('is_biddit_sale')),                     # biddit
        ]

        return [result_text] + form_updates

    except requests.exceptions.HTTPError as e:
        if hasattr(e, 'response') and e.response is not None and e.response.status_code == 403:
            return ["Error: Access blocked by Immoweb (403 Forbidden). "
                    "The site's bot protection may be active. Try again later."] + no_update
        return [f"Error fetching page: {e}"] + no_update
    except ValueError as e:
        return [f"Error: {e}"] + no_update
    except Exception as e:
        return [f"Error: {type(e).__name__}: {e}"] + no_update


def make_prediction(
    livable_area, bedrooms, bathrooms, toilets, terrace_surf, garden_surf,
    parking_in, parking_out, parking_box, epc, lat, lon, year_built, cad_income,
    subtype, condition, kitchen, energy, heating, flood, neighbourhood, street_type,
    solar, double_glaz, living, attic, basement, terrace, garden, pool, biddit
):
    """Manual form prediction (uses old XGBoost model)."""
    input_dict = {
        'livable_area': livable_area,
        'number_of_bedrooms': bedrooms,
        'number_of_bathrooms': bathrooms,
        'number_of_toilets': toilets,
        'terrace_surface': terrace_surf,
        'garden_surface': garden_surf,
        'parking_indoor': parking_in,
        'parking_outdoor': parking_out,
        'parking_closed_box': parking_box,
        'primary_energy_consumption': epc,
        'latitude': lat,
        'longitude': lon,
        'year_built': year_built,
        'cadastral_income': cad_income,
        'property_subtype': subtype,
        'building_condition': condition,
        'kitchen_type': kitchen,
        'energy_label': energy,
        'heating_type': heating,
        'flood_zone_type': flood,
        'neighbourhood': neighbourhood,
        'street_type': street_type,
        'has_photovoltaic_panels': int(solar),
        'has_double_glazing': int(double_glaz),
        'has_living_room': int(living),
        'has_attic': int(attic),
        'has_basement': int(basement),
        'has_terrace': int(terrace),
        'has_garden': int(garden),
        'has_swimming_pool': int(pool),
        'is_biddit_sale': int(biddit),
        'dist_from_center': calculate_dist_from_center(lat, lon),
        'distance_to_street': defaults.get('distance_to_street', 0),
    }

    df_input = pd.DataFrame([input_dict])

    # Ensure all expected columns exist
    for col in feature_cols:
        if col not in df_input.columns:
            df_input[col] = 0

    if old_model is not None:
        try:
            log_pred = old_model.predict(df_input)[0]
            price = np.exp(log_pred)
            return f"EUR {price:,.2f}"
        except Exception as e:
            return f"Error: {str(e)}"
    else:
        return "Error: No model loaded for manual prediction"


# ============================================================================
# NEIGHBOURHOOD ANALYTICS
# ============================================================================

# Module-level caches for pre-computed analytics data
_analytics_df = None          # Full dataset with predictions
_neighbourhood_summary = None # Per-neighbourhood aggregated stats
_today_deals_df = None        # Current listings with predictions


def compute_neighbourhood_analytics():
    """Load full dataset, run AutoGluon predictions, compute neighbourhood stats.

    Returns (full_df, neighbourhood_summary) or (None, None) on failure.
    """
    if not os.path.exists(EMBEDDINGS_CSV_PATH):
        print(f"Embeddings CSV not found at {EMBEDDINGS_CSV_PATH}")
        return None, None

    print("Loading property dataset for neighbourhood analytics...")
    df = pd.read_csv(EMBEDDINGS_CSV_PATH)

    # Ensure numeric types
    for col in ['price', 'listing_duration_days', 'livable_area']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Compute price per m²
    df['price_per_sqm'] = np.where(
        df['livable_area'] > 0,
        df['price'] * 1000 / df['livable_area'],
        np.nan,
    )

    # Run AutoGluon batch prediction if available
    if ag_predictor is not None:
        print("Running batch predictions on full dataset...")
        try:
            # Add macro indicator columns required by AutoGluon
            macro = fetch_macro_indicators()
            if macro is not None and len(macro) > 0:
                latest = macro.iloc[-1]
                for col in ['belgian_cpi_index', 'belgian_10y_yield', 'euribor_12m']:
                    df[col] = latest.get(col)
            else:
                for col in ['belgian_cpi_index', 'belgian_10y_yield', 'euribor_12m']:
                    df[col] = np.nan

            df['predicted_k'] = ag_predictor.predict(df)
            df['delta_eur'] = (df['predicted_k'] - df['price']) * 1000
            df['delta_pct'] = df['delta_eur'] / (df['price'] * 1000) * 100
        except Exception as e:
            print(f"Batch prediction failed: {e}")
            df['predicted_k'] = np.nan
            df['delta_eur'] = np.nan
            df['delta_pct'] = np.nan
    else:
        print("AutoGluon not available — skipping delta computation")
        df['predicted_k'] = np.nan
        df['delta_eur'] = np.nan
        df['delta_pct'] = np.nan

    # Pricing bucket: Underpriced (delta > 10%), Fair (±10%), Overpriced (delta < -10%)
    df['pricing_bucket'] = pd.cut(
        df['delta_pct'],
        bins=[-np.inf, -10, 10, np.inf],
        labels=['Overpriced', 'Fair', 'Underpriced'],
    )

    # Property type label
    df['property_type_label'] = np.where(
        df['property_type_HOUSE'] == 1, 'House', 'Apartment')

    # Boolean coercion for is_under_option
    df['is_under_option'] = df['is_under_option'].fillna(False).astype(bool)

    # Neighbourhood aggregation
    has_delta = df['delta_pct'].notna()
    grouped = df.groupby('neighbourhood')
    summary = grouped.agg(
        count=('price', 'size'),
        median_price_k=('price', 'median'),
        median_price_sqm=('price_per_sqm', 'median'),
        avg_listing_duration=('listing_duration_days', 'mean'),
        median_listing_duration=('listing_duration_days', 'median'),
        pct_under_option=('is_under_option', 'mean'),
    ).reset_index()

    # Delta stats only where predictions exist
    if has_delta.any():
        delta_stats = df[has_delta].groupby('neighbourhood').agg(
            median_delta_pct=('delta_pct', 'median'),
            pct_underpriced=('delta_pct', lambda x: (x > 0).mean()),
        ).reset_index()
        summary = summary.merge(delta_stats, on='neighbourhood', how='left')
    else:
        summary['median_delta_pct'] = np.nan
        summary['pct_underpriced'] = np.nan

    # Format percentages
    summary['pct_under_option'] = (summary['pct_under_option'] * 100).round(1)
    summary['pct_underpriced'] = (summary['pct_underpriced'] * 100).round(1)
    summary['avg_listing_duration'] = summary['avg_listing_duration'].round(1)
    summary['median_listing_duration'] = summary['median_listing_duration'].round(0)
    summary['median_price_sqm'] = summary['median_price_sqm'].round(0)
    summary['median_delta_pct'] = summary['median_delta_pct'].round(1)

    # Sort by median delta descending (best deals first)
    summary = summary.sort_values('median_delta_pct', ascending=False, na_position='last')

    print(f"Analytics ready: {len(df)} properties across {len(summary)} neighbourhoods")
    return df, summary


def _get_polygon_geojson(neighbourhood_name):
    """Return the GeoJSON dict for a neighbourhood polygon, matching by name."""
    for poly, name in neighbourhood_polygons:
        # Match normalized names (spaces removed, dashes to underscores)
        norm_name = name.replace(' ', '').replace('-', '_')
        if norm_name == neighbourhood_name or name == neighbourhood_name:
            return json.loads(json.dumps(poly.__geo_interface__))
    return None


def build_choropleth_map(neighbourhood_summary, metric):
    """Build a Folium choropleth map colored by the selected metric.

    Returns an HTML string suitable for iframe embedding.
    """
    metric_config = {
        'Median Price/m²': ('median_price_sqm', '€/m²', 'YlOrRd'),
        'Avg Listing Duration (days)': ('avg_listing_duration', 'days', 'YlGnBu'),
        'Number of Listings': ('count', 'listings', 'PuBu'),
        '% Underpriced': ('pct_underpriced', '%', 'RdYlGn'),
        'Median Delta %': ('median_delta_pct', '%', 'RdYlGn'),
    }

    col, unit, palette = metric_config.get(
        metric, ('median_price_sqm', '€/m²', 'YlOrRd'))

    m = folium.Map(location=[51.054, 3.725], zoom_start=13, tiles='cartodbpositron')

    values = neighbourhood_summary[col].dropna()
    if len(values) == 0:
        return '<div style="padding:50px;text-align:center;">No data available for this metric.</div>'

    vmin, vmax = values.min(), values.max()
    if vmin == vmax:
        vmax = vmin + 1

    # Choose colors based on palette name
    palette_colors = {
        'YlOrRd': ['#ffffb2', '#fecc5c', '#fd8d3c', '#f03b20', '#bd0026'],
        'YlGnBu': ['#ffffcc', '#a1dab4', '#41b6c4', '#2c7fb8', '#253494'],
        'PuBu': ['#f1eef6', '#bdc9e1', '#74a9cf', '#2b8cbe', '#045a8d'],
        'RdYlGn': ['#d73027', '#fc8d59', '#fee08b', '#d9ef8b', '#1a9850'],
    }
    colors = palette_colors.get(palette, palette_colors['YlOrRd'])
    colormap = cm.LinearColormap(
        colors=colors, vmin=vmin, vmax=vmax,
        caption=f'{metric} ({unit})')
    colormap.add_to(m)

    for _, row in neighbourhood_summary.iterrows():
        nbh = row['neighbourhood']
        val = row.get(col)
        if pd.isna(val):
            continue

        geojson = _get_polygon_geojson(nbh)
        if geojson is None:
            continue

        color = colormap(val)
        tooltip_text = f"<b>{nbh}</b><br>{metric}: {val:,.1f} {unit}"

        folium.GeoJson(
            geojson,
            style_function=lambda x, c=color: {
                'fillColor': c,
                'color': '#333',
                'weight': 1.5,
                'fillOpacity': 0.65,
            },
            tooltip=folium.Tooltip(tooltip_text),
        ).add_to(m)

    # Use get_root().render() for full standalone HTML (not Jupyter _repr_html_)
    html = m.get_root().render()
    html_escaped = html.replace('"', '&quot;')
    return f'<iframe srcdoc="{html_escaped}" width="100%" height="550px" style="border:none;"></iframe>'


def build_pricing_speed_analysis(df):
    """Build the 'Do underpriced properties sell faster?' visualizations.

    Returns (scatter_fig, bar_fig, summary_markdown).
    """
    # Filter to properties with valid delta and listing duration
    mask = df['delta_pct'].notna() & df['listing_duration_days'].notna()
    dfa = df[mask].copy()

    # Cap listing duration at 200 days to reduce outlier distortion
    dfa['duration_capped'] = dfa['listing_duration_days'].clip(upper=200)

    # Shared chart styling
    _chart_font = dict(family='Inter, system-ui, sans-serif', size=12, color='#334155')
    _chart_colors = {'House': '#3b82f6', 'Apartment': '#f59e0b'}
    _chart_layout = dict(
        template='plotly_white',
        font=_chart_font,
        title_font=dict(size=15, color='#1e293b'),
        paper_bgcolor='#ffffff',
        plot_bgcolor='#f8fafc',
        margin=dict(l=50, r=20, t=50, b=50),
        legend=dict(
            orientation='h', yanchor='bottom', y=1.02,
            xanchor='right', x=1, bgcolor='#ffffff',
        ),
    )

    # --- Scatter plot ---
    scatter_fig = px.scatter(
        dfa.sample(n=min(3000, len(dfa)), random_state=42),
        x='delta_pct', y='duration_capped',
        color='property_type_label',
        opacity=0.35,
        trendline='ols',
        labels={
            'delta_pct': 'Price Delta (%)',
            'duration_capped': 'Listing Duration (days)',
            'property_type_label': '',
        },
        title='Price Delta vs Listing Duration',
        color_discrete_map=_chart_colors,
    )
    scatter_fig.update_layout(height=420, **_chart_layout)
    scatter_fig.update_traces(marker=dict(size=5))
    # Style trendlines
    for trace in scatter_fig.data:
        if trace.mode == 'lines':
            trace.update(line=dict(width=2.5, dash='dot'))

    # --- Bar chart: avg duration by pricing bucket x property type ---
    bucket_stats = dfa.groupby(
        ['pricing_bucket', 'property_type_label'], observed=True,
    ).agg(
        avg_duration=('listing_duration_days', 'mean'),
        count=('listing_duration_days', 'size'),
    ).reset_index()

    bucket_order = ['Underpriced', 'Fair', 'Overpriced']
    bucket_stats['pricing_bucket'] = pd.Categorical(
        bucket_stats['pricing_bucket'], categories=bucket_order, ordered=True)
    bucket_stats = bucket_stats.sort_values('pricing_bucket')

    bar_fig = px.bar(
        bucket_stats,
        x='pricing_bucket', y='avg_duration',
        color='property_type_label',
        barmode='group',
        text=bucket_stats['avg_duration'].round(1),
        labels={
            'pricing_bucket': 'Pricing Bucket',
            'avg_duration': 'Avg Listing Duration (days)',
            'property_type_label': '',
        },
        title='Avg Listing Duration by Pricing Bucket',
        color_discrete_map=_chart_colors,
    )
    bar_fig.update_layout(height=380, **_chart_layout)
    bar_fig.update_traces(
        textposition='outside', textfont_size=13,
        marker_line_width=0, opacity=0.9,
    )

    # --- Summary statistics ---
    corr, pval = stats.spearmanr(
        dfa['delta_pct'], dfa['listing_duration_days'])

    under = dfa[dfa['pricing_bucket'] == 'Underpriced']['listing_duration_days']
    fair = dfa[dfa['pricing_bucket'] == 'Fair']['listing_duration_days']
    over = dfa[dfa['pricing_bucket'] == 'Overpriced']['listing_duration_days']

    # Under-option rates per bucket
    option_rates = dfa.groupby('pricing_bucket', observed=True)['is_under_option'].mean() * 100

    summary = f"""### Key Findings

- **Underpriced** properties (delta > +10%): median **{under.median():.0f}** days on market (n={len(under):,})
- **Fair-priced** properties (±10%): median **{fair.median():.0f}** days on market (n={len(fair):,})
- **Overpriced** properties (delta < −10%): median **{over.median():.0f}** days on market (n={len(over):,})
- Spearman correlation (delta vs duration): **r = {corr:.3f}** (p = {pval:.2e})

**Under-option rate** (proxy for sold):
"""
    for bucket in bucket_order:
        rate = option_rates.get(bucket, 0)
        summary += f"- {bucket}: {rate:.1f}%\n"

    return scatter_fig, bar_fig, summary


def build_neighbourhood_deepdive(df, today_deals_df, neighbourhood):
    """Build deep-dive content for a selected neighbourhood.

    Returns (stats_markdown, trend_fig, listings_df).
    """
    nbh_df = df[df['neighbourhood'] == neighbourhood]

    if len(nbh_df) == 0:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title='No data', height=300, template='plotly_white')
        return "No data for this neighbourhood.", empty_fig, pd.DataFrame()

    # --- Stats card ---
    med_price = nbh_df['price'].median()
    min_price = nbh_df['price'].min()
    max_price = nbh_df['price'].max()
    avg_area = nbh_df['livable_area'].mean()
    avg_age = nbh_df['age'].mean() if 'age' in nbh_df.columns else None
    count = len(nbh_df)

    # Most common building condition and energy label
    top_condition = nbh_df['building_condition'].mode().iloc[0] if 'building_condition' in nbh_df.columns and len(nbh_df['building_condition'].dropna()) > 0 else 'N/A'
    top_energy = nbh_df['energy_label'].mode().iloc[0] if 'energy_label' in nbh_df.columns and len(nbh_df['energy_label'].dropna()) > 0 else 'N/A'

    nbh_display = neighbourhood.replace('_', ' ')
    stats_md = f"""<div style="
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 100%);
        border-radius: 12px; padding: 24px 28px; color: white; margin-bottom: 16px;
    ">
        <h2 style="margin:0 0 16px 0; font-size:22px; font-weight:700; color:white;">{nbh_display}</h2>
        <div style="display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr)); gap:12px;">
            <div style="background:rgba(255,255,255,0.15); border-radius:8px; padding:14px; text-align:center;">
                <div style="font-size:24px; font-weight:700; color:white;">{count:,}</div>
                <div style="font-size:12px; color:#cbd5e1; margin-top:4px; font-weight:500;">Properties</div>
            </div>
            <div style="background:rgba(255,255,255,0.15); border-radius:8px; padding:14px; text-align:center;">
                <div style="font-size:24px; font-weight:700; color:white;">{med_price:,.0f}k</div>
                <div style="font-size:12px; color:#cbd5e1; margin-top:4px; font-weight:500;">Median Price</div>
            </div>
            <div style="background:rgba(255,255,255,0.15); border-radius:8px; padding:14px; text-align:center;">
                <div style="font-size:24px; font-weight:700; color:white;">{avg_area:,.0f} m&sup2;</div>
                <div style="font-size:12px; color:#cbd5e1; margin-top:4px; font-weight:500;">Avg Area</div>
            </div>
            <div style="background:rgba(255,255,255,0.15); border-radius:8px; padding:14px; text-align:center;">
                <div style="font-size:24px; font-weight:700; color:white;">{avg_age:,.0f} yr</div>
                <div style="font-size:12px; color:#cbd5e1; margin-top:4px; font-weight:500;">Avg Age</div>
            </div>
            <div style="background:rgba(255,255,255,0.15); border-radius:8px; padding:14px; text-align:center;">
                <div style="font-size:16px; font-weight:700; color:white;">{top_condition.replace('_',' ').title()}</div>
                <div style="font-size:12px; color:#cbd5e1; margin-top:4px; font-weight:500;">Top Condition</div>
            </div>
            <div style="background:rgba(255,255,255,0.15); border-radius:8px; padding:14px; text-align:center;">
                <div style="font-size:24px; font-weight:700; color:white;">{top_energy}</div>
                <div style="font-size:12px; color:#cbd5e1; margin-top:4px; font-weight:500;">Top Energy Label</div>
            </div>
        </div>
        <div style="margin-top:12px; font-size:13px; color:#94a3b8;">
            Price range: {min_price:,.0f}k&euro; &ndash; {max_price:,.0f}k&euro;
        </div>
    </div>"""

    # --- Monthly price trend: neighbourhood vs city-wide ---
    df_dated = df[df['date_obtained'].notna()].copy()
    df_dated['month'] = pd.to_datetime(df_dated['date_obtained']).dt.to_period('M')
    nbh_dated = df_dated[df_dated['neighbourhood'] == neighbourhood]

    city_trend = df_dated.groupby('month')['price'].median().reset_index()
    city_trend.columns = ['month', 'price']
    city_trend['label'] = 'City-wide'

    nbh_trend = nbh_dated.groupby('month')['price'].median().reset_index()
    nbh_trend.columns = ['month', 'price']
    nbh_trend['label'] = neighbourhood.replace('_', ' ')

    trend_data = pd.concat([city_trend, nbh_trend], ignore_index=True)
    trend_data['month_str'] = trend_data['month'].astype(str)

    trend_fig = px.line(
        trend_data, x='month_str', y='price', color='label',
        labels={'month_str': 'Month', 'price': 'Median Price (k€)', 'label': ''},
        title=f'Monthly Median Price: {neighbourhood.replace("_", " ")} vs City-wide',
        color_discrete_sequence=['#cbd5e1', '#3b82f6'],
    )
    _chart_font = dict(family='Inter, system-ui, sans-serif', size=12, color='#334155')
    trend_fig.update_layout(
        height=380, template='plotly_white',
        font=_chart_font,
        title_font=dict(size=15, color='#1e293b'),
        paper_bgcolor='#ffffff',
        plot_bgcolor='#f8fafc',
        margin=dict(l=50, r=20, t=50, b=50),
        legend=dict(
            orientation='h', yanchor='bottom', y=1.02,
            xanchor='right', x=1, bgcolor='#ffffff',
        ),
    )
    trend_fig.update_traces(line=dict(width=2.5))

    # --- Recent listings from today_deals ---
    listings = pd.DataFrame()
    if today_deals_df is not None and len(today_deals_df) > 0:
        listings = today_deals_df[
            today_deals_df['neighbourhood'] == neighbourhood
        ].copy()
        if len(listings) > 0:
            display_cols = [c for c in [
                'property_id', 'price', 'predicted_k', 'delta_pct',
                'livable_area', 'number_of_bedrooms', 'building_condition',
                'energy_label',
            ] if c in listings.columns]
            listings = listings[display_cols].sort_values(
                'delta_pct', ascending=False, na_position='last')
            listings.columns = [c.replace('_', ' ').title() for c in listings.columns]

    return stats_md, trend_fig, listings


# ============================================================================
# PRE-COMPUTE ANALYTICS AT STARTUP
# ============================================================================

print("Pre-computing neighbourhood analytics...")
try:
    _analytics_df, _neighbourhood_summary = compute_neighbourhood_analytics()
except Exception as e:
    print(f"Error computing neighbourhood analytics: {e}")
    _analytics_df, _neighbourhood_summary = None, None

# Load today's deals
try:
    if os.path.exists(TODAY_DEALS_CSV_PATH):
        _today_deals_df = pd.read_csv(TODAY_DEALS_CSV_PATH)
        print(f"Loaded {len(_today_deals_df)} current listings from today_deals.csv")
    else:
        print("today_deals.csv not found")
except Exception as e:
    print(f"Error loading today_deals.csv: {e}")


# ============================================================================
# GRADIO UI
# ============================================================================

CUSTOM_CSS = """
/* --- Force light mode everywhere --- */
:root, :host {
    --body-background-fill: #ffffff !important;
    --background-fill-primary: #ffffff !important;
    --background-fill-secondary: #f8fafc !important;
    --body-text-color: #0f172a !important;
    --block-background-fill: #ffffff !important;
    --block-label-text-color: #1e293b !important;
    --block-title-text-color: #1e293b !important;
    --input-background-fill: #ffffff !important;
    --table-odd-background-fill: #ffffff !important;
    --table-even-background-fill: #f8fafc !important;
    --table-row-focus: #eff6ff !important;
    --border-color-primary: #e2e8f0 !important;
    --color-accent-soft: #eff6ff !important;
    --neutral-50: #f8fafc !important;
    --neutral-100: #f1f5f9 !important;
    --neutral-200: #e2e8f0 !important;
    --neutral-700: #334155 !important;
    --neutral-800: #1e293b !important;
    --neutral-900: #0f172a !important;
    color-scheme: light !important;
}
body, .gradio-container, .main, .contain {
    background: #ffffff !important;
    color: #0f172a !important;
}
.dark { background: #ffffff !important; color: #0f172a !important; }
.gradio-container {
    max-width: 1280px !important;
    background: #ffffff !important;
    color: #0f172a !important;
}
footer { display: none !important; }

/* Force light backgrounds on all Gradio panels and wrappers */
.gradio-container div[class*="panel"],
.gradio-container .block,
.gradio-container .form,
.gradio-container .wrap {
    background-color: transparent !important;
    color: #0f172a !important;
}
.gradio-container label, .gradio-container span:not(.app-header span), .gradio-container p:not(.app-header p) {
    color: #1e293b !important;
}
/* Textbox / input styling */
.gradio-container textarea, .gradio-container input[type="text"], .gradio-container input[type="number"] {
    background: #ffffff !important;
    color: #0f172a !important;
    border-color: #cbd5e1 !important;
}
/* Dropdown styling */
.gradio-container select, .gradio-container .choices, .gradio-container .choices__inner {
    background: #ffffff !important;
    color: #0f172a !important;
}
.gradio-container .choices__list--dropdown {
    background: #ffffff !important;
}
.gradio-container .choices__list--dropdown .choices__item {
    color: #0f172a !important;
}

/* --- App header --- */
.app-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #1d4ed8 100%);
    border-radius: 16px; padding: 32px 36px; margin-bottom: 8px;
    color: white; text-align: center;
}
.app-header h1 { margin: 0; font-size: 28px; font-weight: 800; letter-spacing: -0.5px; color: white !important; }
.app-header p  { margin: 6px 0 0; font-size: 15px; color: #e2e8f0 !important; }
.app-header span { color: #e2e8f0 !important; }

/* --- Section cards --- */
.section-card {
    background: #ffffff !important; border: 1px solid #e2e8f0; border-radius: 14px;
    padding: 24px 28px; margin: 10px 0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04), 0 4px 12px rgba(0,0,0,0.03);
}
.section-card h3 {
    margin: 0 0 4px 0 !important; font-size: 18px !important; font-weight: 700 !important;
    color: #0f172a !important;
}
.section-card .section-sub {
    font-size: 14px; color: #475569 !important; margin: 0 0 18px 0; line-height: 1.5;
}

/* --- KPI strip --- */
.kpi-strip {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
    gap: 12px; margin: 12px 0 8px;
}
.kpi-card {
    background: linear-gradient(135deg, #f8fafc, #f1f5f9) !important;
    border: 1px solid #e2e8f0; border-radius: 12px;
    padding: 18px 16px; text-align: center;
}
.kpi-card .kpi-value {
    font-size: 26px; font-weight: 800; color: #0f172a !important;
    line-height: 1.1;
}
.kpi-card .kpi-label {
    font-size: 12px; color: #475569 !important; text-transform: uppercase;
    letter-spacing: 0.4px; margin-top: 6px; font-weight: 600;
}

/* --- Dropdowns inside section cards: keep visible --- */
.section-card .dropdown-label {
    display: block; font-size: 13px; font-weight: 600; color: #334155 !important;
    margin-bottom: 6px;
}

/* --- Findings card --- */
.findings-box {
    background: linear-gradient(135deg, #f0f9ff, #eff6ff) !important;
    border: 1px solid #bfdbfe; border-left: 4px solid #3b82f6;
    border-radius: 10px; padding: 20px 24px;
    color: #0f172a !important;
}
.findings-box h3 { color: #1e40af !important; margin: 0 0 12px 0 !important; font-size: 16px !important; }
.findings-box strong { color: #1e3a8a !important; }
.findings-box div { color: #0f172a !important; }

/* --- Comparison table polish --- */
.comparison-table {
    background: #ffffff !important;
}
.comparison-table table {
    font-size: 14px !important;
    background: #ffffff !important;
}
.comparison-table thead th {
    background: #f1f5f9 !important; font-weight: 700 !important;
    color: #1e293b !important; text-transform: uppercase; font-size: 12px !important;
    letter-spacing: 0.3px;
}
.comparison-table tbody tr {
    background: #ffffff !important;
}
.comparison-table tbody tr:nth-child(even) {
    background: #f8fafc !important;
}
.comparison-table td {
    color: #1e293b !important;
}

/* --- Plotly charts: force light backgrounds --- */
.gradio-container .plot-container,
.gradio-container .js-plotly-plot {
    background: #ffffff !important;
}
.gradio-container .js-plotly-plot .plotly .main-svg {
    background: #ffffff !important;
}

/* --- Tab styling --- */
.tab-nav button { font-weight: 600 !important; font-size: 15px !important; color: #334155 !important; }
.tab-nav button.selected {
    border-bottom: 3px solid #2563eb !important; color: #2563eb !important;
}

/* --- Deep dive stats (rendered as HTML) --- */
.deepdive-stats .prose { max-width: none !important; }

/* --- Dataframe label and headers --- */
.gradio-container .dataframe-container {
    background: #ffffff !important;
}
"""

# Pre-build KPI strip HTML
def _format_findings_html(markdown_text):
    """Convert the summary markdown to styled HTML for the findings box."""
    import re as _re
    html = markdown_text.strip()
    # Convert markdown bold
    html = _re.sub(r'\*\*(.+?)\*\*', r'<strong style="color:#1e3a8a;">\1</strong>', html)
    # Convert headers
    html = _re.sub(r'^### (.+)$', r'<h3>\1</h3>', html, flags=_re.MULTILINE)
    # Convert bullet points
    html = _re.sub(
        r'^- (.+)$',
        r'<div style="margin:6px 0;padding:6px 0 6px 14px;border-left:3px solid #60a5fa;">\1</div>',
        html, flags=_re.MULTILINE,
    )
    # Wrap paragraphs
    html = html.replace('\n\n', '<br><br>')
    return f'<div style="font-size:14px;line-height:1.7;color:#0f172a;">{html}</div>'


def _build_kpi_strip():
    if _analytics_df is None or _neighbourhood_summary is None:
        return ''
    total = len(_analytics_df)
    nbh_count = len(_neighbourhood_summary)
    med_price = _analytics_df['price'].median()
    avg_dur = _analytics_df['listing_duration_days'].mean()
    pct_option = _analytics_df['is_under_option'].mean() * 100
    med_delta = _analytics_df['delta_pct'].median() if _analytics_df['delta_pct'].notna().any() else 0
    return f"""<div class="kpi-strip">
        <div class="kpi-card">
            <div class="kpi-value">{total:,}</div>
            <div class="kpi-label">Properties Analyzed</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-value">{nbh_count}</div>
            <div class="kpi-label">Neighbourhoods</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-value">{med_price:,.0f}k</div>
            <div class="kpi-label">Median Price</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-value">{avg_dur:.0f}d</div>
            <div class="kpi-label">Avg Listing Duration</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-value">{pct_option:.1f}%</div>
            <div class="kpi-label">Under Option</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-value">{med_delta:+.1f}%</div>
            <div class="kpi-label">Median Delta</div>
        </div>
    </div>"""


CUSTOM_THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.blue,
    secondary_hue=gr.themes.colors.slate,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont('Inter'), 'system-ui', 'sans-serif'],
)

FORCE_LIGHT_JS = """
() => {
    // Force light mode: remove dark class from all elements
    document.querySelectorAll('.dark').forEach(el => el.classList.remove('dark'));
    document.documentElement.classList.remove('dark');
    document.body.classList.remove('dark');
    document.body.style.backgroundColor = '#ffffff';
    document.body.style.color = '#0f172a';
    // Override Gradio dark mode CSS variables
    document.documentElement.style.setProperty('--body-background-fill', '#ffffff');
    document.documentElement.style.setProperty('--background-fill-primary', '#ffffff');
    document.documentElement.style.setProperty('--background-fill-secondary', '#f8fafc');
    document.documentElement.style.setProperty('--body-text-color', '#0f172a');
    document.documentElement.style.setProperty('--block-background-fill', '#ffffff');
    document.documentElement.style.setProperty('--block-label-text-color', '#1e293b');
    document.documentElement.style.setProperty('--input-background-fill', '#ffffff');
    document.documentElement.style.setProperty('--table-odd-background-fill', '#ffffff');
    document.documentElement.style.setProperty('--table-even-background-fill', '#f8fafc');
    document.documentElement.style.setProperty('--table-row-focus', '#eff6ff');
    document.documentElement.style.setProperty('--border-color-primary', '#e2e8f0');
    document.documentElement.style.setProperty('--color-accent-soft', '#eff6ff');
}
"""

with gr.Blocks(title="Ghent Real Estate Intelligence", js=FORCE_LIGHT_JS) as demo:
    gr.HTML("""<div class="app-header">
        <h1>Ghent Real Estate Intelligence</h1>
        <p>Price predictions &bull; Neighbourhood analytics &bull; Market insights</p>
    </div>""")

    with gr.Tab("Price Predictor"):
        # --- URL Analysis Section ---
        gr.Markdown("### Analyze Immoweb Listing")
        gr.Markdown("Paste an Immoweb URL to automatically fetch property details, "
                     "run the prediction, and fill in the form below.")
        with gr.Row():
            url_input = gr.Textbox(
                label="Immoweb URL",
                placeholder="https://www.immoweb.be/en/classified/house/for-sale/gent/9000/12345678",
                scale=4,
            )
            analyze_btn = gr.Button("Analyze Listing", variant="primary", scale=1)

        url_result = gr.Textbox(
            label="Analysis Result", show_label=True, interactive=False,
            lines=12, max_lines=20,
        )

        gr.Markdown("---")

        # --- Manual Form Section ---
        gr.Markdown("### Manual Input")
        gr.Markdown("Or fill in details manually to estimate a property's value.")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("#### Location")
                address_input = gr.Textbox(
                    label="Search Address", placeholder="e.g. Veldstraat 1",
                    info="Enter street and number to auto-fill coordinates.")
                find_btn = gr.Button("Find Location", variant="secondary", size="sm")

                with gr.Row():
                    lat_input = gr.Number(
                        label="Latitude",
                        value=defaults.get('latitude', 51.05), precision=5)
                    lon_input = gr.Number(
                        label="Longitude",
                        value=defaults.get('longitude', 3.72), precision=5)

                nbr_input = gr.Dropdown(
                    label="Neighbourhood",
                    choices=options.get('neighbourhood', ag_neighbourhood_names),
                    value=options.get('neighbourhood', ['Unknown'])[0])
                street_type_input = gr.Dropdown(
                    label="Street Type",
                    choices=options.get('street_type', ['residential']),
                    value=options.get('street_type', ['residential'])[0])
                flood_input = gr.Dropdown(
                    label="Flood Zone",
                    choices=options.get('flood_zone_type', ['NON_FLOOD_ZONE']),
                    value='NON_FLOOD_ZONE')

            with gr.Column(scale=1):
                gr.Markdown("#### Property Details")
                with gr.Row():
                    livable_area = gr.Number(
                        label="Livable Area (m2)",
                        value=defaults.get('livable_area', 100))
                    year_built = gr.Number(
                        label="Year Built",
                        value=defaults.get('year_built', 1980))

                with gr.Row():
                    bedrooms = gr.Number(
                        label="Bedrooms",
                        value=defaults.get('number_of_bedrooms', 2))
                    bathrooms = gr.Number(
                        label="Bathrooms",
                        value=defaults.get('number_of_bathrooms', 1))

                with gr.Row():
                    subtype = gr.Dropdown(
                        label="Type",
                        choices=options.get('property_subtype', ['HOUSE']),
                        value='HOUSE')
                    condition = gr.Dropdown(
                        label="Condition",
                        choices=options.get('building_condition', ['GOOD']),
                        value='GOOD')

        with gr.Accordion("Advanced Options", open=False):
            with gr.Row():
                with gr.Column():
                    toilets = gr.Number(
                        label="Toilets",
                        value=defaults.get('number_of_toilets', 1))
                    epc = gr.Number(
                        label="EPC Score",
                        value=defaults.get('primary_energy_consumption', 200))
                    cad_income = gr.Number(
                        label="Cadastral Income",
                        value=defaults.get('cadastral_income', 750))
                with gr.Column():
                    kitchen = gr.Dropdown(
                        label="Kitchen",
                        choices=options.get('kitchen_type', ['INSTALLED']),
                        value='INSTALLED')
                    energy = gr.Dropdown(
                        label="Energy Label",
                        choices=options.get('energy_label', ['C']),
                        value='C')
                    heating = gr.Dropdown(
                        label="Heating",
                        choices=options.get('heating_type', ['GAS']),
                        value='GAS')

            gr.Markdown("#### Outdoor & Parking")
            with gr.Row():
                terrace_surf = gr.Number(label="Terrace Surf. (m2)", value=0)
                garden_surf = gr.Number(label="Garden Surf. (m2)", value=0)
            with gr.Row():
                parking_in = gr.Number(label="Indoor Parking", value=0)
                parking_out = gr.Number(label="Outdoor Parking", value=0)
                parking_box = gr.Number(label="Closed Box", value=0)

            gr.Markdown("#### Amenities")
            with gr.Row():
                solar = gr.Checkbox(label="Solar Panels")
                double_glaz = gr.Checkbox(label="Double Glazing", value=True)
                living = gr.Checkbox(label="Living Room", value=True)
                attic = gr.Checkbox(label="Attic")
                basement = gr.Checkbox(label="Basement")
                terrace = gr.Checkbox(label="Terrace")
                garden = gr.Checkbox(label="Garden")
                pool = gr.Checkbox(label="Pool")
                biddit = gr.Checkbox(label="Biddit Sale")

        predict_btn = gr.Button("Predict Price", variant="primary", size="lg")
        output_text = gr.Textbox(
            label="Predicted Price", show_label=True,
            interactive=False, text_align="center")

    # --- Event Wiring ---

    # URL Analysis button → prediction + auto-fill form
    all_form_outputs = [
        lat_input, lon_input, nbr_input, street_type_input, flood_input,
        livable_area, year_built, bedrooms, bathrooms, subtype, condition,
        toilets, epc, cad_income, kitchen, energy, heating,
        terrace_surf, garden_surf, parking_in, parking_out, parking_box,
        solar, double_glaz, living, attic, basement, terrace, garden, pool, biddit,
    ]

    analyze_btn.click(
        analyze_url,
        inputs=[url_input],
        outputs=[url_result] + all_form_outputs,
    )

    # Find Location button
    def on_find_click(addr):
        lat, lon, nbr, st_type = get_location_data(addr)
        return [
            lat if lat else gr.update(),
            lon if lon else gr.update(),
            nbr if nbr else gr.update(),
            st_type if st_type else gr.update(),
        ]

    find_btn.click(
        on_find_click,
        inputs=[address_input],
        outputs=[lat_input, lon_input, nbr_input, street_type_input],
    )

    # Manual Predict button
    predict_btn.click(
        make_prediction,
        inputs=[
            livable_area, bedrooms, bathrooms, toilets, terrace_surf, garden_surf,
            parking_in, parking_out, parking_box, epc, lat_input, lon_input,
            year_built, cad_income, subtype, condition, kitchen, energy, heating,
            flood_input, nbr_input, street_type_input,
            solar, double_glaz, living, attic, basement, terrace, garden, pool, biddit,
        ],
        outputs=output_text,
    )

    with gr.Tab("Neighbourhood Analysis"):
        # ---- Refresh Button ----
        with gr.Row():
            refresh_btn = gr.Button(
                "Refresh with Latest Data", variant="primary", scale=1)
            refresh_status = gr.Textbox(
                label="Refresh Status", interactive=False,
                lines=4, max_lines=8, scale=3)

        # ---- KPI Dashboard Strip ----
        kpi_html = gr.HTML(_build_kpi_strip())

        # ---- Section 1: Choropleth Map ----
        with gr.Group(elem_classes='section-card'):
            gr.HTML("""<h3>Interactive Neighbourhood Map</h3>
                       <p class="section-sub">Explore Ghent's 25 neighbourhoods &mdash; select a metric to recolor the map.</p>""")
            choropleth_metric = gr.Dropdown(
                label="Color map by metric",
                choices=[
                    'Median Price/m\u00b2',
                    'Avg Listing Duration (days)',
                    'Number of Listings',
                    '% Underpriced',
                    'Median Delta %',
                ],
                value='Median Price/m\u00b2',
            )
            choropleth_html = gr.HTML(
                value=(build_choropleth_map(_neighbourhood_summary, 'Median Price/m\u00b2')
                       if _neighbourhood_summary is not None
                       else '<div style="padding:50px;text-align:center;color:#64748b;">Analytics data not available.</div>'),
            )

        def update_choropleth(metric):
            if _neighbourhood_summary is None:
                return '<div style="padding:50px;text-align:center;color:#64748b;">Analytics data not available.</div>'
            return build_choropleth_map(_neighbourhood_summary, metric)

        choropleth_metric.change(
            update_choropleth,
            inputs=[choropleth_metric],
            outputs=[choropleth_html],
        )

        # ---- Section 2: Do Underpriced Properties Sell Faster? ----
        with gr.Group(elem_classes='section-card'):
            gr.HTML("""<h3>Do Underpriced Properties Sell Faster?</h3>
                       <p class="section-sub">Comparing listing duration across pricing buckets &mdash;
                       underpriced (&gt;+10% delta), fair (&plusmn;10%), and overpriced (&lt;&minus;10%).</p>""")

            # Build initial values (dynamic so refresh can update them)
            _init_scatter = go.Figure()
            _init_bar = go.Figure()
            _init_findings = '<div style="padding:30px;text-align:center;color:#64748b;">Pricing analysis not available (requires AutoGluon predictions).</div>'
            if _analytics_df is not None and _analytics_df['delta_pct'].notna().any():
                _init_scatter, _init_bar, _init_summary_text = build_pricing_speed_analysis(_analytics_df)
                _init_findings = f'<div class="findings-box">{_format_findings_html(_init_summary_text)}</div>'

            scatter_plot = gr.Plot(value=_init_scatter, container=False)
            with gr.Row(equal_height=True):
                with gr.Column(scale=3, min_width=400):
                    bar_plot = gr.Plot(value=_init_bar, container=False)
                with gr.Column(scale=2, min_width=300):
                    findings_html = gr.HTML(value=_init_findings)

        # ---- Section 3: Neighbourhood Comparison Table ----
        with gr.Group(elem_classes='section-card'):
            gr.HTML("""<h3>Neighbourhood Comparison</h3>
                       <p class="section-sub">All 25 neighbourhoods ranked &mdash; click column headers to sort.</p>""")

            def _format_comparison_table(summary):
                if summary is None:
                    return pd.DataFrame()
                display_summary = summary.rename(columns={
                    'neighbourhood': 'Neighbourhood',
                    'count': 'Listings',
                    'median_price_k': 'Median Price (k\u20ac)',
                    'median_price_sqm': 'Median \u20ac/m\u00b2',
                    'avg_listing_duration': 'Avg Duration (days)',
                    'pct_underpriced': '% Underpriced',
                    'median_delta_pct': 'Median Delta %',
                    'pct_under_option': '% Under Option',
                })
                display_cols = [
                    'Neighbourhood', 'Listings', 'Median Price (k\u20ac)',
                    'Median \u20ac/m\u00b2', 'Avg Duration (days)',
                    '% Underpriced', 'Median Delta %', '% Under Option',
                ]
                display_cols = [c for c in display_cols if c in display_summary.columns]
                return display_summary[display_cols]

            comparison_table = gr.Dataframe(
                value=_format_comparison_table(_neighbourhood_summary),
                interactive=False,
                elem_classes='comparison-table',
            )

        # ---- Section 4: Neighbourhood Deep Dive ----
        with gr.Group(elem_classes='section-card'):
            gr.HTML("""<h3>Neighbourhood Deep Dive</h3>
                       <p class="section-sub">Select a neighbourhood to see detailed stats, price trends, and current listings.</p>""")
            nbh_choices = (
                sorted(_neighbourhood_summary['neighbourhood'].tolist())
                if _neighbourhood_summary is not None else []
            )
            deepdive_dropdown = gr.Dropdown(
                label="Select a neighbourhood",
                choices=nbh_choices,
                value=nbh_choices[0] if nbh_choices else None,
            )

            deepdive_stats = gr.HTML(elem_classes='deepdive-stats')
            deepdive_trend = gr.Plot(container=False)
            with gr.Accordion("Current Listings", open=False):
                deepdive_listings = gr.Dataframe(
                    label="Current Listings",
                    interactive=False,
                    elem_classes='comparison-table',
                )

        def update_deepdive(neighbourhood):
            if _analytics_df is None or neighbourhood is None:
                empty_fig = go.Figure()
                empty_fig.update_layout(height=300, template='plotly_white')
                return "Select a neighbourhood.", empty_fig, pd.DataFrame()
            return build_neighbourhood_deepdive(
                _analytics_df, _today_deals_df, neighbourhood)

        deepdive_dropdown.change(
            update_deepdive,
            inputs=[deepdive_dropdown],
            outputs=[deepdive_stats, deepdive_trend, deepdive_listings],
        )

        # Trigger initial load for the first neighbourhood
        if nbh_choices:
            _init_stats, _init_trend, _init_listings = build_neighbourhood_deepdive(
                _analytics_df, _today_deals_df, nbh_choices[0])
            deepdive_stats.value = _init_stats
            deepdive_trend.value = _init_trend
            deepdive_listings.value = _init_listings

        # ---- Refresh Button Wiring ----
        def on_refresh(progress=gr.Progress(track_tqdm=True)):
            """Run refresh pipeline and return updated values for all components."""
            status = refresh_pipeline(progress)

            # Rebuild all dynamic components from updated caches
            new_kpi = _build_kpi_strip()

            new_choropleth = (
                build_choropleth_map(_neighbourhood_summary, 'Median Price/m\u00b2')
                if _neighbourhood_summary is not None
                else '<div style="padding:50px;text-align:center;color:#64748b;">Analytics data not available.</div>'
            )

            # Section 2: pricing analysis
            new_scatter = go.Figure()
            new_bar = go.Figure()
            new_findings = '<div style="padding:30px;text-align:center;color:#64748b;">Pricing analysis not available.</div>'
            if _analytics_df is not None and _analytics_df['delta_pct'].notna().any():
                new_scatter, new_bar, summary_text = build_pricing_speed_analysis(_analytics_df)
                new_findings = f'<div class="findings-box">{_format_findings_html(summary_text)}</div>'

            # Section 3: comparison table
            new_table = _format_comparison_table(_neighbourhood_summary)

            # Section 4: update deep dive dropdown choices
            new_nbh_choices = (
                sorted(_neighbourhood_summary['neighbourhood'].tolist())
                if _neighbourhood_summary is not None else []
            )
            new_dropdown = gr.Dropdown(
                choices=new_nbh_choices,
                value=new_nbh_choices[0] if new_nbh_choices else None,
            )

            # Deep dive for first neighbourhood
            new_dd_stats = "Select a neighbourhood."
            new_dd_trend = go.Figure()
            new_dd_trend.update_layout(height=300, template='plotly_white')
            new_dd_listings = pd.DataFrame()
            if new_nbh_choices and _analytics_df is not None:
                new_dd_stats, new_dd_trend, new_dd_listings = (
                    build_neighbourhood_deepdive(
                        _analytics_df, _today_deals_df, new_nbh_choices[0]))

            return [
                status,
                new_kpi,
                new_choropleth,
                new_scatter,
                new_bar,
                new_findings,
                new_table,
                new_dropdown,
                new_dd_stats,
                new_dd_trend,
                new_dd_listings,
            ]

        refresh_btn.click(
            on_refresh,
            inputs=[],
            outputs=[
                refresh_status,
                kpi_html,
                choropleth_html,
                scatter_plot,
                bar_plot,
                findings_html,
                comparison_table,
                deepdive_dropdown,
                deepdive_stats,
                deepdive_trend,
                deepdive_listings,
            ],
        )


if __name__ == "__main__":
    print("Launching Gradio App...")
    demo.launch(share=True, css=CUSTOM_CSS, theme=CUSTOM_THEME)
