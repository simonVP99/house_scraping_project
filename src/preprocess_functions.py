# preprocessing.py
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import geopandas as gpd
from shapely.geometry import Point, shape
import json
import os
import joblib
import osmnx as ox
from datetime import datetime


# =============================================================================
# Stage 1: Data Cleaning & Filtering
# =============================================================================

def handle_dates(df):
    """Parse date_obtained and publication/modification dates."""
    df['date_obtained'] = df['date_obtained'].astype(str).str.zfill(8)
    df['date_obtained'] = pd.to_datetime(df['date_obtained'], format='%d%m%Y', errors='coerce')
    df['publication_date'] = pd.to_datetime(df['publication_date'], errors='coerce')
    df['last_modification_date'] = pd.to_datetime(df['last_modification_date'], errors='coerce')
    df = df.sort_values('date_obtained')
    return df


def calculate_listing_duration(df):
    """Calculate listing duration from publication to last modification date."""
    df['listing_duration_days'] = (
        df['last_modification_date'] - df['publication_date']
    ).dt.days
    # Clamp negatives to 0 (same-day listings or data inconsistencies)
    df['listing_duration_days'] = df['listing_duration_days'].clip(lower=0)
    return df


def handle_duplicates(df, keep='last'):
    """Keep one record per property_id."""
    df = df.drop_duplicates(subset=['property_id'], keep=keep)
    return df


def _is_commercial_property(row):
    """Detect properties where a commercial space is part of the deal."""
    subtype = str(row.get('property_subtype', ''))
    if subtype == 'MIXED_USE_BUILDING':
        return True
    if subtype in ('APARTMENT', 'PENTHOUSE', 'GROUND_FLOOR', 'FLAT_STUDIO',
                   'DUPLEX', 'TRIPLEX', 'SERVICE_FLAT', 'KOT', 'LOFT'):
        return False
    desc = str(row.get('description', '')).lower()
    commercial_terms = (
        'handelspand', 'handelsruimte', 'handelsgelijkvloers', 'winkelruimte',
        'winkelpand', 'kantoorruimte', 'espace commercial', 'commercial space',
        'fond de commerce', 'commercieel gelijkvloers',
    )
    return any(term in desc for term in commercial_terms)


def filter_non_modelable(df):
    """Remove listings with incomparable pricing dynamics."""
    n_before = len(df)
    df = df[df['is_life_annuity_sale'] != True]
    df = df[df['is_public_sale'] != True]
    df = df[df['is_biddit_sale'] != True]
    n_after_sales = len(df)
    print(f"  Removed {n_before - n_after_sales} non-modelable listings (life annuity + public + Biddit sales)")

    commercial_mask = df.apply(_is_commercial_property, axis=1)
    n_commercial = commercial_mask.sum()
    df = df[~commercial_mask]
    print(f"  Removed {n_commercial} commercial/mixed-use properties")

    return df


def filter_hard_bounds(df):
    """Remove data entry errors using geographic and logical bounds."""
    n_before = len(df)

    # Drop rows missing essential fields
    df = df.dropna(subset=['price', 'livable_area', 'latitude', 'longitude'])

    # Geographic bounds for Ghent region
    df = df[(df['latitude'] >= 50.9) & (df['latitude'] <= 51.2)]
    df = df[(df['longitude'] >= 3.5) & (df['longitude'] <= 4.0)]

    # Logical bounds
    df = df[(df['livable_area'] >= 10) & (df['livable_area'] <= 1000)]
    df = df[df['price'] > 25000]

    print(f"  Hard bounds filtering removed {n_before - len(df)} rows")
    return df


# =============================================================================
# Stage 2: Feature Engineering
# =============================================================================

def handle_age_and_year_built(df):
    """Create age feature from year_built, using current year dynamically.
    Returns (df, median_year_built) so the median can be saved for inference."""
    current_year = datetime.now().year
    median_year = float(df['year_built'].median())
    if df['year_built'].isna().any():
        df['year_built'] = df['year_built'].fillna(median_year)
    df['age'] = current_year - df['year_built']
    return df, median_year


def create_price_per_sqm(df):
    """Create price per square meter feature (before price scaling)."""
    df['price_sq'] = df['price'] / df['livable_area']
    return df


def scale_price(df):
    """Scale price to thousands of EUR for model-friendly coefficients."""
    df['price'] = df['price'] / 1000
    df['price_sq'] = df['price_sq'] / 1000
    return df


def convert_boolean_columns(df, add_missing_indicators=True):
    """Convert has_*/is_* flags to boolean, preserving the fact that a flag was
    *not declared* via a companion ``<col>_missing`` indicator.

    Immoweb sends ``null`` for an amenity the agency did not tick.  Coercing that
    to ``False`` asserts "this property has no garden", which is a fabrication:
    "false" never appears in the source payload at all.  Instead we emit both
    the boolean (for tree models, which cannot consume NaN) and an explicit
    missingness indicator so the distinction is still learnable.
    """
    bool_prefixes = ('has_', 'is_')
    indicator_cols = {}

    for col in list(df.columns):
        if not col.startswith(bool_prefixes):
            continue
        series = df[col].replace({'True': True, 'False': False})
        series = series.replace({'true': True, 'false': False})
        series = series.map(lambda v: v if isinstance(v, bool) else None)

        was_missing = series.isna()
        if add_missing_indicators:
            indicator_cols[f'{col}_missing'] = was_missing.astype(int)

        df[col] = pd.Series(series.notna().to_numpy(), index=df.index, dtype=bool)

    for name, values in indicator_cols.items():
        df[name] = values

    return df


def flag_text_recovered_amenities(df):
    """Record whether any tri-state flag was recovered from the description.

    The parser sets ``text_recovered_flags`` to a pipe-separated list of the
    columns it filled from free text.  Kept as an indicator rather than a raw
    string so the model can weigh text-derived values differently from values
    the agency actually declared.
    """
    if 'text_recovered_flags' not in df.columns:
        return df
    recovered = df['text_recovered_flags'].fillna('')
    df['has_text_recovered_amenities'] = (recovered.str.len() > 0).astype(int)
    df['n_text_recovered_amenities'] = recovered.apply(
        lambda value: 0 if not value else len(value.split('|')))
    return df


def handle_missing_values(df):
    """Fill missing values for categorical and numeric fields."""
    # Categorical -> 'Unknown'
    categorical_fill = [
        'heating_type', 'building_condition', 'flood_zone_type',
        'renovation_obligation', 'location_type', 'kitchen_type', 'energy_label'
    ]
    for col in categorical_fill:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')

    # Surface fields -> 0.  Only safe because the flag and its surface are
    # co-null; a filled 0 here means "surface not declared", and the
    # companion has_garden/has_terrace boolean was itself filled to False.
    # garden_surface_missing / terrace_surface_missing record that.
    for col in ['garden_surface', 'terrace_surface']:
        if col in df.columns:
            missing = df[col].isna()
            if missing.any():
                df[f'{col}_missing'] = missing.astype(int)
            df[col] = df[col].fillna(0)

    # Count fields: convert 0 to NaN (0 bedrooms/bathrooms means missing, not zero)
    for col in ['number_of_bathrooms', 'number_of_toilets', 'number_of_bedrooms']:
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan)

    # Cadastral income: 0 likely means missing
    if 'cadastral_income' in df.columns:
        df['cadastral_income'] = df['cadastral_income'].replace(0, np.nan)

    return df


def drop_dead_columns(df, keep_extended=False):
    """Remove columns that are empty, constant, or not useful for modeling.

    ``keep_extended`` retains columns the parser now populates (orientation,
    demand signals, land attributes) which are useful for analysis even when
    they are not price drivers.
    """
    cols_to_drop = [
        # Dead columns (0% or near-0% fill)
        'parking_closed_box',
        # Constant for Ghent-only data
        'transaction_type', 'region', 'province', 'district',
        # IDs and text (not features)
        'external_reference', 'title', 'description', 'text_recovered_flags',
        'city_raw',
        # Redundant after filtering (all same value for remaining rows)
        'is_sold_or_rented', 'is_life_annuity_sale', 'is_public_sale', 'is_biddit_sale',
        # Already used for listing_duration.  publication_date is retained as a
        # *separate* column so true time-on-market can be derived from it.
        'last_modification_date',
        # Replaced by age
        'number_of_rooms',
        # Price range fields (rarely populated)
        'price_min', 'price_max',
        # Transaction subtype (nearly all BUY_REGULAR)
        'transaction_subtype',
    ]
    if not keep_extended:
        cols_to_drop += [
            'terrace_orientation', 'garden_orientation',
            'view_count', 'bookmark_count',
            'land_is_flat', 'land_is_wooded', 'land_facing_street', 'land_width',
            'land_use_designation', 'land_sewer_connection',
            'permit_is_obtained', 'permit_has_obligation_to_construct',
            'breaches_urban_planning', 'fireplace_count',
        ]
    existing_cols = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=existing_cols)
    if existing_cols:
        print(f"  Dropped {len(existing_cols)} dead/unused columns")
    return df


def encode_categorical_variables(df):
    """One-hot encode property_type."""
    df = pd.get_dummies(df, columns=['property_type'], drop_first=True)
    return df


# =============================================================================
# Stage 3: Imputation & Spatial Features
# =============================================================================

def fill_missing_values_poisson(df_non_nan, df_nan, target_column, cols):
    """Impute missing count data using Poisson GLM.
    Returns (imputed_column, fitted_model) so the model can be saved for inference."""
    X_train = df_non_nan[cols]
    y_train = df_non_nan[target_column]
    train_data = pd.concat([X_train, y_train], axis=1)
    formula = f"{target_column} ~ " + ' + '.join(cols)
    poisson_model = smf.glm(formula=formula, data=train_data, family=sm.families.Poisson()).fit()
    X_nan = df_nan[cols].dropna()
    y_pred = poisson_model.predict(X_nan)
    y_pred_rounded = np.round(y_pred).astype('int64')
    y_pred_rounded[y_pred_rounded < 0] = 0
    df_nan.loc[X_nan.index, target_column] = y_pred_rounded
    df_filled = pd.concat([df_non_nan, df_nan]).sort_index()
    return df_filled[target_column], poisson_model


def fill_missing_values_in_numerical_columns(df):
    """Sequential Poisson imputation with enriched predictor sets.
    Returns (df, poisson_models_dict) so models can be saved for inference."""
    base_cols = ['livable_area', 'property_type_HOUSE', 'price', 'price_sq']

    # Add new predictors if available and have enough non-null values
    for extra_col in ['is_newly_built', 'facade_count', 'floor_number']:
        if extra_col in df.columns and df[extra_col].notna().sum() > len(df) * 0.3:
            base_cols.append(extra_col)

    poisson_models = {}

    # Bedrooms
    df_non_nan = df[df['number_of_bedrooms'].notna()]
    df_nan = df[df['number_of_bedrooms'].isna()]
    if len(df_nan) > 0:
        df['number_of_bedrooms'], model = fill_missing_values_poisson(
            df_non_nan, df_nan, 'number_of_bedrooms', base_cols)
        poisson_models['number_of_bedrooms'] = {"model": model, "predictors": list(base_cols)}

    # Toilets
    cols_toilets = base_cols + ['number_of_bedrooms']
    df_non_nan = df[df['number_of_toilets'].notna()]
    df_nan = df[df['number_of_toilets'].isna()]
    if len(df_nan) > 0:
        df['number_of_toilets'], model = fill_missing_values_poisson(
            df_non_nan, df_nan, 'number_of_toilets', cols_toilets)
        poisson_models['number_of_toilets'] = {"model": model, "predictors": list(cols_toilets)}

    # Bathrooms
    cols_bathrooms = cols_toilets + ['number_of_toilets']
    df_non_nan = df[df['number_of_bathrooms'].notna()]
    df_nan = df[df['number_of_bathrooms'].isna()]
    if len(df_nan) > 0:
        df['number_of_bathrooms'], model = fill_missing_values_poisson(
            df_non_nan, df_nan, 'number_of_bathrooms', cols_bathrooms)
        poisson_models['number_of_bathrooms'] = {"model": model, "predictors": list(cols_bathrooms)}

    # Cadastral income
    cols_income = ['price', 'price_sq', 'property_type_HOUSE']
    df_non_nan = df[df['cadastral_income'].notna()]
    df_nan = df[df['cadastral_income'].isna()]
    if len(df_nan) > 0:
        df['cadastral_income'], model = fill_missing_values_poisson(
            df_non_nan, df_nan, 'cadastral_income', cols_income)
        poisson_models['cadastral_income'] = {"model": model, "predictors": list(cols_income)}

    return df, poisson_models


def parse_geometry(geometry_str):
    geometry_dict = json.loads(geometry_str)
    return shape(geometry_dict)


def get_neighborhood(lat, lon, wijken_gdf):
    point = Point(lon, lat)
    possible_matches_index = list(wijken_gdf.sindex.intersection(point.bounds))
    possible_matches = wijken_gdf.iloc[possible_matches_index]
    for idx, row in possible_matches.iterrows():
        if row['geometry'].contains(point):
            return row['wijk']
    return 'Unknown'


def add_neighborhood_information(df, wijken_path='external_data/stadswijken-gent.csv'):
    wijken = pd.read_csv(wijken_path, sep=';')
    wijken['geometry'] = wijken['Geometry'].apply(parse_geometry)
    wijken_gdf = gpd.GeoDataFrame(wijken, geometry='geometry', crs="EPSG:4326")
    df['neighbourhood'] = df.apply(
        lambda row: get_neighborhood(row['latitude'], row['longitude'], wijken_gdf), axis=1)
    df['neighbourhood'] = df['neighbourhood'].str.replace(' ', '', regex=False).str.replace('-', '_', regex=False)
    return df


def add_street_type_information(df, graph_path='external_data/ghent_region_network.graphml'):
    """Adds street_type and distance_to_street via OSM network matching."""
    if 'latitude' not in df.columns or 'longitude' not in df.columns:
        print("Warning: Latitude/Longitude columns not found. Skipping street type.")
        return df

    if os.path.exists(graph_path):
        print("Loading street network from local file...")
        G = ox.load_graphml(graph_path)
    else:
        print("Downloading street network (Ghent, Evergem, Merelbeke)...")
        place_names = ["Ghent, Belgium", "Evergem, Belgium", "Merelbeke, Belgium"]
        G = ox.graph_from_place(place_names, network_type='all')
        os.makedirs(os.path.dirname(graph_path), exist_ok=True)
        ox.save_graphml(G, graph_path)

    nodes, edges = ox.graph_to_gdfs(G)
    edges['highway'] = edges['highway'].apply(lambda h: h[0] if isinstance(h, list) else h)

    df_geo = df.dropna(subset=['latitude', 'longitude']).copy()
    geometry = [Point(lon, lat) for lon, lat in zip(df_geo['longitude'], df_geo['latitude'])]
    properties_gdf = gpd.GeoDataFrame(df_geo, geometry=geometry, crs=edges.crs)

    print("Matching properties to nearest streets...")

    nearest_edge_indices, distances = ox.nearest_edges(
        G, X=properties_gdf['longitude'], Y=properties_gdf['latitude'], return_dist=True
    )
    df['street_type'] = edges.loc[nearest_edge_indices, 'highway'].values
    df['distance_to_street'] = distances

    # Correction for cycleways/footways
    undesirable_types = ['cycleway', 'footway', 'path', 'pedestrian', 'track', 'steps', 'service', 'trunk']
    needs_correction_mask = df['street_type'].isin(undesirable_types)

    if needs_correction_mask.any():
        print(f"Re-evaluating {needs_correction_mask.sum()} cycleway/footway matches...")
        main_road_edges = edges[~edges['highway'].isin(undesirable_types)]

        if not main_road_edges.empty:
            G_main_roads = ox.graph_from_gdfs(nodes, main_road_edges)
            properties_to_correct = properties_gdf[needs_correction_mask]

            nearest_main_road_indices, main_road_distances = ox.nearest_edges(
                G_main_roads,
                X=properties_to_correct['longitude'],
                Y=properties_to_correct['latitude'],
                return_dist=True
            )

            corrected_street_types = main_road_edges.loc[nearest_main_road_indices, 'highway'].values
            proximity_threshold = 20.0

            correction_df = pd.DataFrame({
                'corrected_type': corrected_street_types,
                'corrected_distance': main_road_distances
            }, index=properties_to_correct.index)

            df['corrected_type'] = correction_df['corrected_type']
            df['corrected_distance'] = correction_df['corrected_distance']

            df['street_type'] = np.where(
                (needs_correction_mask) & (df['corrected_distance'] <= proximity_threshold),
                df['corrected_type'],
                df['street_type']
            )
            df.drop(columns=['corrected_type', 'corrected_distance'], inplace=True)

    df['street_type'] = df['street_type'].fillna('unknown')
    print("Street type assignment complete.")
    return df


def add_distance_from_center_feature(df):
    center_lat = 51.053098573153505
    center_lon = 3.7265139580328865
    df['dist_from_center'] = np.sqrt(
        (df['latitude'] - center_lat)**2 + (df['longitude'] - center_lon)**2
    ) * 100
    return df


def add_spatial_lag_features(df):
    """Compute spatial lag (avg price of k nearest properties) and density proxy."""
    from scipy.spatial import cKDTree

    valid_mask = df['latitude'].notna() & df['longitude'].notna() & df['price'].notna()
    coords = df.loc[valid_mask, ['latitude', 'longitude']].values
    prices = df.loc[valid_mask, 'price'].values

    tree = cKDTree(coords)
    # 10 nearest neighbours + self
    dists, indices = tree.query(coords, k=11)

    # Spatial lag: avg price of 10 nearest (exclude self at index 0)
    neighbour_prices = prices[indices[:, 1:]]
    df.loc[valid_mask, 'spatial_lag_10'] = np.nanmean(neighbour_prices, axis=1)

    # Density proxy: avg distance to 5 nearest (in km)
    df.loc[valid_mask, 'avg_dist_5nn'] = np.mean(dists[:, 1:6] * 111, axis=1)

    # Fill NaN for rows without coordinates
    df['spatial_lag_10'] = df['spatial_lag_10'].fillna(df['price'].median())
    df['avg_dist_5nn'] = df['avg_dist_5nn'].fillna(df['avg_dist_5nn'].median())

    print(f"  Spatial lag: median={df['spatial_lag_10'].median():.1f}k, "
          f"avg_dist_5nn median={df['avg_dist_5nn'].median():.2f}km")
    return df


def filter_statistical_outliers(df):
    """Remove statistical outliers after hard bounds already caught data errors."""
    df['log_price'] = np.log(df['price'])
    upper_log_price = df['log_price'].quantile(0.99)
    upper_price_sq = df['price_sq'].quantile(0.99)
    n_before = len(df)
    df = df[df['log_price'] <= upper_log_price]
    df = df[df['price_sq'] <= upper_price_sq]
    print(f"  Statistical outlier filtering removed {n_before - len(df)} rows")
    return df


# =============================================================================
# Main Pipeline
# =============================================================================

def save_imputation_models(imputation_artifacts, output_dir='outputs/models'):
    """Save imputation artifacts (Poisson GLMs + median year_built) for inference."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'imputation_models.joblib')
    joblib.dump(imputation_artifacts, path)
    print(f"  Saved imputation models to {path}")


def preprocess(input_data, keep='last'):
    """Full preprocessing pipeline."""
    df = pd.DataFrame(input_data)
    print(f"Starting preprocessing: {len(df)} rows")

    # --- Stage 1: Data Cleaning & Filtering ---
    df = handle_dates(df)

    if len(df) > 1:
        df = calculate_listing_duration(df)

    if len(df) > 1:
        df = handle_duplicates(df, keep)
        print(f"  After dedup: {len(df)} rows")

    if len(df) > 1:
        df = filter_non_modelable(df)

    df = filter_hard_bounds(df)
    print(f"  After cleaning: {len(df)} rows")

    # --- Stage 2: Feature Engineering ---
    df, median_year_built = handle_age_and_year_built(df)
    df = create_price_per_sqm(df)
    df = scale_price(df)
    df = convert_boolean_columns(df)
    df = flag_text_recovered_amenities(df)
    df = handle_missing_values(df)
    df = drop_dead_columns(df)
    df = encode_categorical_variables(df)
    df['postal_code'] = df['postal_code'].astype(str)

    # --- Stage 3: Imputation & Spatial Features ---
    poisson_models = {}
    if len(df) > 1:
        df, poisson_models = fill_missing_values_in_numerical_columns(df)

    # Save imputation artifacts for use at inference time
    imputation_artifacts = {
        "median_year_built": median_year_built,
        "poisson_models": poisson_models,
    }
    save_imputation_models(imputation_artifacts)

    df = add_neighborhood_information(df)
    df = add_street_type_information(df)
    df = add_distance_from_center_feature(df)

    if len(df) > 1:
        df = add_spatial_lag_features(df)

    if len(df) > 1:
        df = filter_statistical_outliers(df)

    print(f"Preprocessing complete: {len(df)} rows, {len(df.columns)} columns")
    return df
