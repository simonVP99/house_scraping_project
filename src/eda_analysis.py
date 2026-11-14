"""
Comprehensive EDA for Ghent Real Estate Data
Generates both console output (statistical findings) and plots.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from scipy import stats

warnings.filterwarnings('ignore')
pd.set_option('future.no_silent_downcasting', True)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs', 'eda_plots')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Consistent style
sns.set_theme(style="whitegrid", font_scale=1.1)
PALETTE = sns.color_palette("viridis", 10)


# ============================================================================
# SECTION 1: LOAD DATA
# ============================================================================

def load_data():
    """Load both preprocessed and raw parsed data."""
    print("=" * 80)
    print("LOADING DATA")
    print("=" * 80)

    df = pd.read_csv(os.path.join(BASE_DIR, 'outputs', 'property_data_preprocessed.csv'))
    df['date_obtained'] = pd.to_datetime(df['date_obtained'])
    print(f"Preprocessed data: {len(df):,} rows x {len(df.columns)} columns")

    # Raw parsed data for Biddit analysis (those are filtered out of preprocessed)
    df_raw = pd.read_csv(
        os.path.join(BASE_DIR, 'outputs', 'property_data_parsed.csv'),
        low_memory=False
    )
    df_raw['date_obtained'] = pd.to_datetime(df_raw['date_obtained'], errors='coerce')
    print(f"Raw parsed data:   {len(df_raw):,} rows x {len(df_raw.columns)} columns")

    return df, df_raw


# ============================================================================
# SECTION 2: FEATURE CORRELATIONS WITH PRICE
# ============================================================================

def analyze_correlations(df):
    """Comprehensive correlation analysis with price."""
    print("\n" + "=" * 80)
    print("FEATURE CORRELATIONS WITH PRICE")
    print("=" * 80)

    # --- 2a. Pearson correlations for all numeric features ---
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude = ['property_id', 'latitude', 'longitude', 'postal_code', 'log_price']
    numeric_cols = [c for c in numeric_cols if c not in exclude]

    corr_with_price = df[numeric_cols].corr()['price'].drop('price').sort_values(ascending=False)
    print("\nTop 15 positive correlations with price:")
    for feat, val in corr_with_price.head(15).items():
        print(f"  {feat:<35s} r = {val:+.3f}")

    print("\nTop 10 negative correlations with price:")
    for feat, val in corr_with_price.tail(10).items():
        print(f"  {feat:<35s} r = {val:+.3f}")

    # --- 2b. Full correlation heatmap ---
    # Select features with meaningful correlation (|r| > 0.05)
    relevant = corr_with_price[corr_with_price.abs() > 0.05].index.tolist()
    relevant = ['price'] + relevant

    fig, ax = plt.subplots(figsize=(16, 14))
    corr_matrix = df[relevant].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, vmin=-1, vmax=1, square=True, linewidths=0.5, ax=ax,
                annot_kws={'size': 8})
    ax.set_title('Feature Correlations (|r| > 0.05 with price)', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'correlation_matrix.png'), dpi=150)
    plt.close()

    # --- 2c. Correlation of boolean features with price ---
    bool_cols = df.select_dtypes(include=['bool']).columns.tolist()
    if bool_cols:
        print("\nBoolean feature correlations with price:")
        bool_corr = {}
        for col in bool_cols:
            true_prices = df.loc[df[col] == True, 'price']
            false_prices = df.loc[df[col] == False, 'price']
            if len(true_prices) > 10 and len(false_prices) > 10:
                mean_diff = true_prices.mean() - false_prices.mean()
                pct_true = df[col].mean() * 100
                bool_corr[col] = (mean_diff, pct_true, true_prices.mean(), false_prices.mean())

        sorted_bool = sorted(bool_corr.items(), key=lambda x: abs(x[1][0]), reverse=True)
        for col, (diff, pct, mean_t, mean_f) in sorted_bool:
            sign = "+" if diff > 0 else ""
            print(f"  {col:<30s} {sign}{diff:>10,.0f}€  (True: {mean_t:>10,.0f}€  False: {mean_f:>10,.0f}€  {pct:5.1f}% True)")

    # --- 2d. Categorical features vs price ---
    print("\nCategorical feature effects on price:")
    cat_cols = ['building_condition', 'kitchen_type', 'energy_label', 'heating_type',
                'flood_zone_type', 'property_subtype', 'street_type', 'location_type',
                'renovation_obligation', 'p_score', 'g_score']
    for col in cat_cols:
        if col in df.columns:
            grouped = df.groupby(col)['price'].agg(['mean', 'median', 'count'])
            grouped = grouped[grouped['count'] >= 20].sort_values('median', ascending=False)
            if len(grouped) > 1:
                price_range = grouped['median'].max() - grouped['median'].min()
                print(f"\n  {col} (median price range: {price_range:,.0f}€):")
                for idx, row in grouped.iterrows():
                    print(f"    {str(idx):<25s} median: {row['median']:>10,.0f}€  (n={int(row['count']):>5,})")

    return corr_with_price


# ============================================================================
# SECTION 3: FEATURE IMPORTANCE (XGBoost)
# ============================================================================

def analyze_feature_importance(df):
    """Quick XGBoost feature importance analysis."""
    print("\n" + "=" * 80)
    print("FEATURE IMPORTANCE (XGBoost)")
    print("=" * 80)

    try:
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.preprocessing import LabelEncoder

        # Prepare features
        feature_cols = [c for c in df.columns if c not in
                        ['property_id', 'date_obtained', 'price', 'log_price', 'price_sq',
                         'street', 'house_number', 'box', 'city', 'immo_office_name']]

        df_model = df[feature_cols + ['price']].copy()

        # Encode categoricals
        le_dict = {}
        for col in df_model.select_dtypes(include=['object', 'category']).columns:
            le = LabelEncoder()
            df_model[col] = le.fit_transform(df_model[col].astype(str))
            le_dict[col] = le

        # Convert booleans to int
        for col in df_model.select_dtypes(include=['bool']).columns:
            df_model[col] = df_model[col].astype(int)

        X = df_model.drop('price', axis=1).fillna(-1)
        y = df_model['price']

        # Quick GBM for feature importance
        model = GradientBoostingRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            subsample=0.8, random_state=42
        )
        model.fit(X, y)
        score = model.score(X, y)
        print(f"\nQuick GBM R² (on training data): {score:.4f}")

        importances = pd.Series(model.feature_importances_, index=X.columns)
        importances = importances.sort_values(ascending=False)

        print("\nTop 25 most important features:")
        for i, (feat, imp) in enumerate(importances.head(25).items()):
            bar = "█" * int(imp * 200)
            print(f"  {i+1:2d}. {feat:<35s} {imp:.4f} {bar}")

        # Plot
        fig, ax = plt.subplots(figsize=(12, 10))
        top30 = importances.head(30)
        colors = sns.color_palette("viridis", len(top30))
        ax.barh(range(len(top30)), top30.values, color=colors)
        ax.set_yticks(range(len(top30)))
        ax.set_yticklabels(top30.index)
        ax.invert_yaxis()
        ax.set_xlabel('Feature Importance (Gini)')
        ax.set_title(f'Top 30 Feature Importances (GBM R²={score:.3f})', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'feature_importance.png'), dpi=150)
        plt.close()

        return importances

    except Exception as e:
        print(f"  Error in feature importance: {e}")
        return None


# ============================================================================
# SECTION 4: TEMPORAL ANALYSIS
# ============================================================================

def analyze_temporal(df):
    """Analyze time-based patterns in listing and pricing."""
    print("\n" + "=" * 80)
    print("TEMPORAL ANALYSIS")
    print("=" * 80)

    # --- 4a. Monthly price trends ---
    df_time = df.set_index('date_obtained').sort_index()
    monthly = df_time.resample('ME').agg(
        avg_price=('price', 'mean'),
        median_price=('price', 'median'),
        count=('price', 'count'),
        avg_area=('livable_area', 'mean'),
        avg_price_sq=('price_sq', 'mean')
    )
    # Filter months with enough data
    monthly = monthly[monthly['count'] >= 30]

    print("\nMonthly price trends:")
    for date, row in monthly.iterrows():
        print(f"  {date.strftime('%Y-%m')}  median: {row['median_price']:>10,.0f}€  "
              f"avg: {row['avg_price']:>10,.0f}€  count: {int(row['count']):>4d}  "
              f"avg €/m²: {row['avg_price_sq']:>6,.0f}")

    # Price trend test
    if len(monthly) > 3:
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            range(len(monthly)), monthly['median_price']
        )
        print(f"\n  Price trend: {slope:+,.0f}€/month (R²={r_value**2:.3f}, p={p_value:.4f})")
        if p_value < 0.05:
            direction = "increasing" if slope > 0 else "decreasing"
            print(f"  -> Statistically significant {direction} trend")
        else:
            print(f"  -> No statistically significant trend")

    # Plot monthly trends
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    ax1.plot(monthly.index, monthly['median_price'] / 1000, 'b-o', label='Median Price', markersize=5)
    ax1.fill_between(monthly.index,
                     monthly['avg_price'] / 1000 - monthly['avg_price'].std() / 1000,
                     monthly['avg_price'] / 1000 + monthly['avg_price'].std() / 1000,
                     alpha=0.2)
    ax1.set_ylabel('Price (K€)')
    ax1.set_title('Monthly Price & Volume Trends')
    ax1.legend()
    ax1r = ax1.twinx()
    ax1r.bar(monthly.index, monthly['count'], alpha=0.3, color='gray', width=20, label='Volume')
    ax1r.set_ylabel('Listings Count')
    ax1r.legend(loc='upper left')

    ax2.plot(monthly.index, monthly['avg_price_sq'], 'g-o', label='Avg €/m²', markersize=5)
    ax2.set_ylabel('Price per m² (€)')
    ax2.set_xlabel('Date')
    ax2.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'time_trends.png'), dpi=150)
    plt.close()

    # --- 4b. Day of week effects ---
    df['dow'] = df['date_obtained'].dt.day_name()
    dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dow_stats = df.groupby('dow')['price'].agg(['median', 'count']).reindex(dow_order)
    print("\nDay-of-week listing patterns:")
    for day, row in dow_stats.iterrows():
        print(f"  {day:<12s}  median: {row['median']:>10,.0f}€  listings: {int(row['count']):>4,}")

    # --- 4c. Seasonal (month of year) ---
    df['month_num'] = df['date_obtained'].dt.month
    df['month_name'] = df['date_obtained'].dt.month_name()
    month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']
    month_stats = df.groupby('month_name')['price'].agg(['median', 'count']).reindex(month_order)
    month_stats = month_stats.dropna()
    print("\nSeasonal patterns:")
    for month, row in month_stats.iterrows():
        bar = "█" * int(row['count'] / month_stats['count'].max() * 30)
        print(f"  {month:<12s}  median: {row['median']:>10,.0f}€  n={int(row['count']):>4,}  {bar}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    existing_months = [m for m in month_order if m in df['month_name'].values]
    sns.boxplot(data=df, x='month_name', y='price', order=existing_months, ax=ax1, palette='Set3')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    ax1.set_title('Price Distribution by Month')
    ax1.set_ylabel('Price (€)')
    ax1.set_xlabel('')

    existing_days = [d for d in dow_order if d in df['dow'].values]
    sns.boxplot(data=df, x='dow', y='price', order=existing_days, ax=ax2, palette='Set2')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
    ax2.set_title('Price Distribution by Day of Week')
    ax2.set_ylabel('Price (€)')
    ax2.set_xlabel('')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'seasonal_price_trends.png'), dpi=150)
    plt.close()

    # --- 4d. Listing duration analysis ---
    print("\nListing duration stats:")
    dur = df['listing_duration_days'].describe()
    print(f"  Mean: {dur['mean']:.1f} days, Median: {dur['50%']:.1f} days")
    print(f"  90th percentile: {df['listing_duration_days'].quantile(0.9):.0f} days")

    # Duration vs price
    corr, pval = stats.pearsonr(
        df['listing_duration_days'].dropna(),
        df.loc[df['listing_duration_days'].notna(), 'price']
    )
    print(f"  Correlation with price: r={corr:.3f} (p={pval:.4f})")

    # Duration by property type
    for ptype, group in df.groupby('property_type_HOUSE'):
        label = "Houses" if ptype else "Apartments"
        med = group['listing_duration_days'].median()
        print(f"  {label}: median {med:.0f} days")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    df['listing_duration_days'].clip(upper=365).hist(bins=60, ax=ax1, color='steelblue', edgecolor='white')
    ax1.axvline(df['listing_duration_days'].median(), color='red', linestyle='--', label=f'Median: {df["listing_duration_days"].median():.0f}d')
    ax1.set_title('Listing Duration Distribution')
    ax1.set_xlabel('Days Listed')
    ax1.legend()

    # Duration vs price scatter
    ax2.scatter(df['listing_duration_days'].clip(upper=365), df['price'] / 1000, alpha=0.1, s=5)
    ax2.set_xlabel('Days Listed')
    ax2.set_ylabel('Price (K€)')
    ax2.set_title('Listing Duration vs Price')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'listing_duration_analysis.png'), dpi=150)
    plt.close()

    # Clean up temp columns
    df.drop(columns=['dow', 'month_num', 'month_name'], inplace=True, errors='ignore')


# ============================================================================
# SECTION 5: BIDDIT & SALE TYPE ANALYSIS (from raw data)
# ============================================================================

def analyze_biddit(df_raw):
    """Analyze Biddit sales and other special sale types from raw parsed data."""
    print("\n" + "=" * 80)
    print("BIDDIT & SPECIAL SALE TYPE ANALYSIS")
    print("=" * 80)

    # Deduplicate raw data for fair comparison
    df_raw = df_raw.sort_values('date_obtained').drop_duplicates(subset='property_id', keep='last')
    print(f"\nDeduplicated raw data: {len(df_raw):,} rows")

    # Filter to Ghent region with valid prices
    mask = (
        df_raw['price'].notna() &
        (df_raw['price'] > 25000) &
        df_raw['latitude'].notna() &
        (df_raw['latitude'] >= 50.9) & (df_raw['latitude'] <= 51.2) &
        (df_raw['longitude'] >= 3.5) & (df_raw['longitude'] <= 4.0)
    )
    df_ghent = df_raw[mask].copy()
    print(f"Ghent region with valid prices: {len(df_ghent):,} rows")

    # --- 5a. Sale type breakdown ---
    print("\nSale type breakdown:")
    is_biddit = df_ghent['is_biddit_sale'] == True
    is_life_annuity = df_ghent['is_life_annuity_sale'] == True
    is_public = df_ghent['is_public_sale'] == True
    is_normal = ~(is_biddit | is_life_annuity | is_public)

    categories = {
        'Normal sale': is_normal,
        'Biddit sale': is_biddit,
        'Public sale': is_public,
        'Life annuity': is_life_annuity,
    }

    results = []
    for label, mask_cat in categories.items():
        subset = df_ghent[mask_cat]
        if len(subset) > 0:
            results.append({
                'type': label,
                'count': len(subset),
                'pct': len(subset) / len(df_ghent) * 100,
                'median_price': subset['price'].median(),
                'mean_price': subset['price'].mean(),
                'median_area': subset['livable_area'].median() if 'livable_area' in subset.columns else np.nan,
            })
            print(f"  {label:<20s}  n={len(subset):>5,}  ({len(subset)/len(df_ghent)*100:.1f}%)  "
                  f"median: {subset['price'].median():>10,.0f}€  mean: {subset['price'].mean():>10,.0f}€")

    # --- 5b. Biddit price comparison ---
    if is_biddit.sum() > 5:
        biddit_prices = df_ghent.loc[is_biddit, 'price']
        normal_prices = df_ghent.loc[is_normal, 'price']

        print(f"\nBiddit vs Normal price comparison:")
        print(f"  Biddit  median: {biddit_prices.median():>10,.0f}€  mean: {biddit_prices.mean():>10,.0f}€")
        print(f"  Normal  median: {normal_prices.median():>10,.0f}€  mean: {normal_prices.mean():>10,.0f}€")
        diff_pct = (biddit_prices.median() - normal_prices.median()) / normal_prices.median() * 100
        print(f"  Biddit median is {diff_pct:+.1f}% vs normal")

        # Statistical test
        stat, pval = stats.mannwhitneyu(biddit_prices, normal_prices, alternative='two-sided')
        print(f"  Mann-Whitney U test: p={pval:.4f} {'(significant)' if pval < 0.05 else '(not significant)'}")

        # Biddit property types
        print(f"\n  Biddit property subtypes:")
        for subtype, count in df_ghent.loc[is_biddit, 'property_subtype'].value_counts().head(5).items():
            print(f"    {subtype}: {count}")

        # Biddit building condition
        if 'building_condition' in df_ghent.columns:
            print(f"\n  Biddit building condition:")
            for cond, count in df_ghent.loc[is_biddit, 'building_condition'].value_counts().items():
                pct = count / is_biddit.sum() * 100
                print(f"    {str(cond):<25s} {count:>3d} ({pct:.0f}%)")

    # --- 5c. Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Sale type price distributions
    plot_data = []
    for label, mask_cat in categories.items():
        subset = df_ghent[mask_cat]
        if len(subset) > 5:
            for price in subset['price'].values:
                plot_data.append({'Sale Type': label, 'Price': price})
    plot_df = pd.DataFrame(plot_data)

    if not plot_df.empty:
        order = plot_df.groupby('Sale Type')['Price'].median().sort_values(ascending=False).index
        sns.boxplot(data=plot_df, x='Sale Type', y='Price', order=order, ax=axes[0], palette='Set2')
        axes[0].set_title('Price Distribution by Sale Type')
        axes[0].set_ylabel('Price (€)')
        axes[0].tick_params(axis='x', rotation=30)

    # Sale type counts
    counts = pd.Series({label: mask_cat.sum() for label, mask_cat in categories.items()})
    counts = counts[counts > 0]
    axes[1].pie(counts.values, labels=counts.index, autopct='%1.1f%%', colors=sns.color_palette('Set2'))
    axes[1].set_title('Sale Type Distribution')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'biddit_sale_types.png'), dpi=150)
    plt.close()


# ============================================================================
# SECTION 6: IMMO AGENCY ANALYSIS
# ============================================================================

def analyze_agencies(df):
    """Analyze real estate agency patterns."""
    print("\n" + "=" * 80)
    print("REAL ESTATE AGENCY ANALYSIS")
    print("=" * 80)

    agency_stats = df.groupby('immo_office_name').agg(
        count=('price', 'count'),
        median_price=('price', 'median'),
        mean_price=('price', 'mean'),
        median_area=('livable_area', 'median'),
        median_price_sq=('price_sq', 'median'),
        median_duration=('listing_duration_days', 'median'),
        pct_house=('property_type_HOUSE', 'mean'),
    ).sort_values('count', ascending=False)

    # --- 6a. Market concentration ---
    total = agency_stats['count'].sum()
    top5_share = agency_stats.head(5)['count'].sum() / total * 100
    top10_share = agency_stats.head(10)['count'].sum() / total * 100
    top20_share = agency_stats.head(20)['count'].sum() / total * 100
    print(f"\nMarket concentration:")
    print(f"  Total agencies: {len(agency_stats):,}")
    print(f"  Top 5 share:  {top5_share:.1f}%")
    print(f"  Top 10 share: {top10_share:.1f}%")
    print(f"  Top 20 share: {top20_share:.1f}%")

    # --- 6b. Top 20 agencies ---
    top20 = agency_stats.head(20)
    print(f"\nTop 20 agencies by volume:")
    print(f"  {'Agency':<40s} {'Count':>5s}  {'Med €':>10s}  {'Med €/m²':>8s}  {'Med Days':>8s}  {'% House':>7s}")
    print(f"  {'-'*40} {'-'*5}  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*7}")
    for name, row in top20.iterrows():
        print(f"  {str(name)[:40]:<40s} {int(row['count']):>5d}  "
              f"{row['median_price']:>10,.0f}  {row['median_price_sq']:>8,.0f}  "
              f"{row['median_duration']:>8.0f}  {row['pct_house']*100:>6.1f}%")

    # --- 6c. Price segment analysis ---
    # Classify agencies into price segments
    min_listings = 20
    active = agency_stats[agency_stats['count'] >= min_listings].copy()
    q33 = active['median_price'].quantile(0.33)
    q67 = active['median_price'].quantile(0.67)
    active['segment'] = pd.cut(active['median_price'], bins=[0, q33, q67, float('inf')],
                               labels=['Budget', 'Mid-range', 'Premium'])

    print(f"\nAgency price segments (agencies with >={min_listings} listings):")
    for seg in ['Budget', 'Mid-range', 'Premium']:
        seg_data = active[active['segment'] == seg]
        if len(seg_data) > 0:
            print(f"  {seg:<12s}: {len(seg_data):>3d} agencies, "
                  f"median price range: {seg_data['median_price'].min():,.0f}€ - {seg_data['median_price'].max():,.0f}€")

    # --- 6d. Speed analysis: which agencies sell fastest? ---
    fast_agencies = active.nsmallest(10, 'median_duration')
    slow_agencies = active.nlargest(10, 'median_duration')

    print(f"\nFastest-selling agencies (median listing duration):")
    for name, row in fast_agencies.iterrows():
        print(f"  {str(name)[:40]:<40s}  {row['median_duration']:>5.0f} days  (n={int(row['count'])})")

    print(f"\nSlowest-selling agencies:")
    for name, row in slow_agencies.iterrows():
        print(f"  {str(name)[:40]:<40s}  {row['median_duration']:>5.0f} days  (n={int(row['count'])})")

    # --- 6e. Correlation: does agency affect price beyond property characteristics? ---
    # Compare agency median price vs their median livable area
    if len(active) > 10:
        corr, pval = stats.pearsonr(active['median_price'], active['median_area'])
        print(f"\nAgency median_price vs median_area: r={corr:.3f} (p={pval:.4f})")
        print(f"  -> {'Agencies with pricier listings also have larger properties' if corr > 0.3 else 'Price differences partially independent of property size'}")

    # --- 6f. Plots ---
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    # Top 20 by volume
    top20_plot = agency_stats.head(20).sort_values('count')
    axes[0, 0].barh(range(len(top20_plot)), top20_plot['count'], color=sns.color_palette('viridis', len(top20_plot)))
    axes[0, 0].set_yticks(range(len(top20_plot)))
    axes[0, 0].set_yticklabels([str(n)[:35] for n in top20_plot.index], fontsize=9)
    axes[0, 0].set_xlabel('Number of Listings')
    axes[0, 0].set_title('Top 20 Agencies by Volume')

    # Top 20 median price
    top20_price = active.nlargest(20, 'median_price').sort_values('median_price')
    axes[0, 1].barh(range(len(top20_price)), top20_price['median_price'] / 1000,
                    color=sns.color_palette('magma', len(top20_price)))
    axes[0, 1].set_yticks(range(len(top20_price)))
    axes[0, 1].set_yticklabels([str(n)[:35] for n in top20_price.index], fontsize=9)
    axes[0, 1].set_xlabel('Median Price (K€)')
    axes[0, 1].set_title('Top 20 Agencies by Median Price')

    # Duration by top agencies
    top15 = agency_stats.head(15)
    top15_sorted = top15.sort_values('median_duration')
    axes[1, 0].barh(range(len(top15_sorted)), top15_sorted['median_duration'],
                    color=sns.color_palette('cool', len(top15_sorted)))
    axes[1, 0].set_yticks(range(len(top15_sorted)))
    axes[1, 0].set_yticklabels([str(n)[:35] for n in top15_sorted.index], fontsize=9)
    axes[1, 0].set_xlabel('Median Listing Duration (days)')
    axes[1, 0].set_title('Listing Duration by Top 15 Agencies')

    # Price vs duration scatter for agencies
    axes[1, 1].scatter(active['median_price'] / 1000, active['median_duration'],
                       s=active['count'] * 2, alpha=0.5, c=active['pct_house'], cmap='RdYlBu')
    axes[1, 1].set_xlabel('Median Price (K€)')
    axes[1, 1].set_ylabel('Median Duration (days)')
    axes[1, 1].set_title('Agency Price vs Duration\n(size=volume, color=% houses)')
    plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1], label='% Houses')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'agency_analysis.png'), dpi=150)
    plt.close()


# ============================================================================
# SECTION 7: SPATIAL ANALYSIS
# ============================================================================

def analyze_spatial(df):
    """Neighborhood and spatial patterns."""
    print("\n" + "=" * 80)
    print("SPATIAL ANALYSIS")
    print("=" * 80)

    # --- 7a. Neighbourhood price analysis ---
    nbh = df.groupby('neighbourhood').agg(
        count=('price', 'count'),
        median_price=('price', 'median'),
        median_price_sq=('price_sq', 'median'),
        median_area=('livable_area', 'median'),
        pct_house=('property_type_HOUSE', 'mean'),
        median_age=('age', 'median'),
    ).sort_values('median_price', ascending=False)

    print(f"\nNeighbourhood analysis ({len(nbh)} areas):")
    print(f"  {'Neighbourhood':<50s} {'N':>5s}  {'Med €':>10s}  {'Med €/m²':>8s}  {'Med m²':>6s}  {'Med Age':>7s}  {'% House':>7s}")
    print(f"  {'-'*50} {'-'*5}  {'-'*10}  {'-'*8}  {'-'*6}  {'-'*7}  {'-'*7}")
    for name, row in nbh.iterrows():
        print(f"  {str(name)[:50]:<50s} {int(row['count']):>5d}  "
              f"{row['median_price']:>10,.0f}  {row['median_price_sq']:>8,.0f}  "
              f"{row['median_area']:>6.0f}  {row['median_age']:>7.0f}  {row['pct_house']*100:>6.1f}%")

    # --- 7b. Distance from center effect ---
    corr_dist, pval_dist = stats.pearsonr(df['dist_from_center'], df['price'])
    corr_dist_sq, pval_dist_sq = stats.pearsonr(df['dist_from_center'], df['price_sq'])
    print(f"\nDistance from center effects:")
    print(f"  Price vs distance:    r={corr_dist:.3f} (p={pval_dist:.4f})")
    print(f"  Price/m² vs distance: r={corr_dist_sq:.3f} (p={pval_dist_sq:.4f})")

    # Price by distance bins
    df_temp = df.copy()
    df_temp['dist_bin'] = pd.cut(df_temp['dist_from_center'], bins=10)
    dist_price = df_temp.groupby('dist_bin', observed=True)['price'].agg(['median', 'count'])
    print(f"\n  Price by distance from center:")
    for dist_bin, row in dist_price.iterrows():
        if row['count'] >= 10:
            print(f"    {str(dist_bin):<25s}  median: {row['median']:>10,.0f}€  (n={int(row['count'])})")

    # --- 7c. Street type effect ---
    street_stats = df.groupby('street_type').agg(
        count=('price', 'count'),
        median_price=('price', 'median'),
        median_price_sq=('price_sq', 'median'),
    ).sort_values('median_price', ascending=False)

    print(f"\nStreet type effects:")
    for stype, row in street_stats.iterrows():
        if row['count'] >= 10:
            print(f"  {str(stype):<20s}  median: {row['median_price']:>10,.0f}€  "
                  f"€/m²: {row['median_price_sq']:>6,.0f}  (n={int(row['count'])})")

    # --- 7d. Plots ---
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    # Neighbourhood median prices (top 20 by volume)
    top_nbh = nbh[nbh['count'] >= 50].sort_values('median_price')
    axes[0, 0].barh(range(len(top_nbh)), top_nbh['median_price'] / 1000,
                    color=sns.color_palette('viridis', len(top_nbh)))
    axes[0, 0].set_yticks(range(len(top_nbh)))
    axes[0, 0].set_yticklabels(top_nbh.index, fontsize=9)
    axes[0, 0].set_xlabel('Median Price (K€)')
    axes[0, 0].set_title('Median Price by Neighbourhood (n≥50)')

    # Price per m² by neighbourhood
    top_nbh_sq = nbh[nbh['count'] >= 50].sort_values('median_price_sq')
    axes[0, 1].barh(range(len(top_nbh_sq)), top_nbh_sq['median_price_sq'],
                    color=sns.color_palette('magma', len(top_nbh_sq)))
    axes[0, 1].set_yticks(range(len(top_nbh_sq)))
    axes[0, 1].set_yticklabels(top_nbh_sq.index, fontsize=9)
    axes[0, 1].set_xlabel('Median Price/m² (€)')
    axes[0, 1].set_title('Median Price/m² by Neighbourhood (n≥50)')

    # Distance from center vs price
    axes[1, 0].scatter(df['dist_from_center'], df['price'] / 1000, alpha=0.05, s=5, c='steelblue')
    # Rolling median
    df_sorted = df.sort_values('dist_from_center')
    rolling_med = df_sorted['price'].rolling(200, center=True).median()
    axes[1, 0].plot(df_sorted['dist_from_center'], rolling_med / 1000, 'r-', linewidth=2, label='Rolling median (200)')
    axes[1, 0].set_xlabel('Distance from Center (km)')
    axes[1, 0].set_ylabel('Price (K€)')
    axes[1, 0].set_title('Price vs Distance from City Center')
    axes[1, 0].legend()

    # Street type boxplot
    street_order = street_stats[street_stats['count'] >= 10].sort_values('median_price', ascending=False).index
    sns.boxplot(data=df[df['street_type'].isin(street_order)], x='street_type', y='price',
                order=street_order, ax=axes[1, 1], palette='Set2')
    axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), rotation=45, ha='right')
    axes[1, 1].set_title('Price by Street Type')
    axes[1, 1].set_ylabel('Price (€)')
    axes[1, 1].set_xlabel('')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'spatial_analysis.png'), dpi=150)
    plt.close()


# ============================================================================
# SECTION 8: PROPERTY CHARACTERISTICS DEEP DIVE
# ============================================================================

def analyze_property_characteristics(df):
    """Deep dive into key property features."""
    print("\n" + "=" * 80)
    print("PROPERTY CHARACTERISTICS DEEP DIVE")
    print("=" * 80)

    # --- 8a. House vs Apartment ---
    for is_house, label in [(True, 'Houses'), (False, 'Apartments')]:
        subset = df[df['property_type_HOUSE'] == is_house]
        print(f"\n{label} (n={len(subset):,}):")
        print(f"  Median price: {subset['price'].median():,.0f}€  Mean: {subset['price'].mean():,.0f}€")
        print(f"  Median area:  {subset['livable_area'].median():.0f}m²")
        print(f"  Median €/m²:  {subset['price_sq'].median():,.0f}")
        print(f"  Median age:   {subset['age'].median():.0f} years")
        print(f"  % newly built:{subset['is_newly_built'].mean()*100:.1f}%")

    # --- 8b. Building condition effect ---
    cond_stats = df.groupby('building_condition').agg(
        count=('price', 'count'),
        median_price=('price', 'median'),
        median_price_sq=('price_sq', 'median'),
        median_age=('age', 'median'),
    ).sort_values('median_price', ascending=False)

    print(f"\nBuilding condition effects:")
    for cond, row in cond_stats.iterrows():
        print(f"  {str(cond):<25s}  median: {row['median_price']:>10,.0f}€  "
              f"€/m²: {row['median_price_sq']:>6,.0f}  age: {row['median_age']:>3.0f}y  (n={int(row['count'])})")

    # --- 8c. Energy label effect ---
    energy_stats = df.groupby('energy_label').agg(
        count=('price', 'count'),
        median_price=('price', 'median'),
        median_price_sq=('price_sq', 'median'),
    )
    label_order = ['A++', 'A+', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'UNKNOWN']
    energy_stats = energy_stats.reindex([l for l in label_order if l in energy_stats.index])

    print(f"\nEnergy label effects:")
    for label, row in energy_stats.iterrows():
        bar = "█" * int(row['median_price_sq'] / energy_stats['median_price_sq'].max() * 25)
        print(f"  {str(label):<8s}  median: {row['median_price']:>10,.0f}€  "
              f"€/m²: {row['median_price_sq']:>6,.0f}  (n={int(row['count']):>4,})  {bar}")

    # --- 8d. New construction premium ---
    new = df[df['is_newly_built'] == True]
    old = df[df['is_newly_built'] == False]
    if len(new) > 20 and len(old) > 20:
        print(f"\nNew construction premium:")
        print(f"  Newly built: median {new['price'].median():>10,.0f}€  €/m²: {new['price_sq'].median():>6,.0f}  (n={len(new)})")
        print(f"  Existing:    median {old['price'].median():>10,.0f}€  €/m²: {old['price_sq'].median():>6,.0f}  (n={len(old)})")
        premium_pct = (new['price_sq'].median() - old['price_sq'].median()) / old['price_sq'].median() * 100
        print(f"  New build premium per m²: {premium_pct:+.1f}%")

    # --- 8e. Key scatter plots ---
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # Price vs Area by type
    for is_house, color, label in [(True, 'steelblue', 'House'), (False, 'coral', 'Apartment')]:
        subset = df[df['property_type_HOUSE'] == is_house]
        axes[0, 0].scatter(subset['livable_area'], subset['price'] / 1000, alpha=0.15, s=8, label=label, color=color)
    axes[0, 0].set_xlabel('Livable Area (m²)')
    axes[0, 0].set_ylabel('Price (K€)')
    axes[0, 0].set_title('Price vs Area by Property Type')
    axes[0, 0].legend()
    axes[0, 0].set_xlim(0, 500)

    # Building condition boxplot
    cond_order = cond_stats.index.tolist()
    sns.boxplot(data=df, x='building_condition', y='price', order=cond_order,
                ax=axes[0, 1], palette='RdYlGn_r')
    axes[0, 1].set_xticklabels(axes[0, 1].get_xticklabels(), rotation=45, ha='right')
    axes[0, 1].set_title('Price by Building Condition')
    axes[0, 1].set_ylabel('Price (€)')
    axes[0, 1].set_xlabel('')

    # Energy label boxplot
    existing_labels = [l for l in label_order if l in df['energy_label'].values]
    sns.boxplot(data=df, x='energy_label', y='price', order=existing_labels,
                ax=axes[0, 2], palette='RdYlGn_r')
    axes[0, 2].set_title('Price by Energy Label')
    axes[0, 2].set_ylabel('Price (€)')

    # Number of bedrooms vs price
    bed_data = df[df['number_of_bedrooms'] <= 6]
    sns.boxplot(data=bed_data, x='number_of_bedrooms', y='price', ax=axes[1, 0], palette='viridis')
    axes[1, 0].set_title('Price by Number of Bedrooms')
    axes[1, 0].set_ylabel('Price (€)')

    # Age vs price
    axes[1, 1].scatter(df['age'], df['price'] / 1000, alpha=0.05, s=5, c='steelblue')
    df_age_sorted = df.sort_values('age')
    rolling_med = df_age_sorted['price'].rolling(200, center=True).median()
    axes[1, 1].plot(df_age_sorted['age'], rolling_med / 1000, 'r-', linewidth=2, label='Rolling median')
    axes[1, 1].set_xlabel('Building Age (years)')
    axes[1, 1].set_ylabel('Price (K€)')
    axes[1, 1].set_title('Price vs Building Age')
    axes[1, 1].legend()

    # New build premium
    new_old_data = df.copy()
    new_old_data['Build Type'] = new_old_data['is_newly_built'].map({True: 'New Build', False: 'Existing'})
    sns.boxplot(data=new_old_data, x='Build Type', y='price_sq', ax=axes[1, 2], palette='Set2')
    axes[1, 2].set_title('Price/m² by Build Type')
    axes[1, 2].set_ylabel('Price per m² (€)')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'property_characteristics.png'), dpi=150)
    plt.close()


# ============================================================================
# SECTION 9: MULTI-FEATURE INTERACTION ANALYSIS
# ============================================================================

def analyze_interactions(df):
    """Analyze interesting feature interactions."""
    print("\n" + "=" * 80)
    print("FEATURE INTERACTIONS")
    print("=" * 80)

    # --- 9a. Condition × Energy label ---
    print("\nMedian price by Condition × Energy Label:")
    pivot = df.pivot_table(values='price', index='building_condition',
                          columns='energy_label', aggfunc='median')
    label_order = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    existing = [l for l in label_order if l in pivot.columns]
    pivot = pivot[existing]
    print(pivot.to_string(float_format=lambda x: f'{x:,.0f}'))

    # --- 9b. Floor number effect (apartments) ---
    apts = df[(df['property_type_HOUSE'] == False) & (df['floor_number'].notna())]
    if len(apts) > 50:
        floor_stats = apts.groupby('floor_number')['price'].agg(['median', 'count'])
        floor_stats = floor_stats[floor_stats['count'] >= 10]
        print(f"\nFloor number effect on apartment price:")
        for floor, row in floor_stats.iterrows():
            print(f"  Floor {int(floor):>2d}: median {row['median']:>10,.0f}€  (n={int(row['count'])})")

    # --- 9c. Facade count effect ---
    if 'facade_count' in df.columns:
        facade_stats = df[df['facade_count'].notna()].groupby('facade_count')['price'].agg(['median', 'count'])
        facade_stats = facade_stats[facade_stats['count'] >= 20]
        if len(facade_stats) > 1:
            print(f"\nFacade count effect:")
            for fc, row in facade_stats.iterrows():
                print(f"  {int(fc)} facades: median {row['median']:>10,.0f}€  (n={int(row['count'])})")

    # --- 9d. Garden/terrace premium ---
    for feature, name in [('has_garden', 'Garden'), ('has_terrace', 'Terrace'),
                          ('has_swimming_pool', 'Swimming Pool')]:
        with_feat = df[df[feature] == True]['price']
        without_feat = df[df[feature] == False]['price']
        if len(with_feat) > 20 and len(without_feat) > 20:
            diff = with_feat.median() - without_feat.median()
            pct = diff / without_feat.median() * 100
            print(f"\n  {name} premium: {diff:+,.0f}€ ({pct:+.1f}%)")
            print(f"    With:    median {with_feat.median():>10,.0f}€  (n={len(with_feat)})")
            print(f"    Without: median {without_feat.median():>10,.0f}€  (n={len(without_feat)})")

    # --- 9e. Interaction heatmap ---
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Condition x Energy heatmap
    pivot_sq = df.pivot_table(values='price_sq', index='building_condition',
                             columns='energy_label', aggfunc='median')
    existing = [l for l in label_order if l in pivot_sq.columns]
    pivot_sq = pivot_sq[existing]
    sns.heatmap(pivot_sq, annot=True, fmt=',.0f', cmap='YlOrRd', ax=axes[0])
    axes[0].set_title('Median €/m² by Condition × Energy Label')

    # Neighbourhood x Property type
    nbh_type = df.pivot_table(values='price', index='neighbourhood',
                             columns='property_type_HOUSE', aggfunc='median')
    nbh_type.columns = ['Apartment', 'House']
    nbh_type = nbh_type.dropna()
    nbh_type['spread'] = nbh_type['House'] - nbh_type['Apartment']
    nbh_type = nbh_type.sort_values('spread', ascending=False).head(15)

    x = range(len(nbh_type))
    axes[1].barh([i - 0.2 for i in x], nbh_type['House'] / 1000, height=0.4, label='House', color='steelblue')
    axes[1].barh([i + 0.2 for i in x], nbh_type['Apartment'] / 1000, height=0.4, label='Apartment', color='coral')
    axes[1].set_yticks(list(x))
    axes[1].set_yticklabels(nbh_type.index, fontsize=9)
    axes[1].set_xlabel('Median Price (K€)')
    axes[1].set_title('House vs Apartment Price Gap by Neighbourhood')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'feature_interactions.png'), dpi=150)
    plt.close()


# ============================================================================
# SECTION 10: INTEREST RATE vs ASKING PRICE ANALYSIS
# ============================================================================

def fetch_ecb_rates():
    """Fetch Euribor rates from ECB Statistical Data Warehouse API."""
    import urllib.request
    import csv
    import io

    def _fetch(url, name):
        req = urllib.request.Request(url, headers={'Accept': 'text/csv'})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = resp.read().decode()
        reader = csv.DictReader(io.StringIO(data))
        result = {}
        for row in reader:
            result[row['TIME_PERIOD']] = float(row['OBS_VALUE'])
        return result

    euribor_12m = _fetch(
        'https://data-api.ecb.europa.eu/service/data/FM/M.U2.EUR.RT.MM.EURIBOR1YD_.HSTA?startPeriod=2024-01&endPeriod=2026-03&format=csvdata',
        'Euribor 12m'
    )
    euribor_3m = _fetch(
        'https://data-api.ecb.europa.eu/service/data/FM/M.U2.EUR.RT.MM.EURIBOR3MD_.HSTA?startPeriod=2024-01&endPeriod=2026-03&format=csvdata',
        'Euribor 3m'
    )

    # Build a DataFrame
    rates_df = pd.DataFrame([
        {'year_month': period, 'euribor_12m': euribor_12m.get(period), 'euribor_3m': euribor_3m.get(period)}
        for period in sorted(set(list(euribor_12m.keys()) + list(euribor_3m.keys())))
    ])
    rates_df['year_month'] = pd.to_datetime(rates_df['year_month'])
    return rates_df


def analyze_interest_rates(df):
    """Analyze relationship between ECB interest rates and asking prices."""
    print("\n" + "=" * 80)
    print("INTEREST RATE vs ASKING PRICE ANALYSIS")
    print("=" * 80)

    try:
        rates_df = fetch_ecb_rates()
    except Exception as e:
        print(f"  Could not fetch ECB rates: {e}")
        return

    print(f"\nFetched {len(rates_df)} months of Euribor data")
    print(f"  Euribor 12m range: {rates_df['euribor_12m'].min():.2f}% - {rates_df['euribor_12m'].max():.2f}%")
    print(f"  Euribor 3m range:  {rates_df['euribor_3m'].min():.2f}% - {rates_df['euribor_3m'].max():.2f}%")

    # Merge rates with property data by month
    df_temp = df.copy()
    df_temp['year_month'] = df_temp['date_obtained'].dt.to_period('M').dt.to_timestamp()
    df_merged = df_temp.merge(rates_df, on='year_month', how='left')
    df_merged = df_merged.dropna(subset=['euribor_12m'])
    print(f"  Matched {len(df_merged):,} properties with rate data ({len(df_merged)/len(df)*100:.1f}%)")

    # --- Monthly aggregation ---
    monthly = df_merged.groupby('year_month').agg(
        median_price=('price', 'median'),
        mean_price=('price', 'mean'),
        median_price_sq=('price_sq', 'median'),
        count=('price', 'count'),
        euribor_12m=('euribor_12m', 'first'),
        euribor_3m=('euribor_3m', 'first'),
    ).sort_index()
    monthly = monthly[monthly['count'] >= 30]

    print(f"\nMonthly price vs Euribor 12m rate:")
    print(f"  {'Month':<10s}  {'Euribor 12m':>11s}  {'Median Price':>12s}  {'Med €/m²':>10s}  {'Count':>5s}")
    print(f"  {'-'*10}  {'-'*11}  {'-'*12}  {'-'*10}  {'-'*5}")
    for date, row in monthly.iterrows():
        print(f"  {date.strftime('%Y-%m'):<10s}  {row['euribor_12m']:>10.2f}%  "
              f"{row['median_price']:>12,.0f}  {row['median_price_sq']:>10,.0f}  {int(row['count']):>5,}")

    # --- Correlation tests ---
    if len(monthly) > 5:
        # Pearson correlation between rates and prices
        corr_price, p_price = stats.pearsonr(monthly['euribor_12m'], monthly['median_price'])
        corr_sq, p_sq = stats.pearsonr(monthly['euribor_12m'], monthly['median_price_sq'])
        corr_vol, p_vol = stats.pearsonr(monthly['euribor_12m'], monthly['count'])

        print(f"\nCorrelation analysis (monthly aggregates, n={len(monthly)}):")
        print(f"  Euribor 12m vs median price:    r={corr_price:+.3f} (p={p_price:.4f})"
              f" {'*' if p_price < 0.05 else ''}")
        print(f"  Euribor 12m vs median €/m²:     r={corr_sq:+.3f} (p={p_sq:.4f})"
              f" {'*' if p_sq < 0.05 else ''}")
        print(f"  Euribor 12m vs listing volume:   r={corr_vol:+.3f} (p={p_vol:.4f})"
              f" {'*' if p_vol < 0.05 else ''}")

        if corr_price < -0.1 and p_price < 0.1:
            print(f"  -> As rates fell, asking prices tended to rise (inverse relationship)")
        elif corr_price > 0.1 and p_price < 0.1:
            print(f"  -> As rates fell, asking prices also fell (positive relationship)")
        else:
            print(f"  -> No clear directional relationship between rates and asking prices")

    # --- Individual property-level correlation ---
    corr_ind, p_ind = stats.pearsonr(df_merged['euribor_12m'], df_merged['price'])
    corr_ind_sq, p_ind_sq = stats.pearsonr(df_merged['euribor_12m'], df_merged['price_sq'])
    print(f"\nIndividual property-level correlation (n={len(df_merged):,}):")
    print(f"  Euribor 12m vs price:    r={corr_ind:+.3f} (p={p_ind:.4f})")
    print(f"  Euribor 12m vs €/m²:     r={corr_ind_sq:+.3f} (p={p_ind_sq:.4f})")

    # --- Estimate monthly mortgage cost impact ---
    # Typical Belgian mortgage: 80% LTV, 25 years
    raw_median = df_merged['price'].median()
    # Detect if prices are in K€ (median < 5000) or EUR (median > 5000)
    typical_price = raw_median * 1000 if raw_median < 5000 else raw_median

    loan_amount = typical_price * 0.80  # 80% LTV
    term_months = 25 * 12

    rate_high = rates_df['euribor_12m'].max() / 100 + 0.015  # Euribor + 1.5% bank margin
    rate_low = rates_df['euribor_12m'].min() / 100 + 0.015

    def monthly_payment(principal, annual_rate, months):
        r = annual_rate / 12
        if r == 0:
            return principal / months
        return principal * (r * (1 + r) ** months) / ((1 + r) ** months - 1)

    payment_high = monthly_payment(loan_amount, rate_high, term_months)
    payment_low = monthly_payment(loan_amount, rate_low, term_months)

    print(f"\nMortgage cost impact (median property {typical_price:,.0f}€, 80% LTV, 25yr):")
    print(f"  At peak rate ({rate_high*100:.1f}% = Euribor {rates_df['euribor_12m'].max():.1f}% + 1.5% margin):")
    print(f"    Monthly payment: {payment_high:,.0f}€")
    print(f"  At lowest rate ({rate_low*100:.1f}% = Euribor {rates_df['euribor_12m'].min():.1f}% + 1.5% margin):")
    print(f"    Monthly payment: {payment_low:,.0f}€")
    print(f"  Savings from rate drop: {payment_high - payment_low:,.0f}€/month ({(payment_high - payment_low) * term_months:,.0f}€ total)")

    # --- Plots ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Dual axis: price trend + interest rate
    ax1 = axes[0, 0]
    color1 = 'steelblue'
    ax1.plot(monthly.index, monthly['median_price'] / 1000, 'o-', color=color1, linewidth=2, markersize=5)
    ax1.set_ylabel('Median Asking Price (K€)', color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_title('Asking Prices vs Euribor 12-Month Rate')
    ax1r = ax1.twinx()
    color2 = 'firebrick'
    ax1r.plot(monthly.index, monthly['euribor_12m'], 's--', color=color2, linewidth=2, markersize=5)
    ax1r.set_ylabel('Euribor 12m Rate (%)', color=color2)
    ax1r.tick_params(axis='y', labelcolor=color2)
    ax1r.invert_yaxis()  # Invert so falling rates go up visually

    # 2. Price/m² vs rate
    ax2 = axes[0, 1]
    ax2.scatter(df_merged['euribor_12m'], df_merged['price_sq'], alpha=0.03, s=5, c='steelblue')
    # Add monthly medians
    ax2.scatter(monthly['euribor_12m'], monthly['median_price_sq'], s=100, c='red',
                zorder=5, edgecolors='black', label='Monthly median')
    # Trend line on monthly
    if len(monthly) > 3:
        z = np.polyfit(monthly['euribor_12m'], monthly['median_price_sq'], 1)
        p = np.poly1d(z)
        x_range = np.linspace(monthly['euribor_12m'].min(), monthly['euribor_12m'].max(), 50)
        ax2.plot(x_range, p(x_range), 'r--', linewidth=2, label=f'Trend (r={corr_sq:+.2f})')
    ax2.set_xlabel('Euribor 12m Rate (%)')
    ax2.set_ylabel('Price per m² (€)')
    ax2.set_title('Price/m² vs Interest Rate')
    ax2.legend()

    # 3. Volume vs rate
    ax3 = axes[1, 0]
    color1 = 'steelblue'
    ax3.bar(monthly.index, monthly['count'], width=20, color=color1, alpha=0.7)
    ax3.set_ylabel('Monthly Listings', color=color1)
    ax3.tick_params(axis='y', labelcolor=color1)
    ax3.set_title('Listing Volume vs Interest Rate')
    ax3r = ax3.twinx()
    ax3r.plot(monthly.index, monthly['euribor_12m'], 's--', color='firebrick', linewidth=2, markersize=5)
    ax3r.set_ylabel('Euribor 12m Rate (%)', color='firebrick')
    ax3r.tick_params(axis='y', labelcolor='firebrick')

    # 4. Monthly mortgage payment simulation
    ax4 = axes[1, 1]
    rates_for_plot = rates_df.dropna(subset=['euribor_12m']).copy()
    rates_for_plot['mortgage_rate'] = rates_for_plot['euribor_12m'] / 100 + 0.015
    rates_for_plot['monthly_payment'] = rates_for_plot['mortgage_rate'].apply(
        lambda r: monthly_payment(loan_amount, r, term_months)
    )
    ax4.fill_between(rates_for_plot['year_month'], rates_for_plot['monthly_payment'],
                     alpha=0.3, color='steelblue')
    ax4.plot(rates_for_plot['year_month'], rates_for_plot['monthly_payment'],
             'o-', color='steelblue', linewidth=2, markersize=4)
    ax4.set_ylabel('Monthly Payment (€)')
    ax4.set_title(f'Estimated Monthly Mortgage Payment\n(Median property {typical_price:,.0f}€, 80% LTV, 25yr)')
    ax4.set_xlabel('Date')

    for ax in [axes[0, 0], axes[1, 0], ax4]:
        ax.tick_params(axis='x', rotation=30)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'interest_rate_analysis.png'), dpi=150)
    plt.close()
    print(f"\n  Plot saved: interest_rate_analysis.png")


# ============================================================================
# SECTION 11: KEY FINDINGS SUMMARY
# ============================================================================

def summarize_findings(df, corr_with_price, importances):
    """Print a summary of the most actionable findings."""
    print("\n" + "=" * 80)
    print("KEY FINDINGS SUMMARY")
    print("=" * 80)

    print(f"""
Dataset: {len(df):,} properties in Ghent region
  - {df['property_type_HOUSE'].sum():,} houses, {(~df['property_type_HOUSE']).sum():,} apartments
  - Price range: {df['price'].min():,.0f}€ - {df['price'].max():,.0f}€ (median: {df['price'].median():,.0f}€)
  - Area range: {df['livable_area'].min():.0f} - {df['livable_area'].max():.0f} m²
  - Date range: {df['date_obtained'].min().strftime('%Y-%m-%d')} to {df['date_obtained'].max().strftime('%Y-%m-%d')}
""")

    if corr_with_price is not None:
        print("Top price drivers (correlation):")
        for feat, val in corr_with_price.head(5).items():
            print(f"  - {feat}: r={val:.3f}")

    if importances is not None:
        print("\nTop price drivers (XGBoost importance):")
        for feat, val in importances.head(5).items():
            print(f"  - {feat}: {val:.4f}")

    print("\nAll plots saved to: outputs/eda_plots/")
    print("  - correlation_matrix.png")
    print("  - feature_importance.png")
    print("  - time_trends.png")
    print("  - seasonal_price_trends.png")
    print("  - listing_duration_analysis.png")
    print("  - biddit_sale_types.png")
    print("  - agency_analysis.png")
    print("  - spatial_analysis.png")
    print("  - property_characteristics.png")
    print("  - feature_interactions.png")


# ============================================================================
# MAIN
# ============================================================================

def main():
    df, df_raw = load_data()

    corr_with_price = analyze_correlations(df)
    importances = analyze_feature_importance(df)
    analyze_temporal(df)
    analyze_biddit(df_raw)
    analyze_agencies(df)
    analyze_spatial(df)
    analyze_property_characteristics(df)
    analyze_interactions(df)
    analyze_interest_rates(df)
    summarize_findings(df, corr_with_price, importances)


if __name__ == "__main__":
    main()
