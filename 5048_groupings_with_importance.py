"""
✅ ENHANCED VERSION: SOM Analysis with Attribute Importance
============================================================================
Includes:
- Fast period detection using MiniSom
- Realistic macro-group classification (A, B, C, D)
- Attribute importance calculation for each period
- All 7 attributes in output CSVs
"""

import pandas as pd
import numpy as np
from scipy.stats import f_oneway, wasserstein_distance
from minisom import MiniSom
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# ATTRIBUTE IMPORTANCE CALCULATION
# ============================================================================

def calculate_attribute_importance(country_profile, mapping_df, feature_cols):
    """
    Calculate attribute importance for macro-group classification
    
    Methods:
    1. ANOVA F-statistic: Between-group variance / Within-group variance
    2. Eta-squared: Proportion of variance explained by grouping
    3. Standardized range: How much groups differ on each feature
    
    Returns: DataFrame with importance scores for each attribute
    """
    
    importance_results = []
    
    # Get macro group assignments
    country_groups = mapping_df.set_index('country')['macro_group_id']
    
    for feature in feature_cols:
        # Get feature values for all countries
        feature_values = country_profile[feature].copy()
        
        # Match with group assignments
        valid_countries = feature_values.dropna().index.intersection(country_groups.index)
        
        if len(valid_countries) < 10:  # Need enough data
            continue
            
        values = feature_values.loc[valid_countries].values
        groups = country_groups.loc[valid_countries].values
        
        # Method 1: ANOVA F-statistic
        group_data = {}
        for group_id in ['A', 'B', 'C', 'D']:
            group_data[group_id] = values[groups == group_id]
        
        # Only include groups that have data
        group_lists = [data for data in group_data.values() if len(data) > 0]
        
        if len(group_lists) > 1:
            f_stat, p_value = f_oneway(*group_lists)
        else:
            f_stat, p_value = 0, 1
        
        # Method 2: Eta-squared (effect size)
        # SS_between / SS_total
        grand_mean = np.mean(values)
        ss_total = np.sum((values - grand_mean) ** 2)
        
        ss_between = 0
        for group_id in ['A', 'B', 'C', 'D']:
            group_values = values[groups == group_id]
            if len(group_values) > 0:
                group_mean = np.mean(group_values)
                ss_between += len(group_values) * (group_mean - grand_mean) ** 2
        
        eta_squared = ss_between / ss_total if ss_total > 0 else 0
        
        # Method 3: Standardized range (max group mean - min group mean) / overall std
        group_means = []
        for group_id in ['A', 'B', 'C', 'D']:
            group_values = values[groups == group_id]
            if len(group_values) > 0:
                group_means.append(np.mean(group_values))
        
        if len(group_means) > 1:
            std_range = (max(group_means) - min(group_means)) / (np.std(values) + 1e-10)
        else:
            std_range = 0
        
        # Composite importance score (0-100)
        # Normalize each metric and combine
        importance_score = (
            min(f_stat / 100, 1) * 0.4 +  # F-statistic contribution
            eta_squared * 0.4 +             # Variance explained contribution  
            min(std_range / 5, 1) * 0.2     # Standardized range contribution
        ) * 100
        
        importance_results.append({
            'attribute': feature,
            'f_statistic': f_stat,
            'p_value': p_value,
            'eta_squared': eta_squared,
            'variance_explained_pct': eta_squared * 100,
            'standardized_range': std_range,
            'importance_score': importance_score,
            'rank': 0  # Will be filled later
        })
    
    importance_df = pd.DataFrame(importance_results)
    
    # Rank by importance score
    importance_df = importance_df.sort_values('importance_score', ascending=False)
    importance_df['rank'] = range(1, len(importance_df) + 1)
    
    return importance_df

# ============================================================================
# DATA LOADING
# ============================================================================

def load_and_prepare_data(filepath='5048_data.xlsx'):
    """Load data without imputation"""
    print("="*80)
    print("LOADING DATA")
    print("="*80)
    
    df = pd.read_excel(filepath)
    
    feature_cols = [
        'gdp_current_usd', 'inflation_cpi_pct', 'unemployment_pct',
        'trade_pct_gdp', 'current_account_pct_gdp',
        'gdp_per_capita_current_usd', 'fdi_net_inflows_pct_gdp'
    ]
    
    df_clean = df[['country_name', 'year'] + feature_cols].copy()
    
    print(f"Records: {len(df_clean)}")
    print(f"Countries: {df_clean['country_name'].nunique()}")
    print(f"Years: {df_clean['year'].min()}-{df_clean['year'].max()}")
    
    # Apply log transforms (but keep NaN)
    df_clean['gdp_per_capita_log'] = np.log1p(df_clean['gdp_per_capita_current_usd'].fillna(0))
    df_clean.loc[df_clean['gdp_per_capita_current_usd'].isna(), 'gdp_per_capita_log'] = np.nan
    
    df_clean['gdp_total_log'] = np.log1p(df_clean['gdp_current_usd'].fillna(0))
    df_clean.loc[df_clean['gdp_current_usd'].isna(), 'gdp_total_log'] = np.nan
    
    df_clean = df_clean.drop(['gdp_per_capita_current_usd', 'gdp_current_usd'], axis=1)
    
    feature_cols_final = [
        'gdp_per_capita_log', 'inflation_cpi_pct', 'unemployment_pct',
        'trade_pct_gdp', 'fdi_net_inflows_pct_gdp',
        'current_account_pct_gdp', 'gdp_total_log'
    ]
    
    # Build yearly profiles WITHOUT imputation
    yearly_profiles = {}
    for year in sorted(df_clean['year'].unique()):
        year_data = df_clean[df_clean['year'] == year].copy()
        country_data = year_data.set_index('country_name')[feature_cols_final]
        country_data = country_data[country_data.notna().sum(axis=1) > 0]
        
        if len(country_data) < 30:
            continue
        
        countries = country_data.index.tolist()
        
        # Scale without imputation
        scaled_data = np.zeros_like(country_data.values)
        for j, col in enumerate(feature_cols_final):
            col_data = country_data[col].values
            mask = ~np.isnan(col_data)
            if np.sum(mask) > 0:
                available_values = col_data[mask]
                median = np.median(available_values)
                q25, q75 = np.percentile(available_values, [25, 75])
                iqr = q75 - q25
                if iqr > 0:
                    scaled_values = (col_data - median) / iqr
                    scaled_data[:, j] = scaled_values
        
        scaled_data[country_data.isna().values] = np.nan
        
        yearly_profiles[year] = {
            'scaled_data': scaled_data,
            'countries': countries,
            'raw_data': country_data,
            'n_countries': len(countries)
        }
    
    print(f"✓ Processed {len(yearly_profiles)} years")
    
    return df_clean, feature_cols_final, yearly_profiles

# ============================================================================
# FAST PERIOD DETECTION USING MINISOM
# ============================================================================

def detect_periods_som(yearly_profiles):
    """Detect periods using FAST MiniSom library"""
    print(f"\n{'='*80}")
    print("STEP 1: DETECTING PERIODS WITH FAST SOM")
    print(f"{'='*80}")
    
    # Collect all data
    all_data = []
    all_metadata = []
    
    for year, profile in sorted(yearly_profiles.items()):
        for i, country in enumerate(profile['countries']):
            all_data.append(profile['scaled_data'][i])
            all_metadata.append({'year': year, 'country': country})
    
    all_data = np.array(all_data)
    
    # Fill NaN with median (MiniSom doesn't handle NaN)
    all_data_filled = all_data.copy()
    for j in range(all_data.shape[1]):
        col_mask = ~np.isnan(all_data[:, j])
        if np.sum(col_mask) > 0:
            all_data_filled[~col_mask, j] = np.nanmedian(all_data[:, j])
    
    print(f"Training SOM on {len(all_data)} observations...")
    print("(Using optimized MiniSom library - should take 2-3 minutes)")
    
    # Use MiniSom
    som = MiniSom(x=20, y=20, input_len=all_data.shape[1],
                  sigma=3.0, learning_rate=0.5, random_seed=42)
    som.random_weights_init(all_data_filled)
    som.train_batch(all_data_filled, num_iteration=5000, verbose=True)
    
    print("✓ SOM training complete!")
    
    # Map countries to neurons
    for i, data_point in enumerate(all_data_filled):
        winner = som.winner(data_point)
        all_metadata[i]['som_neuron'] = winner[0] * 20 + winner[1]
    
    metadata_df = pd.DataFrame(all_metadata)
    
    # Calculate distribution changes between consecutive years
    years = sorted(metadata_df['year'].unique())
    changes = []
    
    print(f"\nCalculating distribution changes across {len(years)} years...")
    
    for i in range(1, len(years)):
        prev_year, curr_year = years[i-1], years[i]
        
        prev_data = metadata_df[metadata_df['year'] == prev_year]
        curr_data = metadata_df[metadata_df['year'] == curr_year]
        
        prev_act = np.zeros(400)
        curr_act = np.zeros(400)
        
        for neuron_id, count in prev_data['som_neuron'].value_counts().items():
            prev_act[int(neuron_id)] = count
        for neuron_id, count in curr_data['som_neuron'].value_counts().items():
            curr_act[int(neuron_id)] = count
        
        prev_act = prev_act / (prev_act.sum() + 1e-10)
        curr_act = curr_act / (curr_act.sum() + 1e-10)
        
        euclidean_dist = np.linalg.norm(prev_act - curr_act)
        cosine_dist = 1 - np.dot(prev_act, curr_act) / (np.linalg.norm(prev_act) * np.linalg.norm(curr_act) + 1e-10)
        wasserstein_dist = wasserstein_distance(np.arange(400), np.arange(400), prev_act, curr_act)
        
        composite_score = euclidean_dist * 0.4 + cosine_dist * 0.3 + wasserstein_dist * 0.3
        
        changes.append({'year': curr_year, 'composite_score': composite_score})
    
    changes_df = pd.DataFrame(changes)
    
    # Select top 6 breaks
    changes_sorted = changes_df.sort_values('composite_score', ascending=False)
    break_years = sorted(changes_sorted.head(6)['year'].tolist())
    
    print(f"\n✓ Detected {len(break_years)} breakpoints:")
    for year in break_years:
        score = changes_df[changes_df['year'] == year]['composite_score'].values[0]
        print(f"  - {year} (change score: {score:.3f})")
    
    # Define periods
    min_year, max_year = min(years), max(years)
    period_starts = [min_year] + break_years
    period_ends = break_years + [max_year]
    
    periods = []
    for i, (start, end) in enumerate(zip(period_starts, period_ends)):
        periods.append({
            'period_id': i + 1,
            'start_year': start,
            'end_year': end,
            'duration': end - start + 1,
            'name': f"Period {i+1} ({start}-{end})"
        })
    
    periods_df = pd.DataFrame(periods)
    
    print(f"\n✓ {len(periods)} periods defined:")
    for _, p in periods_df.iterrows():
        print(f"  {p['name']}")
    
    return periods_df, changes_df

# ============================================================================
# CLUSTERING WITH FIXED MACRO-GROUPS
# ============================================================================

def map_to_macro_group(avg_gdp_pc, avg_inflation, avg_unemployment):
    """Map to macro group based on GDP and stability"""
    # Rule 1: High-Income Advanced (GDP > $25K)
    if avg_gdp_pc > 25000:
        return "A", "High-Income Advanced", '#2E7D32', 1
    
    # Rule 2: Emerging & Upper-Middle Income ($8K-$25K + stable)
    elif 8000 < avg_gdp_pc <= 25000 and avg_inflation < 12:
        return "B", "Emerging & Upper-Middle Income", '#1976D2', 2
    
    # Rule 3: High-Inflation Vulnerable (crisis indicator)
    elif avg_inflation > 15 or avg_unemployment > 15:
        return "D", "High-Inflation Vulnerable", '#C62828', 4
    
    # Rule 4: Developing & Lower-Middle Income (everyone else)
    else:
        return "C", "Developing & Lower-Middle Income", '#F57C00', 3

def cluster_period_with_fixed_labels(df, period_info, feature_cols):
    """Cluster countries and assign to macro-groups A, B, C, D"""
    print(f"\n{'='*80}")
    print(f"CLUSTERING: {period_info['name']}")
    print(f"{'='*80}")
    
    start_year = period_info['start_year']
    end_year = period_info['end_year']
    
    period_df = df[(df['year'] >= start_year) & (df['year'] <= end_year)].copy()
    
    # Average across years
    country_profile = period_df.groupby('country_name')[feature_cols].mean()
    country_profile = country_profile[country_profile.notna().sum(axis=1) > 0]
    
    countries = country_profile.index.tolist()
    print(f"Countries: {len(countries)}")
    
    # Scale without imputation
    scaled_data = np.zeros_like(country_profile.values)
    for j, col in enumerate(feature_cols):
        col_data = country_profile[col].values
        mask = ~np.isnan(col_data)
        if np.sum(mask) > 0:
            available_values = col_data[mask]
            median = np.median(available_values)
            q25, q75 = np.percentile(available_values, [25, 75])
            iqr = q75 - q25
            if iqr > 0:
                scaled_values = (col_data - median) / iqr
                scaled_data[:, j] = scaled_values
    
    scaled_data[country_profile.isna().values] = np.nan
    
    # Fill NaN for MiniSom
    scaled_data_filled = scaled_data.copy()
    for j in range(scaled_data.shape[1]):
        col_mask = ~np.isnan(scaled_data[:, j])
        if np.sum(col_mask) > 0:
            scaled_data_filled[~col_mask, j] = np.nanmedian(scaled_data[:, j])
    
    # Train 3×3 SOM
    som = MiniSom(x=3, y=3, input_len=len(feature_cols),
                  sigma=1.0, learning_rate=0.5, random_seed=42)
    som.random_weights_init(scaled_data_filled)
    som.train_batch(scaled_data_filled, num_iteration=3000, verbose=False)
    
    # Label each country individually
    all_country_labels = []
    
    for i, country in enumerate(countries):
        winner = som.winner(scaled_data_filled[i])
        som_cluster_id = winner[0] * 3 + winner[1]
        
        country_data = country_profile.loc[country]
        
        # Get INDIVIDUAL country characteristics
        country_gdp_log = country_data['gdp_per_capita_log']
        country_gdp = np.expm1(country_gdp_log) if not np.isnan(country_gdp_log) else 0
        country_inflation = country_data['inflation_cpi_pct'] if not np.isnan(country_data['inflation_cpi_pct']) else 0
        country_unemployment = country_data['unemployment_pct'] if not np.isnan(country_data['unemployment_pct']) else 0
        country_trade = country_data['trade_pct_gdp'] if not np.isnan(country_data['trade_pct_gdp']) else 0
        country_ca = country_data['current_account_pct_gdp'] if not np.isnan(country_data['current_account_pct_gdp']) else 0
        country_fdi = country_data['fdi_net_inflows_pct_gdp'] if not np.isnan(country_data['fdi_net_inflows_pct_gdp']) else 0
        
        # Label based on INDIVIDUAL GDP
        macro_id, macro_name, color, priority = map_to_macro_group(
            country_gdp, country_inflation, country_unemployment
        )
        
        all_country_labels.append({
            'country': country,
            'som_cluster_id': som_cluster_id,
            'macro_group_id': macro_id,
            'macro_group_name': macro_name,
            'avg_gdp_per_capita': country_gdp,
            'avg_inflation': country_inflation,
            'avg_unemployment': country_unemployment,
            'avg_trade_openness': country_trade,
            'avg_current_account': country_ca,
            'avg_fdi': country_fdi
        })
    
    mapping_df = pd.DataFrame(all_country_labels)
    
    # Create cluster analysis for display
    cluster_analysis = []
    
    for macro_id in ['A', 'B', 'C', 'D']:
        group_countries = mapping_df[mapping_df['macro_group_id'] == macro_id]
        
        if len(group_countries) > 0:
            cluster_analysis.append({
                'macro_group_id': macro_id,
                'macro_group_name': group_countries['macro_group_name'].iloc[0],
                'n_countries': len(group_countries),
                'countries': sorted(group_countries['country'].tolist()),
                'avg_gdp_per_capita': group_countries['avg_gdp_per_capita'].median(),
                'avg_inflation': group_countries['avg_inflation'].median(),
                'avg_unemployment': group_countries['avg_unemployment'].median(),
                'avg_trade_openness': group_countries['avg_trade_openness'].median(),
                'avg_current_account': group_countries['avg_current_account'].median(),
                'avg_fdi': group_countries['avg_fdi'].median()
            })
    
    print(f"\n✓ Labeled {len(countries)} countries individually into macro-groups:")
    
    for cluster in sorted(cluster_analysis, key=lambda x: ['A','B','C','D'].index(x['macro_group_id'])):
        print(f"\n  [{cluster['macro_group_id']}] {cluster['macro_group_name']}")
        print(f"      Countries: {cluster['n_countries']}")
        print(f"      Median GDP/capita: ${cluster['avg_gdp_per_capita']:,.0f}")
        print(f"      Median Inflation: {cluster['avg_inflation']:.1f}%")
        print(f"      GDP range: ${min([c['avg_gdp_per_capita'] for c in all_country_labels if c['macro_group_id']==cluster['macro_group_id']]):,.0f} - ${max([c['avg_gdp_per_capita'] for c in all_country_labels if c['macro_group_id']==cluster['macro_group_id']]):,.0f}")
        
        if 'China' in cluster['countries']:
            print(f"      🇨🇳 INCLUDES CHINA")
        if 'United States' in cluster['countries']:
            print(f"      🇺🇸 INCLUDES USA")
        
        print(f"      Examples: {', '.join(cluster['countries'][:5])}")
    
    return {
        'clusters': cluster_analysis,
        'mapping_df': mapping_df,
        'som': som,
        'country_profile': country_profile
    }

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("\n" + "="*80)
    print("✅ ENHANCED SOM ANALYSIS WITH ATTRIBUTE IMPORTANCE")
    print("="*80)
    
    output_dir = '.'  # Current directory - change if needed
    
    # Load data
    df, feature_cols, yearly_profiles = load_and_prepare_data()
    
    # Detect periods automatically
    periods_df, changes_df = detect_periods_som(yearly_profiles)
    
    # Cluster each period
    print(f"\n{'='*80}")
    print("STEP 2: CLUSTERING WITH FIXED MACRO-GROUPS")
    print(f"{'='*80}")
    
    all_period_results = {}
    all_country_assignments = []
    all_importance_results = []
    
    for _, period in periods_df.iterrows():
        period_info = period.to_dict()
        results = cluster_period_with_fixed_labels(df, period_info, feature_cols)
        all_period_results[period_info['name']] = results
        
        # Calculate attribute importance for this period
        importance_df = calculate_attribute_importance(
            results['country_profile'], 
            results['mapping_df'], 
            feature_cols
        )
        
        # Add period information
        importance_df['period_id'] = period_info['period_id']
        importance_df['period_name'] = period_info['name']
        importance_df['start_year'] = period_info['start_year']
        importance_df['end_year'] = period_info['end_year']
        
        all_importance_results.append(importance_df)
        
        # Print importance summary
        print(f"\n  📊 ATTRIBUTE IMPORTANCE for {period_info['name']}:")
        for _, row in importance_df.head(7).iterrows():
            print(f"      #{row['rank']}: {row['attribute']:30s} - Score: {row['importance_score']:5.1f} (Variance Explained: {row['variance_explained_pct']:5.1f}%)")
        
        # Save assignments
        for _, row in results['mapping_df'].iterrows():
            all_country_assignments.append({
                'period_id': period_info['period_id'],
                'period_name': period_info['name'],
                'start_year': period_info['start_year'],
                'end_year': period_info['end_year'],
                'country': row['country'],
                'macro_group_id': row['macro_group_id'],
                'macro_group_name': row['macro_group_name']
            })
        
        # Save period CSV with all 7 attributes
        period_output = []
        for cluster in results['clusters']:
            cluster_countries = cluster['countries']
            country_profile_local = results['country_profile']
            
            for country in cluster_countries:
                if country in country_profile_local.index:
                    country_data = country_profile_local.loc[country]
                    
                    # Extract all 7 attributes
                    country_gdp_per_capita_log = country_data['gdp_per_capita_log']
                    country_gdp_per_capita = np.expm1(country_gdp_per_capita_log) if not np.isnan(country_gdp_per_capita_log) else 0
                    
                    country_gdp_total_log = country_data['gdp_total_log']
                    country_gdp_total = np.expm1(country_gdp_total_log) if not np.isnan(country_gdp_total_log) else 0
                    
                    country_inflation = country_data['inflation_cpi_pct'] if not np.isnan(country_data['inflation_cpi_pct']) else 0
                    country_unemployment = country_data['unemployment_pct'] if not np.isnan(country_data['unemployment_pct']) else 0
                    country_trade = country_data['trade_pct_gdp'] if not np.isnan(country_data['trade_pct_gdp']) else 0
                    country_fdi = country_data['fdi_net_inflows_pct_gdp'] if not np.isnan(country_data['fdi_net_inflows_pct_gdp']) else 0
                    country_current_account = country_data['current_account_pct_gdp'] if not np.isnan(country_data['current_account_pct_gdp']) else 0
                    
                    period_output.append({
                        'country': country,
                        'macro_group_id': cluster['macro_group_id'],
                        'macro_group_name': cluster['macro_group_name'],
                        'avg_gdp_per_capita': country_gdp_per_capita,
                        'avg_inflation': country_inflation,
                        'avg_unemployment': country_unemployment,
                        'avg_trade_pct_gdp': country_trade,
                        'avg_fdi_net_inflows_pct_gdp': country_fdi,
                        'avg_current_account_pct_gdp': country_current_account,
                        'avg_gdp_total': country_gdp_total
                    })
        
        period_df_out = pd.DataFrame(period_output)
        csv_path = f"{output_dir}/Period_{period_info['period_id']}_MacroGroups.csv"
        period_df_out.to_csv(csv_path, index=False)
        print(f"  ✓ Saved {csv_path}")
    
    # Save attribute importance across all periods
    all_importance_df = pd.concat(all_importance_results, ignore_index=True)
    importance_path = f"{output_dir}/Attribute_Importance_All_Periods.csv"
    all_importance_df.to_csv(importance_path, index=False)
    print(f"\n✅ Saved {importance_path}")
    
    # Create summary showing average importance across all periods
    avg_importance = all_importance_df.groupby('attribute').agg({
        'importance_score': 'mean',
        'variance_explained_pct': 'mean',
        'f_statistic': 'mean'
    }).sort_values('importance_score', ascending=False)
    
    avg_importance['avg_rank'] = range(1, len(avg_importance) + 1)
    
    summary_path = f"{output_dir}/Attribute_Importance_Summary.csv"
    avg_importance.to_csv(summary_path)
    print(f"✅ Saved {summary_path}")
    
    # Display overall importance ranking
    print(f"\n{'='*80}")
    print("📊 OVERALL ATTRIBUTE IMPORTANCE (Averaged Across All Periods)")
    print(f"{'='*80}")
    for attr, row in avg_importance.iterrows():
        print(f"  #{int(row['avg_rank'])}: {attr:30s} - Avg Score: {row['importance_score']:5.1f} (Variance: {row['variance_explained_pct']:5.1f}%)")
    
    # Save tracking
    tracking_df = pd.DataFrame(all_country_assignments)
    tracking_path = f"{output_dir}/Complete_Country_Tracking_ABCD.csv"
    tracking_df.to_csv(tracking_path, index=False)
    print(f"\n✓ Saved {tracking_path}")
    
    # Check China's trajectory
    print(f"\n{'='*80}")
    print("🇨🇳 CHINA'S TRAJECTORY")
    print(f"{'='*80}")
    
    china_trajectory = tracking_df[tracking_df['country'] == 'China']
    if len(china_trajectory) > 0:
        print("\nChina's group assignments:")
        path = []
        for _, row in china_trajectory.iterrows():
            print(f"  {row['period_name']}: Group {row['macro_group_id']} - {row['macro_group_name']}")
            path.append(row['macro_group_id'])
        
        print(f"\nTrajectory: {' → '.join(path)}")
        
        if path.count('C') >= 2 and 'B' in path:
            print("✅ Realistic: Shows gradual development!")
        else:
            print("⚠️ Warning: May need threshold adjustment")
    
    # Summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")
    
    for pid in range(1, len(periods_df) + 1):
        period_data = tracking_df[tracking_df['period_id'] == pid]
        counts = period_data['macro_group_id'].value_counts().to_dict()
        period_name = period_data['period_name'].iloc[0]
        print(f"\n{period_name}:")
        print(f"  [A] High-Income:     {counts.get('A', 0):3d} countries")
        print(f"  [B] Emerging:        {counts.get('B', 0):3d} countries")
        print(f"  [C] Developing:      {counts.get('C', 0):3d} countries")
        print(f"  [D] High-Inflation:  {counts.get('D', 0):3d} countries")
    
    print(f"\n{'='*80}")
    print("✅ ANALYSIS COMPLETE!")
    print(f"{'='*80}")
    print("\n📁 Generated files:")
    print(f"  • {tracking_path}")
    print(f"  • {importance_path} ⭐ NEW")
    print(f"  • {summary_path} ⭐ NEW")
    print(f"  • Period_X_MacroGroups.csv ({len(periods_df)} files with all 7 attributes)")
    
    return periods_df, all_period_results, tracking_df, all_importance_df

if __name__ == "__main__":
    periods_df, all_period_results, tracking_df, importance_df = main()
