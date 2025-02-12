import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd

def get_texas_zipcodes(gdf):
    """Filter GeoDataFrame for Texas ZIP codes"""
    texas_prefixes = ('75', '76', '77', '78', '79', '88')
    texas_mask = gdf['ZCTA5CE10'].str.startswith(texas_prefixes)
    return gdf[texas_mask].copy()

def filter_mainland_us(gdf):
    """Filter GeoDataFrame for mainland US (excluding Alaska and Hawaii)"""
    excluded_prefixes = ('995', '996', '997', '998', '999',  # Alaska
                        '968', '969')  # Hawaii
    mainland_mask = ~gdf['ZCTA5CE10'].str.startswith(excluded_prefixes)
    return gdf[mainland_mask].copy()

class Big5Calculator:
    def __init__(self, responses):
        """Initialize with responses and convert to numeric values"""
        if isinstance(responses, dict):
            self.responses = pd.Series(responses)
        else:
            self.responses = responses
        
        # Filter for only personality indicator columns
        personality_prefixes = ['ext', 'agr', 'cns', 'neu', 'opn']
        self.responses = self.responses[
            [col for col in self.responses.index 
             if any(col.startswith(prefix) for prefix in personality_prefixes) and
             not any(col.startswith(invalid) for invalid in ['tipi', 'bfi2', 'obfi2'])]
        ]
        
        # Convert all values to float (this will handle NaN values properly)
        self.responses = pd.to_numeric(self.responses, errors='coerce')
        
        # Pre-calculate reverse items
        self.reverse_items = self.responses.index[self.responses.index.str.endswith('r')]
        
    def _reverse_score(self, values):
        """Reverse score on 1-5 scale - vectorized version"""
        return 6 - values

    def calculate_scores(self, form='bfi'):
        """Calculate Big Five scores for either BFI or OBFI - optimized version"""
        prefix = 'o' if form == 'obfi' else ''
        scores = self.responses.copy()
        
        # Reverse score items ending with 'r' - vectorized operation
        scores[self.reverse_items] = self._reverse_score(scores[self.reverse_items])
        
        # Define dimensions and their prefixes
        dimensions = {
            'extraversion': f'{prefix}ext',
            'agreeableness': f'{prefix}agr', 
            'conscientiousness': f'{prefix}cns',
            'neuroticism': f'{prefix}neu',
            'openness': f'{prefix}opn'
        }
        
        # Calculate means for each dimension using vectorized operations
        return {
            dim: scores[scores.index.str.startswith(prefix)].mean() 
            if any(scores.index.str.startswith(prefix)) else np.nan
            for dim, prefix in dimensions.items()
        }

def identify_city(zip_code):
    """
    Identify city based on ZIP code prefix.
    Reference: https://en.wikipedia.org/wiki/List_of_ZIP_Code_prefixes
    """
    zip_str = str(zip_code)
    
    # Texas cities
    if zip_str.startswith(('752', '753')):
        return 'Dallas'
    elif zip_str.startswith(('770', '771', '772')):
        return 'Houston'
    elif zip_str.startswith('782'):
        return 'San Antonio'
    elif zip_str.startswith('787'):
        return 'Austin'
    
    # Other major US cities
    elif zip_str.startswith(('100', '101', '102')):
        return 'New York'
    elif zip_str.startswith('606'):
        return 'Chicago'
    elif zip_str.startswith('900'):
        return 'Los Angeles'
    elif zip_str.startswith('941'):
        return 'San Francisco'
    elif zip_str.startswith('980'):
        return 'Seattle'
    elif zip_str.startswith('891'):
        return 'Las Vegas'
    elif zip_str.startswith('602'):
        return 'Phoenix'
    elif zip_str.startswith('331'):
        return 'Miami'
    elif zip_str.startswith('303'):
        return 'Atlanta'
    elif zip_str.startswith('192'):
        return 'Philadelphia'
    elif zip_str.startswith('212'):
        return 'Baltimore'
    elif zip_str.startswith('022'):
        return 'Boston'
    elif zip_str.startswith('802'):
        return 'Denver'
    elif zip_str.startswith('372'):
        return 'Nashville'
    elif zip_str.startswith('631'):
        return 'St. Louis'
    elif zip_str.startswith('482'):
        return 'Detroit'
    elif zip_str.startswith('532'):
        return 'Milwaukee'
    elif zip_str.startswith('550'):
        return 'Minneapolis'
    elif zip_str.startswith('402'):
        return 'Louisville'
    elif zip_str.startswith('432'):
        return 'Columbus'
    elif zip_str.startswith('272'):
        return 'Pittsburgh'
    elif zip_str.startswith('732'):
        return 'Little Rock'
    elif zip_str.startswith('731'):
        return 'Memphis'
    elif zip_str.startswith('832'):
        return 'New Orleans'
    elif zip_str.startswith('322'):
        return 'Jacksonville'
    elif zip_str.startswith('292'):
        return 'Charleston'
    elif zip_str.startswith('232'):
        return 'Richmond'
    elif zip_str.startswith('202'):
        return 'Washington DC'
    elif zip_str.startswith('921'):
        return 'San Diego'
    elif zip_str.startswith('503'):
        return 'Portland'
    elif zip_str.startswith('841'):
        return 'Salt Lake City'
    elif zip_str.startswith('592'):
        return 'Kansas City'
    elif zip_str.startswith('681'):
        return 'Indianapolis'
    elif zip_str.startswith('452'):
        return 'Cincinnati'
    elif zip_str.startswith('441'):
        return 'Cleveland'
    elif zip_str.startswith('152'):
        return 'Albany'
    elif zip_str.startswith('601'):
        return 'Phoenix'
    else:
        return 'Other'

def calculate_personality_scores(row):
    calculator = Big5Calculator(row)
    return pd.Series(calculator.calculate_scores('bfi'))

def calculate_personality_scores_vectorized(df):
    """
    Calculate Big Five personality scores for all rows at once.
    This is a vectorized version that avoids row-by-row processing.
    """
    # Filter for personality indicator columns
    personality_prefixes = ['ext', 'agr', 'cns', 'neu', 'opn']
    cols = [col for col in df.columns 
            if any(col.startswith(prefix) for prefix in personality_prefixes) and
            not any(col.startswith(invalid) for invalid in ['tipi', 'bfi2', 'obfi2'])]
    
    # Select only personality columns and convert to numeric
    scores = df[cols].apply(pd.to_numeric, errors='coerce')
    
    # Identify reverse-scored items
    reverse_items = [col for col in scores.columns if col.endswith('r')]
    
    # Reverse score items ending with 'r'
    if reverse_items:
        scores[reverse_items] = 6 - scores[reverse_items]
    
    # Calculate means for each dimension
    results = pd.DataFrame()
    
    # Calculate means for each trait
    for trait, prefix in {
        'extraversion': 'ext',
        'agreeableness': 'agr',
        'conscientiousness': 'cns',
        'neuroticism': 'neu',
        'openness': 'opn'
    }.items():
        trait_cols = [col for col in scores.columns if col.startswith(prefix)]
        if trait_cols:
            results[trait] = scores[trait_cols].mean(axis=1)
    
    return results

def plot_personality_distributions(now_personality, youth_personality):
    print("\nGenerating personality distribution plots...")
    traits = ['extraversion', 'agreeableness', 'conscientiousness', 'neuroticism', 'openness']
    cities = ['Austin', 'Dallas', 'Houston', 'San Antonio']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for trait in traits:
        print(f"  Creating distribution plot for {trait}...")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Current residence
        for city, color in zip(cities, colors):
            city_data = now_personality[now_personality['city'] == city][trait]
            ax1.hist(city_data, alpha=0.5, label=f"{city} (n={len(city_data)})", color=color, bins=20)
        ax1.set_title(f'Current Residence: {trait}')
        ax1.legend()
        ax1.set_xlabel('Score')
        ax1.set_ylabel('Count')
        
        # Youth residence
        for city, color in zip(cities, colors):
            city_data = youth_personality[youth_personality['city'] == city][trait]
            ax2.hist(city_data, alpha=0.5, label=f"{city} (n={len(city_data)})", color=color, bins=20)
        ax2.set_title(f'Youth Residence: {trait}')
        ax2.legend()
        ax2.set_xlabel('Score')
        ax2.set_ylabel('Count')
        
        plt.tight_layout()
        plt.savefig(f'/media/gisense/koichi/personality/figs/distribution_{trait.lower()}.png')
        plt.close()

def plot_trait_maps_texas(now_personality, youth_personality, map_data, now_zips, youth_zips):
    """Create trait maps for Texas only"""
    print("\nGenerating Texas personality trait maps...")
    traits = ['extraversion', 'agreeableness', 'conscientiousness', 'neuroticism', 'openness']
    
    def plot_trait_map(personality_data, zip_codes, trait, residence_type):
        print(f"  Creating Texas {residence_type} residence map for {trait}...")
        
        # Get Texas ZIP codes
        texas = get_texas_zipcodes(map_data)
        
        # Add ZIP codes to personality data
        personality_data = personality_data.copy()
        personality_data['zip_code'] = zip_codes
        
        # Calculate ZIP code level means and sample sizes
        zip_means = personality_data.groupby('zip_code')[trait].agg(['mean', 'count']).reset_index()
        zip_means.columns = ['ZCTA5CE10', 'trait_value', 'sample_size']
        
        # Calculate alpha values based on sample size
        max_samples = zip_means['sample_size'].max()
        zip_means['alpha'] = np.log1p(zip_means['sample_size']) / np.log1p(max_samples)
        # Ensure minimum visibility
        zip_means['alpha'] = zip_means['alpha'].clip(0.1, 0.9)
        
        # Merge with geographic data
        texas = texas.merge(zip_means, on='ZCTA5CE10', how='left')
        
        # Create figure
        fig, ax = plt.subplots(figsize=(15, 10))
        
        # Plot base map (ZIP codes without data)
        texas.plot(color='lightgray', ax=ax)
        
        # Plot ZIP codes with data
        data_mask = texas['trait_value'].notna()
        for _, row in texas[data_mask].iterrows():
            color = plt.cm.viridis((row['trait_value'] - texas[data_mask]['trait_value'].min()) / 
                                 (texas[data_mask]['trait_value'].max() - texas[data_mask]['trait_value'].min()))
            if pd.notnull(row['alpha']):
                poly = gpd.GeoDataFrame({'geometry': [row['geometry']]})
                poly.plot(ax=ax, color=color, alpha=row['alpha'])
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap='viridis', 
                                  norm=plt.Normalize(vmin=texas[data_mask]['trait_value'].min(), 
                                                   vmax=texas[data_mask]['trait_value'].max()))
        cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', 
                          label=f'{trait} (n={len(zip_means)})')
        cbar.ax.set_xlabel(trait, fontsize=10)
        
        ax.axis('off')
        
        # Add sample size information
        plt.figtext(
            0.02, 0.02,
            f'Total ZIP codes with data: {len(zip_means)}\n' +
            f'Mean samples per ZIP: {zip_means["sample_size"].mean():.1f}\n' +
            f'Median samples per ZIP: {zip_means["sample_size"].median():.1f}',
            fontsize=8
        )
        
        plt.tight_layout()
        plt.savefig(f'/media/gisense/koichi/personality/figs/map_{trait.lower()}_{residence_type.lower()}_texas.png',
                   bbox_inches='tight')
        plt.close()

    # Create maps for each trait and residence type
    for trait in traits:
        plot_trait_map(now_personality, now_zips, trait, 'Current')
        plot_trait_map(youth_personality, youth_zips, trait, 'Youth')

def plot_trait_maps_us(now_personality, youth_personality, map_data, now_zips, youth_zips):
    """Create trait maps for the entire mainland US"""
    print("\nGenerating US-wide personality trait maps...")
    traits = ['extraversion', 'agreeableness', 'conscientiousness', 'neuroticism', 'openness']
    
    def plot_trait_map(personality_data, zip_codes, trait, residence_type):
        print(f"  Creating US-wide {residence_type} residence map for {trait}...")
        
        # Get mainland US ZIP codes
        mainland = filter_mainland_us(map_data)
        
        # Add ZIP codes to personality data
        personality_data = personality_data.copy()
        personality_data['zip_code'] = zip_codes
        
        # Calculate ZIP code level means and sample sizes
        zip_means = personality_data.groupby('zip_code')[trait].agg(['mean', 'count']).reset_index()
        zip_means.columns = ['ZCTA5CE10', 'trait_value', 'sample_size']
        
        # Calculate alpha values based on sample size
        # Use log scale for better visualization since sample sizes might vary greatly
        max_samples = zip_means['sample_size'].max()
        zip_means['alpha'] = np.log1p(zip_means['sample_size']) / np.log1p(max_samples)
        # Ensure minimum visibility
        zip_means['alpha'] = zip_means['alpha'].clip(0.1, 0.9)
        
        # Merge with geographic data
        mainland = mainland.merge(zip_means, on='ZCTA5CE10', how='left')
        
        # Create figure
        fig, ax = plt.subplots(figsize=(20, 12))
        
        # Plot base map (ZIP codes without data)
        mainland.plot(color='lightgray', ax=ax)
        
        # Calculate robust min/max values for color scale (excluding outliers)
        valid_values = mainland['trait_value'].dropna()
        vmin = valid_values.quantile(0.01)  # 1st percentile
        vmax = valid_values.quantile(0.99)  # 99th percentile
        
        # Plot ZIP codes with data
        data_mask = mainland['trait_value'].notna()
        for _, row in mainland[data_mask].iterrows():
            color = plt.cm.viridis((row['trait_value'] - vmin) / (vmax - vmin))
            if pd.notnull(row['alpha']):
                poly = gpd.GeoDataFrame({'geometry': [row['geometry']]})
                poly.plot(ax=ax, color=color, alpha=row['alpha'])
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap='viridis', 
                                  norm=plt.Normalize(vmin=vmin, vmax=vmax))
        cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', 
                          label=f'{trait} (n={len(zip_means)})')
        cbar.ax.set_xlabel(trait, fontsize=10)
        
        ax.axis('off')
        
        # Add sample size and distribution information
        stats_text = (
            f'Total ZIP codes with data: {len(zip_means)}\n'
            f'Mean samples per ZIP: {zip_means["sample_size"].mean():.1f}\n'
            f'Median samples per ZIP: {zip_means["sample_size"].median():.1f}\n'
            f'1st percentile: {vmin:.2f}\n'
            f'99th percentile: {vmax:.2f}'
        )
        plt.figtext(0.02, 0.02, stats_text, fontsize=8)
        
        plt.tight_layout()
        plt.savefig(f'/media/gisense/koichi/personality/figs/map_{trait.lower()}_{residence_type.lower()}_us.png',
                   bbox_inches='tight', dpi=300)
        plt.close()

    # Create maps for each trait and residence type
    for trait in traits:
        plot_trait_map(now_personality, now_zips, trait, 'Current')
        plot_trait_map(youth_personality, youth_zips, trait, 'Youth')

def analyze_big_five(df_now, df_youth, map_data, map_type='texas'):
    """
    Analyze Big Five personality traits and create visualizations.
    
    Parameters:
    -----------
    df_now : pandas.DataFrame
        DataFrame containing current residence data for all US
    df_youth : pandas.DataFrame
        DataFrame containing youth residence data for all US
    map_data : geopandas.GeoDataFrame
        GeoDataFrame containing ZIP code shapes
    map_type : str, optional (default='texas')
        Type of map to create. Options are 'texas' or 'us'
    """
    print("\nCalculating personality scores...")
    
    if map_type.lower() == 'texas':
        # Filter for Texas ZIP codes
        print("  Filtering for Texas ZIP codes...")
        df_now_tx = df_now[df_now['now_zip'].str.startswith(('752', '753',  # Dallas
                                                            '770', '771', '772',  # Houston 
                                                            '782',  # San Antonio
                                                            '787'))]  # Austin
        df_youth_tx = df_youth[df_youth['youth_zip'].str.startswith(('752', '753',  # Dallas
                                                                    '770', '771', '772',  # Houston 
                                                                    '782',  # San Antonio
                                                                    '787'))]  # Austin
        df_for_analysis_now = df_now_tx
        df_for_analysis_youth = df_youth_tx
    else:
        # Use all US data
        print("  Using all US ZIP codes...")
        df_for_analysis_now = df_now
        df_for_analysis_youth = df_youth
    
    # Calculate scores for current residence
    print("  Processing current residence data...")
    now_personality = calculate_personality_scores_vectorized(df_for_analysis_now)
    now_personality['city'] = df_for_analysis_now['now_zip'].apply(identify_city)

    # Calculate scores for youth residence
    print("  Processing youth residence data...")
    youth_personality = calculate_personality_scores_vectorized(df_for_analysis_youth)
    youth_personality['city'] = df_for_analysis_youth['youth_zip'].apply(identify_city)

    # Create visualizations
    if map_type.lower() == 'texas':
        plot_personality_distributions(now_personality, youth_personality)
    
    # Choose mapping function based on map_type
    if map_type.lower() == 'us':
        plot_trait_maps_us(now_personality, youth_personality, map_data, 
                          df_for_analysis_now['now_zip'], df_for_analysis_youth['youth_zip'])
    else:
        plot_trait_maps_texas(now_personality, youth_personality, map_data, 
                            df_for_analysis_now['now_zip'], df_for_analysis_youth['youth_zip'])

    # Print summary statistics
    print("\nCalculating summary statistics...")
    traits = ['extraversion', 'agreeableness', 'conscientiousness', 'neuroticism', 'openness']
    
    if map_type.lower() == 'texas':
        print("\nPersonality Trait Summary by City:")
        print("\nCurrent Residence:")
        current_stats = now_personality.groupby('city')[traits].agg(['mean', 'std', 'count'])
        print(current_stats)
        
        print("\nYouth Residence:")
        youth_stats = youth_personality.groupby('city')[traits].agg(['mean', 'std', 'count'])
        print(youth_stats)
    else:
        print("\nUS-wide Personality Trait Summary:")
        print("\nCurrent Residence:")
        print(now_personality[traits].agg(['mean', 'std', 'count']))
        print("\nYouth Residence:")
        print(youth_personality[traits].agg(['mean', 'std', 'count']))
    
    print("\nAnalysis complete!")