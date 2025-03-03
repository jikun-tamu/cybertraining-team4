import os
import polars as pl
import geopandas as gpd
from pathlib import Path
import pyreadstat
import numpy as np
from tqdm import tqdm
from big_five_analysis import calculate_personality_scores_vectorized
import pandas as pd
from shapely.ops import unary_union
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Configuration
DATA_DIR = Path("/media/data/personality")
WORKSPACE_DIR = Path(os.getcwd())
RAW_DATA_DIR = WORKSPACE_DIR / "data/raw"
PROCESSED_DATA_DIR = WORKSPACE_DIR / "data/processed"
SEGMENTATION_DIR = PROCESSED_DATA_DIR / "segmentation"
DETECTION_DIR = PROCESSED_DATA_DIR / "detection"
CHECKPOINT_DIR = WORKSPACE_DIR / "data/checkpoints"
FINAL_INDIVIDUAL_DATA_PATH = WORKSPACE_DIR / "data/processed/model/final_dataset_individual.parquet"
FINAL_ZIPCODE_DATA_PATH = WORKSPACE_DIR / "data/processed/model/final_dataset_zipcode.parquet"

# Create checkpoint directory if it doesn't exist
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

def load_or_create_checkpoint(path: Path, create_func, *args, **kwargs):
    """Helper function to load checkpoint if exists or create and save if not"""
    if path.exists():
        print(f"Loading checkpoint from {path}")
        return pl.read_parquet(path)
    
    print(f"Creating new data for {path}")
    df = create_func(*args, **kwargs)
    print(f"Saving checkpoint to {path}")
    df.write_parquet(path)
    return df

def load_survey_data():
    """Load and preprocess the survey data"""
    checkpoint_path = CHECKPOINT_DIR / "survey_data.parquet"
    
    def _create_survey_data():
        survey_path = DATA_DIR / "OOS Master Dataset Sept 2022/00_OOS_MASTER DATASET_2022_08_22.sav"
        df, meta = pyreadstat.read_sav(str(survey_path))
        return pl.from_pandas(df)
    
    return load_or_create_checkpoint(checkpoint_path, _create_survey_data)

def load_zipcode_shapes():
    """Load and process zipcode shapefile"""
    shp_path = DATA_DIR / "cb_2016_us_zcta510_500k/cb_2016_us_zcta510_500k.shp"
    gdf = gpd.read_file(str(shp_path))
    return gdf

def process_gsv_metadata():
    """Process Google Street View metadata from all cities"""
    checkpoint_path = CHECKPOINT_DIR / "gsv_metadata.parquet"
    
    def _create_gsv_metadata():
        city_dfs = []
        for city_dir in RAW_DATA_DIR.glob("*"):
            if not city_dir.is_dir():
                continue
                
            metadata_file = city_dir / "gsv_pids.csv"
            if not metadata_file.exists():
                continue
                
            df = pl.read_csv(metadata_file)
            df = df.with_columns(pl.lit(city_dir.name).alias('city'))
            city_dfs.append(df)
        
        if not city_dfs:
            raise FileNotFoundError("No GSV metadata files found")
            
        return pl.concat(city_dfs)
    
    return load_or_create_checkpoint(checkpoint_path, _create_gsv_metadata)

def process_zipcode_year(zipcode, year, gdf_metrics, gdf_zipcodes):
    """Process a single zipcode-year combination for spatial metrics"""
    # Get zipcode polygon
    zipcode_polygon = gdf_zipcodes[gdf_zipcodes['ZCTA5CE10'] == zipcode].iloc[0].geometry
    zipcode_area = zipcode_polygon.area  # in square degrees, will convert later if needed
    
    # Filter points for this zipcode and year
    if year is not None:
        zip_year_points = gdf_metrics[(gdf_metrics['zipcode'] == zipcode) & 
                                     (gdf_metrics['gsv_year'] == year)]
    else:
        zip_year_points = gdf_metrics[gdf_metrics['zipcode'] == zipcode]
    
    # Skip if no points
    if len(zip_year_points) == 0:
        return (zipcode, year), None
    
    # Number of points
    point_count = len(zip_year_points)
    
    # Density (points per square unit)
    density = point_count / zipcode_area
    
    # Spatial coverage
    if point_count > 0:
        # Create a buffer of 100 meters around each point
        # First convert to a projected CRS for accurate buffer
        zip_year_points_proj = zip_year_points.to_crs('EPSG:3857')  # Web Mercator
        buffered_points = zip_year_points_proj.buffer(100)  # 100 meter buffer
        
        # Calculate union of all buffers
        coverage_area = unary_union(buffered_points)
        
        # Convert back to original CRS for comparison with zipcode
        coverage_area = gpd.GeoSeries([coverage_area]).set_crs('EPSG:3857').to_crs(gdf_zipcodes.crs)[0]
        
        # Calculate area and percentage
        coverage_ratio = coverage_area.intersection(zipcode_polygon).area / zipcode_polygon.area
    else:
        coverage_ratio = 0.0

    # Return metrics
    return (zipcode, year), {
        'point_count': point_count,
        'density': density,
        'coverage_ratio': coverage_ratio
    }

def spatial_join_gsv_zipcodes(df_gsv, gdf_zipcodes):
    """
    Perform spatial join between GSV points and zipcode polygons
    with temporal matching (+/- 2 years buffer)
    """
    checkpoint_path = CHECKPOINT_DIR / "gsv_with_zipcodes.parquet"
    
    def _create_spatial_join(df_gsv=df_gsv, gdf_zipcodes=gdf_zipcodes):
        # Convert Polars DataFrame to GeoDataFrame
        gdf_gsv = gpd.GeoDataFrame(
            df_gsv.to_pandas(),
            geometry=gpd.points_from_xy(df_gsv['lon'], df_gsv['lat']),
            crs="EPSG:4326"
        )
        
        # Ensure both GeoDataFrames have the same CRS
        if gdf_zipcodes.crs != gdf_gsv.crs:
            gdf_zipcodes = gdf_zipcodes.to_crs(gdf_gsv.crs)
        
        # Perform spatial join
        print("Performing spatial join...")
        gdf_joined = gpd.sjoin(gdf_gsv, gdf_zipcodes[['ZCTA5CE10', 'geometry']], 
                              how='left', predicate='within')
        
        # Already rename the zipcode column in the GeoDataFrame before making a copy
        gdf_joined = gdf_joined.rename(columns={'ZCTA5CE10': 'zipcode'})
        
        # Create a copy of the GeoDataFrame for spatial calculations
        gdf_metrics = gdf_joined.copy()
        
        # Filter only rows with valid zipcodes
        gdf_metrics = gdf_metrics[~gdf_metrics['zipcode'].isna()]
        
        # Convert back to Polars DataFrame (after making the copy for metrics)
        df_joined = pl.from_pandas(gdf_joined.drop(columns=['geometry', 'index_right']))
        
        # Extract year from date column (assuming 'date' exists in the GSV data)
        # If 'date' column doesn't exist in your dataframe, you'll need to create or derive it
        if 'date' in df_joined.columns:
            df_joined = df_joined.with_columns(pl.col('date').dt.year().alias('gsv_year'))
        elif 'year' in df_joined.columns:
            df_joined = df_joined.rename({'year': 'gsv_year'})
        else:
            # If neither date nor year exists, you might need to extract from another field
            # This is a placeholder - adjust according to your actual data
            print("Warning: No date/year column found in GSV data. Temporal matching will not be possible.")
            df_joined = df_joined.with_columns(pl.lit(None).cast(pl.Int32).alias('gsv_year'))
            
            # Also add the year column to the metrics geodataframe
            gdf_metrics['gsv_year'] = None
        
        # Add year column to gdf_metrics if it exists in df_joined
        if 'gsv_year' in df_joined.columns and 'gsv_year' not in gdf_metrics.columns:
            # Convert from polars to pandas for consistency with gdf_metrics
            years_dict = df_joined.select(['panoid', 'gsv_year']).to_pandas().set_index('panoid')['gsv_year'].to_dict()
            gdf_metrics['gsv_year'] = gdf_metrics['panoid'].map(years_dict)
        
        # Calculate spatial metrics for each zipcode
        print("Calculating spatial coverage metrics...")
        
        # Create empty dictionary to store metrics
        zipcode_metrics = {}
        
        # Get unique zipcodes and years
        zipcodes = gdf_metrics['zipcode'].unique()
        gsv_years = gdf_metrics['gsv_year'].dropna().unique() if 'gsv_year' in gdf_metrics.columns else [None]
        
        # Create list of all zipcode-year combinations to process
        zipcode_year_combinations = []
        for zipcode in zipcodes:
            for year in gsv_years:
                zipcode_year_combinations.append((zipcode, year))
        
        # Determine number of workers (use fewer than all cores to avoid system overload)
        max_workers = max(1, min(multiprocessing.cpu_count() - 1, 16))
        print(f"Processing with {max_workers} workers...")
        
        # Process zipcode-year combinations in parallel
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit tasks
            future_to_combo = {
                executor.submit(process_zipcode_year, zipcode, year, gdf_metrics, gdf_zipcodes): 
                (zipcode, year) for zipcode, year in zipcode_year_combinations
            }
            
            # Process results as they complete
            for future in tqdm(as_completed(future_to_combo), 
                               total=len(zipcode_year_combinations),
                               desc="Calculating metrics by zipcode-year"):
                key, result = future.result()
                if result is not None:
                    zipcode_metrics[key] = result
        
        # Convert metrics to DataFrame
        metrics_rows = []
        for (zipcode, year), metrics in zipcode_metrics.items():
            if metrics is not None:
                metrics_rows.append({
                    'zipcode': zipcode,
                    'gsv_year': year,
                    'gsv_point_count': metrics['point_count'],
                    'gsv_density': metrics['density'],
                    'gsv_coverage_ratio': metrics['coverage_ratio']
                })
        
        df_metrics = pl.from_pandas(pd.DataFrame(metrics_rows))
        
        # Join metrics back to the original dataframe
        if 'gsv_year' in df_joined.columns:
            join_keys = ['zipcode', 'gsv_year']
        else:
            join_keys = ['zipcode']
            
        # Ensure metrics dataframe has the right schema for the join
        if len(metrics_rows) > 0:
            df_joined = df_joined.join(
                df_metrics,
                on=join_keys,
                how='left'
            )
        
        return df_joined
    
    return load_or_create_checkpoint(checkpoint_path, _create_spatial_join)

def process_segmentation_results():
    """Process segmentation results from all cities and batches"""
    checkpoint_path = CHECKPOINT_DIR / "segmentation_results.parquet"
    
    def _create_segmentation_results():
        results_dfs = []
        
        for city_dir in SEGMENTATION_DIR.glob("*"):
            if not city_dir.is_dir():
                continue
                
            for batch_dir in city_dir.glob("batch_*"):
                if not batch_dir.is_dir():
                    continue
                    
                pixel_ratios = batch_dir / "pixel_ratios.csv"
                label_counts = batch_dir / "label_counts.csv"
                
                if not pixel_ratios.exists() or not label_counts.exists():
                    continue
                    
                # Read with suffixes for all columns
                df_ratios = pl.read_csv(pixel_ratios)
                df_counts = pl.read_csv(label_counts)
                
                # Add suffixes to all columns except the joining key
                ratio_cols = [col for col in df_ratios.columns if col != 'filename_key']
                count_cols = [col for col in df_counts.columns if col != 'filename_key']
                
                df_ratios = df_ratios.rename({col: f"{col}_ratios" for col in ratio_cols})
                df_counts = df_counts.rename({col: f"{col}_counts" for col in count_cols})
                
                # Merge ratios and counts
                df = df_ratios.join(df_counts, on='filename_key')
                df = df.with_columns([
                    pl.lit(city_dir.name).alias('city'),
                    pl.lit(batch_dir.name).alias('batch')
                ])
                print(f"Processing {city_dir.name}/{batch_dir.name}: {df.shape}")
                results_dfs.append(df)
        
        if not results_dfs:
            raise FileNotFoundError("No segmentation result files found")
            
        # Concatenate with align=True to handle different columns
        print("Concatenating all results...")
        df_concat = pl.concat(results_dfs, how="diagonal")
        
        # Fill null values with 0
        print("Filling null values with 0...")
        numeric_cols = [col for col in df_concat.columns 
                       if col not in ['filename_key', 'city', 'batch'] 
                       and df_concat[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]]
        df_concat = df_concat.with_columns([
            pl.col(col).fill_null(0) for col in numeric_cols
        ])
        
        return df_concat
    
    return load_or_create_checkpoint(checkpoint_path, _create_segmentation_results)

def calculate_visual_complexity(df):
    """Calculate Shannon index for each image based on pixel ratios"""
    # Get ratio columns
    ratio_cols = [col for col in df.columns if col.endswith('_ratios')]
    
    # Convert to numpy for faster computation
    ratios_array = df[ratio_cols].to_numpy()
    
    # Calculate Shannon index for each row
    complexity = np.array([shannon_index(row) for row in ratios_array])
    
    return pl.Series(complexity)

def shannon_index(ratios):
    # Remove zero values and normalize
    nonzero_ratios = ratios[ratios > 0]
    if len(nonzero_ratios) == 0:
        return 0
    normalized = nonzero_ratios / nonzero_ratios.sum()
    # Calculate Shannon index
    return -(normalized * np.log(normalized)).sum()

def aggregate_by_zipcode(df_gsv_with_zipcode, df_segments):
    """
    Aggregate segmentation results by zipcode with temporal matching to survey data
    """
    checkpoint_path = CHECKPOINT_DIR / "segment_stats_by_zipcode.parquet"
    
    def _create_aggregation():
        # Load survey data to get years
        survey_data = load_survey_data()
        
        # Extract years from survey data (assuming there's a year column)
        # Adjust this based on your actual survey data structure
        if 'year' in survey_data.columns:
            survey_years = survey_data['year'].unique().to_list()
        elif 'date' in survey_data.columns:
            survey_years = survey_data.select(pl.col('date').dt.year()).unique().to_list()
        else:
            # If no year information in survey, use a reasonable default range
            print("Warning: No year information found in survey data. Using all available GSV years.")
            survey_years = None
        
        # Merge GSV metadata with segmentation results
        df_merged = df_gsv_with_zipcode.join(df_segments, left_on='panoid', right_on='filename_key')
        
        # Calculate visual complexity for each image
        print("  Calculating visual complexity...")
        df_merged = df_merged.with_columns(
            calculate_visual_complexity(df_merged).alias('visual_complexity')
        )
        
        # Get numeric columns for aggregation
        numeric_cols = [col for col in df_segments.columns 
                       if df_segments[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]
                       and col != 'filename_key']
        numeric_cols.append('visual_complexity')
        
        # Add the new spatial metrics columns if they exist
        spatial_metrics = ['gsv_point_count', 'gsv_density', 'gsv_coverage_ratio']
        for metric in spatial_metrics:
            if metric in df_merged.columns:
                if metric not in numeric_cols:
                    numeric_cols.append(metric)
        
        # If we have survey years and GSV years, we'll create year-specific aggregations
        year_specific_dfs = []
        
        if 'gsv_year' in df_merged.columns and survey_years is not None:
            print("  Creating year-matched aggregations...")
            
            # For each survey year, find GSV data within +/- 2 years
            for survey_year in tqdm(survey_years, desc="Processing survey years"):
                # Filter GSV data to +/- 2 years of survey year
                year_min = survey_year - 2
                year_max = survey_year + 2
                
                df_year_filtered = df_merged.filter(
                    (pl.col('gsv_year') >= year_min) & 
                    (pl.col('gsv_year') <= year_max)
                )
                
                # Skip if no data for this year range
                if len(df_year_filtered) == 0:
                    print(f"  No GSV data found within +/- 2 years of survey year {survey_year}")
                    continue
                
                # Group by zipcode and aggregate
                agg_exprs = [
                    pl.col('panoid').count().alias('image_count'),
                    pl.lit(survey_year).alias('survey_year'),
                    # Take the most common city for each zipcode
                    pl.col('city').mode().first().alias('city')
                ]
                
                # Add mean expressions for numeric columns
                agg_exprs.extend([
                    pl.col(col).mean().alias(f"{col}_mean") 
                    for col in numeric_cols
                ])
                
                # Add std expressions for numeric columns
                agg_exprs.extend([
                    pl.col(col).std(ddof=1).cast(pl.Float64).alias(f"{col}_std")
                    for col in numeric_cols
                ])
                
                # Aggregate for this year range
                df_year_agg = df_year_filtered.groupby('zipcode').agg(agg_exprs)
                year_specific_dfs.append(df_year_agg)
        
        # If we have year-specific dataframes, concatenate them
        if year_specific_dfs:
            df_zipcode_years = pl.concat(year_specific_dfs)
            
            print(f"  Created year-matched dataset with {len(df_zipcode_years)} zipcode-year combinations")
            
            # Also create an aggregated version without year matching for backward compatibility
            print("  Creating overall aggregation for backward compatibility...")
        
        # Standard aggregation without year matching (for backward compatibility)
        agg_exprs = [
            pl.col('panoid').count().alias('image_count'),
            # Take the most common city for each zipcode
            pl.col('city').mode().first().alias('city')
        ]
        
        # Add mean expressions
        agg_exprs.extend([
            pl.col(col).mean().alias(f"{col}_mean") 
            for col in numeric_cols
        ])
        
        # Add std expressions
        agg_exprs.extend([
            pl.col(col).std(ddof=1).cast(pl.Float64).alias(f"{col}_std")
            for col in numeric_cols
        ])
        
        df_zipcode_overall = df_merged.groupby('zipcode').agg(agg_exprs)
        
        # Return appropriate dataframe(s)
        if year_specific_dfs:
            # Save both versions
            df_zipcode_years.write_parquet(CHECKPOINT_DIR / "segment_stats_by_zipcode_years.parquet")
            return df_zipcode_years  # Return the year-matched version as primary
        else:
            return df_zipcode_overall
    
    return load_or_create_checkpoint(checkpoint_path, _create_aggregation)

def create_final_dataset(df_survey, df_segment_stats):
    """Create final datasets at both individual and zipcode levels with temporal matching"""
    print("Creating final datasets...")
    
    # Calculate Big Five personality traits
    print("  Calculating Big Five traits...")
    checkpoint_path = CHECKPOINT_DIR / "big_five_traits.parquet"
    
    def _create_big_five():
        return pl.from_pandas(calculate_personality_scores_vectorized(df_survey.to_pandas()))
    
    df_big_five = load_or_create_checkpoint(checkpoint_path, _create_big_five)
    
    # Add Big Five traits to survey data
    df_combined = df_survey.hstack(df_big_five)
    
    # Check if we have year-specific segment stats
    has_temporal_data = 'survey_year' in df_segment_stats.columns
    
    # Extract survey years from record_time column if available
    if 'record_time' in df_survey.columns:
        print("  Extracting year from record_time column...")
        # First, fix any potential year format issues (like " 200" for year 2000)
        df_combined = df_combined.with_columns(
            pl.col('record_time').str.replace(r' 200\b', ' 2000').alias('fixed_record_time')
        )
        # Then extract the year from the fixed date string
        df_combined = df_combined.with_columns(
            pl.col('fixed_record_time').str.extract(r'(\d{4})').cast(pl.Int32).alias('survey_year')
        )
        survey_year_column = 'survey_year'
    elif 'year' in df_survey.columns:
        df_combined = df_combined.with_columns(pl.col('year').alias('survey_year'))
        survey_year_column = 'survey_year'
    elif 'date' in df_survey.columns:
        df_combined = df_combined.with_columns(pl.col('date').dt.year().alias('survey_year'))
        survey_year_column = 'survey_year'
    else:
        survey_year_column = None
    
    # Merge with street view indicators
    print("  Merging with street view indicators...")
    
    if has_temporal_data and survey_year_column:
        print("  Using temporal matching for merging...")
        # Merge on both zipcode and year
        df_individual = df_combined.join(
            df_segment_stats,
            left_on=['now_zip', survey_year_column],
            right_on=['zipcode', 'survey_year'],
            how='inner'  # Only keep rows with both personality and street view data
        )
    else:
        # Fallback to original merging strategy
        df_individual = df_combined.join(
            df_segment_stats,
            left_on='now_zip',
            right_on='zipcode',
            how='inner'  # Only keep rows with both personality and street view data
        )
    
    # Drop rows with missing Big Five indicators
    big_five_traits = ['extraversion', 'agreeableness', 'conscientiousness', 'neuroticism', 'openness']
    df_individual = df_individual.drop_nulls(subset=big_five_traits)
    
    print(f"  Individual dataset shape: {df_individual.shape}")
    print(f"  Number of participants with complete data: {len(df_individual)}")
    
    # Create zipcode level dataset
    print("  Creating zipcode level dataset...")
    
    # First, aggregate personality traits by zipcode
    if has_temporal_data and survey_year_column:
        # Group by both zipcode and year
        groupby_cols = ['now_zip', survey_year_column]
    else:
        groupby_cols = ['now_zip']
    
    agg_exprs = [
        pl.col('now_zip').count().alias('participant_count'),
        *[pl.col(trait).mean().alias(f"{trait}_mean") for trait in big_five_traits],
        *[pl.col(trait).std().alias(f"{trait}_std") for trait in big_five_traits]
    ]
    
    df_zipcode = df_individual.groupby(groupby_cols).agg(agg_exprs)
    
    # Then join with the pre-aggregated segmentation statistics
    if has_temporal_data and survey_year_column:
        # If we have temporal data, join on both zipcode and year
        df_zipcode = df_zipcode.join(
            df_segment_stats.select([
                pl.col('zipcode'),
                pl.col('survey_year'),
                *[pl.col(col) for col in df_segment_stats.columns 
                  if col not in ['zipcode', 'survey_year']]
            ]),
            left_on=['now_zip', survey_year_column],
            right_on=['zipcode', 'survey_year'],
            how='left'
        )
    else:
        # Otherwise, join just on zipcode
        df_zipcode = df_zipcode.join(
            df_segment_stats.select([
                pl.col('zipcode'),
                *[pl.col(col) for col in df_segment_stats.columns if col != 'zipcode']
            ]),
            left_on='now_zip',
            right_on='zipcode',
            how='left'
        )
    
    print(f"  Zipcode dataset shape: {df_zipcode.shape}")
    if has_temporal_data:
        print(f"  Number of unique zipcode-year combinations: {len(df_zipcode)}")
        print(f"  Number of unique zipcodes: {df_zipcode['now_zip'].n_unique()}")
    else:
        print(f"  Number of unique zipcodes: {len(df_zipcode)}")
    
    return df_individual, df_zipcode

def main():
    # Check if final datasets already exist
    if (FINAL_INDIVIDUAL_DATA_PATH.exists() and 
        FINAL_ZIPCODE_DATA_PATH.exists()):
        print("Loading existing final datasets...")
        df_individual = pl.read_parquet(FINAL_INDIVIDUAL_DATA_PATH)
        df_zipcode = pl.read_parquet(FINAL_ZIPCODE_DATA_PATH)
        print(f"Individual dataset loaded: {df_individual.shape}")
        print(f"Zipcode dataset loaded: {df_zipcode.shape}")
        return df_individual, df_zipcode
    
    print("Processing data from scratch...")
    
    # Load survey data
    print("Loading survey data...")
    df_survey = load_survey_data()
    
    # Process GSV data and spatial join
    print("Loading zipcode shapes...")
    gdf_zipcodes = load_zipcode_shapes()
    
    print("Processing GSV metadata...")
    df_gsv = process_gsv_metadata()
    
    print("Processing segmentation results...")
    df_segments = process_segmentation_results()
    
    # Perform spatial join
    df_gsv_with_zipcode = spatial_join_gsv_zipcodes(df_gsv, gdf_zipcodes)
    
    # Aggregate segmentation results
    print("Aggregating segmentation results...")
    df_segment_stats = aggregate_by_zipcode(df_gsv_with_zipcode, df_segments)
    
    # Create final datasets
    df_individual, df_zipcode = create_final_dataset(df_survey, df_segment_stats)
    
    # Save final datasets
    print("Saving final datasets...")
    FINAL_INDIVIDUAL_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_individual.write_parquet(FINAL_INDIVIDUAL_DATA_PATH)
    df_zipcode.write_parquet(FINAL_ZIPCODE_DATA_PATH)
    
    print("Dataset preparation completed!")
    return df_individual, df_zipcode

if __name__ == "__main__":
    main() 