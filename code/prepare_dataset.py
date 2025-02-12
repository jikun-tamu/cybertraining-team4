import os
import polars as pl
import geopandas as gpd
from pathlib import Path
import pyreadstat
import numpy as np
from tqdm import tqdm
from big_five_analysis import calculate_personality_scores_vectorized

# Configuration
DATA_DIR = Path("/media/data/personality")
WORKSPACE_DIR = Path(os.getcwd())
RAW_DATA_DIR = WORKSPACE_DIR / "data/raw"
PROCESSED_DATA_DIR = WORKSPACE_DIR / "data/processed"
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

def spatial_join_gsv_zipcodes(df_gsv, gdf_zipcodes):
    """Perform spatial join between GSV points and zipcode polygons"""
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
        
        # Convert back to Polars DataFrame
        df_joined = pl.from_pandas(gdf_joined.drop(columns=['geometry', 'index_right']))
        return df_joined.rename({'ZCTA5CE10': 'zipcode'})
    
    return load_or_create_checkpoint(checkpoint_path, _create_spatial_join)

def process_segmentation_results():
    """Process segmentation results from all cities and batches"""
    checkpoint_path = CHECKPOINT_DIR / "segmentation_results.parquet"
    
    def _create_segmentation_results():
        results_dfs = []
        
        for city_dir in PROCESSED_DATA_DIR.glob("*"):
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
    """Aggregate segmentation results by zipcode"""
    checkpoint_path = CHECKPOINT_DIR / "segment_stats_by_zipcode.parquet"
    
    def _create_aggregation():
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
        
        # Group by zipcode and aggregate mean and std
        print("  Aggregating statistics by zipcode...")
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
        
        # Add std expressions - ensure we get single values
        agg_exprs.extend([
            pl.col(col).std(ddof=1).cast(pl.Float64).alias(f"{col}_std")
            for col in numeric_cols
        ])
        
        return df_merged.groupby('zipcode').agg(agg_exprs)
    
    return load_or_create_checkpoint(checkpoint_path, _create_aggregation)

def create_final_dataset(df_survey, df_segment_stats):
    """Create final datasets at both individual and zipcode levels"""
    print("Creating final datasets...")
    
    # Calculate Big Five personality traits
    print("  Calculating Big Five traits...")
    checkpoint_path = CHECKPOINT_DIR / "big_five_traits.parquet"
    
    def _create_big_five():
        return pl.from_pandas(calculate_personality_scores_vectorized(df_survey.to_pandas()))
    
    df_big_five = load_or_create_checkpoint(checkpoint_path, _create_big_five)
    
    # Add Big Five traits to survey data
    df_combined = df_survey.hstack(df_big_five)
    
    # Merge with street view indicators based on now_zip
    print("  Merging with street view indicators...")
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
    df_zipcode = df_individual.groupby('now_zip').agg([
        pl.col('now_zip').count().alias('participant_count'),
        *[pl.col(trait).mean().alias(f"{trait}_mean") for trait in big_five_traits],
        *[pl.col(trait).std().alias(f"{trait}_std") for trait in big_five_traits]
    ])
    
    # Then join with the pre-aggregated segmentation statistics
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