import os
import pandas as pd
import geopandas as gpd
import sqlite3
from pathlib import Path
import pyreadstat
from datetime import datetime
import numpy as np
from tqdm import tqdm
from big_five_analysis import calculate_personality_scores_vectorized

# Configuration
DATA_DIR = Path("/media/data/personality")
WORKSPACE_DIR = Path(os.getcwd())
RAW_DATA_DIR = WORKSPACE_DIR / "data/raw"
PROCESSED_DATA_DIR = WORKSPACE_DIR / "data/processed"
DB_PATH = WORKSPACE_DIR / "data/personality.db"

def create_database():
    """Create SQLite database"""
    conn = sqlite3.connect(DB_PATH)
    return conn

def load_survey_data():
    """Load and preprocess the survey data"""
    survey_path = DATA_DIR / "OOS Master Dataset Sept 2022/00_OOS_MASTER DATASET_2022_08_22.sav"
    df, meta = pyreadstat.read_sav(str(survey_path))
    return df

def load_zipcode_shapes():
    """Load and process zipcode shapefile"""
    shp_path = DATA_DIR / "cb_2016_us_zcta510_500k/cb_2016_us_zcta510_500k.shp"
    gdf = gpd.read_file(str(shp_path))
    return gdf

def process_gsv_metadata():
    """Process Google Street View metadata from all cities"""
    city_dfs = []
    
    for city_dir in RAW_DATA_DIR.glob("*"):
        if not city_dir.is_dir():
            continue
            
        metadata_file = city_dir / "gsv_pids.csv"
        if not metadata_file.exists():
            continue
            
        df = pd.read_csv(metadata_file)
        df['city'] = city_dir.name
        city_dfs.append(df)
    
    if not city_dfs:
        raise FileNotFoundError("No GSV metadata files found")
        
    return pd.concat(city_dfs, ignore_index=True)

def spatial_join_gsv_zipcodes(df_gsv, gdf_zipcodes):
    """Perform spatial join between GSV points and zipcode polygons"""
    # Create GeoDataFrame from GSV points
    gdf_gsv = gpd.GeoDataFrame(
        df_gsv,
        geometry=gpd.points_from_xy(df_gsv.lon, df_gsv.lat),
        crs="EPSG:4326"
    )
    
    # Ensure both GeoDataFrames have the same CRS
    if gdf_zipcodes.crs != gdf_gsv.crs:
        gdf_zipcodes = gdf_zipcodes.to_crs(gdf_gsv.crs)
    
    # Perform spatial join
    print("Performing spatial join...")
    gdf_joined = gpd.sjoin(gdf_gsv, gdf_zipcodes[['ZCTA5CE10', 'geometry']], 
                          how='left', predicate='within')
    
    # Rename zipcode column
    gdf_joined = gdf_joined.rename(columns={'ZCTA5CE10': 'zipcode'})
    
    # Drop geometry and index_right columns
    return gdf_joined.drop(columns=['geometry', 'index_right'])

def process_segmentation_results():
    """Process segmentation results from all cities and batches"""
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
            df_ratios = pd.read_csv(pixel_ratios)
            df_counts = pd.read_csv(label_counts)
            
            # Add suffixes to all columns except the joining key
            ratio_cols = [col for col in df_ratios.columns if col != 'filename_key']
            count_cols = [col for col in df_counts.columns if col != 'filename_key']
            
            df_ratios = df_ratios.rename(columns={col: f"{col}_ratios" for col in ratio_cols})
            df_counts = df_counts.rename(columns={col: f"{col}_counts" for col in count_cols})
            
            # Merge ratios and counts
            df = pd.merge(df_ratios, df_counts, on='filename_key')
            df['city'] = city_dir.name
            df['batch'] = batch_dir.name
            
            results_dfs.append(df)
    
    if not results_dfs:
        raise FileNotFoundError("No segmentation result files found")
        
    return pd.concat(results_dfs, ignore_index=True)

def aggregate_by_zipcode(df_gsv_with_zipcode, df_segments):
    """Aggregate segmentation results by zipcode"""
    # Merge GSV metadata with segmentation results
    df_merged = pd.merge(df_gsv_with_zipcode, df_segments, left_on='panoid', right_on='filename_key')
    
    # Create aggregation dictionary for all numeric columns
    numeric_cols = df_segments.select_dtypes(include=[np.number]).columns
    agg_dict = {
        'panoid': 'count'  # Count of images
    }
    
    # Add mean aggregation for all numeric columns from segmentation results
    for col in numeric_cols:
        if col != 'filename_key':  # Skip the key column
            agg_dict[col] = 'mean'
    
    # Group by zipcode
    df_agg = df_merged.groupby('zipcode').agg(agg_dict).reset_index()
    
    # Rename the count column
    df_agg = df_agg.rename(columns={'panoid': 'image_count'})
    
    return df_agg

def create_final_dataset(df_survey, df_segment_stats):
    """Create final dataset with Big Five traits and street view indicators"""
    print("Creating final dataset...")
    
    # Calculate Big Five personality traits
    print("  Calculating Big Five traits...")
    df_big_five = calculate_personality_scores_vectorized(df_survey)
    
    # Add Big Five traits to survey data
    df_combined = pd.concat([df_survey, df_big_five], axis=1)
    
    # Merge with street view indicators based on now_zip
    print("  Merging with street view indicators...")
    df_final = pd.merge(
        df_combined,
        df_segment_stats,
        left_on='now_zip',
        right_on='zipcode',
        how='inner'  # Only keep rows with both personality and street view data
    )
    
    # Drop rows with missing Big Five indicators
    big_five_traits = ['extraversion', 'agreeableness', 'conscientiousness', 'neuroticism', 'openness']
    df_final = df_final.dropna(subset=big_five_traits)
    
    print(f"  Final dataset shape: {df_final.shape}")
    print(f"  Number of participants with complete data: {len(df_final)}")
    
    return df_final

def main():
    # Create database
    conn = create_database()
    
    # Get list of existing tables
    existing_tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn)['name'].values
    
    # Load and process data only if not already in database
    if 'survey_data' in existing_tables:
        print("Survey data already loaded in the database.")
        df_survey = pd.read_sql("SELECT * FROM survey_data;", conn)
    else:
        print("Loading survey data...")
        df_survey = load_survey_data()
        df_survey.to_sql('survey_data', conn, if_exists='replace', index=False)
        print("Survey data saved to database.")
    
    if 'gsv_panoramas' in existing_tables:
        print("GSV panorama data already loaded in the database.")
        df_gsv_with_zipcode = pd.read_sql("SELECT * FROM gsv_panoramas;", conn)
    else:
        print("Loading zipcode shapes...")
        gdf_zipcodes = load_zipcode_shapes()
        
        print("Processing GSV metadata...")
        df_gsv = process_gsv_metadata()
        
        # Perform spatial join
        df_gsv_with_zipcode = spatial_join_gsv_zipcodes(df_gsv, gdf_zipcodes)
        df_gsv_with_zipcode.to_sql('gsv_panoramas', conn, if_exists='replace', index=False)
        print("GSV panorama data saved to database.")
    
    if 'segmentation_results' in existing_tables:
        print("Segmentation results already loaded in the database.")
        df_segments = pd.read_sql("SELECT * FROM segmentation_results;", conn)
    else:
        print("Processing segmentation results...")
        df_segments = process_segmentation_results()
        df_segments.to_sql('segmentation_results', conn, if_exists='replace', index=False)
        print("Segmentation results saved to database.")
    
    if 'segment_stats_by_zipcode' in existing_tables:
        print("Segment stats already loaded in the database.")
        df_segment_stats = pd.read_sql("SELECT * FROM segment_stats_by_zipcode;", conn)
    else:
        print("Aggregating segmentation results...")
        df_segment_stats = aggregate_by_zipcode(df_gsv_with_zipcode, df_segments)
        df_segment_stats.to_sql('segment_stats_by_zipcode', conn, if_exists='replace', index=False)
        print("Segment stats saved to database.")
    
    if 'final_dataset' in existing_tables:
        print("Final dataset already exists in the database.")
        df_final = pd.read_sql("SELECT * FROM final_dataset;", conn)
    else:
        print("Creating final dataset...")
        df_final = create_final_dataset(df_survey, df_segment_stats)
        df_final.to_sql('final_dataset', conn, if_exists='replace', index=False)
        print("Final dataset saved to database.")
    
    # Create indices
    print("Creating indices...")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_survey_youth_zip ON survey_data(youth_zip);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_survey_now_zip ON survey_data(now_zip);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_stats_zipcode ON segment_stats_by_zipcode(zipcode);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_final_now_zip ON final_dataset(now_zip);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_final_youth_zip ON final_dataset(youth_zip);")
    
    conn.commit()
    conn.close()
    
    print("Dataset preparation completed!")

if __name__ == "__main__":
    main() 