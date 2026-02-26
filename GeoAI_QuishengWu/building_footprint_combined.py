import os
import glob
import geoai
from tqdm.auto import tqdm

# Define the input and output directories
input_dir = "/media/gisense/xihan/250812_tamu_cybertraining_team4/data/interim/chips_600m"
output_dir = "/media/gisense/xihan/250812_tamu_cybertraining_team4/data/interim/Building_Segmentation/GeoAI_QuishengWu_ESRI"

# Specify the path to the locally downloaded model
model_path = "building_footprints_usa.pth"
extractor = geoai.BuildingFootprintExtractor(model_path=model_path)

# --- Change the chip size here ---
# The default is (512, 512). You can change it to another value, e.g., (1024, 1024).
# Note: Performance may vary as the model was trained on 512x512 chips.
extractor.model.chip_size = (2048, 2048)

# Find all the pre.tif files
search_pattern = os.path.join(input_dir, "**/pre/*.tif")
tif_files = glob.glob(search_pattern, recursive=True)

# Initialize the building footprint extractor if not already done
# Assuming 'extractor' is already created and model is loaded from previous cells.
# If not, you would initialize it here:
# model_path = "building_footprints_usa.pth"
# extractor = geoai.BuildingFootprintExtractor(model_path=model_path)

# Process each tif file
for tif_path in tqdm(tif_files, desc="Processing files"):
    
    # Get the cell number and pre/post part from the path
    parts = tif_path.split(os.sep)
    cell_folder_name = parts[-3] # e.g., cell_00046
    pre_folder_name = parts[-2] # e.g., pre
    
    # Create the output directory
    output_cell_dir = os.path.join(output_dir, f"{cell_folder_name}_{pre_folder_name}")
    os.makedirs(output_cell_dir, exist_ok=True)
    
    # Define the output file path
    base_filename = os.path.basename(tif_path).replace('.tif', '')
    output_geojson_path = os.path.join(output_cell_dir, f"{base_filename}.geojson")
    
    # Extract building footprints as vector
    gdf = extractor.process_raster(
        tif_path,
        output_path=output_geojson_path, # Temporarily save here, will be overwritten by regularized version
        batch_size=16,
        confidence_threshold=0.25,
        overlap=0.25,
        nms_iou_threshold=0.5,
        min_object_area=100,
        max_object_area=None,
        mask_threshold=0.5,
        simplify_tolerance=1.0,
    )

    # Regularize building footprints
    if gdf is not None and not gdf.empty:
        gdf_regularized = extractor.regularize_buildings(
            gdf=gdf,
            min_area=100,
            angle_threshold=15,
            orthogonality_threshold=0.3,
            rectangularity_threshold=0.7,
        )
        
        # Save the regularized footprints
        gdf_regularized.to_file(output_geojson_path, driver='GeoJSON')
        
        # Generate and save visualization
        output_png_path = os.path.join(output_cell_dir, f"{base_filename}_visualization.png")
        extractor.visualize_results(
            tif_path, 
            gdf_regularized, 
            output_path=output_png_path
        )

print("All files processed.")