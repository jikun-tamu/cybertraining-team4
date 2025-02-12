from zensvi.download import GSVDownloader
import os 

downloader = GSVDownloader(grid=True, grid_size = 100)
place_list = ["Austin, Texas", "Dallas, Texas", "Houston, Texas", "San Antonio, Texas"]
for place in place_list:
    place_clean = place.lower().replace(", ", "_").replace(" ", "_")
    output_dir = f"data/raw/{place_clean}/"
    # make sure to create the output directory
    os.makedirs(output_dir, exist_ok=True)
    downloader.download_svi(output_dir,
                            input_place_name=place)
