from zensvi.download import GSVDownloader
import os 

api_key = os.environ["GSV_API_KEY"]
downloader = GSVDownloader(grid=True, grid_size = 100, gsv_api_key=api_key)
place_list = ["Austin, Texas", "Dallas, Texas", "Houston, Texas", "San Antonio, Texas"]
for place in place_list:
    place_clean = place.lower().replace(", ", "_").replace(" ", "_")
    output_dir = f"data/raw/{place_clean}/"
    # make sure to create the output directory
    os.makedirs(output_dir, exist_ok=True)
    # downloader.download_svi(output_dir,
    #                         input_place_name=place,
    #                         augment_metadata=True,
    #                         )
    downloader.update_metadata(input_pid_file=f"data/raw/{place_clean}/gsv_pids.csv",
                        output_pid_file=f"data/raw/{place_clean}/gsv_pids_augmented.csv",
                        verbosity=1
                        )
