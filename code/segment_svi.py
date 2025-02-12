from zensvi.cv import Segmenter
import os

segmenter = Segmenter(task="panoptic", dataset="mapillary", device="cuda")
place_list = ["Austin, Texas", "Dallas, Texas", "Houston, Texas", "San Antonio, Texas"]
for place in place_list:
    place_clean = place.lower().replace(", ", "_").replace(" ", "_")
    # get a list of all the folders in f"data/raw/{place_clean}/gsv_panorama/"
    folders_list = os.listdir(f"data/raw/{place_clean}/gsv_panorama/")
    for folder in folders_list:
        # make sure to create the output directory
        os.makedirs(f"data/processed/{place_clean}/{folder}/", exist_ok=True)
        input_dir = f"data/raw/{place_clean}/gsv_panorama/{folder}/"
        output_dir = f"data/processed/{place_clean}/{folder}/"
        segmenter.segment(
            input_dir,
            dir_summary_output=output_dir,
            batch_size=8,
            save_format="csv",
            csv_format="wide",
            max_workers=2,
        )
