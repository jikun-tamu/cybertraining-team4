from zensvi.cv import ObjectDetector
import os

detector = ObjectDetector(
    text_prompt="American flag .",  # specify the object(s) (e.g., single type: "building", multi-type: "car . tree")
    box_threshold=0.45,
    text_threshold=0.25
)
place_list = ["Austin, Texas", "Dallas, Texas", "Houston, Texas", "San Antonio, Texas"]
for place in place_list:
    place_clean = place.lower().replace(", ", "_").replace(" ", "_")
    # get a list of all the folders in f"data/raw/{place_clean}/gsv_panorama/"
    # folders_list = os.listdir(f"data/raw/{place_clean}/gsv_panorama/")
    folders_list = os.listdir(f"/media/data/streetview/{place_clean}/gsv_panorama/")
    for folder in folders_list:
        # make sure to create the output directory
        os.makedirs(f"data/processed/detection/{place_clean}/{folder}/", exist_ok=True)
        # input_dir = f"data/raw/{place_clean}/gsv_panorama/{folder}/"
        input_dir = f"/media/data/streetview/{place_clean}/gsv_panorama/{folder}/"
        output_dir = f"data/processed/detection/{place_clean}/{folder}/"
        output_image_dir = f"data/processed/detection/{place_clean}/{folder}/images/"

        detector.detect_objects(
            dir_input=input_dir,
            dir_summary_output=output_dir,
            # dir_image_output=output_image_dir,
            group_by_object = True,
            max_workers= 8,
            save_format="csv" # or "csv"
        )