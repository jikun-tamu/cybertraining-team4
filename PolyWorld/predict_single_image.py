import json
import time
import numpy as np
from pathlib import Path

import torch
from PIL import Image

from models.matching import OptimalMatching
from models.backbone import R2U_Net, NonMaxSuppression, DetectionBranch


# --------------------------------------------------------
# Utility to compute bounding box
# --------------------------------------------------------
def bounding_box_from_points(points):
    points = np.array(points).flatten()
    xs = points[0::2]
    ys = points[1::2]
    return [int(xs.min()), int(ys.min()), int(xs.max() - xs.min()), int(ys.max() - ys.min())]


def single_annotation(image_id, poly):
    return {
        "image_id": int(image_id),
        "category_id": 100,
        "score": 1,
        "segmentation": poly,
        "bbox": bounding_box_from_points(poly),
    }


# --------------------------------------------------------
# Preprocess image to match CrowdAI loader
# Resize → 320 × 320, normalize to [0,1]
# --------------------------------------------------------
def load_and_prepare_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((320, 320), Image.BILINEAR)
    arr = np.array(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)[None]  # [1,3,320,320]
    return img, tensor


# --------------------------------------------------------
# Main Single-Image Prediction Function
# --------------------------------------------------------
def run_single_image_prediction(image_path, output_json_path, weights_dir):

    image_path = Path(image_path)
    output_json_path = Path(output_json_path)
    weights_dir = Path(weights_dir)

    print(f"➡️ Running PolyWorld on: {image_path}")

    # ----------------------------
    # Load pretrained PolyWorld modules
    # ----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    backbone = R2U_Net().to(device).train()  # train() for BatchNorm behaviour
    head = DetectionBranch().to(device).train()
    suppression = NonMaxSuppression().to(device)
    matching = OptimalMatching().to(device).train()

    print("Loading pretrained weights...")
    backbone.load_state_dict(torch.load(weights_dir / "polyworld_backbone", map_location=device))
    head.load_state_dict(torch.load(weights_dir / "polyworld_seg_head", map_location=device))
    matching.load_state_dict(torch.load(weights_dir / "polyworld_matching", map_location=device))

    # ----------------------------
    # Prepare the image
    # ----------------------------
    img_resized, img_tensor = load_and_prepare_image(image_path)
    img_tensor = img_tensor.to(device).float()

    # ----------------------------
    # Run the model
    # ----------------------------
    t0 = time.time()
    with torch.no_grad():
        features = backbone(img_tensor)
        occupancy_grid = head(features)
        _, graph_points = suppression(occupancy_grid)
        polygons_batch = matching.predict(img_tensor, features, graph_points)

    elapsed = time.time() - t0
    print(f"⏱ Inference time: {elapsed:.4f} seconds")

    # ----------------------------
    # Build JSON predictions
    # ----------------------------
    preds = []
    image_id = 1  # arbitrary since this is single-image testing
    polygons = polygons_batch[0] if len(polygons_batch) > 0 else []

    for poly in polygons:
        preds.append(single_annotation(image_id, [poly]))

    # ----------------------------
    # Save results
    # ----------------------------
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(json.dumps(preds))

    print(f"✅ Saved predictions to: {output_json_path}")


# --------------------------------------------------------
# ENTRY POINT
# --------------------------------------------------------
if __name__ == "__main__":

    run_single_image_prediction(
        image_path="/media/data/building_instance_tamu/test/images/hurricane-harvey_00000023_pre_disaster.png",
        output_json_path="/media/gisense/xihan/250812_tamu_cybertraining_team4/PolyWorld/sample_test/predictions.json",
        weights_dir="/media/gisense/xihan/250812_tamu_cybertraining_team4/PolyWorld/trained_weights"
    )
