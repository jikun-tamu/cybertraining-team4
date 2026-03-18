# Mask_R-CNN_train — Building Instance Segmentation

End-to-end Mask R-CNN training pipeline for building instance segmentation
using the `geoai` library. Trained on NAIP imagery; evaluated on xView2 test set.

## Structure

```
notebooks/
  251209_train_instance_segmentation_model_xihan.ipynb  ← PRIMARY (run this)
  train_instance_segmentation_model.ipynb               ← reference template (opengeos/geoai)
data/
  naip_rgb_train.tif                  ← training imagery (2503×1126, 0.6m, EPSG:26911)
  naip_test.tif                       ← test imagery (1024×1024)
  naip_train_buildings.geojson        ← ground-truth building polygons (735 features)
outputs/
  naip_test_instance_prediction.tif   ← predictions at conf ≥ 0.5 (1023 buildings)
  naip_test_instance_prediction_high_conf.tif  ← conf ≥ 0.7 (948 buildings)
  naip_test_instance_prediction.geojson        ← vectorized + orthogonalized polygons
  training_results.png                ← loss/IoU curves (10 epochs)
  training_results_continue.png       ← extended training curves
buildings_instance/                   ← training dataset (auto-generated from data/)
  images/   36 tiles (512×512 RGB GeoTIFF)
  labels/   36 instance masks
  annotations/  36 Pascal VOC XML files
  instance_models/
    best_model.pth    ← best checkpoint (best val IoU: 79.68%)
    final_model.pth   ← last epoch checkpoint
    training_history.pth
    visualizations/   5 sample prediction overlays
```

## Key results

| Metric | Value |
|--------|-------|
| Architecture | Mask R-CNN (ResNet50 + FPN, COCO pretrained) |
| Training set | 28 tiles (80/20 split from 36 tiles) |
| Epochs | 10 (batch size 4, lr 0.005) |
| **Best val IoU** | **79.68%** |
| Test detections | 1023 @ 0.5 conf, 948 @ 0.7 conf |

## Entry point

Open `notebooks/251209_train_instance_segmentation_model_xihan.ipynb`.

Training data is at `/media/data/building_instance_tamu/Mask_R-CNN_BuildingInstance_Train/`
(absolute path, pre-existing on the machine).

## Quick inference (no retraining)

```python
import geoai
model_path = "buildings_instance/instance_models/best_model.pth"
geoai.instance_segmentation(
    input_path="data/naip_test.tif",
    output_path="/tmp/test_pred.tif",
    model_path=model_path,
    num_classes=2, num_channels=3,
    window_size=512, overlap=256, confidence_threshold=0.5, batch_size=4,
)
```

## Dependencies

```
geoai, torch, torchvision, rasterio, geopandas, tqdm
```
Activate the `geoai_sam` conda environment before running.
