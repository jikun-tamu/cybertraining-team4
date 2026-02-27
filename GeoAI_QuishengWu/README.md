# GeoAI_QuishengWu — Building Footprint Extraction

Explores building footprint detection using the `geoai` library
(`BuildingFootprintExtractor`, pretrained USA model).

## Structure

```
notebooks/
  building_footprints_usa.ipynb       ← main reference (end-to-end, has output)
  building_regularization.ipynb       ← exploratory: regularization methods demo
scripts/
  building_footprint_combined.py      ← batch production script for chips_600m
models/
  building_footprints_usa.pth         ← pretrained model (169 MB, HF download)
data/
  naip_train.tif                      ← NAIP input imagery
  naip_train_buildings.geojson        ← reference building labels
outputs/
  buildings.geojson                   ← raw detections (749 buildings)
  building_masks.{tif,geojson}        ← mask raster + vectorized masks
  naip_buildings*.png                 ← visualization overlays
```

## Key entry point

Open `notebooks/building_footprints_usa.ipynb` — runs end-to-end:
download NAIP → detect → regularize → visualize. Achieved **92.8% rectangularity**
on 664 regularized buildings.

## Smoke test

```bash
cd notebooks
jupyter nbconvert --to notebook --execute building_footprints_usa.ipynb \
    --ExecutePreprocessor.timeout=120
```

Or open the notebook and run cells 1–18 (detection + regularization).
The model is cached at `../models/building_footprints_usa.pth`.

## Dependencies

```
geoai, leafmap, torch, rasterio, geopandas
```
Activate the `geoai_sam` conda environment before running.
