# SAM3 Final Pipeline (Pre‑Disaster Building Footprints)

This folder contains a reusable pipeline that converts pre‑disaster imagery into building footprint polygons using SAM3. It is based on the two SAM3 notebooks and is designed for batch inference with optional tiling and polygon regularization.

## What It Does
- Runs SAM3 inference to detect building instances.
- Saves instance masks and score rasters.
- Polygonizes masks into building footprints.
- Optionally regularizes polygons (orthogonalization or simplification).
- Exports GeoJSON and GeoPackage outputs.

## Inputs
- PNG/JPG/TIFF images. PNGs often have **no georeferencing**.
- If georeferencing can be recovered, outputs are in real-world coordinates.
- If georeferencing cannot be recovered, outputs are still exported but are **image‑pixel coordinates**.

## Georeferencing Recovery (PNG Inputs)
The pipeline tries these methods, in order:
1. World file sidecar (`.wld`, `.pgw`, `.tfw`, `.jgw`) with same base name.
2. GeoTIFF with the same base name in the same folder.
3. Optional metadata table (CSV/JSON) passed via `--metadata`.

If none are found, outputs are still valid GeoJSON/GeoPackage but with pixel coordinates and properties:
- `pixel_coord_system: "image"`
- `transform_source: "none"`

## Setup
Use a Python environment with GPU support for SAM3 if available.

Minimal requirements:
- `samgeo`
- `torch`
- `huggingface_hub`
- `rasterio`
- `shapely`
- `numpy`
- `Pillow`
- `geopandas` (for GeoPackage export)
- `geoai-py` (optional, for orthogonalization)

Install:
```bash
pip install -r SAM3_Final_20260226/requirements.txt
```

## Hugging Face Access
SAM3 weights require Hugging Face access. Provide your token via environment variable:
```bash
export HF_TOKEN=YOUR_TOKEN
```

## Run (Single Image or Folder)
```bash
python SAM3_Final_20260226/scripts/run_sam3_building_infer.py \
  --input /path/to/images \
  --output /path/to/output \
  --prompt building \
  --min-size 100 \
  --tile-size 1024 \
  --overlap 128 \
  --regularize geoai \
  --epsilon 2
```

If you do not want to use Hugging Face download:
```bash
python SAM3_Final_20260226/scripts/run_sam3_building_infer.py \
  --input /path/to/images \
  --output /path/to/output \
  --no-hf
```

## Outputs
- `buildings.geojson` (always)
- `buildings.gpkg` (if `geopandas` is installed)
- `masks/` and `annotations/` directories
- `run_summary.json`

Each feature includes:
- `image_id`, `tile_id`
- `instance_id`, `confidence`
- `area`, `perimeter`
- processing parameters (prompt, min_size, regularize, epsilon)
- georeferencing provenance (`georef_source`, `pixel_coord_system` when applicable)

## Regularization Options
- `none`: no geometry changes.
- `simplify`: shapely simplification with `epsilon`.
- `min_rot_rect`: minimum rotated rectangle per footprint.
- `geoai`: uses `geoai.orthogonalize()` for right‑angle regularization (recommended if available).

## Notes
- The pipeline is designed for **pre‑disaster** building footprint extraction only.
- The notebook workflows remain unchanged and are used as the source of truth.
- No hard‑coded secrets or paths are used; use CLI args + env vars.
