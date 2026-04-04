# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Disaster impact assessment pipeline using satellite imagery, building damage prediction, and demographic data.

**Active deliverables:**
- **`stage1/`** — batch building-detection package using SAM3 (Segment Anything Model 3) via samgeo (formerly `SAM3_Claude/`)
- **`pipeline/`** — combined Stage 1 + Stage 2 pipeline (formerly `II_package/`), collaborator-integrated
- **`evaluation/`** — xView2 benchmark scripts and results

**Exploration only** (not production): `archive/` (formerly `exploration/`) contains Mask R-CNN, PolyWorld, GeoAI_QuishengWu, and earlier SAM3 variants

## Environments

**Always use `geoai_sam`** for SAM3 work. The other env (`geoai_sam3`) has an older samgeo without `SamGeo3`.

```bash
conda activate geoai_sam          # interactive
conda run -n geoai_sam <command>  # non-interactive / scripted
```

## stage1 — SAM3 Building Detection Package

**Location**: `stage1/`  (formerly `SAM3_Claude/`)
**Install** (editable, no deps): `pip install -e stage1 --no-deps`
**One-time HF login** (caches token): `python -c "from huggingface_hub import login; login()"`

### Running the pipeline

```bash
# Dry-run (list images, no inference)
python -m sam3_building_identifier --input-dir <dir> --output-dir <dir> --dry-run

# Process N images
python -m sam3_building_identifier \
    --input-dir /media/data/building_instance_tamu/test/images \
    --output-dir /tmp/sam3_out \
    --max-images 10

# Disaster-type filter (auto-detects xView2 _pre_disaster/_post_disaster suffixes)
python -m sam3_building_identifier --disaster-type pre --device cuda:1 ...

# Re-run over existing outputs (no skip)
python -m sam3_building_identifier --no-skip ...
```

### Smoke test (3 images, ~30s on GPU)

```bash
conda run -n geoai_sam python stage1/tests/smoke_test.py
```

### Key default parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `--prompt` | `"building"` | Text prompt for SAM3 |
| `--tile-size` | `512` | Tile size; SAM3 resizes to 1024 internally, so tiling preserves detail. 0 to disable |
| `--overlap` | `64` | Overlap between adjacent tiles |
| `--min-size` | `100` | Min mask area in pixels |
| `--min-polygon-area` | `100.0` | Filters tile-boundary stitching artifacts |
| `--epsilon` | `2.0` | Douglas-Peucker polygon approx |
| `--batch-size` | `1` | Keep at 1 for A6000 |
| `--disaster-type` | `auto` | Filters images by filename suffix |

## Package Architecture (`sam3_building_identifier/`)

```
config.py          — PipelineConfig dataclass (all tuneable params + computed dirs)
model.py           — SAM3Model: lazy-loads SamGeo3, wraps single/batch inference
pipeline.py        — run_pipeline(): batch loop with tiling support, JSON output, run_summary.json
tiling.py          — generate_tiles(), stitch_masks() for tile-based inference
mask_to_polygon.py — masks_to_instances(): geoai.orthogonalize() → cv2 fallback
utils.py           — discover_images(), timer() context manager, log()
__main__.py        — argparse CLI → PipelineConfig → run_pipeline()
```

**Critical SamGeo3 API behavior**:
- `SamGeo3.generate_masks()` returns **None** — results stored in `self.masks`, `self.boxes`, `self.scores`
- `SamGeo3.generate_masks_batch()` results stored in `self.batch_results` (list of per-image dicts)
- Count masks with `len(getattr(sam3, 'masks', None) or [])`

**Vectorization**: prefers `geoai.orthogonalize()` (matches notebook ground-truth); falls back to OpenCV `cv2.findContours` + Shapely if geoai unavailable.

## Output Schema

Per-image: `predictions/<stem>_prediction.json`
Run aggregate: `run_summary.json`

```json
{
  "image": {"path", "stem", "width", "height", "disaster_type"},
  "instances": [{"id", "uid", "bbox_xyxy":[x1,y1,x2,y2], "polygon":[[x,y],...], "area_px", "confidence"}],
  "timing": {"inference_sec", "postprocess_sec", "total_sec"},
  "summary": {"num_instances", "status"}
}
```

Other outputs per image (when detections > 0):
- `masks/<stem>_mask.tif` — instance mask raster
- `masks/<stem>_scores.tif` — confidence scores raster
- `annotations/<stem>_ann.png` — visualization overlay

## Test Data

- **1866 images** at `/media/data/building_instance_tamu/test/images/` (933 pre, 933 post; 1024×1024 PNG, xView2 naming)
- **Notebook ground-truth outputs**: `/media/data/building_instance_tamu/sam3/test/`

## Other Components

| Directory | Purpose |
|-----------|---------|
| `pipeline/` | Combined Stage 1+2 pipeline (collaborator-integrated); LA fire validation outputs in `pipeline/outputs/` |
| `evaluation/` | xView2 benchmark script (`evaluate_predictions.py`) + results in `evaluation/results/sam3_eval/` |
| `results/` | LA fire run outputs (masks, overlays, logs) |
| `src/cybertraining_team4/` | Core project package (training, evaluation stages) |
| `notebooks/` | EDA and validation notebooks |
| `data/` | LA fire data: raw/ (immutable) → interim/ → processed/ |
| `archive/` | Experimental/comparison work, formerly `exploration/` (not production) |
| `archive/SAM3_notebooks/` | Original SAM3 Jupyter notebooks |
| `archive/SAM3_Final/` | Alternative SAM3 pipeline with georeferencing recovery and GeoJSON/GPKG output |
| `archive/PolyWorld/` | PolygonGNN (CVPR 2022) building extraction |
| `archive/Mask_R-CNN/` | Mask R-CNN training code |
| `archive/GeoAI_QuishengWu/` | GeoAI experiments |
| `archive/corrected_model/` | Early Mask R-CNN checkpoint (corrected_building_segmentation_model.pth) + eval results |

## GPU Notes

- 2× NVIDIA RTX A6000 (47.5 GB each); specify `--device cuda:0` or `cuda:1`
- GPU memory cleared between images via `torch.cuda.empty_cache()` + `gc.collect()`
