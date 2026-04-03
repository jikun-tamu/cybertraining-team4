# Stage 1 — SAM3 Building Detection: Technical Summary

**Date**: 2026-04-01

---

## What SAM3 Does in This Project

SAM3 (Segment Anything Model 3) performs **zero-shot building instance segmentation** from satellite imagery. It requires no training — instead, it uses a text prompt ("building") to identify and segment individual building footprints.

The project wraps SAM3 via the `samgeo` (segment-geospatial) library's `SamGeo3` class, which handles:
- Model loading from HuggingFace Hub
- Text-prompted mask generation
- Tiled inference for large images

**Key insight**: SAM3 is a foundation model. Stage 1 does not train anything — it is purely inference-time segmentation with post-processing to extract clean building polygons.

---

## Data Format

### Input
- **xView2**: 1024x1024 PNG images, RGB, named `{base}_pre_disaster.png` / `{base}_post_disaster.png`
- **LA Fire**: GeoTIFF tiles (600m cells), variable size, RGB visual products from Maxar ARD

### Output (per image)
| File | Format | Content |
|------|--------|---------|
| `predictions/{stem}_prediction.json` | JSON | Instance polygons, bboxes, confidence, timing |
| `masks/{stem}_mask.tif` | GeoTIFF (8-bit) | Instance mask raster (pixel values = instance IDs) |
| `masks/{stem}_scores.tif` | GeoTIFF (32-bit float) | Per-pixel confidence scores |
| `annotations/{stem}_ann.png` | PNG | Visualization overlay |
| `run_summary.json` | JSON | Aggregate statistics for entire run |

### Instance JSON Schema
```json
{
  "image": {"path": "...", "stem": "...", "width": 1024, "height": 1024, "disaster_type": "pre"},
  "instances": [
    {
      "id": 0,
      "uid": "guatemala-volcano_00000000_pre_disaster__inst_000",
      "bbox_xyxy": [x1, y1, x2, y2],
      "polygon": [[x, y], ...],
      "area_px": 1234,
      "confidence": 0.87
    }
  ],
  "timing": {"inference_sec": 0.95, "postprocess_sec": 0.08, "total_sec": 1.03},
  "summary": {"num_instances": 15, "status": "ok"}
}
```

---

## How "Training" Works (It Doesn't)

SAM3 is used as a **frozen foundation model**. There is no fine-tuning or training step.

The only tuning is prompt engineering:
- Default prompt: `"building"` (22,824 buildings detected on test set)
- Tested alternatives:
  - `"house"`: 24,999 detections (higher recall, lower precision)
  - `"rooftop"`: 3,699 detections (very low recall)
  - `"structure"`: 9,760 detections
  - `"building_rooftop"` (dual prompt): 5,085 detections

Results stored at `/media/data/building_instance_tamu/sam3_prompt_experiments/`.

The `"house"` prompt showed the best F1 in experiments (see `evaluation/results/prompt_experiments/`).

---

## How Inference Works

### Pipeline Flow (`stage1/sam3_building_identifier/pipeline.py`)

```
1. Discover images (discover_images() in utils.py)
   - Glob for *.png, *.tif, *.jpg
   - Filter by disaster_type (pre/post/auto)
   - Skip already-processed images (unless --no-skip)

2. For each image:
   a. Load with PIL/rasterio
   b. Initialize SamGeo3 model (lazy, first image only)
      - Backend: "meta" (HuggingFace Hub)
      - Device: cuda:0 or cuda:1

   c. Generate masks via SamGeo3.generate_masks()
      - Text prompt: "building"
      - CRITICAL: returns None — results stored in self.masks, self.boxes, self.scores

   d. Post-process masks → polygons (mask_to_polygon.py)
      - Filter by min_size (100px default)
      - Vectorize: geoai.orthogonalize() preferred, cv2.findContours fallback
      - Simplify: Douglas-Peucker with epsilon=2.0
      - Extract bounding boxes, areas, confidence scores

   e. Save outputs:
      - Instance mask TIF + scores TIF
      - Prediction JSON with polygons
      - Annotation overlay PNG

   f. Clear GPU memory (torch.cuda.empty_cache() + gc.collect())

3. Write run_summary.json with aggregate statistics
```

### Key Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `--prompt` | `"building"` | Text prompt for SAM3 |
| `--min-size` | `100` | Minimum mask area in pixels to keep |
| `--epsilon` | `2.0` | Douglas-Peucker polygon simplification tolerance |
| `--batch-size` | `1` | Images per batch (keep at 1 for A6000) |
| `--device` | `cuda:0` | GPU device |
| `--disaster-type` | `auto` | Filter: `pre`, `post`, or `auto` (by filename suffix) |
| `--no-skip` | `false` | Re-process existing outputs |

### CLI Entry Point

```bash
# Standard usage
conda run -n geoai_sam python -m sam3_building_identifier \
    --input-dir /media/data/building_instance_tamu/test/images \
    --output-dir /tmp/sam3_out \
    --max-images 10

# GeoTIFF batch (LA fire)
python stage1/run_all_geotiffs.py
```

---

## Output Artifacts Handed to Stage 2

Stage 2 consumes Stage 1 outputs through `generate_shared_instance_subimages.py`, which reads:

1. **Prediction JSON files** — specifically the `instances[].polygon` field (WKT pixel coordinates)
2. **Pre-disaster images** — for cropping around each building centroid
3. **Post-disaster images** — same crop geometry applied to each post date

From these, Stage 2 generates:
- 256x256 pre/post RGB crops centered on each building
- Binary building mask (M) rasterized from the polygon
- Binary ring mask (R) = dilated M minus M (48px radius context)

**The exact handoff is the prediction JSON** — everything else is derived from it plus the raw imagery.

---

## Package Architecture

```
stage1/sam3_building_identifier/
├── __main__.py         CLI entry point (argparse → PipelineConfig → run_pipeline)
├── config.py           PipelineConfig dataclass (all parameters + computed directories)
├── model.py            SAM3Model: lazy SamGeo3 initialization, single/batch inference
├── pipeline.py         run_pipeline(): main batch loop, JSON output, run_summary
├── mask_to_polygon.py  masks_to_instances(): vectorization with geoai/cv2 fallback
└── utils.py            discover_images(), timer(), log()
```

### Critical API Behavior

```python
# SamGeo3.generate_masks() returns NONE
sam3.generate_masks(image, text_prompt="building")
# Results are in: sam3.masks, sam3.boxes, sam3.scores (NOT the return value)

# Count masks:
n = len(getattr(sam3, 'masks', None) or [])
```

---

## Evaluation Results (xView2)

### Test Set (933 images)
| Metric | Value |
|--------|-------|
| Precision | 0.682 |
| Recall | 0.284 |
| F1 | 0.401 |
| Mean IoU | 0.759 |

### Per-Disaster Breakdown
| Disaster | F1 | Notes |
|----------|-----|-------|
| hurricane-florence | 0.75 | Best — suburban, spread-out buildings |
| socal-fire | 0.59 | Good |
| hurricane-harvey | 0.53 | Good |
| palu-tsunami | 0.42 | Moderate |
| hurricane-michael | 0.38 | Moderate |
| hurricane-matthew | 0.36 | Moderate |
| midwest-flooding | 0.30 | Below average |
| santa-rosa-wildfire | 0.27 | Below average |
| guatemala-volcano | 0.14 | Poor — dense structures |
| mexico-earthquake | 0.06 | Very poor — dense urban |

### Root Cause of Low Recall
- ~30% of images produce zero predictions (SAM3 fails entirely on dense urban)
- Shape quality is excellent when detected (IoU 0.759)
- Problem is **detection completeness**, not detection accuracy

---

## Legacy / Obsolete Stage 1 Branches

### 1. Mask R-CNN (`exploration/Mask_R-CNN/`)
- **What**: Traditional instance segmentation trained on xView2
- **Status**: OBSOLETE — replaced by SAM3
- **Results**: Best IoU 0.644 (vs SAM3's 0.759)
- **Artifacts**: Model checkpoints (best_model.pth, final_model.pth) at `/media/data/building_instance_tamu/Mask_R-CNN_BuildingInstance_Train/models/`
- **Why obsolete**: Required supervised training, lower accuracy, not generalizable

### 2. PolyWorld (`exploration/PolyWorld/`)
- **What**: Polygon-based building extraction (CVPR 2022 PolygonGNN)
- **Status**: OBSOLETE — explored but not adopted
- **Artifacts**: Finetuned model at `/media/data/building_instance_tamu/PolyWorld/finetuned_v2/`
- **Why obsolete**: Complex pipeline, required fine-tuning, not clearly better than SAM3

### 3. GeoAI Building Footprints (`exploration/GeoAI_QuishengWu/`)
- **What**: Building footprint extraction using geoai library
- **Status**: OBSOLETE — superseded by SAM3 integration via samgeo
- **Why obsolete**: Less flexible, SAM3 provides better zero-shot capability

### 4. SAM3_Final (`exploration/SAM3_Final/`)
- **What**: Earlier SAM3 pipeline variant with georeferencing recovery and GeoJSON/GPKG output
- **Status**: PARTIALLY SUPERSEDED — georef features not in current stage1
- **Note**: A copy exists at `pipeline/stage1/SAM3_Final_20260226/` and is used by the combined pipeline's `run_instance_impact_driver.py`
- **Why superseded**: Current `stage1/` package is cleaner, but lacks GeoJSON export

### 5. SAM3 Notebooks (`exploration/SAM3_notebooks/`)
- **What**: Original interactive SAM3 exploration notebooks
- **Status**: ARCHIVE — useful for understanding SAM3 API, not production code
- **Why archived**: Evolved into the `stage1/` package

### 6. Corrected Model (`exploration/corrected_model/`)
- **What**: Early Mask R-CNN checkpoint with corrections
- **Status**: OBSOLETE — same Mask R-CNN branch, just corrected weights
