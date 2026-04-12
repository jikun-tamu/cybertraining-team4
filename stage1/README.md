# SAM3 Building Identifier

Batch pipeline that uses **SAM3** (via `samgeo`) to detect buildings in
satellite imagery and output per-instance bounding boxes, polygons, and
confidence scores as JSON.

## What it does

1. Scans an input folder for images (filters to `_pre_disaster` images by
   default; falls back to all images if that naming is absent).
2. **Tiles** each image into 512px overlapping patches (SAM3 internally
   resizes to ~1024px, so tiling preserves detail on large images).
3. Runs SAM3 text-prompted segmentation (`prompt="building"`) on each tile.
4. **Stitches** tile masks back into full-image masks using score-based
   replacement at overlaps.
5. Saves:
   - `masks/`       -- instance mask TIF + per-pixel scores TIF
   - `annotations/` -- PNG visualization with colored masks and scores
   - `predictions/` -- one JSON per image with bboxes, polygons, confidence
   - `run_summary.json` -- aggregate stats for the whole run

## Requirements

Use the **`geoai_sam`** conda environment:

```bash
conda activate geoai_sam
```

Install this package (once, development mode):

```bash
pip install -e stage1 --no-deps
```

A Hugging Face login is required the first time SAM3 weights are downloaded:

```bash
python -c "from huggingface_hub import login; login()"
```

## Quick test (3 images)

```bash
conda run -n geoai_sam python -m sam3_building_identifier \
    --input-dir /media/data/building_instance_tamu/test/images \
    --output-dir /tmp/sam3_test \
    --max-images 3
```

Or run the smoke test:

```bash
conda run -n geoai_sam python stage1/tests/smoke_test.py
```

## Run on a full folder

```bash
conda run -n geoai_sam python -m sam3_building_identifier \
    --input-dir /media/data/building_instance_tamu/test/images \
    --output-dir /media/data/building_instance_tamu/xview2_sam3_outputs/test
```

The pipeline skips images whose `_prediction.json` already exists
(`--no-skip` to override). Re-runs are safe.

## Key parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--input-dir` | (see config) | Folder of input images |
| `--output-dir` | (see config) | Root output folder |
| `--prompt` | `"building"` | Text prompt for SAM3 |
| `--tile-size` | `512` | Tile size in pixels (0 to disable) |
| `--overlap` | `64` | Overlap between tiles in pixels |
| `--min-size` | `100` | Minimum mask area (pixels) |
| `--min-polygon-area` | `100.0` | Min polygon area after vectorization |
| `--epsilon` | `2.0` | Polygon simplification tolerance |
| `--disaster-type` | `auto` | `pre`, `post`, `all`, or `auto` |
| `--device` | auto | `cuda`, `cuda:1`, `cpu` |
| `--max-images N` | all | Process only the first N images |
| `--no-skip` | -- | Re-process already-done images |
| `--dry-run` | -- | List images that would be processed, then exit |

Full list: `python -m sam3_building_identifier --help`

## Output format

Each image produces a `predictions/<stem>_prediction.json`:

```json
{
  "image": {"path": "...", "stem": "...", "width": 1024, "height": 1024, "disaster_type": "pre"},
  "instances": [
    {
      "id": 1,
      "uid": "b90f65d8-...",
      "bbox_xyxy": [429, 83, 495, 134],
      "polygon": [[453, 83], [453, 84], ...],
      "area_px": 2128.0,
      "confidence": 0.8094
    }
  ],
  "timing": {"inference_sec": 0.44, "postprocess_sec": 5.0, "total_sec": 5.9},
  "summary": {"num_instances": 3, "status": "ok"}
}
```

## Package architecture

```
config.py          -- PipelineConfig dataclass (all tuneable params)
model.py           -- SAM3Model: lazy-loads SamGeo3, wraps inference
pipeline.py        -- run_pipeline(): batch loop with tiling support
tiling.py          -- generate_tiles(), stitch_masks() for tile-based inference
mask_to_polygon.py -- masks_to_instances(): geoai.orthogonalize() -> cv2 fallback
utils.py           -- discover_images(), timer(), log()
__main__.py        -- argparse CLI -> PipelineConfig -> run_pipeline()
```

## Why tiling?

SAM3's vision encoder internally resizes inputs to ~1024x1024. For large
images (e.g., 1966x1966 LA fire chips), this causes 2x downscaling and
small buildings are missed. Tiling to 512px means each tile gets 2x
**upscaled** to 1024, preserving fine detail. Score-based stitching
resolves overlaps by keeping the higher-confidence detection at each pixel.

## Prompt experiments

Initial experiments on cell_00365 (LA fire, mixed residential):

| Prompt | Detections | Notes |
|--------|-----------|-------|
| `"building"` | 387 | Current default, consistent with xView2 benchmarks |
| `"house"` | 410 | +6% recall, best single prompt |
| `"apartment building"` | 17 | Only finds large structures |
| `"rooftop"` | 182 | Misses many buildings |
| Ensemble (building+house) | 419 | +2.2% over best single, 2x inference cost |

Full comparison figures in `results/prompt_experiment/`.
Future work: multi-prompt ensemble with score-based mask merging (same
algorithm as tile stitching). See config.py docstring for details.

## Notes

- **GPU**: 2x NVIDIA RTX A6000 (47.5 GB each). Keep `--batch-size 1`.
  GPU memory cleared between images via `torch.cuda.empty_cache()`.
- **Tiling**: Default 512px tiles with 64px overlap. Set `--tile-size 0`
  to process full images (not recommended for images > 1024px).
- **Polygon method**: `geoai.orthogonalize()` with `epsilon=2` (matches
  notebooks). Falls back to OpenCV contours if geoai is unavailable.
