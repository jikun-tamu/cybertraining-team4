# SAM3 Building Identifier

Batch pipeline that uses **SAM3** (via `samgeo.SamGeo3`) to detect buildings in
satellite imagery and write per-instance polygons and confidence scores as JSON.
Version 0.2.0 (2026-09): rewritten tiling/merging after the September 2026 audit,
see `reports/sam3_audit_2026-09.md`.

## What it does

1. Finds input images (xView2 `_pre_disaster` images by default; all images if
   that naming is absent). PNG, JPEG and 8-bit GeoTIFF are supported.
2. Splits each image into overlapping 512 px windows. SAM3 resizes every input to
   1008 px, so a 1024 px image loses small buildings unless it is tiled.
3. Runs SAM3 with a text prompt (`"building"`) on every window.
4. Merges detections across windows: two instances from different windows are the
   same building when their masks agree (IoU >= 0.5) inside the region both
   windows see. Merged instances take the mask union and the highest score.
5. Vectorizes each instance with `geoai.orthogonalize()` and writes:
   - `predictions/<stem>_prediction.json` per image (pixel coordinates)
   - `masks/<stem>.tif` (int32 instance labels) and `masks/<stem>_scores.tif`,
     georeferenced when the input is a GeoTIFF
   - `annotations/<stem>_ann.png` polygon overlay
   - `run_summary.json` with the full config, code version and totals

## Requirements

Use the **`geoai_sam`** conda environment (samgeo 1.0.1, SAM 3 weights).
`geoai_sam31` (samgeo 1.4.2, sam3 0.1.4) is only needed for `--model facebook/sam3.1`.

```bash
pip install -e stage1 --no-deps      # once, or set PYTHONPATH=stage1
python -c "from huggingface_hub import login; login()"   # first weight download only
```

## Run

```bash
conda run -n geoai_sam python -m sam3_building_identifier \
    --input-dir /media/data/building_instance_tamu/test/images \
    --output-dir /tmp/sam3_test --max-images 3
```

Tests:

```bash
conda run -n geoai_sam python stage1/tests/test_tiling.py   # merge logic, no GPU
conda run -n geoai_sam python stage1/tests/smoke_test.py    # 3 images end to end
```

Images that already have a prediction JSON are skipped (`--no-skip` to re-run).

## Key parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--prompt` | `building` | SAM3 text prompt |
| `--confidence-threshold` | `0.4` | SAM3 score threshold (SamGeo3 default is 0.5) |
| `--model` | `facebook/sam3` | or `facebook/sam3.1` (needs `geoai_sam31`) |
| `--tile-size` | `512` | Window size; `0` runs on the full image |
| `--overlap` | `64` | Minimum overlap between windows |
| `--merge-iou` | `0.5` | Cross-window merge threshold |
| `--min-size` | `100` | Minimum mask area (px) |
| `--min-polygon-area` | `100` | Minimum polygon area (px^2) |
| `--epsilon` | `2.0` | Polygon simplification tolerance (px) |
| `--disaster-type` | `auto` | `pre`, `post`, `all`, or `auto` |
| `--device` | auto | `cuda`, `cuda:1`, `cpu` |
| `--no-masks`, `--no-annotations` | | Skip the TIF / PNG outputs |

Full list: `python -m sam3_building_identifier --help`.

## Benchmark (xView2 test, 933 pre-disaster images, IoU >= 0.5)

| Configuration | Precision | Recall | F1 |
|---|---:|---:|---:|
| Published Feb 2026 (full image, old code) | 0.682 | 0.284 | 0.401 |
| Full image, fixed code, threshold 0.5 | 0.871 | 0.284 | 0.428 |
| Tiled, old pixel stitching | 0.639 | 0.492 | 0.556 |
| Tiled, instance merge, threshold 0.5 | 0.794 | 0.507 | 0.619 |
| **Tiled, instance merge, threshold 0.4 (default)** | **0.737** | **0.565** | **0.640** |
| Same, prompt `house` | 0.708 | 0.593 | 0.645 |
| Full image, SAM 3.1 | 0.869 | 0.281 | 0.425 |

Results: `evaluation/results/sam3_eval/` (default config) and
`evaluation/results/prompt_experiments/`. Predictions:
`/media/data/building_instance_tamu/xview2_sam3_outputs_v2/test/`.

## Output format

```json
{
  "image": {"path": "...", "stem": "...", "width": 1024, "height": 1024, "disaster_type": "pre"},
  "instances": [
    {"id": 1, "uid": "b90f65d8-...", "bbox_xyxy": [429, 83, 495, 134],
     "polygon": [[453, 83], [453, 84], ...], "area_px": 2128.0, "confidence": 0.8094}
  ],
  "timing": {"inference_sec": 7.1, "postprocess_sec": 0.4, "total_sec": 7.5},
  "summary": {"num_instances": 3, "num_windows": 9, "num_raw_detections": 7, "status": "ok"}
}
```

## Package layout

```
config.py          PipelineConfig dataclass (every tuneable parameter)
model.py           SAM3Model: loads SamGeo3 once, predict(image) -> [(mask, score)]
tiling.py          tile_windows(), merge_instances(), paint_labels()
pipeline.py        run_pipeline(): per-image detect -> merge -> vectorize -> save
mask_to_polygon.py labels_to_instances(): geoai.orthogonalize() per label
utils.py           discover_images(), infer_disaster_type(), log()
__main__.py        CLI -> PipelineConfig -> run_pipeline()
```

## Notes

- SamGeo3 `generate_masks()` returns None; results are on `.masks` / `.scores`.
- Meta's SAM3 crashes on `cuda:1` unless the current CUDA device is set;
  `SAM3Model.load()` does this. `CUDA_VISIBLE_DEVICES=1 --device cuda` also works.
- Tiled inference costs about 9x a full-image pass (about 8 s per 1024 px image on an A6000).
