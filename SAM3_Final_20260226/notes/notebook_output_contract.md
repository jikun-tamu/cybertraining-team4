# SAM3 Notebook Output Contract (Source of Truth)

This document reverse‑engineers outputs and formats from the two source‑of‑truth notebooks:

- `SAM3/251210_sam3_batch_segmentation_Xihan.ipynb`
- `SAM3/251210_sam3_image_segmentation_xihan.ipynb`

It lists all outputs written, naming patterns, formats, and key rules we must preserve.

## Notebook 1: 251210_sam3_batch_segmentation_Xihan.ipynb (Batch)

### External dependencies / folders
- Input images: `/media/data/building_instance_tamu/test/images` (PNG)
  - Pattern: `*_pre_disaster.png`
- Output root: `/media/data/building_instance_tamu/sam3/test`
  - Subdirs created: `masks/`, `annotations/`, `labels/`

### Outputs

**Masks (per image)**
- Writer: `sam3.save_masks_batch(...)`
- Directory: `{output_dir}/masks/`
- Naming: `prefix=original_filename` (image basename)
- Format: GeoTIFF (`.tif`)
- Unique instance labeling: `unique=True`
- Coordinate system: pixel‑space (PNG input; no CRS handling in notebook)
- Size: expected to match the original PNG dimensions (no cropping shown)

**Scores (per image)**
- Not explicitly saved in this notebook’s main batch loop.
- However, vectorization expects `_scores.tif` next to each mask.
  - Path: `{mask_stem}_scores.tif`
  - If missing, vectorization skips the mask.

**Annotations (per image)**
- Writer: `sam3.show_anns_batch(output_dir=..., prefix=f"{original_filename}_ann", dpi=150)`
- Directory: `{output_dir}/annotations/`
- Naming: `{original_filename}_ann*.png` (implementation may add suffix)
- Resolution: same as input image; no tiling in notebook

**Vector outputs (WKT JSON)**
- Function: `vectorize_sam3_masks(...)`
- Inputs: mask `.tif` + score `.tif` in `{output_dir}/masks/`
- Writes temp GeoJSON: `{output_dir}/labels/temp_{mask_stem}.geojson` (deleted after)
- Output JSON: `{output_dir}/labels/{mask_stem}_prediction.json`
- Format:
  - `features.xy`: list of objects, each with
    - `properties`: `feature_type`, `uid`, `label`, `prob`
    - `wkt`: polygon WKT
  - `features.lng_lat`: empty list
  - `metadata`: `original_width`, `original_height`, `width`, `height`, `img_name`
- Geometry regularization: `geoai.orthogonalize(...)` with `epsilon=2`
- Coordinate system: pixel‑space

**Evaluation outputs**
- None in this notebook.

### Key rules inferred
- Per‑image outputs only (no per‑tile artifacts).
- Mask, scores, and annotations are full‑size, aligned to the original image.
- Vector JSON format is WKT‑based with the specific metadata structure above.

---

## Notebook 2: 251210_sam3_image_segmentation_xihan.ipynb (Single‑Image Loop)

### External dependencies / folders
- Input images: `/media/data/building_instance_tamu/test/images` (PNG)
  - Pattern: `*_pre_disaster.png`
- Output root: `/media/data/building_instance_tamu/sam3/test`
  - Subdirs created: `masks/`, `annotations/`, `labels/`

### Outputs

**Masks (per image)**
- Writer: `sam3.save_masks(output=..., unique=True)`
- Directory: `{output_dir}/masks/`
- Naming: `{original_filename}.tif`
- Format: GeoTIFF
- Unique instance labeling: `unique=True`
- Coordinate system: pixel‑space
- Size: matches original PNG dimensions

**Scores (per image)**
- Writer: `sam3.save_masks(..., save_scores=str(score_output), unique=True)`
- Directory: `{output_dir}/masks/`
- Naming: `{original_filename}_scores.tif`
- Format: GeoTIFF with float scores

**Annotations (per image)**
- Writer: `sam3.show_anns(output=str(ann_output), dpi=150, font_size=8)`
- Directory: `{output_dir}/annotations/`
- Naming: `{original_filename}_ann.png`
- Resolution: same as input image (no tiling in notebook)

**Vector outputs (WKT JSON)**
- Same `vectorize_sam3_masks(...)` as notebook 1.
- Output JSON: `{output_dir}/labels/{mask_stem}_prediction.json` with WKT + metadata.
- Requires `_scores.tif` to exist.
- Regularization: `geoai.orthogonalize(...)` with `epsilon=2`.

**Evaluation outputs**
- Visualization for a random sample:
  - Reads predictions from `{output_dir}/labels/*_prediction.json`
  - Reads GT from `/media/data/building_instance_tamu/test/labels/{base}.json`
  - Outputs inline plots only (no files written in this cell)
- Summary evaluation:
  - Writes CSV: `/media/data/building_instance_tamu/sam3/test/iou_results.csv`
  - Columns: `image`, `gt_count`, `pred_count`, `overall_iou`, `mean_poly_iou`

### Key rules inferred
- Per‑image outputs only (no per‑tile artifacts).
- Mask + scores + annotations are full‑size, aligned to original PNG.
- Vector JSON format is WKT‑based with the specific metadata structure above.

---

## Contract Comparison (Notebook 1 vs 2)

**Common contract**
- Input images are PNGs with no CRS/transform; all outputs are pixel‑space.
- Output root contains: `masks/`, `annotations/`, `labels/`.
- Mask naming: `{image_id}.tif` (full size, unique instance labels).
- Score naming: `{image_id}_scores.tif` (required for vectorization).
- Annotation naming: `{image_id}_ann.png` (full size).
- Vector output: `{image_id}_prediction.json` in labels/ using WKT JSON format with specific metadata.
- No tiling artifacts are written in notebooks.

**Differences**
- Notebook 1 uses batch API and doesn’t explicitly save scores in the main loop, but vectorization expects them.
- Notebook 2 explicitly saves scores.
- Notebook 2 generates IoU CSV; notebook 1 does not.

---

## Output Contract We Will Preserve (Pipeline Target)

- **Notebook output style (default)**:
  - Per‑image outputs only: no per‑tile artifacts.
  - `masks/{image_id}.tif` (unique labels, full size)
  - `masks/{image_id}_scores.tif` (full size)
  - `annotations/{image_id}_ann.png` (full size)
  - `labels/{image_id}_prediction.json` (WKT JSON format + metadata)
  - Pixel‑space coordinates unless georef exists.

- **Tiled output style (optional)**:
  - May emit per‑tile masks/annotations for debugging or performance.

Any deviations will be documented in the pipeline report and/or CLI help.
