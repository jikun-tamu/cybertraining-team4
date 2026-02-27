# SAM3 Building Identifier

Batch pipeline that uses **SAM3** (via `samgeo`) to detect buildings in
satellite imagery and output per-instance bounding boxes, polygons, and
confidence scores as JSON.

Built from the source-of-truth notebooks in `../SAM3/`.

---

## What it does

1. Scans an input folder for images (filters to `_pre_disaster` images by
   default; falls back to all images if that naming is absent).
2. Runs SAM3 text-prompted segmentation (`prompt="building"`) on each image.
3. Saves:
   - `masks/`       — instance mask TIF + per-pixel scores TIF
   - `annotations/` — PNG visualization with colored masks and scores
   - `predictions/` — one JSON per image with bboxes, polygons, confidence
   - `run_summary.json` — aggregate stats for the whole run

---

## Requirements

Use the **`geoai_sam`** conda environment (has `samgeo==1.0.1` with SAM3,
CUDA, `geoai`, `rasterio`, `shapely`, `opencv`):

```bash
conda activate geoai_sam
```

Install this package (once, development mode):

```bash
pip install -e /media/gisense/xihan/250812_tamu_cybertraining_team4/SAM3_Claude --no-deps
```

A Hugging Face login is required the first time SAM3 weights are downloaded:

```bash
python -c "from huggingface_hub import login; login()"
```

---

## Quick test (3 images)

```bash
conda run -n geoai_sam python -m sam3_building_identifier \
    --input-dir /media/data/building_instance_tamu/test/images \
    --output-dir /media/data/building_instance_tamu/sam3_claude/test3 \
    --max-images 3
```

Or run the smoke test directly:

```bash
conda run -n geoai_sam python \
    /media/gisense/xihan/250812_tamu_cybertraining_team4/SAM3_Claude/tests/smoke_test.py
```

---

## Run on a full folder

```bash
conda run -n geoai_sam python -m sam3_building_identifier \
    --input-dir /media/data/building_instance_tamu/train/images \
    --output-dir /media/data/building_instance_tamu/sam3_claude/train
```

The pipeline skips images whose `_prediction.json` already exists
(`--no-skip` to override). Re-runs are safe.

### Common options

| Flag | Default | Description |
|------|---------|-------------|
| `--input-dir` | (see config) | Folder of input images |
| `--output-dir` | (see config) | Root output folder |
| `--disaster-type` | `auto` | `pre`, `post`, `all`, or `auto` |
| `--max-images N` | all | Process only the first N images |
| `--device` | auto | `cuda`, `cuda:1`, `cpu` |
| `--min-size` | 100 | Minimum mask area (pixels) |
| `--epsilon` | 2.0 | Polygon simplification tolerance |
| `--no-skip` | — | Re-process already-done images |
| `--dry-run` | — | List images that would be processed, then exit |

Full list: `python -m sam3_building_identifier --help`

---

## Output format

Each image produces a `predictions/<stem>_prediction.json`:

```json
{
  "image": {
    "path": "/abs/path/image.png",
    "stem": "hurricane-florence_00000004_pre_disaster",
    "width": 1024,
    "height": 1024,
    "disaster_type": "pre"
  },
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
  "timing": {
    "inference_sec": 0.44,
    "postprocess_sec": 5.0,
    "total_sec": 5.9
  },
  "summary": {
    "num_instances": 3,
    "status": "ok"
  }
}
```

Images with zero detections produce a valid JSON with `"instances": []`.

---

## Notes

- **GPU**: Automatically uses CUDA if available. With a 47 GB A6000, one
  image at a time is sufficient (`--batch-size 1`, the default).
- **Image sizes**: Any size is accepted; SAM3 handles rescaling internally.
- **Disaster naming**: If the folder uses `_pre_disaster` / `_post_disaster`
  suffixes, `--disaster-type auto` (default) will select pre-disaster images.
  If no such naming exists, all images are processed.
- **Polygon method**: `geoai.orthogonalize()` with `epsilon=2` (matches
  notebooks). Falls back to OpenCV contours if geoai is unavailable.
