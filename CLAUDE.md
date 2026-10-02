# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> **PROJECT CONCLUDED (2026-07-10), Stage 1 revisited (2026-09)** — read `PROJECT_CONCLUSION.md`
> first, then `reports/sam3_audit_2026-09.md` for the Stage 1 rewrite, the new benchmark
> and the LA fire re-run. The commands below are valid for re-running the pipeline.

## Project Overview

Disaster impact assessment pipeline using satellite imagery, building damage prediction, and demographic data.

**Active deliverables:**
- **`stage1/`** — batch building-detection package using SAM3 via samgeo (`sam3_building_identifier`, v0.2)
- **`pipeline/`** — combined Stage 1 + Stage 2 pipeline (formerly `II_package/`), collaborator-integrated
- **`evaluation/`** — xView2 benchmark scripts and results

**Exploration only** (not production): `archive/` contains Mask R-CNN, PolyWorld, GeoAI_QuishengWu, and earlier SAM3 variants (`SAM3_Final`, notebooks)

## Environments

**Use `geoai_sam`** (samgeo 1.0.1, SAM 3) for all pipeline work.
`geoai_sam31` (samgeo 1.4.2, sam3 0.1.4) is only needed for `--model facebook/sam3.1`
(gated on Hugging Face; access granted to account `xyaoaf`). SAM 3.1 gave no gain on xView2.

```bash
conda activate geoai_sam          # interactive
conda run -n geoai_sam <command>  # non-interactive (conda is not on PATH in plain ssh:
                                  # export PATH=/media/gisense/xihan/miniconda3/bin:$PATH)
```

Edit files over ssh, not through the Finder/rclone mount: macOS writes `._*` AppleDouble
files into the repo (and `.git/`) when saving through the mount.

## stage1 — SAM3 Building Detection Package

**Install** (editable, no deps): `pip install -e stage1 --no-deps`, or `PYTHONPATH=stage1`
**One-time HF login** (caches token): `python -c "from huggingface_hub import login; login()"`

```bash
# Dry-run (list images, no inference)
python -m sam3_building_identifier --input-dir <dir> --output-dir <dir> --dry-run

# Process N images
python -m sam3_building_identifier \
    --input-dir /media/data/building_instance_tamu/test/images \
    --output-dir /tmp/sam3_out --max-images 10

# Predictions only (no mask TIFs / overlay PNGs), second GPU
python -m sam3_building_identifier ... --no-masks --no-annotations --device cuda:1
```

### Tests

```bash
conda run -n geoai_sam python stage1/tests/test_tiling.py   # merge logic, no GPU
conda run -n geoai_sam python stage1/tests/smoke_test.py    # 3 images end to end
```

### Key default parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `--prompt` | `"building"` | `house` has equal F1 on xView2 but lower wildfire recall |
| `--confidence-threshold` | `0.4` | SamGeo3 default is 0.5; 0.4 gave best tiled F1 |
| `--model` | `facebook/sam3` | `facebook/sam3.1` needs `geoai_sam31` |
| `--tile-size` | `512` | 0 runs on the full image (recall drops from 0.565 to ~0.34) |
| `--overlap` | `64` | Minimum window overlap; windows are evenly spaced, no slivers |
| `--merge-iou` | `0.5` | Cross-window merge: IoU inside the shared window overlap |
| `--min-size` | `100` | Min mask area in pixels |
| `--min-polygon-area` | `100.0` | Drops tiny polygons (the Feb 2026 benchmark had 4,412 of <10 px) |
| `--epsilon` | `2.0` | Douglas-Peucker tolerance for `geoai.orthogonalize()` |
| `--disaster-type` | `auto` | Filters images by xView2 `_pre_disaster` / `_post_disaster` suffix |

## Package Architecture (`sam3_building_identifier/`)

```
config.py          — PipelineConfig dataclass (all tuneable params)
model.py           — SAM3Model: loads SamGeo3 once, predict(rgb array) -> [(mask, score)]
tiling.py          — tile_windows(), merge_instances() (cross-window), paint_labels()
pipeline.py        — run_pipeline(): detect -> merge -> vectorize -> save, run_summary.json
mask_to_polygon.py — labels_to_instances(): geoai.orthogonalize() per label value
utils.py           — discover_images(), infer_disaster_type(), log()
__main__.py        — argparse CLI → PipelineConfig → run_pipeline()
```

**SamGeo3 API behavior**:
- `generate_masks()` returns **None** — results are stored in `.masks`, `.boxes`, `.scores`
- Meta's SAM3 crashes on `cuda:1` unless the current CUDA device is set; `SAM3Model.load()`
  calls `torch.cuda.set_device()`. `CUDA_VISIBLE_DEVICES=1 ... --device cuda` also works.
- `rasterio.features.shapes` (used by orthogonalize) rejects uint32; labels are int32.

## Output Schema

Per-image: `predictions/<stem>_prediction.json`; run aggregate: `run_summary.json`
(full config + `git describe` code version).

```json
{
  "image": {"path", "stem", "width", "height", "disaster_type"},
  "instances": [{"id", "uid", "bbox_xyxy":[x1,y1,x2,y2], "polygon":[[x,y],...], "area_px", "confidence"}],
  "timing": {"inference_sec", "postprocess_sec", "total_sec"},
  "summary": {"num_instances", "num_windows", "num_raw_detections", "status"}
}
```

Other outputs per image (when detections > 0): `masks/<stem>.tif` (int32 labels),
`masks/<stem>_scores.tif`, `annotations/<stem>_ann.png`. Masks keep the input's CRS/transform.

## Benchmark and Data

- **xView2 test**: 1,866 images at `/media/data/building_instance_tamu/test/images/` (933 pre used), labels in `test/labels/`
- **Current Stage 1 predictions**: `/media/data/building_instance_tamu/xview2_sam3_outputs_v2/test/`
  (P 0.737 / R 0.565 / F1 0.640); prompt runs in `sam3_prompt_experiments_v2/`. The pre-fix
  Feb/Mar 2026 outputs (`xview2_sam3_outputs/`, `sam3_prompt_experiments/`) were deleted 2026-10-02.
- **Experiments**: `/media/data/building_instance_tamu/tiling_experiments/` (runs A–L, see audit report)
- **LA fire**: `la_fire_2025/stage2_damage/multidate_full_run_v2/` (re-run with stage1 v0.2);
  the April 2026 run (`multidate_full_run/`) was deleted 2026-10-02; its combined product is in
  git history (`results/final_product/` at commit `0dc7630`).

```bash
python evaluation/evaluate_predictions.py                        # default: test split, v2 outputs
python evaluation/evaluate_predictions.py --pred-dir <dir>/predictions --name run
python evaluation/run_prompt_experiments.py --eval-only          # prompts, same config for all
```

## Other Components

| Directory | Purpose |
|-----------|---------|
| `pipeline/` | Combined Stage 1+2 pipeline; `scripts/run_multidate_experiment.py` runs LA fire (Stage 1 via `stage1/`) |
| `evaluation/` | `evaluate_predictions.py` (single evaluation implementation) + `run_prompt_experiments.py` |
| `results/` | LA fire figures, final product (`results/final_product/`), prompt overlays |
| `reports/` | M2b validation, I-GUIDE audit, SAM3 audit (2026-09) |
| `src/cybertraining_team4/` | Early training code + collaborator's original Stage 2 handoff |
| `notebooks/` | EDA and validation notebooks |
| `archive/` | Experimental/comparison work (not production) |

## GPU Notes

- 2× NVIDIA RTX A6000 (47.5 GB each), shared with other lab members — check `nvidia-smi` first
- Tiled inference: ~8 s per 1024 px image; full LA run (295 cells, Stage 1+2) ~6 h on one GPU
