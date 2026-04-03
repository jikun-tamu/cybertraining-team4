# Phase 0 — Baseline Freeze Record

**Date**: 2026-04-02

---

## Canonical Entry Points

### 1. Single-pair inference
**Script**: `pipeline/scripts/run_instance_impact_driver.py`
**Wrapper**: `pipeline/run_pipeline.sh` (thin bash wrapper that calls the above)
**Working directory**: Must be `pipeline/` (relative paths depend on it)
**Example**:
```bash
cd pipeline && python scripts/run_instance_impact_driver.py \
  --pre_image /path/to/pre.png --post_image /path/to/post.png \
  --run_id demo --out_root outputs/driver_runs
```

### 2. LA fire earliest-post workflow
**Script**: `pipeline/run_la_fire_batch.py`
**Working directory**: `pipeline/` (enforced via `cwd=str(PKG_ROOT)` in subprocess)
**Calls**: `bash run_pipeline.sh` → `run_instance_impact_driver.py`
**Hardcoded**: `LA_FIRE_ROOT = Path("/media/data/la_fire_2025")`

### 3. LA fire multidate workflow
**Script**: `pipeline/scripts/run_full_pipeline_launcher.py` (batch parallel)
**Per-cell**: `pipeline/scripts/run_multidate_experiment.py`
**Working directory**: Uses `PKG_ROOT = Path(__file__).resolve().parents[1]`

---

## Critical Import Chain

```
infer_stage2_ensemble.py
  └── from train_stage2 import (
        SiameseDamageModel,    # Class defined at line 506
        collate_batch,         # Function at line 437
        coral_probs_from_logits,  # Function at line 491
        load_mask_tensor,      # Function at line 280
        load_rgb_tensor,       # Function at line 273
        macro_f1_from_cm,      # Function at line 595
        confusion_matrix,      # Function at line 588
        qwk_from_cm,           # Function at line 607
        read_rows,             # Function at line 245
      )
```

**Resolution mechanism**: `train_stage2` is resolved as a **sibling file** in the same `pipeline/scripts/` directory. When `run_instance_impact_driver.py` or `run_multidate_experiment.py` call this script via `subprocess`, the `cwd` is set to `pipeline/` and the script is invoked as `scripts/infer_stage2_ensemble.py`, so Python adds `scripts/` to `sys.path` and finds `train_stage2.py` there.

**CRITICAL**: The `train_stage2.py` DUPLICATES model classes (not imported from `src/.../2-stage package/`). The active versions differ from the modular `src/` versions.

---

## Active Model Code (in train_stage2.py)

| Symbol | Line | Differs from src/ |
|--------|------|-------------------|
| `SiameseDamageModel` | 506-585 | YES: has `change_fusion`, `pooling_mode`, `diff_abs_scale`; uses `ReLU` |
| `CoralHead` | 466-478 | YES: uses `ReLU`, different `__init__` signature (`hidden_dim` not `hidden`) |
| `coral_targets` | 481-488 | YES: adds `label_smoothing` parameter |
| `coral_probs_from_logits` | 491-503 | YES: adds `clamp_min` and re-normalization |
| `downsample_mask` | 450-451 | Same behavior, standalone function |
| `masked_avg_pool` | 454-459 | Same behavior, standalone function |

---

## Expected Output Schema (infer_stage2_ensemble.py)

Per-sample JSONL record:
```json
{
  "bldg_uid": "...",
  "tile_id": "...",
  "event_id": "...",
  "y_pred_ensemble": 0,
  "ensemble_probs": [0.85, 0.10, 0.03, 0.02],
  "ensemble_logits_cum": [...],
  "per_model": [
    {"logits_cum": [...], "probs_cal": [...]}
  ],
  "pmax": 0.85,
  "margin": 0.75,
  "entropy": 0.45,
  "var_predicted_class_prob_weighted": ...,
  "var_expected_severity_weighted": ...
}
```

---

## Key Intermediate Files

| File | Producer | Consumer |
|------|----------|----------|
| `stage1/labels/*_prediction.json` | Stage 1 (SAM3) | `generate_shared_instance_subimages.py` |
| `shared_base/shared_instance_samples.csv` | `generate_shared_instance_subimages.py` | Stage 2a, Stage 2b |
| `shared_base/crops_pre/*.png` | `generate_shared_instance_subimages.py` | Stage 2a, Stage 2b |
| `shared_base/masks_M/*.png` | `generate_shared_instance_subimages.py` | Stage 2a, Stage 2b |
| `shared_base/masks_R/*.png` | `generate_shared_instance_subimages.py` | Stage 2b only |
| `stage2a_predictions.csv` | `infer_stage2a.py` | `present_instance_results.py` |
| `stage2b_ensemble.jsonl` | `infer_stage2_ensemble.py` | `aggregate_multidate_predictions.py` |
| `aggregated_predictions.csv` | `aggregate_multidate_predictions.py` | `build_combined_dataset.py` |

---

## Relative Path Dependencies (from pipeline/ root)

```
stage1/SAM3_Final_20260226/scripts/run_sam3_building_infer.py
scripts/generate_shared_instance_subimages.py
scripts/build_stage2a_infer_csv.py
scripts/infer_stage2a.py
scripts/infer_stage2_ensemble.py
scripts/generate_post_crops_for_date.py
scripts/aggregate_multidate_predictions.py
models/stage2a/stage2a_best_model.pt
models/stage2b/inference0.7273.pt
models/stage2b/inference0.7066_seed9999.pt
models/stage2b/inference0.7034_seed7777.pt
configs/stage2b/run019_seed2025_train_config.json
configs/stage2b/seed9999_train_config.json
configs/stage2b/seed7777_train_config.json
calibration/calibration_run019_r48/
calibration/calibration_seed9999_r48/
calibration/calibration_seed7777_r48/
```
