# LEGACY — src/cybertraining_team4/

**Status**: PARTIALLY LEGACY
**Date marked**: 2026-04-02

## Active files (still used)
- `process_chips_600m.py` — Creates 600m chips from raw Maxar imagery for LA fire workflow
- `prune_bad_pre_cells.py` — Removes blank pre-disaster cells

## Legacy files (superseded)
- `stage1_train.py` — Stage 1 training script (SAM3 requires no training)
- `train_xihan_gpu_testing.py` — GPU testing script (one-off)
- `run_custom_pipeline.py` — Earlier pipeline runner, replaced by `pipeline/scripts/`

## Collaborator delivery (important for reproducibility)
- `2-stage package/` — Original Stage 2 code from collaborator
  - `scripts/src/models/` — **Original modular model definitions** (but NOTE: the active production
    versions are the **duplicates** inside `pipeline/scripts/train_stage2.py`, which have diverged)
  - `scripts/src/data/` — Original dataset class
  - `scripts/train_stage2.py` — Legacy training script (270 lines, simpler than production version)
  - `scripts/calibrate_*.py` — Calibration scripts (still referenced for understanding calibration)

**CRITICAL**: `pipeline/scripts/train_stage2.py` does NOT import from `2-stage package/scripts/src/`.
It contains **duplicate, diverged implementations** of the model classes. The production-active
model code is the version in `pipeline/scripts/train_stage2.py`.
