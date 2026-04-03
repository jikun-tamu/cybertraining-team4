# Structure Cleanup & Reorganization Plan

**Date**: 2026-04-01
**Principle**: Conservative, reversible changes. No blind deletion.

---

## Current Problems

1. **Duplicated code**: Stage 2 model code exists in both `src/cybertraining_team4/2-stage package/` and `pipeline/scripts/`. The `pipeline/` copy is the production version but imports model code from the `2-stage package` via path manipulation.

2. **Misleading directory names**: `src/cybertraining_team4/` suggests a clean Python package but actually contains a messy mix of scripts and an archived zip-delivered collaborator package.

3. **Scattered entry points**: Multiple scripts do similar things (`run_la_fire_batch.py`, `run_instance_impact_driver.py`, `run_full_pipeline_launcher.py`, `run_multidate_experiment.py`) with unclear which is authoritative.

4. **Large binary in repo**: `II_package.zip` (1 GB) at project root is the original collaborator delivery, now unpacked into `pipeline/`.

5. **Exploration sprawl**: `exploration/` contains 4.7 GB of archived experiments with model checkpoints that are no longer needed for production.

6. **No clear docs directory**: Documentation scattered across READMEs, markdown files in results/, and CLAUDE.md.

7. **Redundant copies**: `pipeline/stage1/SAM3_Final_20260226/` is a copy of `exploration/SAM3_Final/` — both are superseded by `stage1/`.

---

## Recommended Conceptual Structure

```
250812_tamu_cybertraining_team4/
│
├── stage1/                         [KEEP AS-IS] Production Stage 1
│   └── sam3_building_identifier/   SAM3 package (current, clean)
│
├── stage2/                         [NEW — extract from pipeline/]
│   ├── models/                     Model architecture source code
│   │   ├── siamese_stage2.py
│   │   ├── backbone.py
│   │   ├── coral_head.py
│   │   └── masked_pool.py
│   ├── data/
│   │   └── stage2_dataset.py
│   ├── training/
│   │   ├── train_stage2.py
│   │   ├── build_stage2_index.py
│   │   ├── preprocess_stage2_crops.py
│   │   └── calibrate_*.py
│   └── configs/
│       └── stage2b/*.json
│
├── pipeline/                       [SLIM DOWN] Combined pipeline scripts
│   ├── scripts/                    Runtime scripts only
│   │   ├── run_multidate_experiment.py
│   │   ├── run_full_pipeline_launcher.py
│   │   ├── generate_shared_instance_subimages.py
│   │   ├── generate_post_crops_for_date.py
│   │   ├── infer_stage2a.py
│   │   ├── infer_stage2_ensemble.py
│   │   ├── aggregate_multidate_predictions.py
│   │   ├── build_combined_dataset.py
│   │   ├── generate_maps.py
│   │   ├── quality_filter.py
│   │   └── present_instance_results.py
│   ├── checkpoints/                Trained model weights
│   │   ├── stage2a/
│   │   └── stage2b/
│   └── calibration/                Temperature scaling artifacts
│
├── evaluation/                     [KEEP AS-IS] xView2 benchmark
│
├── docs/                           [NEW] All documentation
│   ├── project_audit/              This audit
│   └── ...
│
├── notebooks/                      [KEEP] Validation notebooks
│
├── archive/                        [NEW — move legacy here]
│   ├── exploration/                All exploration experiments
│   ├── II_package.zip              Original collaborator delivery
│   ├── src_legacy/                 Old src/cybertraining_team4/
│   └── README.md                   What was archived and why
│
├── results/                        [KEEP] Reports and logs
│
├── CLAUDE.md                       [UPDATE after restructure]
├── README.md                       [UPDATE]
└── pyproject.toml                  [KEEP]
```

---

## Legacy Archive Recommendations

### KEEP ACTIVE

| Item | Reason |
|------|--------|
| `stage1/` | Production Stage 1 package |
| `pipeline/scripts/` | Production inference scripts |
| `pipeline/models/` | Trained model checkpoints (required for inference) |
| `pipeline/calibration/` | Calibration artifacts |
| `pipeline/configs/` | Training configs (reproducibility) |
| `evaluation/` | Benchmark evaluation |
| `notebooks/` | Validation notebooks |
| `results/` | Reports and metrics |

### KEEP FOR REPRODUCIBILITY

| Item | Reason | Action |
|------|--------|--------|
| `src/.../2-stage package/scripts/src/models/` | Stage 2 model architecture source | Extract to `stage2/models/` |
| `src/.../2-stage package/scripts/src/data/` | Dataset class | Extract to `stage2/data/` |
| `src/.../2-stage package/scripts/train_stage2.py` | Legacy training script | Keep in archive |
| `src/.../2-stage package/scripts/preprocess_stage2_crops.py` | Preprocessing | Extract to `stage2/training/` |
| `src/.../2-stage package/scripts/build_stage2_index*.py` | Index builders | Extract to `stage2/training/` |
| `src/.../2-stage package/scripts/calibrate_*.py` | Calibration scripts | Extract to `stage2/training/` |
| `pipeline/configs/stage2b/*.json` | Training hyperparameters | Keep active |

### ARCHIVE

| Item | Size | Reason |
|------|------|--------|
| `exploration/Mask_R-CNN/` | ~200 MB | Superseded by SAM3; includes model checkpoints |
| `exploration/PolyWorld/` | ~100 MB | Explored, not adopted |
| `exploration/GeoAI_QuishengWu/` | ~200 MB | Superseded by samgeo |
| `exploration/GeoAI_building_segmentation/` | Unknown | Early exploration |
| `exploration/SAM3_notebooks/` | ~50 MB | Evolved into stage1/ package |
| `exploration/SAM3_Final/` | ~100 MB | Superseded by stage1/; copy exists in pipeline/ |
| `exploration/corrected_model/` | ~170 MB | Early Mask R-CNN checkpoint |
| `pipeline/stage1/SAM3_Final_20260226/` | ~100 MB | Redundant copy of exploration/SAM3_Final/ |
| `src/cybertraining_team4/stage1_train.py` | Small | Legacy script |
| `src/cybertraining_team4/train_xihan_gpu_testing.py` | Small | GPU testing script |
| `src/cybertraining_team4/2-stage package/2-stage package.zip` | Large | Nested zip archive |
| `.ipynb_checkpoints/` | Small | Jupyter auto-saves |

### SAFE TO DELETE ONLY AFTER CONFIRMATION

| Item | Size | Reason |
|------|------|--------|
| `II_package.zip` (project root) | 1 GB | Original collaborator delivery; unpacked into `pipeline/` |
| `data/case_overlays.zip` | 734 MB | Possibly duplicated in `la_fire_2025/grids/case_overlays/` |
| Model checkpoints in `exploration/` | ~1.5 GB | `Mask_R-CNN/buildings_instance/instance_models/`, `GeoAI_QuishengWu/models/`, `corrected_model/` |
| `/media/data/building_instance_tamu/Mask_R-CNN_BuildingInstance_Train/models/` | 1.4 GB | Mask R-CNN checkpoints no longer needed |
| `/media/data/building_instance_tamu/xview2_challenge_*.tar.gz` | 10.5 GB | Raw archives if data already extracted |

---

## Entry-Point Cleanup

### Current Confusion

There are at least 5 ways to run the pipeline:

1. `pipeline/run_pipeline.sh` — bash wrapper → `run_instance_impact_driver.py` (single pair)
2. `pipeline/run_la_fire_batch.py` — LA fire batch runner (single post-date per cell)
3. `pipeline/scripts/run_instance_impact_driver.py` — single pre/post pair end-to-end
4. `pipeline/scripts/run_full_pipeline_launcher.py` — parallel multi-date across all cells
5. `pipeline/scripts/run_multidate_experiment.py` — multi-date per single cell

### Recommended Official Entry Points

| Purpose | Script | Notes |
|---------|--------|-------|
| **Stage 1 inference** | `python -m sam3_building_identifier` | Already clean |
| **Stage 1 smoke test** | `stage1/tests/smoke_test.py` | Already clean |
| **Single pair (demo)** | `pipeline/scripts/run_instance_impact_driver.py` | Keep as canonical single-pair runner |
| **Multi-date single cell** | `pipeline/scripts/run_multidate_experiment.py` | Keep as canonical per-cell runner |
| **Full batch (multi-GPU)** | `pipeline/scripts/run_full_pipeline_launcher.py` | Keep as production batch runner |
| **Stage 2 training** | `pipeline/scripts/train_stage2.py` | Production training script |
| **xView2 evaluation** | `evaluation/evaluate_predictions.py` | Already clean |

### Scripts to Deprecate
| Script | Reason |
|--------|--------|
| `pipeline/run_la_fire_batch.py` | Superseded by `run_full_pipeline_launcher.py` (multi-date) |
| `pipeline/run_pipeline.sh` | Thin wrapper; just use `run_instance_impact_driver.py` directly |
| `src/cybertraining_team4/run_custom_pipeline.py` | Earlier version; logic now in pipeline/ |
| `src/cybertraining_team4/stage1_train.py` | No Stage 1 training needed (SAM3 is zero-shot) |

---

## Documentation Gaps

### Must Document Next

1. **Stage 2 training reproduction guide**: Step-by-step instructions to retrain Stage 2 from scratch on new data (currently no single document explains this end-to-end)

2. **Environment setup guide**: Current `environment.yml` files are incomplete or environment-specific. Need a clean setup guide for both `geoai_sam` and Stage 2 dependencies.

3. **Data dictionary**: Formal schema documentation for:
   - `shared_instance_samples.csv` columns
   - `building_damage_all_cells.csv` columns
   - `stage2b_ensemble.jsonl` fields
   - `aggregated_predictions.csv` columns

4. **Multi-date aggregation explanation**: The four methods (M1, M1b, M2, M3) need a clear decision guide for when to use which.

5. **Known limitations document**: Formal list of known failure modes, edge cases, and methodological caveats (especially wildfire domain gap).

6. **Collaborator handoff protocol**: How to deliver Stage 1 outputs to collaborator and receive Stage 2 updates.

7. **GPU resource guide**: Which scripts need which GPU, memory requirements, batch size constraints.

---

## Migration Steps (If Approved)

### Phase 1: Documentation (No File Moves)
- [x] Create `docs/project_audit/` with this audit (current work)
- [ ] Update `README.md` with clearer project map
- [ ] Update `CLAUDE.md` to reflect current state

### Phase 2: Extract Stage 2 Source (Low Risk)
- [ ] Copy (not move) model source from `src/.../2-stage package/scripts/src/` to `stage2/`
- [ ] Copy training/preprocessing scripts to `stage2/training/`
- [ ] Verify pipeline imports still work with original paths

### Phase 3: Archive Legacy (Medium Risk)
- [ ] Create `archive/` directory
- [ ] Move `exploration/` to `archive/exploration/`
- [ ] Move `src/cybertraining_team4/` to `archive/src_legacy/`
- [ ] Move `II_package.zip` to `archive/`
- [ ] Create `archive/README.md` explaining what was moved and why
- [ ] Update all import paths in pipeline scripts
- [ ] Run smoke test to verify nothing breaks

### Phase 4: Cleanup (Higher Risk — Defer)
- [ ] Remove redundant `pipeline/stage1/SAM3_Final_20260226/` after verifying `stage1/` is used everywhere
- [ ] Consolidate `.ipynb_checkpoints/`
- [ ] Consider deleting large binaries after confirming backups

**Each phase should be a separate git commit with clear message.**
