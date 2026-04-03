# Read This First — Project Orientation

**Date**: 2026-04-01

---

## What Is This Project?

A two-stage pipeline for **building-level disaster damage assessment** from satellite imagery:

- **Stage 1**: Detect buildings using SAM3 (zero-shot, no training needed)
- **Stage 2**: Compare pre/post-disaster images to classify damage per building (Siamese ConvNeXt + CORAL ordinal regression)

Trained on **xView2** (multi-disaster benchmark). Applied to **2025 LA wildfires** (real-world case study).

---

## Quick Map of What Matters

| What you want to do | Where to look |
|---------------------|---------------|
| Run Stage 1 (building detection) | `stage1/` — `python -m sam3_building_identifier` |
| Understand Stage 2 architecture | `docs/project_audit/stage2_technical_reverse_engineering.md` |
| Run combined pipeline (single pair) | `pipeline/scripts/run_instance_impact_driver.py` |
| Run full multi-date batch | `pipeline/scripts/run_full_pipeline_launcher.py` |
| Train Stage 2 from scratch | `pipeline/scripts/train_stage2.py` + configs in `pipeline/configs/stage2b/` |
| Stage 2 model source code | `src/cybertraining_team4/2-stage package/scripts/src/models/` |
| Trained model checkpoints | `pipeline/models/stage2a/` and `pipeline/models/stage2b/` |
| xView2 evaluation | `evaluation/evaluate_predictions.py` |
| LA fire results | `/media/data/la_fire_2025/final_maps/` |
| All exploration/legacy work | `exploration/` (archive candidate) |

---

## Environment

```bash
conda activate geoai_sam    # For ALL SAM3 and pipeline work
# Do NOT use geoai_sam3 — it has an older samgeo without SamGeo3
```

GPUs: 2x NVIDIA RTX A6000 (47.5 GB each) — use `--device cuda:0` or `cuda:1`

---

## Key Numbers

| Metric | Value |
|--------|-------|
| Stage 1 F1 on xView2 | 0.40 (precision 0.68, recall 0.28) |
| Stage 1 mean IoU | 0.76 (good shape quality, low detection completeness) |
| Stage 2b best single model | ~70.7% macro F1 |
| LA fire buildings assessed | 10,607 across 120 grid cells |
| LA fire temporal instability | 53% of buildings have conflicting labels across dates |

---

## Critical Caveats

1. **Stage 2 has never seen wildfire damage** — trained on earthquake/flood/hurricane only
2. **53% temporal instability** in LA fire results — predictions change across post-disaster dates
3. **Stage 1 misses dense urban areas** — 30% of xView2 images produce zero predictions
4. **M1b is the correct aggregation method** — M1 produces false "destroyed" from nodata artifacts

---

## Audit Documents (This Directory)

| Document | What it covers |
|----------|---------------|
| `project_overview.md` | Full pipeline explanation, data flow, results |
| `stage1_sam3_summary.md` | Stage 1 technical details, parameters, legacy branches |
| `stage2_technical_reverse_engineering.md` | Complete Stage 2 reverse engineering (architecture, training, innovation analysis) |
| `la_fire_realworld_pipeline.md` | Real-world workflow differences, preprocessing, outputs |
| `structure_cleanup_plan.md` | Proposed reorganization with archive recommendations |
| `directory_inventory.md` | File-level inventory of all directories |

---

## What to Read Next

- If you're **continuing development**: Start with `project_overview.md`, then `stage2_technical_reverse_engineering.md`
- If you're **running the pipeline**: See `pipeline/README.md` and the entry-point table in `structure_cleanup_plan.md`
- If you're **cleaning up the repo**: See `structure_cleanup_plan.md`
- If you're **evaluating results**: See `results/project_evaluation_report.md`
