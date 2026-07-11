# Project Conclusion

**Status: CONCLUDED — no active development.**
**Date concluded**: 2026-07-10
**Location**: moved to `/media/gisense/xihan/archive/250812_tamu_cybertraining_team4`
on 2026-07-10. Active docs use the new path; archived notebooks and old reports may
still reference the pre-move location.

This document is the single entry point for anyone returning to this project. It records
what was built, what the final results were, why we stopped, where every asset lives, and
what the one worthwhile continuation path would be.

---

## 1. What this project is

A two-stage disaster impact assessment pipeline built by CyberTraining Team 4:

- **Stage 1** — zero-shot building detection with SAM3 (text prompt, no training)
- **Stage 2a** — building type + population estimation (EfficientNet-B0, multi-task)
- **Stage 2b** — per-building damage classification (Siamese ConvNeXt ensemble, CORAL
  ordinal head, mask + ring pooling, trained on flood-only xBD)
- **Aggregation** — multi-date majority vote (M2b) with image-quality filtering

The pipeline was deployed end-to-end on the January 2025 Los Angeles wildfires
(Maxar Open Data) and submitted to the **I-GUIDE Spatial AI Challenge 2025-26**
(final submission 2026-04-15, via the collaborator repo — see §5).

## 2. Final results (verified 2026-07-10)

### Stage 1 — SAM3 zero-shot detection, xView2 test benchmark (933 images)

| Prompt | Precision | Recall | F1 | Mean IoU (matched) |
|---|---:|---:|---:|---:|
| "building" (default) | 0.682 | 0.284 | 0.401 | 0.759 |
| "house" | 0.698 | 0.318 | 0.437 | — |

Source: `evaluation/results/sam3_eval/eval_test.json`, `evaluation/results/prompt_experiments/`.
Detected 22,824 buildings of 54,862 ground truth. **Recall ~28–32% is the pipeline's
hard ceiling** — roughly 70% of buildings never enter Stage 2.

### Stage 2b — flood-only xBD validation (collaborator's work)

Best single model (run019): **macro-F1 0.727, QWK 0.826** on flood validation split.
Deployed as a 3-checkpoint ensemble (weights 4:3:2) with temperature calibration and
per-building uncertainty metrics. Full training logbook with all sweeps and ablations:
`pipeline/docs/stage2b.md`.

### LA Fire 2025 deployment (no ground-truth validation performed)

295 cells attempted, 120 cells with detected buildings, **21,797 building instances**.
M2b damage distribution: 16,246 no_damage / 4,174 minor / 39 major / 93 destroyed /
1,245 unknown (no valid-coverage dates).

Final product:
`/media/data/building_instance_tamu/la_fire_2025/stage2_damage/multidate_full_run/building_damage_all_cells.{csv,geojson,gpkg}`
— CSV + GPKG copies are also committed in-repo at `results/final_product/` (2026-07-10),
so the deliverable survives independently of the data disk.

**These predictions were never validated against ground truth.** That is the main
unfinished scientific work (see §4).

## 3. Why the project concluded

Assessed honestly in July 2026 (full analysis was done by comparing against the BRIGHT
Challenge project, `/media/gisense/xihan/archive/260515_BRIGHT_BuildingDamage_Challenge`):

1. **Separated stages cap accuracy.** Stage 1 is untrained (zero-shot) with a ~30% recall
   ceiling. Stage 2b was trained on ground-truth xBD footprints but receives SAM3's
   imperfect polygons at deployment — a mask-distribution shift its mask-anchored pooling
   is directly sensitive to. The BRIGHT project demonstrated the same failure mode
   empirically: a SAM→classifier two-stage design scored 0.09 mAP (46% recall ceiling)
   while a jointly trained end-to-end Mask R-CNN scored 0.244.
2. **Domain shift was never measured.** A flood-trained model was applied to a fire event
   on different imagery with no fine-tuning and no ground-truth check.
3. **The methods contribution is not competitive.** xBD damage classification is a
   saturated literature; the field has moved to end-to-end instance-level models.
4. **Team disengaged.** Remaining publication value did not justify the opportunity cost.

What *was* good: Stage 2b's training methodology (CORAL + ring pooling + calibration +
uncertainty) is careful, reproducible work; the multi-date quality-filtered aggregation
solved a real operational failure (a blank early post-date caused "destroyed" predictions
for every building in a cell — see `pipeline/docs/multidate_experiment_report.md`).

## 4. If you pick this up: the one worthwhile path

**Validate the LA fire predictions against CAL FIRE DINS data.** CAL FIRE published
per-structure Damage Inspection (DINS) data for the Palisades and Eaton fires (public
GeoJSON/CSV, tens of thousands of inspected structures with damage categories).
Nothing in this repo uses it. Concrete steps:

1. Download DINS data for the January 2025 LA fires (CAL FIRE / LA County GIS portals).
2. Spatial-join DINS points against `building_damage_all_cells.geojson` (WGS84).
3. Map DINS categories (No damage / Affected / Minor / Major / Destroyed) to the 4-class
   scheme; compute confusion matrix, per-class F1, and detection recall vs the DINS
   structure inventory (also compare Stage 1 recall against Microsoft/OSM footprints).
4. Frame as a real-event evaluation of a zero-shot + transfer-learning rapid damage
   assessment pipeline. Candidate venues: *Int. J. Applied Earth Observation and
   Geoinformation*, *Natural Hazards*, *IJDRR*.

Estimated effort: 2–4 weeks analysis + writing on top of the completed run. Expect the
numbers to be mediocre (see §3) — the paper would be an honest evaluation/case study, not
a methods paper. Smaller alternatives: a short workshop paper on zero-shot SAM3 building
detection (all benchmark numbers already exist in `evaluation/results/`), or the
multi-prompt ensemble experiment sketched in `results/prompt_experiment/` (+2.2%
detections over best single prompt).

## 5. Asset inventory

### This repo (`/media/gisense/xihan/archive/250812_tamu_cybertraining_team4`)

| Path | What | In git? |
|---|---|---|
| `stage1/` | SAM3 detection package (`sam3_building_identifier`) | yes |
| `pipeline/` | Combined Stage 1+2 pipeline, scripts, docs, configs | yes (code/docs) |
| `pipeline/models/stage2b/` | 3 ensemble checkpoints (3×372 MB) | **no — local only** |
| `pipeline/models/stage2a/` | Stage 2a checkpoint (19 MB) | **no — local only** |
| `pipeline/calibration/` | Per-checkpoint calibration artifacts | yes |
| `evaluation/` | xView2 benchmark script + results | yes |
| `reports/` | M2b validation + I-GUIDE competition audit | yes |
| `results/` | LA fire figures, prompt experiment | yes (small files) |
| `src/cybertraining_team4/` | Early training code + collaborator's original Stage 2 handoff (`2-stage package/`, incl. 2 checkpoints, ~470 MB) | code yes, checkpoints **local only** |
| `archive/` | Exploration (Mask R-CNN, PolyWorld, earlier SAM3 variants) | partially |
| `II_package/` | Local clone of collaborator repo, commit `e83271f`, **has uncommitted notebook edits** | no (standalone git repo) |
| `archive/II_package_20260415_github_clone/` | Clone of the final I-GUIDE submission, commit `abf301f` | no (standalone git repo) |

> **Checkpoint backup — verified 2026-07-10**: all 4 Stage 2 checkpoints (stage2a +
> 3× stage2b ensemble) are tracked via Git LFS in the collaborator's GitHub repo
> (`jikun-tamu/Instance-Impact-IGUIDE-SpatialAIChallenge2026`, under `II_package/models/`),
> confirmed with `git lfs ls-files` in the local clone. Local copies in
> `pipeline/models/` are gitignored convenience copies.

### External / data (not in git)

| Location | What |
|---|---|
| `/media/data/building_instance_tamu/la_fire_2025/` (66 GB) | LA fire chips, grids, manifest, full-run outputs, QC overlays, maps |
| `/media/data/building_instance_tamu/test/images/` | xView2 test set (1,866 images) |
| `github.com/jikun-tamu/Instance-Impact-IGUIDE-SpatialAIChallenge2026` | Collaborator's repo = I-GUIDE submission (incl. model LFS files) |
| Anvil HPC (collaborator's account) | Stage 2b training data (`flood_stage2_prep_r48`) and sweep outputs — regenerable from public xBD via `pipeline/docs/stage2b.md` commands |

**Path gotcha**: the chips manifest hardcodes an old, deleted directory
(`/media/gisense/xihan/250812_CyberTraining_Team4/...`). All pipeline scripts remap it
via `pipeline/scripts/la_fire_paths.py` (`rewrite_la_fire_path()`); if you write new
scripts, use that helper.

### Environment

Conda env **`geoai_sam`** (samgeo 1.0.1 with `SamGeo3`, CUDA, geoai, rasterio, shapely).
Do not use `geoai_sam3` (older samgeo, no SamGeo3). SAM3 weights auto-download from
Hugging Face after one-time `huggingface-cli login`.

## 6. How to re-run

```bash
# Stage 1 smoke test (3 images, ~30 s on GPU)
conda run -n geoai_sam python stage1/tests/smoke_test.py

# Full LA fire pipeline (auto-skips existing outputs)
cd pipeline
conda run -n geoai_sam python scripts/run_multidate_experiment.py --device cuda:0 --workflow realworld

# QC overlays after a run
conda run -n geoai_sam python scripts/analysis/generate_qc_overlays.py

# Rebuild the combined georeferenced product
conda run -n geoai_sam python scripts/analysis/build_combined_dataset.py

# xView2 Stage 1 benchmark
conda run -n geoai_sam python evaluation/evaluate_predictions.py   # see evaluation/ README
```

See `CLAUDE.md` and `pipeline/README.md` for parameters and troubleshooting.

## 7. Known loose ends

- No LICENSE file, though `README.md` references MIT.
- `II_package/` working tree is dirty (2 modified notebooks, untracked `runs/`); the
  edits were demo tweaks and were superseded by the final submission clone in `archive/`.
- Stage 2a was integrated for wiring but never quality-tuned; treat its type/population
  outputs as rough exposure signals only.
- 1,245 buildings (5.7%) are `unknown` in the final product (no post date passed both
  tile- and crop-quality checks).
- `reports/m2b_validation.md` describes an earlier 10,607-building run; the current
  canonical product has 21,797 buildings (Stage 1 was re-run with tiling enabled).
