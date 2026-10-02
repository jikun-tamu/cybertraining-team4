# Stage 1 (SAM3) Audit and Rewrite — September 2026

**Scope**: the SAM3 building-detection stage (`stage1/`), its xView2 benchmark and prompt
experiments, and the LA fire 2025 deployment that depends on it.
**Branch**: `sam3-stage1-fixes` (uncommitted for review).
**Experiment data**: `/media/data/building_instance_tamu/tiling_experiments/` (runs A–L, logs,
`eval/<run>/eval_<run>.json`, `run_tiling.sh`).

## 1. Summary

The published Stage 1 numbers (Feb 2026: P 0.682 / R 0.284 / F1 0.401) did not describe the
code that was deployed on LA fire, and the deployed code had a stitching bug. After fixing the
code and re-tuning, Stage 1 reaches **P 0.737 / R 0.565 / F1 0.640** on the same xView2 test
set. The "~30% recall ceiling" in `PROJECT_CONCLUSION.md` was an artifact of running SAM3 on
full 1024 px images; tiled inference roughly doubles recall.

## 2. Findings

| # | Finding | Impact |
|---|---|---|
| 1 | Benchmark (Feb 26) and prompt experiments (Mar 19) were run on full images with an earlier code version; LA (Apr 4) ran tiled with `min_size=30`, `min_polygon_area=100`. | Published metrics did not describe the deployed configuration. |
| 2 | `tiling.stitch_masks()` kept the higher score per pixel in overlaps, splitting any building that crossed a window seam into several labels. On 60 LA masks, 66% of labels touched another label, 75% of those contacts lay in the overlap strips (30% of the area), and 9,876 labels became 12,958 polygons. | LA building count inflated; fragments were sent to Stage 2 as separate buildings. |
| 3 | The Feb benchmark predictions contained 4,412 polygons smaller than 10 px (no post-filter at the time), all false positives; 1,385 instances had confidence 0. The prompt runs had the same problem (e.g. `house`: 4,137). | Published precision understated (0.682; 0.846 for the same config without junk). |
| 4 | Confidence was read at the polygon centroid; for concave shapes the centroid can fall outside the mask (63 LA instances with confidence 0; old cell_00365 min 0.023). | Wrong `sam3_confidence` passed to Stage 2. |
| 5 | Meta's SAM3 crashes on `--device cuda:1` (decoder tensors created on the current device). | Second GPU unusable as documented. |
| 6 | Tiled mode never wrote annotation PNGs; stitched masks lost the GeoTIFF CRS/transform; `--batch-size`, `save_predictions` and the batch methods in `model.py` were dead code; `total_wall_time_sec` was a sum of per-image times. | Docs/CLI promised behaviour that did not exist. |
| 7 | `pipeline/scripts/infer/infer_stage2_ensemble.py` looked for `stage2b_model` one directory too high after the April 12 script reorganisation (`c8867d5`). | The LA pipeline could not run Stage 2b after April 12. Fixed. |
| 8 | Duplicated code/results: three copies of `SAM3_Final`, the evaluation logic copy-pasted into `run_prompt_experiments.py`, `results/sam3_eval` identical to `evaluation/results/sam3_eval`. Stale paths: `/media/data/.../sam3/test` (never existed under that name), env `geoai_sam3` (deleted), `SAM3_Claude` in `sys.path`. | Confusing; see cleanup list. |
| 9 | Editing through the Finder/rclone mount creates `._*` AppleDouble files; 33 were found inside `.git/objects`. | Git warnings; `._*` added to `.gitignore`. |

## 3. Changes (stage1 v0.2.0)

- **Instance-level merging** (`tiling.merge_instances`): per-window instances are mapped to
  image coordinates; two instances from different windows merge when their masks have IoU ≥ 0.5
  inside the region both windows see. Windows are evenly spaced (no sliver tiles).
- Inference runs in memory on RGB arrays (no temporary tile PNGs/TIFs). Labels are painted in
  score order; each label keeps its own SAM3 score and only its largest polygon.
- Masks are written compressed and georeferenced; annotations work in tiled mode.
- New options `--model` (`facebook/sam3` / `facebook/sam3.1`), `--confidence-threshold`
  (default 0.4), `--merge-iou`; `--device cuda:N` works.
- `run_summary.json` records the full config and `git describe`.
- Evaluation: one implementation (`evaluate_predictions.py`, now with `--pred-dir`);
  `run_prompt_experiments.py` runs every prompt (including the baseline) with the same config.
- Tests: `stage1/tests/test_tiling.py` (no GPU) and an updated `smoke_test.py`.

## 4. Benchmark — xView2 test, 933 pre-disaster images, IoU ≥ 0.5

| Run | Configuration | Pred | P | R | F1 | mIoU |
|---|---|---:|---:|---:|---:|---:|
| (Feb) | Published, full image | 22,824 | 0.682 | 0.284 | 0.401 | 0.759 |
| A | Old code, full image (thr 0.5) | 18,410 | 0.846 | 0.284 | 0.425 | 0.759 |
| E | New code, full image | 17,866 | 0.871 | 0.284 | 0.428 | 0.759 |
| F | New code, full image, SAM 3.1 | 17,755 | 0.869 | 0.281 | 0.425 | 0.759 |
| G | New code, full image, thr 0.4 | 23,019 | 0.818 | 0.343 | 0.483 | 0.751 |
| H | New code, full image, thr 0.3 | 29,218 | 0.750 | 0.400 | 0.521 | 0.744 |
| B | Old code, tiled, pixel stitching | 42,189 | 0.639 | 0.492 | 0.556 | 0.759 |
| C | Old code, tiled, LA config | 36,925 | 0.729 | 0.491 | 0.586 | 0.759 |
| D | New code, tiled, thr 0.5 | 35,036 | 0.794 | 0.507 | 0.619 | 0.763 |
| **J** | **New code, tiled, thr 0.4 (default)** | 42,087 | 0.737 | 0.565 | **0.640** | 0.756 |
| K | New code, tiled, thr 0.3 | 50,351 | 0.656 | 0.603 | 0.628 | 0.750 |
| L | New code, tiled, thr 0.4, prompt `house` | 45,922 | 0.708 | 0.593 | 0.645 | 0.752 |

All runs: `min_size=100`, `epsilon=2`; A–I `min_polygon_area=10`, C/J/K/L `100`;
tiles 512 px, overlap ≥ 64. Run J is the new reference
(`xview2_sam3_outputs_v2/test/`, `evaluation/results/sam3_eval/`).

Per-disaster recall, E → J: palu-tsunami 0.08 → 0.44, mexico-earthquake 0.03 → 0.27,
hurricane-harvey 0.32 → 0.76, santa-rosa-wildfire 0.60 → 0.87, socal-fire 0.59 → 0.74.

**Decisions**
- Tiling with instance merging: largest single gain (E → D: F1 +0.19).
- Threshold 0.4: best F1 among tiled runs.
- Prompt `building` kept: `house` (L) is +0.005 F1 overall but lower recall on both
  wildfire events (santa-rosa 0.845 vs 0.868, socal 0.734 vs 0.737), and LA is a wildfire.
- SAM 3.1 (Mar 2026) not adopted: its release targets video multi-object tracking; on still
  images it matched SAM 3 (F vs E). It remains selectable with `--model facebook/sam3.1`.

Known limitation: the merge rule can join two adjacent buildings when one window segments them
as a single object (seen on large commercial blocks).

## 5. LA fire 2025 re-run

`multidate_full_run_v2/` — all 295 cells, stage1 v0.2 defaults (`--min-size 30` as before),
Stage 2a/2b unchanged, completed 2026-10-01 03:15 with 0 failed steps (~6 h on one A6000 shared
with other jobs). Combined product: `building_damage_all_cells.{csv,geojson,gpkg}`
(also in `results/final_product/`).

| | April 2026 (v1) | October 2026 (v2) |
|---|---:|---:|
| Building instances | 21,797 | 22,024 |
| Cells with buildings | 120 | 126 |
| Median footprint (m²) | 138 | 153 |
| SAM3 confidence, min | 0.00 | 0.40 |
| M2b no damage / minor / major / destroyed / unknown | 16,246 / 4,174 / 39 / 93 / 1,245 | 16,395 / 4,145 / 60 / 126 / 1,298 |

Spatial comparison (UTM 11N): 17,042 v2 footprints match a v1 footprint 1:1 (IoU ≥ 0.5);
2,162 v2 footprints each replace 2+ v1 fragments (4,774 fragments in total); 2,435 v2
footprints are new detections; 409 v1 polygons have no v2 counterpart (mostly shadow/fence
false positives). The near-equal totals hide both effects: about 2,600 duplicate fragments
removed and about 2,400 buildings added.

The 169 cells with no buildings are wildland: spot checks show chaparral/forest or images
with partial coverage (30 cells), consistent with the Palisades and Eaton fire perimeters
reaching into the Santa Monica and San Gabriel mountains.

Damage labels barely moved: Stage 2b still calls ~74% of buildings "no damage" and only
126 (0.6%) "destroyed", while CAL FIRE reported thousands of destroyed structures in these
fires. Better footprints do not fix the Stage 2b domain gap (§6).

Storage: the April run directory (`multidate_full_run/`, 19 GB), the pre-fix xView2 and prompt
outputs (7.6 GB) and the experiment mask TIFs of runs A–C and E–I were deleted on 2026-10-02.
The April combined product stays in git history (`results/final_product/` at `0dc7630`).

Also fixed for this run: `infer_stage2_ensemble.py` import path (finding 7) and
`build_combined_dataset.py` `PKG_ROOT`.

## 6. Open issues outside Stage 1

- **Stage 2b on wildfire**: QC overlays show clearly burned parcels classified "no damage"
  (old and new runs alike; cell_00365: 362 of 405 buildings "no damage"). Stage 2b was trained on
  flood-only xBD and has never been validated on fire. Validating against CAL FIRE DINS
  (`PROJECT_CONCLUSION.md` §4) and retraining on fire events are the next steps.
- xView2 train-split predictions have not been re-run with v0.2 (`eval_train.json` is stale).
- Prompts `rooftop`, `building rooftop`, `structure` not re-run with v0.2 (clearly worse in the
  earlier full-image runs).
