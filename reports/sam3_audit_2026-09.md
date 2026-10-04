# Stage 1 (SAM3) Audit and Rewrite — September 2026

**Scope**: the SAM3 building-detection stage (`stage1/`), its xView2 benchmark and prompt
experiments, the LA fire 2025 deployment that depends on it, and (October) field validation and
Stage 2b retraining.
**Branch**: `sam3-stage1-fixes` (uncommitted for review).
**Experiment data**: `/media/data/building_instance_tamu/tiling_experiments/` (runs A–L, logs,
`eval/<run>/eval_<run>.json`, `run_tiling.sh`).

## 1. Summary

The published Stage 1 numbers (Feb 2026: P 0.682 / R 0.284 / F1 0.401) did not describe the
code that was deployed on LA fire, and the deployed code had a stitching bug. After fixing the
code and re-tuning, Stage 1 reaches **P 0.737 / R 0.565 / F1 0.640** on the same xView2 test
set. The "~30% recall ceiling" in `PROJECT_CONCLUSION.md` was an artifact of running SAM3 on
full 1024 px images; tiled inference roughly doubles recall.

October follow-up (§6–9): validated against CAL FIRE DINS for both LA fires, Stage 1 footprints
reach F1 0.60 against county outlines after correcting a ~4 m imagery offset. The April flood-only
Stage 2b found almost none of the destroyed buildings; the same model retrained on all xView2
events gives the new LA product (`multidate_full_run_v3/`) destroyed-building F1 0.96 on detected
buildings and 0.86 end to end. A training-free persistence check reaches 0.82–0.86.

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

## 6. Field validation against CAL FIRE DINS (2026-10-02)

**Ground truth**: LA County `WildFire_BUILDINGS_100M_Plus_DINS_MaxMinDAMAGE` feature service
(LARIAC building outlines within 100 m of the 2025 perimeters with CAL FIRE DINS max/min damage,
27 Feb 2025; 34,313 buildings) and the WFIGS perimeters for Eaton and Palisades.
**Region**: imaged area (non-zero pixels of the 295 pre chips) ∩ (perimeters + 100 m) = 83.3 km²,
14,369 buildings. Data, scripts and results: `/media/data/building_instance_tamu/la_fire_2025/validation/`;
scripts and result JSONs are also in `evaluation/la_fire_validation/`.

**Geolocation offset.** Maxar footprints sit a median 4.3 m from the LARIAC outlines, with a
different offset per 600 m cell (x from −4.7 to +0.7 m). All IoU metrics below apply one
translation per cell (median centroid offset of matched pairs, cells with ≥ 30 pairs). Without it,
IoU-based F1 falls to ~0.20. Pre/post Maxar chips are co-registered to < 1 px in most pairs.

**Stage 1 vs LARIAC** (IoU ≥ 0.5, co-registered): April v1 P 0.570 / R 0.557 / F1 0.563;
v0.2 P 0.606 / R 0.591 / F1 0.598. Centroid-in-footprint recall 0.790, overlap precision 0.881.
Recall by area: < 50 m² 0.60, 50–100 m² 0.70, 100–200 m² 0.84, 200–400 m² 0.90, > 400 m² 0.91.
DINS-inspected structures: 0.81–0.88.

## 7. Damage experiments against DINS

Destroyed vs not destroyed; DINS "Destroyed (>50%)" is positive. Only (building, date) pairs whose
footprint has valid post-fire pixels on a tile that passed the quality filter count.

**Experiment 1 — building persistence (training-free).** SAM 3 (same stage1 v0.2 config) run on
all 673 post-fire chips of the 124 evaluated cells (`la_fire_2025/postfire_sam3/`). Persistence =
share of a pre-fire footprint still segmented as "building". Destroyed if < 0.5 (fixed, not tuned).
On 9,168 matched buildings: first valid date P 0.799 / R 0.882 / F1 0.839; max over dates
P 0.942 / R 0.792 / F1 0.861; AUC 0.87–0.90. Eaton 0.86–0.88, Palisades 0.54–0.57 (272 destroyed).
Thresholds tuned on one fire and tested on the other give 0.65–0.75 on Palisades, 0.87–0.90 on Eaton.
No post date looked like reused pre-fire imagery.

**Experiment 5 — single vs multiple dates (April Stage 2b).** On single dates the flood model's few
"destroyed" calls (recall 0.07–0.09) came almost entirely from crops with no image data; restricted
to valid imagery it found 1 of 4,084 destroyed buildings. The M2b majority vote removed every
"destroyed" call. The model was more confident on DINS-destroyed buildings it got wrong
(mean top-class probability 0.69) than on undamaged ones (0.65), after temperature calibration.

**Experiment 2 — oracle footprints.** LARIAC outlines, shifted onto the Maxar grid, replace Stage 1
(14,361 footprints in 124 cells; `la_fire_2025/oracle_footprints/`, Stage 2b run in
`stage2_damage/multidate_oracle/`). On 10,876 DINS buildings (4,988 destroyed):
persistence F1 0.851 (max over dates) vs 0.768 end to end with SAM 3 footprints; flood Stage 2b
F1 0.078 even with oracle footprints. Stage 1 misses 18% of destroyed buildings (detected share
0.818 vs 0.863 for others), which is most of the end-to-end loss. Palisades stays low for
persistence with oracle footprints (0.57), so its difficulty is in the post-fire imagery, not Stage 1.

## 8. Stage 2b retrained on xView2 (2026-10-03)

Same architecture and run019 settings (ConvNeXt-tiny Siamese, CORAL, mask+ring pooling, weighted
sampler, EMA, early stopping), single GPU. Training crops for the 10 xView2 tier1 events were made
with the LA inference crop generator from ground-truth pre-disaster polygons
(`/media/data/building_instance_tamu/stage2_training_data/`, scripts in `evaluation/stage2b_retraining/`).
Three event sets: A all hazards (159,794 buildings, 13,227 destroyed), B fire only (22,998; 4,800),
C all but fire (136,796; 8,427). The April model had 502 destroyed examples.

Destroyed F1 against DINS, first valid date, single checkpoint, no calibration:

| Model | Training events | Oracle footprints | SAM 3 end to end | Eaton | Palisades |
|---|---|---:|---:|---:|---:|
| April Stage 2b (run019) | flood (3) | 0.078 | 0.000 | 0.078 | 0.077 |
| C | all but fire (8) | 0.429 | 0.383 | 0.418 | 0.565 |
| Persistence | none | 0.822 | 0.759 | 0.843 | 0.546 |
| A | all hazards (10) | 0.913 | 0.845 | 0.918 | 0.848 |
| B | fire only (2) | 0.940 | 0.874 | 0.950 | 0.805 |

xView2 test split (53,850 buildings): macro-F1 / destroyed F1 — April 0.325 / 0.022,
A 0.772 / 0.829, B 0.385 / 0.592, C 0.702 / 0.757. Fire test events, destroyed F1: April 0.000,
A 0.918, B 0.925, C 0.739. The April model scores 0.40 macro-F1 on the xView2 flood test events
here vs 0.73 on its own flood validation split (different split and preprocessing).

Caveat: the two training fires (Santa Rosa, SoCal, both 2017) are Californian, so A and B are a
same-hazard, nearby-region transfer to LA, not a test on an unfamiliar setting.

`run_multidate_experiment.py` now takes `--stage2b_model {xview2_all, flood_2026_04}`
(default `xview2_all` = model A; weights in `pipeline/models/stage2b_xview2_all/`, git-ignored).

## 9. LA final product v3

`multidate_full_run_v3/` (2026-10-03): v2 footprints, crops and Stage 2a, with Stage 2b replaced by
model A (all-hazard xView2). Built by hard-linking v2 and regenerating only the Stage 2b outputs,
aggregation and combined product (`evaluation/stage2b_retraining/make_v3.sh`, `split_jsonl.py`).
Combined product: 22,024 buildings, M2b classes 14,486 no damage / 122 minor / 370 major /
5,748 destroyed / 1,298 unknown (v2: 126 destroyed). Copied to `results/final_product/`.

Against DINS, M2b destroyed vs not destroyed:

| Product | Buildings | Precision | Recall | F1 | Eaton | Palisades |
|---|---:|---:|---:|---:|---:|---:|
| v2 (April Stage 2b), detected buildings | 9,168 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| v3 (model A), detected buildings | 9,168 | 0.964 | 0.952 | 0.958 | 0.959 | 0.947 |
| v3, end to end (Stage 1 misses count as not detected) | 10,876 | 0.964 | 0.779 | 0.862 | 0.859 | 0.895 |

With model A the M2b multi-date vote helps (F1 0.909 for the per-date probability average M1,
0.958 for M2b), the opposite of the flood model, where the vote removed every destroyed call.
Any-damage F1 (DINS affected or worse) is 0.916. The remaining gap is Stage 1: 18% of destroyed
buildings are never detected on the pre-fire image (`results` in `la_fire_2025/validation/v3_product_validation.json`).

## 10. Open issues

- Stage 1 recall on small and destroyed buildings (18% of destroyed buildings never reach Stage 2).
- Palisades: fewer imaged cells and lower scores for every method; check imagery dates and angles.
- Full xBD tier3 adds three wildfires including Woolsey (2018, Los Angeles); download needs an
  xview2.org account.
- The two training fires are Californian; a fire outside California would test generalisation.
- xView2 train-split SAM 3 predictions were not re-run with v0.2.
- Prompts `rooftop`, `building rooftop`, `structure` not re-run with v0.2.
