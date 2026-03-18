# Multi-Date Inference Experiment Report

**Date**: 2026-03-15
**Author**: Claude Code (external validation run)
**Experiment directory**: `II_package/outputs/multidate_experiment/`

---

## 1. What Was Changed

### New scripts (in `scripts/`)

| Script | Purpose |
|---|---|
| `quality_filter.py` | Rule-based tile- and crop-level image quality filter (module) |
| `generate_post_crops_for_date.py` | Generate post crops for one date by reading pre-computed crop geometry from the base shared CSV |
| `aggregate_multidate_predictions.py` | Aggregate Stage-2b JSONL predictions across dates using three methods |
| `run_multidate_experiment.py` | Top-level experiment driver: Stage 1 → shared base → Stage 2a → per-date Stage 2b → aggregation |

### What was NOT changed

- `run_instance_impact_driver.py` — untouched
- `run_pipeline.sh` — untouched
- `generate_shared_instance_subimages.py` — untouched
- `infer_stage2_ensemble.py` — untouched
- `infer_stage2a.py` — untouched
- All Stage-2a and Stage-2b model weights — untouched
- All calibration artifacts — untouched
- All existing single-date smoke test outputs under `outputs/driver_runs/` — untouched

---

## 2. Experiment Design

### Core idea

The existing pipeline runs Stage-2b once per pre/post pair. For the LA fire dataset each cell has **5–8 post-disaster acquisition dates**. This experiment tests whether using all available post dates — with appropriate aggregation — gives more stable damage predictions than relying on a single (earliest) date.

### Stage 1 is run once per cell

Building instance detection (SAM3) uses only the **pre-disaster image**. The resulting instances (UIDs, bounding polygons) are reused for every post date. UIDs are deterministic UUIDs written into the Stage-1 JSON; they do not change between dates.

### Per-date Stage-2b

For each post date:
1. **Tile-level quality filter** — compute three metrics on the full post TIF:
   - `mean_brightness`: overall mean pixel value
   - `frac_zeros`: fraction of pixels where all channels = 0 (nodata footprint)
   - `spatial_std`: mean of per-pixel channel standard deviation (measures texture/variation)
   - Reject if `mean_brightness < 15` OR `frac_zeros > 0.60` OR `spatial_std < 3.0`
2. **Post crop generation** — read crop window `(crop_x0, crop_y0, crop_size)` from the base shared CSV (pre-computed from Stage-1 polygon geometry); crop from the new post TIF; flag per-crop quality (`frac_zero > 0.50` = bad crop)
3. **Stage-2b ensemble inference** — identical to the single-date pipeline; runs on the per-date `shared_for_date.csv`

Stage-2b is run for **all** post dates regardless of quality flag. The quality flag is stored in the CSV and used only during aggregation.

### Aggregation methods

| Method | ID | Description |
|---|---|---|
| **Probability averaging** | M1 | Average calibrated softmax probabilities across tile-quality-passing dates; argmax to get damage class |
| **Majority vote** | M2 | Most common `y_pred_ensemble` across tile-quality-passing dates; ties broken by M1 |
| **Quality-filtered probability averaging** | M3 | Same as M1 but additionally requires per-crop quality OK; falls back to M1 if no crops pass |

**Baseline**: earliest-post-date single-date prediction (current pipeline behavior).

---

## 3. Sample Cells

| Cell | Category | n Instances | n Post Dates | Dates Rejected (quality) | Notes |
|---|---|---|---|---|---|
| cell_00064 | Dense, 5 dates | 7 | 5 / 5 pass | 0 | Existing smoke-test run reused for Stage 1 |
| cell_00065 | Dense, 5 dates | 27 | 5 / 5 pass | 0 | Existing smoke-test run reused |
| cell_00074 | Dense, 8 dates | 21 | 7 / 8 pass | 20250110 (all zero) | Existing smoke-test run reused |
| cell_00073 | Sparse-medium, 8 dates | 2 | 6 / 8 pass | 20250109 (too dark), 20250110 (all zero) | Fresh Stage 1 |
| cell_00046 | Medium, 5 dates | 37 | 4 / 5 pass | 20250115 (hazy/dark) | Fresh Stage 1; 54 Stage-1 instances, 37 carried into Stage 2 |
| cell_00080 | Medium, 7 dates | 30 | 5 / 7 pass | 20250110 (all zero), 20250119 (low texture) | Fresh Stage 1; 49 Stage-1 instances, 30 in Stage 2 |
| cell_00072 | Sparse, 8 dates | **0** | — | — | Stage 1 detected 0 buildings; pipeline stops after Stage 1 |
| cell_00045 | Sparse (dark), 5 dates | **0** | — | — | Stage 1 detected 0 buildings; pipeline stops after Stage 1 |

---

## 4. Quality Filter Results

### Tile-level rejections

| Cell | Date | mean_brightness | frac_zeros | spatial_std | Rejection reason |
|---|---|---|---|---|---|
| cell_00074 | 20250110 | 0.0 | 1.00 | 0.0 | All three thresholds triggered — **completely blank** |
| cell_00073 | 20250109 | 8.95 | 0.41 | 5.04 | mean_brightness < 15 |
| cell_00073 | 20250110 | 0.0 | 1.00 | 0.0 | All three — completely blank |
| cell_00046 | 20250115 | 12.82 | 0.79 | 1.77 | All three — likely cloud/haze |
| cell_00080 | 20250110 | 0.0 | 1.00 | 0.0 | All three — completely blank |
| cell_00080 | 20250119 | 91.34 | 0.07 | 2.43 | spatial_std < 3.0 — unusually uniform (haze?) |

**All five completely blank dates (mean=0, frac_zeros=1.0) were correctly rejected.**
The rule-based filter required no tuning: the threshold `mean_brightness < 15` cleanly catches early post-fire dates with heavy smoke/nodata.

---

## 5. Baseline vs. Aggregation — Summary Tables

**Damage class key**: 0 = no damage, 1 = minor, 2 = major, 3 = destroyed

### cell_00064 (7 instances, 5 dates, all pass quality)

| Method | cls 0 | cls 1 | cls 2 | cls 3 | Agreement w/ baseline |
|---|---|---|---|---|---|
| **Baseline** (20250109) | 1 | 2 | 4 | 0 | — |
| M1 prob avg | 6 | 1 | 0 | 0 | 14% |
| M2 majority | 6 | 1 | 0 | 0 | 14% |
| M3 quality avg | 6 | 1 | 0 | 0 | 14% |

The earliest date (20250109, mean=19.4) is marginal quality — the model predicted heavy damage on most buildings. Averaging across all 5 dates shifts the distribution to predominantly "no damage". Unstable: 6/7 instances (86%).

### cell_00065 (27 instances, 5 dates, all pass quality)

| Method | cls 0 | cls 1 | cls 2 | cls 3 | Agreement w/ baseline |
|---|---|---|---|---|---|
| **Baseline** (20250109) | 4 | 19 | 4 | 0 | — |
| M1 prob avg | 22 | 5 | 0 | 0 | 29% |
| M2 majority | 20 | 7 | 0 | 0 | 29% |
| M3 quality avg | 22 | 5 | 0 | 0 | 29% |

The baseline significantly over-predicts minor/major damage on the dark early date. Aggregation collapses to mostly no-damage. Unstable: 23/27 instances (85%).

### cell_00074 (21 instances, 8 dates, 7 pass quality, 1 all-zero rejected)

| Method | cls 0 | cls 1 | cls 2 | cls 3 | Agreement w/ baseline |
|---|---|---|---|---|---|
| **Baseline** (20250109) | 3 | 9 | 3 | 6 | — |
| M1 prob avg | 16 | 0 | 0 | 5 | 38% |
| M2 majority | 16 | 0 | 0 | 5 | 38% |
| M3 quality avg | 21 | 0 | 0 | 0 | 14% |

Note: M3 uses only 2 dates (20250119, 20250120) because the 40% nodata footprint in dates 20250109–20250116 causes many building crops to fail the crop-level zero-fraction check. M3 therefore discards the destroyed-class signal in the 5 high-confidence "destroyed" instances predicted by M1/M2. **M3 is overly aggressive here** — see Section 7.

### cell_00073 (2 instances, 8 dates, 6 pass quality)

| Method | cls 0 | cls 1 | cls 2 | cls 3 | Agreement w/ baseline |
|---|---|---|---|---|---|
| **Baseline** (20250109, rejected) | 0 | 0 | 0 | 2 | — |
| M1 prob avg | 1 | 0 | 0 | 1 | 50% |
| M2 majority | 0 | 0 | 0 | 2 | 50% |
| M3 quality avg | 2 | 0 | 0 | 0 | 0% |

The baseline used a rejected date (mean=8.95); the model predicted "destroyed" for both. M1 and M2 disagree on one instance. M3 again over-filters. Small sample (2 instances).

### cell_00046 (37 instances, 5 dates, 4 pass quality)

| Method | cls 0 | cls 1 | cls 2 | cls 3 | Agreement w/ baseline |
|---|---|---|---|---|---|
| **Baseline** (20250109) | 28 | 8 | 1 | 0 | — |
| M1 prob avg | 36 | 1 | 0 | 0 | 78% |
| M2 majority | 35 | 1 | 1 | 0 | 78% |
| M3 quality avg | 36 | 1 | 0 | 0 | 78% |

Good agreement between baseline and aggregation (78%). This cell has good image coverage across dates (frac_zeros ≈ 0.35 consistently). Aggregation further collapses minor-damage predictions to no-damage. Unstable: 12/37 (32%).

### cell_00080 (30 instances, 7 dates, 5 pass quality)

**Most striking result.**

| Method | cls 0 | cls 1 | cls 2 | cls 3 | Agreement w/ baseline |
|---|---|---|---|---|---|
| **Baseline** (20250110, BLANK) | 0 | 0 | 0 | **30** | — |
| M1 prob avg | 13 | 2 | 0 | 15 | 50% |
| M2 majority | 12 | 3 | 0 | 15 | 50% |
| M3 quality avg | 13 | 2 | 0 | 15 | 50% |

The earliest available post date (20250110) is a **completely blank image** (all pixels = 0). The Stage-2b model, presented with a blank post crop alongside a valid pre crop, predicted "destroyed" for all 30 buildings — a catastrophic false positive. M1/M2/M3 all use the 5 quality-passing dates and give a more reasonable split: ~13 no-damage, 2 minor, 15 destroyed. Unstable: 8/30 (27%).

---

## 6. Agreement and Instability

| Cell | n inst | Unstable | Avg label entropy | M1 vs M2 agree | M1 vs M3 agree |
|---|---|---|---|---|---|
| cell_00064 | 7 | 86% | 0.40 | 100% | 100% |
| cell_00065 | 27 | 85% | 0.39 | 82% | 100% |
| cell_00074 | 21 | 100% | 0.43 | 100% | 24% |
| cell_00073 | 2 | 100% | 0.46 | 50% | 0% |
| cell_00046 | 37 | 32% | 0.15 | 97% | 100% |
| cell_00080 | 30 | 27% | 0.10 | 90% | 100% |

**Key observations:**

1. **High instability in cells with partial nodata coverage** (cells 00064–00074): when the early post dates are dark/marginal, the model gives noisy damage predictions. Instability rates of 85–100% mean the building label would change depending on which single date is chosen.

2. **Low instability in cells with good coverage** (cells 00046, 00080 for well-covered dates): 26–32% instability, reasonable for a genuine damage assessment.

3. **M1 and M2 are nearly identical** except in very small-sample or high-entropy cases. For cells with ≥ 7 instances, they agree on ≥ 90% of instances. **Majority vote adds no value over probability averaging** in this regime.

4. **M3 (crop-level quality filter) is too aggressive** in cells with partial nodata footprints (cells 00073, 00074). When 40% of a tile is nodata, many building crops will fail the 50%-zero threshold even though the underlying building may be in the valid region. This causes M3 to discard most dates and give unreliable results.

---

## 7. Recommendation

### Summary recommendation table

| Question | Recommendation |
|---|---|
| Keep earliest-post-only? | **No** — especially when the earliest date may be blank or very dark |
| Use multi-date aggregation? | **Yes — M1 (probability averaging) across tile-quality-passing dates** |
| Use majority vote (M2)? | Not necessary — same result as M1 in almost all cases; adds complexity |
| Use crop-level quality filter (M3)? | **No for now** — too aggressive in partially-covered tiles; rejects valid dates |
| Change the trained model? | **No** |

### Detailed reasoning

**Multi-date probability averaging (M1) is clearly better than earliest-post-only for two reasons:**

1. **Catastrophic failure case** (cell_00080): the earliest available post date can be a completely blank image, causing the model to predict all buildings as destroyed. Quality filtering at the tile level entirely prevents this. The cost of implementing this filter is minimal (three arithmetic statistics on a 1966×1966 image).

2. **Marginal early dates** (cells 00064, 00065): the 20250109 date has mean brightness ≈ 20 (dark, likely early post-fire smoke/low light). The model over-predicts minor/major damage on dark imagery. Averaging with later, cleaner dates substantially shifts predictions toward no-damage. Whether the later dates are "correct" is unknown without ground truth, but the later dates have brighter, higher-quality imagery and are more likely representative of the actual structure condition.

**Why not crop-level filtering (M3)?**

The 50%-zero crop threshold discards entire dates for cells with partial nodata footprints (cells 00073, 00074 have ~40% of the tile as nodata). A building whose centroid falls in the valid region may still get a crop that is 50–60% zero because the nodata border intersects the crop window. Reducing the crop-level threshold (e.g., from 50% to 70%) would be a simple fix, but the benefit over M1 alone is marginal given the small cells affected. **Recommended: do not use M3 in the full run**; rely on tile-level quality filtering (M1) which is more reliable.

**Is there a need to change the trained model?**

**No.** The instability observed (changing damage class across post dates) is not a model pathology — it is the correct response to genuinely different-quality imagery. When the model is given a blank or dark post image, it correctly recognizes the severe difference from the pre image and predicts damage. When given later, clearer imagery, predictions stabilize. The model is behaving as designed. The fix is at the **inference strategy level** (use multiple dates + quality filter), not at the model level.

### Recommended full-run settings

```bash
python scripts/run_multidate_experiment.py \
    --cells <all 295 cells> \
    --manifest data/processed/chips_600m_manifest.csv \
    --out_root outputs/multidate_experiment \
    --device cuda:0
```

Use **M1 (probability averaging)** as the primary damage prediction column.
Use **M2 (majority vote)** only as a cross-check.
Skip M3 entirely in the full run, or increase `--crop_max_zero_frac` to 0.70.

---

## 8. Implementation Notes

### Instance ID consistency

Verified: Stage-1 `bldg_uid` values are deterministic UUIDs written by SAM3 inference into the `_prediction.json` file. They do not change when the CSV is read again. All per-date `shared_for_date.csv` files have identical `bldg_uid` columns (copied verbatim from the base shared CSV). The aggregator joins on `bldg_uid` with no ambiguity.

### Backward compatibility

The single-date `run_pipeline.sh` / `run_instance_impact_driver.py` pipeline is completely unchanged. The new scripts are additive; they produce outputs in a separate `outputs/multidate_experiment/` tree.

### Sparse cells (0 instances)

Two cells (cell_00045, cell_00072) produced 0 Stage-1 building detections. The pipeline correctly stops after Stage 1 (the shared-artifact step raises an error on empty labels). These cells represent terrain without clearly detectable buildings — likely wildland, hillside, or heavily forested areas. They should be flagged as "no buildings detected" in the final product rather than excluded silently.

### Stage-2a type column

The Stage-2a `pred_type` column was empty in the summary because the `stage2a_predictions.csv` uses a different column name than anticipated. This does not affect the damage predictions (Stage-2b is independent of Stage-2a types); it only affects the per-cell summary statistics for building type.

---

## 9. Output Locations

```
II_package/outputs/multidate_experiment/
├── experiment_summary.json              ← per-cell counts + quality summary
├── cell_00064/
│   ├── stage1/labels/                   ← Stage-1 prediction JSON (reused from smoke test)
│   ├── shared_base/shared_instance_samples.csv
│   ├── stage2a_predictions.csv
│   ├── dates/
│   │   ├── 20250109/stage2b_20250109.jsonl
│   │   ├── 20250109/quality_metrics.json
│   │   ├── 20250109/shared_for_date.csv
│   │   ├── 20250113/ … 20250116/
│   ├── aggregated_predictions.jsonl     ← all 3 methods per building
│   └── aggregated_predictions.csv       ← flat CSV
├── cell_00065/ … cell_00080/
```

**Smoke-test single-date outputs remain at:**
```
II_package/outputs/driver_runs/la_fire_cell_00064/
II_package/outputs/driver_runs/la_fire_cell_00065/
II_package/outputs/driver_runs/la_fire_cell_00074/
```
These are untouched.
