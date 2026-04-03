# M2b Full Workflow Report

**Date**: 2026-04-03
**Dataset**: LA Fire 2025, 10,607 buildings across 120 cells (295 total cells, 175 with zero buildings)

---

## 1. Validation Summary

See `reports/m2b_validation.md` for the detailed validation report. Key findings:

- **M2b = coverage-aware majority vote** is the authoritative real-world aggregation rule
- Valid date = `tile_quality_ok AND crop_quality_ok`
- 0 valid dates → NOT_IDENTIFIABLE (-1)
- Otherwise → majority vote across valid per-date labels, tie-break = highest damage class (conservative)
- 432 NOT_IDENTIFIABLE buildings (4.1%) — genuine coverage gaps, identical to M1b
- 612 false "destroyed" labels from M1 (nodata artifacts) correctly eliminated
- 1,269 buildings shift from no_damage (M1b) to minor (M2b) — legitimate: majority vote captures discrete per-date signals that probability averaging dilutes
- 91.6% of buildings have 4+ valid dates — strong statistical basis

### Final M2b Distribution

| Class | Count | % |
|-------|-------|---|
| NOT_IDENTIFIABLE (-1) | 432 | 4.1% |
| No damage (0) | 7,827 | 73.8% |
| Minor (1) | 2,315 | 21.8% |
| Major (2) | 33 | 0.3% |
| Destroyed (3) | 0 | 0.0% |

## 2. Workflow Definition

### Modes (`--workflow` flag in `run_multidate_experiment.py`)

| Mode | Post dates | Quality filtering | Aggregation | Use case |
|------|-----------|-------------------|-------------|----------|
| `training` | First available only | None | None (single-date label) | xView2 benchmark, model training |
| `realworld` (default) | All available | Tile + crop quality | M2b majority vote | Operational disaster assessment |

### Key Scripts

| Script | Purpose |
|--------|---------|
| `run_multidate_experiment.py` | End-to-end pipeline: Stage 1 → shared instances → per-date Stage 2b → aggregation |
| `aggregate_multidate_predictions.py` | Computes all 5 methods (M1, M1b, M2, M2b, M3) per cell |
| `re_aggregate_all.py` | Re-runs aggregation on all 120 cells (no re-inference) |
| `build_combined_dataset.py` | Merges all cells into `building_damage_all_cells.csv` |
| `generate_qc_overlays.py` | One QC overlay image per tile |

## 3. What Was Cleaned / Archived

| Action | What | Why |
|--------|------|-----|
| Renamed | `exploration/` → `archive/` | Clearly separates experimental work from production |
| Created | `archive/LEGACY_NOTE.md` | Documents what's in the archive and why |
| Created | `pipeline/stage1/LEGACY_NOTE.md` | Documents SAM3_Final legacy adapter |
| Created | `src/cybertraining_team4/LEGACY_NOTE.md` | Documents early-stage package |
| Updated | `CLAUDE.md` | All references updated to reflect new paths |
| Preserved | All code and data | Nothing was deleted; experimental work is archived, not removed |

## 4. Result Locations

| Output | Path |
|--------|------|
| Per-cell aggregated predictions | `/media/data/la_fire_2025/stage2_damage/multidate_full_run/cell_XXXXX/aggregated_predictions.jsonl` |
| Per-cell CSV | `/media/data/la_fire_2025/stage2_damage/multidate_full_run/cell_XXXXX/aggregated_predictions.csv` |
| Combined CSV (all cells) | `/media/data/la_fire_2025/stage2_damage/multidate_full_run/building_damage_all_cells.csv` |
| QC overlay images | `/media/data/la_fire_2025/qc_overlays_m2b/` (119 PNGs, ~669 MB) |
| Validation report | `reports/m2b_validation.md` |
| This report | `reports/m2b_full_workflow.md` |

## 5. Sample Inspection Set — Top 10 Tiles to Check First

Selected by **highest absolute damage count** (minor + major), which surfaces the tiles where the most buildings are affected and where visual QC is most informative:

| Rank | Cell | Buildings | Damaged | Damage % | Distribution | Why inspect |
|------|------|-----------|---------|----------|--------------|-------------|
| 1 | cell_00524 | 140 | 111 | 79.3% | 109 minor, 2 major | Highest damage count AND high %; check if minor labels are credible |
| 2 | cell_00457 | 168 | 101 | 60.1% | 99 minor, 2 major | Large tile, 60% damaged — verify building detection quality |
| 3 | cell_00495 | 144 | 77 | 53.5% | 75 minor, 2 major | >50% damaged in a dense tile |
| 4 | cell_00458 | 92 | 65 | 70.7% | 64 minor, 1 major | High damage rate, check major classification |
| 5 | cell_00516 | 137 | 56 | 40.9% | 56 minor | All minor, no major — check borderline cases |
| 6 | cell_00443 | 141 | 55 | 39.0% | 55 minor | Similar profile to cell_00507 |
| 7 | cell_00507 | 141 | 55 | 39.0% | 55 minor | Paired with cell_00443 for consistency check |
| 8 | cell_00456 | 144 | 54 | 37.5% | 53 minor, 1 major | Large tile, moderate damage |
| 9 | cell_00525 | 72 | 52 | 72.2% | 50 minor, 2 major | High % in smaller tile — check for false positives |
| 10 | cell_00417 | 144 | 51 | 35.4% | 51 minor | Baseline comparison — lower damage rate |

**Inspection guidance**: Open the QC overlay PNG for each cell. Verify that:
- Building polygons align with visible structures in the post-disaster image
- "Minor" labels correspond to visible smoke damage, debris, or discoloration (not false positives from image artifacts)
- "Major" labels show clear structural damage
- NOT_IDENTIFIABLE buildings are in genuinely occluded areas

---

## Required Answers

### Q1: Is M2b now the default real-world workflow?

**Yes.** M2b (coverage-aware majority vote) is the default aggregation method for `--workflow realworld`. It is computed by `aggregate_multidate_predictions.py` and reported in the `m2b_coverage_vote_class` field of every `aggregated_predictions.jsonl`. The combined CSV (`building_damage_all_cells.csv`) includes `m2b_damage_class` and `m2b_damage_label` columns.

### Q2: What was archived vs deleted?

**Archived (not deleted):**
- `exploration/` directory renamed to `archive/` — contains Mask R-CNN, PolyWorld, GeoAI_QuishengWu, SAM3_notebooks, SAM3_Final, corrected_model experiments
- Each legacy directory has a `LEGACY_NOTE.md` explaining its status

**Nothing was deleted.** All experimental code, notebooks, and outputs are preserved in `archive/` for reference.

### Q3: Did the full workflow cover all tiles?

**Yes.** All 120 cells with buildings have M2b predictions in their `aggregated_predictions.jsonl`. The remaining 175 cells in the run root have zero building instances (confirmed by `zero_instances.marker` files). Total: 10,607 buildings across 120 cells with M2b classifications.

### Q4: Are all QC images generated? Where?

**119 of 120 cells have QC overlay images** at `/media/data/la_fire_2025/qc_overlays_m2b/`. The one skipped cell (`cell_00531`) had no valid post-disaster image available. Each image shows building polygons colored by M2b damage class over the best-quality post-disaster satellite image, with damage class labels at building centroids and a legend.

### Q5: Which 10 tiles should be checked first, and why?

See **Section 5** above. The 10 tiles are selected by highest absolute damage count (not just percentage), because:
1. **Statistical significance**: Tiles with more damaged buildings provide more evidence to validate the classification model
2. **Impact prioritization**: These tiles represent the most affected areas in the LA fire
3. **Error detection**: If the model has systematic biases, they are most visible where damage is concentrated
4. **Mix of profiles**: The list includes tiles with both high damage percentage (cell_00524 at 79%) and moderate percentage (cell_00417 at 35%), enabling comparison across severity levels
