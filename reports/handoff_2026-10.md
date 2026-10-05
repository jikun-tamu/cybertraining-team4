# Handoff — October 2026

Status of the LA fire 2025 damage assessment work after the September–October 2026 revisit,
for whoever continues it. Work is on branch `sam3-stage1-fixes` (not merged into `main`).
Server: `xihan@10.158.11.52`, env `geoai_sam`.

## Read first

1. `reports/sam3_audit_2026-09.md` — everything that changed and every number, §1–10.
2. `reports/literature_review_2026-10.md` — 42 references, what already exists, positioning options.
3. Results page with figures (private; ask Xihan for access).

## Where things stand

| Part | Status | Key number (vs CAL FIRE DINS, LA fires) |
|---|---|---|
| Stage 1, SAM 3 footprints (`stage1/`, v0.2) | Rewritten and validated | F1 0.60 vs county outlines; recall 0.84–0.91 for buildings > 100 m² |
| Stage 2b damage, retrained on all xView2 events | Done; default in the pipeline | Destroyed F1 0.96 on detected buildings, 0.86 end to end |
| Training-free persistence baseline | Done | Destroyed F1 0.82–0.86 |
| LA product v3 | Done (`results/final_product/`) | 5,748 of 22,024 buildings destroyed |
| Population / exposure (Stage 3) | Planned, not started | — |

Main data on `/media/data/building_instance_tamu/`:
`la_fire_2025/stage2_damage/multidate_full_run_v3/` (product), `la_fire_2025/validation/`
(DINS ground truth, matches, experiment results), `la_fire_2025/postfire_sam3/`,
`la_fire_2025/oracle_footprints/`, `stage2_damage/multidate_oracle/`,
`stage2_training_data/` (xView2 crops, CSVs, runs A_all / B_fire / C_nofire, 47 GB).
Scripts for all of it are in `evaluation/la_fire_validation/` and `evaluation/stage2b_retraining/`.

## Next step that was agreed: population exposure per building

Goal: how many people, and who, lived in the destroyed buildings.

Decisions already made (2026-10-04):
- **Footprints**: SAM 3 v3 footprints for the main analysis; LARIAC outlines as a sensitivity
  check (how much do footprint errors change the exposed-population estimate?).
- **Residential filter**: LA County Assessor parcels (use code, units, year built) as primary;
  Stage 2a predicted type as the "no local data" alternative. Parcels are not downloaded yet;
  fetch only the study region from the county parcel feature service into
  `la_fire_2025/census/` (not into the shared census repo).
- **Allocation weight**: footprint area × stories (stories from the LARIAC `HEIGHT` field), or
  assessor units where available.
- **Population**: 2020 Decennial block `pop20` / `housing20`.
- **Composition**: ACS 2022 5-year block group (B01001 age, B03002 race/ethnicity, B19013 income,
  B25003 tenure, B25077 value); tract DP02/DP05/S1701 for disability and poverty.

Census data comes from the lab's shared, read-only repo `/media/data/us_census` (see its
`CLAUDE.md` and `CATALOG.md`; DuckDB at `processed/duckdb/census.duckdb`, env `gisenv`).
B01001 at block group was added there on 2026-10-04 for this project. If something is missing,
download it into that repo with its ingest scripts first, then copy into this project.
Project-specific outputs never go back into the shared repo.

## Other open items before writing

- Confidence intervals and spatial-block validation for the DINS metrics.
- Compare against another footprint source (e.g. Microsoft building footprints).
- Validate the minor/major classes (only destroyed vs not destroyed is validated).
- Stage 1 misses 18% of destroyed buildings; improving small-building recall is the largest
  remaining gain (e.g. combining the "building" and "house" prompts).
- Full xBD tier3 adds three wildfires, including Woolsey (2018, Los Angeles); needs an
  xview2.org account to download.
- Both training fires (Santa Rosa, SoCal 2017) are Californian; state this when reporting the
  LA results.
