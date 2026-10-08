# Working Together on This Repository

The LA fires 2025 paper is led by Jinyu Zhou (TAMU). Xihan Yao (UT Austin, GISense Lab) runs
the models on the GISense lab server. Everything in this repository and everything listed below
can be used for the paper.

## Start here

| File | What it is |
|---|---|
| `reports/results_summary_2026-10.md` | One page with every number worth citing |
| `reports/sam3_audit_2026-09.md` | Full record: changes, experiments, caveats (§1–10) |
| `reports/handoff_2026-10.md` | Status, open items, the population-exposure plan |
| `reports/literature_review_2026-10.md` | 42 references and positioning options |
| `results/final_product/` | LA product v3: 22,024 buildings with damage class (CSV + GPKG) |
| `evaluation/la_fire_validation/data/` | LA ground truth (LARIAC + CAL FIRE DINS), matched pairs |

## What is in git and what is not

**In git**: all code, reports, result JSONs, the LA product and the LA ground truth (about 100 MB).

**On the lab server only** (too large for git; shared on request through OneDrive):

| Data | Size | Notes |
|---|---:|---|
| LA pre/post Maxar image chips, 295 cells × all dates | 18.5 GB | Already on OneDrive (`chips_600m`) |
| LA run directory v3 (crops, per-date predictions, masks) | 17 GB | `multidate_full_run_v3/` |
| SAM 3 detections on the post-fire images (persistence experiment) | 283 MB | |
| LARIAC outlines shifted onto the Maxar grid (oracle footprints) | 7 MB | |
| SAM 3 predictions, xView2 test split | 56 MB | JSON per image |
| Tiling experiments A–L on xView2 | 757 MB | |
| Stage 2b model A, all hazards (current default) | 355 MB | `.pt`, git-ignored |
| Stage 2b models B (fire only) and C (all but fire), original flood ensemble | ~1.1 GB | |
| Stage 2a model (building type + population, original) | 19 MB | |
| xView2 training crops for Stage 2b (10 events) | 51 GB | |

Ask for anything in this table and it will be shared through OneDrive.

## Lab server resources

- 2 × NVIDIA RTX A6000 (48 GB each), shared with other lab members
- 32 CPU cores, 500 GB RAM, about 2.8 TB free on the data disk
- Conda environments: `geoai_sam` (SAM 3 via samgeo, Stage 1 and 2) and `gisenv` (GIS, census)
- US Census data already on the server: 2020 blocks (`pop20`, `housing20`), ACS 5-year 2020 and
  2022 at block group and tract, CDC PLACES, SVI
- Typical run times: Stage 1 about 8 s per 1024 px image; the full LA run (295 cells, Stage 1 + 2)
  about 6 h on one GPU

## Asking for a run

1. Push your code to a branch (for example `stage2a-jinyu`) with a short `RUN.md`:
   the command to run, the inputs it expects, the environment (`requirements.txt` or
   `environment.yml`) and what output you want back.
2. Large model weights do not go in git; share them through OneDrive or a download link.
3. Email Xihan. The run happens on the lab server; results come back in the same branch
   (small files) or through OneDrive (large files).

Inputs already available for LA: pre- and post-fire image chips (GeoTIFF, Maxar, UTM 11N),
Stage 1 footprints (GPKG and per-image JSON), 256 px pre/post crops with mask channels per
building, and the DINS ground truth.

## Stage 2A (building type and population)

To validate a building-type model on LA we need ground-truth use types. LARIAC outlines in the
ground truth carry the assessor parcel number (`AIN`), so LA County Assessor use codes
(residential / commercial / institutional, number of units) can be joined per building. The
parcels are not downloaded yet; that is the next data step once the Stage 2A code is in.
