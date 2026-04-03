# Project Overview — Disaster Impact Assessment Pipeline

**Date**: 2026-04-01
**Scope**: Full pipeline audit and technical documentation

---

## What This Project Does

This project assesses **building-level disaster damage** from satellite imagery. It combines:

1. **Stage 1** — Automatic building segmentation from satellite images using SAM3 (Segment Anything Model 3)
2. **Stage 2** — Pre/post-disaster image comparison to classify damage severity per building

The pipeline was developed for and validated against:
- **xView2 dataset** — multi-disaster satellite imagery benchmark (training/evaluation)
- **2025 Los Angeles wildfires** — real-world case study using Maxar open data

---

## Pipeline Architecture

```
                        STAGE 1                              STAGE 2
                   Building Detection               Damage Classification

  Satellite     ┌──────────────────┐    Building   ┌──────────────────────┐   Damage
  Imagery  ───> │  SAM3 (samgeo)   │ ──polygons──> │  Siamese ConvNeXt    │ ──class──>  Final
  (pre/post)    │  Text prompt:    │    + masks     │  + CORAL ordinal     │   per       Report
                │  "building"      │                │  regression          │   building
                └──────────────────┘                └──────────────────────┘
                                                          │
                                                    Also includes:
                                                    Stage 2a: building type
                                                              + population
                                                    Stage 2b: damage class
                                                              + uncertainty
```

### Stage 1: Building Segmentation

- **Model**: SAM3 via `samgeo` (segment-geospatial) library
- **Approach**: Zero-shot text-prompted segmentation — no training required
- **Input**: Single satellite image (pre-disaster preferred)
- **Output**: Per-building instance masks, polygons (WKT), bounding boxes, confidence scores
- **Key parameters**: prompt="building", min_size=100px, polygon_epsilon=2.0

### Stage 2a: Building Type & Population (auxiliary)

- **Model**: EfficientNet-B0 with 4-channel input (RGB + binary mask)
- **Input**: 256x256 pre-disaster crop + building mask
- **Output**: Building type (5 classes: residential_small, residential_multi, commercial, institutional, other) + population estimate (log-scale regression)

### Stage 2b: Damage Classification

- **Model**: Siamese ConvNeXt-Tiny with CORAL ordinal regression head
- **Input**: 256x256 pre-disaster crop, post-disaster crop, building mask (M), context ring mask (R)
- **Output**: Damage class probabilities (4 classes: no-damage, minor, major, destroyed)
- **Ensemble**: 3 checkpoints (seeds 2025, 7777, 9999) with weighted averaging (4:3:2)
- **Calibration**: Temperature scaling + vector temperature on CORAL logits

### Multi-Date Aggregation

For real-world scenarios with multiple post-disaster captures:
- Quality filtering at tile-level (brightness, nodata fraction, texture) and crop-level (coverage)
- Four aggregation methods:
  - **M1**: Probability averaging across all quality-passing dates
  - **M2**: Majority vote with M1 tiebreak
  - **M3**: Quality-filtered probability averaging (tile + crop filtering)
  - **M1b**: Coverage-aware averaging (strictest — preferred for final results)

---

## Data Flow Between Stages

```
Stage 1 output (per image):
  predictions/{stem}_prediction.json
    └── instances[].polygon (WKT pixel coords)
    └── instances[].bbox_xyxy
    └── instances[].confidence

        │
        ▼  generate_shared_instance_subimages.py

Shared artifacts (per building):
  crops_pre/{tile}__{uid}.png       (256x256 RGB crop)
  crops_post/{tile}__{uid}.png      (256x256 RGB crop)
  masks_M/{tile}__{uid}.png         (binary building mask)
  masks_R/{tile}__{uid}.png         (binary 48px ring mask)
  shared_instance_samples.csv       (index with geometry + paths)

        │
        ▼  infer_stage2a.py / infer_stage2_ensemble.py

Stage 2 output (per building):
  stage2a_predictions.csv           (type + population)
  stage2b_ensemble.jsonl            (damage class + uncertainty)
```

---

## Datasets

### xView2 (Training & Evaluation)

- **Location**: `/media/data/building_instance_tamu/`
- **Split**: 2,799 image pairs (train) / 933 image pairs (test)
- **Format**: 1024x1024 PNG, xView2 naming (`{base}_pre_disaster.png`, `{base}_post_disaster.png`)
- **Labels**: JSON with WKT polygons + damage subtypes (no-damage, minor, major, destroyed)
- **Disaster types**: 10 types including hurricane-florence, socal-fire, guatemala-volcano, mexico-earthquake
- **Usage**:
  - Stage 1: SAM3 evaluated against polygon ground truth (Precision 0.68, Recall 0.28, F1 0.40)
  - Stage 2: Siamese model trained on building crops extracted from these pairs

### LA Fire 2025 (Real-World Case Study)

- **Location**: `/media/data/la_fire_2025/`
- **Source**: Maxar open data (ARD visual products, UTM Zone 11)
- **Structure**: 295 grid cells at 600m resolution, 1 pre + 4-13 post dates per cell
- **Key difference from xView2**: Multi-date temporal coverage, GeoTIFF format, real-world nodata/quality issues
- **Results**: 10,607 buildings across 120 cells; 85% no damage, 10% minor, 0.2% major (M1b method)
- **Critical finding**: 53% temporal instability (conflicting labels across dates)

---

## How xView2 Relates to LA Fire

| Aspect | xView2 (Training) | LA Fire (Real-World) |
|--------|-------------------|---------------------|
| Format | PNG 1024x1024 | GeoTIFF 600m cells |
| Dates | 1 pre + 1 post per tile | 1 pre + 4-13 post per cell |
| Labels | Ground truth available | No ground truth |
| Disasters | Multi-type (earthquake, flood, etc.) | Wildfire only |
| Quality | Clean, curated | Nodata, cloud, haze issues |
| Stage 1 | Evaluated on xView2 labels | Run zero-shot on Maxar imagery |
| Stage 2 | Trained on xView2 crops | Applied zero-shot (no wildfire training) |

**Key limitation**: Stage 2 was trained on earthquake/flood data. It has never seen wildfire damage patterns, which may explain low major/destroyed predictions in LA fire results.

---

## Key Results

### Stage 1 (SAM3 on xView2)
- Precision: 0.682, Recall: 0.284, F1: 0.401, Mean IoU: 0.759
- Shape quality is excellent when buildings are detected; main weakness is detection completeness
- Best on hurricane-florence (F1=0.75), worst on mexico-earthquake (F1=0.057, dense urban)

### Stage 2 (Damage on xView2)
- Best single model: ~70.7% macro F1 (quadratic weighted kappa)
- 3-checkpoint ensemble with temperature calibration improves reliability
- CORAL ordinal regression constrains predictions to respect damage ordering

### Combined Pipeline (LA Fire)
- 10,607 buildings assessed across 120/295 grid cells
- M1b damage distribution: 85.7% no damage, 10.1% minor, 0.2% major, 0% destroyed
- 53% prediction instability across dates — significant methodological concern

---

## Repository Structure (Current)

```
250812_tamu_cybertraining_team4/
├── stage1/                 ← Production Stage 1 (SAM3 package)
├── pipeline/               ← Combined Stage 1+2 pipeline (from collaborator)
├── evaluation/             ← xView2 benchmark evaluation
├── src/cybertraining_team4/  ← Original package + archived 2-stage code
├── exploration/            ← Legacy experiments (Mask R-CNN, PolyWorld, etc.)
├── results/                ← Aggregated reports
├── notebooks/              ← Validation notebooks
├── data/                   ← LA fire overlay archive
└── docs/                   ← This documentation
```

See `directory_inventory.md` for detailed file-level inventory.
See `structure_cleanup_plan.md` for proposed reorganization.
