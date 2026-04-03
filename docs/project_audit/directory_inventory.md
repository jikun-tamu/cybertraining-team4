# Directory Inventory

**Date**: 2026-04-01

---

## Main Project: `/media/gisense/xihan/250812_tamu_cybertraining_team4/`

### Root Files
| File | Size | Purpose |
|------|------|---------|
| `README.md` | 5.1 KB | Project overview |
| `CLAUDE.md` | 5.3 KB | Claude Code guidance |
| `NEXT_PHASE.md` | 9.0 KB | Phase planning (prompt experiments + population estimation) |
| `pyproject.toml` | 539 B | Python package metadata |
| `environment.yml` | 368 B | Conda environment definition |
| `.gitignore` | 1.9 KB | Git ignore rules |
| `II_package.zip` | **1 GB** | Original collaborator delivery (archived) |

---

### `stage1/` — Production Stage 1 (7.3 MB) [ACTIVE]
```
stage1/
├── README.md
├── setup.py                              Editable install config
├── run_all_geotiffs.py                   GeoTIFF batch runner
├── run_on_geotiff.py                     Single GeoTIFF runner
├── sam3_building_identifier/
│   ├── __init__.py
│   ├── __main__.py                       CLI entry point
│   ├── config.py                         PipelineConfig dataclass
│   ├── model.py                          SAM3Model wrapper
│   ├── pipeline.py                       Main batch pipeline
│   ├── mask_to_polygon.py                Vectorization
│   └── utils.py                          Utilities
├── tests/
│   └── smoke_test.py                     Quick GPU validation
└── notebooks/
    └── eval_and_viz.ipynb                Evaluation visualization
```

---

### `pipeline/` — Combined Stage 1+2 Pipeline (1.4 GB) [ACTIVE]
```
pipeline/
├── README.md                             (7.8 KB) Pipeline documentation
├── requirements.txt                      (8.1 KB) Dependencies
├── environment.yml                       (8.9 KB) Conda env
├── run_pipeline.sh                       Bash wrapper
├── run_la_fire_batch.py                  LA fire batch runner (legacy)
│
├── scripts/                              (18 Python files)
│   ├── run_full_pipeline_launcher.py     Multi-GPU parallel batch runner
│   ├── run_multidate_experiment.py       Per-cell multi-date orchestrator
│   ├── run_instance_impact_driver.py     Single-pair end-to-end
│   ├── generate_shared_instance_subimages.py  Stage 1→2 bridge
│   ├── generate_post_crops_for_date.py   Per-date crop generation
│   ├── infer_stage2a.py                  Building type + population
│   ├── infer_stage2_ensemble.py          Damage ensemble inference
│   ├── aggregate_multidate_predictions.py  Multi-date aggregation (M1/M1b/M2/M3)
│   ├── build_combined_dataset.py         Geospatial integration
│   ├── build_stage2a_infer_csv.py        Stage 2a input preparation
│   ├── generate_maps.py                  Final map generation
│   ├── generate_presentation_overlays.py Visualization overlays
│   ├── present_instance_results.py       Result merging + risk scoring
│   ├── quality_filter.py                 Image quality assessment
│   ├── re_aggregate_all.py               Batch re-aggregation
│   ├── train_stage2.py                   (53 KB) Production training
│   ├── visualize_stage2_overlays.py      Instance-level visualization
│   └── build_combined_dataset.py         Georeferenced output builder
│
├── models/
│   ├── stage2a/
│   │   └── stage2a_best_model.pt         EfficientNet-B0 (type + population)
│   └── stage2b/
│       ├── inference0.7273.pt            Seed 2025 (weight 4)
│       ├── inference0.7066_seed9999.pt   Seed 9999 (weight 2)
│       └── inference0.7034_seed7777.pt   Seed 7777 (weight 3)
│
├── configs/stage2b/
│   ├── run019_seed2025_train_config.json
│   ├── seed7777_train_config.json
│   └── seed9999_train_config.json
│
├── calibration/                          Temperature scaling artifacts
│   ├── calibration_run019_r48/
│   ├── calibration_seed7777_r48/
│   └── calibration_seed9999_r48/
│
├── stage1/
│   └── SAM3_Final_20260226/              (REDUNDANT copy of exploration/SAM3_Final)
│
├── outputs/
│   ├── driver_runs/                      Single-pair run outputs
│   └── multidate_experiment/             Multi-date experiment outputs
│
├── docs/                                 Pipeline-specific docs
└── example_image_pair/                   Demo input pair
```

---

### `evaluation/` — xView2 Benchmark (548 KB) [ACTIVE]
```
evaluation/
├── README.md
├── requirements.txt
├── environment.yml
├── evaluate_predictions.py               (18 KB) SAM3 vs ground truth evaluation
├── run_prompt_experiments.py             (18 KB) Prompt ablation framework
├── results/
│   ├── sam3_eval/                        Benchmark results (train + test)
│   └── prompt_experiments/               Prompt experiment results
└── docs/
```

---

### `src/cybertraining_team4/` — Original Package (906 MB) [LEGACY]
```
src/cybertraining_team4/
├── __init__.py
├── stage1_train.py                       Legacy Stage 1 training script
├── train_xihan_gpu_testing.py            GPU testing
├── run_custom_pipeline.py                Early pipeline runner
├── process_chips_600m.py                 Chip creation from Maxar imagery
├── prune_bad_pre_cells.py                Bad cell removal
│
└── 2-stage package/                      (COLLABORATOR DELIVERY)
    ├── corrected_building_segmentation_model.pth  (170 MB)
    ├── inference_vector_calibrated.pt             Calibrated checkpoint
    ├── iguide-gpu-environment.yml                 Environment spec
    ├── iguide-gpu-pip.txt                         Pip requirements
    ├── 2-stage package.zip                        Nested archive
    │
    └── scripts/
        ├── stage1_train.py               Mask R-CNN Stage 1 training
        ├── stage1_infer_tile_masks.py     Mask R-CNN inference
        ├── train_stage2.py               Legacy Stage 2 training (270 lines)
        ├── preprocess_stage2_crops.py     Crop + mask generation
        ├── build_stage2_index.py          Index from labeled data
        ├── build_stage2_index_from_pred.py  Index from predictions
        ├── polygonize_stage1_masks.py     Mask → polygon conversion
        ├── calibrate_temperature_stage2.py     Scalar calibration
        ├── calibrate_vector_temperature_stage2.py  Vector calibration
        ├── eval_full_stage2.py            Full evaluation
        ├── eval_ece_stage2.py             ECE evaluation
        ├── eval_nll_stage2.py             NLL evaluation
        ├── forward_check_stage2.py        Forward pass sanity check
        ├── infer_overlay_stage2.py        Inference with overlays
        ├── check_stage2_dataset.py        Dataset validation
        ├── run_full_pipeline.py           Legacy full pipeline
        ├── sweep_train_stage2.sh          Hyperparameter sweep
        │
        └── src/                           MODEL SOURCE CODE
            ├── data/
            │   └── stage2_dataset.py      Dataset class
            └── models/
                ├── siamese_stage2.py      Siamese damage model
                ├── backbone.py            ConvNeXt backbone wrapper
                ├── coral_head.py          CORAL ordinal regression
                └── masked_pool.py         Masked average pooling
```

---

### `exploration/` — Experiments (4.7 GB) [ARCHIVE CANDIDATE]
```
exploration/
├── SAM3_Final/                    Earlier SAM3 pipeline with georef
│   ├── src/sam3_final/            (11 Python modules)
│   ├── scripts/run_sam3_building_infer.py
│   └── notebooks/01_visual_qc_and_eval.ipynb
│
├── SAM3_notebooks/                Original SAM3 notebooks (6 notebooks)
│   ├── sam3_image_segmentation.ipynb
│   ├── sam3_batch_segmentation.ipynb
│   ├── sam3_interactive.ipynb
│   └── 251210_sam3_*_xihan.ipynb  (Xihan's versions)
│
├── PolyWorld/                     Polygon-based building extraction
│   ├── prediction.py, predict_*.py
│   ├── models/backbone.py, matching.py
│   └── notebooks/ (4 notebooks)
│
├── Mask_R-CNN/                    Instance segmentation
│   ├── notebooks/ (2 notebooks)
│   └── buildings_instance/instance_models/  (best_model.pth, etc.)
│
├── GeoAI_QuishengWu/             Building footprint extraction
│   ├── scripts/building_footprint_combined.py
│   ├── models/building_footprints_usa.pth
│   └── notebooks/ (2 notebooks)
│
├── GeoAI_building_segmentation/   Early exploration
│
└── corrected_model/               Early Mask R-CNN checkpoint
    └── corrected_building_segmentation_model.pth
```

---

### Other Directories
```
results/                           (10 MB) [ACTIVE]
├── project_evaluation_report.md   Comprehensive evaluation report
├── sam3_eval/                     SAM3 evaluation results
├── _staging/                      Staging area
└── run_all.log                    (1.9 MB) Processing log

notebooks/                         (18 MB) [ACTIVE]
├── stage1.ipynb                   Stage 1 analysis
├── validation_case.ipynb          Case study validation
├── validation_LA_fire.ipynb       LA fire validation
└── 251209_TestingDataDownload.ipynb  Data download helper

data/                              (735 MB) [KEEP]
├── README.md
└── case_overlays.zip              (734 MB) LA fire overlays archive

tests/                             (empty — tests live in stage1/tests/)
```

---

## Training/Test Data: `/media/data/building_instance_tamu/` (33.2 GB)

```
building_instance_tamu/
├── train/                         (8.1 GB) xView2 training split
│   ├── images/                    5,598 PNG (1024x1024)
│   ├── labels/                    5,598 JSON (WKT polygons + damage)
│   └── targets/                   5,598 binary masks
│
├── test/                          (2.7 GB) xView2 test split
│   ├── images/                    1,866 PNG
│   ├── labels/                    1,866 JSON
│   └── targets/                   1,866 binary masks
│
├── sam3_claude/                   (4.6 GB) SAM3 outputs (default prompt)
│   ├── test/                      933 predictions + 1,866 masks + 933 annotations
│   └── train/                     2,799 predictions + 5,598 masks + 2,799 annotations
│
├── sam3_prompt_experiments/       (3.0 GB) Prompt ablations on test set
│   ├── house/                     24,999 buildings detected
│   ├── rooftop/                   3,699 buildings
│   ├── structure/                 9,760 buildings
│   └── building_rooftop/          5,085 buildings
│
├── Mask_R-CNN_BuildingInstance_Train/  (4.9 GB) [ARCHIVE CANDIDATE]
│   ├── images/                    2,283 PNG
│   ├── labels/                    2,283 PNG (16-bit instance masks)
│   ├── models/                    (1.4 GB) best_model.pth, etc.
│   └── test/                      Evaluation outputs
│
├── PolyWorld/                     (329 MB) [ARCHIVE CANDIDATE]
│   ├── finetuned_v2/              (150 MB) Model weights
│   └── test_*/                    COCO-format prediction JSONs
│
├── xview2_challenge_train.tar.gz  (7.9 GB) [DELETE CANDIDATE if extracted]
└── xview2_challenge_test.tar.gz   (2.6 GB) [DELETE CANDIDATE if extracted]
```

---

## LA Fire Data: `/media/data/la_fire_2025/` (69 GB)

```
la_fire_2025/
├── raw/                           (27 GB) 538 Maxar GeoTIFFs
│   └── maxar_opendata/events/.../ard/11/{tile}/{date}/*.tif
│
├── chips/                         (18 GB) 295 cells, 2,011 GeoTIFFs
│   └── cell_NNNNN/{pre,post}/*.tif
│
├── grids/                         (806 MB) Grid definitions + manifests
│   ├── *.gpkg, *.csv, *.html     Geospatial data
│   └── case_overlays/             8,686 prediction overlay PNGs
│
├── stage1_sam3/                   (4.0 GB) Building detection
│   ├── predictions/               2,011 JSON files
│   ├── raster_masks/              1,258 GeoTIFFs
│   ├── annotations/               629 PNGs
│   └── run_summary.json           21,154 buildings total
│
├── stage2_damage/                 (14 GB) Damage classification
│   └── multidate_full_run/
│       ├── building_damage_all_cells.csv
│       ├── experiment_summary.json
│       └── cell_NNNNN/            298 cell directories
│
├── damage_overlays_v1/            (3.9 GB) Early overlays
├── overlays_v2/                   (843 MB) Refined overlays
├── panels/                        (1.2 GB) Cell visualization panels
└── final_maps/                    (246 MB) Final analysis output
    └── maps/*.{geojson,gpkg,png,md}
```

---

## Model Checkpoints Inventory

| File | Size | Status | Location |
|------|------|--------|----------|
| `pipeline/models/stage2a/stage2a_best_model.pt` | ~50 MB | **ACTIVE** | Stage 2a inference |
| `pipeline/models/stage2b/inference0.7273.pt` | ~50 MB | **ACTIVE** | Stage 2b ensemble |
| `pipeline/models/stage2b/inference0.7066_seed9999.pt` | ~50 MB | **ACTIVE** | Stage 2b ensemble |
| `pipeline/models/stage2b/inference0.7034_seed7777.pt` | ~50 MB | **ACTIVE** | Stage 2b ensemble |
| `src/.../inference_vector_calibrated.pt` | ~50 MB | Reproducibility | Calibrated checkpoint |
| `src/.../corrected_building_segmentation_model.pth` | 170 MB | Archive | Old Mask R-CNN |
| `exploration/corrected_model/*.pth` | 170 MB | Archive | Duplicate of above |
| `exploration/Mask_R-CNN/.../best_model.pth` | 169 MB | Archive | Mask R-CNN |
| `exploration/Mask_R-CNN/.../final_model.pth` | 169 MB | Archive | Mask R-CNN |
| `exploration/GeoAI_QuishengWu/models/*.pth` | ~200 MB | Archive | GeoAI model |
| `/media/data/.../Mask_R-CNN_.../models/best_checkpoint.pth` | 335 MB | Archive | Full checkpoint |
| `/media/data/.../PolyWorld/finetuned_v2/` | 150 MB | Archive | PolyWorld weights |

**Total active checkpoints**: ~200 MB
**Total archivable checkpoints**: ~1.5 GB (in project) + ~1.9 GB (in data)
