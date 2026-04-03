# LA Fire 2025 — Real-World Pipeline Documentation

**Date**: 2026-04-01

---

## How the LA Fire Analysis Differs from Training/Evaluation

The LA fire case study applies the xView2-trained pipeline to real-world Maxar satellite imagery of the January 2025 Los Angeles wildfires. This introduces significant differences from the controlled xView2 benchmark environment.

---

## Data Source Differences

| Aspect | xView2 (Training) | LA Fire (Real-World) |
|--------|-------------------|---------------------|
| **Provider** | DigitalGlobe/Maxar (curated) | Maxar Open Data (raw ARD) |
| **Format** | PNG, 1024x1024, 8-bit RGB | GeoTIFF, variable size, 8-bit RGB |
| **CRS** | Pixel coordinates only | UTM Zone 11 (EPSG:32611) |
| **Resolution** | ~0.3m GSD (WorldView-3) | ~0.3m GSD (WorldView-3, visual ARD) |
| **Coverage** | Single tile = single scene | Multiple overlapping scenes per area |
| **Temporal** | 1 pre + 1 post per tile | 1 pre + 4-13 post dates per cell |
| **Quality** | Curated, minimal nodata | Nodata gaps, cloud, haze, partial coverage |
| **Disasters** | 10 types (earthquake, flood, etc.) | Wildfire only (unseen in training) |
| **Labels** | Ground truth available | No ground truth |

### Critical Implication
Stage 2 has **never seen wildfire damage** during training. The 85% "no damage" result may reflect genuine minimal damage, model inability to recognize fire damage, or both. This is the single most important caveat for the LA fire results.

---

## Image Alignment / Scale / Tiling

### Chip Creation (`process_chips_600m.py`)

Raw Maxar imagery is converted to a regular grid:

1. **Grid definition**: 600m x 600m cells in UTM Zone 11 covering the fire perimeter
2. **Source indexing**: All 538 Maxar GeoTIFFs indexed by geographic bounds and CRS
3. **Pre/post classification**: Images classified by:
   - Directory name containing `"pre-event"` or `"post-event"`
   - Date parsing from filename (threshold: 2025-01-02)
4. **Mosaicking**: Multiple overlapping source rasters mosaicked per cell
5. **Clipping**: Mosaic clipped to exact 600m cell bounds
6. **Output**: LZW-compressed GeoTIFF with 512px tile blocks

```
Output structure:
chips/cell_00045/
  pre/cell_00045_pre.tif           (single pre-disaster baseline)
  post/cell_00045_post_20250109.tif
  post/cell_00045_post_20250113.tif
  post/cell_00045_post_20250114.tif
  ...                               (4-13 post dates per cell)
```

### Key Differences from xView2 Tiling
- xView2 tiles are exactly 1024x1024px at a fixed GSD
- LA fire chips are 600m in ground coordinates (pixel count depends on GSD)
- LA fire requires mosaicking from multiple source scenes
- Geographic registration via GeoTIFF affine transforms

---

## Ad Hoc Preprocessing

### 1. Bad Pre-Cell Pruning (`prune_bad_pre_cells.py`)

Detects and removes cells where the pre-disaster image is blank/nodata:
- Checks max pixel value against threshold
- If blank: removes entire cell directory (pre + all post)
- Also cleans associated panels, grid rows, and CSV entries

### 2. Quality Filtering at Multiple Levels

**Tile-level** (`quality_filter.py`):
```python
min_mean_brightness = 15.0      # Reject dark/empty tiles
max_zero_fraction = 0.60        # Reject >60% nodata
min_spatial_std = 3.0           # Reject featureless/hazy
```

**Crop-level** (per building instance):
```python
max_zero_fraction = 0.50        # Reject crops with >50% black pixels
```

**Results**: 47 of 656 post-date images (7%) rejected by tile-level quality filter.

### 3. Path Rewriting (`run_la_fire_batch.py`)

The batch runner fixes stale absolute paths from earlier project structure:
```python
# Old path (from when project was at different location)
/media/gisense/xihan/250812_CyberTraining_Team4/data/chips_600m
# Fixed to
/media/data/la_fire_2025/chips
```

---

## Thresholds and Heuristics

### Stage 1 Parameters (Modified from xView2 Defaults)

| Parameter | xView2 Default | LA Fire Setting | Reason |
|-----------|---------------|-----------------|--------|
| `min_size` | 100 px | 30 px | LA fire chips may have smaller buildings at edge |
| `tile_size` | N/A | 512 px | SAM3 internal tiling for GeoTIFF processing |
| `overlap` | N/A | 64 px | Tile overlap to avoid edge artifacts |

### Stage 2b Ensemble Weights
```python
model_weights = [4, 3, 2]  # seed 2025, seed 9999, seed 7777
# Weighted average on CORAL cumulative logits before probability conversion
```

### Multi-Date Aggregation Thresholds
- **Instability**: Building classified as unstable if >=2 different damage classes across dates
- **Quality gate**: M1b requires both tile-level AND crop-level quality pass
- **NOT_IDENTIFIABLE**: Returns -1 when no usable date exists for a building

---

## Custom Postprocessing

### Multi-Date Aggregation (`aggregate_multidate_predictions.py`)

This is the most significant LA-fire-specific addition. xView2 has only 1 post date per tile; LA fire has 4-13.

Four methods implemented:

| Method | Tile Filter | Crop Filter | Logic |
|--------|-------------|-------------|-------|
| **M1** | Yes | No | Average calibrated probs across quality-passing dates → argmax |
| **M2** | Yes | No | Majority vote across dates; M1 probs for tiebreak |
| **M3** | Yes | Yes | Average probs only from high-quality crops; fallback to M1 |
| **M1b** | Yes | Yes | Average probs from coverage-valid dates; no fallback (returns -1) |

**M1b is the recommended method** — it avoids false "destroyed" labels that M1 produces when nodata buildings are misclassified.

### Geospatial Integration (`build_combined_dataset.py`)

Converts pixel-space results to georeferenced formats:
1. Reads GeoTIFF affine transforms from pre-disaster chips
2. Transforms pixel polygons → UTM coordinates → WGS84
3. Computes centroids in both CRS
4. Outputs: CSV (flat), GeoJSON (WGS84), GeoPackage (UTM)

### Map Generation (`generate_maps.py`)

Produces 5 map types at 3 zoom levels (full, east cluster, west cluster):
1. **Building damage map**: Colored polygons by damage class
2. **Damage density**: Hexbin heatmap of building density + scatter of damaged buildings
3. **Per-cell damage percentage**: Circle markers sized by building count, colored by % damaged
4. **Uncertainty map**: Dots colored by prediction stability (red=unstable, green=stable)
5. **Highlighted areas**: Annotated clusters by damage/uncertainty category

### Risk Scoring (`present_instance_results.py`)

Computes `driver_exposure_damage_score = population * expected_severity` per building, combining Stage 2a population estimates with Stage 2b damage predictions.

---

## Fragile Assumptions

### 1. No Wildfire Training Data (CRITICAL)
Stage 2 was trained on earthquake, flood, tsunami, volcano, and hurricane damage. Wildfire damage (charred structures, ash coverage, partial burns) presents fundamentally different visual signatures. The very low major/destroyed predictions (0.2%/0%) may be a model limitation, not reality.

### 2. Pre/Post Registration Assumed Perfect
The pipeline assumes pre and post images of the same cell are perfectly aligned. Maxar ARD products are generally well-registered, but sub-pixel shifts between dates can introduce noise, especially for small buildings.

### 3. Single Pre-Disaster Baseline
Each cell has exactly one pre-disaster image. If that image has quality issues (cloud, haze, partial coverage), it propagates to all post-date comparisons.

### 4. Crop Geometry Fixed Across Dates
Building centroids and crop windows are computed once from the pre-disaster image. If a building is partially destroyed and its footprint changes, the crop may not optimally frame the post-disaster state.

### 5. 53% Temporal Instability
Over half of buildings receive conflicting damage labels across different post-disaster dates. This is a fundamental reliability concern that is not resolved — only documented.

**Possible causes**:
- Varying observation angles between dates
- Atmospheric/illumination differences
- Actual temporal changes (cleanup, secondary damage)
- Model sensitivity to input variation (not robust)

### 6. Quality Filter Thresholds Are Heuristic
The brightness, nodata, and texture thresholds were hand-tuned, not validated against ground truth. They may be too aggressive (rejecting usable data) or too permissive (allowing bad data through).

---

## Outputs Produced in `/media/data/la_fire_2025`

### Directory Structure and Contents

```
la_fire_2025/                                    (69 GB total)
├── raw/                          (27 GB)  538 Maxar GeoTIFFs (source imagery)
│   └── maxar_opendata/events/WildFires-LosAngeles-Jan-2025/
│       └── ard/11/{tile_id}/{date}/{scene}-visual.tif
│
├── chips/                        (18 GB)  295 cells × (1 pre + N post) = 2,011 GeoTIFFs
│   └── cell_NNNNN/{pre,post}/*.tif
│
├── grids/                        (806 MB) Grid definitions + manifests
│   ├── chips_600m_manifest.csv            Cell-to-source mapping
│   ├── *.gpkg                             Grid boundaries (UTM11)
│   └── case_overlays/                     8,686 prediction overlay PNGs
│
├── stage1_sam3/                  (4.0 GB) Building detection results
│   ├── run_summary.json                   21,154 buildings across 2,011 images
│   ├── predictions/              (2,011)  Per-image JSON with polygons
│   ├── raster_masks/             (1,258)  Binary masks + confidence TIFFs
│   ├── annotations/              (629)    Visualization overlays
│   └── sam3_overlays/            (629)    Alternative overlays
│
├── stage2_damage/                (14 GB)  Damage classification results
│   └── multidate_full_run/
│       ├── building_damage_all_cells.csv  Complete flat table
│       ├── experiment_summary.json        Pipeline execution summary
│       └── cell_NNNNN/                    (298 cell directories)
│           ├── pair_inputs/               Symlinked pre/post images
│           ├── shared_base/               Crops, masks, instance CSV
│           └── stage1/                    Per-cell Stage 1 outputs
│
├── damage_overlays_v1/           (3.9 GB) Early damage visualization (629 PNGs)
├── overlays_v2/                  (843 MB) Refined overlays (119 PNGs)
├── panels/                       (1.2 GB) Cell visualization panels (295 PNGs)
└── final_maps/                   (246 MB) Final analysis output
    └── maps/
        ├── building_damage.geojson        Georeferenced damage polygons
        ├── building_damage.gpkg           GeoPackage (UTM11)
        ├── building_damage_map_*.png      Damage maps (3 zoom levels)
        ├── damage_density_map_*.png       Density maps
        ├── per_cell_damage_pct_*.png      Per-cell charts
        ├── uncertainty_map_*.png          Stability maps
        └── damage_summary.md              Summary report
```

### Key Output Files

| File | Purpose |
|------|---------|
| `building_damage_all_cells.csv` | Master table: every building with damage class, probabilities, quality flags |
| `building_damage.geojson` | Georeferenced polygons in WGS84 for GIS |
| `building_damage.gpkg` | Same in UTM11 GeoPackage |
| `damage_summary.md` | Human-readable summary with statistics |
| `experiment_summary.json` | Machine-readable pipeline execution metadata |

### Summary Statistics

| Metric | Value |
|--------|-------|
| Grid cells | 295 total, 120 with buildings |
| Buildings detected (Stage 1) | 21,154 across all images |
| Buildings assessed (Stage 2) | 10,607 unique buildings |
| Post-date images evaluated | 656 |
| Dates rejected by quality | 47 (7%) |
| Damage: no damage (M1b) | 85.7% |
| Damage: minor (M1b) | 10.1% |
| Damage: major (M1b) | 0.2% |
| Damage: destroyed (M1b) | 0% |
| Damage: unknown (M1b) | 4% |
| Temporal instability | 53% of buildings |

---

## Workflow Diagram

```
Maxar Open Data (538 TIFFs)
        │
        ▼
process_chips_600m.py ──► chips/ (295 cells × N dates)
        │
        ▼
prune_bad_pre_cells.py ──► Remove blank pre-cells
        │
        ▼
run_full_pipeline_launcher.py
  (parallel on 2 GPUs, chunks of 30 cells)
        │
        ▼  For each cell:
        │
        ├── Stage 1 (SAM3): building detection on pre-image
        │     └── stage1/labels/{tile}_prediction.json
        │
        ├── generate_shared_instance_subimages.py
        │     └── shared_base/{crops, masks, CSV}
        │
        ├── Stage 2a: building type + population
        │     └── stage2a_predictions.csv
        │
        ├── For each post date (4-13):
        │     ├── generate_post_crops_for_date.py + quality filter
        │     └── infer_stage2_ensemble.py → stage2b_{date}.jsonl
        │
        └── aggregate_multidate_predictions.py
              └── aggregated_predictions.{csv,jsonl}
        │
        ▼
build_combined_dataset.py
        │
        ▼
generate_maps.py ──► final_maps/
```
