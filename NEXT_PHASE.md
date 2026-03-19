# Next Phase Research Directions
**Planned start**: ~2026-04-01
**Current status**: Stage 1 + Stage 2 pipeline complete, LA fire validation done.
**Baseline metrics**: Stage 1 F1=0.40 (test), Precision=0.68, Recall=0.28, mIoU=0.76

---

## Stage 1 — Improving SAM3 Building Detection

### Problem recap
- Recall is the bottleneck (0.28 overall; drops to 0.03–0.09 on dense urban scenes)
- ~30% of images produce zero predictions
- Shape quality is good when detection fires (mIoU=0.76)

### Direction 1: Prompt Engineering Experiments

**What to try** (in `stage1/sam3_building_identifier/config.py`, `--prompt` flag):

| Prompt | Hypothesis |
|--------|-----------|
| `"building"` | Baseline |
| `"house"` | Biased toward residential — may help suburban recall |
| `"rooftop"` | Top-down satellite view — model may respond better to this framing |
| `"building rooftop"` | Multi-concept prompt |
| `"residential building"` | Scope narrowing |
| `"structure"` | Broader — may catch industrial/commercial missed by "building" |
| `"house rooftop structure"` | Ensemble-style prompt |

**Evaluation protocol**: Run on the 10 xView2 test disaster types; compare F1 per disaster.
Pay special attention to: `mexico-earthquake`, `palu-tsunami` (currently F1 < 0.16).

**Script**: `evaluation/evaluate_predictions.py` already handles this. Just point at new
output dirs per prompt experiment.

**Suggested experiment structure**:
```
/media/data/building_instance_tamu/sam3_prompt_experiments/
  prompt_building/       ← already done (baseline)
  prompt_house/
  prompt_rooftop/
  prompt_building_rooftop/
```

### Direction 2: Threshold & Parameter Tuning

- `--min-size`: Try 50 (more detections) vs 200 (fewer, higher precision)
- `--epsilon`: Try 1.0 (finer polygons) for dense urban scenes
- `--tile-size`: Try 256 or 384 for dense scenes (smaller tiles = more overlap = better coverage)

### Direction 3: Hybrid Stage 1 — SAM3 + Building Footprint Databases

For dense urban areas where SAM3 recall collapses:
- **Microsoft Global Building Footprints** (free, global coverage): covers Mexico City, Palu
- **OpenStreetMap buildings**: dense areas often well-mapped
- **Strategy**: use SAM3 for detection; fall back to OSM/Microsoft footprints when
  SAM3 returns 0 detections or < N buildings for a tile

**Merge logic**: Union of SAM3 + external footprints, deduplicated by IoU overlap.
This could push recall from 0.28 → 0.50+ on dense disaster types.

### Direction 4: Zero-Prediction Image Investigation

276 test images (30%) produce zero SAM3 predictions. Before prompts — diagnose why:
```python
# Quick diagnostic: for zero-prediction images, what do they look like?
# Are they truly empty, dark, cloudy, or dense urban?
# Script: evaluation/diagnose_zero_predictions.py (to be written)
```

---

## Stage 2a — Building-Level Population Estimation

### Problem recap
Current Stage 2a is binary damage classification (damaged/undamaged per building).
The goal is to estimate **how many people were affected** per building — not just whether
it was damaged.

### Data Sources to Combine

| Source | Granularity | What it provides |
|--------|-------------|-----------------|
| **ACS (American Community Survey)** | Census tract (~4000 people) | Avg household size, tenure (owner/renter), age, income |
| **Census block groups** | Sub-tract (~600–3000 people) | Finer demographics |
| **Parcel data** | Building footprint level | Land use (residential/commercial), # units, building type |
| **Building polygons** (SAM3/OSM) | Individual building | Footprint area, geometry |
| **County assessor data** | Parcel level | # units, bedrooms, year built |

### Proposed Pipeline

```
ACS tract data (avg HH size, occupancy rate)
      ↓ areal interpolation to block group
Block group demographics
      ↓ dasymetric mapping using parcel land use
Parcel-level unit counts
      ↓ distribute people across buildings within parcel
Building-level population estimate
      ↓ × damage probability (Stage 2b output)
Affected population estimate per building
```

**Dasymetric mapping**: Redistribute census population using building footprint area
as weights within each block/tract. Buildings with larger footprints (or more floors)
get more of the population.

### Step-by-Step Implementation Plan

#### Step 1: Acquire data
```python
# ACS via Census API (already has Python wrapper: census, cenpy)
# Parcel data: LA County Assessor open data portal
# https://assessor.lacounty.gov/GIS/SHP/PAIS/

# Key ACS variables (Table B25010):
# B25010_001E = average household size
# B25010_002E = owner-occupied avg HH size
# B25010_003E = renter-occupied avg HH size
# B25077_001E = median home value
# B19013_001E = median household income
```

#### Step 2: Spatial join — buildings to census tracts
```python
import geopandas as gpd
buildings = gpd.read_file("building_damage.geojson")     # our Stage 1 output
tracts = gpd.read_file("census_tracts_la.gpkg")
buildings = buildings.sjoin(tracts[['GEOID','avg_hh_size','total_pop']], how='left')
```

#### Step 3: Parcel join — get unit count per building
```python
parcels = gpd.read_file("la_county_parcels.gpkg")
# Each parcel has: UseCode (land use), Units (# residential units), SQFTmain
buildings = buildings.sjoin(parcels[['UseCode','Units','SQFTmain']], how='left')
```

#### Step 4: Building-level population estimation
```python
# For single-family residential (UseCode == 0100):
#   population = avg_household_size (from ACS tract)
# For multi-family (apartment, UseCode == 0400+):
#   population = Units × avg_household_size × occupancy_rate
# For unknown: use footprint area as proxy
#   population = (building_area / avg_unit_size) × avg_household_size

def estimate_population(row, avg_hh_size, occupancy_rate=0.94):
    if row['UseCode'] in SINGLE_FAMILY_CODES:
        return avg_hh_size
    elif row['Units'] > 0:
        return row['Units'] * avg_hh_size * occupancy_rate
    else:
        # fallback: area-based
        return max(1, (row['area_m2'] / 185) * avg_hh_size)  # 185m² ≈ median US unit
```

#### Step 5: Affected population = population × P(damage)
```python
# Use calibrated Stage 2b probabilities (M1b)
buildings['p_damaged'] = buildings['m1b_prob_minor'] + buildings['m1b_prob_major'] + buildings['m1b_prob_destroyed']
buildings['pop_at_risk'] = buildings['pop_estimate'] × buildings['p_damaged']
```

#### Step 6: Aggregate to census tract for validation
```python
tract_affected = buildings.groupby('GEOID').agg(
    n_buildings=('bldg_uid','count'),
    n_damaged=('m1b_damage_class', lambda x: (x>=1).sum()),
    pop_total=('pop_estimate','sum'),
    pop_at_risk=('pop_at_risk','sum'),
).reset_index()
# Compare to FEMA disaster declarations for validation
```

### Key Challenges and Mitigation

| Challenge | Mitigation |
|-----------|-----------|
| Census tracts don't align with building footprints | Dasymetric mapping with parcel weights |
| Parcel data may be incomplete / outdated | Fall back to ACS block-group estimates |
| Mixed-use buildings (commercial + residential) | UseCode filtering + floor ratio |
| Vacation homes / seasonal (low occupancy) | ACS vacancy rate by tract |
| Renters vs owners displaced differently | Use ACS tenure-specific HH size |

### Data Files to Acquire Before Starting
- [ ] LA County Parcel data (Assessor portal, free SHP download)
- [ ] ACS 5-Year Estimates 2023 for LA County (Census API or data.census.gov)
- [ ] Census tract / block group shapefiles for LA County (Census TIGER/Line)
- [ ] Optional: CoreLogic or Zillow data for unit counts (more accurate than assessor)

### Output Schema Extension
Add to `building_damage_all_cells.csv`:
```
pop_estimate          float   # estimated residents in building
pop_estimate_source   str     # "parcel_units" | "area_proxy" | "acs_tract"
pop_at_risk           float   # pop_estimate × P(damage ≥ minor)
acs_tract_geoid       str     # which census tract
acs_avg_hh_size       float   # from ACS
parcel_use_code       str     # land use classification
parcel_units          int     # number of units (0 if unknown)
```

---

## Suggested Research Timeline (Starting ~2026-04-01)

| Week | Task |
|------|------|
| Week 1 | Prompt experiments (house, rooftop, structure) — rerun SAM3, evaluate |
| Week 1 | Acquire LA County parcel + ACS data; spatial join to existing buildings |
| Week 2 | Analyze prompt results; pick best prompt or ensemble strategy |
| Week 2 | Implement dasymetric population estimation; validate against census |
| Week 2 | Generate affected-population maps for LA fire cells |

---

## Code Entry Points

| Task | File |
|------|------|
| Run SAM3 with new prompt | `conda run -n geoai_sam python -m sam3_building_identifier --prompt "house" ...` |
| Evaluate results | `python evaluation/evaluate_predictions.py --split test` |
| Generate overlays | `python pipeline/scripts/generate_presentation_overlays.py` |
| Full LA fire pipeline | `python pipeline/run_la_fire_batch.py` |
| Data lives at | `/media/data/la_fire_2025/` and `/media/data/building_instance_tamu/` |
