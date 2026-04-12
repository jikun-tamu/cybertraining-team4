# II_package Competition Notebook Audit
**Date**: 2026-04-11  |  **Final submission deadline**: **2026-04-15 (4 days)**  
**Competition**: I-GUIDE Spatial AI Challenge 2025-26  
**Notebook**: `II_package/II_package/present_instance.ipynb`

---

## 1. What the Competition Notebook Does Right Now

The notebook is a **single pre/post image pair demo** running the full 3-stage pipeline end-to-end on one bundled example (Nepal flooding, xView2 tile `00000408`).

### Pipeline walkthrough

| Phase | Cells | What happens |
|---|---|---|
| Setup / deps | 4–10 | Creates `.nb_vendor/` overlay, detects and installs missing `transformers`, `samgeo`, `timm`. Prompts for `HF_TOKEN` via `getpass`. |
| Configuration | 12–18 | Auto-detects `PACKAGE_ROOT`. Sets `PRE_IMAGE`, `POST_IMAGE`, `RUN_ID`, `PYTHON_BIN`, `CUDA_VISIBLE_DEVICES`. Pulls Git LFS model files if missing. Imports torch/numpy/PIL; detects GPU. |
| Stage 1 — SAM3 | 20–23 | Runs `stage1/SAM3_Final_20260226/scripts/run_sam3_building_infer.py` via subprocess. Outputs per-instance JSON (bbox, polygon, confidence). Text prompt = "building", min-size = 100 px, tiling disabled. |
| Shared artifacts | 25–28 | Generates per-instance 256×256 crops (pre + post) and binary masks (M = building, R = ring buffer). One row per building in `shared_instance_samples.csv`. |
| Stage 2a | 30–33 | Runs EfficientNet-B0 (4-channel: RGB + mask) to predict **building type** (5 classes) and **population estimate** (regression). |
| Stage 2b | 35–38 | Runs **3-model Siamese ConvNeXt ensemble** (weights 4:3:2, temperature calibration). Outputs damage class (0–3) + calibrated confidence, entropy, variance-weighted severity per instance. |
| Synthesis | 40–42 | Joins all three stages by `instance_id`. Identifies top-30 uncertain instances. Generates PNG overlay per instance (pre/post crops, polygon, metrics). |
| Reporting | 44–53 | Prints summary statistics. Renders 3-panel distribution chart (damage classes, confidence, population). Displays top-20 uncertain instances table. Shows 6 sample overlay PNGs. |

### Final outputs

| File | Description |
|---|---|
| `instance_results_presented.csv` | Full table: all buildings × all stage predictions |
| `instance_results_top_uncertain.csv` | Top-30 by entropy (priority review queue) |
| `vis_instance_level/` | PNG overlays — one per building instance |
| `stage2b_ensemble.jsonl` | Raw ensemble damage logits + calibrated probabilities |
| `stage2a_predictions.csv` | Population + building type predictions |

### What input it uses

- **One image pair**: `example_image_pair/nepal-flooding_00000408_pre_disaster.png` + `_post_disaster.png`
- Bundled in the repo — judges run this out of the box
- No geospatial context: images are pixel-space PNG, not georeferenced

---

## 2. What You've Built That Is NOT in the Notebook Yet

This is the gap. Everything below exists in the research pipeline at `/media/gisense/xihan/250812_tamu_cybertraining_team4/` and `/media/data/building_instance_tamu/` but has not been incorporated into the competition notebook.

### A. Real-world LA fire 2025 application (biggest gap)

| What exists | Where |
|---|---|
| Full pipeline run on **295 cells × 600m** covering ~197 km² of LA fire area | `pipeline/scripts/`, data at `/media/data/building_instance_tamu/la_fire_2025/` |
| **21,797 buildings** assessed for damage | `stage2_damage/multidate_full_run/building_damage_all_cells.{csv,geojson,gpkg}` |
| **15 publication-quality maps** (building damage, density, uncertainty, per-cell %, highlighted areas) at 3 zoom levels | `/media/data/building_instance_tamu/la_fire_2025/maps/*.png` |
| Damage summary statistics | `maps/damage_summary.md` |
| 120 QC overlays (post-disaster imagery + damage polygons) | `qc_overlays_m2b/` |

**The notebook demos only on Nepal flooding. Not a single LA fire result is shown to the judges.** The competition explicitly asks for disaster response applications — you have a complete real wildfire case study.

### B. Multi-date temporal aggregation

| What exists | Where |
|---|---|
| Multi-date pipeline (5 post-dates: Jan 9/13/14/15/16) | `pipeline/scripts/run_multidate_experiment.py` |
| Probability averaging (M1) and majority vote (M2b) aggregation | `pipeline/scripts/build_combined_dataset.py` |
| Per-date damage predictions for 120 cells | `stage2_damage/multidate_full_run/cell_*/dates/` |
| Inter-date stability metric (51% conflicting labels) | `building_damage_all_cells.csv` |

The competition notebook runs on a **single** post-disaster image. The research pipeline shows how predictions stabilize/diverge across multiple acquisition dates — a methodological contribution the judges don't see.

### C. Quantitative Stage 1 validation on xView2

| What exists | Where |
|---|---|
| F1 = 0.401, Precision = 0.682, Recall = 0.284 on 933 test images | `evaluation/results/sam3_eval/` |
| Mean IoU = 0.759 on matched prediction-label pairs | same |
| Prompt ablation (building vs. house vs. rooftop vs. structure) | `evaluation/results/prompt_experiments/`, data in `sam3_prompt_experiments/` |
| Evaluation scripts | `evaluation/evaluate_predictions.py`, `run_prompt_experiments.py` |

The notebook makes no quantitative claims about Stage 1 accuracy.

### D. Portability and reproducibility fixes

| What was fixed | Where |
|---|---|
| `sam3_building_identifier` `input_dir`/`output_dir` defaults changed from hardcoded `/media/...` to `""` (fail-fast) | `stage1/sam3_building_identifier/config.py` |
| `run_multidate_experiment.py` uses `conda run -n <env>` instead of absolute conda path | `pipeline/scripts/run_multidate_experiment.py` |
| `la_fire_paths.py` reads `LA_FIRE_ROOT` env var | `pipeline/scripts/la_fire_paths.py` |

These are not in `II_package/` — it still uses the archived `SAM3_Final_20260226` Stage 1 script.

### E. Active vs. archived Stage 1 package

The competition notebook calls `stage1/SAM3_Final_20260226/scripts/run_sam3_building_infer.py` (archived, inside II_package).  
The research pipeline uses `stage1/sam3_building_identifier` (active package, pip-installable).  
The two are functionally close but the active package has the portability fixes. The notebook's SAM3_Final variant works and the II_package README explains this — but judges can't verify the active package runs correctly from the notebook.

---

## 3. Judging Criteria vs. Current Notebook

| Criterion | Current state | What's missing |
|---|---|---|
| **Technical excellence** | Pipeline logic is sound; 3-stage ensemble with calibration is sophisticated | No quantitative validation numbers shown; LA fire scale not demonstrated |
| **Creativity** | Multi-task Stage 2a (type + population) + calibrated ensemble Stage 2b is genuinely novel | The notebook doesn't explain *why* this architecture; no ablation or comparison shown |
| **FAIR data principles** | Notebook is self-contained + outputs are structured | No DOI/persistent ID for data; GeoJSON/GPKG outputs exist but judges can't see them from the notebook |
| **Open science** | Notebook is clean; code is all open | No mention of xView2 dataset license; Stage 2b model training provenance not described |
| **Reproducibility** | `.nb_vendor` bootstrap + LFS pull is clever | Notebook only works if HF token obtained; no fallback path for judges without GPU |

---

## 4. Recommendations (ranked by impact, given 4-day deadline)

### Must-do (high impact, low effort)

**R1. Add a LA fire results section to the notebook**  
After the Nepal demo, add 3–5 markdown + code cells that:
- Load `building_damage_all_cells.csv` (or a small GeoJSON subset)
- Display the summary stats table (cells, buildings, damage distribution)
- Show 2–3 of the map PNGs inline with `IPython.display.Image`

This turns "Nepal demo" into "Nepal demo + real wildfire deployment at scale." Judges see the research pipeline's output without you having to re-run anything.

**R2. Add xView2 evaluation numbers**  
One markdown cell with a results table (F1, precision, recall, IoU) for Stage 1. Reference the evaluation scripts. Takes 10 minutes to write.

**R3. Add a methods narrative**  
The notebook has good structural markdown but doesn't explain *design choices*: why ensemble? why temperature calibration? why multi-date? Add 1–2 sentences per stage explaining the "why." Judges reward technical depth.

### Should-do (medium impact, medium effort)

**R4. Show multi-date temporal analysis**  
Add one cell that plots the per-date damage class distributions for 2–3 LA fire cells. Shows the temporal aggregation method working and surfaces the domain-mismatch insight (flood-trained model, wildfire application) honestly.

**R5. Add a GeoJSON visualization**  
Load a small subset of `building_damage_all_cells.geojson` (e.g., a 5-cell neighborhood) and plot it with geopandas/folium. Demonstrates FAIR/interoperable data format. Folium renders inline in Jupyter.

**R6. Document the FAIR properties explicitly**  
Add a markdown cell at the end: Findable (what's the repo URL), Accessible (notebook is self-contained), Interoperable (GeoJSON/GPKG standard formats), Reusable (MIT license, reproducible with one HF token).

### Nice-to-have (low-priority given deadline)

**R7. Replace SAM3_Final Stage 1 with active package**  
Change Cell 20 to call `python -m sam3_building_identifier ...` instead of `stage1/SAM3_Final_20260226/scripts/run_sam3_building_infer.py`. Requires ensuring II_package's `stage1/` is the active pip-installable version. Low risk but needs testing.

**R8. Add CPU fallback note**  
If judges run without a GPU, the notebook currently silently falls back but is very slow. Add a markdown note estimating runtime (~30–60 min on CPU for a single tile) so judges set expectations.

---

## 5. II_package Structure Summary

```
II_package/II_package/
├── present_instance.ipynb          ← competition notebook (54 cells)
├── run_pipeline.sh                 ← thin wrapper → scripts/run_instance_impact_driver.py
├── scripts/
│   ├── run_instance_impact_driver.py    ← CLI orchestrator (7-stage)
│   ├── generate_shared_instance_subimages.py
│   ├── build_stage2a_infer_csv.py
│   ├── infer_stage2a.py
│   ├── infer_stage2_ensemble.py
│   ├── present_instance_results.py
│   └── visualize_stage2_overlays.py
├── stage1/
│   └── SAM3_Final_20260226/        ← archived SAM3 variant (not the active package)
├── models/
│   ├── stage2a/stage2a_best_model.pt    (19 MB, via LFS)
│   └── stage2b/  (3 × 355 MB checkpoints, via LFS)
├── calibration/  (3 × temperature scaling artifacts)
├── configs/stage2b/  (3 training config JSONs)
├── example_image_pair/  (Nepal flooding pre/post PNG)
├── environment.yml
└── requirements.txt
```

---

## 6. Known Issues in II_package

| Issue | Severity | Fix |
|---|---|---|
| Stage 1 uses archived `SAM3_Final_20260226` script, not active `sam3_building_identifier` package | Low | Cosmetic; both work. Active package has portability fixes. |
| `CUDA_VISIBLE_DEVICES` hardcoded to "0" in Cell 20 | Low | Edit Cell 18/20 for different GPU index |
| MULTIPOLYGON instances silently dropped in shared subimage generation | Low | ~10 instances/tile; acceptable |
| No fallback if HF token unavailable | Medium | Mention in notebook; judges need a valid HF account |
| Notebook demo uses Nepal flooding, not wildfire | High | Add LA fire section (R1 above) |
| No quantitative benchmark numbers presented | High | Add xView2 results table (R2 above) |
