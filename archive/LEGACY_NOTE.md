# LEGACY — Archive Directory (formerly exploration/)

**Status**: ARCHIVED (not production code)
**Date marked**: 2026-04-02

This directory contains experimental work that has been **superseded by production code**:

| Subdirectory | What it was | Superseded by |
|-------------|-------------|---------------|
| `SAM3_Final/` | Earlier SAM3 pipeline with georef | `stage1/` (production SAM3 package) |
| `SAM3_notebooks/` | Original interactive SAM3 exploration | `stage1/` |
| `Mask_R-CNN/` | Instance segmentation training | SAM3 (zero-shot, no training needed) |
| `PolyWorld/` | Polygon-based building extraction | SAM3 |
| `GeoAI_QuishengWu/` | Building footprint extraction | SAM3 via samgeo |
| `GeoAI_building_segmentation/` | Early building segmentation | SAM3 |
| `corrected_model/` | Corrected Mask R-CNN checkpoint | SAM3 |

**Do not delete** without first confirming that no reproducibility requirements exist.
**Do not import** from these directories in new code.
