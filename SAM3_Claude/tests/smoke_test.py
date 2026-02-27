"""
Smoke test: run the pipeline on 3 pre-disaster images and validate outputs.

Run with:
    conda run -n geoai_sam python tests/smoke_test.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow running from the SAM3_Claude root dir without installation
sys.path.insert(0, str(Path(__file__).parent.parent))

from sam3_building_identifier import PipelineConfig, run_pipeline

SMOKE_OUTPUT = "/tmp/sam3_smoke_test"

cfg = PipelineConfig(
    input_dir="/media/data/building_instance_tamu/test/images",
    output_dir=SMOKE_OUTPUT,
    disaster_type="pre",
    max_images=3,
    skip_existing=False,   # always re-run for a clean test
    save_masks=True,
    save_annotations=True,
)

print("\n=== Running smoke test on 3 images ===\n")
summary = run_pipeline(cfg)

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
errors = []

pred_dir = Path(SMOKE_OUTPUT) / "predictions"
pred_files = list(pred_dir.glob("*_prediction.json"))

if len(pred_files) != 3:
    errors.append(f"Expected 3 prediction files, found {len(pred_files)}")

for pf in sorted(pred_files):
    with open(pf) as fh:
        doc = json.load(fh)

    stem = pf.name.replace("_prediction.json", "")

    # Check top-level keys
    for key in ("image", "instances", "timing", "summary"):
        if key not in doc:
            errors.append(f"{pf.name}: missing key '{key}'")

    # Check each instance has required fields
    for i, inst in enumerate(doc.get("instances", [])):
        for field in ("id", "uid", "bbox_xyxy", "polygon", "area_px", "confidence"):
            if field not in inst:
                errors.append(f"{pf.name} instance[{i}]: missing field '{field}'")

        bbox = inst.get("bbox_xyxy", [])
        if len(bbox) != 4:
            errors.append(f"{pf.name} instance[{i}]: bbox_xyxy must have 4 values")
        elif not (bbox[0] < bbox[2] and bbox[1] < bbox[3]):
            errors.append(f"{pf.name} instance[{i}]: invalid bbox {bbox}")

        poly = inst.get("polygon", [])
        if len(poly) < 3:
            errors.append(f"{pf.name} instance[{i}]: polygon too short ({len(poly)} pts)")

    print(f"  {pf.name}")
    print(f"    instances : {doc['summary']['num_instances']}")
    print(f"    timing    : infer={doc['timing'].get('inference_sec', '?')}s  "
          f"post={doc['timing'].get('postprocess_sec', '?')}s  "
          f"total={doc['timing'].get('total_sec', '?')}s")
    if doc["instances"]:
        first = doc["instances"][0]
        print(f"    sample[0] : bbox={first['bbox_xyxy']}  "
              f"area={first['area_px']}  conf={first['confidence']}")

# Check mask and annotation files exist — only required when detections > 0
mask_dir = Path(SMOKE_OUTPUT) / "masks"
ann_dir = Path(SMOKE_OUTPUT) / "annotations"
for pf in sorted(pred_files):
    stem = pf.name.replace("_prediction.json", "")
    with open(pf) as fh:
        doc = json.load(fh)
    n = doc["summary"]["num_instances"]
    if n > 0:
        for suffix in (".tif", "_scores.tif"):
            mf = mask_dir / f"{stem}{suffix}"
            if not mf.exists():
                errors.append(f"Missing mask file (expected, {n} detections): {mf.name}")
        af = ann_dir / f"{stem}_ann.png"
        if not af.exists():
            errors.append(f"Missing annotation file (expected, {n} detections): {af.name}")
    else:
        print(f"  NOTE: {stem} — 0 detections, mask/ann files correctly absent")

# Check run_summary.json
summary_path = Path(SMOKE_OUTPUT) / "run_summary.json"
if not summary_path.exists():
    errors.append("run_summary.json not created")
else:
    with open(summary_path) as fh:
        rs = json.load(fh)
    print(f"\nrun_summary.json:")
    print(f"  images_processed : {rs['totals']['images_processed']}")
    print(f"  total_buildings  : {rs['totals']['total_buildings']}")
    print(f"  total_wall_time  : {rs['totals']['total_wall_time_sec']}s")
    print(f"  avg_per_image    : {rs['totals']['avg_time_per_image_sec']}s")

print()
if errors:
    print(f"FAIL — {len(errors)} validation error(s):")
    for e in errors:
        print(f"  ✗ {e}")
    sys.exit(1)
else:
    print("PASS — all validations passed.")
    sys.exit(0)
