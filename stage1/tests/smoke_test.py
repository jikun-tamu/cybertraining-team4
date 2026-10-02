"""
Smoke test: run the pipeline on 3 xView2 pre-disaster images and validate outputs.

    conda run -n geoai_sam python stage1/tests/smoke_test.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run without pip install

from sam3_building_identifier import PipelineConfig, run_pipeline

OUT = Path("/tmp/sam3_smoke_test")
shutil.rmtree(OUT, ignore_errors=True)

cfg = PipelineConfig(
    input_dir="/media/data/building_instance_tamu/test/images",
    output_dir=str(OUT),
    disaster_type="pre",
    max_images=3,
    skip_existing=False,
)
run_pipeline(cfg)

errors = []
preds = sorted(cfg.predictions_dir.glob("*_prediction.json"))
if len(preds) != 3:
    errors.append(f"expected 3 prediction files, found {len(preds)}")

for pf in preds:
    doc = json.loads(pf.read_text())
    stem = doc["image"]["stem"]
    for i, inst in enumerate(doc["instances"]):
        missing = {"id", "uid", "bbox_xyxy", "polygon", "area_px", "confidence"} - inst.keys()
        if missing:
            errors.append(f"{stem}[{i}]: missing {sorted(missing)}")
        x1, y1, x2, y2 = inst["bbox_xyxy"]
        if not (x1 < x2 and y1 < y2):
            errors.append(f"{stem}[{i}]: invalid bbox {inst['bbox_xyxy']}")
        if len(inst["polygon"]) < 4:
            errors.append(f"{stem}[{i}]: polygon has {len(inst['polygon'])} points")
        if not 0 < inst["confidence"] <= 1:
            errors.append(f"{stem}[{i}]: confidence {inst['confidence']}")
    if doc["instances"]:
        for path in (cfg.masks_dir / f"{stem}.tif", cfg.masks_dir / f"{stem}_scores.tif",
                     cfg.annotations_dir / f"{stem}_ann.png"):
            if not path.exists():
                errors.append(f"missing {path.name}")
    print(f"  {stem}: {doc['summary']['num_instances']} instances, {doc['timing']['total_sec']}s")

if not cfg.run_summary_path.exists():
    errors.append("run_summary.json not written")

if errors:
    print(f"FAIL: {len(errors)} error(s)")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
print("PASS")
