"""
sam3_building_identifier
========================
A product-like pipeline that wraps SamGeo3 to detect buildings in
xView2-style satellite imagery and outputs per-instance bounding boxes,
polygons, and confidence scores as structured JSON.

Quick start
-----------
    from sam3_building_identifier import run_pipeline, PipelineConfig

    cfg = PipelineConfig(
        input_dir="/media/data/building_instance_tamu/test/images",
        output_dir="/tmp/sam3_output",
        max_images=3,
    )
    run_pipeline(cfg)

CLI
---
    python -m sam3_building_identifier --help
"""

from sam3_building_identifier.config import PipelineConfig
from sam3_building_identifier.pipeline import run_pipeline

__all__ = ["PipelineConfig", "run_pipeline"]
__version__ = "0.1.0"
