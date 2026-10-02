"""
sam3_building_identifier: detect buildings in satellite imagery with SAM3
(via samgeo.SamGeo3) and write per-instance polygons and confidence as JSON.

    from sam3_building_identifier import PipelineConfig, run_pipeline
    run_pipeline(PipelineConfig(input_dir="...", output_dir="...", max_images=3))

CLI: python -m sam3_building_identifier --help
"""

from sam3_building_identifier.config import PipelineConfig
from sam3_building_identifier.pipeline import run_pipeline

__all__ = ["PipelineConfig", "run_pipeline"]
__version__ = "0.2.0"
