"""
PipelineConfig: every tuneable parameter in one place.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class PipelineConfig:
    # --- I/O ---
    input_dir: str = ""
    """Directory containing input images (required)."""

    output_dir: str = ""
    """Root output directory; masks/, annotations/, predictions/ are created in it (required)."""

    disaster_type: str = "auto"
    """'auto': pre-disaster images if xView2 *_pre_disaster naming exists, else all.
    'pre' / 'post': filter by xView2 suffix. 'all': every image."""

    image_extensions: tuple = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

    max_images: Optional[int] = None
    """Process at most this many images (for tests)."""

    # --- SAM3 model ---
    backend: str = "meta"
    """SamGeo3 backend: 'meta' (Meta's official implementation) or 'transformers'."""

    model_id: str = "facebook/sam3"
    """'facebook/sam3' (Nov 2025) or 'facebook/sam3.1' (Mar 2026; needs samgeo>=1.4,
    sam3>=0.1.4 and the geoai_sam31 env; Meta backend only)."""

    confidence_threshold: float = 0.4
    """SAM3 detection score threshold (SamGeo3 default 0.5). On the xView2 test
    set with tiling, 0.4 gave the best F1 (0.640 vs 0.619 at 0.5, 0.628 at 0.3)."""

    device: Optional[str] = None
    """Torch device ('cuda', 'cuda:1', 'cpu'). None: cuda if available."""

    load_from_hf: bool = True
    """Load SAM3 weights from Hugging Face (requires a cached HF login)."""

    checkpoint_path: Optional[str] = None
    """Local checkpoint path. None: default HF download."""

    # --- Inference ---
    text_prompt: str = "building"
    """Text prompt. With the default config 'house' scores about the same F1 on
    xView2 (0.645 vs 0.640) but lower recall on the two wildfire events, so
    'building' stays the default (evaluation/results/prompt_experiments/)."""

    min_size: int = 100
    """Minimum mask area in pixels, applied per window and again after merging."""

    max_size: Optional[int] = None
    """Maximum mask area in pixels. None: no limit."""

    tile_size: Optional[int] = 512
    """Window size for tiled inference. None: run on the full image."""

    tile_overlap: int = 64
    """Minimum overlap between adjacent windows."""

    merge_iou: float = 0.5
    """IoU (inside the shared window overlap) above which two instances from
    different windows are merged into one."""

    # --- Polygons ---
    polygon_epsilon: float = 2.0
    """Douglas-Peucker tolerance (pixels) for geoai.orthogonalize()."""

    min_polygon_area: float = 100.0
    """Drop polygons smaller than this (square pixels)."""

    simplify_tolerance: Optional[float] = None
    """Optional extra Shapely simplify() tolerance. None: skip."""

    # --- Outputs ---
    save_masks: bool = True
    """Write masks/<stem>.tif (int32 labels) and masks/<stem>_scores.tif."""

    save_annotations: bool = True
    """Write annotations/<stem>_ann.png (polygon overlay)."""

    skip_existing: bool = True
    """Skip images whose prediction JSON already exists."""

    @property
    def masks_dir(self) -> Path:
        return Path(self.output_dir) / "masks"

    @property
    def annotations_dir(self) -> Path:
        return Path(self.output_dir) / "annotations"

    @property
    def predictions_dir(self) -> Path:
        return Path(self.output_dir) / "predictions"

    @property
    def run_summary_path(self) -> Path:
        return Path(self.output_dir) / "run_summary.json"

    def make_output_dirs(self) -> None:
        self.predictions_dir.mkdir(parents=True, exist_ok=True)
        if self.save_masks:
            self.masks_dir.mkdir(parents=True, exist_ok=True)
        if self.save_annotations:
            self.annotations_dir.mkdir(parents=True, exist_ok=True)
