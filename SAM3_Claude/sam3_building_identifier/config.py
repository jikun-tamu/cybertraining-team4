"""
PipelineConfig — all tuneable parameters in one place.

Every parameter has a docstring and a default that matches the source-of-truth
notebooks (251210_sam3_batch_segmentation_Xihan.ipynb and
251210_sam3_image_segmentation_xihan.ipynb).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class PipelineConfig:
    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------
    input_dir: str = "/media/data/building_instance_tamu/test/images"
    """Directory containing input images."""

    output_dir: str = "/media/data/building_instance_tamu/sam3_claude/test"
    """Root output directory.  Sub-folders are created automatically."""

    # ------------------------------------------------------------------
    # Image filtering
    # ------------------------------------------------------------------
    disaster_type: str = "auto"
    """Which images to process.

    'auto' (default) — inspect the folder at runtime:
                       if any *_pre_disaster.* files exist → use pre only;
                       otherwise → process all images in the folder.
    'pre'  → *_pre_disaster.* only (warns if none found, returns empty list)
    'post' → *_post_disaster.* only (warns if none found, returns empty list)
    'all'  → every image file in input_dir
    """

    image_extensions: tuple = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
    """Accepted image file extensions (lower-cased)."""

    # ------------------------------------------------------------------
    # SAM3 model
    # ------------------------------------------------------------------
    backend: str = "meta"
    """SAM3 backend. 'meta' uses Meta's official implementation."""

    device: Optional[str] = None
    """Torch device string ('cuda', 'cuda:0', 'cpu').
    None → auto-detect: cuda if available, else cpu."""

    load_from_hf: bool = True
    """Load SAM3 weights from Hugging Face (requires prior huggingface-cli login)."""

    checkpoint_path: Optional[str] = None
    """Optional local checkpoint path. None → use default HF download."""

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    text_prompt: str = "building"
    """Text prompt passed to SamGeo3.generate_masks()."""

    min_size: int = 100
    """Minimum mask area in pixels. Masks smaller than this are discarded."""

    max_size: Optional[int] = None
    """Maximum mask area in pixels. None → no upper limit."""

    # ------------------------------------------------------------------
    # Polygon / vectorization
    # ------------------------------------------------------------------
    polygon_epsilon: float = 2.0
    """Douglas-Peucker simplification tolerance (pixels) passed to
    geoai.orthogonalize().  Matches notebooks (epsilon=2)."""

    min_polygon_area: float = 10.0
    """Minimum polygon area (sq-pixels) to keep after vectorization.
    Passed to geoai.orthogonalize(min_area=...)."""

    simplify_tolerance: Optional[float] = None
    """Extra Shapely simplify() tolerance after orthogonalize.
    None → skip the extra simplification step."""

    # ------------------------------------------------------------------
    # Output control
    # ------------------------------------------------------------------
    save_masks: bool = True
    """Save instance mask TIF and scores TIF to masks/ sub-dir."""

    save_annotations: bool = True
    """Save annotation PNG to annotations/ sub-dir."""

    save_predictions: bool = True
    """Save enhanced prediction JSON to predictions/ sub-dir."""

    annotation_dpi: int = 150
    """DPI for saved annotation PNGs."""

    skip_existing: bool = True
    """Skip images whose prediction JSON already exists in output_dir."""

    # ------------------------------------------------------------------
    # Batch / system
    # ------------------------------------------------------------------
    batch_size: int = 1
    """Images per SAM3 batch call.  Notebooks use 1 due to GPU memory."""

    max_images: Optional[int] = None
    """Cap the number of images to process (useful for testing). None → all."""

    # ------------------------------------------------------------------
    # Derived paths (computed from output_dir — do not set manually)
    # ------------------------------------------------------------------
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
        """Create all output sub-directories."""
        for d in (self.masks_dir, self.annotations_dir, self.predictions_dir):
            d.mkdir(parents=True, exist_ok=True)
