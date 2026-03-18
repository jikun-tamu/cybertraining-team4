"""
SAM3 model loader and thin inference wrapper.

Responsibilities
----------------
* Detect the best available device (CUDA → CPU).
* Load SamGeo3 exactly as the source-of-truth notebooks do.
* Expose a clean run_inference() method that returns raw results
  (masks array, scores array) without touching the file system.
* Expose save helpers (masks TIF, annotation PNG) that mirror notebook calls.
* Handle GPU memory cleanup between images.
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Optional

import torch

from sam3_building_identifier.config import PipelineConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------

def detect_device(requested: Optional[str] = None) -> str:
    """Return a torch device string.

    If *requested* is provided, use it directly (after validating CUDA is
    available if a cuda device is requested).  Otherwise auto-detect.
    """
    if requested is not None:
        if requested.startswith("cuda") and not torch.cuda.is_available():
            logger.warning(
                "Requested device %r but CUDA is not available. Falling back to cpu.",
                requested,
            )
            return "cpu"
        return requested

    if torch.cuda.is_available():
        device = "cuda"
        props = torch.cuda.get_device_properties(0)
        mem_gb = props.total_memory / 1024 ** 3
        logger.info("CUDA available: %s (%.1f GB)", props.name, mem_gb)
    else:
        device = "cpu"
        logger.info("CUDA not available — using CPU.")
    return device


def log_gpu_memory(label: str = "") -> None:
    """Log current GPU memory usage (no-op if CUDA unavailable)."""
    if not torch.cuda.is_available():
        return
    allocated = torch.cuda.memory_allocated(0) / 1024 ** 3
    reserved = torch.cuda.memory_reserved(0) / 1024 ** 3
    tag = f" [{label}]" if label else ""
    logger.debug("GPU memory%s — allocated: %.2f GB, reserved: %.2f GB",
                 tag, allocated, reserved)


def clear_gpu_memory() -> None:
    """Free cached GPU memory between images."""
    torch.cuda.empty_cache()
    gc.collect()


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

class SAM3Model:
    """Thin wrapper around SamGeo3 that mirrors notebook usage exactly.

    Parameters
    ----------
    cfg:
        PipelineConfig instance.  Only the model-related fields are used here.
    """

    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg
        self._sam3 = None  # lazy-loaded

    # ------------------------------------------------------------------
    # Lazy initialisation
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Instantiate SamGeo3.  Call once before processing images."""
        if self._sam3 is not None:
            logger.debug("SAM3 model already loaded — skipping.")
            return

        # Import here so the rest of the package can be imported even when
        # samgeo is not installed (e.g. for config-only use).
        from samgeo import SamGeo3  # noqa: PLC0415

        device = detect_device(self.cfg.device)
        logger.info(
            "Loading SamGeo3 (backend=%r, device=%r, load_from_HF=%s)",
            self.cfg.backend,
            device,
            self.cfg.load_from_hf,
        )

        self._sam3 = SamGeo3(
            backend=self.cfg.backend,
            device=device,
            checkpoint_path=self.cfg.checkpoint_path,
            load_from_HF=self.cfg.load_from_hf,
        )
        log_gpu_memory("after model load")
        logger.info("SamGeo3 loaded successfully.")

    @property
    def sam3(self):
        if self._sam3 is None:
            raise RuntimeError("Model not loaded. Call SAM3Model.load() first.")
        return self._sam3

    # ------------------------------------------------------------------
    # Single-image inference  (mirrors notebook single-image path)
    # ------------------------------------------------------------------

    def set_image(self, image_path: Path) -> None:
        """Load one image into the model (notebook: sam3.set_image(img_path))."""
        self.sam3.set_image(str(image_path))

    def generate_masks(self) -> int:
        """Run mask generation on the currently loaded image.

        Returns the number of instances found.

        Notebook equivalent::
            sam3.generate_masks(prompt="building", min_size=100)

        Note: generate_masks() returns None and stores results in self.sam3.masks.
        We read the count from self.sam3.masks after calling.
        """
        self.sam3.generate_masks(
            prompt=self.cfg.text_prompt,
            min_size=self.cfg.min_size,
            max_size=self.cfg.max_size,
        )
        masks = getattr(self.sam3, "masks", None)
        return len(masks) if masks is not None else 0

    # ------------------------------------------------------------------
    # Batch inference  (mirrors notebook batch path — still 1 image at a time
    # due to GPU memory, but uses the batch API for consistency)
    # ------------------------------------------------------------------

    def set_image_batch(self, image_paths: list[Path]) -> None:
        """Load a batch of images (notebook: sam3.set_image_batch(batch_paths))."""
        self.sam3.set_image_batch([str(p) for p in image_paths])

    def generate_masks_batch(self) -> list[int]:
        """Run batch mask generation.

        Returns a list of instance counts, one per image in the batch.

        Notebook equivalent::
            sam3.generate_masks_batch("building", min_size=100)
        """
        self.sam3.generate_masks_batch(
            self.cfg.text_prompt,
            min_size=self.cfg.min_size,
            max_size=self.cfg.max_size,
        )
        # Results stored in self.sam3.batch_results (list of per-image dicts)
        batch_results = getattr(self.sam3, "batch_results", None) or []
        return [
            len(r.get("masks") or [])
            for r in batch_results
        ]

    # ------------------------------------------------------------------
    # Saving masks TIF  (mirrors notebook save_masks / save_masks_batch)
    # ------------------------------------------------------------------

    def save_masks(self, stem: str, masks_dir: Path) -> tuple[Path, Path]:
        """Save instance mask TIF and scores TIF for a single image.

        Notebook equivalent::
            sam3.save_masks(output=mask_output, save_scores=score_output, unique=True)

        Returns
        -------
        (mask_path, scores_path)
        """
        mask_path = masks_dir / f"{stem}.tif"
        score_path = masks_dir / f"{stem}_scores.tif"
        self.sam3.save_masks(
            output=str(mask_path),
            unique=True,
            save_scores=str(score_path),
        )
        return mask_path, score_path

    def save_masks_batch(self, stem: str, masks_dir: Path) -> list[Path]:
        """Save mask TIFs for the current batch.

        Notebook equivalent::
            sam3.save_masks_batch(output_dir=..., prefix=stem, unique=True)

        Returns list of saved mask file paths.
        """
        saved = self.sam3.save_masks_batch(
            output_dir=str(masks_dir),
            prefix=stem,
            unique=True,
        )
        return [Path(p) for p in (saved or [])]

    # ------------------------------------------------------------------
    # Saving annotation PNG  (mirrors notebook show_anns)
    # ------------------------------------------------------------------

    def save_annotation(self, stem: str, annotations_dir: Path) -> Path:
        """Save annotation PNG for the current single image.

        Notebook equivalent::
            sam3.show_anns(output=ann_output, dpi=150, font_size=8)
        """
        ann_path = annotations_dir / f"{stem}_ann.png"
        self.sam3.show_anns(
            output=str(ann_path),
            dpi=self.cfg.annotation_dpi,
            show_bbox=True,
            show_score=True,
        )
        return ann_path

    def save_annotation_batch(self, stem: str, annotations_dir: Path) -> Path:
        """Save annotation PNGs for the current batch.

        Notebook equivalent::
            sam3.show_anns_batch(output_dir=..., prefix=f"{stem}_ann", dpi=150)
        """
        self.sam3.show_anns_batch(
            output_dir=str(annotations_dir),
            prefix=f"{stem}_ann",
            dpi=self.cfg.annotation_dpi,
            show_bbox=True,
            show_score=True,
        )
        return annotations_dir / f"{stem}_ann.png"

    # ------------------------------------------------------------------
    # Memory management
    # ------------------------------------------------------------------

    def clear_memory(self) -> None:
        """Clear GPU cache — call between images to avoid OOM."""
        clear_gpu_memory()
        log_gpu_memory("after clear")
