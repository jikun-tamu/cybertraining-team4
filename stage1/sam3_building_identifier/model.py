"""
SAM3 model loader and in-memory inference wrapper around samgeo.SamGeo3.
"""

from __future__ import annotations

import gc
import logging
from typing import Optional

import numpy as np
import torch

from sam3_building_identifier.config import PipelineConfig

logger = logging.getLogger(__name__)


def detect_device(requested: Optional[str] = None) -> str:
    """Return a torch device string, falling back to CPU without CUDA."""
    if requested is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("Requested %r but CUDA is not available; using cpu.", requested)
        return "cpu"
    return requested


class SAM3Model:
    """Loads SamGeo3 once and runs text-prompted segmentation on RGB arrays."""

    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg
        self._sam3 = None

    def load(self) -> None:
        if self._sam3 is not None:
            return
        from samgeo import SamGeo3

        device = detect_device(self.cfg.device)
        if device.startswith("cuda:"):
            # Meta's SAM3 decoder allocates some tensors on the *current* CUDA
            # device, so cuda:1 crashes with a device mismatch unless the
            # current device is set as well.
            torch.cuda.set_device(device)
        logger.info("Loading SamGeo3 (%s, backend=%s, device=%s, threshold=%s)", self.cfg.model_id,
                    self.cfg.backend, device, self.cfg.confidence_threshold)
        self._sam3 = SamGeo3(
            backend=self.cfg.backend,
            model_id=self.cfg.model_id,
            confidence_threshold=self.cfg.confidence_threshold,
            device=device,
            checkpoint_path=self.cfg.checkpoint_path,
            load_from_HF=self.cfg.load_from_hf,
        )

    def predict(self, image: np.ndarray) -> list[tuple[np.ndarray, float]]:
        """Segment one HxWx3 uint8 image. Returns [(bool mask HxW, score), ...]."""
        if self._sam3 is None:
            raise RuntimeError("Model not loaded. Call SAM3Model.load() first.")
        sam3 = self._sam3
        sam3.set_image(image)
        # generate_masks() returns None; results are stored on the object.
        sam3.generate_masks(
            prompt=self.cfg.text_prompt,
            min_size=self.cfg.min_size,
            max_size=self.cfg.max_size,
        )
        masks = sam3.masks if sam3.masks is not None else []
        scores = sam3.scores if sam3.scores is not None else []

        results = []
        for mask, score in zip(masks, scores):
            mask = np.asarray(mask).squeeze()
            if mask.ndim > 2:
                mask = mask[0]
            results.append((mask > 0, float(np.asarray(score).ravel()[0])))

        torch.cuda.empty_cache()
        gc.collect()
        return results
