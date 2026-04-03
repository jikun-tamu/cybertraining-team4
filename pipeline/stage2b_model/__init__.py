"""Stage 2b damage model — extracted from pipeline/scripts/train_stage2.py.

This module contains the ACTIVE production implementations of:
  - SiameseDamageModel  (Siamese ConvNeXt + dual-mask pooling + CORAL head)
  - CoralHead           (ordinal regression head)
  - Data loading and metric utilities used by inference scripts

These are COPIES of the code in train_stage2.py (lines 245–623), extracted
so that inference scripts can import cleanly without depending on the 1400-line
training monolith.

IMPORTANT: train_stage2.py is NOT modified. Both copies must stay in sync.
If you change model logic, update BOTH this file AND train_stage2.py.

Extracted: 2026-04-02
Source: pipeline/scripts/train_stage2.py
"""

from .model import (
    SiameseDamageModel,
    CoralHead,
    coral_probs_from_logits,
    coral_targets,
    downsample_mask,
    masked_avg_pool,
    global_avg_pool,
)
from .data_utils import (
    load_rgb_tensor,
    load_mask_tensor,
    collate_batch,
    read_rows,
    IMAGENET_MEAN,
    IMAGENET_STD,
)
from .metrics import (
    confusion_matrix,
    macro_f1_from_cm,
    qwk_from_cm,
    ece_score,
)

__all__ = [
    "SiameseDamageModel",
    "CoralHead",
    "coral_probs_from_logits",
    "coral_targets",
    "downsample_mask",
    "masked_avg_pool",
    "global_avg_pool",
    "load_rgb_tensor",
    "load_mask_tensor",
    "collate_batch",
    "read_rows",
    "confusion_matrix",
    "macro_f1_from_cm",
    "qwk_from_cm",
    "ece_score",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
]
