"""Rule-based image quality filter for post-disaster chips.

No ML — purely statistics-based. Designed to reject:
  - All-zero / nodata images (mean_brightness < MIN_MEAN)
  - Images with excessive nodata coverage (frac_zeros > MAX_ZEROS)
  - Featureless / hazy images (spatial_std < MIN_STD)

Also provides per-crop filtering: a crop is "bad" if >50% of its pixels are
near-zero (i.e., the building centroid fell outside the valid image footprint).

Usage (as a module):
    from quality_filter import assess_tile_quality, assess_crop_quality
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np

# ─── Tile-level thresholds ────────────────────────────────────────────────────
MIN_MEAN = 15.0        # reject if overall mean brightness < this
MAX_ZEROS = 0.60       # reject if fraction of all-zero pixels > this
MIN_STD = 3.0          # reject if spatial std (mean per-pixel channel std) < this

# ─── Instance/crop-level threshold ───────────────────────────────────────────
CROP_MAX_ZERO_FRAC = 0.50  # reject a crop if > this fraction is black


def _open_as_array(image_path: Path) -> np.ndarray:
    """Read image as (C, H, W) uint8 array. Supports TIF (rasterio) and PNG (PIL)."""
    path = Path(image_path)
    suffix = path.suffix.lower()
    if suffix in (".tif", ".tiff"):
        import rasterio
        with rasterio.open(path) as ds:
            arr = ds.read()  # (C, H, W)
        return arr.astype(np.float32)
    else:
        from PIL import Image
        img = Image.open(path).convert("RGB")
        arr = np.array(img, dtype=np.float32)  # (H, W, 3)
        return arr.transpose(2, 0, 1)  # (3, H, W)


def assess_tile_quality(
    image_path: Path,
    min_mean: float = MIN_MEAN,
    max_zeros: float = MAX_ZEROS,
    min_std: float = MIN_STD,
) -> Tuple[bool, Dict]:
    """Assess quality of a full post-disaster tile image.

    Returns
    -------
    ok : bool
        True if image passes all thresholds.
    metrics : dict
        {'mean_brightness', 'frac_zeros', 'spatial_std', 'reason'}
    """
    try:
        arr = _open_as_array(image_path)
    except Exception as e:
        return False, {"mean_brightness": -1.0, "frac_zeros": 1.0, "spatial_std": 0.0,
                       "reason": f"open_error:{e}"}

    # (C, H, W) → compute metrics
    mean_brightness = float(arr.mean())
    # frac_zeros: fraction of pixels where ALL channels are 0
    all_zero_mask = (arr.max(axis=0) == 0)
    frac_zeros = float(all_zero_mask.mean())
    # spatial_std: mean of per-pixel std across channels (measures spatial variation)
    spatial_std = float(arr.std(axis=0).mean())

    reasons = []
    if mean_brightness < min_mean:
        reasons.append(f"mean_brightness={mean_brightness:.1f}<{min_mean}")
    if frac_zeros > max_zeros:
        reasons.append(f"frac_zeros={frac_zeros:.2f}>{max_zeros}")
    if spatial_std < min_std:
        reasons.append(f"spatial_std={spatial_std:.1f}<{min_std}")

    ok = len(reasons) == 0
    return ok, {
        "mean_brightness": round(mean_brightness, 2),
        "frac_zeros": round(frac_zeros, 4),
        "spatial_std": round(spatial_std, 2),
        "reason": "; ".join(reasons) if reasons else "ok",
    }


def assess_crop_quality(
    crop_arr: np.ndarray,
    max_zero_frac: float = CROP_MAX_ZERO_FRAC,
) -> Tuple[bool, float]:
    """Assess a single 256×256 crop (H, W, 3) uint8 numpy array.

    Returns (ok, zero_frac).
    """
    all_zero = (crop_arr.max(axis=-1) == 0) if crop_arr.ndim == 3 else (crop_arr == 0)
    zero_frac = float(all_zero.mean())
    return zero_frac <= max_zero_frac, round(zero_frac, 4)
