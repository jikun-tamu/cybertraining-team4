"""
Mask-to-polygon conversion.

Design rationale
----------------
The source-of-truth notebooks use ``geoai.orthogonalize()`` to convert
raster instance masks to vector polygons.  ``orthogonalize`` runs
Douglas-Peucker simplification and optional shape regularisation, producing
clean, mostly-rectilinear building footprints — which matches the xView2
dataset characteristics well.

We replicate that path here exactly, adding:
  * bbox (xyxy) extraction from each polygon via ``shapely.bounds``
  * area in pixels (``shapely.Polygon.area``)
  * confidence score read from the ``_scores.tif`` raster
  * an optional second-pass ``shapely.simplify()`` for tighter polygons

Fallback
--------
If geoai is unavailable (import error), the module falls back to a pure
OpenCV+Shapely approach (``cv2.findContours`` → ``shapely.Polygon``).
The fallback is clearly flagged in log output so the operator knows.
"""

from __future__ import annotations

import logging
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rasterio
from shapely.geometry import mapping

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output schema helpers
# ---------------------------------------------------------------------------

def _make_instance(
    inst_id: int,
    polygon,  # shapely Polygon
    confidence: Optional[float],
    simplify_tolerance: Optional[float] = None,
) -> Dict[str, Any]:
    """Build one entry in the 'instances' list of the output JSON."""
    if simplify_tolerance is not None and simplify_tolerance > 0:
        polygon = polygon.simplify(simplify_tolerance, preserve_topology=True)

    if polygon.is_empty or not polygon.is_valid:
        polygon = polygon.buffer(0)  # attempt repair

    exterior = list(polygon.exterior.coords)
    minx, miny, maxx, maxy = polygon.bounds
    return {
        "id": inst_id,
        "uid": str(uuid.uuid4()),
        "bbox_xyxy": [round(minx), round(miny), round(maxx), round(maxy)],
        "polygon": [[round(x, 2), round(y, 2)] for x, y in exterior],
        "area_px": round(polygon.area, 2),
        "confidence": round(confidence, 4) if confidence is not None else None,
    }


# ---------------------------------------------------------------------------
# Score extraction helper
# ---------------------------------------------------------------------------

def _read_scores_for_id(
    score_arr: Optional[np.ndarray],
    mask_arr: np.ndarray,
    mask_id: int,
) -> Optional[float]:
    """Return mean score for pixels belonging to *mask_id*."""
    if score_arr is None:
        return None
    pixels = mask_arr == mask_id
    if not pixels.any():
        return None
    return float(score_arr[pixels].mean())


# ---------------------------------------------------------------------------
# Primary path: geoai.orthogonalize  (matches notebooks exactly)
# ---------------------------------------------------------------------------

def _vectorize_via_orthogonalize(
    mask_path: Path,
    score_path: Optional[Path],
    epsilon: float,
    min_area: float,
    simplify_tolerance: Optional[float],
) -> List[Dict[str, Any]]:
    """Convert mask TIF → instances using geoai.orthogonalize + Shapely.

    This path mirrors the notebook's vectorize_sam3_masks() function.
    """
    import geoai  # noqa: PLC0415
    from shapely import wkt as shapely_wkt  # noqa: PLC0415

    # Read mask raster
    with rasterio.open(mask_path) as src:
        mask_arr = src.read(1)  # shape: (H, W)

    # Read scores raster if available
    score_arr: Optional[np.ndarray] = None
    if score_path is not None and score_path.exists():
        with rasterio.open(score_path) as src:
            score_arr = src.read(1).astype(np.float32)

    # Orthogonalize: raster → GeoDataFrame of polygon features
    with tempfile.NamedTemporaryFile(suffix=".geojson", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        gdf = geoai.orthogonalize(
            str(mask_path),
            output_path=tmp_path,
            epsilon=epsilon,
            min_area=min_area,
        )
    except Exception as exc:
        logger.warning("geoai.orthogonalize failed: %s — falling back to cv2", exc)
        return _vectorize_via_cv2(
            mask_path, score_path, epsilon, min_area, simplify_tolerance
        )

    if gdf is None or len(gdf) == 0:
        logger.debug("orthogonalize returned empty GeoDataFrame for %s", mask_path.name)
        return []

    instances: List[Dict[str, Any]] = []
    for inst_id, row in enumerate(gdf.itertuples(), start=1):
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        # Find which mask label corresponds to this polygon via centroid lookup
        cx, cy = geom.centroid.x, geom.centroid.y
        px, py = int(round(cx)), int(round(cy))
        py_clipped = max(0, min(py, mask_arr.shape[0] - 1))
        px_clipped = max(0, min(px, mask_arr.shape[1] - 1))
        label = int(mask_arr[py_clipped, px_clipped])

        confidence = _read_scores_for_id(score_arr, mask_arr, label)
        inst = _make_instance(inst_id, geom, confidence, simplify_tolerance)
        instances.append(inst)

    return instances


# ---------------------------------------------------------------------------
# Fallback path: pure OpenCV + Shapely
# ---------------------------------------------------------------------------

def _vectorize_via_cv2(
    mask_path: Path,
    score_path: Optional[Path],
    epsilon: float,
    min_area: float,
    simplify_tolerance: Optional[float],
) -> List[Dict[str, Any]]:
    """Convert mask TIF → instances using cv2.findContours + Shapely.

    Used as fallback when geoai is unavailable or orthogonalize fails.
    """
    import cv2  # noqa: PLC0415
    from shapely.geometry import Polygon  # noqa: PLC0415

    with rasterio.open(mask_path) as src:
        mask_arr = src.read(1)

    score_arr: Optional[np.ndarray] = None
    if score_path is not None and score_path.exists():
        with rasterio.open(score_path) as src:
            score_arr = src.read(1).astype(np.float32)

    unique_ids = np.unique(mask_arr)
    unique_ids = unique_ids[unique_ids != 0]  # exclude background

    instances: List[Dict[str, Any]] = []
    inst_id = 1
    for mask_id in unique_ids:
        binary = (mask_arr == mask_id).astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue

        # Largest contour = building exterior
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) < min_area:
            continue

        # Douglas-Peucker approximation (arc_length * epsilon fraction is too
        # aggressive; use epsilon directly as perimeter fraction)
        peri = cv2.arcLength(contour, True)
        approx_eps = epsilon / max(peri, 1.0) * peri  # = epsilon (pixels)
        approx = cv2.approxPolyDP(contour, approx_eps, True)

        coords = [(float(p[0][0]), float(p[0][1])) for p in approx]
        if len(coords) < 3:
            continue

        try:
            poly = Polygon(coords)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.is_empty or poly.area < min_area:
                continue
        except Exception:
            continue

        confidence = _read_scores_for_id(score_arr, mask_arr, int(mask_id))
        inst = _make_instance(inst_id, poly, confidence, simplify_tolerance)
        instances.append(inst)
        inst_id += 1

    return instances


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def masks_to_instances(
    mask_path: Path,
    score_path: Optional[Path],
    epsilon: float = 2.0,
    min_area: float = 10.0,
    simplify_tolerance: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Convert a mask TIF (and optional scores TIF) to a list of instance dicts.

    Tries ``geoai.orthogonalize`` first (notebook-faithful path), then falls
    back to cv2 contours if geoai is unavailable.

    Parameters
    ----------
    mask_path:
        Path to the instance mask GeoTIFF (uint8, one integer label per pixel).
    score_path:
        Path to the per-pixel scores GeoTIFF (float32).  May be None.
    epsilon:
        Polygon simplification tolerance (pixels).
    min_area:
        Minimum polygon area (sq-pixels) to keep.
    simplify_tolerance:
        Optional additional Shapely simplify tolerance.

    Returns
    -------
    List of instance dicts with keys:
        id, uid, bbox_xyxy, polygon, area_px, confidence
    """
    if not mask_path.exists():
        logger.warning("Mask file not found: %s", mask_path)
        return []

    try:
        import geoai  # noqa: F401, PLC0415
        use_orthogonalize = True
    except ImportError:
        logger.warning("geoai not available — using cv2 fallback.")
        use_orthogonalize = False

    if use_orthogonalize:
        instances = _vectorize_via_orthogonalize(
            mask_path, score_path, epsilon, min_area, simplify_tolerance
        )
    else:
        instances = _vectorize_via_cv2(
            mask_path, score_path, epsilon, min_area, simplify_tolerance
        )

    logger.debug(
        "%s → %d instances (method=%s)",
        mask_path.name,
        len(instances),
        "orthogonalize" if use_orthogonalize else "cv2",
    )
    return instances
