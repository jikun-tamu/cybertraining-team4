"""
Label raster -> per-instance polygons (pixel coordinates).

Uses geoai.orthogonalize(), which vectorizes each label value separately and
regularizes the outlines into mostly-rectilinear footprints. Each polygon
keeps its label value, so the confidence is the instance's own SAM3 score.
Only the largest part of each label is kept: a label split into pieces
(e.g. by a higher-score neighbour painted over it) stays one building.
"""

from __future__ import annotations

import logging
import tempfile
import uuid
import warnings
from pathlib import Path
from typing import Any, Optional

import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning

logger = logging.getLogger(__name__)


def _make_instance(inst_id: int, polygon, confidence: float,
                   simplify_tolerance: Optional[float]) -> dict[str, Any]:
    if simplify_tolerance:
        polygon = polygon.simplify(simplify_tolerance, preserve_topology=True)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    if polygon.geom_type == "MultiPolygon":
        polygon = max(polygon.geoms, key=lambda g: g.area)
    minx, miny, maxx, maxy = polygon.bounds
    return {
        "id": inst_id,
        "uid": str(uuid.uuid4()),
        "bbox_xyxy": [round(minx), round(miny), round(maxx), round(maxy)],
        "polygon": [[round(x, 2), round(y, 2)] for x, y in polygon.exterior.coords],
        "area_px": round(polygon.area, 2),
        "confidence": round(confidence, 4),
    }


def labels_to_instances(
    labels: np.ndarray,
    label_scores: dict[int, float],
    epsilon: float = 2.0,
    min_area: float = 100.0,
    simplify_tolerance: Optional[float] = None,
) -> list[dict[str, Any]]:
    """Vectorize an int32 label image into instance dicts for the output JSON."""
    if not label_scores:
        return []
    import geoai

    with tempfile.TemporaryDirectory() as tmp:
        tif = Path(tmp) / "labels.tif"
        h, w = labels.shape
        # No transform on purpose: polygons come out in pixel coordinates.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NotGeoreferencedWarning)
            with rasterio.open(tif, "w", driver="GTiff", height=h, width=w, count=1,
                               dtype="int32") as dst:
                dst.write(labels, 1)
            gdf = geoai.orthogonalize(str(tif), epsilon=epsilon, min_area=min_area)

    if gdf is None or len(gdf) == 0:
        return []

    best: dict[int, Any] = {}  # label -> largest polygon of that label
    for value, geom in zip(gdf["value"], gdf.geometry):
        if geom is None or geom.is_empty or geom.area < min_area:
            continue
        value = int(value)
        if value not in best or geom.area > best[value].area:
            best[value] = geom

    return [
        _make_instance(k, best[value], label_scores[value], simplify_tolerance)
        for k, value in enumerate(sorted(best), start=1)
    ]
