"""
Tiled inference support: window layout and cross-window instance merging.

SAM3 resizes every input to 1008 px, so small buildings in a large image
lose detail. We run SAM3 on overlapping windows and merge the per-window
instances back into one set of full-image instances.

Merging rule: two instances from different windows are the same object when
their masks agree inside the region that both windows see (IoU >= merge_iou
within the shared overlap). A merged instance takes the union of the masks
and the highest score.

This replaces the earlier pixel-wise stitching (keep the higher per-pixel
score at overlaps), which split any building crossing a window seam into
several labels. See reports/sam3_audit_2026-09.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

Window = tuple[int, int, int, int]  # (x, y, w, h) in full-image pixels


@dataclass
class Instance:
    """One detected object, with its mask cropped to its bounding box."""

    mask: np.ndarray  # bool, shape (y1 - y0, x1 - x0)
    x0: int
    y0: int
    score: float
    window: Optional[Window] = None  # source window; None after merging

    @property
    def x1(self) -> int:
        return self.x0 + self.mask.shape[1]

    @property
    def y1(self) -> int:
        return self.y0 + self.mask.shape[0]

    @property
    def area(self) -> int:
        return int(self.mask.sum())


# ---------------------------------------------------------------------------
# Window layout
# ---------------------------------------------------------------------------

def _positions(length: int, tile: int, overlap: int) -> list[int]:
    if length <= tile:
        return [0]
    n = int(np.ceil((length - overlap) / (tile - overlap)))
    return [int(round(p)) for p in np.linspace(0, length - tile, n)]


def tile_windows(width: int, height: int, tile_size: int, overlap: int) -> list[Window]:
    """Evenly spaced windows covering the image.

    Adjacent windows overlap by at least *overlap* pixels. Unlike a fixed
    stride, the last window is aligned to the image edge, so there are no
    narrow sliver tiles (a 1024 px image gives 3x3 full 512 px windows).
    """
    if overlap >= tile_size:
        raise ValueError(f"overlap ({overlap}) must be smaller than tile_size ({tile_size})")
    return [
        (x, y, min(tile_size, width), min(tile_size, height))
        for y in _positions(height, tile_size, overlap)
        for x in _positions(width, tile_size, overlap)
    ]


# ---------------------------------------------------------------------------
# Instance construction and merging
# ---------------------------------------------------------------------------

def instance_from_mask(mask: np.ndarray, window: Window, score: float) -> Optional[Instance]:
    """Build an Instance from a window-sized boolean mask."""
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    return Instance(
        mask=mask[y0:y1, x0:x1].copy(),
        x0=window[0] + x0,
        y0=window[1] + y0,
        score=float(score),
        window=window,
    )


def _crop(inst: Instance, rx0: int, ry0: int, rx1: int, ry1: int) -> np.ndarray:
    """The instance mask restricted to a region, in region coordinates."""
    out = np.zeros((ry1 - ry0, rx1 - rx0), dtype=bool)
    ix0, iy0 = max(inst.x0, rx0), max(inst.y0, ry0)
    ix1, iy1 = min(inst.x1, rx1), min(inst.y1, ry1)
    if ix0 < ix1 and iy0 < iy1:
        out[iy0 - ry0:iy1 - ry0, ix0 - rx0:ix1 - rx0] = \
            inst.mask[iy0 - inst.y0:iy1 - inst.y0, ix0 - inst.x0:ix1 - inst.x0]
    return out


def _same_object(a: Instance, b: Instance, merge_iou: float) -> bool:
    (ax, ay, aw, ah), (bx, by, bw, bh) = a.window, b.window
    # Region that both windows see, limited to the two bounding boxes.
    rx0 = max(ax, bx, min(a.x0, b.x0))
    ry0 = max(ay, by, min(a.y0, b.y0))
    rx1 = min(ax + aw, bx + bw, max(a.x1, b.x1))
    ry1 = min(ay + ah, by + bh, max(a.y1, b.y1))
    if rx0 >= rx1 or ry0 >= ry1:
        return False
    ma = _crop(a, rx0, ry0, rx1, ry1)
    mb = _crop(b, rx0, ry0, rx1, ry1)
    inter = np.count_nonzero(ma & mb)
    if inter == 0:
        return False
    return inter / np.count_nonzero(ma | mb) >= merge_iou


def merge_instances(instances: list[Instance], merge_iou: float = 0.5) -> list[Instance]:
    """Merge instances from different windows that are the same object."""
    n = len(instances)
    if n < 2:
        return list(instances)

    boxes = np.array([[i.x0, i.y0, i.x1, i.y1] for i in instances])
    overlaps = (
        (boxes[:, None, 0] < boxes[None, :, 2]) & (boxes[None, :, 0] < boxes[:, None, 2])
        & (boxes[:, None, 1] < boxes[None, :, 3]) & (boxes[None, :, 1] < boxes[:, None, 3])
    )

    parent = list(range(n))

    def find(k: int) -> int:
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    for i, j in zip(*np.nonzero(np.triu(overlaps, k=1))):
        a, b = instances[i], instances[j]
        if a.window == b.window:
            continue
        if find(i) != find(j) and _same_object(a, b, merge_iou):
            parent[find(i)] = find(j)

    groups: dict[int, list[Instance]] = {}
    for k in range(n):
        groups.setdefault(find(k), []).append(instances[k])

    merged = []
    for members in groups.values():
        if len(members) == 1:
            merged.append(members[0])
            continue
        x0 = min(m.x0 for m in members)
        y0 = min(m.y0 for m in members)
        x1 = max(m.x1 for m in members)
        y1 = max(m.y1 for m in members)
        mask = np.zeros((y1 - y0, x1 - x0), dtype=bool)
        for m in members:
            mask[m.y0 - y0:m.y1 - y0, m.x0 - x0:m.x1 - x0] |= m.mask
        merged.append(Instance(mask, x0, y0, max(m.score for m in members)))
    return merged


def paint_labels(
    instances: list[Instance], height: int, width: int,
) -> tuple[np.ndarray, np.ndarray, dict[int, float]]:
    """Rasterize instances into a label image and a score image.

    Higher-score instances are painted last, so they win where masks overlap.
    Returns (labels int32, scores float32, {label: score}).
    """
    labels = np.zeros((height, width), dtype=np.int32)
    scores = np.zeros((height, width), dtype=np.float32)
    label_scores: dict[int, float] = {}
    for label, inst in enumerate(sorted(instances, key=lambda i: i.score), start=1):
        region = (slice(inst.y0, inst.y1), slice(inst.x0, inst.x1))
        labels[region][inst.mask] = label
        scores[region][inst.mask] = inst.score
        label_scores[label] = inst.score
    return labels, scores, label_scores
