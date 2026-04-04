"""
Image tiling and mask stitching for SAM3 building detection.

SAM3's vision encoder internally resizes images to ~1024×1024. For large
images (e.g. 1966×1966), this downscaling causes small buildings to be
missed. Tiling the image into 512×512 patches with overlap, running SAM3
on each tile, and stitching results back yields much better recall.

Default: tile_size=512, overlap=64 (matching validated legacy pipeline).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image


def generate_tiles(
    image_path: Path,
    tiles_dir: Path,
    tile_size: int = 512,
    overlap: int = 64,
) -> list[dict]:
    """Split an image into overlapping tiles and save as PNGs.

    Returns a list of dicts: {path, x, y, w, h} for each tile.
    """
    tiles_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem

    with Image.open(image_path) as im:
        img_w, img_h = im.size
        stride = max(1, tile_size - overlap)
        tiles = []

        for y in range(0, img_h, stride):
            for x in range(0, img_w, stride):
                x2 = min(x + tile_size, img_w)
                y2 = min(y + tile_size, img_h)
                if x2 - x <= 0 or y2 - y <= 0:
                    continue

                tw, th = x2 - x, y2 - y
                tile = im.crop((x, y, x2, y2))
                tile_name = f"{stem}_x{x}_y{y}_w{tw}_h{th}.png"
                tile_path = tiles_dir / tile_name
                tile.save(tile_path)

                tiles.append({"path": tile_path, "x": x, "y": y, "w": tw, "h": th})

    return tiles


def stitch_masks(
    tile_mask_paths: list[Path],
    tile_score_paths: list[Path],
    image_size_hw: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Stitch per-tile mask and score TIFs back into full-image arrays.

    Uses score-based replacement at overlaps: keeps the higher-confidence
    prediction when tiles overlap.

    Returns (label_mask, score_mask) as full-image numpy arrays.
    """
    import rasterio

    h, w = image_size_hw
    canvas_mask = np.zeros((h, w), dtype=np.int32)
    canvas_score = np.zeros((h, w), dtype=np.float32)
    max_label = 0

    # Build score lookup by matching stems
    score_map = {p.stem.replace("_scores", ""): p for p in tile_score_paths}

    for mp in tile_mask_paths:
        meta = _parse_tile_meta(mp.stem)
        if meta is None:
            continue
        x, y, tw, th = meta

        with rasterio.open(mp) as src:
            tile_mask = src.read(1)

        sp = score_map.get(mp.stem)
        tile_score = None
        if sp and sp.exists():
            with rasterio.open(sp) as ssrc:
                tile_score = ssrc.read(1)

        # Clip to actual tile dimensions
        tile_mask = tile_mask[:th, :tw]
        if tile_score is not None:
            tile_score = tile_score[:th, :tw]

        # Offset labels to keep unique IDs across tiles
        offset_mask = tile_mask.copy()
        offset_mask[offset_mask > 0] += max_label

        region_mask = canvas_mask[y: y + th, x: x + tw]
        region_score = canvas_score[y: y + th, x: x + tw]

        if tile_score is not None:
            # Replace where new tile has higher confidence
            replace = tile_score > region_score
            region_mask[replace] = offset_mask[replace]
            region_score[replace] = tile_score[replace]
        else:
            # Replace only empty pixels
            replace = region_mask == 0
            region_mask[replace] = offset_mask[replace]

        max_label = max(max_label, int(offset_mask.max()))
        canvas_mask[y: y + th, x: x + tw] = region_mask
        canvas_score[y: y + th, x: x + tw] = region_score

    return canvas_mask, canvas_score


def _parse_tile_meta(name: str) -> Optional[tuple[int, int, int, int]]:
    """Extract (x, y, w, h) from tile filename like stem_x0_y448_w512_h512."""
    m = re.search(r"_x(\d+)_y(\d+)_w(\d+)_h(\d+)", name)
    if not m:
        return None
    return tuple(int(v) for v in m.groups())
