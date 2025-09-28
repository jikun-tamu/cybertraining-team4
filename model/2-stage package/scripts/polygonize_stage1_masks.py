#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from tqdm import tqdm


def find_polygons_from_mask(mask: np.ndarray, min_area: int = 50, simplify_tol: float = 1.5) -> List[Polygon]:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys: List[Polygon] = []
    for cnt in contours:
        if len(cnt) < 3:
            continue
        area = cv2.contourArea(cnt)
        if area < float(min_area):
            continue
        pts = cnt.reshape(-1, 2)
        # Convert to shapely polygon in image pixel coords (x, y)
        poly = Polygon([(float(x), float(y)) for x, y in pts])
        if not poly.is_valid or poly.area <= 0:
            continue
        if simplify_tol and simplify_tol > 0:
            poly = poly.simplify(simplify_tol, preserve_topology=True)
            if not poly.is_valid or poly.area <= 0:
                continue
        polys.append(poly)
    return polys


def write_xbd_json(out_path: Path, polygons: List[Polygon]):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    features = []
    for i, poly in enumerate(polygons):
        # Force to MultiPolygon-safe WKT (handles holes automatically)
        wkt_str = poly.wkt
        features.append({
            "properties": {
                "feature_type": "building",
                "uid": f"pred_{i:06d}",
            },
            "wkt": wkt_str,
        })

    data = {
        "features": {
            "xy": features
        },
        "metadata": {}
    }

    with open(out_path, 'w') as f:
        json.dump(data, f)


def main():
    parser = argparse.ArgumentParser(description="Polygonize Stage-1 masks into xBD-style JSON with WKT")
    parser.add_argument('--masks_dir', required=True, help='Directory with *_pre_mask.png files')
    parser.add_argument('--out_json_dir', required=True, help='Directory to write JSON labels')
    parser.add_argument('--min_area', type=int, default=50, help='Minimum component area in px')
    parser.add_argument('--simplify_tol', type=float, default=1.5, help='Polygon simplify tolerance (pixels)')
    args = parser.parse_args()

    masks = sorted([p for p in Path(args.masks_dir).glob('*_pre_mask.png')])
    assert len(masks) > 0, f"No *_pre_mask.png found under {args.masks_dir}"

    processed = 0
    skipped = 0
    it = tqdm(masks, total=len(masks), desc='Polygonize Stage1 masks', unit='mask')
    for mpath in it:
        try:
            mask = cv2.imread(str(mpath), cv2.IMREAD_GRAYSCALE)
            mask = (mask > 127).astype(np.uint8)
            polys = find_polygons_from_mask(mask, min_area=args.min_area, simplify_tol=args.simplify_tol)
            # Match xBD naming: base stem should correspond to the original tile basename
            # Example: <tile>_pre_mask.png -> <tile>_pre_disaster.json
            base = mpath.name.replace('_pre_mask.png', '')
            out_name = f"{base}_pre_disaster.json"
            out_path = Path(args.out_json_dir) / out_name
            write_xbd_json(out_path, polys)
            processed += 1
        except Exception:
            skipped += 1
        try:
            it.set_postfix({"ok": processed, "skip": skipped})
        except Exception:
            pass
    print(f"Wrote JSONs to {args.out_json_dir} | processed={processed} | skipped={skipped}")


if __name__ == '__main__':
    main()


