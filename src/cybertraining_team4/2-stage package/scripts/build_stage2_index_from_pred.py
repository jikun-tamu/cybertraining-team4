#!/usr/bin/env python3
import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, Any, Tuple


def load_json(path: str) -> Dict[str, Any]:
    with open(path, 'r') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description='Build Stage-2 per-building index from predicted Stage-1 pre JSONs (xBD-style)')
    parser.add_argument('--images_root', required=True, help='Root containing images/*.png tiles')
    parser.add_argument('--pred_labels_root', required=True, help='Root containing predicted *_pre_disaster.json')
    parser.add_argument('--out_csv', required=True, help='Output CSV path')
    parser.add_argument('--event_id', default='pred_event', help='Event id to assign')
    parser.add_argument('--width', type=int, default=1024)
    parser.add_argument('--height', type=int, default=1024)
    args = parser.parse_args()

    images_dir = Path(args.images_root)
    labels_dir = Path(args.pred_labels_root)

    rows = []
    # MODIFICATION: The custom pipeline creates a temp dir with one pre/post pair.
    # Find the single pre-image in the temp directory.
    try:
        pre_img = next(images_dir.rglob("pre/*.tif"))
        post_img = next(images_dir.rglob("post/*.tif"))
    except StopIteration:
        print(f"Error: Could not find pre/post .tif pair in {images_dir}")
        # Create an empty CSV and exit gracefully
        pre_img, post_img = None, None

    if pre_img and post_img:
        # From the pre-image name, determine the corresponding JSON file name.
        # The polygonizer uses the mask's name, which is based on the pre-image name.
        # e.g., 'cell_00045_pre.tif' (from stage1_infer) -> 'cell_00045_pre.tif_pre_disaster.json' (from polygonize)
        json_name = f"{pre_img.name}_pre_disaster.json"
        pre_json = labels_dir / json_name

        if not pre_json.exists():
            print(f"Warning: Corresponding JSON file not found at {pre_json}")
        else:
            data = load_json(str(pre_json))
            features = data.get('features', {}).get('xy', [])
            tile_id = pre_img.stem.replace('_pre', '') # e.g., 'cell_00045'

            for i, feat in enumerate(features):
                uid = feat.get('properties', {}).get('uid', f'pred_{i:06d}')
                wkt_pre = feat.get('wkt', '')
                # We do not have post polygons from Stage-1; leave blank
                rows.append({
                    'event_id': args.event_id,
                    'tile_id': tile_id,
                    'width': args.width,
                    'height': args.height,
                    'pre_image': str(pre_img),
                    'post_image': str(post_img),
                    'pre_json': str(pre_json),
                    'post_json': '',
                    'bldg_uid': uid,
                    'damage_subtype': '',
                    'damage_class': '',
                    'polygon_wkt_xy_pre': wkt_pre,
                    'polygon_wkt_xy_post': '',
                })

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', newline='') as f:
        fieldnames = [
            'event_id','tile_id','width','height','pre_image','post_image',
            'pre_json','post_json','bldg_uid','damage_subtype','damage_class',
            'polygon_wkt_xy_pre','polygon_wkt_xy_post'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"Wrote index with {len(rows)} rows to {out_path}")


if __name__ == '__main__':
    main()


