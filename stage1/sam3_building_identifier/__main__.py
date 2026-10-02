"""
CLI entry point.

    python -m sam3_building_identifier --input-dir <dir> --output-dir <dir> [options]
    python -m sam3_building_identifier --help
"""

from __future__ import annotations

import argparse
import logging
import sys

from sam3_building_identifier.config import PipelineConfig
from sam3_building_identifier.pipeline import run_pipeline
from sam3_building_identifier.utils import discover_images, log

D = PipelineConfig()  # defaults


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m sam3_building_identifier",
        description="Detect buildings with SAM3 and write per-instance JSON predictions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input-dir", "-i", required=True, help="Directory of input images.")
    p.add_argument("--output-dir", "-o", required=True, help="Root output directory.")
    p.add_argument("--disaster-type", "-d", choices=["auto", "pre", "post", "all"],
                   default=D.disaster_type, help="Filter by xView2 filename suffix.")
    p.add_argument("--max-images", "-n", type=int, default=None, help="Process at most N images.")

    p.add_argument("--device", default=None, help="Torch device, e.g. cuda, cuda:1, cpu.")
    p.add_argument("--backend", default=D.backend, help="SamGeo3 backend (meta or transformers).")
    p.add_argument("--model", default=D.model_id, dest="model_id",
                   help="facebook/sam3 or facebook/sam3.1.")
    p.add_argument("--confidence-threshold", type=float, default=D.confidence_threshold,
                   help="SAM3 detection score threshold.")
    p.add_argument("--checkpoint-path", default=None, help="Local SAM3 checkpoint.")
    p.add_argument("--no-hf", action="store_false", dest="load_from_hf",
                   help="Do not load weights from Hugging Face.")

    p.add_argument("--prompt", default=D.text_prompt, dest="text_prompt", help="Text prompt.")
    p.add_argument("--min-size", type=int, default=D.min_size, help="Minimum mask area (px).")
    p.add_argument("--max-size", type=int, default=None, help="Maximum mask area (px).")
    p.add_argument("--tile-size", type=int, default=D.tile_size,
                   help="Window size for tiled inference; 0 runs on the full image.")
    p.add_argument("--overlap", type=int, default=D.tile_overlap, help="Minimum window overlap (px).")
    p.add_argument("--merge-iou", type=float, default=D.merge_iou,
                   help="IoU in the shared overlap above which cross-window instances merge.")

    p.add_argument("--epsilon", type=float, default=D.polygon_epsilon, dest="polygon_epsilon",
                   help="Polygon simplification tolerance (px).")
    p.add_argument("--min-polygon-area", type=float, default=D.min_polygon_area,
                   help="Drop polygons smaller than this (px^2).")
    p.add_argument("--simplify-tolerance", type=float, default=None,
                   help="Extra Shapely simplify() tolerance.")

    p.add_argument("--no-masks", action="store_false", dest="save_masks", help="Skip mask TIFs.")
    p.add_argument("--no-annotations", action="store_false", dest="save_annotations",
                   help="Skip annotation PNGs.")
    p.add_argument("--no-skip", action="store_false", dest="skip_existing",
                   help="Re-process images that already have a prediction JSON.")
    p.add_argument("--dry-run", action="store_true", help="List images and exit.")
    p.add_argument("--verbose", "-v", action="store_true", help="DEBUG logging.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(format="%(levelname)-8s %(name)s: %(message)s",
                        level=logging.DEBUG if args.verbose else logging.INFO, stream=sys.stderr)

    options = vars(args)
    dry_run = options.pop("dry_run")
    options.pop("verbose")
    options["tile_overlap"] = options.pop("overlap")
    options["tile_size"] = options["tile_size"] or None
    cfg = PipelineConfig(**options)

    if dry_run:
        images = discover_images(cfg.input_dir, cfg.disaster_type,
                                 tuple(cfg.image_extensions), cfg.max_images)
        log(f"DRY-RUN: would process {len(images)} images")
        for path in images:
            print(path)
        return 0

    totals = run_pipeline(cfg)
    if not totals:
        return 1
    return 0 if totals["images_error"] == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
