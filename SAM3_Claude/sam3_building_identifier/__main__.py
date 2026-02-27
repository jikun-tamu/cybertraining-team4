"""
CLI entrypoint.

Usage examples
--------------
# Run on all pre-disaster images in default dir:
    python -m sam3_building_identifier

# Run on 5 images, send output to /tmp/test_out:
    python -m sam3_building_identifier \\
        --input-dir /media/data/building_instance_tamu/test/images \\
        --output-dir /tmp/test_out \\
        --max-images 5

# Post-disaster images, CUDA device 1:
    python -m sam3_building_identifier \\
        --disaster-type post \\
        --device cuda:1

# Dry-run: discover images but do NOT run inference:
    python -m sam3_building_identifier --dry-run --max-images 10

Full list of options:  python -m sam3_building_identifier --help
"""

from __future__ import annotations

import argparse
import logging
import sys

from sam3_building_identifier.config import PipelineConfig
from sam3_building_identifier.pipeline import run_pipeline
from sam3_building_identifier.utils import discover_images, log


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m sam3_building_identifier",
        description=(
            "SAM3 Building Identifier — detect buildings in xView2-style "
            "satellite imagery and output per-instance JSON predictions."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- I/O ----
    io = p.add_argument_group("I/O")
    io.add_argument(
        "--input-dir", "-i",
        default=PipelineConfig.input_dir,
        help="Directory containing input images.",
    )
    io.add_argument(
        "--output-dir", "-o",
        default=PipelineConfig.output_dir,
        help="Root output directory (sub-folders created automatically).",
    )

    # ---- Filtering ----
    filt = p.add_argument_group("Image filtering")
    filt.add_argument(
        "--disaster-type", "-d",
        choices=["auto", "pre", "post", "all"],
        default="auto",
        help=(
            "Which images to process. "
            "'auto' (default): use pre-disaster if that naming exists, else all. "
            "'pre'/'post': filter by xView2 suffix. "
            "'all': every image in the folder."
        ),
    )
    filt.add_argument(
        "--max-images", "-n",
        type=int,
        default=None,
        metavar="N",
        help="Process at most N images (useful for quick tests).",
    )

    # ---- Model ----
    mdl = p.add_argument_group("Model")
    mdl.add_argument(
        "--device",
        default=None,
        help="Torch device: 'cuda', 'cuda:0', 'cpu'.  None → auto-detect.",
    )
    mdl.add_argument(
        "--backend",
        default="meta",
        help="SamGeo3 backend identifier.",
    )
    mdl.add_argument(
        "--checkpoint-path",
        default=None,
        help="Local checkpoint path.  None → download from Hugging Face.",
    )
    mdl.add_argument(
        "--no-hf",
        action="store_false",
        dest="load_from_hf",
        help="Do not load from Hugging Face (use local checkpoint instead).",
    )

    # ---- Inference ----
    inf = p.add_argument_group("Inference")
    inf.add_argument(
        "--prompt",
        default="building",
        dest="text_prompt",
        help="Text prompt passed to SamGeo3.generate_masks().",
    )
    inf.add_argument(
        "--min-size",
        type=int,
        default=100,
        help="Minimum mask area (pixels) — smaller masks are discarded.",
    )
    inf.add_argument(
        "--max-size",
        type=int,
        default=None,
        help="Maximum mask area (pixels).  None → no upper limit.",
    )
    inf.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of images per SAM3 batch call (recommend 1 for GPU memory).",
    )

    # ---- Polygon ----
    poly = p.add_argument_group("Polygon / vectorization")
    poly.add_argument(
        "--epsilon",
        type=float,
        default=2.0,
        dest="polygon_epsilon",
        help="Polygon simplification tolerance (pixels) for geoai.orthogonalize.",
    )
    poly.add_argument(
        "--min-polygon-area",
        type=float,
        default=10.0,
        help="Minimum polygon area (sq-pixels) after vectorization.",
    )
    poly.add_argument(
        "--simplify-tolerance",
        type=float,
        default=None,
        help="Optional extra Shapely simplify() tolerance.  None → skip.",
    )

    # ---- Output ----
    out = p.add_argument_group("Output control")
    out.add_argument(
        "--no-masks",
        action="store_false",
        dest="save_masks",
        help="Do not save mask/scores TIF files.",
    )
    out.add_argument(
        "--no-annotations",
        action="store_false",
        dest="save_annotations",
        help="Do not save annotation PNG files.",
    )
    out.add_argument(
        "--annotation-dpi",
        type=int,
        default=150,
        help="DPI for annotation PNGs.",
    )
    out.add_argument(
        "--no-skip",
        action="store_false",
        dest="skip_existing",
        help="Re-process images even if prediction JSON already exists.",
    )

    # ---- Misc ----
    misc = p.add_argument_group("Misc")
    misc.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Discover images and print what would be processed, "
            "then exit without running inference."
        ),
    )
    misc.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG logging.",
    )

    return p.parse_args(argv)


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(levelname)-8s %(name)s — %(message)s",
        level=level,
        stream=sys.stderr,
    )


def main(argv=None) -> int:
    args = _parse_args(argv)
    _configure_logging(args.verbose)

    # Build config from CLI args
    cfg = PipelineConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        disaster_type=args.disaster_type,
        max_images=args.max_images,
        device=args.device,
        backend=args.backend,
        checkpoint_path=args.checkpoint_path,
        load_from_hf=args.load_from_hf,
        text_prompt=args.text_prompt,
        min_size=args.min_size,
        max_size=args.max_size,
        batch_size=args.batch_size,
        polygon_epsilon=args.polygon_epsilon,
        min_polygon_area=args.min_polygon_area,
        simplify_tolerance=args.simplify_tolerance,
        save_masks=args.save_masks,
        save_annotations=args.save_annotations,
        save_predictions=True,
        annotation_dpi=args.annotation_dpi,
        skip_existing=args.skip_existing,
    )

    # Dry-run: just list images and exit
    if args.dry_run:
        images = discover_images(
            input_dir=cfg.input_dir,
            disaster_type=cfg.disaster_type,
            extensions=tuple(cfg.image_extensions),
            max_images=cfg.max_images,
        )
        log(f"DRY-RUN: would process {len(images)} images")
        for p in images:
            print(p)
        return 0

    summary = run_pipeline(cfg)

    if not summary:
        return 1
    return 0 if summary.get("errors", 0) == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
