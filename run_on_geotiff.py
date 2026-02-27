"""
Minimal wrapper: run existing SAM3 pipeline on GeoTIFF images, then
reorganise outputs into results/masks/ and results/overlays/.

NO changes to SAM3_Claude package. This script only:
  1. Calls the existing run_pipeline() with appropriate config.
  2. Post-processes outputs: mask TIF → binary PNG, annotation PNG → overlay PNG.

Usage:
    conda run -n geoai_sam python run_on_geotiff.py \
        --input-dir data/interim/chips_600m/cell_00047/pre \
        --output-dir results \
        [--device cuda:0]
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# -- make sure the package is importable even without `pip install -e` --
sys.path.insert(0, str(Path(__file__).parent / "SAM3_Claude"))

from sam3_building_identifier.config import PipelineConfig
from sam3_building_identifier.pipeline import run_pipeline
from sam3_building_identifier.utils import log


def postprocess(stem: str, tmp_dir: Path, out_dir: Path) -> None:
    """Convert pipeline outputs to mask.png + overlay.png with expected naming."""
    masks_out = out_dir / "masks"
    overlays_out = out_dir / "overlays"
    masks_out.mkdir(parents=True, exist_ok=True)
    overlays_out.mkdir(parents=True, exist_ok=True)

    # --- mask TIF → binary PNG ---
    mask_tif = tmp_dir / "masks" / f"{stem}.tif"
    if mask_tif.exists():
        arr = np.array(Image.open(mask_tif))
        # Instance IDs > 0 are buildings; convert to binary 0/255
        binary = (arr > 0).astype(np.uint8) * 255
        out_path = masks_out / f"{stem}_mask.png"
        Image.fromarray(binary).save(out_path)
        log(f"  mask  → {out_path}")
    else:
        log(f"  WARN  mask TIF not found for {stem} (zero detections?)")

    # --- annotation PNG → overlay PNG ---
    ann_png = tmp_dir / "annotations" / f"{stem}_ann.png"
    if ann_png.exists():
        out_path = overlays_out / f"{stem}_overlay.png"
        shutil.copy(ann_png, out_path)
        log(f"  overlay → {out_path}")
    else:
        log(f"  WARN  annotation PNG not found for {stem} (zero detections?)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SAM3 pipeline on GeoTIFF images.")
    parser.add_argument("--input-dir", required=True,
                        help="Directory containing .tif images.")
    parser.add_argument("--output-dir", default="results",
                        help="Root output directory (default: results/).")
    parser.add_argument("--device", default=None,
                        help="Torch device, e.g. cuda:0 (default: auto-detect).")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Limit number of images to process.")
    parser.add_argument("--no-skip", action="store_true",
                        help="Re-run even if prediction JSON already exists.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    tmp_dir = output_dir / "_pipeline_tmp"

    log(f"Input  dir : {input_dir}")
    log(f"Output dir : {output_dir}")
    log(f"Tmp dir    : {tmp_dir}")

    cfg = PipelineConfig(
        input_dir=str(input_dir),
        output_dir=str(tmp_dir),
        disaster_type="all",          # no xView2 naming → process everything
        device=args.device,
        max_images=args.max_images,
        skip_existing=not args.no_skip,
        save_masks=True,
        save_annotations=True,
    )

    # --- run the unmodified pipeline ---
    run_pipeline(cfg)

    # --- post-process each image to results/masks/ and results/overlays/ ---
    log("\nPost-processing outputs ...")
    tif_files = sorted(input_dir.glob("*.tif")) + sorted(input_dir.glob("*.tiff"))
    if args.max_images is not None:
        tif_files = tif_files[: args.max_images]

    for tif in tif_files:
        postprocess(tif.stem, tmp_dir, output_dir)

    log(f"\nDone. Results in {output_dir}/masks/ and {output_dir}/overlays/")


if __name__ == "__main__":
    main()
