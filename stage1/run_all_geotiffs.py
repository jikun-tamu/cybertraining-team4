"""
Batch runner: apply SAM3 pipeline to ALL pre/post GeoTIFFs under chips_600m/.

Strategy
--------
1. Collect every pre/*.tif and post/*.tif under --chips-dir.
2. Symlink them into a single flat temp directory.
3. Run the existing pipeline ONCE (model loads once → efficient).
4. Post-process outputs → results/masks/<stem>_mask.png
                         results/overlays/<stem>_overlay.png

Usage
-----
    conda run -n geoai_sam python run_all_geotiffs.py \
        --chips-dir data/interim/chips_600m \
        --output-dir results \
        --device cuda:0

Resume after interruption:
    Add --skip-existing (default: on) — images whose mask/overlay already
    exist in results/ are skipped in the post-processing step.
    To rerun inference from scratch also pass --no-skip to the pipeline.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent / "SAM3_Claude"))

from sam3_building_identifier.config import PipelineConfig
from sam3_building_identifier.pipeline import run_pipeline
from sam3_building_identifier.utils import log


# ---------------------------------------------------------------------------
# Collect all TIF paths
# ---------------------------------------------------------------------------

def collect_tifs(chips_dir: Path) -> list[Path]:
    """Walk chips_dir and return all pre/*.tif and post/*.tif paths."""
    tifs = []
    for cell_dir in sorted(chips_dir.iterdir()):
        if not cell_dir.is_dir():
            continue
        for sub in ("pre", "post"):
            sub_dir = cell_dir / sub
            if sub_dir.is_dir():
                tifs.extend(sorted(sub_dir.glob("*.tif")))
                tifs.extend(sorted(sub_dir.glob("*.tiff")))
    return tifs


# ---------------------------------------------------------------------------
# Post-process one image result
# ---------------------------------------------------------------------------

def postprocess(stem: str, tmp_dir: Path, out_dir: Path) -> None:
    masks_out = out_dir / "masks"
    overlays_out = out_dir / "overlays"
    masks_out.mkdir(parents=True, exist_ok=True)
    overlays_out.mkdir(parents=True, exist_ok=True)

    mask_tif = tmp_dir / "masks" / f"{stem}.tif"
    if mask_tif.exists():
        arr = np.array(Image.open(mask_tif))
        binary = (arr > 0).astype(np.uint8) * 255
        Image.fromarray(binary).save(masks_out / f"{stem}_mask.png")

    ann_png = tmp_dir / "annotations" / f"{stem}_ann.png"
    if ann_png.exists():
        shutil.copy(ann_png, overlays_out / f"{stem}_overlay.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run SAM3 on all GeoTIFFs under chips_600m/."
    )
    parser.add_argument(
        "--chips-dir",
        default="data/interim/chips_600m",
        help="Root directory containing cell_XXXXX/pre/ and cell_XXXXX/post/ folders.",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Root output directory (default: results/).",
    )
    parser.add_argument("--device", default=None,
                        help="Torch device, e.g. cuda:0 (default: auto-detect).")
    parser.add_argument("--no-skip", action="store_true",
                        help="Re-run inference even if prediction JSON already exists.")
    args = parser.parse_args()

    chips_dir = Path(args.chips_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    tmp_dir = output_dir / "_pipeline_tmp"

    # -- collect all TIFs --
    all_tifs = collect_tifs(chips_dir)
    log(f"Found {len(all_tifs)} TIF files under {chips_dir}")

    if not all_tifs:
        log("Nothing to do.")
        return

    # -- symlink into a flat staging directory --
    staging_dir = output_dir / "_staging"
    staging_dir.mkdir(parents=True, exist_ok=True)

    new_links = 0
    for tif in all_tifs:
        link = staging_dir / tif.name
        if not link.exists():
            link.symlink_to(tif)
            new_links += 1
    log(f"Staging dir: {staging_dir}  ({new_links} new symlinks, "
        f"{len(all_tifs) - new_links} already present)")

    # -- run the pipeline ONCE on the flat staging dir --
    cfg = PipelineConfig(
        input_dir=str(staging_dir),
        output_dir=str(tmp_dir),
        disaster_type="all",
        device=args.device,
        skip_existing=not args.no_skip,
        save_masks=True,
        save_annotations=True,
    )
    run_pipeline(cfg)

    # -- post-process every TIF → mask.png + overlay.png --
    log("\nPost-processing outputs ...")
    masks_out = output_dir / "masks"
    overlays_out = output_dir / "overlays"
    done = skipped = empty = 0

    for tif in all_tifs:
        stem = tif.stem
        mask_out_path = masks_out / f"{stem}_mask.png"
        overlay_out_path = overlays_out / f"{stem}_overlay.png"

        # skip if both outputs already exist
        if mask_out_path.exists() and overlay_out_path.exists():
            skipped += 1
            continue

        mask_tif = tmp_dir / "masks" / f"{stem}.tif"
        ann_png = tmp_dir / "annotations" / f"{stem}_ann.png"

        if not mask_tif.exists() and not ann_png.exists():
            empty += 1  # zero-detection image — nothing to save
            continue

        postprocess(stem, tmp_dir, output_dir)
        done += 1

    log(f"  Post-processed : {done}")
    log(f"  Already existed: {skipped}")
    log(f"  Empty (0 detections): {empty}")
    log(f"\nDone. Outputs in:")
    log(f"  {masks_out}")
    log(f"  {overlays_out}")


if __name__ == "__main__":
    main()
