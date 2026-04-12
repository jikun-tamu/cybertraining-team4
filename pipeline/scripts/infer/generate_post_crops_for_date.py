#!/usr/bin/env python3
"""Generate post-disaster crops for one date, reusing pre-computed crop geometry.

This is a lightweight companion to generate_shared_instance_subimages.py.
Instead of re-reading Stage-1 JSON and recomputing polygon geometry, it reads
crop_x0 / crop_y0 / crop_size directly from an existing shared_instance_samples.csv
(the "base" shared CSV produced for any prior post date).

Only the post_crop column changes; pre_crop, mask_M, mask_R are unchanged.

Instance-level quality: if a crop has > CROP_MAX_ZERO_FRAC zero pixels, the row
is written with quality_ok=false so the aggregator can skip it.

Output
------
  {out_dir}/post_crops/{tile_id}__{bldg_uid}.png  — one PNG per building
  {out_dir}/shared_for_date.csv                   — base CSV with post_crop swapped
  {out_dir}/quality_metrics.json                  — tile-level quality result
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Allow importing shared modules (quality_filter, la_fire_paths) from scripts/ root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from quality_filter import assess_tile_quality, assess_crop_quality


def parse_args():
    p = argparse.ArgumentParser(description="Generate post crops for one post date.")
    p.add_argument("--base_shared_csv", type=Path, required=True,
                   help="Existing shared_instance_samples.csv (has crop_x0, crop_y0, …)")
    p.add_argument("--post_image", type=Path, required=True,
                   help="Path to the post-disaster image for this date (TIF or PNG).")
    p.add_argument("--date", type=str, required=True, help="Date string, e.g. 20250109.")
    p.add_argument("--out_dir", type=Path, required=True,
                   help="Output directory for this date.")
    p.add_argument("--png_compress_level", type=int, default=1)
    p.add_argument("--overwrite", action="store_true")
    # Quality filter thresholds (tile-level)
    p.add_argument("--min_mean", type=float, default=15.0)
    p.add_argument("--max_zeros", type=float, default=0.60)
    p.add_argument("--min_std", type=float, default=3.0)
    # Quality filter threshold (crop-level)
    p.add_argument("--crop_max_zero_frac", type=float, default=0.50)
    return p.parse_args()


def open_image_rgb(path: Path) -> Image.Image:
    """Open any image (TIF or PNG) as PIL RGB."""
    suffix = path.suffix.lower()
    if suffix in (".tif", ".tiff"):
        import rasterio
        with rasterio.open(path) as ds:
            arr = ds.read()  # (C, H, W) uint8
        if arr.shape[0] >= 3:
            arr = arr[:3]
        arr = arr.transpose(1, 2, 0)  # (H, W, C)
        return Image.fromarray(arr.astype(np.uint8), mode="RGB")
    else:
        return Image.open(path).convert("RGB")


def main():
    args = parse_args()

    if not args.base_shared_csv.exists():
        sys.exit(f"ERROR: base_shared_csv not found: {args.base_shared_csv}")
    if not args.post_image.exists():
        sys.exit(f"ERROR: post_image not found: {args.post_image}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    crops_dir = args.out_dir / "post_crops"
    crops_dir.mkdir(exist_ok=True)

    out_csv = args.out_dir / "shared_for_date.csv"
    quality_json = args.out_dir / "quality_metrics.json"

    # ── Tile-level quality filter ─────────────────────────────────────────────
    tile_ok, tile_metrics = assess_tile_quality(
        args.post_image, args.min_mean, args.max_zeros, args.min_std
    )
    tile_metrics["date"] = args.date
    tile_metrics["post_image"] = str(args.post_image)
    tile_metrics["tile_quality_ok"] = tile_ok
    quality_json.write_text(json.dumps(tile_metrics, indent=2))
    print(f"[quality] date={args.date} tile_ok={tile_ok}  {tile_metrics['reason']}")

    # ── Load post image (even if tile fails, generate crops but mark quality_ok=false) ──
    try:
        post_img = open_image_rgb(args.post_image)
    except Exception as e:
        print(f"[error] Cannot open post image: {e}")
        # Write an empty CSV so the aggregator knows this date failed completely
        with open(args.base_shared_csv) as f_in:
            reader = csv.DictReader(f_in)
            base_fields = reader.fieldnames or []
        out_fields = base_fields + ["post_crop", "quality_ok", "crop_zero_frac"]
        # Remove duplicates preserving order
        seen = set()
        out_fields = [x for x in out_fields if not (x in seen or seen.add(x))]
        with open(out_csv, "w", newline="") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=out_fields)
            writer.writeheader()
        return

    # ── Process each building instance ───────────────────────────────────────
    with open(args.base_shared_csv, newline="") as f_in:
        reader = csv.DictReader(f_in)
        base_fields = list(reader.fieldnames or [])
        rows = list(reader)

    # Build output fieldnames: base fields + new columns (avoid duplicates)
    extra_cols = ["post_crop", "quality_ok", "crop_zero_frac"]
    out_fields = base_fields.copy()
    for col in extra_cols:
        if col not in out_fields:
            out_fields.append(col)
    # post_crop is already in base_fields — it will be overwritten in the row dict

    written = skipped_geo = skipped_existing = 0

    with open(out_csv, "w", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=out_fields)
        writer.writeheader()

        for row in rows:
            bldg_uid = row.get("bldg_uid", "")
            tile_id = row.get("tile_id", "")
            stem = f"{tile_id}__{bldg_uid}".replace("/", "_")
            crop_out = crops_dir / f"{stem}.png"

            # Parse crop geometry from base CSV
            try:
                x0 = int(row["crop_x0"])
                y0 = int(row["crop_y0"])
                crop_size = int(row["crop_size"])
            except (KeyError, ValueError):
                skipped_geo += 1
                continue

            if not args.overwrite and crop_out.exists():
                # Reuse existing crop, but still compute quality flag
                try:
                    existing = np.array(Image.open(crop_out).convert("RGB"))
                    crop_ok, zero_frac = assess_crop_quality(existing, args.crop_max_zero_frac)
                except Exception:
                    crop_ok, zero_frac = False, 1.0
                out_row = dict(row)
                out_row["post_crop"] = str(crop_out)
                out_row["quality_ok"] = str(tile_ok and crop_ok).lower()
                out_row["crop_zero_frac"] = str(zero_frac)
                writer.writerow(out_row)
                written += 1
                skipped_existing += 1
                continue

            # Crop from post image
            w, h = post_img.size
            x1 = min(x0 + crop_size, w)
            y1 = min(y0 + crop_size, h)
            crop_pil = post_img.crop((x0, y0, x1, y1))

            # Pad if needed (edge case: crop window exceeds image boundary)
            if crop_pil.size != (crop_size, crop_size):
                padded = Image.new("RGB", (crop_size, crop_size), (0, 0, 0))
                padded.paste(crop_pil, (0, 0))
                crop_pil = padded

            crop_arr = np.array(crop_pil)
            crop_ok, zero_frac = assess_crop_quality(crop_arr, args.crop_max_zero_frac)

            crop_pil.save(crop_out, compress_level=args.png_compress_level)

            out_row = dict(row)
            out_row["post_crop"] = str(crop_out)
            out_row["quality_ok"] = str(tile_ok and crop_ok).lower()
            out_row["crop_zero_frac"] = str(zero_frac)
            writer.writerow(out_row)
            written += 1

    print(f"[done] date={args.date}  written={written}  skipped_geo={skipped_geo}"
          f"  skipped_existing={skipped_existing}  out={out_csv}")


if __name__ == "__main__":
    main()
