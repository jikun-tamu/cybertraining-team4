#!/usr/bin/env python3
"""Prune cells whose PRE chip is blank (no useful pixels).

For each `cell_XXXXX` under `data/chips_600m` this script will:
- open the pre chip (expected at `pre/cell_XXXXX_pre.tif`)
- if the image is blank or near-zero (all channels max <= threshold),
  - delete the entire cell folder (pre + post)
  - delete the panel PNG under `data/derived/pre_post_panels/{cell_name}.png` if present
  - remove that cell row from the grid GeoPackage layer (writes a temp gpkg and replaces original)
  - remove any rows referencing that cell_id in `data/derived/chips_600m_manifest.csv` and `data/derived/chips_600m_progress.csv`

Use carefully — this permanently deletes files.
"""

import argparse
import csv
import os
import shutil
from pathlib import Path

import geopandas as gpd
import rasterio
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Prune cells with blank PRE chips")
    p.add_argument("--base", type=Path, required=False,
                   default=Path("/media/gisense/xihan/250812_CyberTraining_Team4"),
                   help="Project base path")
    p.add_argument("--chips", type=Path, default=None, help="Chips root (defaults under base)")
    p.add_argument("--panels", type=Path, default=None, help="Panels folder (defaults under derived)")
    p.add_argument("--grid", type=Path, default=None, help="Grid GPKG path (defaults under data/interim)")
    p.add_argument("--threshold", type=float, default=1e-6,
                   help="Maximum pixel value threshold (values <= threshold considered blank). For uint8 images use 1, for float use small epsilon")
    p.add_argument("--dry-run", action="store_true", help="Don't delete, just print what would be removed")
    return p.parse_args()


def is_blank_pre(pre_tif: Path, threshold: float) -> bool:
    try:
        with rasterio.open(pre_tif) as ds:
            # read a reduced window to avoid huge memory; read all bands but downsample if large
            # We'll read the full dataset but with decimation if very large
            w = ds.width
            h = ds.height
            factor = max(1, int(max(w, h) / 1024))
            if factor > 1:
                arr = ds.read(out_shape=(ds.count, int(h / factor), int(w / factor)))
            else:
                arr = ds.read()
            # compute max across bands
            maxv = float(np.nanmax(arr))
            return maxv <= threshold
    except Exception:
        # If the file can't be opened treat it as blank
        return True


def remove_cell_folder(cell_dir: Path, dry_run: bool):
    if dry_run:
        print(f"[DRY] Would remove folder: {cell_dir}")
        return
    if cell_dir.exists():
        shutil.rmtree(cell_dir)
        print(f"Removed folder: {cell_dir}")


def remove_panel(panel_dir: Path, cell_name: str, dry_run: bool):
    png = panel_dir / f"{cell_name}.png"
    if png.exists():
        if dry_run:
            print(f"[DRY] Would remove panel: {png}")
        else:
            png.unlink()
            print(f"Removed panel: {png}")


def prune_grid(gpkg: Path, cell_ids_to_remove, dry_run: bool):
    if not gpkg.exists():
        print(f"Grid gpkg not found: {gpkg}; skipping grid prune")
        return
    import fiona
    layers = list(fiona.listlayers(str(gpkg)))
    layer = None
    if layers:
        layer = "grid_600m_clip" if "grid_600m_clip" in layers else layers[0]
    else:
        print(f"No layers found in {gpkg}; skipping grid prune")
        return
    gdf = gpd.read_file(gpkg, layer=layer)
    orig_len = len(gdf)
    gdf = gdf[~gdf["cell_id"].isin(cell_ids_to_remove)].copy()
    new_len = len(gdf)
    if dry_run:
        print(f"[DRY] Would write new grid gpkg removing {orig_len - new_len} rows (keep {new_len})")
        return
    # write to temp gpkg then replace
    tmp = gpkg.with_suffix(".tmp.gpkg")
    if tmp.exists():
        tmp.unlink()
    gdf.to_file(tmp, layer=layer, driver="GPKG")
    # remove original and rename
    gpkg.unlink()
    tmp.rename(gpkg)
    print(f"Updated grid gpkg: removed {orig_len - new_len} cells")


def clean_csv(csv_path: Path, cell_ids_to_remove, key_name="cell_id", dry_run=False):
    if not csv_path.exists():
        return
    rows = []
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                if str(r.get(key_name, "")) in {str(x) for x in cell_ids_to_remove}:
                    continue
            except Exception:
                pass
            rows.append(r)
    if dry_run:
        print(f"[DRY] Would rewrite {csv_path} removing rows for cells: {cell_ids_to_remove}")
        return
    # write back
    if rows:
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
    else:
        # no rows left — remove file
        csv_path.unlink()
    print(f"Cleaned CSV: {csv_path}")


def main():
    args = parse_args()
    BASE = args.base
    CHIPS = args.chips or (BASE / "data" / "chips_600m")
    PANELS = args.panels or (BASE / "data" / "derived" / "pre_post_panels")
    GPKG = args.grid or (BASE / "data" / "interim" / "la_fire_grid_600m_clip_utm11.gpkg")
    DERIVED = BASE / "data" / "derived"
    manifest = DERIVED / "chips_600m_manifest.csv"
    progress = DERIVED / "chips_600m_progress.csv"

    if not CHIPS.exists():
        print(f"Chips folder not found: {CHIPS}")
        return

    # Find candidate cells
    cells = sorted([p for p in CHIPS.iterdir() if p.is_dir() and p.name.startswith("cell_")])
    to_remove = []
    for cell in cells:
        pre = cell / "pre" / f"{cell.name}_pre.tif"
        if not pre.exists():
            # treat missing pre as blank
            print(f"Missing pre chip -> will remove: {cell}")
            to_remove.append(cell)
            continue
        blank = is_blank_pre(pre, threshold=args.threshold)
        if blank:
            print(f"Blank pre chip detected -> will remove: {cell}")
            to_remove.append(cell)

    if not to_remove:
        print("No blank pre chips found — nothing to do.")
        return

    # collect cell ids
    cell_ids = [int(c.name.split("_")[1]) for c in to_remove]

    # Remove folders and panels
    for c in to_remove:
        remove_cell_folder(c, args.dry_run)
        remove_panel(PANELS, c.name, args.dry_run)

    # Update geopackage
    prune_grid(GPKG, cell_ids, args.dry_run)

    # Clean manifest and progress
    clean_csv(manifest, cell_ids, key_name="cell_id", dry_run=args.dry_run)
    clean_csv(progress, cell_ids, key_name="cell_id", dry_run=args.dry_run)

    print("Prune complete. Removed cells:", [c.name for c in to_remove])


if __name__ == "__main__":
    main()
