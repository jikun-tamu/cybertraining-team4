#!/usr/bin/env python3
import argparse
import csv
import json
import os
from datetime import datetime
from pathlib import Path

import fiona
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.io import MemoryFile
from rasterio.mask import mask as rio_mask
from rasterio.merge import merge as rio_merge
from shapely.geometry import mapping, box as sbox
from tqdm import tqdm
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds


def parse_args():
    p = argparse.ArgumentParser(description="Mosaic + clip rasters to 600m grid with pre/post classification and progress logging")
    p.add_argument("--base", type=Path, default=Path("/media/gisense/xihan/250812_CyberTraining_Team4"), help="Project base path")
    p.add_argument("--grid", type=Path, default=None, help="Path to grid GPKG (defaults to data/interim/la_fire_grid_600m_clip_utm11.gpkg under base)")
    p.add_argument("--raw", type=Path, default=None, help="Raw data root (defaults to base/data/raw)")
    p.add_argument("--out", type=Path, default=None, help="Chips output root (defaults to base/data/chips_600m)")
    p.add_argument("--derived", type=Path, default=None, help="Derived output root (defaults to base/data/derived)")
    p.add_argument("--limit-cells", type=int, default=None, help="Limit number of cells (for testing). Use -1 or None for all.")
    p.add_argument("--cell-ids", type=str, default=None, help="Comma-separated cell IDs to process (overrides --limit-cells if provided)")
    p.add_argument("--skip-existing", action="store_true", help="Skip writing outputs that already exist")
    p.add_argument("--threshold", type=str, default="2025-01-02", help="Date threshold YYYY-MM-DD for pre/post classification")
    return p.parse_args()


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_grid(grid_path: Path):
    if not grid_path.exists():
        raise FileNotFoundError(f"Grid file not found: {grid_path}")
    layers = list(fiona.listlayers(str(grid_path)))
    layer = "grid_600m_clip" if "grid_600m_clip" in layers else (layers[0] if layers else None)
    grid = gpd.read_file(grid_path, layer=layer) if layer else gpd.read_file(grid_path)
    if grid.crs is None:
        grid.set_crs(32611, inplace=True)
    elif grid.crs.to_epsg() != 32611:
        grid = grid.to_crs(32611)
    if "cell_id" not in grid.columns:
        grid = grid.reset_index().rename(columns={"index": "cell_id"})
    return grid


def build_raster_index(raw_root: Path):
    # Collect all GeoTIFFs under raw; restrict to plausible ARD visual and general *.tif
    raster_paths = [p for p in raw_root.rglob("*.tif")]
    r_infos = []
    for rp in raster_paths:
        try:
            with rasterio.open(rp) as ds:
                if ds.crs is None:
                    continue
                b = ds.bounds
                geom = sbox(b.left, b.bottom, b.right, b.top)
                gdf = gpd.GeoDataFrame(geometry=[geom], crs=ds.crs)
                gdf_utm = gdf.to_crs(32611)
                r_infos.append({
                    "path": str(rp),
                    "geometry": gdf_utm.geometry.values[0],
                    "crs": ds.crs,
                })
        except Exception:
            continue
    rf_gdf = gpd.GeoDataFrame(r_infos, geometry="geometry", crs=32611)
    return rf_gdf


def classify_pre_post(path: Path, thresh_dt: datetime):
    s = str(path).lower()
    if "post-event" in s:
        return "post", None
    if "pre-event" in s:
        return "pre", None
    # Try to parse yyyymmdd or yyyy-mm-dd
    import re
    m = re.search(r"(20\d{2})[-_/]?(\d{2})[-_/]?(\d{2})", s)
    if m:
        try:
            dt = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            return ("post" if dt >= thresh_dt else "pre"), dt
        except Exception:
            pass
    return None, None


def clip_mosaic(cell_geom_32611, paths, out_path: Path, mosaic=True) -> bool:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if mosaic:
            srcs = [rasterio.open(str(p)) for p in paths]
            if not srcs:
                return False
            # enforce same CRS
            crs_set = {s.crs.to_string() for s in srcs if s.crs}
            if len(crs_set) > 1:
                # pick the majority CRS and filter others out
                from collections import Counter
                majority_crs = Counter([s.crs.to_string() for s in srcs if s.crs]).most_common(1)[0][0]
                keep = []
                for s in srcs:
                    if s.crs and s.crs.to_string() == majority_crs:
                        keep.append(s)
                    else:
                        s.close()
                srcs = keep
                if not srcs:
                    return False
            tgt = srcs[0]
            cell_geom = gpd.GeoSeries([cell_geom_32611], crs=32611).to_crs(tgt.crs).geometry.values[0]
            mosaic_img, mosaic_transform = rio_merge(srcs, bounds=cell_geom.bounds)
            meta = tgt.meta.copy()
            for s in srcs:
                s.close()
            meta.update({
                "height": mosaic_img.shape[1],
                "width": mosaic_img.shape[2],
                "transform": mosaic_transform,
                "crs": tgt.crs,
                "count": mosaic_img.shape[0],
            })
            with MemoryFile() as memfile:
                with memfile.open(**meta) as tmp:
                    tmp.write(mosaic_img)
                    out_img, out_transform = rio_mask(tmp, [mapping(cell_geom)], crop=True)
                    out_meta = meta.copy()

                    # enforce exact cell bounds (in target CRS) so output covers 600m x 600m grid cell
                    cell_bounds = cell_geom.bounds
                    # determine a sensible source pixel size (absolute)
                    src_px = abs(out_transform.a) if out_transform.a != 0 else None
                    if src_px is None or src_px <= 0:
                        # fallback to transform from mosaic
                        src_px = abs(mosaic_transform.a) if mosaic_transform.a != 0 else 1.0

                    target_w = max(1, int(round((cell_bounds[2] - cell_bounds[0]) / src_px)))
                    target_h = max(1, int(round((cell_bounds[3] - cell_bounds[1]) / src_px)))
                    target_transform = from_bounds(cell_bounds[0], cell_bounds[1], cell_bounds[2], cell_bounds[3], target_w, target_h)

                    # reproject/resample masked output to exact cell extent
                    dst_array = np.zeros((out_img.shape[0], target_h, target_w), dtype=out_img.dtype)
                    for i in range(out_img.shape[0]):
                        reproject(
                            source=out_img[i],
                            destination=dst_array[i],
                            src_transform=out_transform,
                            src_crs=meta.get('crs'),
                            dst_transform=target_transform,
                            dst_crs=meta.get('crs'),
                            resampling=Resampling.bilinear,
                        )

                    out_meta.update({
                        "height": target_h,
                        "width": target_w,
                        "transform": target_transform,
                        "compress": "lzw",
                        "tiled": True,
                        "blockxsize": min(512, target_w),
                        "blockysize": min(512, target_h),
                    })
                with rasterio.open(out_path, "w", **out_meta) as dst:
                    dst.write(dst_array)
        else:
            with rasterio.open(str(paths[0])) as ds:
                geom = gpd.GeoSeries([cell_geom_32611], crs=32611).to_crs(ds.crs).geometry.values[0]
                out_img, out_transform = rio_mask(ds, [mapping(geom)], crop=True)
                meta = ds.meta.copy()

                cell_bounds = geom.bounds
                src_px = abs(out_transform.a) if out_transform.a != 0 else abs(ds.transform.a)
                target_w = max(1, int(round((cell_bounds[2] - cell_bounds[0]) / src_px)))
                target_h = max(1, int(round((cell_bounds[3] - cell_bounds[1]) / src_px)))
                target_transform = from_bounds(cell_bounds[0], cell_bounds[1], cell_bounds[2], cell_bounds[3], target_w, target_h)

                dst_array = np.zeros((out_img.shape[0], target_h, target_w), dtype=out_img.dtype)
                for i in range(out_img.shape[0]):
                    reproject(
                        source=out_img[i],
                        destination=dst_array[i],
                        src_transform=out_transform,
                        src_crs=ds.crs,
                        dst_transform=target_transform,
                        dst_crs=ds.crs,
                        resampling=Resampling.bilinear,
                    )

                meta.update({
                    "height": target_h,
                    "width": target_w,
                    "transform": target_transform,
                    "compress": "lzw",
                    "tiled": True,
                    "blockxsize": min(512, target_w),
                    "blockysize": min(512, target_h),
                })
            with rasterio.open(out_path, "w", **meta) as dst:
                dst.write(dst_array)
        return True
    except Exception as e:
        print("Clip/mosaic failed for", out_path, "→", e)
        return False


# NOTE: Pixel-based 600x600 resampling helpers removed — outputs are resampled
# to the grid cell extent (600 m × 600 m) inside `clip_mosaic` so files
# represent the same geographic area as the grid.


def main():
    args = parse_args()
    BASE = args.base
    RAW = args.raw or (BASE / "data" / "raw")
    OUT_CHIPS = ensure_dir(args.out or (BASE / "data" / "chips_600m"))
    DERIVED = ensure_dir((args.derived or (BASE / "data" / "derived")))
    GRID = args.grid or (BASE / "data" / "interim" / "la_fire_grid_600m_clip_utm11.gpkg")
    THRESH = datetime.fromisoformat(args.threshold)
    progress_csv = DERIVED / "chips_600m_progress.csv"
    manifest_csv = DERIVED / "chips_600m_manifest.csv"

    grid = load_grid(GRID)
    rf = build_raster_index(RAW)

    # classify
    rf["when"] = None
    rf["date"] = None
    for i, row in rf.iterrows():
        w, dt = classify_pre_post(row["path"], THRESH)
        rf.at[i, "when"] = w
        rf.at[i, "date"] = dt.isoformat() if isinstance(dt, datetime) else None
    rf_known = rf[rf["when"].isin(["pre", "post"])].copy()

    # spatial join to cells
    pre_join = gpd.sjoin(rf_known[rf_known["when"] == "pre"], grid[["cell_id", "geometry"]], predicate="intersects")
    post_join = gpd.sjoin(rf_known[rf_known["when"] == "post"], grid[["cell_id", "geometry"]], predicate="intersects")

    # select cells
    cells = grid.copy()
    if args.cell_ids:
        ids = [int(x) for x in args.cell_ids.split(",") if x.strip()]
        cells = cells[cells["cell_id"].isin(ids)].copy()
    elif args.limit_cells is not None and args.limit_cells >= 0:
        cells = cells.head(int(args.limit_cells)).copy()

    # prepare progress CSV
    write_header = not progress_csv.exists()
    with progress_csv.open("a", newline="") as pf:
        pw = csv.writer(pf)
        if write_header:
            pw.writerow([
                "ts_start", "ts_end", "duration_s", "cell_id", "pre_inputs", "post_inputs",
                "pre_output", "post_outputs", "pre_count", "post_count", "status", "error"
            ])

        manifest_rows = []
        for _, cell in tqdm(cells.iterrows(), total=len(cells), desc="Processing cells"):
            ts0 = datetime.utcnow()
            cid = int(cell["cell_id"])
            cell_dir = OUT_CHIPS / f"cell_{cid:05d}"
            pre_dir = cell_dir / "pre"
            post_dir = cell_dir / "post"
            pre_dir.mkdir(parents=True, exist_ok=True)
            post_dir.mkdir(parents=True, exist_ok=True)

            cell_pre = pre_join[pre_join["cell_id"] == cid]
            pre_paths = [Path(p) for p in cell_pre["path"].tolist()]
            pre_out = pre_dir / f"cell_{cid:05d}_pre.tif"

            cell_post = post_join[post_join["cell_id"] == cid]
            post_paths = [Path(p) for p in cell_post["path"].tolist()]

            status = "ok"
            err = ""
            pre_written = None
            post_written = []

            try:
                # Pre
                if pre_paths and (not args.skip_existing or not pre_out.exists()):
                    ok = clip_mosaic(cell.geometry, pre_paths, pre_out, mosaic=True)
                    if ok:
                        # ensure size
                        try:
                            ensure_600x600(pre_out, size=(600, 600))
                        except Exception:
                            pass
                        pre_written = pre_out
                        manifest_rows.append({"cell_id": cid, "type": "pre", "path": str(pre_out), "count": len(pre_paths)})
                # Post
                for j, rp in enumerate(post_paths, start=1):
                    # rp is a string path in rf; parse date component like 2025-01-16 from path
                    date_part = None
                    try:
                        import re
                        m = re.search(r"(20\d{2})[-_/]?(\d{2})[-_/]?(\d{2})", str(rp))
                        if m:
                            date_part = f"{m.group(1)}{m.group(2)}{m.group(3)}"
                    except Exception:
                        date_part = None

                    if date_part:
                        outp = post_dir / f"cell_{cid:05d}_post_{date_part}.tif"
                    else:
                        outp = post_dir / f"cell_{cid:05d}_post_{j:02d}.tif"
                    if args.skip_existing and outp.exists():
                        post_written.append(outp)
                        manifest_rows.append({"cell_id": cid, "type": "post", "path": str(outp), "source": str(rp)})
                        continue
                    ok = clip_mosaic(cell.geometry, [rp], outp, mosaic=False)
                    if ok:
                        post_written.append(outp)
                        manifest_rows.append({"cell_id": cid, "type": "post", "path": str(outp), "source": str(rp), "date": date_part})
            except Exception as e:
                status = "error"
                err = str(e)

            ts1 = datetime.utcnow()
            dur = (ts1 - ts0).total_seconds()
            # write progress row
            pw.writerow([
                ts0.isoformat() + "Z",
                ts1.isoformat() + "Z",
                f"{dur:.2f}",
                cid,
                len(pre_paths),
                len(post_paths),
                str(pre_written) if pre_written else "",
                ";".join(str(p) for p in post_written) if post_written else "",
                len(pre_paths),
                len(post_paths),
                status,
                err,
            ])
            pf.flush()

        # Write manifest at end
        if manifest_rows:
            with manifest_csv.open("w", newline="") as f:
                fieldnames = sorted({k for r in manifest_rows for k in r.keys()})
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                for r in manifest_rows:
                    w.writerow(r)

    print("Done. Progress:", progress_csv)
    print("Manifest:", manifest_csv)


if __name__ == "__main__":
    main()
