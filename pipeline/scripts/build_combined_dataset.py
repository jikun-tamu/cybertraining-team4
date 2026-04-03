#!/usr/bin/env python3
"""Combine per-cell multidate results into a single georeferenced dataset.

For each processed cell:
  1. Reads aggregated_predictions.jsonl (M1 damage class + probs)
  2. Reads shared_base/shared_instance_samples.csv (pixel polygon WKT + Stage-1 confidence)
  3. Reads stage2a_predictions.csv (building type + population)
  4. Reads per-date quality_metrics.json (date quality flags)
  5. Converts pixel-space polygon → UTM → WGS84 using the pre TIF geotransform
  6. Joins all fields into one row per building instance

Outputs
-------
  outputs/multidate_full_run/building_damage_all_cells.csv   — flat CSV
  outputs/maps/building_damage.geojson                       — WGS84 GeoJSON
  outputs/maps/building_damage.gpkg                          — UTM GeoPackage
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

PKG_ROOT = Path(__file__).resolve().parents[1]
OLD_PREFIX = "/media/gisense/xihan/250812_CyberTraining_Team4/data/chips_600m"
NEW_PREFIX = "/media/data/building_instance_tamu/la_fire_2025/chips"


def fix_path(p: str) -> str:
    return p.replace(OLD_PREFIX, NEW_PREFIX)


# ── Geometry helpers ──────────────────────────────────────────────────────────

def parse_wkt_polygon(wkt: str) -> list[tuple[float, float]] | None:
    """Parse POLYGON ((x y, ...)) → list of (x, y) pairs."""
    import re
    if not wkt or not wkt.strip().upper().startswith("POLYGON"):
        return None
    nums = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", wkt)]
    if len(nums) < 6 or len(nums) % 2 != 0:
        return None
    return [(nums[i], nums[i + 1]) for i in range(0, len(nums), 2)]


def pixel_to_utm(pts_px: list[tuple[float, float]], transform) -> list[tuple[float, float]]:
    """Apply rasterio Affine transform: pixel (col, row) → UTM (x, y)."""
    return [(transform.c + pt[0] * transform.a + pt[1] * transform.b,
             transform.f + pt[0] * transform.d + pt[1] * transform.e)
            for pt in pts_px]


def utm_to_wgs84(pts_utm: list[tuple[float, float]], epsg: int = 32611):
    """Convert UTM points to WGS84 using pyproj."""
    from pyproj import Transformer
    t = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    return [t.transform(x, y) for x, y in pts_utm]


def pts_to_geojson_polygon(pts_wgs84: list[tuple[float, float]]) -> dict:
    coords = [[lon, lat] for lon, lat in pts_wgs84]
    if coords[0] != coords[-1]:
        coords.append(coords[0])
    return {"type": "Polygon", "coordinates": [coords]}


def pts_to_wkt_utm(pts_utm: list[tuple[float, float]]) -> str:
    inner = ", ".join(f"{x:.3f} {y:.3f}" for x, y in pts_utm)
    # Close ring
    if pts_utm[0] != pts_utm[-1]:
        x0, y0 = pts_utm[0]
        inner += f", {x0:.3f} {y0:.3f}"
    return f"POLYGON (({inner}))"


# ── Data loading helpers ──────────────────────────────────────────────────────

def load_agg(jsonl_path: Path) -> dict[str, dict]:
    records = {}
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            records[obj["bldg_uid"]] = obj
    return records


def load_stage2a(csv_path: Path) -> dict[str, dict]:
    records = {}
    if not csv_path.exists():
        return records
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            uid = row.get("bldg_uid", "")
            if uid:
                records[uid] = row
    return records


def load_shared_csv(csv_path: Path) -> dict[str, dict]:
    records = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            uid = row.get("bldg_uid", "")
            if uid:
                records[uid] = row
    return records


def get_pre_transform(cell_id: str, manifest_pre: dict[str, Path]):
    """Return (rasterio_transform, epsg) for a cell's pre TIF."""
    import rasterio
    pre_path = manifest_pre.get(cell_id)
    if pre_path is None or not pre_path.exists():
        return None, None
    with rasterio.open(pre_path) as ds:
        return ds.transform, int(ds.crs.to_epsg())


def load_manifest_pre(manifest_path: Path) -> dict[str, Path]:
    mapping = {}
    with open(manifest_path, newline="") as f:
        for row in csv.DictReader(f):
            if row["type"] == "pre":
                cid = f"cell_{int(row['cell_id']):05d}"
                mapping[cid] = Path(fix_path(row["path"]))
    return mapping


# ── Main ──────────────────────────────────────────────────────────────────────

DAMAGE_LABEL = {-1: "unknown", 0: "no_damage", 1: "minor", 2: "major", 3: "destroyed"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_root", type=Path,
                   default=PKG_ROOT / "outputs/multidate_full_run")
    p.add_argument("--out_csv", type=Path,
                   default=PKG_ROOT / "outputs/multidate_full_run/building_damage_all_cells.csv")
    p.add_argument("--out_geojson", type=Path,
                   default=PKG_ROOT / "outputs/maps/building_damage.geojson")
    p.add_argument("--out_gpkg", type=Path,
                   default=PKG_ROOT / "outputs/maps/building_damage.gpkg")
    p.add_argument("--manifest", type=Path,
                   default=PKG_ROOT.parent / "data/processed/chips_600m_manifest.csv")
    args = p.parse_args()

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_geojson.parent.mkdir(parents=True, exist_ok=True)

    manifest_pre = load_manifest_pre(args.manifest)
    print(f"Manifest: {len(manifest_pre)} pre-image paths")

    # Discover all processed cells
    cell_dirs = sorted([d for d in args.run_root.iterdir()
                        if d.is_dir() and d.name.startswith("cell_")])
    print(f"Cell directories found: {len(cell_dirs)}")

    csv_fields = [
        "cell_id", "bldg_uid",
        "m1_damage_class", "m1_damage_label",
        "m1_prob_no_damage", "m1_prob_minor", "m1_prob_major", "m1_prob_destroyed",
        "m1b_damage_class", "m1b_damage_label",
        "m2_majority_class", "m2b_damage_class", "m2b_damage_label",
        "m3_quality_filtered_class",
        "n_dates_total", "n_dates_used", "n_dates_rejected", "n_dates_valid_coverage",
        "label_entropy", "is_unstable",
        "sam3_confidence",
        "stage2a_type", "stage2a_population",
        "polygon_wkt_utm", "centroid_utm_x", "centroid_utm_y",
        "centroid_lon", "centroid_lat",
        "crop_x0", "crop_y0",
    ]

    geojson_features = []
    all_rows = []
    n_skipped_geo = 0
    n_zero_instance_cells = 0

    for cell_dir in cell_dirs:
        cell_id = cell_dir.name

        # Skip zero-instance cells
        if (cell_dir / "shared_base/zero_instances.marker").exists():
            n_zero_instance_cells += 1
            continue

        agg_jsonl = cell_dir / "aggregated_predictions.jsonl"
        shared_csv = cell_dir / "shared_base/shared_instance_samples.csv"
        stage2a_csv = cell_dir / "stage2a_predictions.csv"

        if not agg_jsonl.exists() or not shared_csv.exists():
            continue

        agg = load_agg(agg_jsonl)
        shared = load_shared_csv(shared_csv)
        s2a = load_stage2a(stage2a_csv)

        # Get geotransform for this cell
        transform, epsg = get_pre_transform(cell_id, manifest_pre)

        for uid, rec in agg.items():
            shared_row = shared.get(uid, {})
            s2a_row = s2a.get(uid, {})

            # Parse pixel polygon
            wkt_px = shared_row.get("polygon_wkt_xy_pre", "")
            pts_px = parse_wkt_polygon(wkt_px)

            polygon_wkt_utm = ""
            centroid_utm_x = centroid_utm_y = centroid_lon = centroid_lat = ""
            geom_wgs84 = None

            if pts_px and transform and epsg:
                try:
                    pts_utm = pixel_to_utm(pts_px, transform)
                    polygon_wkt_utm = pts_to_wkt_utm(pts_utm)
                    cx_utm = np.mean([p[0] for p in pts_utm])
                    cy_utm = np.mean([p[1] for p in pts_utm])
                    centroid_utm_x = round(cx_utm, 2)
                    centroid_utm_y = round(cy_utm, 2)
                    pts_wgs84 = utm_to_wgs84(pts_utm, epsg)
                    geom_wgs84 = pts_to_geojson_polygon(pts_wgs84)
                    clon, clat = utm_to_wgs84([(cx_utm, cy_utm)], epsg)[0]
                    centroid_lon = round(clon, 7)
                    centroid_lat = round(clat, 7)
                except Exception as e:
                    n_skipped_geo += 1
                    geom_wgs84 = None
            elif not pts_px:
                n_skipped_geo += 1

            # M1 probs
            m1_probs = rec.get("m1_prob_avg_probs", [0.25, 0.25, 0.25, 0.25])
            m1_class = rec.get("m1_prob_avg_class", -1)
            m1b_class = rec.get("m1b_coverage_class", m1_class)  # fall back to m1 if field absent

            row = {
                "cell_id": cell_id,
                "bldg_uid": uid,
                "m1_damage_class": m1_class,
                "m1_damage_label": DAMAGE_LABEL.get(int(m1_class) if m1_class != "" else -1, "unknown"),
                "m1_prob_no_damage": round(m1_probs[0], 4) if len(m1_probs) > 0 else "",
                "m1_prob_minor": round(m1_probs[1], 4) if len(m1_probs) > 1 else "",
                "m1_prob_major": round(m1_probs[2], 4) if len(m1_probs) > 2 else "",
                "m1_prob_destroyed": round(m1_probs[3], 4) if len(m1_probs) > 3 else "",
                "m1b_damage_class": m1b_class,
                "m1b_damage_label": DAMAGE_LABEL.get(int(m1b_class) if m1b_class != "" else -1, "unknown"),
                "m2_majority_class": rec.get("m2_majority_class", ""),
                "m2b_damage_class": rec.get("m2b_coverage_vote_class", ""),
                "m2b_damage_label": DAMAGE_LABEL.get(
                    int(rec.get("m2b_coverage_vote_class", -1)), "unknown"),
                "m3_quality_filtered_class": rec.get("m3_quality_filtered_class", ""),
                "n_dates_total": rec.get("n_dates_total", ""),
                "n_dates_used": rec.get("n_dates_used_m1m2", ""),
                "n_dates_rejected": (rec.get("n_dates_total", 0) -
                                     rec.get("n_dates_used_m1m2", 0)),
                "n_dates_valid_coverage": rec.get("n_dates_valid_coverage", ""),
                "label_entropy": rec.get("label_entropy", ""),
                "is_unstable": rec.get("is_unstable", ""),
                "sam3_confidence": shared_row.get("sam3_confidence", ""),
                "stage2a_type": s2a_row.get("pred_type", s2a_row.get("type", "")),
                "stage2a_population": s2a_row.get("pred_population", ""),
                "polygon_wkt_utm": polygon_wkt_utm,
                "centroid_utm_x": centroid_utm_x,
                "centroid_utm_y": centroid_utm_y,
                "centroid_lon": centroid_lon,
                "centroid_lat": centroid_lat,
                "crop_x0": shared_row.get("crop_x0", ""),
                "crop_y0": shared_row.get("crop_y0", ""),
            }
            all_rows.append(row)

            if geom_wgs84:
                geojson_features.append({
                    "type": "Feature",
                    "geometry": geom_wgs84,
                    "properties": {k: v for k, v in row.items()
                                   if k not in ("polygon_wkt_utm",)},
                })

    # ── Write CSV ─────────────────────────────────────────────────────────────
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"[done] CSV: {args.out_csv}  rows={len(all_rows)}")
    print(f"  zero-instance cells: {n_zero_instance_cells}")
    print(f"  geo-skipped instances: {n_skipped_geo}")

    # ── Write GeoJSON ─────────────────────────────────────────────────────────
    geojson = {"type": "FeatureCollection", "features": geojson_features}
    args.out_geojson.write_text(json.dumps(geojson))
    print(f"[done] GeoJSON: {args.out_geojson}  features={len(geojson_features)}")

    # ── Write GeoPackage (via geopandas) ──────────────────────────────────────
    try:
        import geopandas as gpd
        from shapely.geometry import shape
        geoms = [shape(f["geometry"]) for f in geojson_features]
        props = [f["properties"] for f in geojson_features]
        gdf = gpd.GeoDataFrame(props, geometry=geoms, crs="EPSG:4326")
        gdf_utm = gdf.to_crs("EPSG:32611")
        gdf_utm.to_file(args.out_gpkg, driver="GPKG", layer="building_damage")
        print(f"[done] GeoPackage: {args.out_gpkg}  rows={len(gdf_utm)}")
    except Exception as e:
        print(f"[warn] GeoPackage write failed: {e}")


if __name__ == "__main__":
    main()
