#!/usr/bin/env python3
"""Combine per-cell canonical LA Fire multidate results into a single georeferenced dataset.

For each processed cell:
  1. Reads aggregated_predictions.jsonl (M1 damage class + probs)
  2. Reads shared_base/shared_instance_samples.csv (pixel polygon WKT + Stage-1 confidence)
  3. Reads stage2a_predictions.csv (building type + population)
  4. Reads per-date quality_metrics.json (date quality flags)
  5. Converts pixel-space polygon → UTM → WGS84 using the pre TIF geotransform
  6. Joins all fields into one row per building instance

Outputs
-------
  /media/data/building_instance_tamu/la_fire_2025/stage2_damage/multidate_full_run/building_damage_all_cells.csv
  /media/data/building_instance_tamu/la_fire_2025/stage2_damage/multidate_full_run/building_damage_all_cells.geojson
  /media/data/building_instance_tamu/la_fire_2025/stage2_damage/multidate_full_run/building_damage_all_cells.gpkg
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from la_fire_paths import canonical_manifest_path, canonical_run_root, rewrite_la_fire_path

PKG_ROOT = Path(__file__).resolve().parents[1]


def fix_path(p: str) -> str:
    return str(rewrite_la_fire_path(p))


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _parse_ring_text(text: str) -> list[tuple[float, float]] | None:
    import re
    pts = []
    for chunk in text.split(","):
        nums = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", chunk)]
        if len(nums) >= 2:
            pts.append((nums[0], nums[1]))
    if len(pts) >= 2 and pts[0] == pts[-1]:
        pts = pts[:-1]
    return pts or None


def _extract_wkt_rings(wkt: str, geom_type: str) -> list[list[tuple[float, float]] | None]:
    body = wkt.strip()[len(geom_type):].strip()
    if not (body.startswith("(") and body.endswith(")")):
        return []
    groups = []
    start = None
    depth = 0
    for idx, ch in enumerate(body):
        if ch == "(":
            if depth == 0:
                start = idx
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0 and start is not None:
                groups.append(body[start:idx + 1])
                start = None
    if geom_type == "POLYGON":
        return [_parse_ring_text(groups[0][1:-1])] if groups else []
    rings = []
    for poly_group in groups:
        inner = poly_group[1:-1].strip()
        sub_groups = []
        start = None
        depth = 0
        for idx, ch in enumerate(inner):
            if ch == "(":
                if depth == 0:
                    start = idx
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0 and start is not None:
                    sub_groups.append(inner[start:idx + 1])
                    start = None
        if sub_groups:
            rings.append(_parse_ring_text(sub_groups[0][1:-1]))
    return rings


def parse_wkt_geometry(wkt: str) -> list[list[tuple[float, float]]] | None:
    if not wkt:
        return None
    try:
        from shapely import wkt as shapely_wkt  # type: ignore

        geom = shapely_wkt.loads(wkt)
        if geom.geom_type == "Polygon":
            coords = list(geom.exterior.coords[:-1])
            return [coords] if len(coords) >= 3 else None
        if geom.geom_type == "MultiPolygon":
            polys = []
            for poly in geom.geoms:
                coords = list(poly.exterior.coords[:-1])
                if len(coords) >= 3:
                    polys.append(coords)
            return polys or None
    except Exception:
        pass
    upper = wkt.strip().upper()
    if upper.startswith("POLYGON"):
        rings = _extract_wkt_rings(wkt, "POLYGON")
    elif upper.startswith("MULTIPOLYGON"):
        rings = _extract_wkt_rings(wkt, "MULTIPOLYGON")
    else:
        return None
    valid = [ring for ring in rings if ring and len(ring) >= 3]
    return valid or None


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


def pts_to_geojson_polygon(pts_wgs84: list[tuple[float, float]]) -> list[list[float]]:
    coords = [[lon, lat] for lon, lat in pts_wgs84]
    if coords[0] != coords[-1]:
        coords.append(coords[0])
    return coords


def pts_to_wkt_utm(pts_utm: list[tuple[float, float]]) -> str:
    inner = ", ".join(f"{x:.3f} {y:.3f}" for x, y in pts_utm)
    # Close ring
    if pts_utm[0] != pts_utm[-1]:
        x0, y0 = pts_utm[0]
        inner += f", {x0:.3f} {y0:.3f}"
    return f"POLYGON (({inner}))"


def geometry_to_geojson(geoms_wgs84: list[list[tuple[float, float]]]) -> dict:
    polygons = [[pts_to_geojson_polygon(poly)] for poly in geoms_wgs84]
    if len(polygons) == 1:
        return {"type": "Polygon", "coordinates": polygons[0]}
    return {"type": "MultiPolygon", "coordinates": polygons}


def geometry_to_wkt_utm(geoms_utm: list[list[tuple[float, float]]]) -> str:
    if len(geoms_utm) == 1:
        return pts_to_wkt_utm(geoms_utm[0])
    parts = []
    for pts in geoms_utm:
        coords = list(pts)
        if coords[0] != coords[-1]:
            coords.append(coords[0])
        parts.append("((" + ", ".join(f"{x:.3f} {y:.3f}" for x, y in coords) + "))")
    return "MULTIPOLYGON (" + ", ".join(parts) + ")"


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
                   default=canonical_run_root())
    p.add_argument("--out_csv", type=Path,
                   default=canonical_run_root() / "building_damage_all_cells.csv")
    p.add_argument("--out_geojson", type=Path,
                   default=canonical_run_root() / "building_damage_all_cells.geojson")
    p.add_argument("--out_gpkg", type=Path,
                   default=canonical_run_root() / "building_damage_all_cells.gpkg")
    p.add_argument("--manifest", type=Path,
                   default=canonical_manifest_path())
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
            geoms_px = parse_wkt_geometry(wkt_px)

            polygon_wkt_utm = ""
            centroid_utm_x = centroid_utm_y = centroid_lon = centroid_lat = ""
            geom_wgs84 = None

            if geoms_px and transform and epsg:
                try:
                    geoms_utm = [pixel_to_utm(poly, transform) for poly in geoms_px]
                    polygon_wkt_utm = geometry_to_wkt_utm(geoms_utm)
                    all_utm_pts = [pt for poly in geoms_utm for pt in poly]
                    cx_utm = np.mean([p[0] for p in all_utm_pts])
                    cy_utm = np.mean([p[1] for p in all_utm_pts])
                    centroid_utm_x = round(cx_utm, 2)
                    centroid_utm_y = round(cy_utm, 2)
                    geoms_wgs84 = [utm_to_wgs84(poly, epsg) for poly in geoms_utm]
                    geom_wgs84 = geometry_to_geojson(geoms_wgs84)
                    clon, clat = utm_to_wgs84([(cx_utm, cy_utm)], epsg)[0]
                    centroid_lon = round(clon, 7)
                    centroid_lat = round(clat, 7)
                except Exception as e:
                    n_skipped_geo += 1
                    geom_wgs84 = None
            elif not geoms_px:
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
