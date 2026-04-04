#!/usr/bin/env python3
"""Generate ONE QC overlay image per tile: best post-disaster image + M2b damage polygons.

Reads pixel-space polygons from shared_instance_samples.csv and M2b damage classes
from aggregated_predictions.jsonl. Uses the best-quality post-disaster image as background.

Usage:
    python scripts/generate_qc_overlays.py
    python scripts/generate_qc_overlays.py --cells cell_00046 cell_00064
    python scripts/generate_qc_overlays.py --out_dir /path/to/output
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as patheffects
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection

from la_fire_paths import canonical_chips_root, canonical_la_fire_root, canonical_run_root

# ── Paths ────────────────────────────────────────────────────────────────────
LA_FIRE_ROOT = canonical_la_fire_root()
CHIPS_ROOT   = canonical_chips_root()
RUN_ROOT     = canonical_run_root()

DAMAGE_COLORS = {
    0:  (0.18, 0.80, 0.44),   # green  — no damage
    1:  (0.95, 0.61, 0.07),   # amber  — minor
    2:  (0.90, 0.49, 0.13),   # orange — major
    3:  (0.91, 0.30, 0.24),   # red    — destroyed
    -1: (0.58, 0.65, 0.65),   # grey   — not identifiable
}
DAMAGE_LABELS = {
    0: "No damage", 1: "Minor", 2: "Major",
    3: "Destroyed", -1: "Not identifiable",
}

WKT_FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def parse_args():
    p = argparse.ArgumentParser(description="Generate QC overlay images per tile.")
    p.add_argument("--run_root", type=Path, default=RUN_ROOT)
    p.add_argument("--chips_root", type=Path, default=CHIPS_ROOT)
    p.add_argument("--out_dir", type=Path,
                   default=LA_FIRE_ROOT / "qc_overlays_m2b")
    p.add_argument("--cells", nargs="*", default=None,
                   help="Only these cells (default: all)")
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args()


# ── Helpers ──────────────────────────────────────────────────────────────────

def _parse_ring_text(text: str) -> np.ndarray | None:
    pts = []
    for chunk in text.split(","):
        nums = [float(x) for x in WKT_FLOAT_RE.findall(chunk)]
        if len(nums) >= 2:
            pts.append((nums[0], nums[1]))
    if len(pts) >= 2 and pts[0] == pts[-1]:
        pts = pts[:-1]
    return np.array(pts, dtype=np.float32) if pts else None


def _extract_wkt_rings(wkt: str, geom_type: str) -> list[np.ndarray | None]:
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


def parse_wkt_geometry_xy(wkt: str) -> list[np.ndarray] | None:
    if not wkt:
        return None
    try:
        from shapely import wkt as shapely_wkt  # type: ignore

        geom = shapely_wkt.loads(wkt)
        if geom.geom_type == "Polygon":
            arr = np.array(geom.exterior.coords[:-1], dtype=np.float32)
            return [arr] if len(arr) >= 3 else None
        if geom.geom_type == "MultiPolygon":
            polys = []
            for poly in geom.geoms:
                arr = np.array(poly.exterior.coords[:-1], dtype=np.float32)
                if len(arr) >= 3:
                    polys.append(arr)
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
    valid = [ring for ring in rings if ring is not None and len(ring) >= 3]
    return valid or None


def geometry_centroid(polys: list[np.ndarray]) -> tuple[float, float]:
    pts = np.vstack(polys)
    return float(pts[:, 0].mean()), float(pts[:, 1].mean())


def best_post_date(cell_id: str, run_root: Path, chips_root: Path):
    """Return (date_str, tif_path) for best-quality post image."""
    dates_dir = run_root / cell_id / "dates"
    if not dates_dir.exists():
        return None

    candidates = []
    for d in sorted(dates_dir.iterdir()):
        if not d.is_dir():
            continue
        qm_path = d / "quality_metrics.json"
        if not qm_path.exists():
            continue
        qm = json.loads(qm_path.read_text())
        if not qm.get("tile_quality_ok", False):
            continue
        tif_path = chips_root / cell_id / "post" / f"{cell_id}_post_{d.name}.tif"
        if not tif_path.exists():
            continue
        # Score by crop coverage fraction then brightness
        n_ok = n_total = 0
        shared_csv = d / "shared_for_date.csv"
        if shared_csv.exists():
            with open(shared_csv) as f:
                rows = list(csv.DictReader(f))
            n_total = len(rows)
            n_ok = sum(1 for r in rows if r.get("quality_ok", "false").lower() == "true")
        coverage = n_ok / n_total if n_total > 0 else 0.0
        brightness = float(qm.get("mean_brightness", 0))
        candidates.append((coverage, brightness, d.name, tif_path))

    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return candidates[0][2], candidates[0][3]


def read_tif_rgb(tif_path: Path):
    import rasterio
    with rasterio.open(tif_path) as ds:
        return ds.read([1, 2, 3]).transpose(1, 2, 0)


def enhance_rgb(arr: np.ndarray) -> np.ndarray:
    out = np.zeros_like(arr, dtype=np.float32)
    for c in range(3):
        ch = arr[:, :, c].astype(np.float32)
        valid = ch[ch > 0]
        if len(valid) == 0:
            continue
        lo, hi = np.percentile(valid, [2, 98])
        if hi > lo:
            out[:, :, c] = np.clip((ch - lo) / (hi - lo), 0, 1)
        else:
            out[:, :, c] = ch / 255.0
    return out


def load_buildings(cell_id: str, run_root: Path):
    """Load buildings with pixel polygons and M2b damage classes.

    Returns list of {uid, polygon_xy, m2b_class, m2b_confidence, n_valid_dates}.
    """
    shared_csv = run_root / cell_id / "shared_base/shared_instance_samples.csv"
    agg_jsonl = run_root / cell_id / "aggregated_predictions.jsonl"

    if not shared_csv.exists() or not agg_jsonl.exists():
        return []

    # Load M2b predictions
    m2b_data = {}
    with open(agg_jsonl) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            uid = rec.get("bldg_uid", "")
            m2b_class = rec.get("m2b_coverage_vote_class", -1)
            # Confidence = max of M1b probs (coverage-aware)
            probs = rec.get("m1b_coverage_probs", [0.25]*4)
            m2b_data[uid] = {
                "class": m2b_class,
                "confidence": max(probs) if probs else 0.25,
                "n_valid": rec.get("m2b_n_valid_dates", 0),
            }

    # Load pixel polygons
    buildings = []
    with open(shared_csv) as f:
        for row in csv.DictReader(f):
            uid = row.get("bldg_uid", "")
            wkt = row.get("polygon_wkt_xy_pre", "")
            polys = parse_wkt_geometry_xy(wkt)
            if polys is None:
                continue
            # Filter out polygons with georeferenced (non-pixel) coordinates.
            # Pixel coords should be in [0, ~2000]; geo coords are 1e6+.
            MAX_PIXEL = 10000
            bad = False
            for pts in polys:
                for x, y in pts:
                    if abs(x) > MAX_PIXEL or abs(y) > MAX_PIXEL:
                        bad = True
                        break
                if bad:
                    break
            if bad:
                continue
            info = m2b_data.get(uid, {"class": -1, "confidence": 0.25, "n_valid": 0})
            buildings.append({
                "uid": uid,
                "polygon_geoms": polys,
                "m2b_class": info["class"],
                "confidence": info["confidence"],
                "n_valid": info["n_valid"],
            })
    return buildings


def make_overlay(cell_id: str, buildings: list[dict], bg_img: np.ndarray,
                 date_str: str, out_path: Path, dpi: int = 150):
    """Generate one QC overlay image."""
    h, w = bg_img.shape[:2]
    enhanced = enhance_rgb(bg_img)

    fig_w = 12
    fig_h = fig_w * h / w
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h), dpi=dpi)
    ax.imshow(enhanced, extent=[0, w, h, 0], interpolation="bilinear")

    # Draw polygons
    for b in buildings:
        cls = b["m2b_class"]
        color = DAMAGE_COLORS.get(cls, DAMAGE_COLORS[-1])
        conf = b["confidence"]
        alpha = 0.15 + 0.55 * max(0, (conf - 0.25) / 0.75)
        alpha = max(0.15, min(alpha, 0.70))

        polys = b["polygon_geoms"]
        for pts in polys:
            poly = MplPolygon(pts, closed=True, facecolor=(*color, alpha),
                              edgecolor=(*color, min(alpha + 0.3, 1.0)),
                              linewidth=0.8)
            ax.add_patch(poly)

        # Label at centroid
        cx, cy = geometry_centroid(polys)
        label_cls = str(cls) if cls >= 0 else "?"
        label = f"{label_cls} {conf:.2f}"
        ax.text(cx, cy, label, fontsize=4, ha="center", va="center",
                color="white", fontweight="bold",
                path_effects=[patheffects.withStroke(
                    linewidth=1.5, foreground="black")])

    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_axis_off()

    # Title
    class_counts = Counter(b["m2b_class"] for b in buildings)
    count_str = "  ".join(f"{DAMAGE_LABELS.get(k, '?')}:{v}"
                          for k, v in sorted(class_counts.items()))
    ax.set_title(f"{cell_id}  |  date={date_str}  |  {len(buildings)} buildings  |  {count_str}",
                 fontsize=8, pad=4)

    # Legend
    legend_patches = [mpatches.Patch(facecolor=DAMAGE_COLORS[k], label=DAMAGE_LABELS[k])
                      for k in sorted(DAMAGE_COLORS.keys()) if k in class_counts]
    if legend_patches:
        ax.legend(handles=legend_patches, loc="lower right", fontsize=5,
                  framealpha=0.7, edgecolor="none")

    fig.tight_layout(pad=0.3)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Discover cells
    all_cells = sorted([
        d.name for d in args.run_root.iterdir()
        if d.is_dir() and d.name.startswith("cell_")
        and (d / "dates").exists()
        and not (d / "shared_base/zero_instances.marker").exists()
    ])

    if args.cells:
        all_cells = [c for c in all_cells if c in args.cells]

    print(f"Generating QC overlays for {len(all_cells)} cells → {args.out_dir}")

    n_ok = n_skip = 0
    all_buildings = []  # collect for summary report

    for i, cell_id in enumerate(all_cells, 1):
        buildings = load_buildings(cell_id, args.run_root)
        if not buildings:
            n_skip += 1
            continue

        # Tag each building with its cell for the report
        for b in buildings:
            b["cell_id"] = cell_id
        all_buildings.extend(buildings)

        result = best_post_date(cell_id, args.run_root, args.chips_root)
        if result is None:
            print(f"  [{i}/{len(all_cells)}] {cell_id}: no valid post image — skipped")
            n_skip += 1
            continue

        date_str, tif_path = result
        bg_img = read_tif_rgb(tif_path)
        out_path = args.out_dir / f"{cell_id}_qc_m2b.png"

        try:
            make_overlay(cell_id, buildings, bg_img, date_str, out_path, dpi=args.dpi)
            n_ok += 1
        except Exception as e:
            print(f"  [{i}/{len(all_cells)}] {cell_id}: ERROR — {e}")
            n_skip += 1
            continue

        if i % 10 == 0 or i == len(all_cells):
            print(f"  [{i}/{len(all_cells)}] done={n_ok} skip={n_skip}")

    print(f"\n[done] {n_ok} overlays generated, {n_skip} skipped")
    print(f"[done] output: {args.out_dir}")

    # ── Summary report ──────────────────────────────────────────────────
    total = len(all_buildings)
    class_counts = Counter(b["m2b_class"] for b in all_buildings)
    cells_with_bldgs = len(set(b["cell_id"] for b in all_buildings))

    report_lines = [
        "=" * 60,
        "  PIPELINE RUN SUMMARY",
        "=" * 60,
        f"  Total cells in manifest:       {len(all_cells)}",
        f"  Cells with buildings:          {cells_with_bldgs}",
        f"  QC overlays generated:         {n_ok}",
        f"  Cells skipped:                 {n_skip}",
        "",
        f"  Total buildings detected:      {total}",
        "-" * 60,
        "  Damage classification (M2b):",
    ]
    for cls_id in sorted(class_counts.keys()):
        label = DAMAGE_LABELS.get(cls_id, f"class_{cls_id}")
        count = class_counts[cls_id]
        pct = 100.0 * count / total if total else 0
        report_lines.append(f"    {label:<22s}  {count:>7,d}  ({pct:5.1f}%)")
    report_lines.append("-" * 60)

    # Per-cell top damaged
    cell_damage = Counter()
    cell_destroyed = Counter()
    for b in all_buildings:
        if b["m2b_class"] >= 2:  # major or destroyed
            cell_damage[b["cell_id"]] += 1
        if b["m2b_class"] == 3:
            cell_destroyed[b["cell_id"]] += 1

    if cell_damage:
        report_lines.append("  Most damaged cells (major + destroyed):")
        for cell_id, count in cell_damage.most_common(10):
            n_dest = cell_destroyed.get(cell_id, 0)
            report_lines.append(f"    {cell_id}:  {count} damaged  ({n_dest} destroyed)")
    report_lines.append("=" * 60)

    report_text = "\n".join(report_lines)
    print(f"\n{report_text}")

    # Write report to file
    report_path = args.out_dir / "run_summary_report.txt"
    with open(report_path, "w") as f:
        f.write(report_text + "\n")
    print(f"\n[done] Report saved to {report_path}")


if __name__ == "__main__":
    main()
