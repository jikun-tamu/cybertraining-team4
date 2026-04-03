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

# ── Paths ────────────────────────────────────────────────────────────────────
LA_FIRE_ROOT = Path("/media/data/building_instance_tamu/la_fire_2025")
CHIPS_ROOT   = LA_FIRE_ROOT / "chips"
RUN_ROOT     = LA_FIRE_ROOT / "stage2_damage/multidate_full_run"

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

def parse_wkt_polygon_xy(wkt: str) -> np.ndarray | None:
    if not wkt or not wkt.strip().upper().startswith("POLYGON"):
        return None
    nums = [float(x) for x in WKT_FLOAT_RE.findall(wkt)]
    if len(nums) < 6 or len(nums) % 2 != 0:
        return None
    return np.array(nums, dtype=np.float32).reshape(-1, 2)


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
            pts = parse_wkt_polygon_xy(wkt)
            if pts is None:
                continue
            info = m2b_data.get(uid, {"class": -1, "confidence": 0.25, "n_valid": 0})
            buildings.append({
                "uid": uid,
                "polygon_xy": pts,
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

        pts = b["polygon_xy"]
        poly = MplPolygon(pts, closed=True, facecolor=(*color, alpha),
                          edgecolor=(*color, min(alpha + 0.3, 1.0)),
                          linewidth=0.8)
        ax.add_patch(poly)

        # Label at centroid
        cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
        label = str(cls) if cls >= 0 else "?"
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
    for i, cell_id in enumerate(all_cells, 1):
        buildings = load_buildings(cell_id, args.run_root)
        if not buildings:
            n_skip += 1
            continue

        result = best_post_date(cell_id, args.run_root, args.chips_root)
        if result is None:
            print(f"  [{i}/{len(all_cells)}] {cell_id}: no valid post image — skipped")
            n_skip += 1
            continue

        date_str, tif_path = result
        bg_img = read_tif_rgb(tif_path)
        out_path = args.out_dir / f"{cell_id}_qc_m2b.png"

        make_overlay(cell_id, buildings, bg_img, date_str, out_path, dpi=args.dpi)
        n_ok += 1

        if i % 10 == 0 or i == len(all_cells):
            print(f"  [{i}/{len(all_cells)}] done={n_ok} skip={n_skip}")

    print(f"\n[done] {n_ok} overlays generated, {n_skip} skipped")
    print(f"[done] output: {args.out_dir}")


if __name__ == "__main__":
    main()
