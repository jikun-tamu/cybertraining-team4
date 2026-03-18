#!/usr/bin/env python3
"""Generate visualization maps from the combined building damage dataset.

Produces (full + east + west zoom for each):
  1. building_damage_map_{full,east,west}.png  — per-building damage class
  2. damage_density_map_{full,east,west}.png   — spatial density
  3. per_cell_damage_pct_{full,east,west}.png  — per-cell % damaged
  4. uncertainty_map_{full,east,west}.png      — prediction stability
  5. highlighted_areas_{full,east,west}.png    — clusters labeled
  6. damage_summary.md                         — summary report
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
import matplotlib.cm as cm

PKG_ROOT = Path(__file__).resolve().parents[1]

# Damage class styling
DAMAGE_COLORS = {
    0: "#2ecc71",   # no damage — green
    1: "#f39c12",   # minor — amber
    2: "#e67e22",   # major — orange
    3: "#e74c3c",   # destroyed — red
    -1: "#95a5a6",  # unknown — grey
}
DAMAGE_LABELS = {0: "No damage", 1: "Minor", 2: "Major", 3: "Destroyed", -1: "Unknown"}

# Data-driven bounding boxes (lon_min, lon_max, lat_min, lat_max)
# West cluster: Malibu/coastal  data: lon -118.645–-118.542, lat 34.059–34.094
BBOX_WEST = (-118.68, -118.50, 34.03, 34.11)
# East cluster: Pasadena/Altadena  data: lon -118.144–-118.037, lat 34.162–34.225
BBOX_EAST = (-118.17, -117.99, 34.14, 34.25)

ZOOM_CONFIGS = [
    ("full",  None,      ""),
    ("west",  BBOX_WEST, " — West Cluster (Malibu / Coastal)"),
    ("east",  BBOX_EAST, " — East Cluster (Pasadena / Altadena)"),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_geojson(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)["features"]


def safe_float(v, default=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def safe_int(v, default=-1):
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def is_unstable_flag(v) -> bool:
    return v if isinstance(v, bool) else str(v).lower() == "true"


def geojson_polygon_to_pts(geom: dict) -> list[tuple[float, float]]:
    return [(c[0], c[1]) for c in geom["coordinates"][0]]


def filter_features(features: list[dict], bbox: tuple | None) -> list[dict]:
    """Return features whose centroid falls within bbox (lon_min,lon_max,lat_min,lat_max)."""
    if bbox is None:
        return features
    lo, hi, la, lb = bbox
    out = []
    for f in features:
        lon = safe_float(f["properties"].get("centroid_lon"))
        lat = safe_float(f["properties"].get("centroid_lat"))
        if lon is not None and lat is not None and lo <= lon <= hi and la <= lat <= lb:
            out.append(f)
    return out


def figsize_for_bbox(bbox: tuple | None, features: list[dict] | None = None,
                     base_w: float = 14) -> tuple[float, float]:
    """Compute figsize so geographic proportions are correct (no stretching).

    At LA latitude (~34.15°) one degree of longitude spans cos(34.15°) ≈ 0.828
    times the distance of one degree of latitude.  The correct figure width/height
    ratio is therefore:
        (lon_span * cos_lat) / lat_span

    Args:
        bbox: (lon_min, lon_max, lat_min, lat_max).  If None, auto-computed from
              feature centroids.
        features: used only when bbox is None.
        base_w: fixed figure width in inches; height is derived from aspect.
    """
    if bbox is None:
        if features:
            lons = [safe_float(f["properties"].get("centroid_lon"))
                    for f in features if safe_float(f["properties"].get("centroid_lon"))]
            lats = [safe_float(f["properties"].get("centroid_lat"))
                    for f in features if safe_float(f["properties"].get("centroid_lat"))]
            if lons and lats:
                pad = 0.05   # degrees of padding on each side
                bbox = (min(lons) - pad, max(lons) + pad,
                        min(lats) - pad, max(lats) + pad)
        if bbox is None:
            return (base_w, base_w * 0.5)   # safe fallback

    lon_span  = bbox[1] - bbox[0]
    lat_span  = bbox[3] - bbox[2]
    lat_mid   = (bbox[2] + bbox[3]) / 2
    cos_lat   = math.cos(math.radians(lat_mid))
    # geographic aspect ratio: width-in-distance / height-in-distance
    geo_aspect = (lon_span * cos_lat) / lat_span if lat_span > 0 else 1.0
    h = base_w / geo_aspect
    # Clamp: avoid absurdly tall or flat figures
    h = max(4.0, min(h, base_w * 1.8))
    return (base_w, round(h, 1))


def setup_fig(figsize=(16, 14)) -> tuple:
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("auto")
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#1a1a2e")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#555")
    return fig, ax


def apply_bbox_limits(ax, bbox: tuple | None):
    if bbox:
        ax.set_xlim(bbox[0], bbox[1])
        ax.set_ylim(bbox[2], bbox[3])


def add_legend_damage(ax):
    patches = [mpatches.Patch(color=DAMAGE_COLORS[c], label=DAMAGE_LABELS[c])
               for c in [0, 1, 2, 3]]
    ax.legend(handles=patches, loc="lower left", framealpha=0.85,
              facecolor="#222", labelcolor="white", title="Damage Class",
              title_fontsize=10, fontsize=9)


def save_fig(fig, out_path: Path, dpi: int = 200):
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[map] {out_path}")


# ── Map 1: Building-level damage map ─────────────────────────────────────────

def map_building_damage(features: list[dict], out_path: Path,
                        bbox: tuple | None = None, subtitle: str = ""):
    feats = filter_features(features, bbox)
    fig, ax = setup_fig(figsize_for_bbox(bbox, feats))

    for feat in feats:
        try:
            pts = geojson_polygon_to_pts(feat["geometry"])
            cls = safe_int(feat["properties"].get("m1b_damage_class", -1))
            color = DAMAGE_COLORS.get(cls, DAMAGE_COLORS[-1])
            xs, ys = zip(*pts)
            ax.fill(xs, ys, color=color, alpha=0.80, linewidth=0)
            ax.plot(xs, ys, color="white", alpha=0.20, linewidth=0.4)
        except Exception:
            pass

    apply_bbox_limits(ax, bbox)
    add_legend_damage(ax)
    title = "Building-Level Damage Assessment — LA Fire 2025" + subtitle + "\n"
    title += "(M1: Probability Averaging across post-disaster dates)"
    ax.set_title(title, color="white", fontsize=13, pad=12)
    ax.set_xlabel("Longitude", color="white")
    ax.set_ylabel("Latitude", color="white")
    save_fig(fig, out_path, dpi=200)


# ── Map 2: Damage density ─────────────────────────────────────────────────────

def map_damage_density(features: list[dict], out_path: Path,
                       bbox: tuple | None = None, subtitle: str = ""):
    feats = filter_features(features, bbox)
    lons_ok, lats_ok, lons_dmg, lats_dmg = [], [], [], []

    for feat in feats:
        props = feat["properties"]
        lon = safe_float(props.get("centroid_lon"))
        lat = safe_float(props.get("centroid_lat"))
        if lon is None or lat is None:
            continue
        cls = safe_int(props.get("m1b_damage_class", -1))
        if cls in (2, 3):
            lons_dmg.append(lon); lats_dmg.append(lat)
        else:
            lons_ok.append(lon); lats_ok.append(lat)

    fig, ax = setup_fig(figsize_for_bbox(bbox, feats))
    gridsize = 40 if bbox else 60

    if lons_ok:
        hb = ax.hexbin(lons_ok, lats_ok, gridsize=gridsize, cmap="Blues",
                       mincnt=1, alpha=0.5, linewidths=0)
        cb = plt.colorbar(hb, ax=ax, label="Building count (no/minor damage)", fraction=0.03)
        cb.ax.yaxis.set_tick_params(color="white")
        cb.ax.yaxis.label.set_color("white")

    pt_size = 12 if bbox else 6
    if lons_dmg:
        ax.scatter(lons_dmg, lats_dmg, c="#e74c3c", s=pt_size, alpha=0.9,
                   linewidth=0, label="Major/Destroyed", zorder=5)

    apply_bbox_limits(ax, bbox)
    ax.set_title("Damage Density — LA Fire 2025" + subtitle +
                 "\nRed dots = major/destroyed buildings",
                 color="white", fontsize=12)
    ax.set_xlabel("Longitude", color="white")
    ax.set_ylabel("Latitude", color="white")
    ax.legend(fontsize=9, facecolor="#333", labelcolor="white", loc="lower left")
    save_fig(fig, out_path, dpi=200)


# ── Map 3: Per-cell damage percentage ────────────────────────────────────────

def map_per_cell_damage_pct(features: list[dict], out_path: Path,
                             bbox: tuple | None = None, subtitle: str = ""):
    feats = filter_features(features, bbox)
    cell_stats: dict = defaultdict(lambda: {"total":0,"damaged":0,"lons":[],"lats":[]})

    for feat in feats:
        props = feat["properties"]
        cid = props.get("cell_id", "")
        lon = safe_float(props.get("centroid_lon"))
        lat = safe_float(props.get("centroid_lat"))
        if not cid or lon is None or lat is None:
            continue
        cls = safe_int(props.get("m1b_damage_class", -1))
        cell_stats[cid]["total"] += 1
        if cls in (2, 3):
            cell_stats[cid]["damaged"] += 1
        cell_stats[cid]["lons"].append(lon)
        cell_stats[cid]["lats"].append(lat)

    fig, ax = setup_fig(figsize_for_bbox(bbox, feats))
    cmap = plt.get_cmap("RdYlGn_r")
    norm = Normalize(vmin=0, vmax=100)

    label_threshold = 3 if bbox else 5
    for cid, stats in cell_stats.items():
        if not stats["lons"]:
            continue
        pct = 100.0 * stats["damaged"] / stats["total"] if stats["total"] else 0
        cx = np.mean(stats["lons"])
        cy = np.mean(stats["lats"])
        color = cmap(norm(pct))
        size = max(30, min(400, stats["total"] * (5 if bbox else 3)))
        ax.scatter(cx, cy, c=[color], s=size, alpha=0.85, linewidth=0.5,
                   edgecolors="white")
        if stats["total"] >= label_threshold:
            fs = 7 if bbox else 5
            ax.text(cx, cy, f"{pct:.0f}%", ha="center", va="center",
                    fontsize=fs, color="white", weight="bold")

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, fraction=0.03,
                      label="% buildings major/destroyed")
    cb.ax.yaxis.set_tick_params(color="white")
    cb.ax.yaxis.label.set_color("white")

    apply_bbox_limits(ax, bbox)
    ax.set_title("Per-Cell Damage Percentage — LA Fire 2025" + subtitle + "\n"
                 "(circle size ∝ n buildings; color = % major/destroyed)",
                 color="white", fontsize=12)
    ax.set_xlabel("Longitude", color="white")
    ax.set_ylabel("Latitude", color="white")
    save_fig(fig, out_path, dpi=200)


# ── Map 4: Uncertainty map ────────────────────────────────────────────────────

def map_uncertainty(features: list[dict], out_path: Path,
                    bbox: tuple | None = None, subtitle: str = ""):
    feats = filter_features(features, bbox)
    fig, ax = setup_fig(figsize_for_bbox(bbox, feats))

    for feat in feats:
        props = feat["properties"]
        lon = safe_float(props.get("centroid_lon"))
        lat = safe_float(props.get("centroid_lat"))
        if lon is None or lat is None:
            continue
        entropy = safe_float(props.get("label_entropy"), 0.0)
        unstable = is_unstable_flag(props.get("is_unstable", False))
        alpha = max(0.15, min(1.0, entropy * 1.5))
        color = "#e74c3c" if unstable else "#2ecc71"
        size = max(6, entropy * (50 if bbox else 30))
        ax.scatter(lon, lat, c=color, s=size, alpha=alpha, linewidth=0)

    apply_bbox_limits(ax, bbox)
    patches = [
        mpatches.Patch(color="#e74c3c", label="Unstable (conflicting dates)"),
        mpatches.Patch(color="#2ecc71", label="Stable (consistent dates)"),
    ]
    ax.legend(handles=patches, loc="lower left", facecolor="#333",
              labelcolor="white", fontsize=9)
    ax.set_title("Prediction Uncertainty — LA Fire 2025" + subtitle + "\n"
                 "(dot size ∝ entropy; red = conflicting damage labels across dates)",
                 color="white", fontsize=12)
    ax.set_xlabel("Longitude", color="white")
    ax.set_ylabel("Latitude", color="white")
    save_fig(fig, out_path, dpi=200)


# ── Map 5: Highlighted areas ──────────────────────────────────────────────────

def _cluster_label(pct_destroyed: float, pct_unstable: float) -> str | None:
    if pct_destroyed >= 50:
        return "HIGH DAMAGE"
    if pct_unstable >= 50:
        return "UNCERTAIN"
    if 15 <= pct_destroyed < 50:
        return "MIXED"
    return None


def map_highlighted_areas(features: list[dict], out_path: Path,
                           bbox: tuple | None = None, subtitle: str = ""):
    feats = filter_features(features, bbox)
    cells: dict = defaultdict(lambda: {"total":0,"destroyed":0,"unstable":0,
                                        "lons":[],"lats":[]})
    for feat in feats:
        props = feat["properties"]
        cid = props.get("cell_id", "")
        lon = safe_float(props.get("centroid_lon"))
        lat = safe_float(props.get("centroid_lat"))
        if not cid or lon is None or lat is None:
            continue
        cls = safe_int(props.get("m1b_damage_class", -1))
        unstable = is_unstable_flag(props.get("is_unstable", False))
        cells[cid]["total"] += 1
        if cls == 3:
            cells[cid]["destroyed"] += 1
        if unstable:
            cells[cid]["unstable"] += 1
        cells[cid]["lons"].append(lon)
        cells[cid]["lats"].append(lat)

    fig, ax = setup_fig(figsize_for_bbox(bbox, feats))

    all_lons = [safe_float(f["properties"].get("centroid_lon")) for f in feats
                if safe_float(f["properties"].get("centroid_lon"))]
    all_lats = [safe_float(f["properties"].get("centroid_lat")) for f in feats
                if safe_float(f["properties"].get("centroid_lat"))]
    ax.scatter(all_lons, all_lats, c="#555", s=3, alpha=0.35, linewidth=0)

    cluster_colors = {"HIGH DAMAGE": "#e74c3c", "MIXED": "#f39c12", "UNCERTAIN": "#9b59b6"}
    highlighted = []
    circle_r = 0.002 if bbox else 0.003
    label_fs = 7 if bbox else 5

    for cid, stats in cells.items():
        if stats["total"] < 3:
            continue
        pct_d = 100 * stats["destroyed"] / stats["total"]
        pct_u = 100 * stats["unstable"] / stats["total"]
        label = _cluster_label(pct_d, pct_u)
        if label is None:
            continue
        cx = np.mean(stats["lons"])
        cy = np.mean(stats["lats"])
        color = cluster_colors[label]
        circle = plt.Circle((cx, cy), circle_r, color=color, alpha=0.25, linewidth=0)
        ax.add_patch(circle)
        ax.scatter(cx, cy, c=color, s=100, alpha=0.9, linewidth=0.5,
                   edgecolors="white", zorder=5)
        ax.text(cx + circle_r * 0.5, cy + circle_r * 0.5,
                cid.replace("cell_", ""), fontsize=label_fs,
                color=color, alpha=0.9, zorder=6, weight="bold")
        highlighted.append((cid, label, pct_d, pct_u, stats["total"]))

    apply_bbox_limits(ax, bbox)
    patches = [mpatches.Patch(color=v, label=k) for k, v in cluster_colors.items()]
    patches.append(mpatches.Patch(color="#555", label="Other buildings"))
    ax.legend(handles=patches, loc="lower left", facecolor="#333",
              labelcolor="white", fontsize=9)
    ax.set_title(f"Representative Areas — LA Fire 2025" + subtitle + f"\n"
                 f"Highlighted: {len(highlighted)} notable cells",
                 color="white", fontsize=12)
    ax.set_xlabel("Longitude", color="white")
    ax.set_ylabel("Latitude", color="white")
    save_fig(fig, out_path, dpi=200)
    print(f"         highlighted={len(highlighted)} cells")
    return highlighted


# ── Summary statistics report ─────────────────────────────────────────────────

def write_summary_report(features: list[dict], highlighted: list,
                          out_path: Path, run_root: Path):
    rows = [f["properties"] for f in features]
    n_total = len(rows)
    damage_counts = Counter(safe_int(r.get("m1b_damage_class", -1)) for r in rows)
    n_unstable = sum(1 for r in rows if is_unstable_flag(r.get("is_unstable", False)))
    n_cells = len(set(r.get("cell_id") for r in rows))

    n_zero = sum(1 for d in run_root.iterdir()
                 if d.is_dir() and (d / "shared_base/zero_instances.marker").exists())

    n_dates_total = n_dates_rejected = 0
    for d in run_root.iterdir():
        if not d.is_dir():
            continue
        for qj in ((d / "dates").glob("*/quality_metrics.json")
                   if (d / "dates").exists() else []):
            try:
                m = json.loads(qj.read_text())
                n_dates_total += 1
                if not m.get("tile_quality_ok", True):
                    n_dates_rejected += 1
            except Exception:
                pass

    def pct(n, d): return f"{100*n//d}%" if d else "—"

    lines = [
        "# LA Fire 2025 — Building Damage Assessment",
        f"**Generated**: 2026-03-15  |  **Method**: Multi-date probability averaging (M1)  |  **Model**: unchanged (no retraining)",
        "",
        "## Dataset Overview",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Cells processed | {n_cells} of 295 |",
        f"| Cells with 0 Stage-1 detections (sparse terrain) | {n_zero} |",
        f"| Total building instances | {n_total:,} |",
        f"| Post-date images evaluated | {n_dates_total} |",
        f"| Post-date images rejected by quality filter | {n_dates_rejected} ({pct(n_dates_rejected, n_dates_total)}) |",
        "",
        "## Damage Distribution (M1 Probability Averaging)",
        "",
        "| Damage Class | Count | % |",
        "|---|---|---|",
    ]
    for cls in [0, 1, 2, 3, -1]:
        cnt = damage_counts.get(cls, 0)
        lines.append(f"| {DAMAGE_LABELS[cls]} ({cls}) | {cnt:,} | {pct(cnt, n_total)} |")

    lines += [
        "",
        "## Prediction Stability",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Instances with conflicting labels across dates (unstable) | {n_unstable:,} ({pct(n_unstable, n_total)}) |",
        f"| Instances with stable labels | {n_total - n_unstable:,} ({pct(n_total - n_unstable, n_total)}) |",
        "",
        "## Notable Areas",
        "",
        "| Cell | Category | % Destroyed | % Unstable | n Buildings |",
        "|---|---|---|---|---|",
    ]
    for cid, label, pct_d, pct_u, n_bldg in sorted(highlighted, key=lambda x: -x[2])[:20]:
        lines.append(f"| {cid} | {label} | {pct_d:.0f}% | {pct_u:.0f}% | {n_bldg} |")

    lines += [
        "",
        "## Maps Generated",
        "",
        "| File | Description |",
        "|---|---|",
        "| `building_damage_map_{full,east,west}.png` | Per-building damage class (colored polygons) |",
        "| `damage_density_map_{full,east,west}.png` | Spatial density hexbins + major/destroyed scatter |",
        "| `per_cell_damage_pct_{full,east,west}.png` | Circle per cell, color = % major+destroyed |",
        "| `uncertainty_map_{full,east,west}.png` | Prediction stability: red = conflicting labels |",
        "| `highlighted_areas_{full,east,west}.png` | Annotated map of notable clusters |",
        "| `overlay_east_*.png` | Post-disaster imagery + damage polygon overlay |",
        "| `overlay_west_*.png` | Post-disaster imagery + damage polygon overlay |",
        "",
        "## Files",
        "",
        "| File | Description |",
        "|---|---|",
        "| `outputs/multidate_full_run/building_damage_all_cells.csv` | Flat CSV, all building instances |",
        "| `outputs/maps/building_damage.geojson` | WGS84 GeoJSON polygons |",
        "| `outputs/maps/building_damage.gpkg` | UTM GeoPackage |",
        "",
        "## Notes",
        "",
        "- Damage classes: 0=no damage, 1=minor, 2=major, 3=destroyed",
        "- M1 = probability averaging across all quality-passing post dates",
        "- Quality filter: rejects dates with mean_brightness < 15, frac_zeros > 0.60, or spatial_std < 3.0",
        "- Stage-1 building detection: SAM3, text prompt 'building', tile_size=512, overlap=64, min_size=30",
        "- Stage-2b ensemble: 3 checkpoints, weights 4:3:2, temperature calibration",
        "- **No model retraining** — all weights unchanged from Jikun's trained ensemble",
    ]

    out_path.write_text("\n".join(lines))
    print(f"[done] Summary report: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--geojson", type=Path,
                   default=PKG_ROOT / "outputs/maps/building_damage.geojson")
    p.add_argument("--out_dir", type=Path,
                   default=PKG_ROOT / "outputs/maps")
    p.add_argument("--run_root", type=Path,
                   default=PKG_ROOT / "outputs/multidate_full_run")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading GeoJSON from {args.geojson}...")
    features = load_geojson(args.geojson)
    print(f"Loaded {len(features)} features")

    all_highlighted = []
    for variant, bbox, subtitle in ZOOM_CONFIGS:
        tag = f"_{variant}"
        map_building_damage(features,
                            args.out_dir / f"building_damage_map{tag}.png",
                            bbox=bbox, subtitle=subtitle)
        map_damage_density(features,
                           args.out_dir / f"damage_density_map{tag}.png",
                           bbox=bbox, subtitle=subtitle)
        map_per_cell_damage_pct(features,
                                args.out_dir / f"per_cell_damage_pct{tag}.png",
                                bbox=bbox, subtitle=subtitle)
        map_uncertainty(features,
                        args.out_dir / f"uncertainty_map{tag}.png",
                        bbox=bbox, subtitle=subtitle)
        highlighted = map_highlighted_areas(features,
                                            args.out_dir / f"highlighted_areas{tag}.png",
                                            bbox=bbox, subtitle=subtitle)
        if variant == "full":
            all_highlighted = highlighted

    write_summary_report(features, all_highlighted,
                         args.out_dir / "damage_summary.md", args.run_root)


if __name__ == "__main__":
    main()
