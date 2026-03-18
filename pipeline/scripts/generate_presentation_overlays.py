#!/usr/bin/env python3
"""Generate presentation-quality overlays: post-disaster imagery + damage polygons.

Per-building confidence is shown two ways:
  1. Polygon opacity — high opacity = confident, low opacity = uncertain
  2. Text label at centroid — shows exact confidence % (e.g. "73%")

Confidence = max(calibrated probability vector) from M1b aggregation.
Damage class from M1b (coverage-aware: only uses dates where the building
footprint has ≤50% zero/nodata pixels).

Generates overlays for all cells in the GeoJSON (not just a curated subset).
East cluster = lon > -118.35 (Altadena / Eaton Fire)
West cluster = lon ≤ -118.35 (Pacific Palisades / Malibu)
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
from matplotlib.patheffects import withStroke

# ── Paths ─────────────────────────────────────────────────────────────────────
LA_FIRE_ROOT   = Path("/media/data/la_fire_2025")
CHIPS_ROOT     = LA_FIRE_ROOT / "chips"
GEOJSON        = LA_FIRE_ROOT / "final_maps/maps/building_damage.geojson"
RUN_ROOT       = LA_FIRE_ROOT / "stage2_damage/multidate_full_run"
OUT_DIR        = LA_FIRE_ROOT / "overlays_v2"

# Longitude threshold separating west (Pacific Palisades) from east (Altadena)
LON_SPLIT = -118.35

DAMAGE_COLORS = {
    0:  (0.18, 0.80, 0.44),   # green  — no damage
    1:  (0.95, 0.61, 0.07),   # amber  — minor
    2:  (0.90, 0.49, 0.13),   # orange — major
    3:  (0.91, 0.30, 0.24),   # red    — destroyed
    -1: (0.58, 0.65, 0.65),   # grey   — not identifiable
}
DAMAGE_LABELS = {
    0: "No damage", 1: "Minor damage", 2: "Major damage",
    3: "Destroyed", -1: "Not identifiable",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def safe_int(v, default=-1):
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def load_geojson(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)["features"]


def load_confidence(cell_id: str) -> dict[str, float]:
    """Return {bldg_uid: confidence} from JSONL aggregated predictions.

    Confidence = max(m1b_coverage_probs).
    Falls back to 0.25 (uniform) if record missing.
    """
    jsonl = RUN_ROOT / cell_id / "aggregated_predictions.jsonl"
    result: dict[str, float] = {}
    if not jsonl.exists():
        return result
    for line in jsonl.read_text().strip().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        uid = rec.get("bldg_uid", "")
        probs = rec.get("m1b_coverage_probs", [0.25, 0.25, 0.25, 0.25])
        result[uid] = float(max(probs)) if probs else 0.25
    return result


def load_stability(cell_id: str) -> dict[str, bool]:
    """Return {bldg_uid: is_stable}."""
    jsonl = RUN_ROOT / cell_id / "aggregated_predictions.jsonl"
    result: dict[str, bool] = {}
    if not jsonl.exists():
        return result
    for line in jsonl.read_text().strip().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        uid = rec.get("bldg_uid", "")
        result[uid] = not rec.get("is_unstable", False)
    return result


def confidence_to_alpha(conf: float) -> float:
    """Map confidence [0.25, 1.0] → polygon alpha [0.15, 0.78]."""
    alpha = 0.15 + 0.63 * (conf - 0.25) / 0.75
    return max(0.12, min(alpha, 0.80))


def best_post_date(cell_id: str) -> tuple[str, Path] | None:
    """Return (date_str, tif_path) maximising building coverage then brightness."""
    dates_dir = RUN_ROOT / cell_id / "dates"
    if not dates_dir.exists():
        return None

    candidates = []
    for d in sorted(dates_dir.iterdir()):
        qm_path = d / "quality_metrics.json"
        if not qm_path.exists():
            continue
        qm = json.loads(qm_path.read_text())
        if not qm.get("tile_quality_ok", False):
            continue
        tif_path = CHIPS_ROOT / cell_id / "post" / f"{cell_id}_post_{d.name}.tif"
        if not tif_path.exists():
            continue

        n_ok = n_total = 0
        shared_csv = d / "shared_for_date.csv"
        if shared_csv.exists():
            rows = list(csv.DictReader(open(shared_csv)))
            n_total = len(rows)
            n_ok = sum(1 for r in rows if r.get("quality_ok", "false").lower() == "true")
        coverage_frac = n_ok / n_total if n_total > 0 else 0.0
        brightness = float(qm.get("mean_brightness", 0))
        candidates.append((coverage_frac, brightness, d.name, tif_path))

    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    cov, brt, date_str, tif_path = candidates[0]
    print(f"         date={date_str}  coverage={cov:.0%}  brightness={brt:.1f}")
    return date_str, tif_path


def read_tif_rgb(tif_path: Path):
    import rasterio
    with rasterio.open(tif_path) as ds:
        arr = ds.read([1, 2, 3]).transpose(1, 2, 0)
        bounds = ds.bounds
        crs = ds.crs
        transform = ds.transform
    return arr, bounds, crs, transform


def utm_bounds_to_wgs84(bounds, crs):
    from pyproj import Transformer
    t = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lon_min, lat_min = t.transform(bounds.left, bounds.bottom)
    lon_max, lat_max = t.transform(bounds.right, bounds.top)
    return lon_min, lon_max, lat_min, lat_max


def get_cell_features(features: list[dict], cell_id: str) -> list[dict]:
    return [f for f in features if f["properties"].get("cell_id") == cell_id]


def enhance_rgb(arr: np.ndarray) -> np.ndarray:
    out = np.zeros_like(arr, dtype=np.float32)
    for c in range(3):
        ch = arr[:, :, c].astype(np.float32)
        valid = ch[ch > 0]
        if len(valid) == 0:
            continue
        lo = np.percentile(valid, 2)
        hi = np.percentile(valid, 98)
        if hi > lo:
            out[:, :, c] = np.clip((ch - lo) / (hi - lo), 0, 1)
        else:
            out[:, :, c] = ch / 255.0
    return out


def load_pre_tif(cell_id: str):
    pre_dir = CHIPS_ROOT / cell_id / "pre"
    candidates = sorted(pre_dir.glob("*.tif")) if pre_dir.exists() else []
    if not candidates:
        return None, None, None
    return read_tif_rgb(candidates[0])


def make_composite_rgb(post_arr: np.ndarray, pre_arr: np.ndarray | None,
                       brightness_scale: float = 0.6) -> np.ndarray:
    post_enhanced = enhance_rgb(post_arr)
    nodata_mask = (post_arr.sum(axis=2) == 0)
    if pre_arr is None or not nodata_mask.any():
        return post_enhanced
    pre_enhanced = enhance_rgb(pre_arr) * brightness_scale
    composite = post_enhanced.copy()
    composite[nodata_mask] = pre_enhanced[nodata_mask]
    return composite


def polygon_area_degrees(pts: np.ndarray) -> float:
    """Shoelace formula for polygon area in degree² (proxy for screen size)."""
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


# ── Main overlay function ─────────────────────────────────────────────────────

def make_overlay(cluster: str, cell_id: str, description: str,
                 features: list[dict], out_dir: Path):
    """Create one presentation overlay figure.

    Confidence is shown as:
      - polygon fill opacity (existing method, retained)
      - text label at building centroid showing exact % (new)
    """
    print(f"[overlay] {cell_id}")
    result = best_post_date(cell_id)
    if result is None:
        print(f"  [skip] {cell_id}: no valid post date found")
        return
    date_str, tif_path = result

    post_arr, bounds, crs, _ = read_tif_rgb(tif_path)
    pre_arr, _, _, _ = load_pre_tif(cell_id)
    nodata_frac = (post_arr.sum(axis=2) == 0).mean()
    print(f"         nodata={nodata_frac:.1%}  pre_composite={'yes' if pre_arr is not None else 'no'}")

    rgb = make_composite_rgb(post_arr, pre_arr)
    lon_min, lon_max, lat_min, lat_max = utm_bounds_to_wgs84(bounds, crs)

    confidence = load_confidence(cell_id)
    stability  = load_stability(cell_id)
    cell_feats = get_cell_features(features, cell_id)
    n_buildings = len(cell_feats)

    # ── Figure setup ──────────────────────────────────────────────────────────
    lon_span = lon_max - lon_min
    lat_span = lat_max - lat_min
    lat_mid  = (lat_min + lat_max) / 2
    cos_lat  = math.cos(math.radians(lat_mid))
    geo_aspect = (lon_span * cos_lat) / lat_span if lat_span > 0 else 1.0
    fig_w = 14
    fig_h = max(4.0, min(fig_w / geo_aspect, fig_w * 1.5))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("#0d0d0d")
    ax.set_facecolor("#0d0d0d")

    # ── Background imagery ────────────────────────────────────────────────────
    ax.imshow(rgb, extent=[lon_min, lon_max, lat_min, lat_max],
              origin="upper", aspect="auto", interpolation="bilinear", zorder=1)

    # ── Damage polygons + confidence labels ───────────────────────────────────
    patches: list = []
    facecolors: list = []
    classes_present: set[int] = set()
    all_confs: list[float] = []
    n_stable = n_identifiable = 0

    # Pixel size in degrees (used to decide label font size)
    deg_per_px_lon = lon_span / (fig_w * 100)   # rough: 100 dpi equivalent
    deg_per_px_lat = lat_span / (fig_h * 100)
    min_label_area = deg_per_px_lon * deg_per_px_lat * 400  # ~20×20 px² threshold

    for feat in cell_feats:
        try:
            coords = feat["geometry"]["coordinates"][0]
            pts = np.array([[c[0], c[1]] for c in coords])
            props = feat["properties"]
            cls  = safe_int(props.get("m1b_damage_class", -1))
            uid  = props.get("bldg_uid", "")
            conf = confidence.get(uid, 0.25)
            alpha = confidence_to_alpha(conf)
            color_rgb = DAMAGE_COLORS.get(cls, DAMAGE_COLORS[-1])

            patches.append(MplPolygon(pts, closed=True))
            facecolors.append((*color_rgb, alpha))
            classes_present.add(cls)

            if cls != -1:
                all_confs.append(conf)
                n_identifiable += 1
                if stability.get(uid, True):
                    n_stable += 1

            # ── Confidence text label at centroid ──────────────────────────
            clon = props.get("centroid_lon")
            clat = props.get("centroid_lat")
            if clon and clat and polygon_area_degrees(pts) > min_label_area:
                # White text, thin black outline for readability on any background
                txt = ax.text(
                    float(clon), float(clat),
                    f"{conf:.0%}",
                    ha="center", va="center",
                    fontsize=4.5, color="white", fontweight="bold",
                    zorder=4,
                )
                txt.set_path_effects([
                    withStroke(linewidth=1.8, foreground="black")
                ])
        except Exception:
            pass

    if patches:
        coll = PatchCollection(patches, facecolors=facecolors,
                               edgecolors="white", linewidths=0.5, zorder=2)
        ax.add_collection(coll)

    mean_conf = statistics.mean(all_confs) if all_confs else 0.0
    pct_stable = n_stable / n_identifiable if n_identifiable > 0 else 0.0

    # ── Legend — damage class ─────────────────────────────────────────────────
    damage_order = [0, 1, 2, 3, -1]
    legend_patches = [
        mpatches.Patch(facecolor=(*DAMAGE_COLORS[c], 0.70), edgecolor="white",
                       linewidth=0.5, label=DAMAGE_LABELS[c])
        for c in damage_order if c in classes_present
    ]
    leg1 = ax.legend(handles=legend_patches, loc="lower left", framealpha=0.85,
                     facecolor="#111", labelcolor="white",
                     title="Damage Class (Stage 2)", title_fontsize=9, fontsize=8)
    leg1.get_title().set_color("white")
    ax.add_artist(leg1)

    # ── Legend — confidence key (opacity + label example) ────────────────────
    conf_handles = []
    for v in [0.30, 0.50, 0.75, 1.00]:
        lbl = f"conf ≈ {v:.0%}  (α={confidence_to_alpha(v):.2f})"
        conf_handles.append(
            mpatches.Patch(facecolor=(0.85, 0.85, 0.85, confidence_to_alpha(v)),
                           edgecolor="white", linewidth=0.5, label=lbl)
        )
    leg2 = ax.legend(handles=conf_handles, loc="lower right", framealpha=0.85,
                     facecolor="#111", labelcolor="white",
                     title="Stage-2 confidence\n(opacity + label %)", title_fontsize=9,
                     fontsize=8)
    leg2.get_title().set_color("#ccc")

    # ── Axes styling ──────────────────────────────────────────────────────────
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.tick_params(colors="#aaa", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.set_xlabel("Longitude", color="#aaa", fontsize=9)
    ax.set_ylabel("Latitude", color="#aaa", fontsize=9)

    # ── Title ─────────────────────────────────────────────────────────────────
    cluster_name = ("West Cluster (Pacific Palisades)"
                    if cluster == "west" else "East Cluster (Altadena / Eaton Fire)")
    ax.set_title(
        f"Building Damage Assessment — LA Wildfires 2025  |  {cluster_name}\n"
        f"{description}  |  Post-disaster date: {date_str}\n"
        f"Stage-1 detected: {n_buildings} buildings  "
        f"|  Mean model confidence: {mean_conf:.0%}  "
        f"|  Stable predictions: {pct_stable:.0%}",
        color="white", fontsize=11, pad=10,
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = out_dir / f"overlay_{cluster}_{cell_id}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[saved] {out_path.name}  conf={mean_conf:.0%}  stable={pct_stable:.0%}")


# ── Auto-discover all cells from GeoJSON ──────────────────────────────────────

def discover_cells(features: list[dict]) -> list[tuple[str, str, str]]:
    """Return [(cluster, cell_id, description)] for all cells in the GeoJSON."""
    # Group by cell
    cell_feats: dict[str, list] = defaultdict(list)
    cell_lons: dict[str, list] = defaultdict(list)
    for feat in features:
        props = feat["properties"]
        cid = props.get("cell_id", "")
        if not cid:
            continue
        cell_feats[cid].append(props)
        if props.get("centroid_lon"):
            cell_lons[cid].append(float(props["centroid_lon"]))

    result = []
    for cid in sorted(cell_feats.keys()):
        props_list = cell_feats[cid]
        n = len(props_list)
        n_damaged = sum(1 for p in props_list
                        if safe_int(p.get("m1b_damage_class", 0)) >= 1)
        pct = n_damaged / n if n > 0 else 0.0

        mean_lon = statistics.mean(cell_lons[cid]) if cell_lons[cid] else -118.3
        cluster = "west" if mean_lon <= LON_SPLIT else "east"
        area_name = "Pacific Palisades" if cluster == "west" else "Altadena"

        description = f"{area_name} — {pct:.0%} damaged ({n} buildings)"
        result.append((cluster, cid, description))

    return result


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    import argparse
    p = argparse.ArgumentParser(
        description="Generate damage overlay maps for all LA fire grid cells"
    )
    p.add_argument("--out_dir", type=Path, default=OUT_DIR)
    p.add_argument("--cells", nargs="*", default=None,
                   help="Process only these cell_ids (default: all cells in GeoJSON)")
    p.add_argument("--cluster", choices=["east", "west", "all"], default="all",
                   help="Filter by cluster (default: all)")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    features = load_geojson(GEOJSON)
    print(f"Loaded {len(features)} features from GeoJSON")

    cells = discover_cells(features)
    print(f"Discovered {len(cells)} cells")

    if args.cells:
        cells = [(cl, cid, desc) for cl, cid, desc in cells if cid in args.cells]
    if args.cluster != "all":
        cells = [(cl, cid, desc) for cl, cid, desc in cells if cl == args.cluster]

    print(f"Generating {len(cells)} overlays → {args.out_dir}\n")
    n_ok = n_skip = 0
    for cluster, cell_id, description in cells:
        try:
            make_overlay(cluster, cell_id, description, features, args.out_dir)
            n_ok += 1
        except Exception as e:
            print(f"  [error] {cell_id}: {e}")
            n_skip += 1

    print(f"\n[done] {n_ok} saved, {n_skip} skipped → {args.out_dir}")


if __name__ == "__main__":
    main()
