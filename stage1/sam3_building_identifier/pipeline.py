"""
Batch pipeline: for each image, run SAM3 (tiled or full image), merge
instances, save masks / annotation / prediction JSON, then write
run_summary.json for the whole run.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import subprocess
import time
import traceback
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import rasterio
from PIL import Image, ImageDraw
from rasterio.errors import NotGeoreferencedWarning
from tqdm import tqdm

from sam3_building_identifier.config import PipelineConfig
from sam3_building_identifier.mask_to_polygon import labels_to_instances
from sam3_building_identifier.model import SAM3Model
from sam3_building_identifier.tiling import (
    instance_from_mask, merge_instances, paint_labels, tile_windows,
)
from sam3_building_identifier.utils import discover_images, infer_disaster_type, log, log_section

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def read_rgb(path: Path) -> tuple[np.ndarray, dict]:
    """Read an image as HxWx3 uint8, plus the georeferencing needed to write masks."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.open(path) as src:
            data = src.read()
            georef = {"crs": src.crs, "transform": src.transform} if src.crs else {}
    if data.dtype != np.uint8:
        raise ValueError(f"{path.name}: expected 8-bit imagery, got {data.dtype}")
    if data.shape[0] == 1:
        data = np.repeat(data, 3, axis=0)
    return np.ascontiguousarray(np.moveaxis(data[:3], 0, -1)), georef


def _write_raster(path: Path, arr: np.ndarray, georef: dict) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.open(
            path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1],
            count=1, dtype=arr.dtype, compress="deflate", **georef,
        ) as dst:
            dst.write(arr, 1)


def _write_annotation(path: Path, image: np.ndarray, instances: list[dict]) -> None:
    """Polygon overlay: translucent fill, outline, and score label."""
    base = Image.fromarray(image).convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    for inst in instances:
        pts = [tuple(p) for p in inst["polygon"]]
        draw.polygon(pts, fill=(255, 64, 64, 70), outline=(255, 255, 0, 255))
        x1, y1 = inst["bbox_xyxy"][:2]
        draw.text((x1 + 2, y1 + 1), f"{inst['confidence']:.2f}", fill=(255, 255, 255, 255))
    Image.alpha_composite(base, layer).convert("RGB").save(path)


def _code_version() -> Optional[str]:
    try:
        return subprocess.run(
            ["git", "describe", "--always", "--dirty"], cwd=Path(__file__).parent,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Per-image processing
# ---------------------------------------------------------------------------

def _detect(image: np.ndarray, model: SAM3Model, cfg: PipelineConfig):
    """Run SAM3 on every window and merge. Returns (instances, n_windows, n_raw)."""
    h, w = image.shape[:2]
    windows = tile_windows(w, h, cfg.tile_size, cfg.tile_overlap) if cfg.tile_size else [(0, 0, w, h)]
    raw = []
    for win in windows:
        x, y, ww, wh = win
        for mask, score in model.predict(image[y:y + wh, x:x + ww]):
            inst = instance_from_mask(mask, win, score)
            if inst is not None:
                raw.append(inst)
    merged = merge_instances(raw, cfg.merge_iou) if len(windows) > 1 else raw
    merged = [i for i in merged if i.area >= cfg.min_size]
    return merged, len(windows), len(raw)


def _process_one_image(image_path: Path, model: SAM3Model, cfg: PipelineConfig) -> dict[str, Any]:
    stem = image_path.stem
    pred_path = cfg.predictions_dir / f"{stem}_prediction.json"
    if cfg.skip_existing and pred_path.exists():
        return {"stem": stem, "status": "skipped"}

    record: dict[str, Any] = {"stem": stem, "status": "error"}
    t0 = time.perf_counter()
    try:
        image, georef = read_rgb(image_path)
        h, w = image.shape[:2]

        found, n_windows, n_raw = _detect(image, model, cfg)
        t1 = time.perf_counter()

        labels, scores, label_scores = paint_labels(found, h, w)
        instances = labels_to_instances(
            labels, label_scores,
            epsilon=cfg.polygon_epsilon,
            min_area=cfg.min_polygon_area,
            simplify_tolerance=cfg.simplify_tolerance,
        )
        if instances and cfg.save_masks:
            _write_raster(cfg.masks_dir / f"{stem}.tif", labels, georef)
            _write_raster(cfg.masks_dir / f"{stem}_scores.tif", scores, georef)
        if instances and cfg.save_annotations:
            _write_annotation(cfg.annotations_dir / f"{stem}_ann.png", image, instances)
        t2 = time.perf_counter()

        timing = {
            "inference_sec": round(t1 - t0, 3),
            "postprocess_sec": round(t2 - t1, 3),
            "total_sec": round(t2 - t0, 3),
        }
        doc = {
            "image": {
                "path": str(image_path.resolve()),
                "stem": stem,
                "width": w,
                "height": h,
                "disaster_type": infer_disaster_type(stem),
            },
            "instances": instances,
            "timing": timing,
            "summary": {
                "num_instances": len(instances),
                "num_windows": n_windows,
                "num_raw_detections": n_raw,
                "status": "ok",
            },
        }
        pred_path.write_text(json.dumps(doc, indent=2))
        record.update(status="ok", num_instances=len(instances), **timing)
        log(f"  DONE  {stem}: {len(instances)} buildings ({n_raw} raw in {n_windows} windows) "
            f"in {timing['total_sec']:.1f}s")
    except Exception:
        logger.error("Error processing %s:\n%s", stem, traceback.format_exc())
        record["error"] = traceback.format_exc().splitlines()[-1]
    return record


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_pipeline(cfg: PipelineConfig) -> dict[str, Any]:
    """Run the batch pipeline and write run_summary.json. Returns the totals."""
    if not cfg.input_dir or not cfg.output_dir:
        raise ValueError("input_dir and output_dir are required")
    started_at = datetime.now().isoformat(timespec="seconds")
    wall_start = time.perf_counter()
    log_section("SAM3 Building Identifier")

    images = discover_images(cfg.input_dir, cfg.disaster_type,
                             tuple(cfg.image_extensions), cfg.max_images)
    if not images:
        log("No images found; exiting.")
        return {}

    cfg.make_output_dirs()
    model = SAM3Model(cfg)
    model.load()

    records = []
    for path in tqdm(images, unit="img", ncols=80):
        records.append(_process_one_image(path, model, cfg))

    ok = [r for r in records if r["status"] == "ok"]
    totals = {
        "images_found": len(records),
        "images_processed": len(ok),
        "images_skipped": sum(r["status"] == "skipped" for r in records),
        "images_error": sum(r["status"] == "error" for r in records),
        "total_buildings": sum(r["num_instances"] for r in ok),
        "total_wall_time_sec": round(time.perf_counter() - wall_start, 2),
        "avg_time_per_image_sec": round(sum(r["total_sec"] for r in ok) / len(ok), 2) if ok else 0,
    }
    summary = {
        "started_at": started_at,
        "code_version": _code_version(),
        "config": dataclasses.asdict(cfg),
        "totals": totals,
        "per_image": records,
    }
    cfg.run_summary_path.write_text(json.dumps(summary, indent=2, default=str))

    log_section("Run complete")
    for k, v in totals.items():
        log(f"  {k:24s} {v}")
    return {**totals, "run_summary_path": str(cfg.run_summary_path)}
