"""
Main batch pipeline.

Responsibilities
----------------
* Discover images matching the configured filter.
* Load the SAM3 model once.
* For each image (or batch):
    1. Run inference → save mask TIF + scores TIF
    2. Save annotation PNG
    3. Convert masks → polygon instances
    4. Write enhanced prediction JSON
    5. Record per-image timing and status
* Write a run_summary.json at the end.

The pipeline is deliberately synchronous and single-process.  Processing one
image at a time (batch_size=1) matches the notebooks and avoids GPU OOM.
"""

from __future__ import annotations

import json
import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from sam3_building_identifier.config import PipelineConfig
from sam3_building_identifier.mask_to_polygon import masks_to_instances
from sam3_building_identifier.model import SAM3Model
from sam3_building_identifier.utils import (
    discover_images,
    get_image_size,
    infer_disaster_type,
    log,
    log_section,
    timer,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output JSON builder
# ---------------------------------------------------------------------------

def _build_prediction_json(
    image_path: Path,
    instances: List[Dict[str, Any]],
    timing: Dict[str, float],
    status: str = "ok",
) -> Dict[str, Any]:
    """Build the enhanced prediction JSON document for one image."""
    w, h = get_image_size(image_path)
    return {
        "image": {
            "path": str(image_path.resolve()),
            "stem": image_path.stem,
            "width": w,
            "height": h,
            "disaster_type": infer_disaster_type(image_path.stem),
        },
        "instances": instances,
        "timing": timing,
        "summary": {
            "num_instances": len(instances),
            "status": status,
        },
    }


def _write_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2)


# ---------------------------------------------------------------------------
# Single-image processing step
# ---------------------------------------------------------------------------

def _process_one_image(
    image_path: Path,
    model: SAM3Model,
    cfg: PipelineConfig,
) -> Dict[str, Any]:
    """Process one image end-to-end.  Returns a status/timing record."""
    stem = image_path.stem
    pred_path = cfg.predictions_dir / f"{stem}_prediction.json"

    # --- skip if already done ---
    if cfg.skip_existing and pred_path.exists():
        log(f"  SKIP {stem} (prediction exists)")
        return {"stem": stem, "status": "skipped", "num_instances": None,
                "inference_sec": 0, "postprocess_sec": 0, "total_sec": 0}

    record: Dict[str, Any] = {"stem": stem, "status": "error"}
    overall_start = __import__("time").perf_counter()

    try:
        # -- 1. Inference --
        w, h = get_image_size(image_path)
        log(f"  IMG   {stem} ({w}×{h}px)")
        model.clear_memory()
        with timer() as t_infer:
            if cfg.batch_size == 1:
                model.set_image(image_path)
                n_found = model.generate_masks()
            else:
                model.set_image_batch([image_path])
                counts = model.generate_masks_batch()
                n_found = counts[0] if counts else 0
        record["inference_sec"] = t_infer["elapsed"]

        log(f"  INF   {stem} — {n_found} candidates found in {t_infer['elapsed']:.1f}s")

        if n_found == 0:
            log(f"  WARN  {stem} — no buildings detected by SAM3")
            # Write an empty prediction and skip file-saving steps
            instances: List[Dict] = []
            record["postprocess_sec"] = 0.0
        else:
            # -- 2. Save masks TIF --
            if cfg.save_masks:
                if cfg.batch_size == 1:
                    mask_path, score_path = model.save_masks(stem, cfg.masks_dir)
                else:
                    saved = model.save_masks_batch(stem, cfg.masks_dir)
                    mask_path = saved[0] if saved else cfg.masks_dir / f"{stem}.tif"
                    score_path = cfg.masks_dir / f"{stem}_scores.tif"
            else:
                mask_path = cfg.masks_dir / f"{stem}.tif"
                score_path = cfg.masks_dir / f"{stem}_scores.tif"

            # -- 3. Save annotation PNG --
            if cfg.save_annotations:
                try:
                    if cfg.batch_size == 1:
                        model.save_annotation(stem, cfg.annotations_dir)
                    else:
                        model.save_annotation_batch(stem, cfg.annotations_dir)
                except Exception as exc:
                    logger.warning("Annotation save failed for %s: %s", stem, exc)

            # -- 4. Mask → polygons --
            with timer() as t_post:
                sp = score_path if score_path.exists() else None
                instances = masks_to_instances(
                    mask_path=mask_path,
                    score_path=sp,
                    epsilon=cfg.polygon_epsilon,
                    min_area=cfg.min_polygon_area,
                    simplify_tolerance=cfg.simplify_tolerance,
                )
            record["postprocess_sec"] = t_post["elapsed"]

        # -- 5. Write prediction JSON --
        timing = {
            "inference_sec": record.get("inference_sec", 0),
            "postprocess_sec": record.get("postprocess_sec", 0),
            "total_sec": round(__import__("time").perf_counter() - overall_start, 3),
        }
        prediction = _build_prediction_json(image_path, instances, timing)
        _write_json(prediction, pred_path)

        record.update({
            "status": "ok",
            "num_instances": len(instances),
            "total_sec": timing["total_sec"],
        })
        log(
            f"  DONE  {stem} — {len(instances)} buildings "
            f"in {timing['total_sec']:.1f}s"
        )

    except Exception:
        tb = traceback.format_exc()
        logger.error("Error processing %s:\n%s", stem, tb)
        record["status"] = "error"
        record["error"] = tb.splitlines()[-1]

    return record


# ---------------------------------------------------------------------------
# Run summary
# ---------------------------------------------------------------------------

def _write_run_summary(
    cfg: PipelineConfig,
    image_records: List[Dict[str, Any]],
    total_sec: float,
    started_at: str,
) -> None:
    ok = [r for r in image_records if r["status"] == "ok"]
    skipped = [r for r in image_records if r["status"] == "skipped"]
    errors = [r for r in image_records if r["status"] == "error"]

    total_buildings = sum(r.get("num_instances", 0) or 0 for r in ok)
    avg_time = (
        sum(r.get("total_sec", 0) or 0 for r in ok) / len(ok) if ok else 0
    )

    summary = {
        "started_at": started_at,
        "config": {
            "input_dir": cfg.input_dir,
            "output_dir": cfg.output_dir,
            "disaster_type": cfg.disaster_type,
            "text_prompt": cfg.text_prompt,
            "min_size": cfg.min_size,
            "polygon_epsilon": cfg.polygon_epsilon,
            "min_polygon_area": cfg.min_polygon_area,
        },
        "totals": {
            "images_found": len(image_records),
            "images_processed": len(ok),
            "images_skipped": len(skipped),
            "images_error": len(errors),
            "total_buildings": total_buildings,
            "total_wall_time_sec": round(total_sec, 2),
            "avg_time_per_image_sec": round(avg_time, 2),
        },
        "per_image": image_records,
    }
    _write_json(summary, cfg.run_summary_path)
    log(f"Run summary written to {cfg.run_summary_path}")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_pipeline(cfg: PipelineConfig) -> Dict[str, Any]:
    """Run the full batch pipeline.

    Parameters
    ----------
    cfg:
        PipelineConfig instance with all parameters set.

    Returns
    -------
    The run-summary dict (same structure as run_summary.json).
    """
    started_at = datetime.now().isoformat(timespec="seconds")
    log_section("SAM3 Building Identifier — Batch Pipeline")
    log(f"Started at {started_at}")

    # -- Discover images --
    images = discover_images(
        input_dir=cfg.input_dir,
        disaster_type=cfg.disaster_type,
        extensions=tuple(cfg.image_extensions),
        max_images=cfg.max_images,
    )
    log(f"Found {len(images)} {cfg.disaster_type}-disaster images in {cfg.input_dir}")

    if not images:
        log("No images found — exiting.")
        return {}

    # -- Prepare output dirs --
    cfg.make_output_dirs()

    # -- Load model --
    log_section("Loading SAM3 model")
    model = SAM3Model(cfg)
    model.load()

    # -- Process images --
    log_section(f"Processing {len(images)} images")
    image_records: List[Dict[str, Any]] = []

    with tqdm(total=len(images), unit="img", ncols=80) as pbar:
        for img_path in images:
            record = _process_one_image(img_path, model, cfg)
            image_records.append(record)
            pbar.set_postfix(status=record["status"])
            pbar.update(1)

    # -- Write run summary --
    import time as _time
    total_sec = sum(r.get("total_sec", 0) or 0 for r in image_records)

    log_section("Run complete")
    ok_count = sum(1 for r in image_records if r["status"] == "ok")
    err_count = sum(1 for r in image_records if r["status"] == "error")
    skip_count = sum(1 for r in image_records if r["status"] == "skipped")
    total_buildings = sum(r.get("num_instances", 0) or 0 for r in image_records)

    log(f"  Processed : {ok_count}")
    log(f"  Skipped   : {skip_count}")
    log(f"  Errors    : {err_count}")
    log(f"  Buildings : {total_buildings}")
    log(f"  Total time: {total_sec:.1f}s")

    _write_run_summary(cfg, image_records, total_sec, started_at)

    return {
        "ok": ok_count,
        "skipped": skip_count,
        "errors": err_count,
        "total_buildings": total_buildings,
        "total_sec": total_sec,
        "run_summary_path": str(cfg.run_summary_path),
    }
