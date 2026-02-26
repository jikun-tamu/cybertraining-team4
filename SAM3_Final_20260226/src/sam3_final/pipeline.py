from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import json
import re

import numpy as np
from rasterio.transform import Affine
from shapely.geometry.base import BaseGeometry

from .export import write_geojson, write_geopackage
from .georef import find_georef
from .infer import Sam3Config, init_sam3, infer_single_image, clear_gpu_cache
from .notebook_outputs import vectorize_mask_to_wkt_json
from .polygonize import PolygonizeConfig, polygonize_mask
from .tiling import generate_tiles
from .utils import ensure_dir, list_images


@dataclass
class PipelineConfig:
    input_path: str
    output_dir: str
    prompt: str = "building"
    min_size: int = 100
    tile_size: int | None = None
    overlap: int = 0
    regularize_method: str = "none"
    epsilon: float = 2.0
    use_geoai: bool = False
    metadata_path: str | None = None
    save_masks: bool = True
    save_scores: bool = True
    save_annotations: bool = True
    run_polygons: bool = True
    clear_cache_every: int = 0
    tile_annotations: bool = False
    full_annotation: bool = True
    output_style: str = "notebook"
    batch_size: int = 1
    max_images: int | None = None
    pattern: str | None = None
    sam3_backend: str = "meta"
    sam3_device: str | None = None
    sam3_checkpoint: str | None = None
    sam3_load_from_hf: bool = True
    hf_token: str | None = None
    exts: tuple[str, ...] = ("png", "jpg", "jpeg", "tif", "tiff")


def _add_props(geom: BaseGeometry, props: dict[str, Any]) -> dict[str, Any]:
    return {
        "geometry": geom,
        "properties": {
            **props,
            "area": float(geom.area),
            "perimeter": float(geom.length),
        },
    }


def _select_images(cfg: PipelineConfig) -> list[Path]:
    p = Path(cfg.input_path)
    if p.is_file():
        images = [p]
    else:
        if cfg.pattern:
            images = sorted(p.rglob(cfg.pattern))
        else:
            images = list_images(cfg.input_path, cfg.exts)
    if cfg.max_images:
        images = images[: cfg.max_images]
    return images


def _write_mask_tif(path: Path, arr: np.ndarray) -> None:
    import rasterio

    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = arr.shape
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=1,
        dtype=arr.dtype,
    ) as dst:
        dst.write(arr, 1)


def _result_to_label_score(result: dict[str, Any], size_hw: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    h, w = size_hw
    label = np.zeros((h, w), dtype=np.uint32)
    score = np.zeros((h, w), dtype=np.float32)

    masks = result.get("masks") or []
    scores = result.get("scores")

    def _mask_from_item(item):
        if isinstance(item, dict):
            if "segmentation" in item:
                return item["segmentation"]
            if "mask" in item:
                return item["mask"]
        return item

    for i, item in enumerate(masks):
        m = _mask_from_item(item)
        if m is None:
            continue
        m = np.asarray(m)
        if m.shape != (h, w):
            continue
        lbl = i + 1
        label[m] = lbl
        sc = None
        if scores is not None and i < len(scores):
            sc = scores[i]
        elif isinstance(item, dict):
            sc = item.get("score") or item.get("predicted_iou") or item.get("stability_score")
        if sc is not None:
            score[m] = float(sc)
    return label, score


def _parse_tile_meta(name: str):
    m = re.search(r"_x(\d+)_y(\d+)_w(\d+)_h(\d+)", name)
    if not m:
        return None
    return tuple(int(x) for x in m.groups())


def _stitch_tiles(tile_mask_paths: Iterable[Path], tile_score_paths: Iterable[Path], size_hw: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    h, w = size_hw
    canvas_mask = np.zeros((h, w), dtype=np.uint32)
    canvas_score = np.zeros((h, w), dtype=np.float32)
    max_label = 0

    score_map = {p.stem.replace("_scores", ""): p for p in tile_score_paths}

    import rasterio

    for mp in tile_mask_paths:
        meta = _parse_tile_meta(mp.stem)
        if meta is None:
            continue
        x, y, tw, th = meta
        with rasterio.open(mp) as src:
            tile_mask = src.read(1)
        sp = score_map.get(mp.stem)
        tile_score = None
        if sp and sp.exists():
            with rasterio.open(sp) as ssrc:
                tile_score = ssrc.read(1)

        tile_mask = tile_mask[:th, :tw]
        if tile_score is not None:
            tile_score = tile_score[:th, :tw]

        # Offset labels to keep uniqueness across tiles
        offset_mask = tile_mask.copy()
        offset_mask[offset_mask > 0] += max_label

        region_mask = canvas_mask[y : y + th, x : x + tw]
        region_score = canvas_score[y : y + th, x : x + tw]

        if tile_score is not None:
            # Replace where score is higher
            replace = tile_score > region_score
            region_mask[replace] = offset_mask[replace]
            region_score[replace] = tile_score[replace]
        else:
            # Replace only empty pixels
            replace = region_mask == 0
            region_mask[replace] = offset_mask[replace]

        max_label = max(max_label, int(offset_mask.max()))
        canvas_mask[y : y + th, x : x + tw] = region_mask
        canvas_score[y : y + th, x : x + tw] = region_score

    return canvas_mask, canvas_score


def _save_full_annotation(img_path: Path, mask: np.ndarray | None, features: list[dict[str, Any]], out_path: Path) -> None:
    from PIL import Image
    from .viz import colorize_instance_mask, draw_polygons

    img = Image.open(img_path).convert("RGB")
    overlay = img.copy()
    if mask is not None:
        mask_vis = colorize_instance_mask(mask)
        overlay = Image.blend(overlay, mask_vis, alpha=0.35)
    if features:
        overlay = draw_polygons(overlay, [f["geometry"] for f in features], color="yellow", width=2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(out_path)


def _run_notebook_style(cfg: PipelineConfig) -> dict[str, Any]:
    output_dir = Path(cfg.output_dir)
    ensure_dir(output_dir)
    images = _select_images(cfg)

    import torch
    print(f"torch_version: {torch.__version__}")
    print(f"cuda_available: {torch.cuda.is_available()}")
    print(f"cuda_device_count: {torch.cuda.device_count()}")
    if cfg.sam3_device and "cuda" in cfg.sam3_device and torch.cuda.is_available():
        try:
            idx = int(cfg.sam3_device.split(":")[1])
            print(f"selected_device: cuda:{idx} ({torch.cuda.get_device_name(idx)})")
        except Exception:
            print(f"selected_device: {cfg.sam3_device}")

    sam3 = init_sam3(
        Sam3Config(
            backend=cfg.sam3_backend,
            device=cfg.sam3_device,
            checkpoint_path=cfg.sam3_checkpoint,
            load_from_hf=cfg.sam3_load_from_hf,
            hf_token=cfg.hf_token,
        )
    )

    timing_images: list[dict[str, Any]] = []
    timing_tiles: list[dict[str, Any]] = []

    summary = {
        "images": len(images),
        "tiles": 0,
        "instances": 0,
        "skipped_images": 0,
    }

    labels_dir = output_dir / "labels"
    masks_dir = output_dir / "masks"
    ann_dir = output_dir / "annotations"

    tile_counter = 0

    if cfg.tile_size is None:
        for start in range(0, len(images), max(1, cfg.batch_size)):
            batch = images[start : start + max(1, cfg.batch_size)]
            batch_paths = [str(p) for p in batch]
            sam3.set_image_batch(batch_paths)
            batch_results = None
            try:
                sam3.generate_masks_batch(cfg.prompt, min_size=cfg.min_size)
                batch_results = sam3.batch_results
            except RuntimeError:
                batch_results = None

            for i, img_path in enumerate(batch):
                img_path = Path(img_path)
                georef = find_georef(img_path, metadata_path=cfg.metadata_path)
                import time
                t0 = time.perf_counter()

                if batch_results is not None:
                    result = batch_results[i]
                else:
                    # Fallback to per-image inference if batch fails (e.g., empty mask tensor bug)
                    sam3.set_image(str(img_path))
                    sam3.generate_masks(prompt=cfg.prompt, min_size=cfg.min_size)
                    result = {"masks": getattr(sam3, "masks", []), "scores": getattr(sam3, "scores", None)}
                from PIL import Image
                w, h = Image.open(img_path).size
                label_mask, score_mask = _result_to_label_score(result, (h, w))

                if cfg.save_masks:
                    _write_mask_tif(masks_dir / f"{img_path.stem}.tif", label_mask)
                    if cfg.save_scores:
                        _write_mask_tif(masks_dir / f"{img_path.stem}_scores.tif", score_mask)

                features = []
                if cfg.run_polygons:
                    try:
                        if cfg.save_scores:
                            vectorize_mask_to_wkt_json(
                                masks_dir / f"{img_path.stem}.tif",
                                masks_dir / f"{img_path.stem}_scores.tif",
                                labels_dir,
                                epsilon=cfg.epsilon,
                            )
                    except Exception:
                        pass

                if cfg.save_annotations and cfg.full_annotation:
                    _save_full_annotation(img_path, label_mask, features, ann_dir / f"{img_path.stem}_ann.png")

                summary["instances"] += int(label_mask.max())
                t1 = time.perf_counter()
                timing_images.append(
                    {
                        "image_id": img_path.stem,
                        "num_tiles": 1,
                        "num_instances": int(label_mask.max()),
                        "t_total_s": t1 - t0,
                    }
                )
        # no tiling; done
        # Write timing outputs
        import pandas as pd
        timing_csv = output_dir / "timing_per_image.csv"
        pd.DataFrame(timing_images).to_csv(timing_csv, index=False)
        timing_json = output_dir / "run_timing_summary.json"
        timing_json.write_text(json.dumps({"summary": summary, "timing_per_image": timing_images, "timing_per_tile": timing_tiles}, indent=2))
        summary["outputs"] = {"masks": str(masks_dir), "annotations": str(ann_dir), "labels": str(labels_dir)}
        return summary

    for img_path in images:
        img_path = Path(img_path)
        georef = find_georef(img_path, metadata_path=cfg.metadata_path)

        import time
        t0 = time.perf_counter()

        # Tiled processing
        tiles = generate_tiles(
            img_path,
            out_dir=output_dir,
            tile_size=cfg.tile_size,
            overlap=cfg.overlap,
            transform=georef.transform,
        )
        summary["tiles"] += len(tiles)

        tile_masks_tmp = []
        tile_scores_tmp = []

        for tile in tiles:
            tile_counter += 1
            if cfg.clear_cache_every and tile_counter % cfg.clear_cache_every == 0:
                clear_gpu_cache()

            result = infer_single_image(
                sam3,
                tile.tile_path,
                output_dir=output_dir / ".tmp_tiles",
                prompt=cfg.prompt,
                min_size=cfg.min_size,
                save_masks=True,
                save_scores=cfg.save_scores,
                save_ann=cfg.save_annotations and cfg.tile_annotations,
            )
            if result is None:
                continue

            tile_masks_tmp.append(result.mask_path)
            if result.score_path:
                tile_scores_tmp.append(result.score_path)

            timing_tiles.append(
                {
                    "image_id": tile.image_id,
                    "tile_id": tile.tile_id,
                    "t_tile_io_s": tile.io_time_s,
                    "t_infer_s": result.t_infer_s,
                    "t_save_s": result.t_save_s,
                    "t_poly_s": 0.0,
                }
            )

        from PIL import Image
        w, h = Image.open(img_path).size
        label_mask, score_mask = _stitch_tiles(tile_masks_tmp, tile_scores_tmp, (h, w))

        if cfg.save_masks:
            _write_mask_tif(masks_dir / f"{img_path.stem}.tif", label_mask)
            if cfg.save_scores:
                _write_mask_tif(masks_dir / f"{img_path.stem}_scores.tif", score_mask)

        if cfg.save_annotations and cfg.full_annotation:
            _save_full_annotation(img_path, label_mask, [], ann_dir / f"{img_path.stem}_ann.png")

        if cfg.run_polygons:
            try:
                if cfg.save_scores:
                    vectorize_mask_to_wkt_json(
                        masks_dir / f"{img_path.stem}.tif",
                        masks_dir / f"{img_path.stem}_scores.tif",
                        labels_dir,
                        epsilon=cfg.epsilon,
                    )
            except Exception:
                pass

        summary["instances"] += int(label_mask.max())
        t1 = time.perf_counter()
        timing_images.append({"image_id": img_path.stem, "num_tiles": len(tiles), "num_instances": int(label_mask.max()), "t_total_s": t1 - t0})

    # Write timing outputs
    import pandas as pd
    timing_csv = output_dir / "timing_per_image.csv"
    pd.DataFrame(timing_images).to_csv(timing_csv, index=False)
    timing_json = output_dir / "run_timing_summary.json"
    timing_json.write_text(json.dumps({"summary": summary, "timing_per_image": timing_images, "timing_per_tile": timing_tiles}, indent=2))

    summary["outputs"] = {
        "masks": str(masks_dir),
        "annotations": str(ann_dir),
        "labels": str(labels_dir),
    }
    return summary


def _run_tiled_style(cfg: PipelineConfig) -> dict[str, Any]:
    output_dir = Path(cfg.output_dir)
    ensure_dir(output_dir)
    images = _select_images(cfg)

    sam3 = init_sam3(
        Sam3Config(
            backend=cfg.sam3_backend,
            device=cfg.sam3_device,
            checkpoint_path=cfg.sam3_checkpoint,
            load_from_hf=cfg.sam3_load_from_hf,
            hf_token=cfg.hf_token,
        )
    )

    poly_cfg = PolygonizeConfig(
        regularize_method=cfg.regularize_method,
        epsilon=cfg.epsilon,
        use_geoai=cfg.use_geoai,
    )

    all_features: list[dict[str, Any]] = []
    crs_set = set()
    summary = {
        "images": len(images),
        "tiles": 0,
        "instances": 0,
        "skipped_images": 0,
    }

    for img_path in images:
        img_path = Path(img_path)
        georef = find_georef(img_path, metadata_path=cfg.metadata_path)
        if georef.crs is not None:
            crs_set.add(str(georef.crs))

        tiles = generate_tiles(
            img_path,
            out_dir=output_dir,
            tile_size=cfg.tile_size,
            overlap=cfg.overlap,
            transform=georef.transform,
        )
        summary["tiles"] += len(tiles)

        for tile in tiles:
            result = infer_single_image(
                sam3,
                tile.tile_path,
                output_dir=output_dir,
                prompt=cfg.prompt,
                min_size=cfg.min_size,
                save_masks=cfg.save_masks,
                save_scores=cfg.save_scores,
                save_ann=cfg.save_annotations,
            )
            if result is None:
                continue

            if not cfg.run_polygons:
                continue

            tile_transform = tile.transform
            if tile_transform is None:
                tile_transform = Affine.translation(tile.x, tile.y)

            features = polygonize_mask(
                result.mask_path,
                result.score_path,
                transform=tile_transform,
                cfg=poly_cfg,
            )

            for f in features:
                props = f["properties"]
                props.update(
                    {
                        "image_id": tile.image_id,
                        "tile_id": tile.tile_id,
                        "width": georef.width,
                        "height": georef.height,
                        "prompt": cfg.prompt,
                        "min_size": cfg.min_size,
                        "regularize": cfg.regularize_method,
                        "epsilon": cfg.epsilon,
                        "georef_source": georef.source,
                    }
                )
                if georef.transform is None:
                    props.update(
                        {
                            "pixel_coord_system": "image",
                            "transform_source": "none",
                        }
                    )
                all_features.append(_add_props(f["geometry"], props))

            summary["instances"] += len(features)

    out_geojson = output_dir / "buildings.geojson"
    out_gpkg = output_dir / "buildings.gpkg"
    gpkg_ok = False
    out_crs = None
    if len(crs_set) == 1:
        out_crs = list(crs_set)[0]
    if cfg.run_polygons:
        write_geojson(all_features, out_geojson, crs=out_crs)
        gpkg_ok = write_geopackage(all_features, out_gpkg, crs=out_crs)

    summary["outputs"] = {
        "geojson": str(out_geojson) if cfg.run_polygons else None,
        "gpkg": str(out_gpkg) if gpkg_ok else None,
    }
    return summary


def run_pipeline(cfg: PipelineConfig) -> dict[str, Any]:
    if cfg.output_style == "tiled":
        return _run_tiled_style(cfg)
    return _run_notebook_style(cfg)
