from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rasterio.transform import Affine
from shapely.geometry.base import BaseGeometry

from .export import write_geojson, write_geopackage
from .georef import find_georef
from .infer import Sam3Config, init_sam3, infer_single_image, clear_gpu_cache
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


def run_pipeline(cfg: PipelineConfig) -> dict[str, Any]:
    output_dir = Path(cfg.output_dir)
    ensure_dir(output_dir)
    images = list_images(cfg.input_path, cfg.exts)

    import torch
    import json
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

    poly_cfg = PolygonizeConfig(
        regularize_method=cfg.regularize_method,
        epsilon=cfg.epsilon,
        use_geoai=cfg.use_geoai,
    )

    all_features: list[dict[str, Any]] = []
    crs_set = set()
    timing_tiles: list[dict[str, Any]] = []
    timing_images: list[dict[str, Any]] = []
    summary = {
        "images": len(images),
        "tiles": 0,
        "instances": 0,
        "skipped_images": 0,
    }

    tile_counter = 0
    for img_path in images:
        georef = find_georef(img_path, metadata_path=cfg.metadata_path)
        if georef.crs is not None:
            crs_set.add(str(georef.crs))

        import time
        t_tile_gen0 = time.perf_counter()
        tiles = generate_tiles(
            img_path,
            out_dir=output_dir,
            tile_size=cfg.tile_size,
            overlap=cfg.overlap,
            transform=georef.transform,
        )
        t_tile_gen1 = time.perf_counter()
        summary["tiles"] += len(tiles)

        image_t_infer = 0.0
        image_t_save = 0.0
        image_t_poly = 0.0
        image_t_tile_io = sum(t.io_time_s for t in tiles)
        image_instances = 0

        for tile in tiles:
            tile_counter += 1
            if cfg.clear_cache_every and tile_counter % cfg.clear_cache_every == 0:
                clear_gpu_cache()

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

            image_t_infer += result.t_infer_s
            image_t_save += result.t_save_s

            if cfg.run_polygons:
                if result.mask_path is None:
                    raise RuntimeError("Polygonization requested but masks were not saved.")
            else:
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
                continue

            # Use georef transform if available; otherwise translate to full-image pixel coords
            tile_transform = tile.transform
            if tile_transform is None:
                tile_transform = Affine.translation(tile.x, tile.y)

            t_poly0 = time.perf_counter()
            features = polygonize_mask(
                result.mask_path,
                result.score_path,
                transform=tile_transform,
                cfg=poly_cfg,
            )
            t_poly1 = time.perf_counter()
            t_poly = t_poly1 - t_poly0
            image_t_poly += t_poly

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
            image_instances += len(features)

            timing_tiles.append(
                {
                    "image_id": tile.image_id,
                    "tile_id": tile.tile_id,
                    "t_tile_io_s": tile.io_time_s,
                    "t_infer_s": result.t_infer_s,
                    "t_save_s": result.t_save_s,
                    "t_poly_s": t_poly,
                }
            )

        timing_images.append(
            {
                "image_id": Path(img_path).stem,
                "num_tiles": len(tiles),
                "num_instances": image_instances,
                "t_tile_io_s": image_t_tile_io,
                "t_infer_s": image_t_infer,
                "t_save_s": image_t_save,
                "t_poly_s": image_t_poly,
                "t_tile_gen_s": t_tile_gen1 - t_tile_gen0,
            }
        )

    # Write outputs
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

    import pandas as pd
    timing_csv = output_dir / "timing_per_image.csv"
    pd.DataFrame(timing_images).to_csv(timing_csv, index=False)

    timing_json = output_dir / "run_timing_summary.json"
    timing_json.write_text(
        json.dumps(
            {
                "summary": summary,
                "timing_per_image": timing_images,
                "timing_per_tile": timing_tiles,
            },
            indent=2,
        )
    )
    return summary
