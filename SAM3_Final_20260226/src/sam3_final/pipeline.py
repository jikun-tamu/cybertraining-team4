from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rasterio.transform import Affine
from shapely.geometry.base import BaseGeometry

from .export import write_geojson, write_geopackage
from .georef import find_georef
from .infer import Sam3Config, init_sam3, infer_single_image
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
    save_annotations: bool = True
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
                save_scores=True,
                save_ann=cfg.save_annotations,
            )
            if result is None:
                continue

            if not cfg.save_masks:
                # Still polygonize; masks are required, so we keep them for this run
                pass

            # Use georef transform if available; otherwise translate to full-image pixel coords
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

    # Write outputs
    out_geojson = output_dir / "buildings.geojson"
    out_crs = None
    if len(crs_set) == 1:
        out_crs = list(crs_set)[0]
    write_geojson(all_features, out_geojson, crs=out_crs)

    out_gpkg = output_dir / "buildings.gpkg"
    gpkg_ok = write_geopackage(all_features, out_gpkg, crs=out_crs)

    summary["outputs"] = {
        "geojson": str(out_geojson),
        "gpkg": str(out_gpkg) if gpkg_ok else None,
    }
    return summary
