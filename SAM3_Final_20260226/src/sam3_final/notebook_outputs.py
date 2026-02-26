from __future__ import annotations

import json
import uuid
from pathlib import Path

import numpy as np
from PIL import Image
import rasterio
from shapely.geometry import Point


def vectorize_mask_to_wkt_json(mask_path: Path, score_path: Path, output_dir: Path, epsilon: float = 2.0) -> Path | None:
    import geoai

    output_dir.mkdir(parents=True, exist_ok=True)
    temp_geojson = output_dir / f"temp_{mask_path.stem}.geojson"

    gdf = geoai.orthogonalize(str(mask_path), str(temp_geojson), epsilon=epsilon)
    if gdf is None or len(gdf) == 0:
        if temp_geojson.exists():
            temp_geojson.unlink()
        return None

    with rasterio.open(mask_path) as msrc, rasterio.open(score_path) as ssrc:
        mask_arr = msrc.read(1)
        score_arr = ssrc.read(1)

    features = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        mask_id = int(mask_arr[msrc.index(*geom.centroid.coords[0])])
        if mask_id == 0:
            mask_id = int(np.bincount(mask_arr[msrc.window(*geom.bounds).round().astype(int)].ravel()).argmax())

        prob = float(score_arr[mask_arr == mask_id].mean()) if (mask_arr == mask_id).any() else None

        features.append(
            {
                "properties": {
                    "feature_type": "building",
                    "uid": str(uuid.uuid4()),
                    "label": mask_id,
                    "prob": prob,
                },
                "wkt": geom.wkt,
            }
        )

    mask_img = np.array(Image.open(mask_path))
    output_json = {
        "features": {"xy": features, "lng_lat": []},
        "metadata": {
            "original_width": mask_img.shape[1],
            "original_height": mask_img.shape[0],
            "width": mask_img.shape[1],
            "height": mask_img.shape[0],
            "img_name": mask_path.stem + ".png",
        },
    }

    out_path = output_dir / f"{mask_path.stem}_prediction.json"
    with open(out_path, "w") as f:
        json.dump(output_json, f, indent=2)

    if temp_geojson.exists():
        temp_geojson.unlink()
    return out_path
