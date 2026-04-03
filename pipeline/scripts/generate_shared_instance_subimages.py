#!/usr/bin/env python3
"""Generate shared per-instance subimages from Stage-1 SAM3 outputs.

This runs crop/mask generation earlier in the pipeline so Stage-2a and Stage-2b
can consume the same standardized instance artifacts.

Protocol compatibility:
- Center crop on polygon centroid
- Default crop size = 256
- Polygon anchor = Stage-1 pre-disaster polygon
- Generates mask_M and mask_R (ring) using same logic as Stage-2b preprocessing
"""

import argparse
import concurrent.futures
import csv
import json
import re
import time
import uuid
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


WKT_FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def parse_args():
    p = argparse.ArgumentParser(description="Generate shared instance subimages from Stage-1 labels.")
    p.add_argument("--stage1_labels_dir", required=True, type=Path, help="Directory with *_prediction.json files.")
    p.add_argument("--pre_images_dir", required=True, type=Path, help="Directory containing pre-disaster images.")
    p.add_argument("--post_images_dir", required=True, type=Path, help="Directory containing post-disaster images.")
    p.add_argument("--out_root", required=True, type=Path, help="Output root directory.")
    p.add_argument("--crop_size", type=int, default=256, help="Square crop size in pixels.")
    p.add_argument("--ring_radius_px", type=int, default=48, help="Ring radius in pixels.")
    p.add_argument("--pre_token", type=str, default="_pre_disaster", help="Token in pre image name.")
    p.add_argument("--post_token", type=str, default="_post_disaster", help="Replacement token for post image name.")
    p.add_argument("--hazard_type", type=str, default="unknown", help="Optional hazard tag.")
    p.add_argument("--event_id", type=str, default="", help="Optional event id.")
    p.add_argument("--strict_images", action="store_true", help="Skip rows when pre/post image files are missing.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing artifacts.")
    p.add_argument(
        "--dilate_backend",
        choices=["auto", "scipy", "opencv", "python"],
        default="auto",
        help="Backend for ring dilation.",
    )
    p.add_argument("--png_compress_level", type=int, default=1, help="PNG compression [0..9].")
    p.add_argument("--num_workers", type=int, default=1, help="Parallel worker count.")
    p.add_argument("--chunk_size", type=int, default=300, help="Rows per chunk in parallel mode.")
    p.add_argument("--log_every", type=int, default=500, help="Progress print frequency.")
    return p.parse_args()


def parse_wkt_polygon_xy(wkt):
    if not wkt or not isinstance(wkt, str):
        return None
    wkt = wkt.strip()
    upper = wkt.upper()
    if upper.startswith("POLYGON"):
        rings = _extract_wkt_rings(wkt, "POLYGON")
        return _largest_valid_ring(rings)
    if upper.startswith("MULTIPOLYGON"):
        rings = _extract_wkt_rings(wkt, "MULTIPOLYGON")
        return _largest_valid_ring(rings)
    return None


def _extract_wkt_rings(wkt, geom_type):
    text = wkt.strip()
    body = text[len(geom_type) :].strip()
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
                groups.append(body[start : idx + 1])
                start = None

    if geom_type == "POLYGON":
        if not groups:
            return []
        return [_parse_ring_text(groups[0][1:-1])]

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
                    sub_groups.append(inner[start : idx + 1])
                    start = None
        if sub_groups:
            rings.append(_parse_ring_text(sub_groups[0][1:-1]))
    return rings


def _parse_ring_text(text):
    pts = []
    for chunk in text.split(","):
        nums = [float(x) for x in WKT_FLOAT_RE.findall(chunk)]
        if len(nums) >= 2:
            pts.append((nums[0], nums[1]))
    if len(pts) >= 2 and pts[0] == pts[-1]:
        pts = pts[:-1]
    return np.array(pts, dtype=np.float32) if pts else None


def _polygon_area_abs(pts):
    if pts is None or len(pts) < 3:
        return 0.0
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _largest_valid_ring(rings):
    best = None
    best_area = -1.0
    for ring in rings:
        if ring is None or len(ring) < 3:
            continue
        area = _polygon_area_abs(ring)
        if area > best_area:
            best = ring
            best_area = area
    return best


def polygon_xy_to_wkt(pts):
    if pts is None or len(pts) < 3:
        return ""
    coords = [tuple(map(float, p)) for p in pts]
    if coords[0] != coords[-1]:
        coords.append(coords[0])
    pts_text = ", ".join(f"{x} {y}" for x, y in coords)
    return f"POLYGON (({pts_text}))"


def polygon_centroid(pts):
    x = pts[:, 0]
    y = pts[:, 1]
    x1 = x[:-1]
    y1 = y[:-1]
    x2 = x[1:]
    y2 = y[1:]
    cross = x1 * y2 - x2 * y1
    area2 = cross.sum()
    if abs(area2) < 1e-6:
        return float(x.mean()), float(y.mean())
    cx = ((x1 + x2) * cross).sum() / (3.0 * area2)
    cy = ((y1 + y2) * cross).sum() / (3.0 * area2)
    return float(cx), float(cy)


def clamp_crop_window(cx, cy, crop_size, w, h):
    half = crop_size // 2
    x0 = int(round(cx)) - half
    y0 = int(round(cy)) - half
    x0 = max(0, min(x0, max(0, w - crop_size)))
    y0 = max(0, min(y0, max(0, h - crop_size)))
    return x0, y0, x0 + crop_size, y0 + crop_size


def local_polygon(pts_global, x0, y0):
    pts = pts_global.copy()
    pts[:, 0] -= float(x0)
    pts[:, 1] -= float(y0)
    return pts


def draw_mask_polygon(size, pts_local):
    img = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(img)
    xy = [tuple(map(float, p)) for p in pts_local]
    if len(xy) >= 3:
        draw.polygon(xy, fill=255)
    return np.array(img, dtype=np.uint8)


def _disk_kernel(radius):
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return ((xx * xx + yy * yy) <= (radius * radius)).astype(np.uint8)


def binary_dilate_python(mask01, radius):
    if radius <= 0:
        return mask01.astype(np.uint8)
    h, w = mask01.shape
    out = np.zeros_like(mask01, dtype=np.uint8)
    ys, xs = np.where(mask01 > 0)
    if ys.size == 0:
        return out
    r2 = radius * radius
    offsets = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy <= r2:
                offsets.append((dy, dx))
    for y, x in zip(ys, xs):
        for dy, dx in offsets:
            yy = y + dy
            xx = x + dx
            if 0 <= yy < h and 0 <= xx < w:
                out[yy, xx] = 1
    return out


def resolve_dilate_backend(choice):
    choice = choice.lower()
    if choice in ("auto", "scipy"):
        try:
            from scipy import ndimage as ndi  # type: ignore

            def dilate_scipy(mask01, radius):
                if radius <= 0:
                    return mask01.astype(np.uint8)
                structure = _disk_kernel(radius).astype(bool)
                out = ndi.binary_dilation(mask01.astype(bool), structure=structure)
                return out.astype(np.uint8)

            return "scipy", dilate_scipy
        except Exception:
            if choice == "scipy":
                raise RuntimeError("Requested scipy backend, but scipy is unavailable.")

    if choice in ("auto", "opencv"):
        try:
            import cv2  # type: ignore

            def dilate_opencv(mask01, radius):
                if radius <= 0:
                    return mask01.astype(np.uint8)
                kernel = _disk_kernel(radius)
                out = cv2.dilate(mask01.astype(np.uint8), kernel, iterations=1)
                return (out > 0).astype(np.uint8)

            return "opencv", dilate_opencv
        except Exception:
            if choice == "opencv":
                raise RuntimeError("Requested opencv backend, but cv2 is unavailable.")

    return "python", binary_dilate_python


def derive_tile_id(pre_image_name):
    stem = Path(pre_image_name).stem
    if stem.endswith("_pre_disaster"):
        return stem[: -len("_pre_disaster")]
    return stem


def derive_post_name(pre_name, pre_token, post_token):
    if pre_token and pre_token in pre_name:
        return pre_name.replace(pre_token, post_token)
    return pre_name


def _is_stage1_package_format(doc):
    """Detect stage1/sam3_building_identifier output format (has 'instances' key)."""
    return "instances" in doc and isinstance(doc["instances"], list)


def _polygon_coords_to_wkt(coords):
    """Convert [[x,y], ...] polygon to WKT string."""
    if not coords or len(coords) < 3:
        return ""
    pts = " ".join(f"{x} {y}" for x, y in coords)
    # Close ring if not already closed
    if coords[0] != coords[-1]:
        pts += f" {coords[0][0]} {coords[0][1]}"
    return f"POLYGON (({pts}))"


def _rows_from_stage1_package_json(args, label_files):
    """Parse stage1/sam3_building_identifier output format."""
    rows = []
    skipped = {
        "missing_img_name": 0,
        "missing_wkt": 0,
        "bad_wkt": 0,
        "missing_image_files": 0,
    }
    for i, path in enumerate(label_files, start=1):
        doc = json.loads(path.read_text(encoding="utf-8"))
        img_info = doc.get("image", {}) or {}
        img_name = img_info.get("stem", "")
        if not img_name:
            skipped["missing_img_name"] += 1
            continue

        # Reconstruct image filename — look for the actual file
        pre_path = None
        for ext in ("png", "tif", "tiff", "jpg", "jpeg"):
            candidate = args.pre_images_dir / f"{img_name}.{ext}"
            if candidate.exists():
                pre_path = candidate
                break
        if pre_path is None:
            pre_path = args.pre_images_dir / f"{img_name}.png"  # fallback

        post_name = derive_post_name(pre_path.name, args.pre_token, args.post_token)
        post_path = args.post_images_dir / post_name
        if args.strict_images and (not pre_path.exists() or not post_path.exists()):
            skipped["missing_image_files"] += 1
            continue

        width = img_info.get("width", "")
        height = img_info.get("height", "")
        tile_id = derive_tile_id(pre_path.name)

        instances = doc.get("instances", []) or []
        for inst in instances:
            polygon_coords = inst.get("polygon", [])
            wkt = _polygon_coords_to_wkt(polygon_coords)
            if not wkt:
                skipped["missing_wkt"] += 1
                continue
            poly = parse_wkt_polygon_xy(wkt)
            if poly is None:
                skipped["bad_wkt"] += 1
                continue
            wkt = polygon_xy_to_wkt(poly)

            uid = inst.get("uid") or f"sam3_{uuid.uuid4()}"
            rows.append(
                {
                    "event_id": args.event_id,
                    "hazard_type": args.hazard_type,
                    "tile_id": tile_id,
                    "width": width,
                    "height": height,
                    "pre_image": str(pre_path),
                    "post_image": str(post_path),
                    "pre_json": "",
                    "post_json": str(path),
                    "bldg_uid": str(uid),
                    "damage_subtype": "",
                    "damage_class": "",
                    "polygon_wkt_xy_pre": wkt,
                    "polygon_wkt_xy_post": wkt,
                    "sam3_confidence": inst.get("confidence", ""),
                }
            )

        if args.log_every > 0 and i % args.log_every == 0:
            print(
                "index_progress",
                f"{i}/{len(label_files)}",
                f"({100.0 * i / len(label_files):.1f}%)",
                "| rows",
                len(rows),
            )
    return rows, skipped, len(label_files)


def _rows_from_prediction_json(args, label_files):
    rows = []
    skipped = {
        "missing_img_name": 0,
        "missing_wkt": 0,
        "bad_wkt": 0,
        "missing_image_files": 0,
    }
    # Auto-detect format from first file
    if label_files:
        first_doc = json.loads(label_files[0].read_text(encoding="utf-8"))
        if _is_stage1_package_format(first_doc):
            print("Detected stage1/sam3_building_identifier output format.")
            return _rows_from_stage1_package_json(args, label_files)

    for i, path in enumerate(label_files, start=1):
        doc = json.loads(path.read_text(encoding="utf-8"))
        meta = doc.get("metadata", {}) or {}
        img_name = meta.get("img_name", "")
        if not img_name:
            skipped["missing_img_name"] += 1
            continue

        pre_path = args.pre_images_dir / img_name
        post_name = derive_post_name(img_name, args.pre_token, args.post_token)
        post_path = args.post_images_dir / post_name
        if args.strict_images and (not pre_path.exists() or not post_path.exists()):
            skipped["missing_image_files"] += 1
            continue

        width = meta.get("width", meta.get("original_width", ""))
        height = meta.get("height", meta.get("original_height", ""))
        tile_id = derive_tile_id(img_name)

        xy = (doc.get("features", {}) or {}).get("xy", []) or []
        for feat in xy:
            props = feat.get("properties", {}) or {}
            wkt = feat.get("wkt", "") or ""
            if not wkt:
                skipped["missing_wkt"] += 1
                continue
            poly = parse_wkt_polygon_xy(wkt)
            if poly is None:
                skipped["bad_wkt"] += 1
                continue
            wkt = polygon_xy_to_wkt(poly)

            uid = props.get("uid") or f"sam3_{uuid.uuid4()}"
            rows.append(
                {
                    "event_id": args.event_id,
                    "hazard_type": args.hazard_type,
                    "tile_id": tile_id,
                    "width": width,
                    "height": height,
                    "pre_image": str(pre_path),
                    "post_image": str(post_path),
                    "pre_json": "",
                    "post_json": str(path),
                    "bldg_uid": str(uid),
                    "damage_subtype": "",
                    "damage_class": "",
                    "polygon_wkt_xy_pre": wkt,
                    "polygon_wkt_xy_post": wkt,
                    "sam3_confidence": props.get("prob", ""),
                }
            )

        if args.log_every > 0 and i % args.log_every == 0:
            print(
                "index_progress",
                f"{i}/{len(label_files)}",
                f"({100.0 * i / len(label_files):.1f}%)",
                "| rows",
                len(rows),
            )
    return rows, skipped, len(label_files)


def _rows_from_mask_tifs(args, mask_files):
    import rasterio
    from rasterio.features import shapes
    from shapely.geometry import shape
    from shapely.ops import unary_union

    rows = []
    skipped = {
        "missing_image_files": 0,
        "empty_mask_files": 0,
    }
    for i, mask_path in enumerate(mask_files, start=1):
        stem = mask_path.stem
        img_name = f"{stem}.png"
        pre_path = args.pre_images_dir / img_name
        post_name = derive_post_name(img_name, args.pre_token, args.post_token)
        post_path = args.post_images_dir / post_name
        if args.strict_images and (not pre_path.exists() or not post_path.exists()):
            skipped["missing_image_files"] += 1
            continue

        with rasterio.open(mask_path) as src:
            arr = src.read(1)
        score_path = mask_path.parent / f"{stem}_scores.tif"
        score_arr = None
        if score_path.exists():
            with rasterio.open(score_path) as ssrc:
                score_arr = ssrc.read(1)

        labels = [int(x) for x in np.unique(arr) if int(x) > 0]
        if not labels:
            skipped["empty_mask_files"] += 1
            continue

        tile_id = derive_tile_id(img_name)
        h, w = arr.shape
        for lbl in labels:
            bin_mask = (arr == lbl).astype(np.uint8)
            geoms = [shape(g) for g, v in shapes(bin_mask, mask=(bin_mask > 0)) if int(v) == 1]
            if not geoms:
                continue
            geom = unary_union(geoms)
            wkt = geom.wkt
            conf = ""
            if score_arr is not None:
                pix = score_arr[arr == lbl]
                if pix.size > 0:
                    conf = float(np.mean(pix))
            rows.append(
                {
                    "event_id": args.event_id,
                    "hazard_type": args.hazard_type,
                    "tile_id": tile_id,
                    "width": w,
                    "height": h,
                    "pre_image": str(pre_path),
                    "post_image": str(post_path),
                    "pre_json": "",
                    "post_json": "",
                    "bldg_uid": f"{stem}_lbl{lbl}",
                    "damage_subtype": "",
                    "damage_class": "",
                    "polygon_wkt_xy_pre": wkt,
                    "polygon_wkt_xy_post": wkt,
                    "sam3_confidence": conf,
                }
            )

        if args.log_every > 0 and i % args.log_every == 0:
            print(
                "index_progress_masks",
                f"{i}/{len(mask_files)}",
                f"({100.0 * i / len(mask_files):.1f}%)",
                "| rows",
                len(rows),
            )
    return rows, skipped, len(mask_files)


def collect_rows(args):
    label_files = sorted(args.stage1_labels_dir.glob("*_prediction.json"))
    if label_files:
        print(f"Found Stage-1 label json files: {len(label_files)}")
        return _rows_from_prediction_json(args, label_files)

    masks_dir = args.stage1_labels_dir.parent / "masks"
    mask_files = sorted([p for p in masks_dir.glob("*.tif") if not p.name.endswith("_scores.tif")])
    if mask_files:
        print(f"No *_prediction.json in {args.stage1_labels_dir}; falling back to mask polygonization from {masks_dir}")
        return _rows_from_mask_tifs(args, mask_files)

    raise RuntimeError(
        f"No *_prediction.json files in {args.stage1_labels_dir} and no mask tif files in {masks_dir}"
    )


def chunk_rows(rows, chunk_size):
    for i in range(0, len(rows), chunk_size):
        yield rows[i : i + chunk_size]


def process_chunk(task):
    rows = task["rows"]
    out_cols = task["out_cols"]
    cfg = task["cfg"]
    out_pre = Path(task["out_pre"])
    out_post = Path(task["out_post"])
    out_m = Path(task["out_m"])
    out_r = Path(task["out_r"])
    tmp_csv_path = Path(task["tmp_csv"])
    _, dilate_fn = resolve_dilate_backend(cfg["dilate_backend"])

    stats = {
        "rows_in": len(rows),
        "written_rows": 0,
        "skipped_missing_images": 0,
        "skipped_open_error": 0,
        "skipped_existing": 0,
    }

    cached_pre_path = None
    cached_post_path = None
    cached_pre_img = None
    cached_post_img = None

    with tmp_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_cols)
        writer.writeheader()

        for row in rows:
            stem = f"{row['tile_id']}__{row['bldg_uid']}".replace("/", "_")
            pre_out = out_pre / f"{stem}.png"
            post_out = out_post / f"{stem}.png"
            m_out = out_m / f"{stem}.png"
            r_out = out_r / f"{stem}.png"

            if (not cfg["overwrite"]) and pre_out.exists() and post_out.exists() and m_out.exists() and r_out.exists():
                out_row = dict(row)
                out_row.update(
                    {
                        "pre_crop": str(pre_out),
                        "post_crop": str(post_out),
                        "mask_M": str(m_out),
                        "mask_R": str(r_out),
                        "crop_x0": "",
                        "crop_y0": "",
                        "crop_size": cfg["crop_size"],
                        "cx": "",
                        "cy": "",
                        "polygon_source_used": "pre_then_post",
                        "m_area_px": "",
                        "r_area_px": "",
                    }
                )
                writer.writerow(out_row)
                stats["written_rows"] += 1
                stats["skipped_existing"] += 1
                continue

            pre_path = Path(row["pre_image"])
            post_path = Path(row["post_image"])
            if not pre_path.exists() or not post_path.exists():
                stats["skipped_missing_images"] += 1
                continue

            poly = parse_wkt_polygon_xy(row.get("polygon_wkt_xy_pre", ""))
            if poly is None:
                continue

            try:
                if cached_pre_path != pre_path:
                    cached_pre_img = Image.open(pre_path).convert("RGB")
                    cached_pre_path = pre_path
                if cached_post_path != post_path:
                    cached_post_img = Image.open(post_path).convert("RGB")
                    cached_post_path = post_path
                pre_img = cached_pre_img
                post_img = cached_post_img
            except Exception:
                stats["skipped_open_error"] += 1
                continue

            w, h = post_img.size
            poly_for_centroid = np.vstack([poly, poly[0]]) if len(poly) >= 2 and not np.allclose(poly[0], poly[-1]) else poly
            cx, cy = polygon_centroid(poly_for_centroid)
            x0, y0, x1, y1 = clamp_crop_window(cx, cy, cfg["crop_size"], w, h)
            pre_crop = pre_img.crop((x0, y0, x1, y1))
            post_crop = post_img.crop((x0, y0, x1, y1))

            poly_local = local_polygon(poly, x0, y0)
            m_mask_255 = draw_mask_polygon(cfg["crop_size"], poly_local)
            m_mask = (m_mask_255 > 0).astype(np.uint8)
            dil = dilate_fn(m_mask, cfg["ring_radius_px"])
            r_mask = np.clip(dil - m_mask, 0, 1).astype(np.uint8)
            r_mask_255 = (r_mask * 255).astype(np.uint8)

            pre_crop.save(pre_out, compress_level=cfg["png_compress_level"])
            post_crop.save(post_out, compress_level=cfg["png_compress_level"])
            Image.fromarray(m_mask_255, mode="L").save(m_out, compress_level=cfg["png_compress_level"])
            Image.fromarray(r_mask_255, mode="L").save(r_out, compress_level=cfg["png_compress_level"])

            out_row = dict(row)
            out_row.update(
                {
                    "pre_crop": str(pre_out),
                    "post_crop": str(post_out),
                    "mask_M": str(m_out),
                    "mask_R": str(r_out),
                    "crop_x0": x0,
                    "crop_y0": y0,
                    "crop_size": cfg["crop_size"],
                    "cx": f"{cx:.3f}",
                    "cy": f"{cy:.3f}",
                    "polygon_source_used": "pre_then_post",
                    "m_area_px": int(m_mask.sum()),
                    "r_area_px": int(r_mask.sum()),
                }
            )
            writer.writerow(out_row)
            stats["written_rows"] += 1

    return {"chunk_id": task["chunk_id"], "tmp_csv": str(tmp_csv_path), "stats": stats}


def main():
    args = parse_args()
    t0 = time.time()
    backend_name, _ = resolve_dilate_backend(args.dilate_backend)

    rows, skipped_index, n_label_files = collect_rows(args)
    if not rows:
        raise RuntimeError("No rows collected from Stage-1 labels.")

    args.out_root.mkdir(parents=True, exist_ok=True)
    out_pre = args.out_root / "crops_pre"
    out_post = args.out_root / "crops_post"
    out_m = args.out_root / "masks_M"
    out_r = args.out_root / "masks_R"
    out_tmp = args.out_root / "_tmp_shared_chunks"
    for p in (out_pre, out_post, out_m, out_r, out_tmp):
        p.mkdir(parents=True, exist_ok=True)

    out_cols = list(rows[0].keys()) + [
        "pre_crop",
        "post_crop",
        "mask_M",
        "mask_R",
        "crop_x0",
        "crop_y0",
        "crop_size",
        "cx",
        "cy",
        "polygon_source_used",
        "m_area_px",
        "r_area_px",
    ]

    tasks = []
    for chunk_id, chunk in enumerate(chunk_rows(rows, args.chunk_size)):
        tasks.append(
            {
                "chunk_id": chunk_id,
                "rows": chunk,
                "out_cols": out_cols,
                "out_pre": str(out_pre),
                "out_post": str(out_post),
                "out_m": str(out_m),
                "out_r": str(out_r),
                "tmp_csv": str(out_tmp / f"chunk_{chunk_id:06d}.csv"),
                "cfg": {
                    "crop_size": args.crop_size,
                    "ring_radius_px": args.ring_radius_px,
                    "dilate_backend": args.dilate_backend,
                    "overwrite": args.overwrite,
                    "png_compress_level": args.png_compress_level,
                },
            }
        )

    print(f"Loaded label files: {n_label_files}")
    print(f"Collected rows: {len(rows)}")
    print(
        "Config:",
        f"crop_size={args.crop_size}",
        f"ring_radius_px={args.ring_radius_px}",
        "polygon_source=pre_then_post",
        f"dilate_backend={backend_name}",
        f"num_workers={args.num_workers}",
        f"chunk_size={args.chunk_size}",
    )
    print(f"Prepared chunks: {len(tasks)}")

    agg = {
        "total_rows": len(rows),
        "written_rows": 0,
        "skipped_missing_images": 0,
        "skipped_open_error": 0,
        "skipped_existing": 0,
    }
    chunk_results = []
    processed_rows = 0
    next_log = args.log_every

    if args.num_workers == 1:
        for task in tasks:
            res = process_chunk(task)
            chunk_results.append(res)
            s = res["stats"]
            processed_rows += s["rows_in"]
            for k in ("written_rows", "skipped_missing_images", "skipped_open_error", "skipped_existing"):
                agg[k] += s[k]
            while args.log_every > 0 and processed_rows >= next_log:
                print(
                    "progress",
                    f"{processed_rows}/{len(rows)}",
                    f"({100.0 * processed_rows / len(rows):.1f}%)",
                    "| written",
                    agg["written_rows"],
                    "| skip_img",
                    agg["skipped_missing_images"],
                    "| skip_open",
                    agg["skipped_open_error"],
                    "| skip_exist",
                    agg["skipped_existing"],
                )
                next_log += args.log_every
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) as ex:
            futures = [ex.submit(process_chunk, t) for t in tasks]
            for fut in concurrent.futures.as_completed(futures):
                res = fut.result()
                chunk_results.append(res)
                s = res["stats"]
                processed_rows += s["rows_in"]
                for k in ("written_rows", "skipped_missing_images", "skipped_open_error", "skipped_existing"):
                    agg[k] += s[k]
                while args.log_every > 0 and processed_rows >= next_log:
                    print(
                        "progress",
                        f"{processed_rows}/{len(rows)}",
                        f"({100.0 * processed_rows / len(rows):.1f}%)",
                        "| written",
                        agg["written_rows"],
                        "| skip_img",
                        agg["skipped_missing_images"],
                        "| skip_open",
                        agg["skipped_open_error"],
                        "| skip_exist",
                        agg["skipped_existing"],
                    )
                    next_log += args.log_every

    chunk_results.sort(key=lambda x: x["chunk_id"])
    out_csv = args.out_root / "shared_instance_samples.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=out_cols)
        writer.writeheader()
        for item in chunk_results:
            tmp = Path(item["tmp_csv"])
            with tmp.open("r", encoding="utf-8", newline="") as in_f:
                for row in csv.DictReader(in_f):
                    writer.writerow(row)
            tmp.unlink(missing_ok=True)

    elapsed = time.time() - t0
    print("Wrote:", out_csv)
    print("Index skipped diagnostics:")
    for k, v in sorted(skipped_index.items()):
        print(k, v)
    for k in sorted(agg):
        print(k, agg[k])
    print("elapsed_seconds", f"{elapsed:.1f}")


if __name__ == "__main__":
    main()
