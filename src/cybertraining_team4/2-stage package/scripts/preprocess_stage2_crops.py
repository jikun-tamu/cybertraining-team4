#!/usr/bin/env python3
import argparse
import csv
import math
import os
from typing import Tuple, List, Optional

import numpy as np
from PIL import Image, ImageDraw
from shapely import wkt
from shapely.geometry import Polygon, MultiPolygon
from tqdm import tqdm


def clamp(v: int, lo: int, hi: int) -> int:
	return max(lo, min(hi, v))


def polygon_from_wkt(wkt_str: str) -> Optional[Polygon]:
	if not wkt_str:
		return None
	try:
		geom = wkt.loads(wkt_str)
		if isinstance(geom, Polygon):
			return geom
		elif isinstance(geom, MultiPolygon):
			# merge into a single polygon by union of bounds; for masks we'll rasterize all parts
			return geom
		else:
			return None
	except Exception:
		return None


def compute_crop_bounds(cx: float, cy: float, crop_size: int, width: int, height: int) -> Tuple[int, int, int, int]:
	half = crop_size // 2
	x0 = int(round(cx)) - half
	y0 = int(round(cy)) - half
	x0 = clamp(x0, 0, max(0, width - crop_size))
	y0 = clamp(y0, 0, max(0, height - crop_size))
	x1 = x0 + crop_size
	y1 = y0 + crop_size
	return x0, y0, x1, y1


def draw_polygon_on_mask(draw: ImageDraw.ImageDraw, poly: Polygon, offset: Tuple[int, int], fill: int):
	# Handles Polygon or MultiPolygon
	def draw_single(p: Polygon):
		x_off, y_off = offset
		exterior = [(x - x_off, y - y_off) for x, y in np.array(p.exterior.coords)]
		draw.polygon(exterior, fill=fill)
		for interior in p.interiors:
			interior_pts = [(x - x_off, y - y_off) for x, y in np.array(interior.coords)]
			draw.polygon(interior_pts, fill=0)

	if isinstance(poly, MultiPolygon):
		for part in poly.geoms:
			draw_single(part)
	else:
		draw_single(poly)


def rasterize_mask_from_polygon(poly: Polygon, crop_box: Tuple[int, int, int, int], size: int) -> np.ndarray:
	mask_img = Image.new('L', (size, size), 0)
	draw = ImageDraw.Draw(mask_img)
	draw_polygon_on_mask(draw, poly, (crop_box[0], crop_box[1]), fill=1)
	return np.array(mask_img, dtype=np.uint8)


def load_image(path: str) -> Image.Image:
	img = Image.open(path).convert('RGB')
	return img


def save_image_array(arr: np.ndarray, path: str):
	Image.fromarray(arr).save(path)


def process_row(row: dict, out_root: str, crop_size: int, ring_radius_px: int) -> Optional[dict]:
	pre_path = row['pre_image']
	post_path = row['post_image']
	width = int(row['width'])
	height = int(row['height'])
	uid = row['bldg_uid']
	tile = row['tile_id']
	# Prefer post polygon for centroid; fallback to pre
	poly_post = polygon_from_wkt(row.get('polygon_wkt_xy_post', ''))
	poly_pre = polygon_from_wkt(row.get('polygon_wkt_xy_pre', ''))
	poly_for_center = poly_post if poly_post is not None else poly_pre
	if poly_for_center is None:
		return None
	centroid = poly_for_center.centroid
	cx, cy = centroid.x, centroid.y
	crop_box = compute_crop_bounds(cx, cy, crop_size, width, height)  # (x0,y0,x1,y1)

	# Load and crop images
	pre_img = load_image(pre_path).crop(crop_box)
	post_img = load_image(post_path).crop(crop_box)

	# Masks: use post polygon for M if available else pre
	poly_for_mask = poly_post if poly_post is not None else poly_pre
	mask_M = rasterize_mask_from_polygon(poly_for_mask, crop_box, crop_size)

	# Ring via buffer then subtract M
	buffer_poly = None
	try:
		buffer_poly = poly_for_mask.buffer(ring_radius_px)
	except Exception:
		buffer_poly = None
	if buffer_poly is not None:
		mask_dil = rasterize_mask_from_polygon(buffer_poly, crop_box, crop_size)
		mask_R = (mask_dil.astype(np.int16) - mask_M.astype(np.int16))
		mask_R = np.clip(mask_R, 0, 1).astype(np.uint8)
	else:
		mask_R = np.zeros_like(mask_M, dtype=np.uint8)

	# Save outputs
	pre_dir = os.path.join(out_root, 'crops_pre')
	post_dir = os.path.join(out_root, 'crops_post')
	m_dir = os.path.join(out_root, 'masks_M')
	r_dir = os.path.join(out_root, 'masks_R')
	for d in [pre_dir, post_dir, m_dir, r_dir]:
		os.makedirs(d, exist_ok=True)

	base_name = f"{tile}__{uid}"
	pre_out = os.path.join(pre_dir, base_name + '.png')
	post_out = os.path.join(post_dir, base_name + '.png')
	m_out = os.path.join(m_dir, base_name + '.png')
	r_out = os.path.join(r_dir, base_name + '.png')

	pre_img.save(pre_out)
	post_img.save(post_out)
	save_image_array((mask_M * 255).astype(np.uint8), m_out)
	save_image_array((mask_R * 255).astype(np.uint8), r_out)

	aug = {
		'pre_crop': pre_out,
		'post_crop': post_out,
		'mask_M': m_out,
		'mask_R': r_out,
		'crop_x0': crop_box[0],
		'crop_y0': crop_box[1],
		'crop_size': crop_size,
		'cx': cx,
		'cy': cy,
	}
	return aug


def run(index_csv: str, out_root: str, crop_size: int, ring_radius_px: int, limit: Optional[int], progress: bool) -> str:
	os.makedirs(out_root, exist_ok=True)
	aug_csv = os.path.join(out_root, 'stage2_samples_floods.csv')

	# Count total data rows for progress bar (exclude header)
	try:
		with open(index_csv, 'r') as _tf:
			total_rows = max(sum(1 for _ in _tf) - 1, 0)
	except Exception:
		total_rows = 0
	if limit is not None and limit > 0:
		effective_total = min(total_rows, limit)
	else:
		effective_total = total_rows

	processed = 0
	skipped = 0
	with open(index_csv, 'r') as f_in, open(aug_csv, 'w', newline='') as f_out:
		reader = csv.DictReader(f_in)
		base_fields = (reader.fieldnames or [])
		fieldnames = base_fields + ['pre_crop', 'post_crop', 'mask_M', 'mask_R', 'crop_x0', 'crop_y0', 'crop_size', 'cx', 'cy']
		writer = csv.DictWriter(f_out, fieldnames=fieldnames)
		writer.writeheader()

		it = reader
		if progress:
			it = tqdm(reader, total=(effective_total if effective_total > 0 else None), desc='Preprocess Stage2 crops', unit='obj')

		for row in it:
			aug = process_row(row, out_root, crop_size, ring_radius_px)
			if aug is None:
				skipped += 1
				# Update live postfix if tqdm is active
				if progress:
					try:
						it.set_postfix({"ok": processed, "skip": skipped})
					except Exception:
						pass
				continue
			row.update(aug)
			writer.writerow(row)
			processed += 1
			if progress:
				try:
					it.set_postfix({"ok": processed, "skip": skipped})
				except Exception:
					pass
			if limit and processed >= limit:
				break

	print(f'Wrote augmented CSV: {aug_csv} | processed={processed} | skipped={skipped}')
	return aug_csv


def main():
	parser = argparse.ArgumentParser(description='Preprocess Stage-2 crops and masks from index CSV.')
	parser.add_argument('--index_csv', default='/anvil/scratch/x-jliu7/test/stage2_index_floods.csv')
	parser.add_argument('--out_root', default='/anvil/scratch/x-jliu7/test_stage2')
	parser.add_argument('--crop_size', type=int, default=256)
	parser.add_argument('--ring_radius_px', type=int, default=48)
	parser.add_argument('--limit', type=int, default=None, help='Process only first N rows')
	parser.add_argument('--no_progress', action='store_true', help='Disable progress bar output')
	args = parser.parse_args()
	aug_csv = run(args.index_csv, args.out_root, args.crop_size, args.ring_radius_px, args.limit, progress=(not args.no_progress))


if __name__ == '__main__':
	main()


