#!/usr/bin/env python3
import argparse
import os
import sys
import subprocess
from typing import Optional, Tuple, List

from PIL import Image


def find_first_image_size(images_root: str) -> Optional[Tuple[int, int]]:
	"""Return (width, height) for the first image found under images_root, or None."""
	valid_ext = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}
	for root, _dirs, files in os.walk(images_root):
		for name in files:
			_, ext = os.path.splitext(name)
			if ext.lower() in valid_ext:
				path = os.path.join(root, name)
				try:
					with Image.open(path) as im:
						w, h = im.size
						return w, h
				except Exception:
					continue
	return None


def ensure_dir(path: str) -> None:
	os.makedirs(path, exist_ok=True)


def run_step(cmd: List[str], desc: str) -> None:
	print(f"\n[run] {desc}\n$ {' '.join(cmd)}", flush=True)
	res = subprocess.run(cmd)
	if res.returncode != 0:
		raise SystemExit(f"Step failed: {desc} (exit code {res.returncode})")


def main():
	parser = argparse.ArgumentParser(description='Run Stage1→Stage2 pipeline end-to-end.')
	parser.add_argument('--images_dir', required=True, help='Root directory containing pre/post image tiles')
	parser.add_argument('--stage1_ckpt', required=True, help='Path to Stage1 segmentation checkpoint (.pth)')
	parser.add_argument('--stage2_ckpt', required=True, help='Path to Stage2 classifier checkpoint (.pt)')
	parser.add_argument('--work_root', default='', help='Base output directory; defaults to <images_dir>/../pipeline_outputs')
	parser.add_argument('--width', type=int, default=0, help='Tile width; if 0, autodetect from images')
	parser.add_argument('--height', type=int, default=0, help='Tile height; if 0, autodetect from images')
	parser.add_argument('--thresh', type=float, default=0.5)
	parser.add_argument('--min_area', type=int, default=50)
	parser.add_argument('--open_ksize', type=int, default=3)
	parser.add_argument('--simplify_tol', type=float, default=1.5)
	parser.add_argument('--crop_size', type=int, default=256)
	parser.add_argument('--ring_radius_px', type=int, default=48)
	parser.add_argument('--classes', type=int, default=4)
	parser.add_argument('--backbone', default='convnext_tiny')
	parser.add_argument('--limit', type=int, default=None, help='Optional limit for preprocess and overlay')
	parser.add_argument('--no_progress', action='store_true', help='Disable progress bar in preprocess step')
	parser.add_argument('--overlays_out', default='', help='Optional output directory for Stage-2 overlays')
	parser.add_argument('--resume_from', choices=['stage1', 'polygonize', 'index', 'preprocess', 'stage2'], default='stage1')
	parser.add_argument('--force', action='store_true', help='Force re-run steps even if outputs exist')
	args = parser.parse_args()

	images_dir = os.path.abspath(args.images_dir)
	if not os.path.isdir(images_dir):
		raise SystemExit(f"images_dir not found: {images_dir}")

	work_root = os.path.abspath(args.work_root) if args.work_root else os.path.abspath(os.path.join(images_dir, os.pardir, 'pipeline_outputs'))
	stage1_masks_dir = os.path.join(work_root, 'stage1_masks')
	stage1_json_dir = os.path.join(work_root, 'stage1_json')
	stage2_root = os.path.join(work_root, 'stage2')
	overlays_out = os.path.abspath(args.overlays_out) if args.overlays_out else os.path.join(work_root, 'overlays_stage2')
	ensure_dir(work_root)

	width = args.width
	height = args.height
	if width <= 0 or height <= 0:
		wh = find_first_image_size(images_dir)
		if wh is None:
			raise SystemExit('Failed to autodetect image size. Provide --width and --height.')
		width, height = wh
	print(f"Using image size: width={width} height={height}")

	py = sys.executable
	scripts_root = os.getcwd()

	steps = [
		('stage1', [
			py, os.path.join(scripts_root, 'stage1_infer_tile_masks.py'),
			'--images_dir', images_dir,
			'--ckpt', os.path.abspath(args.stage1_ckpt),
			'--out_dir', stage1_masks_dir,
			'--thresh', str(args.thresh),
			'--min_area', str(args.min_area),
			'--open_ksize', str(args.open_ksize),
		]),
		('polygonize', [
			py, os.path.join(scripts_root, 'polygonize_stage1_masks.py'),
			'--masks_dir', stage1_masks_dir,
			'--out_json_dir', stage1_json_dir,
			'--min_area', str(args.min_area),
			'--simplify_tol', str(args.simplify_tol),
		]),
		('index', [
			py, os.path.join(scripts_root, 'build_stage2_index_from_pred.py'),
			'--images_root', images_dir,
			'--pred_labels_root', stage1_json_dir,
			'--out_csv', os.path.join(stage2_root, 'stage2_index_pred.csv'),
			'--event_id', 'pred_event',
			'--width', str(width), '--height', str(height),
		]),
		('preprocess', [
			py, os.path.join(scripts_root, 'preprocess_stage2_crops.py'),
			'--index_csv', os.path.join(stage2_root, 'stage2_index_pred.csv'),
			'--out_root', stage2_root,
			'--crop_size', str(args.crop_size),
			'--ring_radius_px', str(args.ring_radius_px),
		] + (['--limit', str(args.limit)] if args.limit is not None else []) + (['--no_progress'] if args.no_progress else [])),
		('stage2', [
			py, os.path.join(scripts_root, 'infer_overlay_stage2.py'),
			'--csv', os.path.join(stage2_root, 'stage2_samples_floods.csv'),
			'--ckpt', os.path.abspath(args.stage2_ckpt),
			'--out_dir', overlays_out,
			'--classes', str(args.classes),
			'--backbone', args.backbone,
		] + (['--limit', str(args.limit)] if args.limit is not None else [])),
	]

	resume_order = {name: i for i, (name, _c) in enumerate(steps)}
	start_idx = resume_order.get(args.resume_from, 0)

	if not args.force:
		ensure_dir(stage1_masks_dir)
		ensure_dir(stage1_json_dir)
		ensure_dir(stage2_root)
		ensure_dir(overlays_out)

	for i, (name, cmd) in enumerate(steps):
		if i < start_idx:
			print(f"[skip] {name} (resuming from {args.resume_from})")
			continue
		if not args.force:
			if name == 'stage1' and os.listdir(stage1_masks_dir):
				print(f"[skip] {name} (found outputs in {stage1_masks_dir}); use --force to re-run")
				continue
			if name == 'polygonize' and os.listdir(stage1_json_dir):
				print(f"[skip] {name} (found outputs in {stage1_json_dir}); use --force to re-run")
				continue
			if name == 'index' and os.path.isfile(os.path.join(stage2_root, 'stage2_index_pred.csv')):
				print(f"[skip] {name} (found CSV); use --force to re-run")
				continue
			if name == 'preprocess' and os.path.isfile(os.path.join(stage2_root, 'stage2_samples_floods.csv')):
				print(f"[skip] {name} (found samples CSV); use --force to re-run")
				continue
			if name == 'stage2' and os.listdir(overlays_out):
				print(f"[skip] {name} (found overlays in {overlays_out}); use --force to re-run")
				continue
		run_step(cmd, desc=name)

	print("\nAll steps completed.")
	print(f"Outputs:\n  Stage1 masks: {stage1_masks_dir}\n  Stage1 polygons: {stage1_json_dir}\n  Stage2 index CSV: {os.path.join(stage2_root, 'stage2_index_pred.csv')}\n  Stage2 samples CSV: {os.path.join(stage2_root, 'stage2_samples_floods.csv')}\n  Overlays: {overlays_out}")


if __name__ == '__main__':
	main()


