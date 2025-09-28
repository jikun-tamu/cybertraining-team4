#!/usr/bin/env python3
import argparse
import os
import sys
import subprocess
from typing import Optional, Tuple, List
from pathlib import Path
import shutil
import shlex

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
	# Quote each argument to handle spaces correctly
	quoted_cmd = ' '.join(shlex.quote(str(c)) for c in cmd)
	print(f"\n[run] {desc}\n$ {quoted_cmd}", flush=True)
	res = subprocess.run(quoted_cmd, shell=True)
	if res.returncode != 0:
		raise SystemExit(f"Step failed: {desc} (exit code {res.returncode})")

def find_pre_post_pairs(images_dir: Path) -> List[Tuple[Path, Path]]:
    """Find all (pre, post) image pairs in the data directory."""
    pairs = []
    post_images = sorted(list(images_dir.rglob("post/*.tif")))
    for post_path in post_images:
        cell_dir = post_path.parent.parent
        pre_path = next(cell_dir.glob("pre/*.tif"), None)
        if pre_path:
            pairs.append((pre_path, post_path))
        else:
            print(f"Warning: No pre-image found for post-image {post_path}")
    return pairs


def main():
	parser = argparse.ArgumentParser(description='Run Stage1→Stage2 pipeline end-to-end for multiple post-disaster images.')
	parser.add_argument('--images_dir', required=True, help='Root directory containing cell folders with pre/post subfolders (e.g., chips_600m)')
	parser.add_argument('--stage1_ckpt', required=True, help='Path to Stage1 segmentation checkpoint (.pth)')
	parser.add_argument('--stage2_ckpt', required=True, help='Path to Stage2 classifier checkpoint (.pt)')
	parser.add_argument('--work_root', default='', help='Base output directory; defaults to <images_dir>/../pipeline_outputs')
	parser.add_argument('--width', type=int, default=1024, help='Tile width; if 0, autodetect from images. Defaulting to 1024 for xBD.')
	parser.add_argument('--height', type=int, default=1024, help='Tile height; if 0, autodetect from images. Defaulting to 1024 for xBD.')
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
	parser.add_argument('--force', action='store_true', help='Force re-run steps even if outputs exist')
	args = parser.parse_args()

	images_dir = Path(args.images_dir).resolve()
	if not images_dir.is_dir():
		raise SystemExit(f"images_dir not found: {images_dir}")

	work_root = Path(args.work_root).resolve() if args.work_root else images_dir.parent / 'pipeline_outputs'
	stage1_masks_dir = work_root / 'stage1_masks'
	stage1_json_dir = work_root / 'stage1_json'
	
	ensure_dir(work_root)
	ensure_dir(stage1_masks_dir)
	ensure_dir(stage1_json_dir)

	width = args.width
	height = args.height
	if width <= 0 or height <= 0:
		wh = find_first_image_size(str(images_dir))
		if wh is None:
			raise SystemExit('Failed to autodetect image size. Provide --width and --height.')
		width, height = wh
	print(f"Using image size: width={width} height={height}")

	py = sys.executable
	# FIX: Point to the directory containing the model's scripts
	scripts_root = "/media/gisense/xihan/250812_CyberTraining_Team4/model/2-stage package/scripts"

	# --- STAGE 1: Run once for all pre-images ---
	print("--- Starting Stage 1: Footprint Extraction ---")
	stage1_cmd = [
		py, os.path.join(scripts_root, 'stage1_infer_tile_masks.py'),
		'--images_dir', str(images_dir),
		'--ckpt', os.path.abspath(args.stage1_ckpt),
		'--out_dir', str(stage1_masks_dir),
		'--thresh', str(args.thresh),
		'--min_area', str(args.min_area),
		'--open_ksize', str(args.open_ksize),
	]
	if not args.force and any(stage1_masks_dir.iterdir()):
		print(f"[skip] stage1 (found outputs in {stage1_masks_dir}); use --force to re-run")
	else:
		if args.force and any(stage1_masks_dir.iterdir()):
			shutil.rmtree(stage1_masks_dir)
			ensure_dir(stage1_masks_dir)
		run_step(stage1_cmd, desc='stage1_infer_tile_masks')

	polygonize_cmd = [
		py, os.path.join(scripts_root, 'polygonize_stage1_masks.py'),
		'--masks_dir', str(stage1_masks_dir),
		'--out_json_dir', str(stage1_json_dir),
		'--min_area', str(args.min_area),
		'--simplify_tol', str(args.simplify_tol),
	]
	if not args.force and any(stage1_json_dir.iterdir()):
		print(f"[skip] polygonize (found outputs in {stage1_json_dir}); use --force to re-run")
	else:
		if args.force and any(stage1_json_dir.iterdir()):
			shutil.rmtree(stage1_json_dir)
			ensure_dir(stage1_json_dir)
		run_step(polygonize_cmd, desc='polygonize_stage1_masks')
	
	print("\n--- Stage 1 Complete. Polygons generated. ---")

	# --- STAGE 2: Loop through each pre-post pair ---
	print("\n--- Starting Stage 2: Damage Classification for each Post-Image ---")
	pre_post_pairs = find_pre_post_pairs(images_dir)
	print(f"Found {len(pre_post_pairs)} pre-post pairs to process for Stage 2.")

	for pre_path, post_path in pre_post_pairs:
		post_date = post_path.stem.split('_post_')[-1]
		cell_name = pre_path.parent.parent.name
		pair_id = f"{cell_name}_{post_date}"
		print(f"\n--- Processing Pair: {pair_id} ---")

		# Define output directories for this specific pair
		stage2_root = work_root / 'stage2' / pair_id
		overlays_out = work_root / 'overlays_stage2' / pair_id
		ensure_dir(stage2_root)
		ensure_dir(overlays_out)

		# Create a temporary, single-pair image directory for the indexing script
		temp_images_dir = work_root / 'temp_pair_dir'
		if temp_images_dir.exists():
			shutil.rmtree(temp_images_dir)
		
		# The indexing script expects a flat pre/post structure, so we create it
		(temp_images_dir / 'pre').mkdir(parents=True)
		(temp_images_dir / 'post').mkdir(parents=True)
		shutil.copy(pre_path, temp_images_dir / 'pre' / pre_path.name)
		shutil.copy(post_path, temp_images_dir / 'post' / post_path.name)

		# Define commands for this pair
		index_cmd = [
			py, os.path.join(scripts_root, 'build_stage2_index_from_pred.py'),
			'--images_root', str(temp_images_dir),
			'--pred_labels_root', str(stage1_json_dir),
			'--out_csv', str(stage2_root / 'stage2_index_pred.csv'),
			'--event_id', pair_id,
			'--width', str(width), '--height', str(height),
		]
		preprocess_cmd = [
			py, os.path.join(scripts_root, 'preprocess_stage2_crops.py'),
			'--index_csv', str(stage2_root / 'stage2_index_pred.csv'),
			'--out_root', str(stage2_root),
			'--crop_size', str(args.crop_size),
			'--ring_radius_px', str(args.ring_radius_px),
		] + (['--limit', str(args.limit)] if args.limit is not None else []) + (['--no_progress'] if args.no_progress else [])
		
		stage2_infer_cmd = [
			py, os.path.join(scripts_root, 'infer_overlay_stage2.py'),
			'--csv', str(stage2_root / 'stage2_samples_floods.csv'),
			'--ckpt', os.path.abspath(args.stage2_ckpt),
			'--out_dir', str(overlays_out),
			'--classes', str(args.classes),
			'--backbone', args.backbone,
		] + (['--limit', str(args.limit)] if args.limit is not None else [])

		# Run steps for the pair
		run_step(index_cmd, f"index for {pair_id}")
		run_step(preprocess_cmd, f"preprocess for {pair_id}")
		run_step(stage2_infer_cmd, f"stage2_infer for {pair_id}")

		# Clean up temp dir
		shutil.rmtree(temp_images_dir)
		print(f"--- Finished Pair: {pair_id}. Overlays in: {overlays_out} ---")

	print("\nAll steps completed for all pairs.")

if __name__ == '__main__':
	main()
