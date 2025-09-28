#!/usr/bin/env python3
import argparse
import os
import json
import sys
from typing import List, Dict

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from tqdm import tqdm

a = '/anvil/scratch/x-jliu7'
if a not in sys.path:
	sys.path.append(a)

from src.data.stage2_dataset import Stage2Dataset
from src.models.siamese_stage2 import SiameseDamageModel


CLASS_NAMES = ['no-damage', 'minor', 'major', 'destroyed']
CLASS_COLORS = {
	0: (0, 200, 0),
	1: (255, 215, 0),
	2: (255, 140, 0),
	3: (220, 20, 60),
}


def overlay_mask_on_image(img: Image.Image, mask: Image.Image, color: tuple, alpha: float = 0.35) -> Image.Image:
	base = img.convert('RGBA')
	mask_arr = np.array(mask.convert('L')) > 127
	overlay = np.zeros((img.height, img.width, 4), dtype=np.uint8)
	overlay[mask_arr] = (*color, int(255 * alpha))
	over = Image.fromarray(overlay, mode='RGBA')
	return Image.alpha_composite(base, over).convert('RGB')


def draw_label(img: Image.Image, text: str, color: tuple) -> Image.Image:
	draw = ImageDraw.Draw(img)
	# try a default font
	try:
		font = ImageFont.truetype("DejaVuSans.ttf", 16)
	except Exception:
		font = ImageFont.load_default()
	padding = 4
	text_size = draw.textbbox((0, 0), text, font=font)
	box_w = text_size[2] - text_size[0] + 2 * padding
	box_h = text_size[3] - text_size[1] + 2 * padding
	# top-left corner
	shape = [(0, 0), (box_w, box_h)]
	bg = (0, 0, 0, 160)
	bg_img = Image.new('RGBA', (box_w, box_h), bg)
	img = img.convert('RGBA')
	img.paste(bg_img, (0, 0), bg_img)
	draw = ImageDraw.Draw(img)
	draw.text((padding, padding), text, fill=color, font=font)
	return img.convert('RGB')


def main():
	parser = argparse.ArgumentParser(description='Overlay Stage-2 predictions on crops with confidence')
	parser.add_argument('--csv', default='/anvil/scratch/x-jliu7/test_stage2/stage2_samples_floods.csv')
	parser.add_argument('--ckpt', default='/anvil/scratch/x-jliu7/outputs_stage2/stage2_best.pt')
	parser.add_argument('--backbone', default='convnext_tiny')
	parser.add_argument('--classes', type=int, default=4)
	parser.add_argument('--out_dir', default='/anvil/scratch/x-jliu7/overlays_stage2')
	parser.add_argument('--limit', type=int, default=50)
	parser.add_argument('--val_only', action='store_true', help='If set, compute overlays/metrics on validation split only')
	parser.add_argument('--metrics_out', default='', help='Optional path to write JSON metrics summary')
	parser.add_argument('--samples_out', default='', help='Optional path to write per-sample JSONL with probs/confidence/entropy/margin/brier')
	parser.add_argument('--seed', type=int, default=42)
	args = parser.parse_args()

	os.makedirs(args.out_dir, exist_ok=True)
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	# Load dataset (no transforms)
	ds = Stage2Dataset(args.csv, return_meta=True)

	# Optionally restrict to validation split
	indices = list(range(len(ds)))
	if args.val_only:
		# tile-level split to reduce leakage
		def _split_by_tile(rows: List[Dict[str, str]], val_frac: float = 0.2, seed: int = 42):
			import random
			tiles = sorted({r['tile_id'] for r in rows})
			random.Random(seed).shuffle(tiles)
			cut = int(round(len(tiles) * (1 - val_frac)))
			train_tiles = set(tiles[:cut])
			val_tiles = set(tiles[cut:])
			train_idx, val_idx = [], []
			for i, r in enumerate(rows):
				(train_idx if r['tile_id'] in train_tiles else val_idx).append(i)
			return train_idx, val_idx
		_, val_idx = _split_by_tile(ds.rows, seed=args.seed)
		indices = val_idx
	# Optional limit
	if args.limit and len(indices) > args.limit:
		indices = indices[:args.limit]

	# Build model and load checkpoint
	model = SiameseDamageModel(backbone_name=args.backbone, num_classes=args.classes).to(device)
	# Trigger head build on a single sample to match state_dict keys
	with torch.no_grad():
		pre0, post0, m0, r0, _y0, _meta0 = ds[0]
		_ = model(pre0.unsqueeze(0).to(device), post0.unsqueeze(0).to(device), m0.unsqueeze(0).to(device), r0.unsqueeze(0).to(device))
	# Safe load with weights_only when available
	try:
		ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)  # type: ignore
	except TypeError:
		ckpt = torch.load(args.ckpt, map_location=device)
	# Allow strict=False to load older checkpoints without temperature buffer
	model.load_state_dict(ckpt['model'], strict=False)
	# If calibration info present, set model.temperature
	calib = ckpt.get('calibration', {}) if isinstance(ckpt, dict) else {}
	T = calib.get('temperature', None)
	if T is not None:
		with torch.no_grad():
			# temperature is a 0-D buffer; use copy_ to assign scalar
			model.temperature.copy_(torch.tensor(float(T), device=model.temperature.device))
	# Optional: class-wise vector temperature in probability space
	vector_T = calib.get('vector_temperature', None)
	vector_T = [float(x) for x in vector_T] if isinstance(vector_T, (list, tuple)) else None
	# Calibration metadata for outputs
	calibration_meta = {
		'type': 'vector_temperature' if vector_T is not None else ('scalar_temperature' if T is not None else 'none'),
		'temperature': float(T) if T is not None else None,
		'vector_alphas': vector_T,
	}
	model.eval()

	# Metrics accumulators (per true class)
	K = args.classes
	conf_correct = [[] for _ in range(K)]
	conf_incorrect = [[] for _ in range(K)]
	tol_hits = [0 for _ in range(K)]
	totals = [0 for _ in range(K)]

	# Prepare optional samples JSONL
	samples_f = open(args.samples_out, 'w') if args.samples_out else None
	for i in tqdm(indices, desc='Overlay infer'):
		pre, post, m, r, _, meta = ds[i]
		# batchify
		with torch.no_grad():
			out = model(pre.unsqueeze(0).to(device), post.unsqueeze(0).to(device), m.unsqueeze(0).to(device), r.unsqueeze(0).to(device))
			probs = out['probs']  # already normalized from CORAL conversion
			if vector_T is not None:
				p = probs.clamp_min(1e-12)
				alphas = torch.tensor(vector_T, device=p.device).view(1, -1)
				p = p.pow(alphas)
				probs = p / p.sum(dim=1, keepdim=True).clamp_min(1e-12)
			probs_np = probs.squeeze(0).detach().cpu().numpy()
			pred = int(probs.argmax(dim=1).item())
			conf = float(probs.max().item())
			# margin and entropy
			probs_sorted, _ = torch.sort(probs.squeeze(0), descending=True)
			margin = float((probs_sorted[0] - probs_sorted[1]).item()) if probs_sorted.numel() > 1 else float('nan')
			entropy = float(-(probs.squeeze(0).clamp_min(1e-12).log() * probs.squeeze(0).clamp_min(1e-12)).sum().item())

		# Load original post image for visualization
		post_img = Image.open(ds.rows[i]['post_crop']).convert('RGB')
		mask_m = Image.open(ds.rows[i]['mask_M']).convert('L')
		color = CLASS_COLORS.get(pred, (0, 255, 255))
		over = overlay_mask_on_image(post_img, mask_m, color=color, alpha=0.35)
		label = CLASS_NAMES[pred] if pred < len(CLASS_NAMES) else str(pred)
		try:
			gt_id = int(ds.rows[i]['damage_class'])
		except Exception:
			gt_id = -1
		gt_label = CLASS_NAMES[gt_id] if 0 <= gt_id < len(CLASS_NAMES) else str(gt_id)
		text = f"pred: {label} | conf={conf:.2f} | gt: {gt_label}\nmargin={margin:.2f} | ent={entropy:.2f}"
		over = draw_label(over, text, color=color)
		out_path = os.path.join(args.out_dir, f"{ds.rows[i]['tile_id']}__{ds.rows[i]['bldg_uid']}.png")
		over.save(out_path)

		# Metrics accumulation (only if gt in range)
		if 0 <= gt_id < K:
			totals[gt_id] += 1
			if pred == gt_id:
				conf_correct[gt_id].append(conf)
			else:
				conf_incorrect[gt_id].append(conf)
			# tolerant accuracy (|pred - gt| <= 1)
			if abs(pred - gt_id) <= 1:
				tol_hits[gt_id] += 1
			# brier score
			one_hot = torch.zeros((1, K), device=probs.device)
			one_hot[0, gt_id] = 1.0
			brier = float(((probs - one_hot) ** 2).sum(dim=1).item())
		else:
			brier = float('nan')

		# Optional per-sample JSONL output
		if samples_f is not None:
			rec = {
				'tile_id': ds.rows[i]['tile_id'],
				'bldg_uid': ds.rows[i]['bldg_uid'],
				'gt_class': gt_id if 0 <= gt_id < K else None,
				'pred_class': pred,
				'probs': [float(x) for x in probs_np.tolist()],
				'max_prob': conf,
				'margin': margin,
				'predictive_entropy': entropy,
				'brier_score': brier,
			}
			import json as _json
			samples_f.write(_json.dumps(rec) + "\n")

	# Summarize metrics
	def _mean_std(x):
		if len(x) == 0:
			return float('nan'), float('nan')
		return float(np.mean(x)), float(np.std(x))

	print("\nValidation confidence stats (by true class):", flush=True)
	for c in range(K):
		mc, sc = _mean_std(conf_correct[c])
		mi, si = _mean_std(conf_incorrect[c])
		tol_acc = (tol_hits[c] / totals[c]) if totals[c] > 0 else float('nan')
		print(f"class={c} correct_conf_mean={mc:.4f} std={sc:.4f} | incorrect_conf_mean={mi:.4f} std={si:.4f} | tolerant_acc(|pred-gt|<=1)={tol_acc:.4f}", flush=True)

	# Optional JSON dump
	if args.metrics_out:
		os.makedirs(os.path.dirname(args.metrics_out), exist_ok=True)
		out = {
			'per_class': {
				str(c): {
					'correct_conf_mean': _mean_std(conf_correct[c])[0],
					'correct_conf_std': _mean_std(conf_correct[c])[1],
					'incorrect_conf_mean': _mean_std(conf_incorrect[c])[0],
					'incorrect_conf_std': _mean_std(conf_incorrect[c])[1],
					'tolerant_acc': (tol_hits[c] / totals[c]) if totals[c] > 0 else float('nan'),
					'count': totals[c],
				}
				for c in range(K)
			},
			'calibration': calibration_meta,
		}
		with open(args.metrics_out, 'w') as f:
			json.dump(out, f, indent=2)
		print(f"Wrote metrics to {args.metrics_out}")

	if samples_f is not None:
		samples_f.close()
	print(f"Wrote overlays to {args.out_dir}")


if __name__ == '__main__':
	main()


