#!/usr/bin/env python3
import argparse
import csv
import os
import random
import sys
from typing import List, Dict, Tuple

import numpy as np
import time
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
try:
	from torch.amp import autocast as amp_autocast, GradScaler as AmpGradScaler
	_AMP_NEW = True
except Exception:
	from torch.cuda.amp import autocast as amp_autocast, GradScaler as AmpGradScaler
	_AMP_NEW = False
from torch.utils.data import DataLoader, Subset

a = '/anvil/scratch/x-jliu7'
if a not in sys.path:
	sys.path.append(a)

from src.data.stage2_dataset import Stage2Dataset
from src.models.siamese_stage2 import SiameseDamageModel
from src.models.coral_head import coral_targets

from torchmetrics.classification import MulticlassF1Score, MulticlassCalibrationError, MulticlassCohenKappa


def set_seed(seed: int):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)


def read_rows(csv_path: str) -> List[Dict[str, str]]:
	rows: List[Dict[str, str]] = []
	with open(csv_path, 'r') as f:
		reader = csv.DictReader(f)
		for r in reader:
			rows.append(r)
	return rows


def split_by_tile(rows: List[Dict[str, str]], val_frac: float = 0.2, seed: int = 42) -> Tuple[List[int], List[int]]:
	# tile-level split to reduce leakage
	tiles = sorted({r['tile_id'] for r in rows})
	random.Random(seed).shuffle(tiles)
	cut = int(round(len(tiles) * (1 - val_frac)))
	train_tiles = set(tiles[:cut])
	val_tiles = set(tiles[cut:])
	train_idx, val_idx = [], []
	for i, r in enumerate(rows):
		if r['tile_id'] in train_tiles:
			train_idx.append(i)
		else:
			val_idx.append(i)
	return train_idx, val_idx


def make_loaders(csv_path: str, batch_size: int, num_workers: int, limit: int = 0, class_balance: bool = False, num_classes: int = 4, balance_alpha: float = 0.5, balance_cap: float = 12.0):
	rows = read_rows(csv_path)
	if limit and len(rows) > limit:
		rows = rows[:limit]
	# write a temp CSV subset if needed
	tmp_csv = csv_path
	if limit:
		tmp_csv = csv_path + f".limit{limit}.tmp"
		with open(tmp_csv, 'w', newline='') as f:
			writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
			writer.writeheader()
			for r in rows:
				writer.writerow(r)

	ds = Stage2Dataset(tmp_csv)
	train_idx, val_idx = split_by_tile(ds.rows)
	train_ds = Subset(ds, train_idx)
	val_ds = Subset(ds, val_idx)
	# Optional class-balanced sampling for training
	if class_balance:
		# Compute class histogram on the training split
		labels_tr = [int(ds.rows[i]['damage_class']) for i in train_idx]
		import numpy as _np
		counts = _np.bincount(labels_tr, minlength=max(num_classes, (max(labels_tr) + 1 if labels_tr else 0)))
		# Tempered inverse-frequency weights: w_c ∝ (1 / count_c)^alpha
		alpha = float(balance_alpha)
		counts_clip = _np.clip(counts, 1, None)
		weights_per_class = counts_clip.astype(_np.float64) ** (-alpha)
		# Normalize so max weight = 1
		if weights_per_class.max() > 0:
			weights_per_class = weights_per_class / weights_per_class.max()
		# Cap the max/min ratio (optional)
		cap = float(balance_cap)
		if cap and cap > 0:
			min_allowed = 1.0 / cap
			weights_per_class = _np.clip(weights_per_class, min_allowed, 1.0)
		# Sample weight per example in train subset order
		sample_weights = [float(weights_per_class[y]) for y in labels_tr]
		from torch.utils.data import WeightedRandomSampler as _WRS
		sampler = _WRS(sample_weights, num_samples=len(sample_weights), replacement=True)
		print(f"train_class_counts={counts.tolist()} class_weights={weights_per_class.tolist()} alpha={alpha} cap={cap}", flush=True)
		train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, shuffle=False, num_workers=num_workers, pin_memory=True)
	else:
		train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
	val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
	return train_loader, val_loader


def train_one_epoch(model, loader, optimizer, device, num_classes: int, epoch: int):
	model.train()
	loss_fn = nn.BCEWithLogitsLoss()
	total_loss = 0.0
	count = 0
	use_cuda = (device.type == 'cuda')
	scaler = AmpGradScaler(enabled=use_cuda)
	progress = tqdm(loader, total=len(loader), desc=f"Epoch {epoch} [train]", leave=False)
	for step, (pre, post, m, r, y) in enumerate(progress, start=1):
		pre = pre.to(device)
		post = post.to(device)
		m = m.to(device)
		r = r.to(device)
		y = y.to(device)
		if _AMP_NEW and use_cuda:
			ctx = amp_autocast('cuda', enabled=True)
		else:
			ctx = amp_autocast(enabled=use_cuda)
		with ctx:
			out = model(pre, post, m, r)
			logits_cum = out['logits_cum']
			targets = coral_targets(y, num_classes)
			loss = loss_fn(logits_cum, targets)
		optimizer.zero_grad(set_to_none=True)
		scaler.scale(loss).backward()
		scaler.step(optimizer)
		scaler.update()
		total_loss += loss.item() * y.size(0)
		count += y.size(0)
		# Update progress bar with running loss
		avg = total_loss / max(count, 1)
		progress.set_postfix({"loss": f"{avg:.4f}"})
	return total_loss / max(count, 1)


def evaluate(model, loader, device, num_classes: int):
	model.eval()
	f1 = MulticlassF1Score(num_classes=num_classes, average='macro').to(device)
	f1_per_class = MulticlassF1Score(num_classes=num_classes, average=None).to(device)
	kappa = MulticlassCohenKappa(num_classes=num_classes, weights='quadratic').to(device)
	# ECE on multiclass probabilities
	try:
		from torchmetrics.classification import MulticlassCalibrationError
		ece = MulticlassCalibrationError(num_classes=num_classes, n_bins=15).to(device)
	except Exception:
		ece = None
	import torch.nn.functional as F
	val_loss_fn = nn.BCEWithLogitsLoss()
	total_loss = 0.0
	count = 0
	updated = False
	with torch.no_grad():
		for pre, post, m, r, y in tqdm(loader, total=len(loader), desc="Eval", leave=False):
			pre = pre.to(device)
			post = post.to(device)
			m = m.to(device)
			r = r.to(device)
			y = y.to(device)
			out = model(pre, post, m, r)
			logits_cum = out['logits_cum']
			probs = out['probs']
			preds = probs.argmax(dim=1)
			targets = coral_targets(y, num_classes)
			loss = val_loss_fn(logits_cum, targets)
			total_loss += loss.item() * y.size(0)
			count += y.size(0)
			f1.update(preds, y)
			f1_per_class.update(preds, y)
			kappa.update(preds, y)
			if ece is not None:
				ece.update(probs, y)
			updated = True
	avg_loss = total_loss / max(count, 1)
	if updated:
		macro_val = float(f1.compute().cpu())
		per_class = [float(x) for x in f1_per_class.compute().cpu().tolist()]
		metrics = {
			'macro_f1': macro_val,
			'per_class_f1': per_class,
			'qwk': float(kappa.compute().cpu()),
		}
		if ece is not None:
			metrics['ece'] = float(ece.compute().cpu())
	else:
		metrics = {'macro_f1': float('nan'), 'qwk': float('nan')}
		if ece is not None:
			metrics['ece'] = float('nan')
	return avg_loss, metrics


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--csv', default='/anvil/scratch/x-jliu7/test_stage2/stage2_samples_floods.csv')
	parser.add_argument('--out_dir', default='/anvil/scratch/x-jliu7/outputs_stage2')
	parser.add_argument('--epochs', type=int, default=1)
	parser.add_argument('--batch_size', type=int, default=16)
	parser.add_argument('--lr', type=float, default=2e-4)
	parser.add_argument('--weight_decay', type=float, default=0.05)
	parser.add_argument('--num_workers', type=int, default=1)
	parser.add_argument('--classes', type=int, default=4)
	parser.add_argument('--class_balance', action='store_true', help='Enable class-balanced sampling for training')
	parser.add_argument('--balance_alpha', type=float, default=0.5, help='Exponent alpha for tempered inverse-frequency weights (0..1)')
	parser.add_argument('--balance_cap', type=float, default=12.0, help='Max weight ratio cap (max/min). Set <=0 to disable.')
	parser.add_argument('--backbone', default='convnext_tiny')
	parser.add_argument('--limit', type=int, default=256, help='Limit rows for quick test. 0 means use all.')
	parser.add_argument('--seed', type=int, default=42)
	args = parser.parse_args()

	os.makedirs(args.out_dir, exist_ok=True)
	set_seed(args.seed)
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	train_loader, val_loader = make_loaders(
		args.csv, args.batch_size, args.num_workers,
		limit=args.limit,
		class_balance=bool(args.class_balance),
		num_classes=args.classes,
		balance_alpha=args.balance_alpha,
		balance_cap=args.balance_cap,
	)
	model = SiameseDamageModel(backbone_name=args.backbone, num_classes=args.classes).to(device)
	if device.type == 'cuda':
		model = model.to(memory_format=torch.channels_last)
	optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

	print(f"train_size={len(train_loader.dataset)} val_size={len(val_loader.dataset)} batches={len(train_loader)}", flush=True)
	best_f1 = -1.0
	for epoch in range(1, args.epochs + 1):
		print(f"\n=== Epoch {epoch}/{args.epochs} ===", flush=True)
		start = time.time()
		train_loss = train_one_epoch(model, train_loader, optimizer, device, args.classes, epoch)
		val_loss, metrics = evaluate(model, val_loader, device, args.classes)
		elapsed = time.time() - start
		per_str = ''
		if 'per_class_f1' in metrics:
			per_vals = ','.join(f"{v:.4f}" for v in metrics['per_class_f1'])
			per_str = f" per_class_f1=[{per_vals}]"
		print(
			f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} macro_f1={metrics['macro_f1']:.4f} qwk={metrics['qwk']:.4f} "
			+ (f"ece={metrics['ece']:.4f} " if 'ece' in metrics else '')
			+ per_str,
			flush=True,
		)
		print(f"epoch={epoch} time_sec={elapsed:.1f}", flush=True)
		# save best
		if (not np.isnan(metrics['macro_f1'])) and metrics['macro_f1'] > best_f1:
			best_f1 = metrics['macro_f1']
			ckpt_path = os.path.join(args.out_dir, 'stage2_best.pt')
			torch.save({'model': model.state_dict(), 'metrics': metrics, 'epoch': epoch}, ckpt_path)
			print(f"saved {ckpt_path}", flush=True)

	# final save
	ckpt_last = os.path.join(args.out_dir, 'stage2_last.pt')
	torch.save({'model': model.state_dict()}, ckpt_last)
	print(f"saved {ckpt_last}")


if __name__ == '__main__':
	main()


