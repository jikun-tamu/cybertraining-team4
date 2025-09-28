#!/usr/bin/env python3
import argparse
import csv
import os
import sys
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

a = '/anvil/scratch/x-jliu7'
if a not in sys.path:
    sys.path.append(a)

from src.data.stage2_dataset import Stage2Dataset
from src.models.siamese_stage2 import SiameseDamageModel
from src.models.coral_head import coral_targets, coral_probs_from_logits


def read_rows(csv_path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def split_by_tile(rows: List[Dict[str, str]], val_frac: float = 0.2, seed: int = 42) -> Tuple[List[int], List[int]]:
    import random
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


def nll_coral(logits_cum: torch.Tensor, y: torch.Tensor, num_classes: int) -> torch.Tensor:
    # Negative log-likelihood for CORAL cumulative logits
    # Convert y to cumulative binary targets and use BCEWithLogitsLoss per threshold, summed
    targets = coral_targets(y, num_classes)
    bce = nn.BCEWithLogitsLoss(reduction='none')
    loss = bce(logits_cum, targets).sum(dim=1)  # sum across thresholds
    return loss.mean()


def main():
    parser = argparse.ArgumentParser(description='Calibrate temperature scaling on validation set for Stage-2 model')
    parser.add_argument('--csv', default='/anvil/scratch/x-jliu7/test_stage2/stage2_samples_floods.csv')
    parser.add_argument('--ckpt_in', default='/anvil/scratch/x-jliu7/outputs_stage2/stage2_best.pt')
    parser.add_argument('--ckpt_out', default='/anvil/scratch/x-jliu7/outputs_stage2/stage2_best_calibrated.pt')
    parser.add_argument('--backbone', default='convnext_tiny')
    parser.add_argument('--classes', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--objective', type=str, default='categorical', choices=['categorical', 'coral'],
                        help="'categorical' minimizes NLL on class probs; 'coral' minimizes BCE over thresholds")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Build dataset and get validation split
    rows = read_rows(args.csv)
    ds = Stage2Dataset(args.csv)
    _, val_idx = split_by_tile(rows, val_frac=0.2, seed=args.seed)
    val_ds = Subset(ds, val_idx)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    print(f"Calibration split size: {len(val_ds)} (batch_size={args.batch_size})")

    # Build model and load checkpoint; warm-build head using one batch
    model = SiameseDamageModel(backbone_name=args.backbone, num_classes=args.classes).to(device)
    # Warm build
    with torch.no_grad():
        pre0, post0, m0, r0, _y0 = ds[0]
        _ = model(pre0.unsqueeze(0).to(device), post0.unsqueeze(0).to(device), m0.unsqueeze(0).to(device), r0.unsqueeze(0).to(device))
    ckpt = torch.load(args.ckpt_in, map_location=device)
    model.load_state_dict(ckpt['model'], strict=False)
    model.eval()
    print(f"Initialized temperature T={float(model.temperature.item()):.6f}")

    # Precompute validation cumulative logits once (faster calibration epochs)
    cached_logits = []
    cached_targets = []
    with torch.no_grad():
        for pre, post, m, r, y in tqdm(val_loader, total=len(val_loader), desc="Calib: computing logits", leave=False):
            pre = pre.to(device)
            post = post.to(device)
            m = m.to(device)
            r = r.to(device)
            y = y.to(device)
            out = model(pre, post, m, r)
            cached_logits.append(out['logits_cum'].detach())
            cached_targets.append(y.detach())
    logits_all = torch.cat(cached_logits, dim=0)
    y_all = torch.cat(cached_targets, dim=0)
    print(f"Cached logits for {int(logits_all.size(0))} samples; optimizing temperature...")

    # Optimize temperature (single scalar) to minimize NLL on validation using cached logits
    # We treat temperature as a positive scalar parameter
    T = torch.nn.Parameter(torch.ones(1, device=device) * float(model.temperature.item()))
    optimizer = torch.optim.LBFGS([T], lr=args.lr, max_iter=100, line_search_fn='strong_wolfe')
    last_loss = [None]

    def closure():
        optimizer.zero_grad(set_to_none=True)
        logits_T = logits_all / T.clamp_min(1e-6)
        if args.objective == 'categorical':
            # Convert to categorical probabilities and minimize NLL over true class
            probs = coral_probs_from_logits(logits_T)
            # Avoid log(0)
            eps = 1e-12
            true_probs = probs[torch.arange(y_all.size(0), device=probs.device), y_all]
            total = -(true_probs.clamp_min(eps).log()).mean()
        else:
            total = nll_coral(logits_T, y_all, args.classes)
        # store loss for progress reporting
        try:
            last_loss[0] = float(total.detach().item())
        except Exception:
            pass
        total.backward()
        return total

    for epoch in range(args.epochs):
        optimizer.step(closure)
        with torch.no_grad():
            T_cur = float(T.clamp_min(1e-6).item())
        cur_loss = last_loss[0]
        if cur_loss is not None:
            print(f"calib_epoch={epoch+1}/{args.epochs} loss={cur_loss:.6f} T={T_cur:.6f}")
        else:
            print(f"calib_epoch={epoch+1}/{args.epochs} T={T_cur:.6f}")

    # Save calibrated temperature in checkpoint
    T_value = float(T.detach().clamp_min(1e-6).item())
    # temperature is a 0-D buffer; use copy_ to assign scalar
    with torch.no_grad():
        model.temperature.copy_(torch.tensor(T_value, device=model.temperature.device))
    # Update and save checkpoint with temperature
    ckpt_out = {
        'model': model.state_dict(),
        'epoch': ckpt.get('epoch', None),
        'metrics': ckpt.get('metrics', None),
        'calibration': {'temperature': T_value},
    }
    os.makedirs(os.path.dirname(args.ckpt_out), exist_ok=True)
    torch.save(ckpt_out, args.ckpt_out)
    print(f"Saved calibrated checkpoint with temperature={T_value:.6f} to {args.ckpt_out}")


if __name__ == '__main__':
    main()


