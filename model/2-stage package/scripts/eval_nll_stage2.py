#!/usr/bin/env python3
import argparse
import os
import sys
from typing import Tuple, List, Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

a = '/anvil/scratch/x-jliu7'
if a not in sys.path:
    sys.path.append(a)

from src.data.stage2_dataset import Stage2Dataset
from src.models.siamese_stage2 import SiameseDamageModel
from src.models.coral_head import coral_probs_from_logits


def split_by_tile(rows: List[Dict[str, str]], val_frac: float = 0.2, seed: int = 42) -> Tuple[List[int], List[int]]:
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


def main():
    parser = argparse.ArgumentParser(description='Compute NLL (Negative Log-Likelihood) on validation split')
    parser.add_argument('--csv', required=True)
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--backbone', default='convnext_tiny')
    parser.add_argument('--classes', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--val_frac', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ds = Stage2Dataset(args.csv)
    _, val_idx = split_by_tile(ds.rows, val_frac=args.val_frac, seed=args.seed)
    val_ds = Subset(ds, val_idx)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    model = SiameseDamageModel(backbone_name=args.backbone, num_classes=args.classes).to(device)
    # Warm build the head with a single forward to ensure state_dict keys match
    with torch.no_grad():
        pre0, post0, m0, r0, _y0 = ds[0]
        _ = model(pre0.unsqueeze(0).to(device), post0.unsqueeze(0).to(device), m0.unsqueeze(0).to(device), r0.unsqueeze(0).to(device))
    # Safe load
    try:
        ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)  # type: ignore
    except TypeError:
        ckpt = torch.load(args.ckpt, map_location=device)
    state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    model.load_state_dict(state, strict=False)

    # Apply stored temperature if present
    calib = ckpt.get('calibration', {}) if isinstance(ckpt, dict) else {}
    T = calib.get('temperature', None)
    if T is not None:
        with torch.no_grad():
            model.temperature.copy_(torch.tensor(float(T), device=model.temperature.device))
    vector_T = calib.get('vector_temperature', None)
    if isinstance(vector_T, (list, tuple)):
        vector_T = [float(x) for x in vector_T]

    model.eval()

    nll_sum = 0.0
    n_count = 0
    eps = 1e-12
    with torch.no_grad():
        for pre, post, m, r, y in val_loader:
            pre = pre.to(device, non_blocking=True)
            post = post.to(device, non_blocking=True)
            m = m.to(device, non_blocking=True)
            r = r.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            out = model(pre, post, m, r)
            probs = out['probs']  # [B, C]
            if vector_T is not None:
                p = probs.clamp_min(1e-12)
                alphas = torch.tensor(vector_T, device=p.device).view(1, -1)
                p = p.pow(alphas)
                probs = p / p.sum(dim=1, keepdim=True).clamp_min(1e-12)
            # Gather true-class probabilities
            true_p = probs.gather(1, y.view(-1, 1)).clamp_min(eps).squeeze(1)
            # NLL per sample = -log p_true
            nll_batch = -true_p.log()
            nll_sum += float(nll_batch.sum().item())
            n_count += y.numel()

    nll = nll_sum / max(n_count, 1)
    print(f"val_size={n_count} NLL={nll:.4f}")


if __name__ == '__main__':
    main()


