#!/usr/bin/env python3
import argparse
import os
import sys
from typing import Tuple, List, Dict

import torch
from torch.utils.data import DataLoader, Subset
from torchmetrics.classification import MulticlassCalibrationError

a = '/anvil/scratch/x-jliu7'
if a not in sys.path:
    sys.path.append(a)

from src.data.stage2_dataset import Stage2Dataset
from src.models.siamese_stage2 import SiameseDamageModel


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
    p = argparse.ArgumentParser(description='Evaluate ECE for a Stage-2 checkpoint (supports calibrated temperature)')
    p.add_argument('--csv', default='/anvil/scratch/x-jliu7/test_stage2/stage2_samples_floods.csv')
    p.add_argument('--ckpt', required=True)
    p.add_argument('--backbone', default='convnext_tiny')
    p.add_argument('--classes', type=int, default=4)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--num_workers', type=int, default=1)
    p.add_argument('--n_bins', type=int, default=15)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ds = Stage2Dataset(args.csv)
    _, val_idx = split_by_tile(ds.rows)
    val_ds = Subset(ds, val_idx)
    loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = SiameseDamageModel(backbone_name=args.backbone, num_classes=args.classes).to(device)
    # Warm build
    with torch.no_grad():
        pre0, post0, m0, r0, _y0 = ds[0]
        _ = model(pre0.unsqueeze(0).to(device), post0.unsqueeze(0).to(device), m0.unsqueeze(0).to(device), r0.unsqueeze(0).to(device))
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt['model'], strict=False)
    calib = ckpt.get('calibration', {}) if isinstance(ckpt, dict) else {}
    T = calib.get('temperature', None)
    if T is not None:
        with torch.no_grad():
            model.temperature.copy_(torch.tensor(float(T), device=model.temperature.device))
    vector_T = calib.get('vector_temperature', None)
    if isinstance(vector_T, (list, tuple)):
        vector_T = [float(x) for x in vector_T]
    model.eval()

    ece = MulticlassCalibrationError(num_classes=args.classes, n_bins=args.n_bins).to(device)
    with torch.no_grad():
        for pre, post, m, r, y in loader:
            out = model(pre.to(device), post.to(device), m.to(device), r.to(device))
            probs = out['probs']
            if vector_T is not None:
                p = probs.clamp_min(1e-12)
                alphas = torch.tensor(vector_T, device=p.device).view(1, -1)
                p = p.pow(alphas)
                probs = p / p.sum(dim=1, keepdim=True).clamp_min(1e-12)
            ece.update(probs, y.to(device))
    ece_val = float(ece.compute().cpu())
    print(f"ECE={ece_val:.4f} (bins={args.n_bins}) | ckpt={args.ckpt}")


if __name__ == '__main__':
    main()


