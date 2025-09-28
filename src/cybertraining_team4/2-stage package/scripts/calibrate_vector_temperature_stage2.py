#!/usr/bin/env python3
import argparse
import os
import sys
from typing import List, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

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


def apply_vector_temperature_probs(probs: torch.Tensor, alphas: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Apply class-wise temperature in probability space: p'_c ∝ p_c^{alpha_c}."""
    # Clamp to avoid zeros, raise to power, then renormalize
    p = probs.clamp_min(eps)
    p_pow = p.pow(alphas.view(1, -1))
    p_norm = p_pow / p_pow.sum(dim=1, keepdim=True).clamp_min(eps)
    return p_norm


def nll_from_probs(probs: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> float:
    true_p = probs.gather(1, y.view(-1, 1)).clamp_min(eps).squeeze(1)
    return float((-true_p.log()).mean().item())


def ece_from_probs(probs: torch.Tensor, y: torch.Tensor, num_bins: int = 15) -> float:
    try:
        from torchmetrics.classification import MulticlassCalibrationError
    except Exception:
        return float('nan')
    K = probs.size(1)
    metric = MulticlassCalibrationError(num_classes=K, n_bins=num_bins).to(probs.device)
    metric.update(probs, y)
    return float(metric.compute().item())


def main():
    p = argparse.ArgumentParser(description='Calibrate class-wise vector temperature (probability space) via greedy grid-search')
    p.add_argument('--csv', required=True)
    p.add_argument('--ckpt_in', required=True)
    p.add_argument('--ckpt_out', required=True)
    p.add_argument('--backbone', default='convnext_tiny')
    p.add_argument('--classes', type=int, default=4)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--num_workers', type=int, default=1)
    p.add_argument('--val_frac', type=float, default=0.2)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--objective', choices=['nll', 'ece'], default='nll')
    p.add_argument('--n_bins', type=int, default=15, help='Bins for ECE if objective=ecc')
    p.add_argument('--passes', type=int, default=3, help='Coordinate grid-search passes over classes')
    p.add_argument('--candidates', type=str, default='0.6,0.8,1.0,1.2,1.5', help='Comma-separated alpha candidates per class')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ds = Stage2Dataset(args.csv)
    _, val_idx = split_by_tile(ds.rows, val_frac=args.val_frac, seed=args.seed)
    val_ds = Subset(ds, val_idx)
    loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # Build model and load checkpoint
    model = SiameseDamageModel(backbone_name=args.backbone, num_classes=args.classes).to(device)
    with torch.no_grad():
        pre0, post0, m0, r0, _y0 = ds[0]
        _ = model(pre0.unsqueeze(0).to(device), post0.unsqueeze(0).to(device), m0.unsqueeze(0).to(device), r0.unsqueeze(0).to(device))
    ckpt = torch.load(args.ckpt_in, map_location=device)
    if isinstance(ckpt, dict) and 'model' in ckpt:
        model.load_state_dict(ckpt['model'], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)

    # Apply scalar temperature if present; vector will be applied post-hoc on probs
    calib_prev = ckpt.get('calibration', {}) if isinstance(ckpt, dict) else {}
    T_scalar = calib_prev.get('temperature', None)
    if T_scalar is not None:
        with torch.no_grad():
            model.temperature.copy_(torch.tensor(float(T_scalar), device=model.temperature.device))

    model.eval()

    # Cache validation probabilities and labels
    all_probs = []
    all_y = []
    with torch.no_grad():
        for pre, post, m, r, y in loader:
            out = model(pre.to(device), post.to(device), m.to(device), r.to(device))
            all_probs.append(out['probs'].detach())
            all_y.append(y.to(device))
    probs = torch.cat(all_probs, dim=0)
    y = torch.cat(all_y, dim=0)

    K = args.classes
    alphas = torch.ones(K, device=probs.device)
    cands = [float(x) for x in args.candidates.split(',') if x.strip()]

    def objective_value(a_vec: torch.Tensor) -> float:
        p_adj = apply_vector_temperature_probs(probs, a_vec)
        if args.objective == 'nll':
            return nll_from_probs(p_adj, y)
        else:
            return ece_from_probs(p_adj, y, num_bins=args.n_bins)

    base_val = objective_value(alphas)
    print(f"Initial {args.objective}={base_val:.6f} | alphas={alphas.tolist()}")

    for it in range(args.passes):
        improved = False
        for c in range(K):
            best_alpha = float(alphas[c].item())
            best_val = base_val
            for cand in cands:
                if cand == best_alpha:
                    continue
                trial = alphas.clone()
                trial[c] = cand
                val = objective_value(trial)
                if val < best_val:
                    best_val = val
                    best_alpha = cand
            if best_alpha != float(alphas[c].item()):
                alphas[c] = best_alpha
                base_val = best_val
                improved = True
                print(f"pass={it+1} class={c} -> alpha={best_alpha:.3f} | {args.objective}={best_val:.6f}")
        if not improved:
            print(f"No improvement in pass {it+1}; stopping early.")
            break

    # Save calibrated checkpoint with vector temperature
    save_ckpt = ckpt if isinstance(ckpt, dict) else {'model': model.state_dict()}
    calib_out = dict(calib_prev)
    calib_out['vector_temperature'] = [float(x) for x in alphas.tolist()]
    calib_out['objective'] = args.objective
    save_ckpt['calibration'] = calib_out
    torch.save(save_ckpt, args.ckpt_out)
    print(f"Saved vector-calibrated checkpoint to {args.ckpt_out} with alphas={calib_out['vector_temperature']}")


if __name__ == '__main__':
    main()


