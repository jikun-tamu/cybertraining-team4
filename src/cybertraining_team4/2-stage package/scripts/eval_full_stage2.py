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


def split_train_val_test_by_tile(rows: List[Dict[str, str]], val_frac: float = 0.2, test_frac: float = 0.2, seed: int = 42) -> Tuple[List[int], List[int], List[int]]:
    import random
    assert 0 < val_frac < 1 and 0 < test_frac < 1 and val_frac + test_frac < 1
    tiles = sorted({r['tile_id'] for r in rows})
    random.Random(seed).shuffle(tiles)
    n = len(tiles)
    n_train = int(round(n * (1 - val_frac - test_frac)))
    n_val = int(round(n * val_frac))
    train_tiles = set(tiles[:n_train])
    val_tiles = set(tiles[n_train:n_train + n_val])
    test_tiles = set(tiles[n_train + n_val:])
    tr, va, te = [], [], []
    for i, r in enumerate(rows):
        if r['tile_id'] in train_tiles:
            tr.append(i)
        elif r['tile_id'] in val_tiles:
            va.append(i)
        else:
            te.append(i)
    return tr, va, te


def apply_vector_temperature_probs(probs: torch.Tensor, alphas: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    p = probs.clamp_min(eps)
    p_pow = p.pow(alphas.view(1, -1))
    return p_pow / p_pow.sum(dim=1, keepdim=True).clamp_min(eps)


def evaluate(ds: Stage2Dataset, indices: List[int], ckpt_path: str, classes: int, backbone: str, n_bins_list = (10, 15, 20)) -> Dict[str, float]:
    from sklearn.metrics import f1_score, cohen_kappa_score
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loader = DataLoader(Subset(ds, indices), batch_size=128, shuffle=False, num_workers=1, pin_memory=True)

    # Build and load model
    model = SiameseDamageModel(backbone_name=backbone, num_classes=classes).to(device)
    with torch.no_grad():
        pre0, post0, m0, r0, _y0 = ds[0]
        _ = model(pre0.unsqueeze(0).to(device), post0.unsqueeze(0).to(device), m0.unsqueeze(0).to(device), r0.unsqueeze(0).to(device))
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    calib = ckpt.get('calibration', {}) if isinstance(ckpt, dict) else {}
    T = calib.get('temperature', None)
    if T is not None:
        with torch.no_grad():
            model.temperature.copy_(torch.tensor(float(T), device=model.temperature.device))
    vT = calib.get('vector_temperature', None)
    vT = [float(x) for x in vT] if isinstance(vT, (list, tuple)) else None

    ys = []
    probs_list = []
    with torch.no_grad():
        for pre, post, m, r, y in loader:
            out = model(pre.to(device), post.to(device), m.to(device), r.to(device))
            p = out['probs']
            if vT is not None:
                p = apply_vector_temperature_probs(p, torch.tensor(vT, device=p.device))
            probs_list.append(p.cpu())
            ys.append(y)

    P = torch.cat(probs_list, dim=0)
    Y = torch.cat(ys, dim=0).long()
    pred = P.argmax(dim=1)

    # Metrics
    macro_f1 = float(__import__('sklearn.metrics').metrics.f1_score(Y.numpy(), pred.numpy(), average='macro'))
    qwk = float(__import__('sklearn.metrics').metrics.cohen_kappa_score(Y.numpy(), pred.numpy(), weights='quadratic'))
    # NLL
    eps = 1e-12
    true_p = P.gather(1, Y.view(-1, 1)).clamp_min(eps).squeeze(1)
    nll = float((-true_p.log()).mean().item())
    # ECE (multiple bin counts)
    try:
        from torchmetrics.classification import MulticlassCalibrationError
        eces = {}
        for nb in n_bins_list:
            e = MulticlassCalibrationError(num_classes=classes, n_bins=nb)
            e.update(P, Y)
            eces[f'ece_{nb}'] = float(e.compute().item())
    except Exception:
        eces = {f'ece_{nb}': float('nan') for nb in n_bins_list}
    # Tolerant accuracy by true class (|pred-gt|<=1)
    K = classes
    totals = [0]*K
    tol_hits = [0]*K
    for y_i, p_i in zip(Y.tolist(), pred.tolist()):
        if 0 <= y_i < K:
            totals[y_i] += 1
            if abs(p_i - y_i) <= 1:
                tol_hits[y_i] += 1
    tol = {f'tolerant_acc_class_{c}': (tol_hits[c]/totals[c] if totals[c] else float('nan')) for c in range(K)}

    out = {'macro_f1': macro_f1, 'qwk': qwk, 'nll': nll, **eces, **tol}
    return out


def main():
    ap = argparse.ArgumentParser(description='Full evaluation: uncalibrated vs vector-calibrated on held-out test split')
    ap.add_argument('--csv', required=True)
    ap.add_argument('--ckpt_uncal', required=True)
    ap.add_argument('--ckpt_vec', required=True)
    ap.add_argument('--backbone', default='convnext_tiny')
    ap.add_argument('--classes', type=int, default=4)
    ap.add_argument('--val_frac', type=float, default=0.2)
    ap.add_argument('--test_frac', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    ds = Stage2Dataset(args.csv)
    tr_idx, va_idx, te_idx = split_train_val_test_by_tile(ds.rows, val_frac=args.val_frac, test_frac=args.test_frac, seed=args.seed)

    print(f"Split sizes (by tiles): train={len(tr_idx)} val={len(va_idx)} test={len(te_idx)}")

    r_uncal = evaluate(ds, te_idx, args.ckpt_uncal, args.classes, args.backbone)
    r_vec   = evaluate(ds, te_idx, args.ckpt_vec,   args.classes, args.backbone)

    def fmt(d):
        keys = ['macro_f1','qwk','nll','ece_10','ece_15','ece_20']
        return ' '.join([f"{k}={d.get(k, float('nan')):.4f}" for k in keys])

    print("\nUncalibrated:")
    print(fmt(r_uncal))
    print("Vector-calibrated:")
    print(fmt(r_vec))

    print("\nPer-class tolerant accuracy (|pred-gt|<=1):")
    for c in range(args.classes):
        a = r_uncal.get(f'tolerant_acc_class_{c}', float('nan'))
        b = r_vec.get(f'tolerant_acc_class_{c}', float('nan'))
        print(f"class={c} uncal={a:.4f} vec={b:.4f}")


if __name__ == '__main__':
    main()


