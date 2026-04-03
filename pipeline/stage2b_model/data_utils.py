"""Stage 2b data loading utilities — copied from pipeline/scripts/train_stage2.py.

Source lines: 36-37 (constants), 245-253 (read_rows), 273-283 (load tensors), 437-447 (collate)
DO NOT modify without also updating train_stage2.py.
"""

import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def read_rows(csv_path, limit=0):
    rows = []
    with Path(csv_path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if limit > 0 and i >= limit:
                break
            rows.append(row)
    return rows


def load_rgb_tensor(path):
    arr = np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # HWC -> CHW
    t = (t - IMAGENET_MEAN) / IMAGENET_STD
    return t


def load_mask_tensor(path):
    arr = np.array(Image.open(path).convert("L"), dtype=np.uint8)
    m = (arr > 0).astype(np.float32)
    return torch.from_numpy(m).unsqueeze(0).contiguous()  # [1,H,W]


def collate_batch(batch):
    pre, post, m, r, y, meta, sample_id = zip(*batch)
    return (
        torch.stack(pre, dim=0),
        torch.stack(post, dim=0),
        torch.stack(m, dim=0),
        torch.stack(r, dim=0),
        torch.tensor(y, dtype=torch.long),
        meta,
        torch.tensor(sample_id, dtype=torch.long),
    )
