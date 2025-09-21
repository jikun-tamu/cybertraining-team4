#!/usr/bin/env python3
# train.py
from pathlib import Path
import json, random
import numpy as np
import cv2
from PIL import Image
from shapely import wkt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import timm
import argparse

# -------------------- data utils --------------------
IMG_DIR = Path("images")
LBL_DIR = Path("labels")

def is_pre(path: Path):
    return "_pre_disaster" in path.name

def paired_pre_files(img_dir=IMG_DIR, lbl_dir=LBL_DIR):
    imgs = sorted([p for p in img_dir.glob("*.png") if is_pre(p)])
    pairs = []
    for ip in imgs:
        jp = lbl_dir / ip.with_suffix(".json").name
        if jp.exists() and is_pre(jp):
            pairs.append((ip, jp))
    return pairs

def load_img(path: Path):
    return np.array(Image.open(path).convert("RGB"))

def mask_from_json(json_path: Path, image_shape):
    meta = json.loads(json_path.read_text())
    H, W = image_shape[:2]
    mask = np.zeros((H, W), dtype=np.uint8)
    for f in meta["features"]["xy"]:
        poly = wkt.loads(f["wkt"])
        pts = np.array(poly.exterior.coords, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 1)
    return mask

def generate_windows(H, W, tile=512, stride=384):
    ys = list(range(0, max(1, H - tile + 1), stride)) or [0]
    xs = list(range(0, max(1, W - tile + 1), stride)) or [0]
    for y in ys:
        for x in xs:
            yield slice(y, y + tile), slice(x, x + tile)

class FootprintDataset(Dataset):
    def __init__(self, pairs, tile=512, stride=384, augment=True, oversample_pos=True, cache_masks=True):
        self.tile, self.stride = tile, stride
        self.augment, self.oversample_pos = augment, oversample_pos
        self.records = []
        self.T = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.Normalize(), ToTensorV2(),
        ]) if augment else A.Compose([A.Normalize(), ToTensorV2()])

        for ip, jp in pairs:
            img = load_img(ip)
            H, W = img.shape[:2]
            mask = mask_from_json(jp, img.shape) if cache_masks else None
            pos, neg = [], []
            for ys, xs in generate_windows(H, W, tile, stride):
                m = (mask[ys, xs] if cache_masks else mask_from_json(jp, img.shape)[ys, xs])
                (pos if m.any() else neg).append((ys, xs))
            self.records.append(dict(ip=ip, jp=jp, H=H, W=W, mask=mask, pos=pos, neg=neg))

    def __len__(self):
        # stochastic sampling; return a reasonable epoch size
        return max(1, sum(max(1, len(r["pos"]) + len(r["neg"])) for r in self.records))

    def __getitem__(self, _):
        r = random.choice(self.records)
        use_pos = (len(r["pos"]) > 0) and (self.oversample_pos or random.random() < 0.5)
        win_list = r["pos"] if (use_pos and r["pos"]) else (r["neg"] or r["pos"])
        ys, xs = random.choice(win_list)
        img = load_img(r["ip"])[ys, xs]
        msk = (r["mask"][ys, xs] if r["mask"] is not None else mask_from_json(r["jp"], (r["H"], r["W"]))[ys, xs])
        data = self.T(image=img, mask=msk)
        x = data["image"].float()                 # (3,H,W), normalized
        y = data["mask"].unsqueeze(0).float()     # (1,H,W)
        return x, y

def split_pairs(pairs, val_ratio=0.1, seed=42):
    rng = random.Random(seed)
    pairs = pairs.copy(); rng.shuffle(pairs)
    nv = max(1, int(len(pairs) * val_ratio))
    return pairs[nv:], pairs[:nv]

# -------------------- model/loss/metric --------------------
def make_model():
    return smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1)

def dice_loss(logits, target, eps=1e-6):
    p = torch.sigmoid(logits)
    num = 2 * (p * target).sum(dim=(2, 3)) + eps
    den = p.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + eps
    return 1 - (num / den).mean()

bce = nn.BCEWithLogitsLoss()
def total_loss(logits, target):
    return 0.5 * bce(logits, target) + 0.5 * dice_loss(logits, target)

@torch.no_grad()
def iou_metric(logits, target, thr=0.5, eps=1e-6):
    p = (torch.sigmoid(logits) > thr).float()
    inter = (p * target).sum(dim=(2, 3))
    union = p.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - inter + eps
    return (inter / union).mean().item()

# -------------------- training --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--stride", type=int, default=384)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--amp", action="store_true")
    args = ap.parse_args()

    random.seed(42); np.random.seed(42); torch.manual_seed(42)

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    pairs = paired_pre_files()
    if not pairs: raise RuntimeError("No (image,json) pairs found under images/ and labels/ matching *_pre_disaster*.png/.json")
    train_pairs, val_pairs = split_pairs(pairs, 0.1, seed=42)

    train_ds = FootprintDataset(train_pairs, tile=args.tile, stride=args.stride, augment=True,  oversample_pos=True)
    val_ds   = FootprintDataset(val_pairs,   tile=args.tile, stride=args.stride, augment=False, oversample_pos=False)

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=args.workers, pin_memory=False)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=False)

    model = make_model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.startswith("cuda")))

    print(f"Found pairs: {len(pairs)} | device: {device}")

    best_iou = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running, seen, step = 0.0, 0, 0
        for xb, yb in train_dl:
            step += 1
            xb, yb = xb.to(device), yb.to(device)
            if yb.ndim == 3: yb = yb.unsqueeze(1).float()

            opt.zero_grad(set_to_none=True)
            if device.startswith("cuda"): torch.cuda.synchronize()
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                logits = model(xb)
                assert logits.shape[-2:] == yb.shape[-2:], f"spatial mismatch: {logits.shape} vs {yb.shape}"
                loss = total_loss(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()

            running += loss.item() * xb.size(0)
            seen    += xb.size(0)
            if step >= args.steps: break

        train_loss = running / max(1, seen)

        model.eval()
        ious = []
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                if yb.ndim == 3: yb = yb.unsqueeze(1).float()
                logits = model(xb)
                ious.append(float(iou_metric(logits, yb)))
        miou = float(np.mean(ious)) if ious else 0.0

        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_mIoU={miou:.4f}")

        if miou > best_iou:
            best_iou = miou
            torch.save(model.state_dict(), "best_unet.pt")

    print("Best IoU:", best_iou)

if __name__ == "__main__":
    # Better C++ traces if anything trips
    import os
    os.environ.setdefault("PYTORCH_SHOW_CPP_STACKTRACES", "1")
    main()
