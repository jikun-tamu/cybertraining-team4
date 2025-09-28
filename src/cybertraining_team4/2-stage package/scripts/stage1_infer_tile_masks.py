#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


# Add the parent directory of the scripts folder to the path, so it can find stage1_train
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)


def list_pre_images(images_dir: str) -> List[Path]:
    images_dir_p = Path(images_dir)
    # Find all .tif files within any 'pre' subdirectory
    return sorted(images_dir_p.rglob("pre/*.tif"))


def load_stage1_model(ckpt_path: str):
    from stage1_train import CorrectedBuildingSegmentationModel  # lazy import

    model = CorrectedBuildingSegmentationModel(model_path=ckpt_path)
    # Load weights directly from the provided path
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.model.load_state_dict(checkpoint['model_state_dict'])
    model.model.eval()
    return model


def predict_mask(model, image_path: str, thresh: float = 0.5) -> np.ndarray:
    pred = model.predict(image_path)  # float in [0,1] at original size
    pred_bin = (pred >= float(thresh)).astype(np.uint8)
    return pred_bin


def postprocess_mask(mask: np.ndarray, min_area: int = 50, open_ksize: int = 3) -> np.ndarray:
    if open_ksize and open_ksize > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    # remove small components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    out = np.zeros_like(mask, dtype=np.uint8)
    for lbl in range(1, num_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area >= int(min_area):
            out[labels == lbl] = 1
    return out


def save_mask(mask: np.ndarray, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask * 255).astype(np.uint8)).save(out_path)


def main():
    parser = argparse.ArgumentParser(description="Batch Stage-1 inference to generate pre-image building masks")
    parser.add_argument('--images_dir', required=True, help='Directory with tile images (*.png)')
    parser.add_argument('--ckpt', required=True, help='Path to corrected_building_segmentation_model.pth')
    parser.add_argument('--out_dir', required=True, help='Directory to write predicted masks')
    parser.add_argument('--thresh', type=float, default=0.5, help='Threshold for binarizing probability mask')
    parser.add_argument('--min_area', type=int, default=50, help='Remove components smaller than this (pixels)')
    parser.add_argument('--open_ksize', type=int, default=3, help='Morphological opening kernel size (0/1 disables)')
    parser.add_argument('--no_progress', action='store_true', help='Disable progress bar output')
    args = parser.parse_args()

    images = list_pre_images(args.images_dir)
    assert len(images) > 0, f"No pre-disaster images found under {args.images_dir}"

    model = load_stage1_model(args.ckpt)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped = 0
    it = images
    if not args.no_progress:
        it = tqdm(images, total=len(images), desc='Stage1 infer masks', unit='img')
    for img_path in it:
        try:
            base = img_path.name.replace('_pre_disaster.png', '')
            out_path = out_dir / f"{base}_pre_mask.png"
            pred = predict_mask(model, str(img_path), thresh=args.thresh)
            pred = postprocess_mask(pred, min_area=args.min_area, open_ksize=args.open_ksize)
            save_mask(pred, out_path)
            processed += 1
        except Exception:
            skipped += 1
        if not args.no_progress:
            try:
                it.set_postfix({"ok": processed, "skip": skipped})
            except Exception:
                pass
    print(f"Wrote masks to {out_dir} | processed={processed} | skipped={skipped}")


if __name__ == '__main__':
    sys.path.append(os.path.dirname(os.path.realpath(__file__)))
    main()


