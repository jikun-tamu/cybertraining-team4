#!/usr/bin/env python3
"""Verify that stage2b_model and train_stage2 produce identical results.

Run from pipeline/ directory:
    conda run -n geoai_sam python stage2b_model/test_sync.py

This test catches accidental divergence between the extracted module
(stage2b_model/) and the monolithic training script (scripts/train_stage2.py).
"""

import json
import sys
from pathlib import Path

import torch

# ── Setup paths ──────────────────────────────────────────────────────────────
PKG_ROOT = Path(__file__).resolve().parents[1]  # pipeline/
sys.path.insert(0, str(PKG_ROOT))
sys.path.insert(0, str(PKG_ROOT / "scripts"))

# ── Import both sources ──────────────────────────────────────────────────────
import stage2b_model as extracted
import train_stage2 as original


def test_coral_probs():
    """coral_probs_from_logits must be identical."""
    torch.manual_seed(0)
    logits = torch.randn(20, 3)
    p1 = original.coral_probs_from_logits(logits)
    p2 = extracted.coral_probs_from_logits(logits)
    assert torch.allclose(p1, p2, atol=1e-7), f"coral_probs mismatch: max diff {(p1-p2).abs().max()}"
    print("  [PASS] coral_probs_from_logits")


def test_coral_targets():
    """coral_targets must be identical (no label smoothing)."""
    y = torch.tensor([0, 1, 2, 3, 0, 3])
    t1 = original.coral_targets(y, 4)
    t2 = extracted.coral_targets(y, 4)
    assert torch.equal(t1, t2), "coral_targets mismatch"
    print("  [PASS] coral_targets")


def test_model_forward():
    """SiameseDamageModel forward pass must be identical."""
    cfg_path = PKG_ROOT / "configs/stage2b/run019_seed2025_train_config.json"
    with open(cfg_path) as f:
        cfg = json.load(f)

    kwargs = dict(
        backbone_name=cfg["backbone"],
        num_classes=cfg["classes"],
        hidden_dim=cfg["hidden_dim"],
        dropout=cfg["dropout"],
        pretrained=False,
        change_fusion=cfg.get("change_fusion", "pre_post_diff"),
        pooling_mode=cfg.get("pooling_mode", "mask_m_ring"),
    )

    m1 = original.SiameseDamageModel(**kwargs)
    m2 = extracted.SiameseDamageModel(**kwargs)
    m2.load_state_dict(m1.state_dict())

    m1.eval()
    m2.eval()

    torch.manual_seed(42)
    pre = torch.randn(2, 3, 256, 256)
    post = torch.randn(2, 3, 256, 256)
    mask = (torch.randn(2, 1, 256, 256) > 0).float()
    ring = (torch.randn(2, 1, 256, 256) > 0).float()

    with torch.no_grad():
        o1 = m1(pre, post, mask, ring)
        o2 = m2(pre, post, mask, ring)

    assert torch.allclose(o1["logits_cum"], o2["logits_cum"], atol=1e-6), \
        f"logits mismatch: max diff {(o1['logits_cum']-o2['logits_cum']).abs().max()}"
    assert torch.allclose(o1["probs"], o2["probs"], atol=1e-6), \
        f"probs mismatch: max diff {(o1['probs']-o2['probs']).abs().max()}"
    print("  [PASS] SiameseDamageModel forward (logits + probs)")


def test_real_checkpoint():
    """Load a real checkpoint through extracted module."""
    cfg_path = PKG_ROOT / "configs/stage2b/run019_seed2025_train_config.json"
    ckpt_path = PKG_ROOT / "models/stage2b/inference0.7273.pt"

    if not ckpt_path.exists():
        print("  [SKIP] real checkpoint test (file not found)")
        return

    with open(cfg_path) as f:
        cfg = json.load(f)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt

    model = extracted.SiameseDamageModel(
        backbone_name=cfg["backbone"],
        num_classes=cfg["classes"],
        hidden_dim=cfg["hidden_dim"],
        dropout=cfg["dropout"],
        pretrained=False,
        change_fusion=cfg.get("change_fusion", "pre_post_diff"),
        pooling_mode=cfg.get("pooling_mode", "mask_m_ring"),
    )
    model.load_state_dict(state)
    model.eval()

    torch.manual_seed(99)
    pre = torch.randn(1, 3, 256, 256)
    post = torch.randn(1, 3, 256, 256)
    mask = (torch.randn(1, 1, 256, 256) > 0).float()
    ring = (torch.randn(1, 1, 256, 256) > 0).float()

    with torch.no_grad():
        out = model(pre, post, mask, ring)

    assert out["probs"].shape == (1, 4), f"unexpected probs shape: {out['probs'].shape}"
    assert abs(out["probs"].sum().item() - 1.0) < 1e-5, f"probs don't sum to 1: {out['probs'].sum()}"
    print("  [PASS] real checkpoint load + forward")


def test_data_utils():
    """read_rows, IMAGENET constants must match."""
    assert torch.equal(original.IMAGENET_MEAN, extracted.IMAGENET_MEAN), "IMAGENET_MEAN mismatch"
    assert torch.equal(original.IMAGENET_STD, extracted.IMAGENET_STD), "IMAGENET_STD mismatch"
    print("  [PASS] IMAGENET constants")


def main():
    print("stage2b_model sync test")
    print("=" * 50)
    test_coral_probs()
    test_coral_targets()
    test_data_utils()
    test_model_forward()
    test_real_checkpoint()
    print("=" * 50)
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
