"""Stage 2b model architecture — copied from pipeline/scripts/train_stage2.py.

Source lines: 450-585 (model), 466-478 (CoralHead), 481-503 (CORAL utils)
DO NOT modify without also updating train_stage2.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


# ── Low-level pooling ────────────────────────────────────────────────────────

def downsample_mask(mask, h, w):
    return F.interpolate(mask, size=(h, w), mode="nearest")


def masked_avg_pool(feat, mask, eps=1e-6):
    # feat: [B,C,H,W], mask: [B,1,Hm,Wm] in {0,1}
    m = downsample_mask(mask, feat.shape[-2], feat.shape[-1])
    num = (feat * m).sum(dim=(2, 3))
    den = m.sum(dim=(2, 3)).clamp_min(eps)
    return num / den


def global_avg_pool(feat):
    return feat.mean(dim=(2, 3))


# ── CORAL ordinal head ───────────────────────────────────────────────────────

class CoralHead(nn.Module):
    def __init__(self, in_dim, num_classes, hidden_dim=512, dropout=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes - 1),
        )

    def forward(self, x):
        return self.net(x)


def coral_targets(y, num_classes, label_smoothing=0.0):
    # target[k] = 1 if y > k else 0 for k in [0..K-2]
    ks = torch.arange(num_classes - 1, device=y.device).view(1, -1)
    tgt = (y.view(-1, 1) > ks).float()
    if label_smoothing > 0:
        eps = float(label_smoothing)
        tgt = tgt * (1.0 - eps) + 0.5 * eps
    return tgt


def coral_probs_from_logits(logits):
    # logits represent P(y > k) via sigmoid
    p_gt = torch.sigmoid(logits)  # [B,K-1]
    b, km1 = p_gt.shape
    k = km1 + 1
    probs = []
    probs.append(1.0 - p_gt[:, 0])
    for c in range(1, k - 1):
        probs.append(p_gt[:, c - 1] - p_gt[:, c])
    probs.append(p_gt[:, -1])
    probs = torch.stack(probs, dim=1).clamp_min(1e-8)
    probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return probs


# ── Siamese damage model ─────────────────────────────────────────────────────

class SiameseDamageModel(nn.Module):
    def __init__(
        self,
        backbone_name,
        num_classes,
        hidden_dim=512,
        dropout=0.1,
        pretrained=False,
        stage_index=-1,
        change_fusion="pre_post_diff",
        diff_abs_scale=1.0,
        pooling_mode="mask_m_ring",
    ):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
        )
        channels = self.backbone.feature_info.channels()
        if stage_index < 0:
            stage_index = len(channels) + stage_index
        if stage_index < 0 or stage_index >= len(channels):
            raise ValueError(f"Invalid stage_index {stage_index}, available [0..{len(channels)-1}]")
        self.stage_index = stage_index
        self.change_fusion = change_fusion
        self.diff_abs_scale = float(diff_abs_scale)
        self.pooling_mode = pooling_mode
        c = channels[stage_index]
        if pooling_mode not in ("rgb_only", "mask_m", "mask_m_ring"):
            raise ValueError(f"Unknown pooling_mode={pooling_mode}")
        if change_fusion not in ("legacy", "pre_post_diff"):
            raise ValueError(f"Unknown change_fusion={change_fusion}")
        if pooling_mode == "mask_m_ring":
            in_dim = c * (6 if change_fusion == "legacy" else 8)
        else:
            in_dim = c * (3 if change_fusion == "legacy" else 4)
        self.head = CoralHead(in_dim=in_dim, num_classes=num_classes, hidden_dim=hidden_dim, dropout=dropout)

    def feat_map(self, x):
        feats = self.backbone(x)
        return feats[self.stage_index]

    def forward(self, pre, post, m, r):
        f_pre = self.feat_map(pre)
        f_post = self.feat_map(post)
        if self.pooling_mode == "rgb_only":
            v_pre = global_avg_pool(f_pre)
            v_post = global_avg_pool(f_post)
            d = v_post - v_pre
            if self.change_fusion == "legacy":
                x = torch.cat([v_pre, v_post, d], dim=1)
            else:
                d_abs = self.diff_abs_scale * torch.abs(d)
                x = torch.cat([v_pre, v_post, d, d_abs], dim=1)
        elif self.pooling_mode == "mask_m":
            v_pre = masked_avg_pool(f_pre, m)
            v_post = masked_avg_pool(f_post, m)
            d = v_post - v_pre
            if self.change_fusion == "legacy":
                x = torch.cat([v_pre, v_post, d], dim=1)
            else:
                d_abs = self.diff_abs_scale * torch.abs(d)
                x = torch.cat([v_pre, v_post, d, d_abs], dim=1)
        else:
            v_pre_m = masked_avg_pool(f_pre, m)
            v_post_m = masked_avg_pool(f_post, m)
            v_pre_r = masked_avg_pool(f_pre, r)
            v_post_r = masked_avg_pool(f_post, r)
            d_m = v_post_m - v_pre_m
            d_r = v_post_r - v_pre_r
            if self.change_fusion == "legacy":
                x = torch.cat([v_pre_m, v_post_m, v_pre_r, v_post_r, d_m, d_r], dim=1)
            else:
                d_abs_m = self.diff_abs_scale * torch.abs(d_m)
                d_abs_r = self.diff_abs_scale * torch.abs(d_r)
                x = torch.cat([v_pre_m, v_post_m, v_pre_r, v_post_r, d_m, d_r, d_abs_m, d_abs_r], dim=1)
        logits = self.head(x)
        probs = coral_probs_from_logits(logits)
        return {"logits_cum": logits, "probs": probs}
