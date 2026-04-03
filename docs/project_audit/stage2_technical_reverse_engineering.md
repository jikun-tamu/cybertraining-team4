# Stage 2 — Damage Assessment: Technical Reverse Engineering

**Date**: 2026-04-01
**Purpose**: Complete technical documentation of Stage 2 for someone who did not build it

---

## Purpose of Stage 2

Stage 2 takes **individual building regions** identified by Stage 1 and classifies the **level of damage** by comparing pre-disaster and post-disaster satellite imagery. It consists of two sub-stages:

- **Stage 2a**: Building type classification + population estimation (auxiliary — not damage-related)
- **Stage 2b**: Damage severity classification (the core damage assessment)

---

## Stage 2b: Damage Classification (Core)

### Input Expected from Stage 1

Stage 2b does **not** consume Stage 1 output directly. An intermediate script (`generate_shared_instance_subimages.py`) transforms Stage 1 output into the format Stage 2 expects:

| Input | Shape | Source |
|-------|-------|--------|
| `pre_crop` | 256x256 RGB PNG | Pre-disaster image, cropped around building centroid |
| `post_crop` | 256x256 RGB PNG | Post-disaster image, same crop window |
| `mask_M` | 256x256 grayscale PNG | Binary mask of building footprint (from Stage 1 polygon) |
| `mask_R` | 256x256 grayscale PNG | Binary ring mask (dilated M by 48px, minus M) — surrounding context |

The crop window is computed by:
1. Computing the centroid of the Stage 1 polygon (WKT pixel coordinates)
2. Centering a 256x256 window on that centroid (clamped to image bounds)
3. Rasterizing the polygon within that window → mask_M
4. Dilating mask_M by 48px, subtracting M → mask_R

### Additional Inputs Needed

- **Trained checkpoints**: 3 `.pt` files in `pipeline/models/stage2b/`
  - `inference0.7273.pt` (seed 2025, weight 4)
  - `inference0.7034_seed7777.pt` (seed 7777, weight 3)
  - `inference0.7066_seed9999.pt` (seed 9999, weight 2)
- **Calibration**: Temperature scaling baked into checkpoints under `calibration.temperature` and `calibration.vector_temperature`

### No Additional Metadata Required

Stage 2b operates purely on the four image/mask inputs. It does not use geolocation, disaster type, or any other metadata for inference.

---

## Model Architecture

### Overview: Siamese ConvNeXt with Dual-Mask CORAL Head

```
┌─────────────┐     ┌─────────────┐
│  Pre-image   │     │  Post-image  │
│  [3,256,256] │     │  [3,256,256] │
└──────┬───────┘     └──────┬───────┘
       │                     │
       ▼                     ▼
┌──────────────────────────────────┐
│     SHARED ConvNeXt-Tiny         │  ← Weight-shared backbone
│     (ImageNet-12K pretrained)    │
│     features_only, stage 3       │
└──────┬──────────────┬────────────┘
       │              │
   f_pre           f_post          [B, 768, H/32, W/32]
       │              │
       ▼              ▼
┌──────────────────────────────────┐
│     DUAL-MASK POOLING            │
│                                  │
│  mask_M (building) ──┐           │
│  mask_R (ring)    ──┐│           │
│                     ││           │
│  v_pre  = pool(f_pre,  M)       │  [B, 768]
│  v_post = pool(f_post, M)       │  [B, 768]
│  v_delta = v_post - v_pre       │  [B, 768]  ← change signal
│                                  │
│  vr_pre  = pool(f_pre,  R)      │  [B, 768]
│  vr_post = pool(f_post, R)      │  [B, 768]
│  vr_delta = vr_post - vr_pre    │  [B, 768]  ← context change
└──────┬───────────────────────────┘
       │
       ▼  concatenate all 6 vectors
   z = [v_pre, v_post, v_delta,    [B, 6*768 = 4608]
        vr_pre, vr_post, vr_delta]
       │
       ▼
┌──────────────────────────────────┐
│     CORAL HEAD (ordinal)         │
│                                  │
│  Linear(4608 → 512) + GELU      │
│  Dropout(0.2)                    │
│  Linear(512 → 3)                 │  ← K-1 cumulative logits
│                                  │
│  Temperature scaling: logits/T   │
│  Sigmoid → cumulative probs      │
│  Convert → class probs [B, 4]   │
└──────────────────────────────────┘
```

### Component Details

#### 1. Shared Backbone (`backbone.py`)

- **Architecture**: ConvNeXt-Tiny from `timm` library
- **Pretrained**: `timm/convnext_tiny.in12k_ft_in1k` (ImageNet-12K → ImageNet-1K)
- **Feature extraction**: `features_only=True` mode, extracting stage 3 features
- **Output**: Feature maps at ~1/32 of input resolution
- **Weight sharing**: Both pre and post images pass through the **exact same** backbone instance

#### 2. Masked Pooling (`masked_pool.py`)

- **Purpose**: Average features only within masked regions (building or ring)
- **Process**:
  1. Downsample binary mask to match feature map spatial dimensions (nearest interpolation)
  2. Element-wise multiply: `masked_feat = features * mask`
  3. Sum spatially, divide by mask pixel count: `output = sum(masked_feat) / sum(mask)`
  4. eps=1e-6 prevents division by zero for empty masks
- **Two masks used**:
  - **M (building)**: Focuses on the building footprint itself
  - **R (ring)**: Captures surrounding context (unchanged buildings, terrain)
- **Result**: 6 feature vectors per sample (pre/post/delta for both M and R regions)

#### 3. CORAL Head (`coral_head.py`)

**CORAL = Consistent Rank-Aligned Logits** — an ordinal regression method.

For K=4 damage classes (no-damage < minor < major < destroyed), CORAL predicts K-1=3 cumulative binary thresholds:

```
logit[0] → P(damage > 0)   i.e., P(at least minor)
logit[1] → P(damage > 1)   i.e., P(at least major)
logit[2] → P(damage > 2)   i.e., P(destroyed)
```

Converting to class probabilities:
```
sig = sigmoid(logits)   → [s0, s1, s2]

P(no-damage) = 1 - s0
P(minor)     = s0 - s1
P(major)     = s1 - s2
P(destroyed) = s2
```

**Why CORAL?** Damage severity is ordinal — misclassifying "no-damage" as "destroyed" is worse than misclassifying as "minor". CORAL enforces monotonic cumulative probabilities, which respects this ordering and improves calibration.

#### 4. Temperature Scaling

Post-hoc calibration applied at inference time:
1. **Scalar temperature**: Divides cumulative logits by learned T before sigmoid
2. **Vector temperature**: Applies per-class power transform to probabilities: `p'_c ∝ p_c^{alpha_c}`

Both are optimized on a held-out validation set using LBFGS (scalar) or grid search (vector).

---

## Innovation Analysis: How Much Did the Collaborator Innovate?

The Stage 2 architecture is a **carefully engineered combination of established techniques**, with several non-trivial design decisions that go beyond a standard Siamese network:

### Standard Components (Not Novel)
1. **Siamese weight sharing** — classic approach for change detection (Bromley et al., 1993)
2. **ConvNeXt-Tiny backbone** — off-the-shelf from timm, ImageNet pretrained
3. **Temperature scaling** — standard post-hoc calibration (Guo et al., 2017)
4. **CORAL ordinal regression** — published method (Cao et al., 2020)
5. **ImageNet normalization** — standard practice

### Non-Standard / Innovative Design Choices

#### 1. Dual-Mask Spatial Focus (Moderate Innovation)
Most Siamese change detection networks use **global average pooling** — they look at the entire image. This architecture uses **two separate masks** to isolate:
- The building region (M) — where damage actually occurs
- The surrounding ring (R) — context that helps distinguish damage from background variation

This dual-mask approach is a meaningful innovation. It allows the model to separately reason about:
- "Did the building change?" (v_delta from M)
- "Did the surroundings change too?" (vr_delta from R)

A building that changed while surroundings stayed the same is likely damaged. Both changing might indicate image acquisition differences.

#### 2. Six-Vector Concatenation (Moderate Innovation)
Rather than just computing a distance metric (typical Siamese), the model concatenates:
```
[v_pre, v_post, v_delta, vr_pre, vr_post, vr_delta]
```

This gives the head access to:
- **Absolute appearance** (v_pre, v_post) — what the building looks like
- **Change magnitude** (v_delta, vr_delta) — what changed
- **Context comparison** (vr_pre vs vr_post) — was change building-specific or global

Most Siamese networks only use difference or concatenation of two vectors. Using six is more expressive.

#### 3. CORAL for Damage Levels (Good Application, Not Novel)
Using CORAL for damage classification is a well-motivated choice (damage is ordinal), but it's not commonly seen in satellite damage assessment literature, which typically uses standard cross-entropy. The collaborator correctly identified that ordinal regression is more appropriate here.

#### 4. Lazy Head Construction (Minor Engineering)
The CORAL head is built dynamically on the first forward pass, allowing backbone swaps without hardcoding feature dimensions. This is clean engineering, not architectural innovation.

#### 5. Production-Grade Training Pipeline (Significant Engineering)
The `pipeline/scripts/train_stage2.py` (1415 lines) includes:
- Focal loss with class weighting
- CORAL label smoothing
- Hard example mining with EMA
- Class-aware batch sampling
- EMA model averaging
- Cosine learning rate scheduling with warmup
- Multi-metric checkpoint selection
- DDP multi-GPU support

This is **substantial engineering work** — well beyond a research prototype.

### Summary: Innovation Level

| Aspect | Level | Notes |
|--------|-------|-------|
| Overall architecture concept | Standard | Siamese for change detection is established |
| Dual-mask spatial focus | Moderate innovation | Uncommon in literature, well-motivated |
| Six-vector feature concatenation | Moderate innovation | More expressive than typical approaches |
| CORAL ordinal regression for damage | Good application | Correct but not first to do so |
| Training pipeline engineering | High quality | Production-grade with many advanced features |
| Calibration pipeline | Thorough | Both scalar and vector temperature |

**Bottom line**: The collaborator did not invent a new architecture class, but made several smart, non-obvious design choices that improve upon a standard Siamese network. The dual-mask approach and six-vector concatenation are the most distinctive elements. The engineering quality of the training pipeline is impressive.

---

## Training Logic

### Data Preparation

1. **Index building**: `build_stage2_index.py` extracts per-building rows from xView2 labels
   - Matches pre/post label JSONs by tile name
   - Extracts WKT polygons and damage subtypes
   - Maps to 4-class schema: {no-damage: 0, minor: 1, major: 2, destroyed: 3}

2. **Crop preprocessing**: `preprocess_stage2_crops.py` generates training artifacts
   - Computes building centroid from post polygon (falls back to pre)
   - Extracts 256x256 crops centered on centroid
   - Generates mask_M (building footprint) and mask_R (48px ring)
   - Outputs CSV linking crop/mask paths to damage labels

### Training Configuration (from production configs)

```
backbone:           convnext_tiny (pretrained on ImageNet-12K)
epochs:             20
batch_size:         16
learning_rate:      5e-5
weight_decay:       0.05
lr_scheduler:       cosine (with 1 epoch warmup)
val_ratio:          0.15 (tile-level split for geographic independence)
sampler:            weighted (class balance alpha=0.2, cap=3.0)
augmentation:       hflip=0.5, rot90=0.25, color_jitter=0.03
loss:               BCE on CORAL targets + label smoothing (0.02)
EMA_decay:          0.999
early_stop:         patience=5 epochs
best_metric:        macro_f1
```

### Training Loop

```python
for epoch in range(num_epochs):
    model.train()
    for pre, post, m, r, label in train_loader:
        # Forward pass
        output = model(pre, post, m, r)

        # CORAL targets: label=2 → [1, 1, 0]
        targets = coral_targets(label, num_classes=4)

        # Loss: BCE on cumulative logits
        loss = F.binary_cross_entropy_with_logits(
            output['logits_cum'], targets
        )

        # Optional: focal weighting, class weighting
        # Optional: label smoothing on targets

        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    metrics = evaluate(model, val_loader)  # macro F1, QWK, ECE

    # Checkpoint
    if metrics['macro_f1'] > best_f1:
        save_checkpoint(model, metrics, path)
```

### Calibration (Post-Training)

1. **Scalar temperature**: LBFGS optimization of single T parameter to minimize NLL on validation set
2. **Vector temperature**: Greedy grid search over per-class alpha exponents to minimize ECE

---

## Label Schema / Damage Categories

```
Class 0: no-damage      — Building intact, no visible change
Class 1: minor-damage   — Minor structural or cosmetic damage
Class 2: major-damage   — Significant structural damage
Class 3: destroyed      — Building destroyed or collapsed
```

These map directly to xView2 damage subtypes. The xView2 dataset also has "un-classified" which is excluded from training.

---

## Evaluation Metrics

- **Macro F1**: Unweighted average of per-class F1 scores (primary metric)
- **Quadratic Weighted Kappa (QWK)**: Penalizes distant misclassifications more (ordinal-aware)
- **ECE (Expected Calibration Error)**: Measures probability calibration quality
- **NLL (Negative Log-Likelihood)**: Standard probabilistic evaluation
- **Per-class Tolerant Accuracy**: Accuracy where ±1 class error counts as correct

---

## Where Code Is Located

### Model Architecture
```
src/cybertraining_team4/2-stage package/scripts/src/models/
├── siamese_stage2.py       Main model class (SiameseDamageModel)
├── backbone.py             Feature backbone wrapper (FeatureBackbone)
├── coral_head.py           CORAL ordinal head + helper functions
└── masked_pool.py          Masked average pooling (MaskedAvgPool)
```

### Dataset
```
src/cybertraining_team4/2-stage package/scripts/src/data/
└── stage2_dataset.py       Stage2Dataset class + make_dataloader()
```

### Training
```
pipeline/scripts/train_stage2.py                Production training (1415 lines)
src/.../2-stage package/scripts/train_stage2.py  Legacy training (270 lines)
```

### Preprocessing
```
src/.../2-stage package/scripts/
├── build_stage2_index.py                  Index from labeled data
├── build_stage2_index_from_pred.py        Index from Stage 1 predictions
└── preprocess_stage2_crops.py             Crop + mask generation
```

### Inference
```
pipeline/scripts/
├── infer_stage2a.py                Stage 2a inference
├── infer_stage2_ensemble.py        Stage 2b ensemble inference
├── generate_shared_instance_subimages.py  Stage 1→2 bridge
└── generate_post_crops_for_date.py        Per-date crop generation
```

### Calibration
```
src/.../2-stage package/scripts/
├── calibrate_temperature_stage2.py        Scalar temperature
└── calibrate_vector_temperature_stage2.py Vector temperature
```

### Evaluation
```
src/.../2-stage package/scripts/
├── eval_full_stage2.py    Full evaluation with calibration comparison
├── eval_ece_stage2.py     ECE-focused evaluation
└── eval_nll_stage2.py     NLL-focused evaluation
```

### Configs
```
pipeline/configs/stage2b/
├── run019_seed2025_train_config.json
├── seed7777_train_config.json
└── seed9999_train_config.json
```

### Trained Models
```
pipeline/models/
├── stage2a/stage2a_best_model.pt          EfficientNet-B0 (type + population)
└── stage2b/
    ├── inference0.7273.pt                 Seed 2025 (best, weight 4)
    ├── inference0.7066_seed9999.pt        Seed 9999 (weight 2)
    └── inference0.7034_seed7777.pt        Seed 7777 (weight 3)
```

---

## Assumptions the Collaborator's Implementation Makes

### Tightly Coupled to xView2

1. **4-class damage schema** hardcoded everywhere — cannot easily add/remove classes
2. **Image size 1024x1024** assumed in index building scripts (configurable but defaulted)
3. **xView2 naming convention** (`_pre_disaster` / `_post_disaster` suffixes) in index builders
4. **xView2 JSON label format** with `features.xy` and `features.lng_lat` structures
5. **Tile-level splits** assume xView2-style tile naming for geographic independence

### Where Stage 2 May Break for Real-World Mixed-Source Imagery

1. **No wildfire training data**: Stage 2 was trained on earthquake, flood, tsunami, volcano, hurricane damage. Wildfire damage patterns (charred structures, ash coverage) are unseen.
2. **Resolution sensitivity**: xView2 images are ~0.3m GSD (WorldView). Different satellite resolutions will produce different feature responses.
3. **Color/radiometric differences**: ImageNet normalization assumes similar color distributions. Smoke-hazed or differently processed imagery may degrade performance.
4. **Crop size assumption**: 256x256 at xView2 resolution captures a building + context. At different resolutions, the same crop may be too large or too small.
5. **Temporal alignment**: The model assumes pre and post images are perfectly registered. Real-world imagery may have sub-pixel misalignment.
6. **Nodata handling**: The model has no explicit handling for nodata pixels (black regions in partial coverage). Mask R helps somewhat, but nodata in the building region will confuse features.

### What Must Be Preserved for Reuse

1. **Model weights** in `pipeline/models/stage2b/` — the three ensemble checkpoints
2. **Calibration parameters** baked into checkpoints
3. **Model source code** in `src/.../2-stage package/scripts/src/models/` — the four model files
4. **Dataset class** in `src/.../2-stage package/scripts/src/data/stage2_dataset.py`
5. **Preprocessing logic**: crop generation + mask rasterization pipeline
6. **Ensemble weights**: 4:3:2 ratio and weighted CORAL logit averaging

---

## How the Combined Pipeline Uses Stage 1 Outputs

```
Stage 1 prediction JSON
        │
        ▼
generate_shared_instance_subimages.py
  - Reads: instances[].polygon (WKT), instances[].confidence
  - Writes: crops_pre/, crops_post/, masks_M/, masks_R/
  - Writes: shared_instance_samples.csv
        │
        ├──► build_stage2a_infer_csv.py → infer_stage2a.py
        │      (building type + population)
        │
        └──► For each post date:
               generate_post_crops_for_date.py
                 (new post_crop, same geometry)
                    │
                    ▼
               infer_stage2_ensemble.py
                 (damage classification)
                    │
                    ▼
             aggregate_multidate_predictions.py
               (M1, M1b, M2, M3 consensus)
                    │
                    ▼
             build_combined_dataset.py
               (georeferenced GeoJSON/GPKG)
```

---

## Stage 2a: Building Type & Population (Auxiliary)

### Architecture
- **Backbone**: EfficientNet-B0 with modified first conv (3→4 channels to accept RGB + mask)
- **Shared features**: 512-dim FC layer
- **Population head**: 512→128→1 (log1p scale regression)
- **Type head**: 512→128→5 (softmax classification)

### Building Types
```
0: residential_small
1: residential_multi
2: commercial
3: institutional
4: other
```

### Input
- Pre-disaster 256x256 crop (RGB, 3 channels)
- Binary building mask (1 channel)
- Concatenated as 4-channel input

### Output
- Population estimate (converted from log-space via expm1)
- Building type class + probabilities

### Checkpoint
- `pipeline/models/stage2a/stage2a_best_model.pt`

**Note**: Stage 2a is independent of damage assessment. It provides auxiliary building metadata that feeds into the risk scoring (`driver_exposure_damage_score = population * expected_severity`).
