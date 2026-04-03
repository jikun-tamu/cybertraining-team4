# Independent Audit — 2026-04-01

## 1. Executive Summary

This project currently has three overlapping code lineages:

1. `pipeline/` is the active packaged runtime bundle for end-to-end inference.
2. `stage1/` is a cleaner standalone SAM3 package, but it is not the Stage 1 implementation invoked by the packaged pipeline.
3. `src/cybertraining_team4/2-stage package/` and `src/cybertraining_team4/run_custom_pipeline.py` are older collaborator-delivered lineage artifacts that still matter for provenance, but they are not the cleanest or safest place to anchor future work.

Verified in code:

- The active end-to-end single-pair workflow is `pipeline/run_pipeline.sh` -> `pipeline/scripts/run_instance_impact_driver.py`.
- Active Stage 1 inside that workflow is `pipeline/stage1/SAM3_Final_20260226/...`, not the top-level `stage1/` package.
- Active Stage 2b architecture is a weight-shared pre/post ConvNeXt-style Siamese model with dual-mask pooling and a CORAL ordinal head.
- The deployed Stage 2b checkpoints do **not** use the simple six-vector fusion described in the prior report. The deployed configs use `change_fusion="pre_post_diff"` and `pooling_mode="mask_m_ring"`, which produces **eight** pooled feature blocks: pre, post, signed diff, and absolute diff for both the building mask and the ring mask.
- The LA fire extension is not just “run more dates.” It is a separate multidate workflow with tile-level and crop-level quality filtering, per-date Stage 2b inference, and aggregation scripts.

My judgment:

- The prior audit captured the broad idea of Stage 2 correctly, but it overstated one architectural detail: the current deployed Stage 2b is not best summarized as a six-vector concatenation model.
- The biggest near-term engineering risk is architectural ambiguity: too many partially overlapping entry points and package deliveries.
- The biggest scientific risk is domain mismatch: the deployed Stage 2b bundle appears to be frozen around flood-focused training/calibration and is then applied to wildfire imagery with severe quality variation.

## 2. Current Pipeline Reconstruction

### 2.1 Active single-pair pipeline

Verified in code:

- `pipeline/run_pipeline.sh` is a thin launcher that simply executes `scripts/run_instance_impact_driver.py`.
- `pipeline/scripts/run_instance_impact_driver.py` defines the current packaged workflow:
  1. Stage 1 SAM3 on the pre image
  2. shared crop + mask generation
  3. Stage 2a inference
  4. Stage 2b ensemble inference
  5. presentation table merge
  6. overlay visualization

Key paths:

- Entry point: `pipeline/run_pipeline.sh`
- Driver: `pipeline/scripts/run_instance_impact_driver.py`
- Stage 1 script default: `pipeline/stage1/SAM3_Final_20260226/scripts/run_sam3_building_infer.py`
- Shared artifact bridge: `pipeline/scripts/generate_shared_instance_subimages.py`
- Stage 2a adapter/inference: `pipeline/scripts/build_stage2a_infer_csv.py`, `pipeline/scripts/infer_stage2a.py`
- Stage 2b inference: `pipeline/scripts/infer_stage2_ensemble.py`
- Final merge/report: `pipeline/scripts/present_instance_results.py`

### 2.2 Stage 1 role

Verified in code:

- The packaged Stage 1 code writes `labels/`, `masks/`, and `annotations/` under the run output directory.
- `run_instance_impact_driver.py` depends specifically on `stage1_out / "labels"` as the input to shared artifact generation.
- `generate_shared_instance_subimages.py` first looks for `*_prediction.json` inside the provided Stage 1 labels directory and only falls back to polygonizing Stage 1 mask TIFFs if those JSONs are absent.

Interpretation:

- Stage 1 is the building-instance proposal system.
- In the active pipeline, Stage 1 supplies pre-disaster building polygons and confidence values.
- Stage 2 consumes derived building-centered crops and masks, not the raw Stage 1 outputs directly.

### 2.3 Stage 2a role

Verified in code:

- `pipeline/scripts/infer_stage2a.py` is an EfficientNet-B0 based 4-channel model using RGB plus the building mask.
- It predicts building type and population-related quantities, not damage.
- The single-pair driver uses pre crops and `mask_M` for Stage 2a.

Interpretation:

- Stage 2a is auxiliary. It enriches building records with type and population estimates.
- It is not required for Stage 2b damage inference itself.
- It matters mainly for downstream presentation and risk ranking (`driver_exposure_damage_score`).

### 2.4 Stage 2b role

Verified in code:

- `pipeline/scripts/infer_stage2_ensemble.py` imports `SiameseDamageModel` and helper functions directly from `pipeline/scripts/train_stage2.py`.
- It loads three checkpoints plus matching train configs and calibration directories.
- It ensembles on cumulative CORAL logits, then computes calibrated class probabilities and uncertainty statistics.

Interpretation:

- Stage 2b is the core damage model.
- The active inference-time architecture lives inside `pipeline/scripts/train_stage2.py`, even though that filename suggests “training only.”
- That design is workable but structurally awkward because model definition, dataset code, metrics, and training loop all live in one script.

### 2.5 LA fire extension

Verified in code:

- `pipeline/run_la_fire_batch.py` is the earliest-post batch runner. It chooses the earliest post-disaster date per cell and runs the standard packaged single-pair driver.
- `pipeline/scripts/run_multidate_experiment.py` is the real multidate extension. It:
  1. runs or reuses Stage 1 once per cell
  2. builds shared base artifacts once
  3. runs Stage 2a once
  4. generates post crops for every post date
  5. runs Stage 2b for every post date
  6. aggregates predictions across dates
- `pipeline/scripts/quality_filter.py`, `generate_post_crops_for_date.py`, and `aggregate_multidate_predictions.py` implement the quality and aggregation logic.

Interpretation:

- There are really two LA fire workflows:
  - a simpler earliest-post batch workflow
  - a richer multidate workflow with aggregation
- The multidate workflow is the scientifically more meaningful one.

### 2.6 Dependency structure

Verified in code:

- `run_instance_impact_driver.py` shells out sequentially to per-stage scripts rather than importing them as a library.
- `infer_stage2_ensemble.py` imports from `train_stage2.py`.
- The packaged SAM3 Stage 1 runner modifies `sys.path` to inject `pipeline/stage1/SAM3_Final_20260226/src`.
- Older lineage code in `src/cybertraining_team4/run_custom_pipeline.py` hardcodes an obsolete absolute script path.

Interpretation:

- The pipeline is runnable, but it is coupled through script-level imports and path assumptions rather than a clean package boundary.
- This is functional, but brittle.

### 2.7 Active vs legacy vs dangerous-to-touch

Most active:

- `pipeline/`
- `pipeline/stage1/SAM3_Final_20260226/`
- LA fire outputs under `/media/data/la_fire_2025/stage1_sam3` and `/media/data/la_fire_2025/stage2_damage/multidate_full_run`

Active but secondary:

- top-level `stage1/` package
- `src/cybertraining_team4/process_chips_600m.py`
- `src/cybertraining_team4/prune_bad_pre_cells.py`

Legacy or provenance-only:

- `src/cybertraining_team4/2-stage package/`
- `src/cybertraining_team4/2-stage package.zip`
- `src/cybertraining_team4/run_custom_pipeline.py`
- large portions of `exploration/`

Dangerous to touch before backup or smoke tests:

- packaged Stage 2b checkpoints in `pipeline/models/stage2b/`
- calibration directories in `pipeline/calibration/`
- packaged Stage 1 runtime under `pipeline/stage1/SAM3_Final_20260226/`
- LA fire result directories under `/media/data/la_fire_2025`

## 3. Stage 2 Technical Reverse Engineering

### 3.1 Is Stage 2b really best described as a Siamese model?

Confirmed, with an important qualification.

Verified in code:

- `pipeline/scripts/train_stage2.py` defines `class SiameseDamageModel`.
- A single `timm.create_model(..., features_only=True)` backbone is stored in `self.backbone`.
- `forward()` computes `f_pre = self.feat_map(pre)` and `f_post = self.feat_map(post)` using the same backbone instance.

Conclusion:

- Yes, this is a Siamese-style shared-backbone pre/post model.
- More precisely, it is a shared-backbone pairwise feature model with mask-conditioned regional pooling and ordinal prediction.

### 3.2 Is dual-mask pooling truly present?

Confirmed.

Verified in code:

- `masked_avg_pool()` downsamples a binary mask to feature-map resolution and averages features only where the mask is positive.
- In `SiameseDamageModel.forward()`, when `pooling_mode == "mask_m_ring"`, the model computes:
  - `v_pre_m = masked_avg_pool(f_pre, m)`
  - `v_post_m = masked_avg_pool(f_post, m)`
  - `v_pre_r = masked_avg_pool(f_pre, r)`
  - `v_post_r = masked_avg_pool(f_post, r)`
- `generate_shared_instance_subimages.py` explicitly produces `mask_M` and `mask_R`, where `mask_R` is a dilation-based ring derived from the Stage 1 building polygon.

How it is implemented:

- `mask_M` is the building footprint rasterized inside the 256x256 crop.
- `mask_R` is `(dilate(mask_M, ring_radius_px) - mask_M)`.
- Both masks are downsampled to the backbone feature-map resolution using nearest-neighbor interpolation before masked average pooling.

### 3.3 Is six-vector concatenation truly present?

Partially confirmed, but **not for the deployed checkpoints**.

Verified in code:

- The old package model at `src/cybertraining_team4/2-stage package/scripts/src/models/siamese_stage2.py` explicitly concatenates six vectors:
  - `v_pre`
  - `v_post`
  - `v_delta`
  - `vr_pre`
  - `vr_post`
  - `vr_delta`
- The active model in `pipeline/scripts/train_stage2.py` supports a `change_fusion == "legacy"` mode that also gives six feature blocks for `mask_m_ring`.
- However, the deployed Stage 2b config JSONs all specify:
  - `change_fusion: "pre_post_diff"`
  - `pooling_mode: "mask_m_ring"`

Therefore, the deployed checkpoints use **eight** concatenated blocks, not six:

1. `v_pre_m`
2. `v_post_m`
3. `v_pre_r`
4. `v_post_r`
5. `d_m = v_post_m - v_pre_m`
6. `d_r = v_post_r - v_pre_r`
7. `d_abs_m = |d_m|`
8. `d_abs_r = |d_r|`

This distinction matters. The active deployed model is more expressive than the six-vector summary in the prior report.

### 3.4 Is CORAL or another ordinal method used?

Confirmed.

Verified in code:

- `CoralHead` outputs `num_classes - 1` logits.
- `coral_targets()` encodes cumulative binary targets.
- `coral_probs_from_logits()` converts cumulative logits into class probabilities.
- Training uses `binary_cross_entropy_with_logits()` against CORAL targets.
- Inference ensembles cumulative CORAL logits before converting them to class probabilities.

Conclusion:

- This is genuinely a CORAL-style ordinal model, not ordinary multiclass cross-entropy.

### 3.5 Most distinctive architectural decisions

Verified in code:

1. Shared pre/post backbone with region-specific pooling rather than full-image pooling.
2. Separate modeling of building interior (`M`) and ring context (`R`).
3. Ordinal damage head rather than nominal classification.
4. Ensemble averaging on cumulative ordinal logits, not only on final class labels.
5. Optional post-hoc scalar temperature and vector-exponent calibration.
6. Deployed fusion includes both signed and absolute change terms.

### 3.6 Standard practice vs thoughtful customization

Mostly standard:

- ConvNeXt-Tiny via `timm`
- ImageNet-style normalization
- AdamW
- cosine LR schedule with warmup
- mixed precision
- DDP support
- EMA

Thoughtful customization:

- using pre-disaster building geometry to define `mask_M` and context `mask_R`
- fusing both building-region and context-region deltas
- keeping ordinal structure through CORAL
- averaging ensemble cumulative logits before class-prob conversion
- explicitly supporting ablations (`rgb_only`, `mask_m`, `mask_m_ring`; `legacy` vs `pre_post_diff`)

### 3.7 What assumptions may break in wildfire imagery?

Verified in code or strongly implied by code/config:

1. The model assumes the pre/post pair is aligned enough that feature differences are semantically meaningful.
2. It assumes the pre-disaster polygon remains a useful anchor for post-disaster evidence.
3. It assumes the building footprint and nearby ring contain most of the discriminative signal.
4. It assumes the post crop is real imagery, not blank/nodata-heavy coverage.
5. The deployed configs point to `outputs/flood_stage2_prep_*`, strongly suggesting the frozen bundle is calibrated around flood-focused training data rather than wildfire.

Wildfire-specific failure modes:

- smoke or haze can mimic “change” without structural damage
- burn scars around a building can dominate the ring signal
- a destroyed or partially collapsed building may no longer align with the pre-disaster mask
- black or nodata-heavy post crops can produce spurious severe-damage predictions
- wildfire visual signatures are different from flood damage cues

## 4. Verification of Prior Report Claims

### Claim-by-claim assessment

| Prior claim | Verdict | Why |
|---|---|---|
| Stage 2 is best understood as a Siamese network with dual-mask spatial focus and CORAL ordinal regression | Confirmed | That is the right high-level description of the active model. |
| Dual-mask pooling using building mask M and ring mask R | Confirmed | Present in active `train_stage2.py` and in shared crop generation. |
| Six-vector concatenation is a key feature | Partially confirmed | True for the older package and the active model’s `legacy` fusion mode, but the deployed checkpoints use 8-block `pre_post_diff` fusion. |
| CORAL ordinal prediction head | Confirmed | Explicitly implemented and used in training and inference. |
| Strong engineering pipeline with focal loss, hard example mining, class-aware sampling, EMA, cosine scheduling, DDP support | Confirmed | All those mechanisms exist in the active training script, though the deployed configs currently have some of them disabled. |
| Stage 2 model code is buried in an awkward path | Confirmed | Active architecture is embedded inside `pipeline/scripts/train_stage2.py`; an older model copy also exists under `src/.../2-stage package/scripts/src/models/`. |
| Redundant zip/package delivery at root | Confirmed | `II_package.zip` duplicates the packaged runtime bundle conceptually and likely materially. |
| Too many unclear entry points | Confirmed | There are multiple overlapping drivers: `pipeline/run_pipeline.sh`, `pipeline/run_la_fire_batch.py`, `pipeline/scripts/run_multidate_experiment.py`, `src/.../run_custom_pipeline.py`, `src/.../2-stage package/scripts/run_full_pipeline.py`, plus the standalone top-level `stage1/`. |
| Large superseded exploration experiments exist | Confirmed | `exploration/` contains several old model families and SAM3 experiment folders. |
| Redundant copies of old SAM3 folders exist | Confirmed | There is a packaged SAM3 runtime under `pipeline/stage1/`, a clean standalone package under `stage1/`, and exploratory SAM3 folders under `exploration/`. |
| Archivable checkpoints/binaries exist | Confirmed | Large `.pt`, `.pth`, calibration, and zip assets are present. |
| Create audit docs as first cleanup step | Supported | This is low-risk and already aligned with current need. |
| Extract Stage 2 source into a clean `stage2/` directory by copying, not moving | Supported | This is reasonable, but only after smoke tests define the current baseline. |
| Archive legacy exploration and redundant package/script material | Supported | Reasonable, but only after provenance and backup decisions are explicit. |
| Defer deletion of large binaries until backups are confirmed | Confirmed | Strongly warranted. |
| Stage 2 has never seen wildfire damage data | Partially confirmed | I did not retrace the original training corpus end-to-end, but the deployed config names and Stage2b docs strongly indicate flood-focused training artifacts. |
| LA fire predictions show strong temporal instability across dates | Partially confirmed | The code explicitly measures instability, and existing internal reports claim it; I did not rerun the experiment independently in this audit. |
| Stage 1 recall is low | Unsupported | I did not independently verify recall metrics from code or rerun evaluation. |
| Production inference depends on fragile `sys.path` imports | Confirmed | Present in packaged Stage 1 and older lineage code; inference also depends on script-level imports. |
| No clean retraining guide exists for Stage 2 | Partially confirmed | There is substantial documentation, but it is more a logbook than a concise, current retraining guide for the packaged runtime. |
| Calibration may be tied to training distribution and may not transfer | Partially confirmed | Calibration artifacts are real; transfer risk is scientifically plausible but not directly testable from code alone. |

## 5. Risks and Uncertainties

### Verified risks

1. Architectural ambiguity
   - Multiple code lineages can easily confuse future edits.

2. Fragile import/runtime structure
   - Stage 1 packaged runtime uses `sys.path` insertion.
   - Stage 2 inference imports model code from a training script.
   - Older lineage code contains obsolete absolute paths.

3. Domain mismatch
   - The deployed Stage 2b bundle appears flood-focused, then is reused on wildfire imagery.

4. Image quality sensitivity
   - The LA fire extension had to add explicit tile-level and crop-level quality filters.

5. Pre-polygon anchoring assumption
   - All downstream crops and masks are anchored to pre-disaster geometry.

### Uncertainties I did not fully resolve in this audit

1. Exact empirical Stage 1 recall/precision tradeoff on the chosen SAM3 freeze.
2. Whether the current packaged Stage 2a is scientifically important enough to keep in the production path.
3. Whether M1 or M1b should be the long-term LA fire aggregation default.
4. Whether `pipeline/` and `II_package.zip` are intended to remain synchronized releases or whether one should become archival only.

## 6. Ranked Next Steps

### 6.1 Structural / engineering

Priority 1:

- Freeze the current runnable baseline before any reorganization.
- Define one canonical runtime entry point for each use case:
  - single-pair packaged inference
  - earliest-post LA fire batch
  - multidate LA fire experiment

Priority 2:

- Extract the active Stage 2b model/library code out of `pipeline/scripts/train_stage2.py` into a small non-breaking `pipeline/stage2b/` or similar package **by copy-first**, then make training and inference import from that copy only after smoke tests pass.

Priority 3:

- Clearly label legacy lineage:
  - `src/cybertraining_team4/2-stage package/`
  - `src/cybertraining_team4/run_custom_pipeline.py`
  - duplicated runtime zip/package artifacts

What should wait:

- deleting checkpoints, zips, or exploration assets
- replacing the packaged Stage 1 runtime with the cleaner top-level `stage1/` package

What should not be touched yet:

- packaged checkpoints
- calibration artifacts
- LA fire result directories

### 6.2 Scientific / modeling

Priority 1:

- Decide whether wildfire deployment is currently “demonstration only” or intended as a credible scientific result.
- If credible inference on wildfire is the goal, test transfer explicitly instead of assuming flood-trained calibration generalizes.

Priority 2:

- Measure sensitivity to date selection using the existing multidate framework.
- Compare earliest-post, M1, and M1b on a curated human-review subset.

Priority 3:

- Audit Stage 1 bottlenecks on wildfire cells with obvious missed structures.
- If Stage 1 recall is materially limiting Stage 2, that should be addressed before Stage 2 retraining.

Priority 4:

- Only after the above, consider Stage 2 retraining or adaptation with wildfire-like data or at least wildfire-adjacent validation.

### 6.3 Reproducibility / documentation

Priority 1:

- Write a short “canonical entry points” doc that says exactly which scripts are current for:
  - Stage 1 standalone
  - packaged single-pair inference
  - LA fire earliest-post batch
  - LA fire multidate workflow

Priority 2:

- Write a short “active vs legacy” map.

Priority 3:

- Add one minimal Stage 2b retraining README for the active code path, not the legacy package path.

Priority 4:

- Record smoke-test commands and expected output files after every future structural change.

## 7. Safe Cleanup / Reorganization Plan

This is a proposal only. No destructive action is recommended yet.

### Step 0. Baseline freeze

Action:

- Record the current canonical commands and expected outputs.
- Copy this audit into the project docs.

Smoke tests:

- `pipeline/run_pipeline.sh --help`
- dry-run or one known-good single-pair inference command
- existence checks for packaged Stage 2b checkpoints/configs/calibration

### Step 1. Mark legacy material without moving anything

Action:

- Add a short `LEGACY_NOT_ACTIVE.md` note inside:
  - `src/cybertraining_team4/2-stage package/`
  - `exploration/`
- Add a top-level doc naming `pipeline/` as the active packaged runtime bundle.

Smoke tests:

- none beyond file existence

### Step 2. Copy active Stage 2b code into a clean package

Action:

- Create a new package directory, for example `pipeline/stage2b_active/`.
- Copy, do not move, the active model/helper pieces out of `pipeline/scripts/train_stage2.py` into:
  - `model.py`
  - `data.py`
  - `metrics.py`
  - `calibration.py`
- Keep the original `train_stage2.py` unchanged at first.

Smoke tests:

- import smoke test for new package
- run `infer_stage2_ensemble.py --help`
- run one tiny forward-pass or inference-row smoke test if available

### Step 3. Switch inference to the copied package first

Action:

- Update `infer_stage2_ensemble.py` to import from the copied package instead of `train_stage2.py`.
- Leave the old implementation in place temporarily.

Smoke tests:

- single-row or tiny-batch inference produces identical schema
- compare one prediction JSONL row before vs after on the same sample

### Step 4. Switch training to the copied package

Action:

- Update `train_stage2.py` to import the copied package rather than defining everything inline.

Smoke tests:

- `train_stage2.py --help`
- one-epoch or limited-row dry smoke test if compute is available

### Step 5. Create an archive namespace

Action:

- Copy legacy code into a clearly named archival location if desired, for example:
  - `archive/legacy_2stage_package/`
  - `archive/exploration/`
- Only after confirmation, replace old top-level legacy locations with notes or leave them in place.

Smoke tests:

- confirm active commands unchanged

### Step 6. Defer binary cleanup

Action:

- Do not delete any large weights, zips, or calibration files until:
  - checksums are recorded
  - backups are confirmed
  - active/archival ownership is explicit

Smoke tests:

- checkpoint/config/calibration path existence check

## 8. Open Questions That Require Human Confirmation

1. Which Stage 1 implementation do you want to be canonical long-term:
   - the cleaner top-level `stage1/` package
   - or the packaged `pipeline/stage1/SAM3_Final_20260226/` freeze?

2. Are the packaged Stage 2b checkpoints intentionally frozen as the production baseline, even though the underlying docs/configs suggest flood-focused preparation?

3. For LA fire, which output should be considered the decision-grade result:
   - earliest-post batch output
   - multidate M1
   - multidate M1b
   - something else

4. Is Stage 2a required in the production path, or is it only useful for downstream presentation/risk scoring?

5. Do you want `pipeline/` and `II_package.zip` both preserved as release artifacts, or should one become archival-only after verification?

6. Have the large binary assets and old package deliveries already been backed up elsewhere?

