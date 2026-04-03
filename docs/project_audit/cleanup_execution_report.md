# Cleanup Execution Report — Phases 0-3

**Date**: 2026-04-02

---

## 1. What Was Verified (Baseline Behavior)

### Canonical Entry Points
| Workflow | Entry Script | Working Dir |
|----------|-------------|-------------|
| Single-pair inference | `pipeline/scripts/run_instance_impact_driver.py` | `pipeline/` |
| LA fire earliest-post | `pipeline/run_la_fire_batch.py` → `run_pipeline.sh` → above | `pipeline/` |
| LA fire multidate | `pipeline/scripts/run_full_pipeline_launcher.py` → `run_multidate_experiment.py` | `pipeline/` |

### Critical Import Chain Verified
- `infer_stage2_ensemble.py` imports 9 symbols from `train_stage2.py` (sibling in `scripts/`)
- `train_stage2.py` **duplicates** model classes (does NOT import from `src/.../2-stage package/`)
- The duplicated versions **differ** from `src/` originals (ReLU vs GELU, extra fusion modes)
- All subprocess calls use relative paths resolved from `pipeline/` root via `PKG_ROOT`

### Model Equivalence Verified
- Built `SiameseDamageModel` from both `train_stage2` and `stage2b_model` with identical config
- Loaded same random weights → forward pass outputs match with **0.00 max difference**
- `coral_probs_from_logits` produces identical results
- Real checkpoint (`inference0.7273.pt`) loads with `strict=True` through extracted module

---

## 2. What Was Changed (Exact Files, Exact Scope)

### Files CREATED (8 files, all new)

| File | Purpose |
|------|---------|
| `docs/project_audit/baseline_freeze.md` | Phase 0 findings record |
| `exploration/LEGACY_NOTE.md` | Legacy marker for exploration directory |
| `src/cybertraining_team4/LEGACY_NOTE.md` | Legacy marker for src directory |
| `pipeline/stage1/LEGACY_NOTE.md` | Legacy marker for redundant SAM3 copy |
| `pipeline/stage2b_model/__init__.py` | Package init with all exports |
| `pipeline/stage2b_model/model.py` | Extracted model architecture (SiameseDamageModel, CoralHead, CORAL utils) |
| `pipeline/stage2b_model/data_utils.py` | Extracted data loading (load_rgb_tensor, load_mask_tensor, etc.) |
| `pipeline/stage2b_model/metrics.py` | Extracted metrics (confusion_matrix, macro_f1, qwk, ece) |

### Files MODIFIED (1 file)

| File | Change | Lines Affected |
|------|--------|---------------|
| `pipeline/scripts/infer_stage2_ensemble.py` | Import block replaced with try/except: tries `stage2b_model` first, falls back to `train_stage2` | Lines 22-32 → 22-43 |

### Files NOT Modified
- `pipeline/scripts/train_stage2.py` — **UNTOUCHED** (original source of truth)
- All model checkpoints — **UNTOUCHED**
- All calibration artifacts — **UNTOUCHED**
- All config files — **UNTOUCHED**
- All other pipeline scripts — **UNTOUCHED**
- Stage 1 code — **UNTOUCHED**
- No files moved or deleted

---

## 3. What Remains Fragile

### High Risk
1. **Dual-copy maintenance burden**: `stage2b_model/` and `train_stage2.py` now contain duplicate code. If someone modifies the model in `train_stage2.py`, they must also update `stage2b_model/`. No automated enforcement.

2. **`pipeline/stage1/SAM3_Final_20260226/`**: The combined pipeline uses this older SAM3 copy (not `stage1/`). If this directory is moved or deleted, the entire pipeline breaks.

3. **Relative path assumptions**: All subprocess calls assume `cwd=pipeline/`. Running scripts from a different directory will break path resolution.

### Medium Risk
4. **Hardcoded `/media/data/la_fire_2025`** in `run_la_fire_batch.py` — not portable.

5. **`HF_HUB_OFFLINE=1`** assumption — SAM3 model must be cached locally; no internet fallback.

6. **Implicit `train_stage2` resolution**: When `infer_stage2_ensemble.py` is called via subprocess from `run_multidate_experiment.py`, Python's module resolution finds `train_stage2.py` in the script directory because `cwd` is `pipeline/` and the script is invoked as `scripts/infer_stage2_ensemble.py`. This is implicit behavior.

### Low Risk
7. **`stage2b_model` path insertion**: The try/except uses `Path(__file__).resolve().parents[1]` to find `pipeline/`. If the script is moved, this will point to the wrong location. The fallback to `train_stage2` mitigates this.

---

## 4. What Was Deliberately NOT Touched

| Item | Reason |
|------|--------|
| `pipeline/scripts/train_stage2.py` | Original source of truth; must not be modified |
| All model checkpoints (`models/stage2a/`, `models/stage2b/`) | Binary assets, must not be moved |
| All calibration directories | Binary assets, must not be moved |
| All config JSONs | Reference data, must not be modified |
| Stage 1 code (`stage1/`, `pipeline/stage1/`) | Out of scope |
| LA fire pipeline scripts (`run_multidate_experiment.py`, etc.) | Out of scope; only inference import was changed |
| `src/cybertraining_team4/2-stage package/` | Legacy code; kept for reproducibility |
| `exploration/` directories | Marked as legacy but not moved |
| No files were deleted | Conservative approach |

---

## 5. Risks Introduced

### Risk 1: Import Ambiguity (LOW)
The try/except import in `infer_stage2_ensemble.py` could mask import errors. If `stage2b_model` partially loads and raises a non-ImportError, the fallback won't trigger.

**Mitigation**: The try/except catches only `ImportError`. Other exceptions (AttributeError, ModuleNotFoundError subclasses) will propagate normally.

### Risk 2: Code Divergence (MEDIUM)
Two copies of model code now exist. Future changes to `train_stage2.py` model classes will NOT automatically propagate to `stage2b_model/`.

**Mitigation**: Added prominent "DO NOT modify without also updating" warnings in every `stage2b_model/*.py` file header.

### Risk 3: Path Resolution Edge Case (LOW)
The `sys.path.insert(0, str(Path(__file__).resolve().parents[1]))` in the try block permanently modifies `sys.path`. If the script is imported as a module (not run as a subprocess), this side effect persists.

**Mitigation**: The pipeline always calls inference scripts via subprocess, never as imported modules. Side effect is contained.

---

## 6. Recommended Next Steps (Prioritized)

### Immediate (before next development cycle)
1. **Add a simple sync-check script** that compares key function signatures between `stage2b_model/` and `train_stage2.py` to detect drift.
2. **Update `pipeline/README.md`** to document the new `stage2b_model/` module and its relationship to `train_stage2.py`.

### Short-term (next sprint)
3. **Migrate `pipeline/stage1/SAM3_Final_20260226/` → `stage1/`**: Update subprocess calls in `run_instance_impact_driver.py` and `run_multidate_experiment.py` to use the production `stage1/` package. This eliminates the most confusing redundancy.
4. **Add integration test**: A lightweight script that loads all 3 checkpoints through `stage2b_model`, runs a dummy forward pass, and verifies output schema. Run before any commit.

### Medium-term (future refactor)
5. **Make `train_stage2.py` import from `stage2b_model`**: Once confidence is high, modify the training script to import model classes from the extracted module, eliminating the duplication. This reverses the dependency: `stage2b_model` becomes the single source of truth.
6. **Parameterize hardcoded paths**: Make `LA_FIRE_ROOT` and `CHIPS_ROOT` CLI arguments instead of constants.
