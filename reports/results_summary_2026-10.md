# Results Summary — LA Fires 2025 Building Damage (October 2026)

One-page summary of the current numbers, for writing. Full details, every run and every caveat:
`reports/sam3_audit_2026-09.md` (sections in brackets below). All numbers can be cited; the
scripts and result JSONs that produce them are in `evaluation/`.

## Pipeline

1. **Stage 1, building footprints**: SAM 3 (`facebook/sam3`, text prompt "building"), tiled
   512 px windows with ≥ 64 px overlap, instances merged across windows, confidence ≥ 0.4,
   polygons orthogonalised. Zero-shot, no training. (`stage1/`)
2. **Stage 2b, damage per building**: Siamese ConvNeXt-tiny on pre/post 256 px crops with the
   footprint mask, ordinal (CORAL) output, trained on all 10 xView2 tier1 events.
3. **Multi-date vote (M2b)**: majority vote over all valid post-fire dates per building.
4. **Stage 2a, building type and population**: present in the pipeline but not validated;
   to be replaced or validated with Jinyu's Stage 2A model.

## Ground truth

- **xView2 test split**: 933 pre-disaster images, 10 events, IoU ≥ 0.5 matching.
- **LA field data**: LA County LARIAC building outlines with CAL FIRE DINS damage inspections
  (27 Feb 2025), within the imaged area ∩ (Eaton + Palisades perimeters + 100 m):
  83.3 km², 14,369 buildings, 5,474 DINS "Destroyed (>50%)". (§6)
- Maxar imagery sits a median 4.3 m off the county outlines; one translation per 600 m cell is
  applied before any IoU metric. Without it IoU-based F1 drops to ~0.20. (§6)

## Stage 1 — building detection

| Test | Precision | Recall | F1 |
|---|---:|---:|---:|
| xView2 test, original version (Feb 2026) | 0.682 | 0.284 | 0.401 |
| xView2 test, current (v0.2) | 0.737 | 0.565 | **0.640** |
| LA vs LARIAC outlines, original | 0.570 | 0.557 | 0.563 |
| LA vs LARIAC outlines, current | 0.606 | 0.591 | **0.598** |

- LA recall by building area: < 50 m² 0.60 · 50–100 m² 0.70 · 100–200 m² 0.84 ·
  200–400 m² 0.90 · > 400 m² 0.91. DINS-inspected structures: 0.81–0.88. (§6)
- The largest single gain is tiling with cross-window instance merging (+0.19 F1). (§4)
- SAM 3.1 gives no gain on still images. Prompt "house" ties on F1 but has lower wildfire recall. (§4)

## Stage 2b — destroyed vs not destroyed, against CAL FIRE DINS

First valid post-fire date, single model. "Oracle" uses the LARIAC outlines instead of Stage 1. (§7–8)

| Model | Training events | Oracle footprints | End to end (SAM 3) | Eaton | Palisades |
|---|---|---:|---:|---:|---:|
| Original Stage 2b | flood only (3) | 0.078 | 0.000 | 0.078 | 0.077 |
| C | all but fire (8) | 0.429 | 0.383 | 0.418 | 0.565 |
| Persistence (no training) | — | 0.822 | 0.759 | 0.843 | 0.546 |
| **A (current default)** | all hazards (10) | **0.913** | **0.845** | 0.918 | 0.848 |
| B | fire only (2) | 0.940 | 0.874 | 0.950 | 0.805 |

- Persistence baseline: share of the pre-fire footprint that SAM 3 still segments after the
  fire; destroyed if < 0.5. AUC 0.87–0.90. (§7)
- xView2 test macro-F1 / destroyed F1: original 0.325 / 0.022, A 0.772 / 0.829. (§8)

## Final LA product (v3)

22,024 buildings in 126 of 295 cells: 14,486 no damage · 122 minor · 370 major ·
**5,748 destroyed** · 1,298 unknown (no valid post-fire imagery). (`results/final_product/`, §9)

| v3 against DINS | Buildings | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Detected buildings | 9,168 | 0.964 | 0.952 | **0.958** |
| End to end (Stage 1 misses count as errors) | 10,876 | 0.964 | 0.779 | **0.862** |

Any-damage F1 (DINS "Affected" or worse): 0.916. The multi-date vote raises F1 from 0.909
(per-date probability average) to 0.958.

## Why the results changed so much

1. **Stage 2b had never seen a fire.** The original model was trained on three xBD flood events
   (502 destroyed examples). On LA it found 1 of 4,084 destroyed buildings, and it was more
   confident when it was wrong. With the same architecture and settings retrained on all 10 xView2
   events (13,227 destroyed examples), destroyed F1 is 0.96. Fire-only training gives 0.94 and
   training on everything except fire gives 0.43, so the fire examples are what matters. (§7–8)
2. **Stage 1 had a tile-stitching bug and an untuned threshold.** Buildings cut by tile edges were
   split into fragments or lost, which capped recall near 0.30. (§2–4)
3. **Validation against field data** (DINS) replaced visual checks, and exposed the 4 m offset that
   otherwise makes every IoU score look bad. (§6)

## Caveats to state in the paper

- Stage 1 misses 18% of destroyed buildings; they never reach Stage 2. This is most of the gap
  between 0.958 and 0.862. (§7)
- Both training fires (Santa Rosa and SoCal, 2017) are in California, so LA is a same-hazard,
  nearby-region transfer. (§8)
- Only destroyed vs not destroyed is validated; minor and major are rare in both training data
  and DINS. (§9)
- Palisades has fewer imaged cells and lower scores for every method. (§7)
- No confidence intervals or spatial-block validation yet.
