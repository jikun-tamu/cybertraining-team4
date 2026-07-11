## Instance Impact Pipeline

Full disaster impact assessment workflow:

- **Stage 1**: Building detection using SAM3 (`sam3_building_identifier` package in `../stage1/`)
- **Shared subimage generation**: 256x256 crops with mask channels
- **Stage 2a**: Building type + population inference
- **Stage 2b**: Damage classification via 3-model Siamese ConvNeXt ensemble
- **Aggregation**: Multi-date M2b majority vote for real-world data

### Package Layout

- `scripts/` -- runtime scripts (multidate experiment, driver, stage scripts)
- `models/stage2a/` -- Stage 2a checkpoint
- `models/stage2b/` -- 3 Stage 2b inference checkpoints
- `configs/stage2b/` -- matching train configs for Stage 2b models
- `calibration/` -- per-checkpoint calibration artifacts
- `docs/` -- technical documentation

### Full Pipeline Run (LA Fire, all cells)

```bash
cd /media/gisense/xihan/archive/250812_tamu_cybertraining_team4/pipeline

conda run -n geoai_sam python scripts/run_multidate_experiment.py \
    --device cuda:0 \
    --workflow realworld
```

This processes all cells in the manifest. To run specific cells:

```bash
conda run -n geoai_sam python scripts/run_multidate_experiment.py \
    --cells cell_00064 cell_00365 \
    --device cuda:0 \
    --workflow realworld
```

### Key Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--cells` | all | Cell IDs to process (omit for all) |
| `--manifest` | auto | Path to chips manifest CSV |
| `--out_root` | auto | Output root directory |
| `--device` | `cuda:0` | GPU device |
| `--workflow` | `realworld` | `realworld` (multidate) or `training` (single post) |
| `--stop_after` | `full` | Stop at: `stage1`, `shared_base`, `stage2a`, or `full` |

### Stage 1 Configuration

Stage 1 always uses the `sam3_building_identifier` package with:
- `--tile-size 512 --overlap 64` (built-in tiling for large images)
- `--prompt building`
- `--min-size 30`

### Main Outputs

Per cell, outputs go to `<out_root>/cell_XXXXX/`:

- `stage1/` -- SAM3 predictions, masks, annotations
- `shared_base/shared_instance_samples.csv` -- instance crops
- `stage2a_predictions.csv` -- building type + population
- `dates/<date>/stage2b_<date>.jsonl` -- per-date damage predictions
- `aggregated/` -- M2b multi-date aggregation results

### Skipping / Re-running

The pipeline auto-skips stages whose output files already exist.
To force re-run, delete the output directory for the target cell:

```bash
rm -rf /media/data/building_instance_tamu/la_fire_2025/stage2_damage/multidate_full_run/cell_00365/
```

### Environment

Use the **`geoai_sam`** conda environment for all pipeline operations.

### Troubleshooting

- **HF / gated model error**: `export HF_TOKEN=...` or run `huggingface-cli login`
- **CUDA OOM**: Reduce batch sizes in the experiment script
- **Too few buildings in Stage 1**: This is fixed by tiling (default 512px)
- **Stale outputs**: Delete the cell's output directory to force re-run
