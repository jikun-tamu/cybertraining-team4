# pipeline/stage1/ — Archived Stage 1 Variants

This directory contains **archived / experimental** Stage 1 implementations.
It is **not** the active Stage 1 package used by the pipeline.

## Active Stage 1

The production Stage 1 package is `sam3_building_identifier`, located at:

```
/media/gisense/xihan/250812_tamu_cybertraining_team4/stage1/
```

Install it with:

```bash
pip install -e ../stage1 --no-deps
```

The pipeline scripts (`scripts/run_multidate_experiment.py`, etc.) import and call
`sam3_building_identifier` directly from that location.

## Contents of This Directory

| Directory | Description |
|-----------|-------------|
| `SAM3_Final_20260226/` | Alternative SAM3 pipeline variant (Feb 2026) with georeferencing recovery and GeoJSON/GPKG output. Archived; see also `../../archive/SAM3_Final/`. |
