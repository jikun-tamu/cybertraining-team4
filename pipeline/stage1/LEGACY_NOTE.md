# LEGACY — pipeline/stage1/SAM3_Final_20260226/

**Status**: ACTIVE BUT REDUNDANT COPY
**Date marked**: 2026-04-02

This is a copy of `archive/SAM3_Final/` (formerly `exploration/SAM3_Final/`) placed here by the collaborator for the combined pipeline.

**Currently used by**: `pipeline/scripts/run_instance_impact_driver.py` and
`pipeline/scripts/run_multidate_experiment.py` as the default Stage 1 backend.

**Relationship to stage1/**:
- `stage1/sam3_building_identifier/` is the newer, cleaner SAM3 package
- This copy is the older SAM3_Final variant with different output format ("notebook" style)
- Pipeline scripts NOW support both: pass `--use_stage1_package` to use the newer package
- Without the flag, scripts default to this SAM3_Final copy for backward compatibility

**NOTE**: Can be removed once all pipeline runs migrate to `--use_stage1_package`.
