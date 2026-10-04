#!/bin/bash
# LA fire final product v3: v2 footprints and crops, Stage 2b replaced by the all-hazard xView2 model (A_all).
set -e
B=/media/data/building_instance_tamu/la_fire_2025/stage2_damage
V2=$B/multidate_full_run_v2; V3=$B/multidate_full_run_v3
T=/media/data/building_instance_tamu/stage2_training_data
R=/media/gisense/xihan/archive/250812_tamu_cybertraining_team4/pipeline
PY=/media/gisense/xihan/geoai_sam/bin/python
# 1. hard-linked copy; remove every file that will be regenerated (removing a link leaves v2 intact)
cp -al $V2 $V3
rm -f $V3/cell_*/dates/*/stage2b_*.jsonl $V3/cell_*/aggregated_predictions.* $V3/building_damage_all_cells.* \
      $V3/experiment_summary.json $V3/*.log
# 2. one CSV with every cell and date
$PY - <<PYEOF
import glob, pandas as pd
parts=[]
for f in sorted(glob.glob("$V3/cell_*/dates/*/shared_for_date.csv")):
    d=pd.read_csv(f,dtype=str)
    if len(d)==0: continue
    d["cell_id"]=f.split("/")[-4]; d["date"]=f.split("/")[-2]; parts.append(d)
df=pd.concat(parts,ignore_index=True); df.to_csv("$V3/v3_alldates.csv",index=False)
print(len(df),"rows",df.cell_id.nunique(),"cells")
PYEOF
# 3. Stage 2b inference with model A (the ensemble script needs 3 checkpoints: same model 3x)
CK=$T/runs/A_all/stage2_best.pt; CF=$T/runs/A_all/train_config.json
cd $R && CUDA_VISIBLE_DEVICES=${GPU:-0} $PY scripts/infer/infer_stage2_ensemble.py --csv $V3/v3_alldates.csv \
  --ckpts $CK,$CK,$CK --weights 1,1,1 --configs $CF,$CF,$CF --calibration_method none \
  --out_jsonl $V3/v3_alldates_stage2b.jsonl --batch_size 128 --num_workers 12 --device cuda --print_examples 0 --log_every_steps 200
# 4. split back per cell/date, aggregate per cell, combine
$PY $T/split_jsonl.py $V3 $V3/v3_alldates.csv $V3/v3_alldates_stage2b.jsonl
for c in $V3/cell_*; do
  ls $c/dates/*/stage2b_*.jsonl >/dev/null 2>&1 || continue
  $PY scripts/infer/aggregate_multidate_predictions.py --cell_run_dir $c --out_jsonl $c/aggregated_predictions.jsonl --out_csv $c/aggregated_predictions.csv > /dev/null
done
$PY scripts/analysis/build_combined_dataset.py --run_root $V3 --out_csv $V3/building_damage_all_cells.csv \
  --out_geojson $V3/building_damage_all_cells.geojson --out_gpkg $V3/building_damage_all_cells.gpkg
echo V3_DONE
