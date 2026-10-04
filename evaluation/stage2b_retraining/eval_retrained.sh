#!/bin/bash
# Run every Stage 2b model (single checkpoint, no calibration) on xView2 test and both LA footprint sets.
# Usage: [MODELS="..."] eval_retrained.sh <gpu>
# infer_stage2_ensemble.py requires exactly 3 checkpoints: a single model is passed 3 times with equal weights.
O=/media/data/building_instance_tamu/stage2_training_data
R=/media/gisense/xihan/archive/250812_tamu_cybertraining_team4/pipeline
export CUDA_VISIBLE_DEVICES=$1; mkdir -p $O/eval
declare -A CK=( [flood_run019]="$R/models/stage2b/inference0.7273.pt|$R/configs/stage2b/run019_seed2025_train_config.json"
                [A_all]="$O/runs/A_all/stage2_best.pt|$O/runs/A_all/train_config.json"
                [B_fire]="$O/runs/B_fire/stage2_best.pt|$O/runs/B_fire/train_config.json"
                [C_nofire]="$O/runs/C_nofire/stage2_best.pt|$O/runs/C_nofire/train_config.json" )
for m in ${MODELS:-flood_run019 A_all B_fire C_nofire}; do
  IFS="|" read ck cf <<< "${CK[$m]}"
  for d in test_all la_sam3_alldates la_oracle_alldates; do
    out=$O/eval/${m}__${d}.jsonl; [ -s $out ] && continue
    cd $R && /media/gisense/xihan/geoai_sam/bin/python scripts/infer/infer_stage2_ensemble.py --csv $O/csv/$d.csv \
      --ckpts $ck,$ck,$ck --weights 1,1,1 --configs $cf,$cf,$cf --calibration_method none --out_jsonl $out \
      --batch_size 128 --num_workers 12 --device cuda --print_examples 0 --log_every_steps 200 >> $O/eval/eval.log 2>&1 \
      || echo "FAILED $m $d" >> $O/eval/eval.log
  done
done
echo ALL_DONE >> $O/eval/eval.log
