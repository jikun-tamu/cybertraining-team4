#!/bin/bash
# Wait for the three training runs, then evaluate all models and compute metrics.
O=/media/data/building_instance_tamu/stage2_training_data
until grep -q "^exit" $O/runs/A_all.log && grep -q "^exit" $O/runs/B_fire.log && grep -q "^exit" $O/runs/C_nofire.log; do sleep 300; done
$O/eval_retrained.sh 0
/media/gisense/xihan/geoai_sam/bin/python $O/metrics_retrained.py > $O/eval/metrics.log 2>&1
echo METRICS_DONE >> $O/eval/metrics.log
