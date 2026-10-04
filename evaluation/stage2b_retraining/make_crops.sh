#!/bin/bash
# Training crops for Stage 2b retraining, made with the same generator the LA inference uses.
D=/media/data/building_instance_tamu; OUT=$D/stage2_training_data
G=/media/gisense/xihan/archive/250812_tamu_cybertraining_team4/pipeline/scripts/infer/generate_shared_instance_subimages.py
for split in train test; do
  /media/gisense/xihan/geoai_sam/bin/python -u $G --stage1_labels_dir $OUT/$split/stage1_labels \
    --pre_images_dir $D/$split/images --post_images_dir $D/$split/images --out_root $OUT/$split/crops \
    --crop_size 256 --ring_radius_px 48 --strict_images --num_workers 20 --log_every 5000 > $OUT/$split/crops.log 2>&1
  echo "exit $? $split" >> $OUT/crops_done.txt
done
