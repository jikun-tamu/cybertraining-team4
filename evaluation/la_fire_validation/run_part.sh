#!/bin/bash
# SAM 3 building detection on post-fire chips (experiment 1: building persistence). Usage: run_part.sh <part>
B=/media/data/building_instance_tamu/la_fire_2025/postfire_sam3
export PYTHONPATH=/media/gisense/xihan/archive/250812_tamu_cybertraining_team4/stage1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=1
/media/gisense/xihan/geoai_sam/bin/python -u -m sam3_building_identifier --input-dir $B/staging/$1 --output-dir $B/$1 \
  --disaster-type all --device cuda --min-size 30 --no-annotations > $B/$1.log 2>&1
