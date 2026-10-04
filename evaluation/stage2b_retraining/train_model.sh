#!/bin/bash
# Retrain Stage 2b with the run019 configuration on a different training CSV.
# Usage: train_model.sh <name> <csv> <gpu> [extra args]
O=/media/data/building_instance_tamu/stage2_training_data
R=/media/gisense/xihan/archive/250812_tamu_cybertraining_team4/pipeline
name=$1; csv=$2; export CUDA_VISIBLE_DEVICES=$3; shift 3
cd $R && /media/gisense/xihan/geoai_sam/bin/python -u scripts/training/train_stage2.py \
  --csv $csv --out_dir $O/runs/$name --epochs 20 --batch_size 16 --lr 5e-5 --weight_decay 0.05 \
  --backbone convnext_tiny --hidden_dim 512 --dropout 0.1 --crop_size 256 --val_ratio 0.15 --seed 2025 \
  --num_workers 12 --amp_dtype bf16 --sampler_mode weighted --class_balance --class_balance_alpha 0.2 \
  --class_balance_cap 2.0 --aug_hflip 0.5 --aug_vflip 0.0 --aug_rot90 0.25 --aug_color_jitter 0.03 \
  --lr_scheduler cosine --warmup_epochs 1 --log_every_steps 200 --best_metric macro_f1 --best_tiebreak_metric qwk \
  --coral_label_smoothing 0.02 --ema_decay 0.999 --event_metrics --save_val_predictions \
  --change_fusion pre_post_diff --pooling_mode mask_m_ring --diff_abs_scale 1.0 --early_stop_patience 5 \
  --print_per_class_f1 --print_confusion_matrix "$@" > $O/runs/$name.log 2>&1
echo "exit $?" >> $O/runs/$name.log
