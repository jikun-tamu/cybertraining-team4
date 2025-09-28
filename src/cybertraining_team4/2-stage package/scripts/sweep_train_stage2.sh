#!/usr/bin/env bash
set -euo pipefail

CSV="/anvil/scratch/x-jliu7/test_stage2/stage2_samples_floods.csv"
OUT_ROOT="/anvil/scratch/x-jliu7/outputs_stage2/sweeps"
BACKBONE="convnext_tiny"
CLASSES=4
EPOCHS=5
BATCH=16

LRS=("1e-5" "5e-5" "1e-6")
WDS=("0.05" "0.10" "0.20")
BALANCE_ALPHA="0.5"
BALANCE_CAP="12.0"

mkdir -p "$OUT_ROOT"

for LR in "${LRS[@]}"; do
  for WD in "${WDS[@]}"; do
    RUN="bs${BATCH}_lr${LR}_wd${WD}_balA${BALANCE_ALPHA}_cap${BALANCE_CAP}"
    OUT_DIR="${OUT_ROOT}/${RUN}"
    mkdir -p "$OUT_DIR"
    echo "=== Running ${RUN} ==="
    python /anvil/scratch/x-jliu7/scripts/train_stage2.py \
      --csv "$CSV" \
      --out_dir "$OUT_DIR" \
      --epochs $EPOCHS \
      --batch_size $BATCH \
      --lr $LR \
      --weight_decay $WD \
      --classes $CLASSES \
      --backbone $BACKBONE \
      --limit 0 \
      --class_balance \
      --balance_alpha ${BALANCE_ALPHA} \
      --balance_cap ${BALANCE_CAP} | tee "${OUT_DIR}/train.log"
  done
done

echo "Sweep complete. Logs in $OUT_ROOT"


