#!/usr/bin/env bash
set -euo pipefail

CUDA_ID="${CUDA_ID:-0}"

mkdir -p logs_lmvd ckpt_lmvd

for seed in 9 42 2024
do
  echo "============================================================"
  echo "Running LMVD Detour-a05 seed=${seed}"
  echo "============================================================"

  python main_lmvd.py \
    --manifest-path LMVD/processed_811/manifest.csv \
    --stats-path LMVD/processed_811/lmvd_stats.npz \
    --save-dir ./ckpt_lmvd \
    --model-name hypervd_lmvd_seq300_detour_a05_selauprc_thrf1pos_seed${seed} \
    --visual-dim 465 \
    --audio-dim 128 \
    --visual-proj-dim 128 \
    --audio-proj-dim 128 \
    --feat-dim 256 \
    --max-seqlen 300 \
    --batch-size 3 \
    --max-epoch 30 \
    --lr 0.0002 \
    --dropout 0.6 \
    --metric f1 \
    --selection-metric auprc \
    --threshold-metric f1_pos \
    --fusion detour_adapted \
    --pooling topk_mean \
    --pool-alpha 0.5 \
    --topk-divisor 16 \
    --adj-mode soft_threshold \
    --adj-threshold 0.8 \
    --graph-branch both \
    --feature-branch-weight 1.0 \
    --temporal-branch-weight 1.0 \
    --train-fold train \
    --val-fold valid \
    --test-fold test \
    --lmvd-norm-clip 10.0 \
    --seed ${seed} \
    --cuda "${CUDA_ID}" \
    2>&1 | tee logs_lmvd/hypervd_lmvd_seq300_detour_a05_selauprc_thrf1pos_seed${seed}.log
done
