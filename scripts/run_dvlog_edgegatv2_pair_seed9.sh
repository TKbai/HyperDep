#!/usr/bin/env bash
set -euo pipefail

CUDA_ID="${CUDA_ID:-0}"
SEED=9

COMMON_ARGS=(
  --data-root dvlog-dataset
  --stats-path dvlog_stats.npz
  --save-dir ./ckpt_dvlog
  --visual-dim 136
  --audio-dim 25
  --visual-proj-dim 128
  --audio-proj-dim 128
  --feat-dim 256
  --max-seqlen 200
  --batch-size 4
  --max-epoch 30
  --lr 0.0002
  --dropout 0.6
  --metric f1
  --selection-metric f1_weighted
  --threshold-metric f1_weighted
  --fusion concat_proj
  --pooling topk_mean
  --pool-alpha 0.3
  --topk-divisor 16
  --adj-mode soft_threshold
  --adj-threshold 0.8
  --graph-branch both
  --feature-branch-weight 1.0
  --temporal-branch-weight 1.0
  --train-num-windows 1
  --eval-num-windows 1
  --window-agg mean
  --seed "${SEED}"
  --cuda "${CUDA_ID}"
)

echo "================ D-Vlog paired baseline ================"

python main_dvlog.py \
  "${COMMON_ARGS[@]}" \
  --model-name hypervd_dvlog_seq200_concat_a03_pair_seed9 \
  --feature-adj-refiner none \
  2>&1 | tee logs/hypervd_dvlog_seq200_concat_a03_pair_seed9.log

echo "================ D-Vlog Edge-GATv2 ======================"

python main_dvlog.py \
  "${COMMON_ARGS[@]}" \
  --model-name hypervd_dvlog_seq200_concat_a03_edgegatv2_seed9 \
  --feature-adj-refiner gatv2 \
  --edge-gatv2-heads 1 \
  --edge-gatv2-hidden 32 \
  --edge-gatv2-dropout 0.1 \
  --edge-gatv2-delta-scale 1.0 \
  --edge-gatv2-use-temporal 0 \
  2>&1 | tee logs/hypervd_dvlog_seq200_concat_a03_edgegatv2_seed9.log
