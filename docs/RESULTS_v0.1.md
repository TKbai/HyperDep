# HyperDep v0.1: HyperVD-DVlog Baseline

## Task

Multimodal depression detection on D-Vlog using acoustic and visual features.

## Base Model

This version migrates HyperVD to D-Vlog by adapting:

- D-Vlog visual feature: 136 dim
- D-Vlog acoustic feature: 25 dim
- visual projection: 136 -> 128
- acoustic projection: 25 -> 128
- fused feature: 256 dim
- hyperbolic GCN branches from HyperVD
- video-level binary depression classification

## Best Configuration

```text
max_seqlen    = 200
pooling       = topk_mean
pool_alpha    = 0.3
topk_divisor  = 16
lr            = 0.0002
dropout       = 0.6
max_epoch     = 30
metric        = weighted F1 on validation set

