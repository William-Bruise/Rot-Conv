#!/usr/bin/env bash
set -e

# Compare F-Conv baseline (EDSR_FCNN) vs SO2 variant (EDSR_SO2)
# Baseline:
# python main.py --model EDSR_FCNN --save EDSR_FCNN_x4 --scale 4 --tranNum 8

# SO2 variant:
python main.py \
  --model EDSR_SO2 \
  --save EDSR_SO2_x4 \
  --scale 4 \
  --tranNum 8
