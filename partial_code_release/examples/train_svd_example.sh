#!/usr/bin/env bash
set -euo pipefail

# Example training command for the partial DualForensics source release.
# Replace /path/to/wang2020_dataset with your own dataset root.
# This release does not include datasets, checkpoints, or private paths.

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd):${PYTHONPATH:-}"

python train.py \
  --name dualforensics_partial_release \
  --wang2020_data_path /path/to/wang2020_dataset \
  --data_mode wang2020 \
  --arch CLIP:ViT-L/14_svd \
  --batch_size 48 \
  --loadSize 256 \
  --cropSize 224 \
  --lr 0.0002 \
  --use_svd
