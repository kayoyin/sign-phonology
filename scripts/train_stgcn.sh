#!/bin/bash
# Train ST-GCN-ASL on ASL Citizen pose graphs.
#
# Required env vars:
#   POSE_DIR      directory of MediaPipe pose .npy files
#   TRAIN_CSV     training split CSV
#   VAL_CSV       validation split CSV
#
# Optional:
#   GLOSS_FILTER  txt file listing glosses to exclude from training
#   SAVE_DIR      checkpoint output dir (default checkpoints/stgcn_asl)
#   LOG_DIR       per-epoch log dir (default logs/stgcn_asl)
#   EPOCHS, BATCH_SIZE, LR

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
: "${POSE_DIR:?POSE_DIR is required}"
: "${TRAIN_CSV:?TRAIN_CSV is required}"
: "${VAL_CSV:?VAL_CSV is required}"

args=(--pose_dir "${POSE_DIR}"
      --train_csv "${TRAIN_CSV}"
      --val_csv "${VAL_CSV}"
      --save_dir "${SAVE_DIR:-checkpoints/stgcn_asl}"
      --log_dir "${LOG_DIR:-logs/stgcn_asl}"
      --epochs "${EPOCHS:-100}"
      --batch_size "${BATCH_SIZE:-32}"
      --lr "${LR:-1e-3}")
[[ -n "${GLOSS_FILTER:-}" ]] && args+=(--gloss_filter "${GLOSS_FILTER}")

python "${REPO_ROOT}/stgcn/train.py" "${args[@]}"
