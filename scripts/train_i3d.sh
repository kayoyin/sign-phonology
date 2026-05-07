#!/bin/bash
# Train I3D-ASL on ASL Citizen, optionally excluding minimal-pair test signs.
#
# Required env vars:
#   VIDEO_DIR     directory of ASL Citizen videos
#   TRAIN_CSV     training split CSV
#   VAL_CSV       validation split CSV
#
# Optional:
#   GLOSS_FILTER  txt file listing glosses to exclude from training
#                 (used to reproduce the paper's "Strict exclusion of test signs" setup)
#   INIT_WEIGHTS  .pt to warm-start from (e.g. Kinetics-pretrained)
#   SAVE_DIR      checkpoint output dir (default checkpoints/i3d_asl)
#   LOG_DIR       per-epoch log dir (default logs/i3d_asl)
#   EPOCHS, BATCH_SIZE, LR

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
: "${VIDEO_DIR:?VIDEO_DIR is required}"
: "${TRAIN_CSV:?TRAIN_CSV is required}"
: "${VAL_CSV:?VAL_CSV is required}"

args=(--video_dir "${VIDEO_DIR}"
      --train_csv "${TRAIN_CSV}"
      --val_csv "${VAL_CSV}"
      --save_dir "${SAVE_DIR:-checkpoints/i3d_asl}"
      --log_dir "${LOG_DIR:-logs/i3d_asl}"
      --epochs "${EPOCHS:-75}"
      --batch_size "${BATCH_SIZE:-32}"
      --lr "${LR:-4e-3}")
[[ -n "${GLOSS_FILTER:-}" ]] && args+=(--gloss_filter "${GLOSS_FILTER}")
[[ -n "${INIT_WEIGHTS:-}" ]] && args+=(--init_weights "${INIT_WEIGHTS}")

python "${REPO_ROOT}/i3d/train.py" "${args[@]}"
