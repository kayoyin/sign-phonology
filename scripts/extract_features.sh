#!/bin/bash
# Extract penultimate-layer features for one (model, split) combination.
#
# Required env vars:
#   MODEL         "i3d" or "stgcn"
#   MODEL_TAG     "rand", "kine", or "asl" (no checkpoint, Kinetics, ASL Citizen)
#   SPLIT_NAME    short tag for the split (e.g. "test", "semlex", "hcs")
#
# I3D-specific:
#   VIDEO_DIR     directory of input videos
#   METADATA_CSV  CSV listing videos to encode
#   I3D_WEIGHTS   .pt checkpoint (omit to use random weights)
#   I3D_CLASSES   logit dimension of the checkpoint (400 for Kinetics, 2731 for ASL)
#
# STGCN-specific:
#   POSE_DIR        directory of pose .npy files to encode
#   METADATA_CSV    CSV listing the poses to encode
#   TRAIN_POSE_DIR  pose dir used at train time (for the gloss vocabulary)
#   TRAIN_CSV       train CSV used to recover gloss -> class mapping
#   STGCN_WEIGHTS   .pt checkpoint (omit to use random weights)
#
# Common:
#   FEATURE_DIR   output directory; the JSON is written as
#                 ${FEATURE_DIR}/${MODEL}_${MODEL_TAG}_${SPLIT_NAME}.json
#   REPO_ROOT     defaults to the directory containing this script's parent

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
: "${MODEL:?MODEL is required (i3d|stgcn)}"
: "${MODEL_TAG:?MODEL_TAG is required (rand|kine|asl)}"
: "${SPLIT_NAME:?SPLIT_NAME is required}"
: "${FEATURE_DIR:?FEATURE_DIR is required}"
: "${METADATA_CSV:?METADATA_CSV is required}"
mkdir -p "${FEATURE_DIR}"
OUT="${FEATURE_DIR}/${MODEL}_${MODEL_TAG}_${SPLIT_NAME}.json"

case "${MODEL}" in
  i3d)
    : "${VIDEO_DIR:?VIDEO_DIR is required for i3d}"
    args=(--video_dir "${VIDEO_DIR}"
          --metadata_csv "${METADATA_CSV}"
          --output_json "${OUT}")
    if [[ -n "${I3D_WEIGHTS:-}" ]]; then
        args+=(--weights "${I3D_WEIGHTS}")
    fi
    if [[ -n "${I3D_CLASSES:-}" ]]; then
        args+=(--num_classes "${I3D_CLASSES}")
    fi
    python "${REPO_ROOT}/i3d/extract_features.py" "${args[@]}"
    ;;
  stgcn)
    : "${POSE_DIR:?POSE_DIR is required for stgcn}"
    : "${TRAIN_POSE_DIR:?TRAIN_POSE_DIR is required for stgcn}"
    : "${TRAIN_CSV:?TRAIN_CSV is required for stgcn}"
    args=(--pose_dir "${POSE_DIR}"
          --metadata_csv "${METADATA_CSV}"
          --output_json "${OUT}"
          --train_pose_dir "${TRAIN_POSE_DIR}"
          --train_csv "${TRAIN_CSV}")
    if [[ -n "${STGCN_WEIGHTS:-}" ]]; then
        args+=(--weights "${STGCN_WEIGHTS}")
    fi
    python "${REPO_ROOT}/stgcn/extract_features.py" "${args[@]}"
    ;;
  *)
    echo "Unknown MODEL: ${MODEL}" >&2; exit 1 ;;
esac
