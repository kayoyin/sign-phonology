#!/bin/bash
# Extract MediaPipe Holistic keypoints for every video in a split CSV.
#
# Required env vars:
#   VIDEO_DIR     directory containing source videos
#   POSE_DIR      output directory for .npy keypoint files
#   METADATA_CSV  split CSV, with column 1 = video filename
#
# Optional env vars:
#   REPO_ROOT     defaults to the directory containing this script's parent
#
# Usage:
#   VIDEO_DIR=/path/to/videos POSE_DIR=/path/to/poses \
#     METADATA_CSV=/path/to/all.csv ./scripts/extract_pose.sh

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
: "${VIDEO_DIR:?VIDEO_DIR is required}"
: "${POSE_DIR:?POSE_DIR is required}"
: "${METADATA_CSV:?METADATA_CSV is required}"

python "${REPO_ROOT}/stgcn/extract_pose.py" \
    --video_dir "${VIDEO_DIR}" \
    --output_dir "${POSE_DIR}" \
    --metadata_csv "${METADATA_CSV}"
