#!/bin/bash
# Run the minimal-pair t-test (and the random non-minimal-pair control) for
# one feature JSON.
#
# Required env vars:
#   FEATURES        feature JSON produced by extract_features.sh
#   MINIMAL_PAIRS   CSV of minimal pairs (data/minimal_pairs_asl_citizen.csv
#                   for ASL Citizen; mapped variant for Sem-Lex)
#   OUTPUT_DIR      where to write ttest_results.csv etc.
#
# Optional:
#   SEMLEX_METADATA  Sem-Lex split CSV — restricts the feature db to the
#                    listed videos, to avoid mismatched extracted features.
#   SKIP_MINIMAL=1   only run the random non-minimal-pair control
#   SKIP_RANDOM=1    only run the minimal-pair t-test
#   N_RANDOM=N       sample N random non-minimal pairs (default 259)

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
: "${FEATURES:?FEATURES is required}"
: "${MINIMAL_PAIRS:?MINIMAL_PAIRS is required}"
: "${OUTPUT_DIR:?OUTPUT_DIR is required}"

args=(--features "${FEATURES}"
      --minimal_pairs "${MINIMAL_PAIRS}"
      --output_dir "${OUTPUT_DIR}"
      --n_random_pairs "${N_RANDOM:-259}")
[[ -n "${SEMLEX_METADATA:-}" ]] && args+=(--semlex_metadata "${SEMLEX_METADATA}")
[[ "${SKIP_MINIMAL:-0}" == "1" ]] && args+=(--skip_minimal)
[[ "${SKIP_RANDOM:-0}" == "1" ]] && args+=(--skip_random)

python "${REPO_ROOT}/analysis/minimal_pair_ttest.py" "${args[@]}"
