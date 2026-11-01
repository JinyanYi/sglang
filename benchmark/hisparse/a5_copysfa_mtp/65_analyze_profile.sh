#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
init_stage analyze-profile
require_value PROFILE_DIR
args=("$PROFILE_DIR" --output "$RUN_DIR/article-profile.json"
  --require-joint --expected-ranks "${TP_SIZE:-8}" --batch "${CONCURRENCY:-16}"
  --min-steps "${PROFILE_MIN_STEPS:-20}" --dp-size 1)
[[ -z "${TARGET_WINDOWS:-}" ]] || args+=(--target-windows "$TARGET_WINDOWS")
[[ -z "${TOKENS_PER_ROUND:-}" ]] || args+=(--tokens-per-round "$TOKENS_PER_ROUND")
run_logged analysis "$PYTHON_BIN" "$HANDOFF_DIR/analyze_profile.py" "${args[@]}"
