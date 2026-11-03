#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
configure_topology
configure_profile_mode
init_stage analyze-profile
require_value PROFILE_DIR
args=("$PROFILE_DIR" --output "$RUN_DIR/article-profile.json"
  "${PROFILE_MODE_ARGS[@]}" --expected-ranks "$TP_SIZE" --batch "$PER_DP_BATCH_SIZE"
  --min-steps "${PROFILE_MIN_STEPS:-20}" --dp-size "$ATTN_DP_SIZE")
[[ -z "${TARGET_WINDOWS:-}" ]] || args+=(--target-windows "$TARGET_WINDOWS")
[[ -z "${TOKENS_PER_ROUND:-}" ]] || args+=(--tokens-per-round "$TOKENS_PER_ROUND")
run_logged analysis "$PYTHON_BIN" "$HANDOFF_DIR/analyze_profile.py" "${args[@]}"
