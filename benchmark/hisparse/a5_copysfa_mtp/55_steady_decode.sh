#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
configure_topology
init_stage steady-decode
require_value MODEL_PATH
source_ascend_environment
verify_running_server

profile_args=()
if [[ "${AUTO_PROFILE:-0}" == 1 ]]; then
  profile_args=(--profile-root "$RUN_DIR" --profile-steps "${PROFILE_STEPS:-20}"
    --profile-warmup-tokens "${PROFILE_WARMUP_TOKENS:-64}"
    --profile-warmup-seconds "${PROFILE_WARMUP_SECONDS:-5}"
    --profile-window-timeout "${PROFILE_WARMUP_TIMEOUT:-900}")
  echo "Automatic no-stack capture: waiting for all streams to progress and remain live."
else
  echo "Wait for ALL_STREAMS_ACTIVE, then start 60_profile.sh in another terminal."
fi
run_logged workload "$PYTHON_BIN" "$HANDOFF_DIR/shared_prefix_workload.py" \
  --base-url "${BASE_URL:-http://127.0.0.1:${PORT:-30000}}" \
  --model-path "$MODEL_PATH" --input-tokens "${INPUT_TOKENS:-65536}" \
  --output-tokens "${OUTPUT_TOKENS:-1024}" \
  --suffix-tokens "${SUFFIX_TOKENS:-128}" \
  --num-prompts "${STEADY_NUM_PROMPTS:-$CONCURRENCY}" --concurrency "$CONCURRENCY" \
  --dp-size "$ATTN_DP_SIZE" \
  --timeout "${TIMEOUT:-3600}" --flush --output "$RUN_DIR/steady-decode.json" "${profile_args[@]}"
