#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
configure_topology
init_stage serving-perf
require_value MODEL_PATH
source_ascend_environment
verify_running_server

run_logged benchmark "$PYTHON_BIN" "$HANDOFF_DIR/shared_prefix_workload.py" \
  --base-url "${BASE_URL:-http://127.0.0.1:${PORT:-30000}}" \
  --model-path "$MODEL_PATH" --input-tokens "${INPUT_TOKENS:-65536}" \
  --output-tokens "${OUTPUT_TOKENS:-1024}" \
  --suffix-tokens "${SUFFIX_TOKENS:-128}" \
  --num-prompts "${NUM_PROMPTS:-$((CONCURRENCY * 4))}" --concurrency "$CONCURRENCY" \
  --dp-size "$ATTN_DP_SIZE" \
  --timeout "${TIMEOUT:-3600}" --flush --output "$RUN_DIR/serving-perf.json"
