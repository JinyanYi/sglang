#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
configure_topology
(( CONCURRENCY == MAX_RUNNING_REQUESTS )) || {
  echo "Single-wave performance requires CONCURRENCY == MAX_RUNNING_REQUESTS" >&2; exit 2;
}
init_stage serving-perf
require_value MODEL_PATH
source_ascend_environment
verify_running_server

echo "Decode warmup: ${DECODE_WARMUP_ROUNDS:-1} full wave(s), then one measured wave of $CONCURRENCY requests."
echo "Ignoring inherited NUM_PROMPTS; native bench_serving metrics, no profiler."
run_logged benchmark "$PYTHON_BIN" "$HANDOFF_DIR/bench_prefix_cache.py" \
  --base-url "${BASE_URL:-http://127.0.0.1:${PORT:-30000}}" \
  --model-path "$MODEL_PATH" --input-tokens "${INPUT_TOKENS:-65536}" \
  --output-tokens "${OUTPUT_TOKENS:-1024}" \
  --decode-warmup-rounds "${DECODE_WARMUP_ROUNDS:-1}" \
  --concurrency "$CONCURRENCY" --page-size "${PAGE_SIZE:-128}" \
  --dp-size "$ATTN_DP_SIZE" \
  --timeout "${TIMEOUT:-3600}" --output "$RUN_DIR/bench-serving.jsonl"
