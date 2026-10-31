#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
init_stage steady-decode
require_value MODEL_PATH

echo "Wait for ALL_STREAMS_ACTIVE, then start 60_profile.sh in another terminal."
run_logged workload "$PYTHON_BIN" "$HANDOFF_DIR/shared_prefix_workload.py" \
  --base-url "${BASE_URL:-http://127.0.0.1:${PORT:-30000}}" \
  --model-path "$MODEL_PATH" --input-tokens "${INPUT_TOKENS:-65536}" \
  --output-tokens "${OUTPUT_TOKENS:-1024}" \
  --suffix-tokens "${SUFFIX_TOKENS:-128}" \
  --num-prompts "${NUM_PROMPTS:-16}" --concurrency "${CONCURRENCY:-16}" \
  --timeout "${TIMEOUT:-3600}" --flush --output "$RUN_DIR/steady-decode.json"
