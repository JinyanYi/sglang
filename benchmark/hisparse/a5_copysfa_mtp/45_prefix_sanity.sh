#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
configure_topology
init_stage prefix-sanity
require_value MODEL_PATH
source_ascend_environment
verify_running_server
BASE_URL="${BASE_URL:-http://127.0.0.1:${PORT:-30000}}"
read -r -a prefix_counts <<< "${PREFIX_COUNTS:-1 2 4}"
run_logged prefix "$PYTHON_BIN" "$HANDOFF_DIR/prefix_sanity.py" \
  --base-url "$BASE_URL" --model-path "$MODEL_PATH" \
  --input-tokens "${INPUT_TOKENS:-65536}" \
  --dp-size "$ATTN_DP_SIZE" \
  --counts "${prefix_counts[@]}" --timeout "${TIMEOUT:-900}" \
  --output-dir "$RUN_DIR"
