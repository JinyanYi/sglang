#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
MODE="${1:-smoke}"
[[ "$MODE" == smoke || "$MODE" == formal ]] || { echo "Usage: $0 [smoke|formal]" >&2; exit 2; }
init_stage "vllm-copysfa-$MODE"
require_value ASCEND_RT_VISIBLE_DEVICES
require_value ORIGINAL_PATHS_ENV
require_file "$ORIGINAL_PATHS_ENV"
{
  check_checkout framework "$FRAMEWORK_ROOT"
  check_checkout kernel "$KERNEL_ROOT"
} 2>&1 | tee "$RUN_DIR/source.log"
source_ascend_environment
set +u
source "$ORIGINAL_PATHS_ENV"
set -u
prepend_framework_python
export FRAMEWORK_ROOT
run_logged preflight "$PYTHON_BIN" "$HANDOFF_DIR/preflight.py" --mode copy
args=("$PYTHON_BIN" "$KERNEL_ROOT/benchmark/copysfa_a5_ab/run_matrix.py"
  --output-root "$RUN_DIR/cases" --batch "${TEST_BATCH_SIZE:-16}"
  --heads "${TEST_HEADS:-8}")
[[ "$MODE" != smoke ]] || args+=(--smoke)
run_logged compare "${args[@]}"
