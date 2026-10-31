#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

MODE="${1:-copy}"
[[ "$MODE" == copy || "$MODE" == model ]] || {
  echo "Usage: $0 [copy|model]" >&2
  exit 2
}
init_stage "preflight-$MODE"
require_dir "$FRAMEWORK_ROOT"
require_dir "$KERNEL_ROOT"
require_value ASCEND_RT_VISIBLE_DEVICES

{
  check_checkout framework "$FRAMEWORK_ROOT"
  check_checkout kernel "$KERNEL_ROOT"
  command -v npu-smi || true
  npu-smi info || true
} 2>&1 | tee "$RUN_DIR/source-and-device.log"

source_ascend_environment
prepend_framework_python
export FRAMEWORK_ROOT
run_logged preflight "$PYTHON_BIN" "$HANDOFF_DIR/preflight.py" --mode "$MODE"
