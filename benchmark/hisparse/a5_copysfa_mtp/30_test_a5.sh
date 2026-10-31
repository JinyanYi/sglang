#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

MODE="${1:-copy}"
[[ "$MODE" == copy || "$MODE" == model ]] || {
  echo "Usage: $0 [copy|model]" >&2
  exit 2
}
init_stage "test-a5-$MODE"
require_value ASCEND_RT_VISIBLE_DEVICES
{
  check_checkout framework "$FRAMEWORK_ROOT"
  check_checkout kernel "$KERNEL_ROOT"
} 2>&1 | tee "$RUN_DIR/source.log"
source_ascend_environment
prepend_framework_python
export FRAMEWORK_ROOT

run_logged preflight "$PYTHON_BIN" "$HANDOFF_DIR/preflight.py" --mode "$MODE"
run_logged kernel_a5 "$PYTHON_BIN" -m pytest -q \
  "$KERNEL_ROOT/tests/test_fused_quant_scatter_copy_sparse_flash_attention.py" \
  "$KERNEL_ROOT/tests/test_fused_quant_scatter_copy_sparse_flash_attention_a5.py"

framework_tests=(
  "$FRAMEWORK_ROOT/test/manual/npu/test_a5_copy_sfa_metadata.py"
)
if [[ "$MODE" == model ]]; then
  framework_tests+=(
    "$FRAMEWORK_ROOT/test/manual/npu/test_a5_external_lim_contract.py"
  )
fi
for path in "${framework_tests[@]}"; do require_file "$path"; done
run_logged framework_a5 "$PYTHON_BIN" -m pytest -q "${framework_tests[@]}"
