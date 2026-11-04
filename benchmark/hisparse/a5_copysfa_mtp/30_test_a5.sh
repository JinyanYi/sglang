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
activate_joint_opp
export FRAMEWORK_ROOT

run_logged preflight "$PYTHON_BIN" "$HANDOFF_DIR/preflight.py" --mode "$MODE"
run_logged kernel_a5 "$PYTHON_BIN" -m pytest -q --junitxml="$RUN_DIR/kernel-a5.xml" \
  "$KERNEL_ROOT/tests/test_fused_quant_scatter_copy_sparse_flash_attention.py" \
  "$KERNEL_ROOT/tests/test_fused_quant_scatter_copy_sparse_flash_attention_a5.py"

framework_tests=(
  "$FRAMEWORK_ROOT/test/manual/npu/test_a5_copy_sfa_metadata.py"
  "$FRAMEWORK_ROOT/test/manual/npu/test_sparse_kv_packed_transport.py"
)
if [[ "$MODE" == model ]]; then
  framework_tests+=(
    "$FRAMEWORK_ROOT/test/manual/npu/test_a5_external_lim_contract.py"
    "$FRAMEWORK_ROOT/test/manual/npu/test_a5_lim_copysfa_joint.py"
  )
fi
for path in "${framework_tests[@]}"; do require_file "$path"; done
run_logged framework_a5 "$PYTHON_BIN" -m pytest -q --junitxml="$RUN_DIR/framework-a5.xml" "${framework_tests[@]}"
if [[ "$MODE" == model ]]; then
  run_logged joint_gate "$PYTHON_BIN" "$HANDOFF_DIR/verify_joint_junit.py" \
    "$RUN_DIR/kernel-a5.xml" "$RUN_DIR/framework-a5.xml" --output "$RUN_DIR/joint-gate.json" \
    --query-widths 1 2 3 4 5 6 8 14
fi
