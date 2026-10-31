#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
init_stage test-cpu
require_dir "$FRAMEWORK_ROOT"
require_dir "$KERNEL_ROOT"
prepend_framework_python

require_file "$FRAMEWORK_ROOT/test/manual/npu/test_a5_copy_sfa_contract.py"
require_file "$FRAMEWORK_ROOT/test/manual/npu/test_sparse_kv_offload_cache.py"
run_logged kernel_cpu "$PYTHON_BIN" -m pytest -q \
  "$KERNEL_ROOT/tests/test_fused_quant_scatter_copy_sparse_flash_attention_cpu.py" \
  "$KERNEL_ROOT/tests/test_copysfa_a5_ab_fixture.py" \
  "$KERNEL_ROOT/tests/test_copysfa_a5_reference_package.py"

run_logged framework_source_cpu "$PYTHON_BIN" -m pytest -q \
  "$FRAMEWORK_ROOT/test/manual/npu/test_a5_copy_sfa_contract.py" \
  "$FRAMEWORK_ROOT/test/manual/npu/test_sparse_kv_offload_cache.py"

runtime_tests=(
  "$FRAMEWORK_ROOT/test/manual/npu/test_sparse_kv_draft_policy.py"
  "$FRAMEWORK_ROOT/test/registered/unit/hardware_backend/test_npu_lim_state.py"
  "$FRAMEWORK_ROOT/test/srt/test_sparse_kv_block_tables.py"
  "$FRAMEWORK_ROOT/test/srt/test_sparse_kv_req_alloc.py"
)
for path in "${runtime_tests[@]}"; do require_file "$path"; done
if [[ "$(uname -s)" == Linux ]]; then
  run_logged framework_runtime_cpu "$PYTHON_BIN" -m pytest -q "${runtime_tests[@]}"
else
  echo "Linux-only runtime policy tests were not run on $(uname -s)." | \
    tee "$RUN_DIR/framework-runtime-not-run.log"
fi
