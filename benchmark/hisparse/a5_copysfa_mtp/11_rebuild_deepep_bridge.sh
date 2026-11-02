#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
init_stage rebuild-deepep-bridge
check_checkout kernel "$KERNEL_ROOT" | tee "$RUN_DIR/kernel-source.log"
source_ascend_environment
helper="$KERNEL_ROOT/tools/deepep_vendor.py"
require_file "$helper"
vendor="$("$PYTHON_BIN" -c 'from importlib.metadata import distribution; print(distribution("deep_ep").locate_file("deep_ep/vendors/hwcomputing"))')"
kernel_so="$("$PYTHON_BIN" -c 'from importlib.metadata import distribution; print(distribution("sgl-kernel-npu").locate_file("sgl_kernel_npu/lib/libsgl_kernel_npu.so"))')"
require_file "$kernel_so"
sha256sum "$kernel_so" > "$RUN_DIR/kernel.before.sha256"
run_logged preserve_vendor "$PYTHON_BIN" "$helper" --source "$vendor" \
  --package-root "$RUN_DIR/backup" --receipt "$RUN_DIR/vendor.before.json"
export SGL_DEEPEP_PREBUILT_VENDOR="$RUN_DIR/backup/vendors/hwcomputing"
stamp="$(mktemp "$RUN_DIR/wheel-start.XXXXXX")"
run_logged build bash -c 'cd "$1" && bash build.sh -a deepep Ascend950' _ "$KERNEL_ROOT"
mapfile -d '' wheels < <(find "$KERNEL_ROOT/output" -maxdepth 1 -type f \
  -name 'deep_ep-*.whl' -newer "$stamp" -print0)
(( ${#wheels[@]} == 1 )) || { echo "Expected exactly one fresh DeepEP wheel" >&2; exit 1; }
sha256sum "${wheels[0]}" | tee "$RUN_DIR/deepep-wheel.sha256"
run_logged install "$PYTHON_BIN" -m pip install --force-reinstall --no-deps "${wheels[0]}"
vendor="$("$PYTHON_BIN" -c 'from importlib.metadata import distribution; print(distribution("deep_ep").locate_file("deep_ep/vendors/hwcomputing"))')"
run_logged vendor_unchanged "$PYTHON_BIN" "$helper" --source "$SGL_DEEPEP_PREBUILT_VENDOR" --compare "$vendor" \
  --baseline-receipt "$RUN_DIR/vendor.before.json" --receipt "$RUN_DIR/vendor-comparison.json"
run_logged kernel_unchanged sha256sum -c "$RUN_DIR/kernel.before.sha256"
RUN_DIR="$RUN_DIR/verify" bash "$HANDOFF_DIR/37_check_deepep.sh"
