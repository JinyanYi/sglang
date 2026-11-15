#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
require_value RECOVERY_DIR
require_dir "$RECOVERY_DIR/backup/vendors/hwcomputing"
require_file "$RECOVERY_DIR/vendor.before.json"
require_file "$RECOVERY_DIR/deepep-wheel.sha256"
require_file "$RECOVERY_DIR/kernel.before.sha256"
init_stage resume-deepep-verification
source_ascend_environment
helper="$KERNEL_ROOT/tools/deepep_vendor.py"
require_file "$helper"
run_logged saved_wheel sha256sum -c "$RECOVERY_DIR/deepep-wheel.sha256"
vendor="$("$PYTHON_BIN" -c 'from importlib.metadata import distribution; print(distribution("deep_ep").locate_file("deep_ep/vendors/hwcomputing"))')"
run_logged vendor_unchanged "$PYTHON_BIN" "$helper" \
  --source "$RECOVERY_DIR/backup/vendors/hwcomputing" --compare "$vendor" \
  --baseline-receipt "$RECOVERY_DIR/vendor.before.json" --receipt "$RUN_DIR/vendor-comparison.json"
run_logged kernel_unchanged sha256sum -c "$RECOVERY_DIR/kernel.before.sha256"
RUN_DIR="$RUN_DIR/verify" bash "$HANDOFF_DIR/37_check_deepep.sh"
