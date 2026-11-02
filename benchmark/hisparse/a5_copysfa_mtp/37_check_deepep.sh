#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
init_stage check-deepep
source_ascend_environment
prepend_framework_python
configure_deepep_environment
activate_joint_opp
run_logged libraries "$PYTHON_BIN" "$HANDOFF_DIR/check_deepep_libraries.py"
