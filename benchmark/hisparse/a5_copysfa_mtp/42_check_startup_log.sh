#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
init_stage check-startup-log
require_value SERVER_LOG
require_file "$SERVER_LOG"
run_logged allocation_roles "$PYTHON_BIN" "$HANDOFF_DIR/check_startup_log.py" "$SERVER_LOG" --mtp "${MTP:-1}"
