#!/usr/bin/env bash
set -euo pipefail
HANDOFF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ $# -ge 1 ]] || { echo "Usage: $0 CASE_ROOT [analysis arguments...]" >&2; exit 2; }
exec "${PYTHON_BIN:-python3}" "$HANDOFF_DIR/article_report.py" "$@"
