#!/usr/bin/env bash
set -euo pipefail
HANDOFF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HANDOFF_DIR/article_no_offload.env.example"
exec bash "$HANDOFF_DIR/40_launch_server.sh" "$@"
