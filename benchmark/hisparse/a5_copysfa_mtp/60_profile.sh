#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
configure_topology
configure_profile_mode

ACTION="${1:-start}"
[[ "$ACTION" == start || "$ACTION" == stop ]] || {
  echo "Usage: $0 [start|stop]" >&2
  exit 2
}
init_stage "profile-$ACTION"
BASE_URL="${BASE_URL:-http://127.0.0.1:${PORT:-30000}}"
if [[ "$ACTION" == stop ]]; then
  run_logged profile_stop "$PYTHON_BIN" "$HANDOFF_DIR/profile_control.py" stop \
    --base-url "$BASE_URL"
  exit 0
fi
verify_running_server

PROFILE_DIR="${PROFILE_DIR:-$RUN_DIR/profile}"
PROFILE_STEPS="${PROFILE_STEPS:-20}"
PROFILE_TIMEOUT="${PROFILE_TIMEOUT:-900}"
[[ ! -e "$PROFILE_DIR" ]] || { echo "Preserve existing capture: $PROFILE_DIR" >&2; exit 2; }
mkdir -p "$PROFILE_DIR"
echo "Run after ALL_STREAMS_ACTIVE; actual batch sizes are checked in the trace."
run_logged profile_start "$PYTHON_BIN" "$HANDOFF_DIR/profile_control.py" start \
  --base-url "$BASE_URL" --output-dir "$PROFILE_DIR" --num-steps "$PROFILE_STEPS" \
  --receipt "$RUN_DIR/profile-request.json"

deadline=$((SECONDS + PROFILE_TIMEOUT))
while true; do
  if "$PYTHON_BIN" "$HANDOFF_DIR/analyze_profile.py" "$PROFILE_DIR" \
      --output "$RUN_DIR/profile-summary.json" \
      "${PROFILE_MODE_ARGS[@]}" --expected-ranks "$TP_SIZE" --batch "$PER_DP_BATCH_SIZE" \
      --dp-size "$ATTN_DP_SIZE" \
      --min-steps "${PROFILE_MIN_STEPS:-$PROFILE_STEPS}" \
      > "$RUN_DIR/profile-validation.log" 2>&1; then
    cat "$RUN_DIR/profile-validation.log"
    break
  fi
  if (( SECONDS >= deadline )); then
    "$PYTHON_BIN" "$HANDOFF_DIR/profile_control.py" stop --base-url "$BASE_URL" \
      > "$RUN_DIR/profile-stop-after-timeout.log" 2>&1 || true
    cat "$RUN_DIR/profile-validation.log"
    echo "Capture incomplete: inspect rank traces, target steps and batch sizes." >&2
    exit 1
  fi
  sleep 2
done
find "$PROFILE_DIR" -type f -printf '%s %p\n' | sort | tee "$RUN_DIR/profile-files.log"
