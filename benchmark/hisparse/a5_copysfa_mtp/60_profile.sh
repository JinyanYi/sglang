#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

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

PROFILE_DIR="${PROFILE_DIR:-$RUN_DIR/profile}"
PROFILE_STEPS="${PROFILE_STEPS:-20}"
PROFILE_TIMEOUT="${PROFILE_TIMEOUT:-900}"
mkdir -p "$PROFILE_DIR"
echo "Run this only after 55_steady_decode.sh prints ALL_STREAMS_ACTIVE."
run_logged profile_start "$PYTHON_BIN" "$HANDOFF_DIR/profile_control.py" start \
  --base-url "$BASE_URL" --output-dir "$PROFILE_DIR" --num-steps "$PROFILE_STEPS"

deadline=$((SECONDS + PROFILE_TIMEOUT))
while ! find "$PROFILE_DIR" -type f -print -quit | grep -q .; do
  if (( SECONDS >= deadline )); then
    "$PYTHON_BIN" "$HANDOFF_DIR/profile_control.py" stop --base-url "$BASE_URL" \
      2>&1 | tee "$RUN_DIR/profile-stop-after-timeout.log" || true
    echo "Timed out waiting for profile output under $PROFILE_DIR" >&2
    exit 1
  fi
  sleep 2
done
# Do not treat an eagerly created trace file as a completed capture. Require
# the file-size manifest to remain unchanged across three polls.
previous_manifest=""
stable_polls=0
while (( stable_polls < 3 )); do
  current_manifest="$(find "$PROFILE_DIR" -type f -printf '%s %p\n' | sort)"
  if [[ "$current_manifest" == "$previous_manifest" ]]; then
    stable_polls=$((stable_polls + 1))
  else
    stable_polls=0
    previous_manifest="$current_manifest"
  fi
  if (( SECONDS >= deadline )); then
    echo "Profile files did not stabilize under $PROFILE_DIR" >&2
    exit 1
  fi
  sleep 2
done
find "$PROFILE_DIR" -type f -printf '%s %p\n' | sort | tee "$RUN_DIR/profile-files.log"
