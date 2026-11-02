#!/usr/bin/env bash

# Shared, side-effect-free helpers for the manual A5 handoff stages.

HANDOFF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_ROOT="${FRAMEWORK_ROOT:-$(cd "$HANDOFF_DIR/../../.." && pwd)}"
KERNEL_ROOT="${KERNEL_ROOT:-$(cd "$FRAMEWORK_ROOT/.." && pwd)/sgl-kernel-npu}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
EXPECTED_BRANCH="${EXPECTED_BRANCH:-feat/a5-copysfa-mtp-radix-20261031}"
EXPECTED_BASE_REF="${EXPECTED_BASE_REF:-origin/fix/mtp3-draft-hbm-tail-20261031}"

require_file() {
  [[ -f "$1" ]] || { echo "Required file is missing: $1" >&2; return 2; }
}

require_dir() {
  [[ -d "$1" ]] || { echo "Required directory is missing: $1" >&2; return 2; }
}

require_value() {
  local name="$1"
  [[ -n "${!name:-}" ]] || { echo "Set $name before running this stage." >&2; return 2; }
}

configure_topology() {
  TP_SIZE="${TP_SIZE:-8}"
  ATTN_DP_SIZE="${ATTN_DP_SIZE:-1}"
  EP_SIZE="${EP_SIZE:-$TP_SIZE}"
  PER_DP_BATCH_SIZE="${PER_DP_BATCH_SIZE:-16}"
  local name
  for name in TP_SIZE ATTN_DP_SIZE EP_SIZE PER_DP_BATCH_SIZE; do
    [[ "${!name}" =~ ^[1-9][0-9]*$ ]] || { echo "$name must be positive" >&2; return 2; }
  done
  (( TP_SIZE % ATTN_DP_SIZE == 0 && TP_SIZE % EP_SIZE == 0 )) || {
    echo "TP_SIZE must be divisible by ATTN_DP_SIZE and EP_SIZE" >&2; return 2;
  }
  MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-$((ATTN_DP_SIZE * PER_DP_BATCH_SIZE))}"
  GRAPH_BATCH_SIZE="${GRAPH_BATCH_SIZE:-$PER_DP_BATCH_SIZE}"
  CONCURRENCY="${CONCURRENCY:-$MAX_RUNNING_REQUESTS}"
  for name in MAX_RUNNING_REQUESTS GRAPH_BATCH_SIZE CONCURRENCY; do
    [[ "${!name}" =~ ^[1-9][0-9]*$ ]] || { echo "$name must be positive" >&2; return 2; }
  done
  (( MAX_RUNNING_REQUESTS == ATTN_DP_SIZE * PER_DP_BATCH_SIZE )) || {
    echo "MAX_RUNNING_REQUESTS is global: expected ATTN_DP_SIZE * PER_DP_BATCH_SIZE" >&2; return 2;
  }
  (( GRAPH_BATCH_SIZE == PER_DP_BATCH_SIZE )) || {
    echo "GRAPH_BATCH_SIZE must equal PER_DP_BATCH_SIZE for this controlled run" >&2; return 2;
  }
  export TP_SIZE ATTN_DP_SIZE EP_SIZE PER_DP_BATCH_SIZE
  export MAX_RUNNING_REQUESTS GRAPH_BATCH_SIZE CONCURRENCY
}

configure_deepep_environment() {
  local package
  package="$("$PYTHON_BIN" -c 'from importlib.metadata import distribution; print(distribution("deep_ep").locate_file("deep_ep"))')" || return $?
  require_dir "$package/vendors/hwcomputing" || return $?
  export SGLANG_DEEPEP_OP_API_LIBRARY="$package/vendors/hwcomputing/op_api/lib/libcust_opapi.so"
  require_file "$SGLANG_DEEPEP_OP_API_LIBRARY" || return $?
  export PYTHONPATH="$package${PYTHONPATH:+:$PYTHONPATH}"
  export LD_LIBRARY_PATH="$package/vendors/hwcomputing/op_api/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export ASCEND_CUSTOM_OPP_PATH="$package/vendors/hwcomputing${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
}

verify_running_server() {
  run_logged server-state "$PYTHON_BIN" "$HANDOFF_DIR/check_server_state.py" \
    --base-url "${BASE_URL:-http://127.0.0.1:${PORT:-30000}}" \
    --tp "$TP_SIZE" --dp "$ATTN_DP_SIZE" --ep "$EP_SIZE" --batch "$PER_DP_BATCH_SIZE" \
    --mtp "${MTP:-1}" --graph "${GRAPH:-1}" --output "$RUN_DIR/server-state.json"
}

init_stage() {
  local stage="$1"
  local stamp
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  RUN_ROOT="${RUN_ROOT:-$PWD/a5-copysfa-mtp-runs}"
  RUN_DIR="${RUN_DIR:-$RUN_ROOT/${stamp}-${stage}}"
  mkdir -p "$RUN_DIR"
  export RUN_DIR
  trap 'rc=$?; trap - EXIT; printf "%s\n" "$rc" > "$RUN_DIR/stage.exitcode"; exit "$rc"' EXIT
  echo "run_dir=$RUN_DIR"
}

run_logged() {
  local name="$1"
  shift
  local rc had_errexit=0
  [[ $- == *e* ]] && had_errexit=1
  set +e
  "$@" 2>&1 | tee "$RUN_DIR/$name.log"
  rc=${PIPESTATUS[0]}
  printf '%s\n' "$rc" > "$RUN_DIR/$name.exitcode"
  if (( had_errexit )); then
    set -e
  else
    set +e
  fi
  return "$rc"
}

activate_joint_opp() {
  require_file "$KERNEL_ROOT/tools/a5_c8/env.py"
  local exports
  exports="$("$PYTHON_BIN" "$KERNEL_ROOT/tools/a5_c8/env.py")" || return $?
  eval "$exports"
  export FRAMEWORK_ROOT KERNEL_ROOT
}

source_ascend_environment() {
  PYTHON_BIN="$("$PYTHON_BIN" -c 'import sys; print(sys.executable)')"
  local toolkit_env="${ASCEND_TOOLKIT_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
  local atb_env="${ATB_ENV:-/usr/local/Ascend/nnal/atb/set_env.sh}"
  require_file "$toolkit_env"
  require_file "$atb_env"
  set +u
  # CANN and ATB may read unset shell variables while initializing.
  source "$toolkit_env"
  source "$atb_env"
  export PATH="$(dirname -- "$PYTHON_BIN"):$PATH"
  set -u
}

prepend_framework_python() {
  export PYTHONPATH="$FRAMEWORK_ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
}

check_checkout() {
  printf '%s_checkout_audit=SKIPPED_BY_USER\n' "$1"
}
