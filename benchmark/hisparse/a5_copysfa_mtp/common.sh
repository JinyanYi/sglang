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

source_ascend_environment() {
  local toolkit_env="${ASCEND_TOOLKIT_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
  local atb_env="${ATB_ENV:-/usr/local/Ascend/nnal/atb/set_env.sh}"
  require_file "$toolkit_env"
  require_file "$atb_env"
  set +u
  # CANN and ATB may read unset shell variables while initializing.
  source "$toolkit_env"
  source "$atb_env"
  set -u
}

prepend_framework_python() {
  export PYTHONPATH="$FRAMEWORK_ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
}

check_checkout() {
  local label="$1"
  local root="$2"
  local branch origin
  git -C "$root" rev-parse --is-inside-work-tree >/dev/null 2>&1 || {
    echo "$label is not a Git worktree: $root" >&2
    return 2
  }
  branch="$(git -C "$root" branch --show-current)"
  origin="$(git -C "$root" remote get-url origin)"
  printf '%s_root=%s\n%s_branch=%s\n%s_head=%s\n%s_origin=%s\n' \
    "$label" "$root" "$label" "$branch" "$label" \
    "$(git -C "$root" rev-parse HEAD)" "$label" "$origin"
  [[ "$branch" == "$EXPECTED_BRANCH" ]] || {
    echo "$label must be on $EXPECTED_BRANCH, got $branch" >&2
    return 2
  }
  [[ "$origin" == "https://github.com/JinyanYi/"* ]] || {
    echo "$label origin is not the expected personal fork: $origin" >&2
    return 2
  }
  local base_sha
  case "$label" in
    framework) base_sha=4d1fefd6b79f214f908359e032f96ba3a64137ee ;;
    kernel) base_sha=db77f6fd2db26b2e60a02c63f821121045482835 ;;
    *) echo "Unknown checkout role: $label" >&2; return 2 ;;
  esac
  git -C "$root" merge-base --is-ancestor "$base_sha" HEAD || {
    echo "$label HEAD is not based on pinned $EXPECTED_BASE_REF ($base_sha)" >&2
    return 2
  }
  printf '%s_base=%s\n' "$label" "$base_sha"
  printf '%s_status_begin\n' "$label"
  git -C "$root" status --short
  printf '%s_status_end\n' "$label"
  printf '%s_tracked_diff_sha256=' "$label"
  git -C "$root" diff --binary | sha256sum | cut -d' ' -f1
  printf '%s_untracked_sha256_begin\n' "$label"
  while IFS= read -r -d '' path; do
    sha256sum "$root/$path"
  done < <(git -C "$root" ls-files --others --exclude-standard -z | sort -z)
  printf '%s_untracked_sha256_end\n' "$label"
}
