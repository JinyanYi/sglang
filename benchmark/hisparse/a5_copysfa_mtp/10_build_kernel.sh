#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
init_stage build-kernel
require_dir "$KERNEL_ROOT"
check_checkout kernel "$KERNEL_ROOT" 2>&1 | tee "$RUN_DIR/kernel-source.log"
source_ascend_environment
build_python_dir="$(dirname -- "$("$PYTHON_BIN" -c 'import sys; print(sys.executable)')")"
[[ -x "$build_python_dir/python3" && -x "$build_python_dir/pip3" ]] || {
  echo "Selected build environment must provide sibling python3 and pip3: $build_python_dir" >&2
  exit 2
}
export PATH="$build_python_dir:$PATH"

# build.sh installs the wheel build helper when it is absent. Refuse that
# mutation here: prepare it in the task-private image/environment beforehand.
command -v python3 >/dev/null
command -v pip3 >/dev/null
python3 -c 'import wheel' || {
  echo "The kernel build interpreter lacks wheel; prepare it outside this stage." >&2
  exit 2
}
pip3 show wheel >/dev/null 2>&1 || {
  echo "pip3 cannot see wheel; align the task-private python3/pip3 pair first." >&2
  exit 2
}

SOC_VERSION="${SOC_VERSION:-Ascend950}"
[[ "$SOC_VERSION" == Ascend950 || "$SOC_VERSION" == Ascend950PR_* || "$SOC_VERSION" == Ascend950DT_* ]] || {
  echo "This handoff builds only an Ascend950 kernel wheel, got $SOC_VERSION" >&2
  exit 2
}

deep_ep_manifest() {
  local output="$1"
  {
    if [[ -d "$KERNEL_ROOT/python/deep_ep" ]]; then
      find "$KERNEL_ROOT/python/deep_ep" -type f \
        ! -path '*/__pycache__/*' ! -name '*.pyc' -print0
    fi
    if [[ -d "$KERNEL_ROOT/output" ]]; then
      find "$KERNEL_ROOT/output" -maxdepth 1 -type f -name 'deep_ep*.whl' -print0
    fi
  } | sort -z | xargs -0 -r sha256sum > "$output"
}

deep_ep_manifest "$RUN_DIR/deep_ep.before.sha256"
run_logged build bash -c 'cd "$1" && ./build.sh -a kernels "$2"' _ "$KERNEL_ROOT" "$SOC_VERSION"
deep_ep_manifest "$RUN_DIR/deep_ep.after.sha256"
run_logged deep_ep_unchanged diff -u "$RUN_DIR/deep_ep.before.sha256" "$RUN_DIR/deep_ep.after.sha256"

wheel="$(find "$KERNEL_ROOT/output" -maxdepth 1 -type f -name 'sgl_kernel_npu*.whl' -printf '%T@ %p\n' | sort -nr | head -n1 | cut -d' ' -f2-)"
[[ -n "$wheel" && -f "$wheel" ]] || { echo "No kernel wheel was produced." >&2; exit 1; }
sha256sum "$wheel" | tee "$RUN_DIR/kernel-wheel.sha256"
printf '%s\n' "$wheel" | tee "$RUN_DIR/kernel-wheel.path"

if [[ -n "${INSTALL_PYTHON:-}" ]]; then
  run_logged install "$INSTALL_PYTHON" -m pip install --no-deps --force-reinstall "$wheel"
else
  echo "Wheel not installed. Set INSTALL_PYTHON to a task-private interpreter to install with --no-deps."
fi
