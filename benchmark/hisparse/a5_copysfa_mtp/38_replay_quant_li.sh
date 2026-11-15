#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
[[ $# == 1 ]] || { echo "Usage: $0 SNAPSHOT_DIRECTORY" >&2; exit 2; }
snapshot_dir="$1"
require_dir "$snapshot_dir"
init_stage replay-quant-li
source_ascend_environment
prepend_framework_python
configure_deepep_environment
activate_joint_opp
export ASCEND_RT_VISIBLE_DEVICES="${LI_REPLAY_DEVICES:-0}"
unset SGLANG_NPU_QUANT_LI_DEBUG_DIR
helper="$FRAMEWORK_ROOT/python/sglang/srt/hardware_backend/npu/quant_li_debug.py"
require_file "$helper"
shopt -s nullglob
fixtures=("$snapshot_dir"/rank0-*.pt)
(( ${#fixtures[@]} > 0 )) || { echo "No rank0 fixtures in $snapshot_dir" >&2; exit 2; }
failed=0
read -r -a bindings <<< "${LI_REPLAY_BINDINGS:-custom torch-npu npu-dispatch}"
printf 'case\texit_code\n' > "$RUN_DIR/summary.tsv"
for fixture in "${fixtures[@]}"; do
  stem="$(basename "$fixture" .pt)"
  for binding in "${bindings[@]}"; do
    for mode in eager graph; do
      extra=()
      [[ "$mode" != graph ]] || extra+=(--graph)
      name="$stem-$binding-$mode"
      if run_logged "$name" "$PYTHON_BIN" "$helper" \
        --fixture "$fixture" --binding "$binding" "${extra[@]}" \
        --output "$RUN_DIR/$name.json"; then
        rc=0
      else
        rc=$?
        failed=1
      fi
      printf '%s\t%s\n' "$name" "$rc" | tee -a "$RUN_DIR/summary.tsv"
    done
  done
done
echo "Execution-only diagnostic; no accuracy or performance conclusion."
echo "Report: $RUN_DIR/summary.tsv"
exit "$failed"
