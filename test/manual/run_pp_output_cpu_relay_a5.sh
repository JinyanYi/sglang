#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
devices=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3}
IFS=, read -r -a device_list <<< "$devices"
if (( ${#device_list[@]} < 3 )); then
  echo "Select at least three free NPUs in ASCEND_RT_VISIBLE_DEVICES" >&2
  exit 2
fi

log=${1:-"$PWD/pp-output-cpu-relay-$(date +%Y%m%d-%H%M%S).log"}
echo "Using NPUs $devices; log: $log"
ASCEND_RT_VISIBLE_DEVICES="$devices" \
SGLANG_PP_OUTPUT_VIA_CPU=1 \
SGLANG_PP_SKIP_PURE_CHUNKED_OUTPUT_COMM=0 \
timeout --signal=TERM --kill-after=10s 180s \
  torchrun --standalone --nproc-per-node="${#device_list[@]}" --max-restarts=0 \
  "$repo_root/test/manual/pp_output_cpu_relay_probe.py" \
  --iterations 20 --sizes 8,4096 --timeout 60 2>&1 | tee "$log"
echo "PASS: every rank completed the scheduler output-ring byte checks"
