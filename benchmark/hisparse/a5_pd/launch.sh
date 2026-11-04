#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PD_ROLE="${1:?Use: launch.sh prefill|decode [--dry-run]}"
shift
[[ "$PD_ROLE" == prefill || "$PD_ROLE" == decode ]] || exit 2
: "${PREFILL_IP:?Set PREFILL_IP to the reachable P-node IPv4 address}"
: "${DECODE_IP:?Set DECODE_IP to the reachable D-node IPv4 address}"
: "${MODEL_PATH:?Set the local A5 GLM-5.2 W4A4C8 model path}"
for address in "$PREFILL_IP" "$DECODE_IP"; do
  [[ "$address" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]] || {
    echo "Replace P/D placeholders with actual IPv4 addresses" >&2; exit 2;
  }
  IFS=. read -r -a octets <<< "$address"
  for octet in "${octets[@]}"; do
    [[ "$octet" =~ ^(0|[1-9][0-9]{0,2})$ ]] && (( 10#$octet <= 255 )) || {
      echo "Invalid P/D IPv4 address: $address" >&2; exit 2;
    }
  done
done

export FRAMEWORK_ROOT="${FRAMEWORK_ROOT:-$(cd "$HERE/../../.." && pwd)}"
export KERNEL_ROOT="${KERNEL_ROOT:-$(dirname "$FRAMEWORK_ROOT")/sgl-kernel-npu}"
export RUN_ROOT="${RUN_ROOT:-$HOME/a5-pd-runs}"
export PYTHON_BIN="${PYTHON_BIN:-python3}"
export EXPECTED_BRANCH=feat/a5-pd-split-kv-20261104
export TP_SIZE="${TP_SIZE:-8}" ATTN_DP_SIZE="${ATTN_DP_SIZE:-4}" EP_SIZE="${EP_SIZE:-8}"
export PER_DP_BATCH_SIZE="${PER_DP_BATCH_SIZE:-16}" GRAPH_BATCH_SIZE="${GRAPH_BATCH_SIZE:-16}"
# Recompute the global admission limit from this invocation's per-DP size;
# a preceding colocated benchmark may have exported a different global limit.
unset MAX_RUNNING_REQUESTS CONCURRENCY NUM_PROMPTS
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export CONTEXT_LENGTH="${CONTEXT_LENGTH:-45056}" CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-4096}"
export MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-1048576}" MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.80}"
export SPARSE_CACHE_SIZE="${SPARSE_CACHE_SIZE:-12288}" MTP=1
export HOST=0.0.0.0 PD_BOOTSTRAP_PORT="${PD_BOOTSTRAP_PORT:-8998}"
export ASCEND_MF_STORE_URL="tcp://$PREFILL_IP:${MF_STORE_PORT:-24669}"
# Same default as the donor A3 recipe. Unsupported MF enums fail explicitly.
export ASCEND_MF_TRANSFER_PROTOCOL="${ASCEND_MF_TRANSFER_PROTOCOL:-device_sdma}"
export MF_HYBM_USE_VMM_SEGMENT=1 SGLANG_NPU_A5_PD_MF=1
export SGL_KERNEL_A5_C8_BACKEND=original
export SGLANG_NPU_PROFILING=0 SGLANG_NPU_USE_MLAPO=0
export no_proxy="localhost,127.0.0.1,::1,$PREFILL_IP,$DECODE_IP${no_proxy:+,$no_proxy}"
export NO_PROXY="$no_proxy"
unset SGLANG_NPU_A5_LIM_LIBRARY
if [[ "$PD_ROLE" == prefill ]]; then
  export SGLANG_HOST_IP="$PREFILL_IP" PORT="${PREFILL_PORT:-30000}" KV_OFFLOAD=0 GRAPH=0
else
  export SGLANG_HOST_IP="$DECODE_IP" PORT="${DECODE_PORT:-30001}" KV_OFFLOAD="${KV_OFFLOAD:-1}" GRAPH=1
fi
if [[ "${1:-}" != --dry-run ]]; then
  source "$HERE/environment.sh"
  "$PYTHON_BIN" "$HERE/doctor.py" --role "$PD_ROLE"
fi
exec bash "$HERE/../a5_copysfa_mtp/40_launch_server.sh" "$@"
