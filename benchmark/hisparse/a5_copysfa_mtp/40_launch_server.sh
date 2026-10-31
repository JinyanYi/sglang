#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

MODEL_PATH="${MODEL_PATH:-}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-a5-c8-copysfa-mtp}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-30000}"
TP_SIZE="${TP_SIZE:-8}"
EP_SIZE="${EP_SIZE:-$TP_SIZE}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-16}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-67584}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-2048}"
MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-32768}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.90}"
QUANTIZATION="${QUANTIZATION:-modelslim}"
SPARSE_CACHE_SIZE="${SPARSE_CACHE_SIZE:-16384}"
GRAPH="${GRAPH:-1}"
MTP="${MTP:-1}"

for name in PORT TP_SIZE EP_SIZE MAX_RUNNING_REQUESTS CONTEXT_LENGTH CHUNKED_PREFILL_SIZE MAX_PREFILL_TOKENS SPARSE_CACHE_SIZE GRAPH MTP; do
  [[ "${!name}" =~ ^[0-9]+$ ]] || { echo "$name must be an integer" >&2; exit 2; }
done
[[ "$GRAPH" == 0 || "$GRAPH" == 1 ]] || { echo "GRAPH must be 0 or 1" >&2; exit 2; }
[[ "$MTP" == 0 || "$MTP" == 1 ]] || { echo "MTP must be 0 or 1" >&2; exit 2; }
(( SPARSE_CACHE_SIZE >= 8192 && SPARSE_CACHE_SIZE <= 32384 && SPARSE_CACHE_SIZE % 128 == 0 )) || {
  echo "A5 Q4 SPARSE_CACHE_SIZE must be a 128-aligned value in [8192, 32384]" >&2
  exit 2
}

args=("$PYTHON_BIN" -m sglang.launch_server
  --model-path "$MODEL_PATH" --served-model-name "$SERVED_MODEL_NAME"
  --attention-backend ascend --device npu
  --tp-size "$TP_SIZE" --ep-size "$EP_SIZE" --pp-size 1 --attn-cp-size 1
  --trust-remote-code
  --dtype bfloat16 --kv-cache-dtype fp8_e4m3 --page-size 128
  --chunked-prefill-size "$CHUNKED_PREFILL_SIZE" --max-prefill-tokens "$MAX_PREFILL_TOKENS"
  --max-running-requests "$MAX_RUNNING_REQUESTS"
  --context-length "$CONTEXT_LENGTH" --mem-fraction-static "$MEM_FRACTION_STATIC"
  --moe-a2a-backend deepep
  --random-seed 42 --enable-metrics --host "$HOST" --port "$PORT")
[[ -z "$QUANTIZATION" || "$QUANTIZATION" == none ]] || args+=(--quantization "$QUANTIZATION")
[[ -z "${MAX_TOTAL_TOKENS:-}" ]] || args+=(--max-total-tokens "$MAX_TOTAL_TOKENS")
if (( MTP )); then
  args+=(--speculative-algorithm NEXTN --speculative-num-steps 3
    --speculative-eagle-topk 1 --speculative-num-draft-tokens 4)
fi
if (( GRAPH )); then
  args+=(--cuda-graph-bs-decode "$MAX_RUNNING_REQUESTS")
else
  args+=(--disable-cuda-graph)
fi

if [[ "${1:-}" == --dry-run ]]; then
  printf 'radix=enabled target_offload=lim+copysfa kv_cache=fp8_e4m3 C=%s metadata_fusions=1 MTP=%s GRAPH=%s draft_offload=disabled profile=off\n' "$SPARSE_CACHE_SIZE" "$MTP" "$GRAPH"
  printf '%q ' "${args[@]}"
  printf '\n'
  exit 0
fi
[[ $# == 0 ]] || { echo "Only --dry-run is accepted." >&2; exit 2; }

require_value MODEL_PATH
require_value ASCEND_RT_VISIBLE_DEVICES
init_stage launch-server
{
  check_checkout framework "$FRAMEWORK_ROOT"
  check_checkout kernel "$KERNEL_ROOT"
} 2>&1 | tee "$RUN_DIR/source.log"
source_ascend_environment
prepend_framework_python
export FRAMEWORK_ROOT

# C8 offload is target-only. The draft worker retains its native HBM KV pool;
# the packed target additionally owns a circular 256-token device tail.
export SGLANG_NPU_ENABLE_SPARSE_KV_OFFLOAD=1
export SGLANG_NPU_SPARSE_KV_INDEXER=lim
export SGLANG_NPU_ENABLE_COPY_SFA=1
export SGLANG_NPU_SPARSE_KV_CACHE_SIZE="$SPARSE_CACHE_SIZE"
export SGLANG_NPU_FUSED_LIM_PRE_METADATA=1
export SGLANG_NPU_FUSED_LIM_COPY_SFA_METADATA=1
export SGLANG_NPU_PROFILING=0
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$RUN_DIR/cache/inductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$RUN_DIR/cache/triton}"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

run_logged preflight "$PYTHON_BIN" "$HANDOFF_DIR/preflight.py" --mode model
printf '%q ' "${args[@]}" | tee "$RUN_DIR/server.command"
printf '\n' | tee -a "$RUN_DIR/server.command"
run_logged server "${args[@]}"
