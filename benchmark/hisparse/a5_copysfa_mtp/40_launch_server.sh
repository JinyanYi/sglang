#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

MODEL_PATH="${MODEL_PATH:-}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-a5-c8-copysfa-mtp}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-30000}"
configure_topology
CONTEXT_LENGTH="${CONTEXT_LENGTH:-67584}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-2048}"
MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-32768}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.90}"
QUANTIZATION="${QUANTIZATION:-modelslim}"
SPARSE_CACHE_SIZE="${SPARSE_CACHE_SIZE:-16384}"
GRAPH="${GRAPH:-1}"
MTP="${MTP:-1}"
KV_OFFLOAD="${KV_OFFLOAD:-1}"
configure_kv_policy
DEEPEP_MODE="${DEEPEP_MODE:-auto}"
export DEEPEP_HYBRID_DEPLOYMENT="${DEEPEP_HYBRID_DEPLOYMENT:-1}"
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-1000}"

for name in PORT TP_SIZE EP_SIZE MAX_RUNNING_REQUESTS CONTEXT_LENGTH CHUNKED_PREFILL_SIZE MAX_PREFILL_TOKENS SPARSE_CACHE_SIZE GRAPH MTP; do
  [[ "${!name}" =~ ^[0-9]+$ ]] || { echo "$name must be an integer" >&2; exit 2; }
done
[[ "$GRAPH" == 0 || "$GRAPH" == 1 ]] || { echo "GRAPH must be 0 or 1" >&2; exit 2; }
[[ "$MTP" == 0 || "$MTP" == 1 ]] || { echo "MTP must be 0 or 1" >&2; exit 2; }
dispatch_batch="$PER_DP_BATCH_SIZE"
(( GRAPH == 0 )) || dispatch_batch="$GRAPH_BATCH_SIZE"
dispatch_min=$((dispatch_batch * (1 + 3 * MTP)))
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK="${SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK:-$dispatch_min}"
[[ "$DEEPEP_MODE" == auto && "$DEEPEP_HYBRID_DEPLOYMENT" == 1 ]] || {
  echo "This joint launcher requires DeepEP auto + hybrid" >&2; exit 2;
}
[[ "$SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK" =~ ^[1-9][0-9]*$ ]] &&
  (( SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK >= dispatch_min && SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK <= 1024 )) || {
  echo "DeepEP dispatch capacity must cover per-DP padded Q tokens ($dispatch_min), at most 1024" >&2; exit 2;
}
if (( KV_OFFLOAD )); then
  (( SPARSE_CACHE_SIZE >= 8192 && SPARSE_CACHE_SIZE <= 32384 && SPARSE_CACHE_SIZE % 128 == 0 )) || {
  echo "A5 Q4 SPARSE_CACHE_SIZE must be a 128-aligned value in [8192, 32384]" >&2
  exit 2
  }
fi

args=("$PYTHON_BIN" -m sglang.launch_server
  --model-path "$MODEL_PATH" --served-model-name "$SERVED_MODEL_NAME"
  --attention-backend ascend --device npu
  --tp-size "$TP_SIZE" --ep-size "$EP_SIZE" --pp-size 1 --attn-cp-size 1
  --attn-dp-size "$ATTN_DP_SIZE" --dp-size 1
  --trust-remote-code
  --dtype bfloat16 --kv-cache-dtype fp8_e4m3 --page-size 128
  --chunked-prefill-size "$CHUNKED_PREFILL_SIZE" --max-prefill-tokens "$MAX_PREFILL_TOKENS"
  --max-running-requests "$MAX_RUNNING_REQUESTS"
  --context-length "$CONTEXT_LENGTH" --mem-fraction-static "$MEM_FRACTION_STATIC"
  --moe-a2a-backend deepep --deepep-mode "$DEEPEP_MODE"
  --random-seed 42 --enable-metrics --host "$HOST" --port "$PORT")
[[ -z "$QUANTIZATION" || "$QUANTIZATION" == none ]] || args+=(--quantization "$QUANTIZATION")
[[ -z "${MAX_TOTAL_TOKENS:-}" ]] || args+=(--max-total-tokens "$MAX_TOTAL_TOKENS")
if (( MTP )); then
  args+=(--speculative-algorithm NEXTN --speculative-num-steps 3
    --speculative-eagle-topk 1 --speculative-num-draft-tokens 4)
fi
if (( GRAPH )); then
  args+=(--cuda-graph-bs-decode "$GRAPH_BATCH_SIZE")
else
  args+=(--disable-cuda-graph)
fi

# Optional two-node P/D handoff. The colocated invocation remains unchanged.
case "${PD_ROLE:-null}" in
  null) ;;
  prefill|decode)
    args+=(--disaggregation-mode "$PD_ROLE"
      --disaggregation-transfer-backend ascend
      --disaggregation-bootstrap-port "${PD_BOOTSTRAP_PORT:-8998}"
      --enable-cache-report)
    if [[ "$PD_ROLE" == decode ]]; then
      args+=(--disaggregation-decode-extra-slots 0
        --disaggregation-decode-retraction-backup cpu_tensor)
      [[ "${PD_DECODE_RADIX:-1}" == 0 ]] || args+=(--disaggregation-decode-enable-radix-cache)
      if [[ "${PD_BUFFERED_WEIGHT_LOAD:-0}" == 1 ]]; then
        args+=(--weight-loader-disable-mmap
          --model-loader-extra-config '{"enable_multithread_load":false}')
      fi
    fi
    ;;
  *) echo "PD_ROLE must be null, prefill or decode" >&2; exit 2 ;;
esac

if [[ "${1:-}" == --dry-run ]]; then
  printf 'world=%s attention_DP=%s attention_TP=%s EP=%s per_DP_batch=%s global_requests=%s graph_bucket=%s DeepEP_dispatch_capacity=%s\n' "$TP_SIZE" "$ATTN_DP_SIZE" "$((TP_SIZE / ATTN_DP_SIZE))" "$EP_SIZE" "$PER_DP_BATCH_SIZE" "$MAX_RUNNING_REQUESTS" "$GRAPH_BATCH_SIZE" "$SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK"
  printf 'radix=enabled KV_OFFLOAD=%s indexer=%s copy_sfa=%s metadata_fusions=%s MTP=%s GRAPH=%s draft_offload=disabled profile=off\n' "$KV_OFFLOAD" "$SGLANG_NPU_SPARSE_KV_INDEXER" "$SGLANG_NPU_ENABLE_COPY_SFA" "$SGLANG_NPU_FUSED_LIM_PRE_METADATA" "$MTP" "$GRAPH"
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
configure_deepep_environment
activate_joint_opp
export FRAMEWORK_ROOT

# Offload, when enabled, is target-only; draft KV remains in HBM in both modes.
# configure_kv_policy already set every sparse switch explicitly, including
# the native baseline case, before Python can cache environment values.
export SGLANG_NPU_PROFILING=0
export SGLANG_NPU_USE_MLAPO=0
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$RUN_DIR/cache/inductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$RUN_DIR/cache/triton}"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

printf 'world=%s attention_DP=%s attention_TP=%s EP=%s per_DP_batch=%s global_requests=%s graph_bucket=%s MTP=%s GRAPH=%s DeepEP_dispatch_capacity=%s HCCL_BUFFSIZE=%s\n' \
  "$TP_SIZE" "$ATTN_DP_SIZE" "$((TP_SIZE / ATTN_DP_SIZE))" "$EP_SIZE" "$PER_DP_BATCH_SIZE" "$MAX_RUNNING_REQUESTS" "$GRAPH_BATCH_SIZE" "$MTP" "$GRAPH" "$SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK" "$HCCL_BUFFSIZE" | tee "$RUN_DIR/topology.log"
printf 'KV_OFFLOAD=%s indexer=%s copy_sfa=%s metadata_fusions=%s\n' \
  "$KV_OFFLOAD" "$SGLANG_NPU_SPARSE_KV_INDEXER" "$SGLANG_NPU_ENABLE_COPY_SFA" "$SGLANG_NPU_FUSED_LIM_PRE_METADATA" | tee "$RUN_DIR/kv-policy.log"

run_logged deepep_libraries "$PYTHON_BIN" "$HANDOFF_DIR/check_deepep_libraries.py"
preflight_mode=model
(( KV_OFFLOAD != 0 )) || preflight_mode=copy
run_logged preflight "$PYTHON_BIN" "$HANDOFF_DIR/preflight.py" --mode "$preflight_mode"
printf '%q ' "${args[@]}" | tee "$RUN_DIR/server.command"
printf '\n' | tee -a "$RUN_DIR/server.command"
run_logged server "${args[@]}"
