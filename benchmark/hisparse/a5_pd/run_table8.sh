#!/usr/bin/env bash
# A5 P/D comparison: native BS5 vs fused sparse offload BS16.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
role="${1:?Usage: run_table8.sh prefill|router|decode|bench [native|hisparse]}"
shift
run_root=/mnt/share/y00970600/a5-pd-runs/table8-20261105
export PREFILL_IP=141.61.81.109 DECODE_IP=141.61.81.111
export PREFILL_PORT=30000 DECODE_PORT=30001 ROUTER_PORT=8000
export PD_BOOTSTRAP_PORT=8998 MF_STORE_PORT=24669
export FRAMEWORK_ROOT="$(cd "$here/../../.." && pwd)"
export KERNEL_ROOT=/mnt/share/y00970600/personal/sgl-kernel-npu
export MODEL_PATH=/mnt/share/w00936111/GLM-5.2-W4A4C8-mxfp4-A5-0731
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TP_SIZE=8 ATTN_DP_SIZE=8 EP_SIZE=8
export CONTEXT_LENGTH=106496 CHUNKED_PREFILL_SIZE=4096 SPARSE_CACHE_SIZE=16000
export MAX_TOTAL_TOKENS=262144 MEM_FRACTION_STATIC=0.80
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=64
export PD_DECODE_RADIX=1
unset SGLANG_TEST_RETRACT SGLANG_TEST_RETRACT_INTERVAL
export no_proxy="localhost,127.0.0.1,::1,$PREFILL_IP,$DECODE_IP${no_proxy:+,$no_proxy}"
export NO_PROXY="$no_proxy"

prepare_run_dir() {
  local name="$1"
  export RUN_DIR="$run_root/$name"
  if [[ "${2:-}" == --dry-run ]]; then
    return
  fi
  mkdir -p "$run_root"
  RUN_DIR="$(mktemp -d "$run_root/$name-XXXXXX")"
  if [[ "$name" == decode-* ]]; then
    printf '%s\n' "${RUN_DIR##*/}" > "$RUN_DIR/latest"
    mv -Tf "$RUN_DIR/latest" "$run_root/$name.latest"
  fi
  echo "run_dir=$RUN_DIR"
}

case "$role" in
  prefill)
    export PER_DP_BATCH_SIZE=16 GRAPH_BATCH_SIZE=16
    prepare_run_dir prefill "${1:-}"
    exec bash "$here/launch.sh" prefill "$@"
    ;;
  router)
    exec bash "$here/router.sh" --policy round_robin --disable-retries "$@"
    ;;
  decode|bench)
    mode="${1:?Choose native or hisparse}"
    shift
    case "$mode" in
      native) bs=5; offload=0 ;;
      hisparse) bs=16; offload=1 ;;
      *) echo 'Choose native or hisparse' >&2; exit 2 ;;
    esac
    if [[ "$role" == decode ]]; then
      export PER_DP_BATCH_SIZE="$bs" GRAPH_BATCH_SIZE="$bs" KV_OFFLOAD="$offload"
      prepare_run_dir "decode-$mode" "${1:-}"
      exec bash "$here/launch.sh" decode "$@"
    fi
    source "$here/../a5_copysfa_mtp/common.sh"
    source_ascend_environment
    prepend_framework_python
    exec "$PYTHON_BIN" "$here/bench_table8.py" "$mode" \
      --root "$run_root" --model-path "$MODEL_PATH" \
      --prefill-url "http://$PREFILL_IP:$PREFILL_PORT" \
      --decode-url "http://$DECODE_IP:$DECODE_PORT" \
      --base-url "http://127.0.0.1:$ROUTER_PORT" "$@"
    ;;
  *) echo 'Usage: run_table8.sh prefill|router|decode|bench [native|hisparse]' >&2; exit 2 ;;
esac
