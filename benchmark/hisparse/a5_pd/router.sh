#!/usr/bin/env bash
set -euo pipefail
: "${PREFILL_IP:?Set PREFILL_IP}"
: "${DECODE_IP:?Set DECODE_IP}"
export no_proxy="localhost,127.0.0.1,::1,$PREFILL_IP,$DECODE_IP${no_proxy:+,$no_proxy}"
export NO_PROXY="$no_proxy"
exec "${PYTHON_BIN:-python3}" -m sglang_router.launch_router \
  --pd-disaggregation --dp-aware --prefill "http://$PREFILL_IP:${PREFILL_PORT:-30000}" \
  --decode "http://$DECODE_IP:${DECODE_PORT:-30001}" \
  --host 0.0.0.0 --port "${ROUTER_PORT:-8000}"
