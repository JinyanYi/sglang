#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
model=${1:-/home/weights/Qwen3-0.6B}
devices=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3}
pp_size=${PP_TEST_PP_SIZE:-4}
port=${PP_TEST_PORT:-30088}
log_dir=${PP_TEST_LOG_DIR:-"$PWD/pp4-e2e-$(date +%Y%m%d-%H%M%S)"}
disable_graph=${PP_TEST_DISABLE_GRAPH:-0}
use_fia=${ASCEND_USE_FIA:-0}
graph_bs=${PP_TEST_GRAPH_BS:-}
if [[ "$disable_graph" != 0 && "$disable_graph" != 1 ]]; then
  echo "PP_TEST_DISABLE_GRAPH must be 0 or 1" >&2
  exit 2
fi
if [[ "$use_fia" != 0 && "$use_fia" != 1 ]]; then
  echo "ASCEND_USE_FIA must be 0 or 1" >&2
  exit 2
fi
graph_args=()
if [[ "$disable_graph" == 1 ]]; then
  graph_args+=(--disable-cuda-graph)
fi
if [[ -n "$graph_bs" ]]; then
  if [[ "$disable_graph" == 1 || ! "$graph_bs" =~ ^[1-9][0-9]*(,[1-9][0-9]*)*$ ]]; then
    echo "PP_TEST_GRAPH_BS requires graph on and comma-separated positive integers" >&2
    exit 2
  fi
  graph_args+=(--cuda-graph-config "{\"decode\":{\"bs\":[${graph_bs}]}}")
fi
IFS=, read -r -a device_list <<< "$devices"
if [[ "$pp_size" != 1 && "$pp_size" != 2 && "$pp_size" != 4 ]]; then
  echo "PP_TEST_PP_SIZE must be 1, 2, or 4" >&2
  exit 2
fi
if (( ${#device_list[@]} != pp_size )); then
  echo "Select exactly $pp_size free NPUs in ASCEND_RT_VISIBLE_DEVICES" >&2
  exit 2
fi
if [[ ! -f "$model/config.json" ]]; then
  echo "Model config not found: $model/config.json" >&2
  exit 2
fi
if ! python - "$port" <<'PY'
import socket
import sys

with socket.socket() as sock:
    sock.bind(("127.0.0.1", int(sys.argv[1])))
PY
then
  echo "Port $port is already in use; set PP_TEST_PORT to another free port" >&2
  exit 2
fi

mkdir -p "$log_dir"
server_log="$log_dir/server.log"
client_log="$log_dir/client.jsonl"
server_pid=
server_pgid=
cleanup() {
  local rc=$?
  trap - EXIT
  if [[ -n "$server_pid" ]]; then
    if [[ "$server_pgid" == "$server_pid" ]]; then
      # Only signal the private group that this launch actually created.
      kill -TERM -- "-$server_pgid" 2>/dev/null || true
      sleep 5
      kill -KILL -- "-$server_pgid" 2>/dev/null || true
    else
      kill -TERM "$server_pid" 2>/dev/null || true
    fi
    wait "$server_pid" 2>/dev/null || true
  fi
  echo "Logs: $log_dir"
  exit "$rc"
}
trap cleanup EXIT

export PYTHONPATH="$repo_root/python${PYTHONPATH:+:$PYTHONPATH}"
if commit=$(git -C "$repo_root" rev-parse --short HEAD 2>/dev/null); then
  echo "Source commit: $commit"
else
  echo "Source commit: unavailable (copied test checkout)"
fi
echo "Model: $model; TP=1 PP=$pp_size; NPUs: $devices; port: $port"
echo "CPU relay requested=1 (active only for NPU PP>2); ASCEND_USE_FIA=$use_fia; disable graph=$disable_graph"
echo "Decode graph capture batch sizes: ${graph_bs:-default}"
echo "Starting server; live output is also saved to $server_log"
ASCEND_RT_VISIBLE_DEVICES="$devices" \
ASCEND_USE_FIA="$use_fia" \
SGLANG_PP_OUTPUT_VIA_CPU=1 \
SGLANG_PP_SKIP_PURE_CHUNKED_OUTPUT_COMM=0 \
SGLANG_ENABLE_PP_SPEC=0 \
setsid python -m sglang.launch_server \
  --model-path "$model" \
  --host 127.0.0.1 --port "$port" \
  --tp-size 1 --pp-size "$pp_size" \
  --context-length 2048 \
  --max-total-tokens 4096 \
  --mem-fraction-static 0.15 \
  --chunked-prefill-size 256 \
  "${graph_args[@]}" \
  > >(tee "$server_log") 2>&1 &
server_pid=$!
server_pgid=$(ps -o pgid= -p "$server_pid" | tr -d '[:space:]')
if [[ "$server_pgid" != "$server_pid" ]]; then
  echo "Could not verify the server's private process group" >&2
  exit 1
fi

ready=0
for ((attempt=0; attempt<180; attempt++)); do
  if curl --silent --show-error --fail --max-time 2 \
      "http://127.0.0.1:$port/ready" >/dev/null 2>&1; then
    ready=1
    break
  fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    echo "Server exited before readiness; inspect $server_log" >&2
    exit 1
  fi
  sleep 2
done
if (( ready == 0 )); then
  echo "Server did not become ready within 360 seconds" >&2
  exit 1
fi

echo "Server ready; sending sequential and concurrent requests"
timeout --signal=TERM --kill-after=10s 420s \
  python "$repo_root/test/manual/pp_output_cpu_relay_e2e_client.py" \
  --url "http://127.0.0.1:$port" --out "$client_log"
curl --silent --show-error --fail --max-time 30 \
  "http://127.0.0.1:$port/health_generate" >/dev/null
echo "PASS: PP$pp_size model requests and final health generation completed"
