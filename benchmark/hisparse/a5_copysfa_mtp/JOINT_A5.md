# A5 original LIM + native CopySFA joint handoff

Both personal repositories use `feat/a5-copysfa-mtp-radix-20261031`.
The integrated kernel commit is `e17cad92960652e0c9c2e935771e10694b30a090`.
The kernel integrates McZyWu `e53e8d8df1d4f25be7c0d40b6fa8234c8750f1fc`
(original LIM PR18383 `7d5a4cd9d7e8c4e2e4f0f6415a71f0e80edcff8e`).
CopySFA retains the A5-tested source from `907a286`, including the 80-byte tiling
ABI and explicit batch scheduling. The two production operators have separate
registrations in one wheel. No vLLM Python installation or external LIM DSO
is required. The private LIM OPP is bundled and hash checked.

This is source/build/test preparation. The agent cannot access A5. Installing
the new wheel and obtaining actual A5 results are required before acceptance.
A3 results and earlier standalone CopySFA results do not establish this joint
A5 implementation's graph/MTP behavior.

## 1. Update, build and install in the same A5 container

Select an unused device for operator tests. Preserve the existing environment
and old wheel/measurements. Use the same Python/CANN/Torch ABI throughout.

```bash
export FRAMEWORK_ROOT=/mnt/share/y00970600/personal/sglang
export KERNEL_ROOT=/mnt/share/y00970600/personal/sgl-kernel-npu
export PYTHON_BIN="$(command -v python3)"
export ASCEND_RT_VISIBLE_DEVICES=0
export RUN_ROOT=/mnt/share/y00970600/a5-joint-runs
for repo in "$FRAMEWORK_ROOT" "$KERNEL_ROOT"; do
  git -C "$repo" fetch origin feat/a5-copysfa-mtp-radix-20261031
  git -C "$repo" switch feat/a5-copysfa-mtp-radix-20261031
  git -C "$repo" merge --ff-only origin/feat/a5-copysfa-mtp-radix-20261031
done
unset SGLANG_NPU_A5_LIM_LIBRARY
cd "$FRAMEWORK_ROOT"
INSTALL_PYTHON="$PYTHON_BIN" bash benchmark/hisparse/a5_copysfa_mtp/10_build_kernel.sh
bash benchmark/hisparse/a5_copysfa_mtp/30_test_a5.sh model
```

The old CopySFA-only wheel must be rebuilt for this integration. Only kernels
are built; existing DeepEP files are checked for changes. The original LIM
build goes under `/tmp/sgl-a5-lim-*/build` to avoid shared-filesystem clock skew.
To reuse that completed LIM build on a subsequent build, set
`SGL_KERNEL_A5_C8_PREBUILT` to that exact build directory. Do not reuse the
CopySFA reference build as a LIM build. Do not source generated files after a
failed build. The framework shell stages activate the installed OPP before
NPU initialization via `tools/a5_c8/env.py`.

The mandatory joint test is `test_a5_lim_copysfa_joint.py`: Q1/Q4 x eager/graph,
real Triton pre-metadata -> original LIM -> real Triton CopySFA metadata ->
native CopySFA, with cold/warm/request-reuse phases at stable Tensor addresses.
It validates causal IDs, tail slots, the full HBM byte image and CPU attention
(atol0.008/rtol0.05/relative-L2<=0.08). The existing metadata tests separately
cover padding, shared canonical pages and changed-input graph replay.
The JUnit gate requires all four joint cases, explicitly including graph/Q4,
and zero skipped/failed/error cases. Retain all logs, XML and `joint-gate.json`.

## 2. Start the model manually in terminal A

Use a real A5 model path and the appropriate already-working DeepEP/HCCL
container environment. The first supported topology is TP8/EP8, DP1/PP1/CP1.
The article's DP4TP2 setup is not this launch topology. No script starts or
kills another service automatically.

```bash
export FRAMEWORK_ROOT=/mnt/share/y00970600/personal/sglang
export KERNEL_ROOT=/mnt/share/y00970600/personal/sgl-kernel-npu
export PYTHON_BIN="$(command -v python3)"
export RUN_ROOT=/mnt/share/y00970600/a5-joint-runs
export MODEL_PATH=/path/to/your/A5/model
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TP_SIZE=8 EP_SIZE=8 PORT=30000
export SPARSE_CACHE_SIZE=16384 MAX_RUNNING_REQUESTS=16
export MTP=1 GRAPH=1
cd "$FRAMEWORK_ROOT"
bash benchmark/hisparse/a5_copysfa_mtp/40_launch_server.sh
```

Both metadata fusion switches are on. Radix stays enabled. Draft KV remains
HBM resident; the target owns the C pool and 256-token MTP tail. Run the
startup-log checker in `42_check_startup_log.sh` against the saved server log.
A ready endpoint is not model acceptance. Repeat manually with MTP/GRAPH
`0/0`, `0/1`, `1/0`, `1/1`, using a fresh output directory per configuration.
Stop only your own server between configurations. Quantization defaults to
modelslim; adjust `QUANTIZATION` only for the model actually being loaded.

## 3. Correctness and unprofiled throughput in terminal B

Export the same framework/model/port/RUN_ROOT variables in this terminal.

```bash
cd "$FRAMEWORK_ROOT"
bash benchmark/hisparse/a5_copysfa_mtp/45_prefix_sanity.sh
bash benchmark/hisparse/a5_copysfa_mtp/50_perf.sh
```

The workload verifies exact output lengths, saves every request's TTFT/TPOT,
requires radix hits, and records live client concurrency. Radix cached_tokens
is not C-pool hit rate. It saves metrics before/after the measured interval,
including acceptance gauges where exposed. They are per-log-interval gauges,
not cumulative acceptance counters. Use acceptance measured for the profiled
window when deriving article throughput. Controlled95/91 C-hit tests remain
standalone tests, not claims about the model's actual hit distribution.

## 4. One no-stack capture: terminals B and C

Keep the server's normal launch. Run the sustained workload in B:

```bash
OUTPUT_TOKENS=4096 NUM_PROMPTS=16 CONCURRENCY=16 \
  bash benchmark/hisparse/a5_copysfa_mtp/55_steady_decode.sh
```

After `ALL_STREAMS_ACTIVE count=16`, run in C (same exports):

```bash
PROFILE_STEPS=20 TP_SIZE=8 CONCURRENCY=16 \
  bash benchmark/hisparse/a5_copysfa_mtp/60_profile.sh start
```

The request receipt explicitly records `with_stack=false`. It waits for eight
complete rank trace files, device kernels and at least20 target DECODE/VERIFY
markers with BS16. DRAFT markers do not count. A truncated/empty trace, missing
rank, short batch or timeout returns nonzero. Client live streams alone do
not prove the server's batch size. Check rank/device mapping against the
server log; trace paths and device process identities are retained.

## 5. Article accounting

The user-local reference is `C:/Users/Administrator/Desktop/work/Offload.txt`,
"decode KVcache low-latency offload practice", section5.1.4. The analyzer
reports per-rank LI/LIM, copy, CopySFA, SFA and other kernel duration sums and
counts, normalized by actual target markers. Those sums may overlap; normal
SFA/other may also contain draft work and are labeled accordingly.

Exact78-layer target elapsed and other-target elapsed need device target
windows, not CPU enqueue-span durations. Export a CSV from MindStudio with
`trace,start_us,end_us`; trace is the relative `trace_view.json` path beneath
PROFILE_DIR and each row spans first-to-last target-layer device task for one
verified round. Include every captured target round for each rank. The
analyzer rejects count mismatches/overlapping windows, derives target elapsed
and consecutive-round ITL, and subtracts the union of named component device
intervals to compute residual time without double counting overlaps.

```bash
export PROFILE_DIR=/path/printed/by/the/capture/script/profile
# These two inputs are optional; absent inputs remain null, never fabricated.
export TARGET_WINDOWS=/path/to/validated-target-windows.csv
export TOKENS_PER_ROUND=3.0  # replace with observed output tokens per round
bash benchmark/hisparse/a5_copysfa_mtp/65_analyze_profile.sh
```

TPOT=ITL/output_tokens_per_round; TPS_per_DP=BS/TPOT_ms*1000. This launch is
DP1, so systemTPS equals TPS_per_DP; never multiply by TP8. The article's
3.0 tokens/round is an assumption, not a consequence of enabling MTP3. Without
validated device windows/acceptance, the report leaves these derived fields
null and lists missing inputs. Compare unprofiled serving throughput separately.

Next acceptance order: joint graph/Q4/fusion tests, four model combinations
with radix, unprofiled throughput, then this single no-stack profiling capture.
