# A5 C8 CopySFA + MTP manual handoff

This directory is a manual validation path, not an A5 result. The code was
prepared without agent access to an Ascend950 host. Every stage writes stdout,
stderr and a numeric exit code beneath a new timestamped run directory.

## Source pair and ownership boundary

Use these two personal-fork checkouts together:

- framework: `feat/a5-copysfa-mtp-radix-20261031`, based on
  `origin/fix/mtp3-draft-hbm-tail-20261031` at
  `4d1fefd6b79f214f908359e032f96ba3a64137ee`;
- kernel: `feat/a5-copysfa-mtp-radix-20261031`, based on the same named remote
  branch at `db77f6fd2db26b2e60a02c63f821121045482835`.

The paired wheel now integrates McZyWu's original A5 LIM OPP with our
A5-tested native CopySFA, retaining independent operator registrations and
the existing metadata fusions. See [JOINT_A5.md](JOINT_A5.md) for the current
build, real LIM-to-CopySFA graph/Q4/fusion tests, manual serving and no-stack
profiling workflow. Rebuild/install this wheel before model-mode preflight;
leave `SGLANG_NPU_A5_LIM_LIBRARY` unset. The OPP is activated before NPU init.
The LIM donor branch name ends in `10-30`, but its integrated HEAD `e53e8d8`
was committed on October 31. It includes the original-OPP/SIMT restoration
`84fa315`, rather than the earlier direct LIM migration. See
`tools/a5_c8/INTEGRATION.json` in the paired kernel checkout for exact provenance.

For the current **40K, C12288, attention DP4/TP2, EP8, MTP3 + graph** launch,
source `article_single_node.env.example` and follow [JOINT_A5.md](JOINT_A5.md).
Its global concurrency is 64 and per-DP graph batch is 16. The direct launch
defaults below remain a DP1 diagnostic configuration. Current A5 joint
operator joint gate passed 52 tests with zero skips/failures/errors in the
user's November 2 A5 run. Whole-model DP4 serving and profiling remain pending;
the earlier donor deployment was Q1 with radix disabled.
After serving checks pass, `AUTO_PROFILE=1 OUTPUT_TOKENS=4096 bash
benchmark/hisparse/a5_copysfa_mtp/55_steady_decode.sh` automatically waits for
continuous full client progress, captures without stacks and validates all
rank traces. See the joint guide for the exact trigger and output paths.

The cross-component address contract is also fixed:

- external LIM emits a logical cache of `C + source-relative-prefix` (`wmc`);
- the framework maps that tail to physical `C + source % 256` slots and rotates
  the last two pages in the HBM block table;
- vLLM-style absolute ring slots are not compatible unless explicitly adjusted;
- every host row is the complete 656-byte payload (FP8 K/V, BF16 RoPE and FP32
  scales); radix aliases share canonical physical host pages, while each live
  request owns an independent `C`-token hot pool.

## Runtime setup

Run inside the intended task-private A5 environment. The scripts source CANN
and ATB first and then prepend the selected framework `python/` directory while
preserving the inherited `PYTHONPATH`. Point `PYTHON_BIN` at the interpreter
containing the selected wheel. Optionally set
`EXPECTED_KERNEL_PACKAGE_ROOT` to make preflight reject any wheel outside a
known private environment.

```bash
export FRAMEWORK_ROOT=/home/y00970600/personal/sglang
export KERNEL_ROOT=/home/y00970600/personal/sgl-kernel-npu
export PYTHON_BIN=/path/to/task-venv/bin/python
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export EXPECTED_KERNEL_PACKAGE_ROOT=/path/to/task-venv
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

Inspect `npu-smi info` and the exact device processes before choosing the list.
None of these scripts kills processes, stops containers, changes THP/sysctls,
or removes global packages.

## Staged execution

Build only the Ascend950 `sgl_kernel_npu` wheel. This calls the repository's
`kernels` target, never `all` or `deepep`, records DeepEP hashes before/after,
and refuses to let the helper install a missing `wheel` build dependency. Set
`INSTALL_PYTHON` only to a task-private interpreter when installation is
wanted; that install uses `--no-deps`.

```bash
bash ./benchmark/hisparse/a5_copysfa_mtp/10_build_kernel.sh
INSTALL_PYTHON="$PYTHON_BIN" bash ./benchmark/hisparse/a5_copysfa_mtp/10_build_kernel.sh
```

Run source/wheel/operator provenance and CPU tests:

```bash
bash ./benchmark/hisparse/a5_copysfa_mtp/00_preflight.sh copy
bash ./benchmark/hisparse/a5_copysfa_mtp/20_test_cpu.sh
```

Run standalone A5 CopySFA numerical tests first if needed. The current combined
wheel already contains McZyWu's original LIM; no separate LIM wheel or DSO is
required. `copy` mode covers CopySFA and packed-tail metadata without invoking
LIM. `model` mode additionally runs the real LIM contract and the four joint
Q1/Q4 eager/graph fusion cases; missing LIM is a hard failure, not a skip:

```bash
bash ./benchmark/hisparse/a5_copysfa_mtp/30_test_a5.sh copy
unset SGLANG_NPU_A5_LIM_LIBRARY
bash ./benchmark/hisparse/a5_copysfa_mtp/00_preflight.sh model
bash ./benchmark/hisparse/a5_copysfa_mtp/30_test_a5.sh model
```

An exit code of zero covers only the named tests. It is not model-serving or
performance evidence.

Compare CopySFA directly with the pinned original vLLM ACLNN/OPP operator
before model serving. This step needs no LIM. First build the offline private
reference as described in the kernel checkout's
`benchmark/copysfa_a5_ab/README.md`, using the same private Python/CANN runtime.
The builder also checks byte-for-byte tiling serialization on CPU. Then run:

```bash
export ORIGINAL_PATHS_ENV=/work/copysfa-original-001/paths.env
bash ./benchmark/hisparse/a5_copysfa_mtp/35_compare_vllm.sh smoke
bash ./benchmark/hisparse/a5_copysfa_mtp/35_compare_vllm.sh formal
```

The formal comparison contains 38 cases: C8192/16384/32384, Q1/Q4,
95/91 selected-prefix C-pool hit rates and eager/graph, plus overlap and
functional boundaries. Accuracy is a hard gate; latency is reported with a
warning above 20% regression by default. It does not block accuracy work.
The actual hit-rate denominators and raw timing samples are retained.

Use kernel `995130e` or later. This includes `be61600`, which fixes the earlier
direct-launch tiling layout (80 serialized bytes with nested padding, rather
than 96), and explicit batch scheduling for the four CopySFA entrypoints.

## Model serving

The launch defaults are target-only LIM + CopySFA, C=16384, FP8 E4M3 KV,
both metadata fusions enabled, radix enabled, and NEXTN steps/top-k/draft
tokens = 3/1/4. Draft-worker offload is disabled by framework policy, so its KV
stays in native HBM; packed target KV owns the separate circular 256-token MTP
tail. The A5 donor defaults are TP/EP 8 and context 67,584, which leaves room
for 65,536 input tokens, 1,024 output tokens and speculative rows. Model path,
TP/EP, devices, concurrency and cache size are runtime inputs. Q4 cache C must
be a 128-aligned value in `[8192, 32384]`.

```bash
export MODEL_PATH=/path/to/model
export TP_SIZE=8 EP_SIZE=8 MAX_RUNNING_REQUESTS=16
export SPARSE_CACHE_SIZE=16384 PORT=30000

# Inspect the exact command without importing a wheel or touching a device.
bash ./benchmark/hisparse/a5_copysfa_mtp/40_launch_server.sh --dry-run

# Model mode rechecks the external LIM ABI before launching.
bash ./benchmark/hisparse/a5_copysfa_mtp/40_launch_server.sh
```

Do not add `--disable-radix-cache`. If memory sizing requires an explicit
`MAX_TOTAL_TOKENS`, set it from the actual model, TP and device capacity rather
than copying a value from another host.

For model integration, run all four `MTP`/`GRAPH` combinations separately:
`0/0`, `0/1`, `1/0`, `1/1`. Both default to 1. Export the selected values
before launching and retain them for the startup-log check, for example
`export MTP=0 GRAPH=0`. MTP=0 omits every speculative argument; MTP=1 uses
NEXTN 3/1/4. GRAPH=0 disables graph capture. Keep radix enabled throughout and
repeat the prefix semantic check for each combination. C8192/16384/32384 are
capacity settings, not guarantees of a measured 95% or 91% model hit rate;
those controlled hit rates belong to the standalone comparison matrix.

While the service is running, validate the allocation-role lines from its log:

```bash
SERVER_LOG=/path/to/launch-run/server.log \
  bash ./benchmark/hisparse/a5_copysfa_mtp/42_check_startup_log.sh
```

This requires every draft allocation to have `offload=False` and positive CKV
storage, and every target allocation to have `offload=True`. C8 embeds RoPE in
the 656-byte CKV record, so its separate KPE allocation is correctly zero.
Server readiness or request count alone is not a pass.

In another shell, first compare deterministic cold and hot-prefix semantics:

```bash
BASE_URL=http://127.0.0.1:30000 \
  bash ./benchmark/hisparse/a5_copysfa_mtp/45_prefix_sanity.sh
```

This builds and saves exactly 65,536 chat-template token IDs with nonce
`73194628` at the beginning and the question at the end. It runs cold1/hot2/hot4
(add hot16 with `PREFIX_COUNTS="1 2 4 16"`), requires every answer to contain
the nonce, and requires every hot response to report `cached_tokens > 0`.
Cold/hot text equality is only diagnostic because native quantization ties can
differ. HTTP success alone is not a pass; review `needle_summary.json`, the
frozen IDs, raw responses, server log and all exit codes.

Run unprofiled serving performance separately from profiling:

```bash
MODEL_PATH="$MODEL_PATH" NUM_PROMPTS=72 CONCURRENCY=16 \
  INPUT_TOKENS=65536 OUTPUT_TOKENS=1024 \
  bash ./benchmark/hisparse/a5_copysfa_mtp/50_perf.sh
```

`50_perf.sh` sends exact token IDs with a 65,408-token shared prefix, validates
all 1,024-token streaming outputs and reported radix hits, and records
request-level TTFT/TPOT plus the full completion count. It runs with no
profiler.

For a decode-only profile, use two terminals. Start the workload first:

```bash
MODEL_PATH="$MODEL_PATH" bash ./benchmark/hisparse/a5_copysfa_mtp/55_steady_decode.sh
```

After all 16 requests emit decode chunks, the client prints
`ALL_STREAMS_ACTIVE count=16`. Only then start a 20-forward profile in the
other terminal. The profiler auto-stops; no flush or warmup occurs inside its
window:

```bash
bash ./benchmark/hisparse/a5_copysfa_mtp/60_profile.sh start
# Only if an active capture must be aborted:
bash ./benchmark/hisparse/a5_copysfa_mtp/60_profile.sh stop
```

Only `50_perf.sh` is the clean serving measurement entry. `60_profile.sh`
does not launch a workload and its profiled timings must not be reported as
unprofiled performance. Confirm the trace contains the intended target BS16
decode markers and 20 forward steps; the HTTP start response and trace-file
existence are not sufficient. A complete handoff report should
include source hashes, wheel/SO hash, exact device list, every exit code,
needle correctness, hot cached-token evidence, completed request count,
throughput/latency metrics, and the raw log/profile paths. Until these manual
stages run, no A5 correctness,
service, graph, radix, throughput or profiling success is claimed.
