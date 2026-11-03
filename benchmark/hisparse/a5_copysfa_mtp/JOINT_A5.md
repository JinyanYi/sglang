# A5 original LIM + native CopySFA joint handoff

Both personal repositories use `feat/a5-copysfa-mtp-radix-20261031`.
The original integration commit is `e17cad92960652e0c9c2e935771e10694b30a090`.
The kernel integrates McZyWu `e53e8d8df1d4f25be7c0d40b6fa8234c8750f1fc`
(original LIM PR18383 `7d5a4cd9d7e8c4e2e4f0f6415a71f0e80edcff8e`).
The donor branch is `feat/hisparse-c8-a5-original-10-30`, but its HEAD was
committed on **2026-10-31 at 17:08:59 +08:00**, not October 30. It includes
`84fa315` restoring the original OPP execution path: the earlier direct
migration disabled the SIMT paths that depend on OPP UB launch settings.
Our build accepts only `original`; it does not fall back to that old direct
LIM. A fresh remote-head check on November 2 still returned `e53e8d8`.
CopySFA retains the A5-tested source from `907a286`, including the 80-byte tiling
ABI and explicit batch scheduling. The two production operators have separate
registrations in one wheel. No vLLM Python installation or external LIM DSO
is required. The private LIM OPP is bundled and hash checked.

The agent cannot access A5. User-run evidence on November 2, stage
`20261102T111122Z-test-a5-model`, reports 25 framework tests passed and a joint
gate with 52 total tests, zero failures/errors/skips and no missing joint cases.
This covers the real Q1/Q4 eager/graph operator/fusion tests. The excerpt does
not include checkout/SO hashes; retain that stage's source and preflight logs
with its JUnits. Whole-model accuracy, DP4 serving and profiling still require
separate validation. The automatic profiling controller has CPU coverage;
this is its first A5 whole-model application.

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

Use the appropriate already-working A5 DeepEP/HCCL container environment.
The article preset requests eight devices, attention DP4/TP2 and EP8.
In this SGLang revision that means `--tp-size 8 --attn-dp-size 4 --ep-size 8`
with replica `--dp-size 1`. Do not set `--tp-size 2` or start four replicas.
BS16 is **per attention-DP worker**: the global request limit and client
concurrency are 64, while the graph bucket is 16. This topology is prepared,
not A5 runtime validated. No script kills another service automatically.

```bash
cd /mnt/share/y00970600/personal/sglang
source benchmark/hisparse/a5_copysfa_mtp/article_single_node.env.example
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
bash benchmark/hisparse/a5_copysfa_mtp/40_launch_server.sh --dry-run
bash benchmark/hisparse/a5_copysfa_mtp/40_launch_server.sh
```

Both metadata fusion switches are on. Radix stays enabled. Draft KV remains
HBM resident; the target owns the C pool and 256-token MTP tail. Run the
startup-log checker in `42_check_startup_log.sh` against the saved server log.
A ready endpoint is not model acceptance. Repeat manually with MTP/GRAPH
`0/0`, `0/1`, `1/0`, `1/1`, using a fresh output directory per configuration.
Stop only your own server between configurations. Quantization defaults to
modelslim; adjust `QUANTIZATION` only for the model actually being loaded.
The preset uses the supplied checkpoint directory
`/mnt/share/w00936111/GLM-5.2-W4A4C8-mxfp4-A5-0731`; its actual model identity
and MTP head count still need checking from that directory's `config.json`.
The agent cannot inspect this A5-only directory. Keep model weights unchanged.

DeepEP auto/hybrid uses the installed `deep_ep` wheel and its own vendor OPP.
The launcher preserves it alongside the LIM OPP and provisions dispatch
capacity for at least 16*4=64 Q4 tokens per DP. The old non-MTP capacity of
32 must not be copied into this preset. `HCCL_BUFFSIZE` defaults to the donor
recipe's 1000 MiB and remains overridable; this is not a measured optimum.
The kernel build stage does not rebuild DeepEP. A container without the
matching installed A5 DeepEP wheel needs that dependency prepared first.

## 3. Correctness and unprofiled throughput in terminal B

Source `article_single_node.env.example` in every terminal so the topology,
model, batch and MTP/graph switches match the server.
ModelSlim supports the W4A4_MXFP4 checkpoint scheme in this framework. The
initial workload uses 40000 input tokens, C12288 and BS16 per DP. This is a
shared-prefix/radix integration workload; article throughput equivalence is
not established merely by matching those dimensions.

```bash
cd /mnt/share/y00970600/personal/sglang
source benchmark/hisparse/a5_copysfa_mtp/article_single_node.env.example
bash benchmark/hisparse/a5_copysfa_mtp/43_check_server.sh
bash benchmark/hisparse/a5_copysfa_mtp/45_prefix_sanity.sh
bash benchmark/hisparse/a5_copysfa_mtp/50_perf.sh
```

The workload sends one wave (64 requests), primes and probes the entire
identical input on every DP, and requires every request to report the full
page-aligned cacheable prefix (39,936 of 40,000 tokens with page size 128).
It preserves native `sglang.benchmark.serving` output throughput and TPOT
in `bench-serving.jsonl` and `benchmark.log`, with a separate
`cache-validation.json` pass/fail gate. See [measurement protocol](NO_OFFLOAD.md).
Radix cached_tokens is not C-pool hit rate. Use acceptance measured for the
profiled window when deriving article throughput. Controlled95/91 C-hit tests remain
standalone tests, not claims about the model's actual hit distribution.
The runtime check requires all four DP scheduler states, the effective BS16
limit, FP8 KV, radix, MTP3 and graph configuration. Silent capacity reductions
or disabled graphs fail this check. It does not prove actual replay: retain
the startup allocation log and trace. Prefix checks run cold/hot on every DP;
the perf/steady clients prime each worker and explicitly route requests evenly
to those workers. A single prime request does not warm four independent caches.

## 4. Automatic no-stack capture in terminal B

Keep the server running in terminal A. After the prefix checks and unprofiled
benchmark pass, run this in terminal B; no third terminal or manual trigger is
required:

```bash
cd /mnt/share/y00970600/personal/sglang
source benchmark/hisparse/a5_copysfa_mtp/article_single_node.env.example
AUTO_PROFILE=1 OUTPUT_TOKENS=4096 PROFILE_STEPS=20 \
  bash benchmark/hisparse/a5_copysfa_mtp/55_steady_decode.sh
```

The client primes all four DP workers, routes one wave of 64 long requests
evenly, and waits until all remain live with at least 64 output tokens each
for five continuous seconds. It then invokes the same `60_profile.sh start`
path with `with_stack=false`, automatically stopping after 20 scheduler steps.
`PROFILE_WARMUP_TOKENS`, `PROFILE_WARMUP_SECONDS` and `PROFILE_WARMUP_TIMEOUT`
(default900 seconds) control the trigger. A completed/failed request before
the trigger, a warmup timeout or invalid rank traces produces a nonzero exit.
Client progress is a trigger only; the trace must still pass the full BS16 and
20-target-step validation on all eight ranks. Capture failure is preserved,
without an automatic second capture or changes to the running service.

The stage prints `run_dir=...`. Within that directory:

- `profile-trigger.json`: observed client progress and continuous warmup.
- `profile/`: raw profiling output.
- `capture/profile-request.json`: explicit no-stack request receipt.
- `capture/profile-summary.json`: per-rank completeness and operator accounting.
- `steady-decode.json`: completed requests, explicitly labeled as profiled.

`AUTO_PROFILE_VALIDATED` means the traces passed the gate; the client then
finishes its remaining output tokens. Timing from this run includes profiler
overhead. Keep `50_perf.sh` results as the unprofiled serving measurement.

### Optional manual trigger: terminals B and C

Keep the server's normal launch. Run the sustained workload in B:

```bash
source benchmark/hisparse/a5_copysfa_mtp/article_single_node.env.example
OUTPUT_TOKENS=4096 STEADY_NUM_PROMPTS=64 \
  bash benchmark/hisparse/a5_copysfa_mtp/55_steady_decode.sh
```

After `ALL_STREAMS_ACTIVE count=64`, run in C (same exports):

```bash
source benchmark/hisparse/a5_copysfa_mtp/article_single_node.env.example
PROFILE_STEPS=20 \
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

TPOT=ITL/output_tokens_per_round; TPS_per_DP=16/TPOT_ms*1000. The analyzer
receives DP4 and per-DP BS16, not global client concurrency64. Its systemTPS
field extrapolates a rank's per-DP result across four balanced DP groups;
do not sum these replicated per-rank estimates or multiply by TP8. Inspect
every independent DP group's trace before treating balance as established.
The article's
3.0 tokens/round is an assumption, not a consequence of enabling MTP3. Without
validated device windows/acceptance, the report leaves these derived fields
null and lists missing inputs. Compare unprofiled serving throughput separately.

Next acceptance order: joint graph/Q4/fusion tests, four model combinations
with radix, unprofiled throughput, then this single no-stack profiling capture.

## Missing DeepEP V2 ACLNN symbols during graph capture

If graph capture reports `aclnnMoeLowLatencyDispatchV2` or its workspace
function missing, inspect library lookup before changing memory sizing or
graph/MTP settings. The November 2 user probe confirmed that DeepEP's vendor
library (SHA prefix `848792579ca1`) has all four V2 dispatch/combine symbols,
but the bare `libcust_opapi.so` handle selected LIM (SHA prefix `6887249a1f24`),
which lacks them. The system library also lacks these custom interfaces.
The old DeepEP bridge cached that wrong handle. A successful LIM/CopySFA
operator test does not exercise the separate MoE bridge.

The fixed DeepEP bridge uses `SGLANG_DEEPEP_OP_API_LIBRARY`, an absolute path
to its own installed vendor, regardless of import order. Standard CANN helper
functions still fall back to `libopapi.so`. No LIM/CopySFA computation, DeepEP
device implementation or graph/MTP setting is changed. Native Linux CPU
loader tests reproduce the old collision and cover both import orders,
standard fallback and invalid pins; they are not A5 device tests.

Update both personal branches, then rebuild only the DeepEP host bridge:

```bash
source benchmark/hisparse/a5_copysfa_mtp/article_single_node.env.example
bash benchmark/hisparse/a5_copysfa_mtp/11_rebuild_deepep_bridge.sh
```

This script saves the currently installed DeepEP vendor and a complete file
hash manifest, rebuilds the bridge with that exact vendor (no device-kernel
compilation), installs only the new DeepEP wheel with `--no-deps`, and verifies
the installed vendor payload hashes and existing `libsgl_kernel_npu.so` are
unchanged. Generated `__pycache__/*.pyc` differences are reported separately
only when their corresponding `.py` source exists and has the same hash on
both sides; all libraries, device binaries, Python sources and configuration
remain protected. Source staging remains a strict whole-tree copy check.
The installed DeepEP baseline `a5c12d19` has the same device sources as this
branch. Do not run `10_build_kernel.sh` for this loader repair. The repaired
bridge requires actual joint serving validation after installation.

### Resume after installation succeeded but vendor comparison failed

The user A5 build of `deep_ep-1.0.0+65f40530.cann.9.1.0.b243` completed and
installed successfully, then the old whole-tree comparison returned
`ValueError: DeepEP vendor bytes changed`. A CPU reproduction with the same
setup script shows that pip creates vendor Python bytecode during install.
The new comparison emits added/missing/changed paths and a complete JSON
receipt, retaining strict payload checks. The actual failed run's differences
must still be checked; the generic old exception alone does not identify them.

Keep the original build directory and resume from its saved vendor baseline:

```bash
export RECOVERY_DIR=/path/to/original-rebuild-deepep-bridge-run
bash benchmark/hisparse/a5_copysfa_mtp/12_resume_deepep_verification.sh
```

This does not rebuild or reinstall anything. It validates the saved wheel,
checks the backup against its original `vendor.before.json`, compares the
installed payload, checks the original kernel SO and runs both import-order
probes. It writes a new run directory and `vendor-comparison.json`, leaving
the failed run's logs/exit codes unchanged. A protected payload difference
still stops the workflow. Do not rerun the build just to create a new baseline
from the already replaced package.

Run `bash benchmark/hisparse/a5_copysfa_mtp/37_check_deepep.sh` after sourcing
the same article environment. It checks LIM-first and DeepEP-first in separate
fresh processes. Both must report `pass_pinned_deepep`, with the actual C++
adapter's four `adapter_libraries` entries pointing to the selected DeepEP
vendor. The raw bare-name probe may still show LIM; that is safe because the
fixed adapter no longer uses it. It records paths, hashes, versions and import
errors without loading the model or executing NPU operators. The complete
reports are `libraries-lim-first.log` and `libraries-deepep-first.log` in the
printed run directory. The launcher also checks the adapter before weights
load, so an old bridge fails early with a specific library report.
