# A5 packed-C8 P/D candidate

This is a **not-yet-device-validated A5 P/D candidate**, based on the working
colocated A5 branch. It keeps the original McZyWu A5 LIM, our CopySFA scheduler,
the two Triton metadata fusions, Q1/Q4, and the independent HBM draft cache.
It does not import the A3 split-BF16 attention or change the CopySFA ABI.

Both personal repositories use `feat/a5-pd-split-kv-20261104`. Provenance and
validation boundaries are in [integration.json](integration.json). The previous
`feat/a5-copysfa-mtp-radix-20261031` branches are unchanged.

## Data path and scope

- P: native HBM packed C8 (656 bytes/token), index K and index scales; no sparse
  offload. P still prepares the MTP draft state required by the existing P/D path.
- D: one MF SHARED packed-C8 host allocation per attention-TP group; only TP0
  allocates the physical DRAM. Export packed target pages first, then device
  index K/scales, then the existing native draft entries. Native allocator page
  IDs and radix prefix mappings remain the address contract.
- Existing Ascend TransferEngine sends P HBM pages to D MF host pages. D does
  not allocate a full target HBM cache merely to receive them. The bounded C
  cache and 256-token causal tail retain the colocated A5 implementation.
- Current target KV writes and accepted-token moves use MF sparse copies. A
  device broadcast orders TP0's shared writes before peer reads, including graph
  replay. Retraction snapshots preserve opaque packed bytes and rank-local
  index/scales; restore invalidates request-local LIM and tail residency.

The first target is two separate nodes, each 8 visible A5 devices, attention
DP4/TP2, EP8, PP1/CP1, MTP 3/1/4, D graph bucket16 and per-DP BS16, C12288,
40K input/1024 output. P graph is disabled; D target/draft graphs use the existing
A5 setup. Matching topology on both peers is required for this first candidate.
Other Q/MTP configurations, heterogeneous TP, DCP, PP, unified memory and role
switching are outside this handoff. Short D requests retain the existing eager
QSFA fallback; this does not add short-request graph support.

## Runtime dependencies

Use the same A5-compatible checkpoint, tokenizer, torch/torch_npu, CANN and
DeepEP stack on both nodes. Keep the tested DeepEP-library bridge from the A5
colocated wheel. The kernel code and ABI are unchanged: an installed wheel
built from kernel base `138319bddb54bec3333e92fb6d072a74c540293a` is reusable in
the same compatible runtime. A different CANN/torch ABI can still require a
rebuild with `benchmark/hisparse/a5_copysfa_mtp/10_build_kernel.sh`.

Additional requirements are an **A5-compatible** `memfabric_hybrid` containing
`offload.OffloadConfig`, `Scene.SHARED`, `offload.empty`, `offload.sparse_copy`,
and the selected `TransferEngine.TransDataOpType`, plus a matching
`sglang_router` supporting `--pd-disaggregation --dp-aware`. `doctor.py` records
package locations/versions and driver version files and checks API presence.
It does not prove GVA mapping, hardware compatibility or transfer success.

Do not install the donor's A3-only MF offload build on A5. Set `MF_PYTHON_PATH`
if using a private matching MF installation. `environment.sh` sources CANN/ATB
first and then adds framework/MF paths, preserving inherited paths.

**HDK and LingQu Computing Network (LCNE): exact A5 versions are unresolved.**
The [public MemFabric hardware matrix](https://github.com/Ascend/memfabric_hybrid#-软件硬件配套说明)
lists A2/A3 requirements, including HDK25.5 and LingQu1.5 for A3 DRAM pooling;
that is not an A5 compatibility statement. The
[A5 upgrade guide](https://support.huaweicloud.com/bestpractice-aics/aics_bp_0104.html)
requires selecting packages for the actual product/architecture. Obtain the
matched A5 MF/HDK/firmware/LCNE/URMA/ubmem bundle from the machine provider before
upgrading anything. This branch neither upgrades drivers nor invents a minimum
A5 version. A normal HCCL all-reduce or a successful import is insufficient.

The A5 launcher defaults to `device_urma`, the cross-host A5 protocol in the
[MemFabric transfer example](https://github.com/Ascend/memfabric_hybrid/blob/9dc22e2a54fb8de28ba79ee77c714cded2086eb9/examples/transfer/test_transfer_pd_separate_hbm_dp.py).
Override `ASCEND_MF_TRANSFER_PROTOCOL` on both peers if the provider's A5
bundle specifies another protocol. The API check lists the installed enum
names, and unsupported protocols fail before model startup. The launcher sets
`MF_CONFIG_STORE_URL` for the shared configuration store.

## Pull the paired branches

In each prepared container, using its actual local paths:

```bash
export FRAMEWORK_ROOT=/mnt/share/y00970600/personal/sglang
export KERNEL_ROOT=/mnt/share/y00970600/personal/sgl-kernel-npu
for repo in "$FRAMEWORK_ROOT" "$KERNEL_ROOT"; do
  git -C "$repo" fetch origin feat/a5-pd-split-kv-20261104
  git -C "$repo" switch feat/a5-pd-split-kv-20261104
  git -C "$repo" merge --ff-only origin/feat/a5-pd-split-kv-20261104
done
```

No scripts kill existing processes or containers, change system network settings,
or perform full checkout audits. Confirm assigned devices and service ports are
free before launching. `/mnt/share` is only an example; use `/home/...` if needed.

## Small device gates before loading the model

In each node's prepared container:

```bash
cd "$FRAMEWORK_ROOT"
export PYTHON_BIN="$(command -v python3)"
export RUN_ROOT=/mnt/share/y00970600/a5-pd-runs
export KERNEL_ROOT
# Normal colocated joint suite first (52 cases in the previous handoff).
ASCEND_RT_VISIBLE_DEVICES=0 bash benchmark/hisparse/a5_copysfa_mtp/30_test_a5.sh model
# Tiny MF pool: exact bytes from MF to the actual AICore reader, then Q1/Q4
# real LIM + CopySFA vs the existing CPU oracle, eager and graph.
ASCEND_RT_VISIBLE_DEVICES=0 bash benchmark/hisparse/a5_pd/check.sh local
# Two devices, matching one attention-TP2 group; checks real shared GVA/fencing.
ASCEND_RT_VISIBLE_DEVICES=0,1 PROBE_TP_SIZE=2 \
  bash benchmark/hisparse/a5_pd/check.sh tp
```

Each numerical MF case runs in a fresh process because MF initialization is
process-global. Any skipped numerical case fails the shell gate. Logs, JUnit
files and stage exit codes go to a new timestamped run directory. The numerical
test reuses the existing A5 LIM/CopySFA oracle and unchanged thresholds. It is
not merely an import test. A local probe still does not prove inter-node MF.

## Start P, D and the router manually

Set the following in each service terminal; replace both address placeholders
with actual container-reachable IPv4 addresses. Model paths may be different
locally but must resolve to identical checkpoint/tokenizer content.

```bash
export FRAMEWORK_ROOT=/mnt/share/y00970600/personal/sglang
export KERNEL_ROOT=/mnt/share/y00970600/personal/sgl-kernel-npu
export MODEL_PATH=/mnt/share/w00936111/GLM-5.2-W4A4C8-mxfp4-A5-0731
export RUN_ROOT=/mnt/share/y00970600/a5-pd-runs
export PREFILL_IP='<P_NODE_IP>'
export DECODE_IP='<D_NODE_IP>'
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TP_SIZE=8 ATTN_DP_SIZE=4 EP_SIZE=8
export PER_DP_BATCH_SIZE=16 GRAPH_BATCH_SIZE=16
export MEM_FRACTION_STATIC=0.80 MAX_TOTAL_TOKENS=1048576
export PYTHON_BIN="$(command -v python3)"
cd "$FRAMEWORK_ROOT"
```

On P, keep running:

```bash
bash benchmark/hisparse/a5_pd/launch.sh prefill
```

P creates the common MF configuration store at `tcp://P_NODE_IP:24669` and
bootstrap registry on8998. HTTP ports default to P30000/D30001/router8000.
No separate manually started config store is required. Allow the selected MF
protocol's data plane and both-way control connectivity as well as these ports.

For the **first same-input reference**, start D with native C8:

```bash
PER_DP_BATCH_SIZE=7 GRAPH_BATCH_SIZE=8 KV_OFFLOAD=0 \
  bash benchmark/hisparse/a5_pd/launch.sh decode
```

In a separate terminal on D, with the same IP variables:

```bash
bash benchmark/hisparse/a5_pd/router.sh
```

Wait for both server readiness messages and router registration of the expected
P/D workers; then run the gate below. After reference completion, stop **only
this D service and its router** using their foreground terminals. Keep P running
and restart D with offload, followed by the same router command:

```bash
PER_DP_BATCH_SIZE=16 GRAPH_BATCH_SIZE=16 KV_OFFLOAD=1 \
  bash benchmark/hisparse/a5_pd/launch.sh decode
```

`launch.sh ... --dry-run` prints the concrete command without importing NPU/MF.
Use different `RUN_DIR` values or leave it unset for each launch.

## Native-reference and offload functional gates

Run in a client terminal with access to the framework Python dependencies:

```bash
cd "$FRAMEWORK_ROOT"
export PYTHONPATH="$FRAMEWORK_ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
export no_proxy="localhost,127.0.0.1,::1,$PREFILL_IP,$DECODE_IP${no_proxy:+,$no_proxy}"
export NO_PROXY="$no_proxy"
export REF_DIR="$RUN_ROOT/$(date -u +%Y%m%dT%H%M%SZ)-pd-native-reference"
python3 benchmark/hisparse/a5_pd/smoke.py \
  --model-path "$MODEL_PATH" --prefill-url "http://$PREFILL_IP:30000" \
  --decode-url "http://$DECODE_IP:30001" --base-url http://127.0.0.1:8000 \
  --concurrency 4 --output "$REF_DIR"
```

Keep `REF_DIR` for the second phase. After restarting D with offload and restarting
the router, compare with **the identical saved input token IDs**:

```bash
python3 benchmark/hisparse/a5_pd/smoke.py \
  --model-path "$MODEL_PATH" --prefill-url "http://$PREFILL_IP:30000" \
  --decode-url "http://$DECODE_IP:30001" --base-url http://127.0.0.1:8000 \
  --concurrency 4 --reference "$REF_DIR/summary.json" \
  --output "$RUN_ROOT/$(date -u +%Y%m%dT%H%M%SZ)-pd-offload-compare"
```

The gate runs natural retrieval inputs512/4096/40000, cold and hot. It requires
64 output tokens, correct retrieval, exact greedy output IDs against native C8,
and every hot response's full page-cacheable prefix (512->384,4096->3968,
40000->39936). It flushes P and D before each length; no profiler is active.
Any error exits nonzero and retains responses already received. DP-aware router
scheduling must actually populate the selected ranks; a cache gate failure is
not waived. These small functional waves do not establish the target BS16 or
model-suite accuracy. Read D logs to confirm productive long-request graph replay.

Then repeat the compare with `--concurrency 64` to exercise the intended full
DP4 x16 wave, retaining the same passing reference. For forced retraction,
restart this offload D with `SGLANG_TEST_RETRACT=1
SGLANG_TEST_RETRACT_INTERVAL=3`, run a fresh compare with `--require-retraction`,
and require positive `meta_info.num_retractions` in the saved responses. Remove these variables and
restart D before measuring performance. Retraction has CPU byte coverage only
until that real-device comparison passes.

Do not report `smoke.py` as bench_serving throughput. P/D timing includes
transport and P compute and has different routing/priming from the old colocated
`50_perf.sh`. Establish accuracy/cache and real per-DP admission first; retain
the old colocated BS7/BS16 performance results as a separate configuration.

## Table 8 settings on two 8-device P/D hosts

`run_table8.sh` uses GLM-5.2 W4A4C8, MTP3, 104000 input tokens, 1024 output
tokens, and the same P configuration for both cases. Each role uses DP8,
attention TP1 and EP8. The article uses 16 decode ranks and derives TPOT and
throughput from ITL. This run uses 8 decode ranks and records client metrics.

Run `run_table8.sh prefill` on P, then `run_table8.sh decode native` and
`run_table8.sh router` in separate terminals on D. Once both workers and the
router are ready, run `run_table8.sh bench native` on D. Stop that D service
and router, keep P running, and repeat with `decode hisparse`, `router` and
`bench hisparse`. Native uses BS5 (40 total requests). HiSparse uses BS16
(128 total requests) and C16000 with LIM, CopySFA and metadata fusion enabled.

The launcher contains the deployment addresses and the shared result directory.
Each service launch gets its own log directory. Stop the previous service for
that role before relaunching. The benchmark follows `decode-<mode>.latest`
and uses a new results subdirectory, with one common token-ID fixture for both
cases. `bench native --input-ids /path/to/104k.json` can supply an existing
104000-ID fixture for cross-framework comparison.

Each case flushes both roles once, primes each D rank, runs two full warmup
waves, then records five single-wave native `bench_serving` runs. Formal
results require 103936 cached tokens on every request and full-batch graph
samples from every D rank. P-side successful KV transfer counters are saved
before and after each wave. The hot payload must match one tail page per request
using the cold 813-page transfer as the calibration. The counters describe
scheduled KV payload, not device-link wire bytes.

Read `summary.json` for the median of the five rounds' native TTFT, TPOT and
output throughput values. Every formal round preserves the original
`bench-serving.stdout.log` and `bench-serving.jsonl`. The PNG renders the
verbatim native result block from that log.

## Matched BS2 with attention DP disabled

Insert `dp1` before each role in `run_table8.sh` to run attention TP8 and EP8
on both nodes. Both decode cases use global BS2 and graph bucket 8. The bucket
aligns single-token draft capture to attention TP8. The run retains the
104000 input tokens, MTP3, cache checks and metrics of the DP8 run.
The router registers one HTTP worker per role and uses the installed wheel.
Restart both model services for this topology and keep the same P service
when switching D from native to HiSparse. The results use the separate
`table8-20261105-dp1` directory.
