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

The default MF protocol `device_sdma` comes from the donor P/D recipe. Override
`ASCEND_MF_TRANSFER_PROTOCOL` on both peers if the provider's A5 bundle requires
another supported protocol. An unsupported enum or failed registration exits
explicitly instead of silently selecting a different transport.

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
