# Native draft LI crash diagnosis

This is an execution diagnostic, not an accuracy or performance gate. It does
not change tensor values, precision thresholds, operator selection, or outputs.

The reported November 3 failure reaches draft extend warmup at
`npu_cudagraph_backend.py:111`, before entering `torch.npu.graph`. Its native
stack includes the recipes `custom_ops` adapter followed by
`aclnnQuantLightningIndexerGetWorkspaceSize` and
`aclnnInnerQuantLightningIndexerGetWorkspaceSize`. A mismatched adapter/library
ABI and unsupported input metadata remain hypotheses. The preceding 02:55 OOM
plogs do not describe the 03:32 segfault.

## Capture the inputs

After stopping the previous failed service, run in the existing A5 container:

```bash
cd /mnt/share/y00970600/personal/sglang
git pull --ff-only origin feat/a5-copysfa-mtp-radix-20261031
source benchmark/hisparse/a5_copysfa_mtp/article_single_node.env.example
export MEM_FRACTION_STATIC=0.80
export SGLANG_NPU_QUANT_LI_DEBUG_DIR="/tmp/a5-quant-li-$(date +%Y%m%d-%H%M%S)"
unset SGLANG_DISABLE_DRAFT_EXTEND_CUDA_GRAPH
echo "Snapshots: $SGLANG_NPU_QUANT_LI_DEBUG_DIR"
bash benchmark/hisparse/a5_copysfa_mtp/40_launch_server.sh
```

Debugging synchronizes once before each draft phase's first native LI call.
Every rank writes a small JSON report; rank 0 also saves full CPU tensor inputs
as `.pt` files. Target LIM/CopySFA calls are excluded. The hook performs no IO
inside an active NPU graph capture. Use local `/tmp` to avoid NFS IO stalls.
The saved files may include model-derived tensors and should stay private.

The JSON `stage` distinguishes failure boundaries:

- `before_sync`: inspect the earlier asynchronous computation or synchronization.
- `synchronized_before_native_li`: synchronization finished; snapshot construction
  has not finished yet.
- `ready_to_call_native_li`: the inputs were recorded before the failing call.

Schemas and module paths identify registered bindings. `mapped_libraries` lists
loaded libraries; it does not prove which symbol a dynamic lookup selected.
Capture these reports even when rank 0 is terminated before its full snapshot
finishes. Do not infer the actual loaded library from a build-time source path.

## Replay without loading the model

Once the failed service's workers have exited and device 0 is free, use the same
terminal (or set `SGLANG_NPU_QUANT_LI_DEBUG_DIR` to the printed directory):

```bash
cd /mnt/share/y00970600/personal/sglang
bash benchmark/hisparse/a5_copysfa_mtp/38_replay_quant_li.sh \
  "$SGLANG_NPU_QUANT_LI_DEBUG_DIR"
```

Each fixture runs in six fresh subprocesses: custom, torch-npu, or explicit
npu-dispatch binding, each with eager execution or an eager warmup followed by
graph capture/replay. The npu-dispatch case imports custom_ops and sgl_kernel_npu
but calls `torch.ops.npu.npu_quant_lightning_indexer` explicitly, as the FP8
framework path now does. It checks whether the stock binding still works when
the custom packages have been loaded into the same process.
The torch-npu subprocess does not explicitly import `custom_ops` or
`sgl_kernel_npu`; the report records modules and schemas to verify that boundary.
Both subprocess kinds inherit the same OPP environment. This comparison changes
the Python/dispatcher binding, not an independently rebuilt CANN installation.
The replay verifies input strides against the snapshot before calling LI.

`run_dir=...-replay-quant-li` contains `summary.tsv`, per-case logs, exit codes,
and progress JSON. A segfault can leave progress JSON at `before_native_li` or
`before_graph_capture`. A completed case has `execution_pass_only`, not an
accuracy claim. A device failure may contaminate later subprocesses; rerun a
case after device recovery before attributing differences to a binding.

If the same Q4 fixture fails in eager execution, focus on inputs, host tiling,
and adapter/library ABI. If only one binding fails, resolve the selected ACLNN
symbols in a native debugger and compare the installed adapter's function
signature with that library's headers. If eager passes and graph fails, inspect
the graph-specific path. A passing Q1 fixture is a control, not proof about Q4.

For normal serving or performance runs, remove the hook:

```bash
unset SGLANG_NPU_QUANT_LI_DEBUG_DIR
```

## Observed binding differential

The manually reported `20261103T061228Z-replay-quant-li` run uses identical
rank0 draft-extend inputs: Q `[64,32,128]` FP8, sixteen requests with Q length 4
and KV length 4, BF16 weights, FP32 scales. Both custom cases exit 139; both
torch-npu cases exit 0 and return indices `[64,1,2048]` INT32. This identifies
a binding-dependent execution failure for that fixture. It does not prove a
specific ABI mismatch or model accuracy. The explicit npu-dispatch case and
the updated full-model FP8 path still require manual A5 validation.

To test only the new explicit dispatcher path against an existing snapshot:

```bash
LI_REPLAY_BINDINGS=npu-dispatch \
  bash benchmark/hisparse/a5_copysfa_mtp/38_replay_quant_li.sh /tmp/<SNAPSHOT_DIRECTORY>
```
