"""Opt-in native LI snapshots and isolated replay; no accuracy or timing gate."""

import argparse
import json
import os
import sys
from pathlib import Path

_SEEN = set()


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, default=str) + "\n")
    temporary.replace(path)


def runtime_info(torch, torch_npu):
    maps = Path("/proc/self/maps")
    libraries = (
        sorted(
            {
                line.split()[-1]
                for line in maps.read_text().splitlines()
                if ".so" in line and "/" in line.split()[-1]
            }
        )
        if maps.exists()
        else []
    )
    return {
        "pid": os.getpid(),
        "torch": str(torch.__version__),
        "torch_npu": str(torch_npu.__version__),
        "callable": str(torch_npu.npu_quant_lightning_indexer),
        "schemas": [
            str(schema)
            for name in torch._C._dispatch_get_all_op_names()
            if "quant_lightning_indexer" in name.lower()
            for schema in torch._C._jit_get_schemas_for_operator(name)
        ],
        "modules": {
            name: getattr(sys.modules.get(name), "__file__", None)
            for name in ("custom_ops", "sgl_kernel_npu", "torch_npu")
        },
        # Mappings are candidates, not proof of which dlsym target was called.
        "mapped_libraries": libraries,
        "environment": {
            key: os.environ.get(key)
            for key in (
                "LD_LIBRARY_PATH",
                "ASCEND_CUSTOM_OPP_PATH",
                "SGLANG_DEEPEP_OP_API_LIBRARY",
                "SGLANG_DISABLE_DRAFT_EXTEND_CUDA_GRAPH",
            )
        },
    }


def describe_call(args, torch):
    result = {}
    for name, value in args.items():
        if not torch.is_tensor(value):
            result[name] = value
            continue
        entry = {
            "shape": list(value.shape),
            "stride": list(value.stride()),
            "dtype": str(value.dtype),
            "device": str(value.device),
            "contiguous": value.is_contiguous(),
        }
        if name.startswith("actual_seq_lengths"):
            entry["values"] = value.detach().cpu().tolist()
        result[name] = entry
    return result


def snapshot_draft_li(args, batch, backend, layer_id, directory, operator=None):
    import torch
    import torch_npu

    if not backend.is_draft_worker:
        return
    mode = batch.forward_mode
    if mode.is_draft_extend_v2():
        phase = "draft_extend"
    elif mode.is_decode():
        phase = "draft_decode"
    else:
        return
    # Never introduce host IO or D2H copies inside graph capture.
    if torch.npu.is_current_stream_capturing() or phase in _SEEN:
        return
    _SEEN.add(phase)
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    stem = f"rank{rank}-{phase}-pid{os.getpid()}"
    path = root / f"{stem}.json"
    report = {"phase": phase, "rank": rank, "layer_id": layer_id}
    report["stage"] = "before_sync"
    write_json(path, report)
    print(f"[quant-li-debug] {path} before_sync", flush=True)
    torch.npu.synchronize()
    report.update(runtime_info(torch, torch_npu))
    report["selected_callable"] = str(operator)
    report["stage"] = "synchronized_before_native_li"
    report["inputs"] = describe_call(args, torch)
    write_json(path, report)
    if rank == 0:
        # Preserve strides on CPU where supported; replay checks them explicitly.
        frozen = {
            name: value.detach().cpu() if torch.is_tensor(value) else value
            for name, value in args.items()
        }
        fixture = root / f"{stem}.pt"
        torch.save({"args": frozen, "report": report}, fixture)
        report["fixture"] = str(fixture)
    report["stage"] = "ready_to_call_native_li"
    write_json(path, report)
    print(f"[quant-li-debug] {path} ready_to_call_native_li", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--binding", choices=("custom", "torch-npu", "npu-dispatch"), required=True
    )
    parser.add_argument("--graph", action="store_true")
    options = parser.parse_args()
    options.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"stage": "imports", "binding": options.binding, "graph": options.graph}
    write_json(options.output, report)
    import torch
    import torch_npu

    # Separate processes are essential: custom_ops may replace registrations.
    if options.binding in ("custom", "npu-dispatch"):
        import custom_ops  # noqa: F401
        import sgl_kernel_npu  # noqa: F401

    torch.npu.set_device(0)
    torch.npu.set_compile_mode(jit_compile=False)
    report.update(runtime_info(torch, torch_npu))
    operator = (
        torch.ops.npu.npu_quant_lightning_indexer
        if options.binding == "npu-dispatch"
        else torch_npu.npu_quant_lightning_indexer
    )
    report["selected_callable"] = str(operator)
    payload = torch.load(options.fixture, map_location="cpu", weights_only=True)
    args = {
        name: value.to("npu:0") if torch.is_tensor(value) else value
        for name, value in payload["args"].items()
    }
    report["fixture"] = str(options.fixture)
    report["inputs"] = describe_call(args, torch)
    for name, metadata in payload["report"]["inputs"].items():
        if isinstance(metadata, dict) and "stride" in metadata:
            if report["inputs"][name]["stride"] != metadata["stride"]:
                raise ValueError(f"replay changed input strides: {name}")
    report["stage"] = "before_native_li"
    write_json(options.output, report)
    print(json.dumps(report, indent=2), flush=True)
    output = operator(**args)
    torch.npu.synchronize()
    if options.graph:
        graph = torch.npu.NPUGraph()
        report["stage"] = "before_graph_capture"
        write_json(options.output, report)
        with torch.npu.graph(graph):
            output = operator(**args)
        graph.replay()
        torch.npu.synchronize()
    outputs = output if isinstance(output, (tuple, list)) else [output]
    report["outputs"] = [
        {"shape": list(item.shape), "dtype": str(item.dtype)}
        if torch.is_tensor(item)
        else {"type": type(item).__name__}
        for item in outputs
    ]
    report["stage"] = "completed"
    report["status"] = "execution_pass_only"
    write_json(options.output, report)
    print(json.dumps(report["outputs"]), flush=True)


if __name__ == "__main__":
    main()
