#!/usr/bin/env python3
"""Report and validate the exact framework, wheel and native-op provenance."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path

COPY_OP = "npu::fused_quant_scatter_copy_sparse_flash_attention"
LIM_OP = "npu::fused_quant_lightning_indexer_manage"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def contains_bytes(path: Path, needle: bytes) -> bool:
    overlap = b""
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            data = overlap + chunk
            if needle in data:
                return True
            overlap = data[-max(0, len(needle) - 1) :]
    return False


def under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def schemas_for(torch, qualified_name: str):
    return [
        schema
        for schema in torch._C._jit_get_all_schemas()
        if schema.name == qualified_name
    ]


def schema_record(schema) -> dict:
    arguments = [str(argument.type) for argument in schema.arguments]
    return {
        "schema": str(schema),
        "argument_types": arguments,
        "tensor_arguments": sum(value == "Tensor" for value in arguments),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("copy", "model"), default="copy")
    args = parser.parse_args()

    framework_root = Path(os.environ["FRAMEWORK_ROOT"]).resolve()
    report: dict[str, object] = {
        "mode": args.mode,
        "python": sys.version,
        "python_executable": sys.executable,
        "python_prefix": sys.prefix,
        "platform": platform.platform(),
        "framework_root": str(framework_root),
        "visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
    }

    import torch
    import torch_npu  # noqa: F401

    report["torch_version"] = torch.__version__
    report["torch_npu_version"] = importlib.metadata.version("torch-npu")
    report["npu_available"] = torch.npu.is_available()
    report["npu_device_count"] = torch.npu.device_count()
    if not report["npu_available"] or report["npu_device_count"] < 1:
        raise RuntimeError("The selected environment cannot enumerate an NPU")
    torch.npu.set_device(0)
    import acl

    report["acl_device_info_601"] = acl.rt.get_device_info(0, 601)
    if tuple(report["acl_device_info_601"]) != (3510, 0):
        raise RuntimeError(
            "This handoff requires Ascend950 (ACL device info 601 == (3510, 0)); "
            f"got {report['acl_device_info_601']!r}"
        )

    import sglang

    framework_module = Path(sglang.__file__).resolve()
    report["sglang_module"] = str(framework_module)
    try:
        report["sglang_distribution"] = importlib.metadata.version("sglang")
    except importlib.metadata.PackageNotFoundError:
        report["sglang_distribution"] = None
    if not under(framework_module, framework_root / "python"):
        raise RuntimeError(
            f"sglang resolves outside FRAMEWORK_ROOT/python: {framework_module}"
        )

    import sgl_kernel_npu

    kernel_module = Path(sgl_kernel_npu.__file__).resolve()
    kernel_so = kernel_module.parent / "lib" / "libsgl_kernel_npu.so"
    report["sgl_kernel_npu_module"] = str(kernel_module)
    report["sgl_kernel_npu_distribution"] = importlib.metadata.version("sgl-kernel-npu")
    report["sgl_kernel_npu_so"] = str(kernel_so)
    if not kernel_so.is_file():
        raise RuntimeError(f"Native kernel library is missing: {kernel_so}")
    report["sgl_kernel_npu_so_sha256"] = sha256(kernel_so)

    expected_kernel_root = os.environ.get("EXPECTED_KERNEL_PACKAGE_ROOT")
    if expected_kernel_root and not under(kernel_module, Path(expected_kernel_root)):
        raise RuntimeError(
            "sgl_kernel_npu resolves outside EXPECTED_KERNEL_PACKAGE_ROOT: "
            f"{kernel_module}"
        )

    copy_schemas = schemas_for(torch, COPY_OP)
    if args.mode == "model":
        lim_library_value = os.environ.get("SGLANG_NPU_A5_LIM_LIBRARY", "")
        if lim_library_value:
            lim_library = Path(lim_library_value).resolve()
            if not lim_library.is_file():
                raise RuntimeError(f"Configured LIM library is missing: {lim_library}")
            # An optional DSO must be LIM-only. Loading a full WMC/vLLM library
            # would attempt to register CopySFA a second time beside this wheel.
            if contains_bytes(lim_library, COPY_OP.split("::", 1)[1].encode()):
                raise RuntimeError(
                    "SGLANG_NPU_A5_LIM_LIBRARY contains the CopySFA operator name. "
                    "Use a LIM-only DSO, or combine the colleague LIM implementation "
                    "into the paired wheel and leave this environment variable unset."
                )
            report["external_lim_library"] = str(lim_library)
            report["external_lim_library_sha256"] = sha256(lim_library)

        # Exercise the framework's loader so an optional LIM-only DSO is
        # registered before its dispatcher schema is inspected.
        from sglang.kernels.ops.attention.npu_quant_sparse import (
            load_quant_lim_operator,
        )

        load_quant_lim_operator()

    lim_schemas = schemas_for(torch, LIM_OP)
    loaded = sorted(str(Path(path).resolve()) for path in torch.ops.loaded_libraries)
    report["loaded_libraries"] = loaded
    if str(kernel_so.resolve()) not in loaded:
        raise RuntimeError(
            "The reported kernel SO is not in torch.ops.loaded_libraries"
        )
    if args.mode == "model" and report.get("external_lim_library"):
        if report["external_lim_library"] not in loaded:
            raise RuntimeError(
                "The configured external LIM DSO was not recorded in "
                "torch.ops.loaded_libraries"
            )
    report["copy_sfa_schemas"] = [schema_record(item) for item in copy_schemas]
    report["lim_schemas"] = [schema_record(item) for item in lim_schemas]
    if not copy_schemas:
        raise RuntimeError(f"Required CopySFA operator is absent: {COPY_OP}")

    # Standalone CopySFA validation deliberately permits LIM to be absent. Model
    # serving is stricter: the colleague's operator must expose exactly the
    # caller-side 19-Tensor ABI, with no vLLM adapter or injected placeholder.
    if args.mode == "model":
        compatible = []
        for schema in lim_schemas:
            types = [str(argument.type) for argument in schema.arguments]
            if len(types) == 19 and all(value == "Tensor" for value in types):
                compatible.append(schema)
        if not compatible:
            raise RuntimeError(
                f"Model mode requires external {LIM_OP} with exactly 19 Tensor "
                "arguments. Install the colleague's paired Ascend950 LIM wheel; "
                "do not install a dummy LIM or vLLM implementation."
            )

    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
