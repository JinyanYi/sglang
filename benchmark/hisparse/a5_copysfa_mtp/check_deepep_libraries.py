#!/usr/bin/env python3
"""Inspect DeepEP ACLNN lookup after LIM loading without running NPU operators."""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

SYMBOLS = [
    name + suffix
    for name in ("aclnnMoeLowLatencyDispatchV2", "aclnnMoeLowLatencyCombineV2")
    for suffix in ("", "GetWorkspaceSize")
]


class LinkMap(ctypes.Structure):
    _fields_ = [("address", ctypes.c_void_p), ("name", ctypes.c_char_p)]


def inspect_library(path):
    result = {"requested": str(path)}
    try:
        library = ctypes.CDLL(str(path))
        libdl = ctypes.CDLL(ctypes.util.find_library("dl") or None)
        dlinfo = libdl.dlinfo
        dlinfo.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
        dlinfo.restype = ctypes.c_int
        link = ctypes.POINTER(LinkMap)()
        if dlinfo(library._handle, 2, ctypes.byref(link)) == 0 and link:
            resolved = link.contents.name.decode()
            result["resolved"] = resolved
            if Path(resolved).is_file():
                with Path(resolved).open("rb") as stream:
                    result["sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
        result["symbols"] = {name: hasattr(library, name) for name in SYMBOLS}
    except (OSError, AttributeError) as exc:
        result["error"] = str(exc)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--order", choices=("lim-first", "deepep-first"), default="lim-first"
    )
    args = parser.parse_args()
    report = {
        "import_order": args.order,
        "boundary": "Library lookup probe only; no NPU operators, model or graph executed",
        "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH"),
        "ASCEND_CUSTOM_OPP_PATH": os.environ.get("ASCEND_CUSTOM_OPP_PATH"),
        "SGLANG_DEEPEP_OP_API_LIBRARY": os.environ.get("SGLANG_DEEPEP_OP_API_LIBRARY"),
    }
    try:
        distribution = importlib.metadata.distribution("deep_ep")
        package = Path(distribution.locate_file("deep_ep")).resolve()
        expected = package / "vendors/hwcomputing/op_api/lib/libcust_opapi.so"
        report.update(
            deep_ep_version=distribution.version, expected_library=str(expected)
        )
        if args.order == "deepep-first":
            import deep_ep
        import sgl_kernel_npu
        import torch

        report["kernel_module"] = sgl_kernel_npu.__file__
        report["lim_libraries"] = dict(torch.ops.npu.resolve_a5_c8_original_libraries())
        report["absolute_deepep"] = inspect_library(expected)
        try:
            import deep_ep
            import deep_ep_cpp

            report["deep_ep_module"] = deep_ep.__file__
            report["deep_ep_cpp"] = deep_ep_cpp.__file__
        except Exception as exc:
            report["deep_ep_import_error"] = f"{type(exc).__name__}: {exc}"
        report["bare_custom"] = inspect_library("libcust_opapi.so")
        report["standard"] = inspect_library("libopapi.so")
        own = report["absolute_deepep"].get("symbols", {})
        bare = report["bare_custom"].get("symbols", {})
        standard = report["standard"].get("symbols", {})
        if "deep_ep_import_error" in report:
            report["status"] = "deep_ep_import_failed"
        elif hasattr(deep_ep_cpp, "resolve_op_api_libraries"):
            resolved = dict(deep_ep_cpp.resolve_op_api_libraries())
            report["adapter_libraries"] = resolved
            report["status"] = (
                "pass_pinned_deepep"
                if all(
                    Path(resolved[name]).resolve() == expected.resolve()
                    for name in SYMBOLS
                )
                else "adapter_resolved_outside_selected_deepep_vendor"
            )
        elif all(bare.get(name) or standard.get(name) for name in SYMBOLS):
            report["status"] = (
                "symbols_available_but_adapter_needs_private_library_support"
            )
        elif all(own.get(name) for name in SYMBOLS):
            report["status"] = "deepep_library_has_symbols_but_bare_lookup_misses"
        else:
            report["status"] = "selected_libraries_missing_symbols_or_dependencies"
    except Exception as exc:
        report.update(status="probe_failed", error=f"{type(exc).__name__}: {exc}")
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["status"] == "pass_pinned_deepep" else 1


if __name__ == "__main__":
    raise SystemExit(main())
