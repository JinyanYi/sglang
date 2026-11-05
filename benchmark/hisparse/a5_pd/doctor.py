#!/usr/bin/env python3
"""Inspect the A5 P/D runtime API; this is not a transfer or hardware test."""

import argparse
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path


def inspect(role):
    import memfabric_hybrid
    import torch  # noqa: F401  # Load libtorch before the MF extension.
    from memfabric_hybrid import TransferEngine, offload

    protocol = (
        os.environ.get("ASCEND_MF_TRANSFER_PROTOCOL", "device_urma").strip().lower()
    )
    protocols = sorted(
        name for name in dir(TransferEngine.TransDataOpType) if name.isupper()
    )
    missing = [
        name
        for name in ("OffloadConfig", "initialize", "empty", "sparse_copy")
        if not hasattr(offload, name)
    ]
    if not hasattr(getattr(offload, "Scene", None), "SHARED"):
        missing.append("Scene.SHARED")
    if not hasattr(TransferEngine.TransDataOpType, protocol.upper()):
        missing.append(f"TransferEngine.TransDataOpType.{protocol.upper()}")
    versions = {}
    for package in (
        "torch",
        "torch-npu",
        "memfabric-hybrid",
        "sgl-kernel-npu",
        "sglang-router",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    files = {}
    for name in (
        "/usr/local/Ascend/driver/version.info",
        "/usr/local/Ascend/firmware/version.info",
        "/usr/local/Ascend/ascend-toolkit/latest/version.cfg",
    ):
        path = Path(name)
        if path.is_file():
            files[name] = path.read_text(errors="replace")
    return dict(
        status="api_present" if not missing else "missing_api",
        boundary="Import/API inspection only; MF GVA, network, graph and accuracy untested",
        role=role,
        python=sys.executable,
        machine=platform.machine(),
        packages=versions,
        memfabric_module=memfabric_hybrid.__file__,
        protocol=protocol,
        available_protocols=protocols,
        missing_api=missing,
        installed_version_files=files,
        environment={
            key: os.environ.get(key)
            for key in (
                "SGLANG_HOST_IP",
                "MF_CONFIG_STORE_URL",
                "MF_HYBM_USE_VMM_SEGMENT",
                "ASCEND_RT_VISIBLE_DEVICES",
                "SGLANG_NPU_A5_PD_MF",
                "LD_LIBRARY_PATH",
            )
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("prefill", "decode"), required=True)
    args = parser.parse_args()
    report = inspect(args.role)
    print(json.dumps(report, indent=2))
    return int(bool(report["missing_api"]))


if __name__ == "__main__":
    raise SystemExit(main())
