"""Run the existing A5 LIM/CopySFA precision gates with actual MF C8 storage."""

import importlib.util
from pathlib import Path

import pytest
import torch

pytest.importorskip("torch_npu")


@pytest.mark.parametrize("width", [1, 4])
@pytest.mark.parametrize("graph_mode", [False, True])
def test_real_lim_copysfa_mf(width, graph_mode, monkeypatch):
    import sgl_kernel_npu.sparsity_driven_kv_offload as transport

    from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.a5_pd_pool import (
        A5PackedMFPool,
    )

    spec = importlib.util.spec_from_file_location(
        "mf_probe_group", Path(__file__).with_name("probe_mf.py")
    )
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    group = probe.ProbeGroup()
    owners = []

    def allocate(shape, dtype, device_id, name=None):
        assert dtype == torch.int8 and shape[-1] == 656
        pool = A5PackedMFPool(
            tokens=shape[0] * shape[1],
            layers=1,
            device=f"npu:{device_id}",
            group=group,
        )
        owners.append(pool)
        tensor = pool.packed[0].view(shape)
        return tensor, tensor.data_ptr(), tensor.data_ptr()

    monkeypatch.setattr(transport, "create_shm_tensor", allocate)
    monkeypatch.setattr(transport, "free_shm", lambda: None)
    root = Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location(
        "a5_joint_oracle", root / "test/manual/npu/test_a5_lim_copysfa_joint.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.test_real_lim_to_copysfa(width, graph_mode)
