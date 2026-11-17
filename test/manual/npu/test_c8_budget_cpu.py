"""A5 persistent buffers and physical Host binding without an NPU runtime."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

SOURCE = (
    Path(__file__).resolve().parents[3]
    / "python/sglang/srt/hardware_backend/npu/sparsity_driven_kv_offload"
)
PKG = "sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload"


@pytest.fixture
def budget(monkeypatch):
    for name in (
        "sglang",
        "sglang.srt",
        "sglang.srt.hardware_backend",
        "sglang.srt.hardware_backend.npu",
        PKG,
        "sglang.srt.utils",
    ):
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    common = ModuleType("sglang.srt.utils.common")
    common.ceil_div = lambda a, b: (a + b - 1) // b
    monkeypatch.setitem(sys.modules, common.__name__, common)
    for name in ("fused_budget", "c8_budget"):
        spec = importlib.util.spec_from_file_location(
            f"{PKG}.{name}", SOURCE / f"{name}.py"
        )
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, spec.name, module)
        spec.loader.exec_module(module)
    return module


def plan(budget, **changes):
    options = dict(
        request_rows=3,
        context_len=8700,
        num_layers=2,
        top_k=2048,
        kv_bytes_per_token=656,
        cache_tokens=8192,
        max_query_width=4,
        copy_sfa_query_heads=8,
        graph_batch_capacity=5,
    )
    options.update(changes)
    return budget.resolve_c8_budget(**options)


@pytest.mark.parametrize("query_width", [1, 4])
def test_actual_declared_allocation_bytes(budget, query_width):
    p = plan(budget, max_query_width=query_width)
    tensors = []
    for specs in (p.manager_buffers, p.adapter_buffers, p.pre_buffers):
        tensors.extend(
            spec.allocate("cpu") for spec in specs.values() for _ in range(spec.count)
        )
    tensors.extend(spec.allocate("cpu") for spec in p.tail_workspace_specs)
    tensors.append(p.query_spec.allocate("cpu"))
    tensors.extend(torch.arange(5, dtype=torch.int32) for _ in range(4))
    assert p.managed_hbm_bytes == sum(x.numel() * x.element_size() for x in tensors)
    assert p.manager_buffers["device_kv_buffer"].shape == (2, 8192 + 256, 1, 656)
    assert p.batch_capacity == 5 and p.request_rows == 3
    assert p.context_len == 8704
    assert not p.prefill_workspace_bytes


@pytest.mark.parametrize("batch_size", [1, 3, 4])
@pytest.mark.parametrize("extra_slots", [0, 2])
def test_null_request_reserves_metadata_but_no_kv_payload(
    budget, batch_size, extra_slots
):
    live_rows = batch_size + extra_slots
    p = plan(budget, request_rows=live_rows + 1)
    payload = p.manager_buffers["device_kv_buffer"]
    assert p.kv_request_rows == live_rows
    assert payload.nbytes == live_rows * (8192 + 256) * 656 * 2
    # Keep the null request and LIM's invalid-request sentinel in metadata only.
    assert p.manager_buffers["device_slot_map"].shape[0] == live_rows + 2
    assert p.manager_buffers["tail_starts"].shape == (live_rows + 2,)


@pytest.mark.parametrize("width", [2, 3, 5, 14])
def test_unsupported_mtp_is_not_silently_admitted(budget, width):
    with pytest.raises(ValueError, match="Q1/Q4"):
        plan(budget, max_query_width=width)


@pytest.mark.parametrize(
    "change",
    [
        dict(kv_bytes_per_token=512),
        dict(page_size=64),
        dict(cache_tokens=32640),
        dict(cache_tokens=2048),
        dict(copy_sfa_query_heads=1),
        dict(context_len=262145),
    ],
)
def test_invalid_c8_geometry_rejected(budget, change):
    with pytest.raises(ValueError):
        plan(budget, **change)


def test_host_follows_native_physical_pages(budget):
    p = plan(budget)
    key = torch.empty((91, 128, 1, 128), dtype=torch.float8_e4m3fn)
    pool = SimpleNamespace(
        dsa_kv_cache_store_fp8=True,
        page_size=128,
        indexer_layer_ids=[0, 4],
        get_index_k_buffer=lambda _: key,
    )
    p.bind_native_pool(pool)
    assert p.host_token_capacity == 91 * 128
    assert p.host_kv_bytes == 2 * 91 * 128 * 656
    assert p.host_kv_spec.shape == (1, 91 * 128, 1, 656)
    p.bind_native_pool(pool)
    pool.get_index_k_buffer = lambda _: key[:90]
    with pytest.raises(ValueError, match="different pool"):
        p.bind_native_pool(pool)


def test_budget_limit_and_fp16_compute(budget):
    p = plan(budget)
    with pytest.raises(ValueError, match="exceeding"):
        plan(budget, hbm_budget_bytes=p.managed_hbm_bytes - 1)
    half = plan(budget, model_config=SimpleNamespace(dtype=torch.float16))
    assert half.manager_buffers["mtp_attn_out"].dtype == torch.float16
    assert half.managed_hbm_bytes == p.managed_hbm_bytes
