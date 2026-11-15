"""C8 retraction preserves raw KV, FP8 index bytes and scale ownership."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

SOURCE = (
    Path(__file__).resolve().parents[3]
    / "python/sglang/srt/hardware_backend/npu/sparsity_driven_kv_offload/a5_pd.py"
)


@pytest.mark.parametrize("shared", [False, True])
def test_overlapping_accepted_token_move_snapshots_all_packed_bytes(
    monkeypatch, shared
):
    calls, publications = [], []

    def copy(source, destination, src, dst, valid):
        source = source.view(-1, 656)
        destination = destination.view(-1, 656)
        destination[dst[valid]] = source[src[valid]].clone()

    def unidex(source, destination, src, dst, valid, *args, **kwargs):
        calls.append(kwargs)
        copy(source, destination, src, dst, valid)

    kernels = ModuleType("sgl_kernel_npu.sparsity_driven_kv_offload")
    kernels.unidex_copy_inplace = unidex
    monkeypatch.setitem(sys.modules, kernels.__name__, kernels)
    spec = importlib.util.spec_from_file_location("a5_overlap_move", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    hosts = [
        torch.randint(-128, 127, (1, 128, 1, 656), dtype=torch.int8) for _ in range(2)
    ]
    before = [host.clone() for host in hosts]
    mf = (
        SimpleNamespace(
            rank=0,
            copy_rows=lambda sources, destinations, src, dst, valid: copy(
                sources[0], destinations[0], src, dst, valid
            ),
            publish=lambda: publications.append(True),
        )
        if shared
        else None
    )
    manager = SimpleNamespace(
        mf_host_pool=mf, host_kv_buffer=hosts, dev_ptr_list=[1234, 5678], device="cpu"
    )
    source = torch.tensor([3, 4, 5])
    target = torch.tensor([4, 5, 3])
    module.move_packed_kv(manager, target, source)
    for actual, original in zip(hosts, before):
        expected = original.clone()
        expected.view(-1, 656)[target] = original.view(-1, 656)[source]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    if shared:
        assert publications == [True] and not calls
    else:
        assert [call["src_ptr"] for call in calls] == [1234, None, 5678, None]
        assert [call["dst_ptr"] for call in calls] == [None, 1234, None, 5678]


@pytest.mark.parametrize("shared", [False, True])
def test_retraction_roundtrip_keeps_all_bytes_and_index_scales(monkeypatch, shared):
    pointer_calls, resets, publications = [], [], []

    def copy(source, destination, source_rows, destination_rows, valid):
        src = source.view(-1, source.shape[-1])
        dst = destination.view(-1, destination.shape[-1])
        dst[destination_rows[valid]] = src[source_rows[valid]].clone()

    def unidex(source, destination, source_rows, destination_rows, valid, *_, **kwargs):
        pointer_calls.append(kwargs)
        copy(source, destination, source_rows, destination_rows, valid)

    kernels = ModuleType("sgl_kernel_npu.sparsity_driven_kv_offload")
    kernels.unidex_copy_inplace = unidex
    monkeypatch.setitem(sys.modules, kernels.__name__, kernels)
    monkeypatch.setattr(
        torch, "npu", SimpleNamespace(synchronize=lambda: None), raising=False
    )
    spec = importlib.util.spec_from_file_location("a5_host_lifecycle", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    host = [
        torch.randint(-128, 127, (1, 256, 1, 656), dtype=torch.int8) for _ in range(2)
    ]
    mf = (
        SimpleNamespace(
            rank=0,
            copy_rows=lambda sources, destinations, src, dst, valid: copy(
                sources[0], destinations[0], src, dst, valid
            ),
            publish=lambda: publications.append(True),
        )
        if shared
        else None
    )
    manager = SimpleNamespace(
        host_kv_buffer=host,
        mf_host_pool=mf,
        dev_ptr_list=[1234, 5678],
        reset_requests=resets.append,
    )
    keys = torch.randint(0, 255, (2, 2, 128, 1, 128), dtype=torch.uint8)
    scales = torch.randn(2, 2, 128, 1)
    pool = SimpleNamespace(
        sparse_kv_manager=manager,
        device="cpu",
        index_k_buffer=keys,
        index_k_scale_buffer=scales,
        indexer_layer_id_to_slot={0: 0, 1: 1},
        start_layer=0,
        cpu_offloading_chunk_size=2,
    )
    selected = torch.tensor([129, 1, 128], dtype=torch.int64)
    original_host = [h.clone() for h in host]
    original_keys, original_scales = keys.clone(), scales.clone()
    snapshot = module.snapshot_packed_kv(pool, selected)
    for h in host:
        h.view(-1, 656)[selected] = 0
    keys.flatten(1, 2)[:, selected] = 0
    scales.flatten(1, 2)[:, selected] = 0
    module.restore_packed_kv(pool, snapshot, selected, 2)
    for actual, expected in zip(host, original_host):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(keys, original_keys, rtol=0, atol=0)
    torch.testing.assert_close(scales, original_scales, rtol=0, atol=0)
    assert resets == [[2]]
    if shared:
        assert publications == [True]
        assert not pointer_calls
    else:
        assert not publications
        assert {x["src_ptr"] for x in pointer_calls if x["src_ptr"] is not None} == {
            1234,
            5678,
        }
        assert {x["dst_ptr"] for x in pointer_calls if x["dst_ptr"] is not None} == {
            1234,
            5678,
        }
