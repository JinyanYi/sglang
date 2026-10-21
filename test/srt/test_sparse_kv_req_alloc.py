from array import array
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload import (
    manager as manager_module,
)
from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.manager import (
    SparseKVCacheManager,
)
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.sampling.sampling_params import SamplingParams


def make_req(rid: str) -> Req:
    return Req(
        rid=rid,
        origin_input_text="test",
        origin_input_ids=array("q", [1]),
        sampling_params=SamplingParams(max_new_tokens=1),
    )


def make_pool(size: int = 4) -> ReqToTokenPool:
    return ReqToTokenPool(
        size=size,
        max_context_len=8,
        device="cpu",
        enable_memory_saver=False,
    )


def install_sparse_reset(pool: ReqToTokenPool, copy_sfa_enabled: bool):
    manager = SparseKVCacheManager.__new__(SparseKVCacheManager)
    manager.device = "cpu"
    manager.layer_num = 2
    manager.copy_sfa_enabled = copy_sfa_enabled
    manager.device_slot_map = [torch.full((pool._alloc_size, 2), 7) for _ in range(2)]
    manager.device_slot_owner = [torch.full((pool._alloc_size, 2), 9) for _ in range(2)]
    manager._install_req_alloc_hook(pool)
    return manager


@pytest.mark.parametrize("copy_sfa_enabled", [False, True])
def test_fresh_req_allocation_resets_only_new_sparse_rows(copy_sfa_enabled: bool):
    pool = make_pool()
    manager = install_sparse_reset(pool, copy_sfa_enabled)
    req = make_req("fresh")

    req_ids = pool.alloc([req])

    assert req_ids == [req.kv.req_pool_idx]
    row = req.kv.req_pool_idx
    for layer_idx in range(manager.layer_num):
        assert torch.equal(
            manager.device_slot_map[layer_idx][row], torch.tensor([-1, -1])
        )
        assert torch.equal(manager.device_slot_map[layer_idx][0], torch.tensor([7, 7]))
        if copy_sfa_enabled:
            assert torch.equal(
                manager.device_slot_owner[layer_idx][row], torch.tensor([-1, -1])
            )
        else:
            assert torch.equal(
                manager.device_slot_owner[layer_idx][row], torch.tensor([9, 9])
            )


def test_reset_requests_keeps_legacy_fixtures_without_optional_lim_state():
    pool = make_pool()
    manager = install_sparse_reset(pool, copy_sfa_enabled=False)
    row = 2
    for layer_idx in range(manager.layer_num):
        manager.device_slot_map[layer_idx][row] = torch.tensor([3, 4])

    manager.reset_requests([row])

    for layer_idx in range(manager.layer_num):
        assert torch.equal(
            manager.device_slot_map[layer_idx][row], torch.tensor([-1, -1])
        )


def test_copy_sfa_enforces_fixed_cache_capacity_for_reference(monkeypatch):
    pool = make_pool()
    monkeypatch.setattr(
        manager_module,
        "envs",
        SimpleNamespace(
            SGLANG_NPU_ENABLE_SPARSE_KV_OFFLOAD=SimpleNamespace(get=lambda: True),
            SGLANG_NPU_ENABLE_COPY_SFA=SimpleNamespace(get=lambda: True),
            SGLANG_NPU_SPARSE_KV_POLICY=SimpleNamespace(get=lambda: "prefix_tail"),
            SGLANG_NPU_SPARSE_KV_CACHE_CAPACITY=SimpleNamespace(get=lambda: 4096),
        ),
    )
    monkeypatch.setattr(
        manager_module, "get_sparse_kv_indexer_impl", lambda: "reference"
    )
    monkeypatch.setattr(
        manager_module.TorchMemorySaverAdapter, "create", lambda **_kwargs: None
    )

    with pytest.raises(ValueError, match="cache capacity C=2048"):
        SparseKVCacheManager(
            pool,
            None,
            sparse_context_len=2048,
            copy_sfa_padded_query_heads=32,
        )


def test_chunk_continuation_reuses_dirty_sparse_row_without_reset():
    pool = make_pool()
    manager = install_sparse_reset(pool, copy_sfa_enabled=True)
    req = make_req("continuation")
    req_id = pool.alloc([req])[0]
    req.kv.kv_allocated_len = 1
    for layer_idx in range(manager.layer_num):
        manager.device_slot_map[layer_idx][req_id] = torch.tensor([3, 4])
        manager.device_slot_owner[layer_idx][req_id] = torch.tensor([5, 6])

    assert pool.alloc([req]) == [req_id]

    for layer_idx in range(manager.layer_num):
        assert torch.equal(
            manager.device_slot_map[layer_idx][req_id], torch.tensor([3, 4])
        )
        assert torch.equal(
            manager.device_slot_owner[layer_idx][req_id], torch.tensor([5, 6])
        )


def test_mixed_reuse_and_fresh_allocation_resets_only_fresh_row():
    pool = make_pool()
    manager = install_sparse_reset(pool, copy_sfa_enabled=True)
    continuation = make_req("continuation")
    continuation_id = pool.alloc([continuation])[0]
    continuation.kv.kv_allocated_len = 1
    fresh = make_req("fresh")
    for layer_idx in range(manager.layer_num):
        manager.device_slot_map[layer_idx][continuation_id] = torch.tensor([3, 4])
        manager.device_slot_owner[layer_idx][continuation_id] = torch.tensor([5, 6])

    req_ids = pool.alloc([continuation, fresh])

    assert req_ids == [continuation_id, fresh.kv.req_pool_idx]
    fresh_id = fresh.kv.req_pool_idx
    for layer_idx in range(manager.layer_num):
        assert torch.equal(
            manager.device_slot_map[layer_idx][continuation_id], torch.tensor([3, 4])
        )
        assert torch.equal(
            manager.device_slot_owner[layer_idx][continuation_id], torch.tensor([5, 6])
        )
        assert torch.equal(
            manager.device_slot_map[layer_idx][fresh_id], torch.tensor([-1, -1])
        )
        assert torch.equal(
            manager.device_slot_owner[layer_idx][fresh_id], torch.tensor([-1, -1])
        )


def test_failed_allocation_does_not_reset_sparse_rows():
    pool = make_pool(size=1)
    manager = install_sparse_reset(pool, copy_sfa_enabled=True)
    allocated = make_req("allocated")
    allocated_id = pool.alloc([allocated])[0]
    for layer_idx in range(manager.layer_num):
        manager.device_slot_map[layer_idx][allocated_id] = torch.tensor([3, 4])
        manager.device_slot_owner[layer_idx][allocated_id] = torch.tensor([5, 6])

    assert pool.alloc([make_req("first"), make_req("second")]) is None

    for layer_idx in range(manager.layer_num):
        assert torch.equal(
            manager.device_slot_map[layer_idx][allocated_id], torch.tensor([3, 4])
        )
        assert torch.equal(
            manager.device_slot_owner[layer_idx][allocated_id], torch.tensor([5, 6])
        )


def test_reallocated_row_resets_stale_copy_sfa_owners():
    pool = make_pool(size=1)
    manager = install_sparse_reset(pool, copy_sfa_enabled=True)
    previous = make_req("previous")
    previous_id = pool.alloc([previous])[0]
    for layer_idx in range(manager.layer_num):
        manager.device_slot_map[layer_idx][previous_id] = torch.tensor([3, 4])
        manager.device_slot_owner[layer_idx][previous_id] = torch.tensor([5, 6])
    pool.free(previous)

    current = make_req("current")
    current_id = pool.alloc([current])[0]

    assert current_id == previous_id
    for layer_idx in range(manager.layer_num):
        assert torch.equal(
            manager.device_slot_map[layer_idx][current_id], torch.tensor([-1, -1])
        )
        assert torch.equal(
            manager.device_slot_owner[layer_idx][current_id], torch.tensor([-1, -1])
        )
