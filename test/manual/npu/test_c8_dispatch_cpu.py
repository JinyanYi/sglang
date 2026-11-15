"""Verify C8 v2 argument ownership and packing fallback without native launch."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

SOURCE = (
    Path(__file__).resolve().parents[3]
    / "python/sglang/srt/hardware_backend/npu/sparsity_driven_kv_offload/c8.py"
)


def fixture(width=4):
    tree = ast.parse(SOURCE.read_text())
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "C8Cache"
    )
    calls = []

    def original(*args, operator=None):
        calls.append(("old", args))
        args[-1].fill_(3)

    def v2(*args):
        calls.append(("v2", args))
        args[17].fill_(3)

    namespace = {
        "torch": torch,
        "fused_quant_scatter_copy_sparse_flash_attention": original,
    }
    exec(
        compile(ast.Module(body=[cls], type_ignores=[]), str(SOURCE), "exec"), namespace
    )
    cache = object.__new__(namespace["C8Cache"])
    rows, heads = 2 * width, 8
    cache.query = torch.full((rows, heads, 576), -17, dtype=torch.bfloat16)
    cache.query_ends = {width: torch.tensor([width, 2 * width], dtype=torch.int32)}
    cache.operator, cache.operator_v2 = object(), v2
    cache.split_query, cache.fused_ring_tail = True, True
    manager = SimpleNamespace(
        lim_adapter=SimpleNamespace(copy_sfa_metadata=lambda manager, plan: None),
        device_kv_buffer=[torch.empty(2, 128, 1, 656, dtype=torch.int8)],
        host_kv_buffer=[torch.empty(2, 128, 1, 656, dtype=torch.int8)],
        dev_ptr_list=[1234],
        cache_capacity=8192,
        planner_hbm_block_table=torch.zeros(2, 66, dtype=torch.int32),
        planner_dram_block_table=torch.zeros(2, 80, dtype=torch.int32),
        planner_actual_seq_lengths_kv=torch.tensor([8500, 0], dtype=torch.int32),
        planner_num_cache_tokens=torch.tensor([8300, 0], dtype=torch.int32),
        mtp_attn_out=torch.empty(rows, heads, 512, dtype=torch.bfloat16),
        tail_starts=[torch.full((4,), -1, dtype=torch.int32)],
        tail_ends=[torch.full((4,), -1, dtype=torch.int32)],
        start_layer=0,
    )
    plan = SimpleNamespace(
        source_ids=torch.zeros(rows, 2048, dtype=torch.int32),
        device_slots=torch.zeros(rows, 2048, dtype=torch.int32),
        topk_misses=torch.zeros(rows, dtype=torch.int32),
        miss_sources=torch.zeros(2, 32768, dtype=torch.int32),
        miss_slots=torch.zeros(2, 32768, dtype=torch.int32),
        miss_counts=torch.zeros(2, dtype=torch.int32),
        valid_rows=torch.tensor([True, False]),
        sparse_rows=torch.tensor([True, False]),
        request_slots=torch.tensor([1, 3], dtype=torch.int32),
    )
    layer = SimpleNamespace(tp_q_head_num=heads, layer_id=0, scaling=0.1)
    q = torch.randn(rows, heads, 512, dtype=torch.bfloat16)
    rope = torch.randn(rows, heads, 64, dtype=torch.bfloat16)
    batch = SimpleNamespace(batch_size=2)
    return cache, manager, plan, layer, q, rope, batch, calls


@pytest.mark.parametrize("width", [1, 4])
@pytest.mark.parametrize("ring", [False, True])
def test_v2_preserves_split_views_and_ring_state_owners(width, ring):
    cache, manager, plan, layer, q, rope, batch, calls = fixture(width)
    current = (
        torch.zeros(q.shape[0], 1, 656, dtype=torch.int8),
        torch.arange(q.shape[0], dtype=torch.int64),
        torch.ones(q.shape[0], dtype=torch.bool),
    )
    output = cache.attention(
        manager, layer, batch, q, rope, plan, current_inputs=current if ring else None
    )
    kind, args = calls[0]
    assert kind == "v2" and len(args) == 25
    assert args[0].data_ptr() == q.data_ptr()
    assert args[1].data_ptr() == rope.data_ptr()
    assert args[4] == manager.dev_ptr_list[0]
    assert args[18] == manager.cache_capacity
    assert torch.all(cache.query == -17), "v2 must not clear or pack the fallback query"
    if ring:
        for actual, expected in zip(
            args[-6:],
            (
                *current,
                plan.request_slots,
                manager.tail_starts[0],
                manager.tail_ends[0],
            ),
        ):
            assert actual.data_ptr() == expected.data_ptr()
    else:
        assert args[-6:] == (None,) * 6
    assert torch.all(output[:width] == 3)
    assert not output[width:].count_nonzero()


def test_noncontiguous_split_inputs_use_existing_packing():
    cache, manager, plan, layer, q, rope, batch, calls = fixture()
    q = torch.randn(q.shape[0], 8, 1024, dtype=q.dtype)[..., ::2]
    assert not q.is_contiguous()
    cache.attention(manager, layer, batch, q, rope, plan)
    assert calls[0][0] == "old"
    torch.testing.assert_close(cache.query[..., :512], q)
    torch.testing.assert_close(cache.query[..., 512:], rope)
