from types import SimpleNamespace

import torch
import pytest
from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.block_tables import (
    fill_request_block_table,
)
from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.copy_sfa_buffers import (
    copy_sfa_host_row_stride,
    view_copy_sfa_output,
)
from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.prefix_tail import (
    derive_prefix_tail_metadata,
    fill_prefix_tail_block_table,
    resident_tail_copy_indices,
    resident_tail_write_mask,
    sparse_batch_capacity,
    valid_sparse_request_rows,
)
from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.lim import (
    LIMDecodeAdapter,
    prepare_lim_rows,
)


def test_fill_request_block_table_multiple_rows_and_padding():
    req_ids = torch.tensor([3, 1, 9], dtype=torch.int32)
    valid_req = torch.tensor([True, True, False])
    offsets = torch.arange(3, dtype=torch.int32)
    output = torch.empty((3, 3), dtype=torch.int32)

    fill_request_block_table(req_ids, valid_req, offsets, 5, output)

    assert torch.equal(
        output,
        torch.tensor([[15, 16, 17], [5, 6, 7], [0, 1, 2]], dtype=torch.int32),
    )


def test_fill_request_block_table_single_page_and_zero_request():
    req_ids = torch.tensor([0, 2], dtype=torch.int32)
    valid_req = torch.tensor([False, True])
    offsets = torch.tensor([0], dtype=torch.int32)
    output = torch.empty((2, 1), dtype=torch.int32)

    fill_request_block_table(req_ids, valid_req, offsets, 4, output)

    assert torch.equal(output, torch.tensor([[0], [8]], dtype=torch.int32))


def test_copy_sfa_output_uses_contiguous_prefix_for_unpadded_heads():
    output_buffer = torch.arange(4 * 8 * 512, dtype=torch.float32).view(4, 8, 512)

    output = view_copy_sfa_output(output_buffer, 2, 3, 512)

    assert output.shape == (2, 3, 512)
    assert output.is_contiguous()
    assert output.stride() == (3 * 512, 512, 1)
    assert output.data_ptr() == output_buffer.data_ptr()
    assert torch.equal(output.reshape(-1), output_buffer.reshape(-1)[: 2 * 3 * 512])


def test_copy_sfa_host_stride_aligns_without_changing_legacy_stride():
    assert copy_sfa_host_row_stride(32772, enabled=True) == 32896
    assert copy_sfa_host_row_stride(32768, enabled=True) == 32768
    assert copy_sfa_host_row_stride(32772, enabled=False) == 32772


def test_prefix_tail_metadata_handles_dense_boundary_and_page_rollover():
    lengths = torch.tensor([0, 1024, 2048, 2049, 2176, 2177], dtype=torch.int32)

    plan = derive_prefix_tail_metadata(
        lengths, cache_capacity=2048, sparse_context_len=2048
    )

    assert torch.equal(
        plan.indexer_seq_lens,
        torch.tensor([0, 1024, 2048, 2048, 2048, 2176], dtype=torch.int32),
    )
    assert torch.equal(
        plan.tail_lens, torch.tensor([0, 0, 0, 1, 128, 1], dtype=torch.int32)
    )
    assert torch.equal(
        plan.cache_tokens, torch.tensor([0, 0, 0, 2048, 2048, 2048], dtype=torch.int32)
    )
    assert torch.equal(
        plan.actual_kv_lens,
        torch.tensor([0, 1024, 2048, 2049, 2176, 2049], dtype=torch.int32),
    )


def test_prefix_tail_page_table_maps_ring_pages_and_padded_rows():
    req_ids = torch.tensor([3, 1, 9], dtype=torch.int32)
    valid_rows = torch.tensor([True, True, False])
    prefixes = torch.tensor([2048, 2176, 0], dtype=torch.int32)
    offsets = torch.arange(18, dtype=torch.int32)
    output = torch.empty((3, 18), dtype=torch.int32)

    fill_prefix_tail_block_table(
        req_ids,
        valid_rows,
        prefixes,
        offsets,
        cache_capacity=2048,
        tail_capacity=256,
        output=output,
    )

    assert torch.equal(output[0, :16], torch.arange(54, 70, dtype=torch.int32))
    assert torch.equal(output[0, 16:], torch.tensor([70, 71], dtype=torch.int32))
    assert torch.equal(output[1, :16], torch.arange(18, 34, dtype=torch.int32))
    assert torch.equal(output[1, 16:], torch.tensor([35, 34], dtype=torch.int32))
    assert torch.equal(output[2, :], torch.arange(18, dtype=torch.int32))


def test_graph_scratch_capacity_exceeds_pool_without_widening_request_ids():
    request_capacity = 9
    assert sparse_batch_capacity(request_capacity, graph_batch_capacity=16) == 16
    assert sparse_batch_capacity(request_capacity) == request_capacity

    req_ids = torch.tensor([0, 8, 9, -1], dtype=torch.int32)
    seq_lens = torch.tensor([1, 1, 1, 1], dtype=torch.int32)
    assert torch.equal(
        valid_sparse_request_rows(req_ids, seq_lens, request_capacity),
        torch.tensor([True, True, False, False]),
    )


def test_prefix_tail_selection_boundary_is_independent_of_cache_capacity():
    lengths = torch.tensor([2048, 2049, 3000, 4097], dtype=torch.int32)

    plan = derive_prefix_tail_metadata(
        lengths, cache_capacity=4096, sparse_context_len=2048
    )

    assert torch.equal(
        plan.indexer_seq_lens,
        torch.tensor([2048, 2048, 2944, 4096], dtype=torch.int32),
    )
    assert torch.equal(plan.sparse_rows, torch.tensor([False, True, True, True]))
    assert torch.equal(plan.tail_lens, torch.tensor([0, 1, 56, 1], dtype=torch.int32))
    assert torch.equal(
        plan.actual_kv_lens,
        torch.tensor([2048, 4097, 4152, 4097], dtype=torch.int32),
    )


def test_resident_tail_prefill_writes_only_latest_256_ring_slots():
    token_positions = torch.arange(512, dtype=torch.int64)
    valid = torch.ones(512, dtype=torch.bool)
    visible_lens = torch.full_like(token_positions, 512)

    write_mask = resident_tail_write_mask(valid, token_positions, visible_lens)
    ring_slots = token_positions[write_mask] % 256

    assert torch.equal(token_positions[write_mask], torch.arange(256, 512))
    assert torch.unique(ring_slots).numel() == 256

    req_ids = torch.zeros_like(token_positions)
    destinations, copy_mask = resident_tail_copy_indices(
        req_ids,
        token_positions,
        visible_lens,
        valid,
        device_kv_stride=4352,
        cache_capacity=4096,
    )
    hbm = torch.full((4352,), -1, dtype=torch.int64)
    hbm[destinations[copy_mask]] = token_positions[copy_mask]
    assert torch.equal(hbm[4096 + ring_slots], torch.arange(256, 512))


def test_prefix_tail_graph_metadata_storage_survives_dense_sparse_transitions():
    initial = derive_prefix_tail_metadata(
        torch.tensor([2048, 0], dtype=torch.int32),
        cache_capacity=4096,
        sparse_context_len=2048,
    )
    pointers = tuple(
        tensor.data_ptr()
        for tensor in (
            initial.indexer_seq_lens,
            initial.prefix_lens,
            initial.tail_lens,
            initial.sparse_rows,
            initial.cache_tokens,
            initial.actual_kv_lens,
        )
    )

    transitioned = derive_prefix_tail_metadata(
        torch.tensor([2049, 3000], dtype=torch.int32),
        cache_capacity=4096,
        sparse_context_len=2048,
        out=initial,
    )

    assert transitioned is initial
    assert pointers == tuple(
        tensor.data_ptr()
        for tensor in (
            transitioned.indexer_seq_lens,
            transitioned.prefix_lens,
            transitioned.tail_lens,
            transitioned.sparse_rows,
            transitioned.cache_tokens,
            transitioned.actual_kv_lens,
        )
    )
    assert torch.equal(
        transitioned.indexer_seq_lens, torch.tensor([2048, 2944], dtype=torch.int32)
    )
    assert torch.equal(transitioned.tail_lens, torch.tensor([1, 56], dtype=torch.int32))


def test_lim_rows_prepare_cold_warm_dense_and_padding_states():
    seq_lens = torch.tensor([2049, 4097, 2048, 0], dtype=torch.int32)
    request_slots = torch.tensor([1, 2, 3, 0], dtype=torch.int32)
    last_prefix = torch.full((5,), -1, dtype=torch.int32)
    last_prefix[1] = 2048

    slots, lengths, prefix, sparse, valid, state, next_prefix = prepare_lim_rows(
        seq_lens, request_slots, last_prefix, capacity=2048, request_rows=4
    )

    assert torch.equal(slots, torch.tensor([1, 2, 3, 4], dtype=torch.int32))
    assert torch.equal(lengths, torch.tensor([2049, 4097, 2048, 0], dtype=torch.int32))
    assert torch.equal(prefix, torch.tensor([2048, 4096, 1920, 0], dtype=torch.int32))
    assert torch.equal(sparse, torch.tensor([True, True, False, False]))
    assert torch.equal(valid, torch.tensor([True, True, True, False]))
    # -1 means retain/warm, -2 means a cold or reset fill, -3 means dense/pad.
    assert torch.equal(state, torch.tensor([-1, -2, -3, -3], dtype=torch.int32))
    assert torch.equal(
        next_prefix, torch.tensor([2048, 4096, -1, -1], dtype=torch.int32)
    )


def _make_reference_lim_adapter(batch_capacity=3, top_k=4):
    adapter = LIMDecodeAdapter.__new__(LIMDecodeAdapter)
    adapter.impl = "reference"
    adapter.is_native = False
    adapter.capacity = 4
    adapter.rows = 3
    adapter.batch_capacity = batch_capacity
    adapter.top_k = top_k
    adapter.source_capacity = 256
    adapter.width = top_k + 128
    adapter.device = torch.device("cpu")
    adapter.source_ids = torch.empty(
        (batch_capacity, 1, top_k), dtype=torch.int32
    )
    adapter.device_slots = torch.empty_like(adapter.source_ids)
    adapter.topk_misses = torch.empty(batch_capacity, dtype=torch.int32)
    adapter.miss_sources = torch.empty((batch_capacity, 8), dtype=torch.int32)
    adapter.miss_slots = torch.empty_like(adapter.miss_sources)
    adapter.miss_counts = torch.empty(batch_capacity, dtype=torch.int32)
    adapter.query_ends = torch.arange(1, batch_capacity + 1, dtype=torch.int32)
    adapter.block_table = torch.empty((batch_capacity, 2), dtype=torch.int32)
    adapter.last_prefix = torch.full((1, adapter.rows + 1), -1, dtype=torch.int32)
    return adapter


def test_lim_reference_graph_scratch_reuses_storage_and_clears_padded_ids():
    adapter = _make_reference_lim_adapter()
    manager = SimpleNamespace(
        start_layer=0,
        size=adapter.rows,
        indexer_impl="reference",
    )
    forward_batch = SimpleNamespace(
        req_pool_indices=torch.tensor([1, 2, 0], dtype=torch.int32),
        seq_lens=torch.tensor([2049, 2048, 0], dtype=torch.int32),
    )
    query = torch.zeros((3, 32, 128), dtype=torch.bfloat16)
    weights = torch.zeros((3, 32), dtype=torch.bfloat16)
    key = torch.zeros((2, 128, 1, 128), dtype=torch.bfloat16)
    block_table = torch.zeros((3, 2), dtype=torch.int32)
    pointers = (adapter.source_ids.data_ptr(), adapter.block_table.data_ptr())

    def indexer(result):
        def run(**kwargs):
            return result

        return run

    first = torch.tensor([[[1, 2, 3, 4]], [[5, 6, 7, 8]], [[9, 9, 9, 9]]])
    plan = adapter.select(
        manager,
        query=query,
        weights=weights,
        key=key,
        block_table=block_table,
        forward_batch=forward_batch,
        layer_id=0,
        reference_indexer=indexer(first),
    )
    assert torch.equal(
        plan.source_ids,
        torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [-1, -1, -1, -1]], dtype=torch.int32),
    )

    second = torch.tensor([[[11, 12, 13, 14]], [[15, 16, 17, 18]], [[19, 19, 19, 19]]])
    replay = adapter.select(
        manager,
        query=query,
        weights=weights,
        key=key,
        block_table=block_table,
        forward_batch=forward_batch,
        layer_id=0,
        reference_indexer=indexer(second),
    )
    assert (adapter.source_ids.data_ptr(), adapter.block_table.data_ptr()) == pointers
    assert torch.equal(
        replay.source_ids,
        torch.tensor(
            [[11, 12, 13, 14], [15, 16, 17, 18], [-1, -1, -1, -1]],
            dtype=torch.int32,
        ),
    )


def test_lim_copy_sfa_reuses_anchor_plan_with_layer_local_kv_and_fresh_sources(
    monkeypatch,
):
    pytest.importorskip("sgl_kernel_npu")
    from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.manager import (
        SparseKVCacheManager,
    )

    batch_size = 2
    cache_capacity = 2048
    context_len = 2560
    device_stride = cache_capacity + 256
    manager = SparseKVCacheManager.__new__(SparseKVCacheManager)
    manager.start_layer = 0
    manager.size = 3
    manager.batch_capacity = batch_size
    manager.cache_capacity = cache_capacity
    manager.sparse_context_len = cache_capacity
    manager.prefix_tail_enabled = True
    manager.resident_tail_capacity = 256
    manager.logical_max_context_len = context_len
    manager.max_context_len = context_len
    manager.device_kv_stride = device_stride
    manager.device = torch.device("cpu")
    manager.head_num = 1
    manager.head_dim = 576
    manager.kv_lora_rank = 512
    manager.store_dtype = torch.bfloat16
    manager.dev_ptr_list = [101, 202]
    manager.device_kv_buffer = [
        torch.zeros((manager.size, device_stride, 1, 576), dtype=torch.bfloat16),
        torch.ones((manager.size, device_stride, 1, 576), dtype=torch.bfloat16),
    ]
    manager.planner_attn_out = torch.full(
        (batch_size, 4, manager.kv_lora_rank), 99, dtype=torch.bfloat16
    )
    manager.planner_topk_src_ids = torch.full(
        (batch_size, 1, cache_capacity), -1, dtype=torch.int32
    )
    manager.planner_topk_dst_slots = torch.full_like(manager.planner_topk_src_ids, -1)
    manager.planner_topk_miss_counts = torch.zeros(batch_size, dtype=torch.int32)
    manager.planner_miss_src_ids = torch.full(
        (batch_size, cache_capacity), -1, dtype=torch.int32
    )
    manager.planner_miss_dst_slots = torch.full_like(manager.planner_miss_src_ids, -1)
    manager.planner_miss_counts = torch.zeros(batch_size, dtype=torch.int32)
    manager.planner_actual_seq_lengths_query = torch.arange(
        1, batch_size + 1, dtype=torch.int32
    )
    manager.planner_actual_seq_lengths_kv = torch.empty(batch_size, dtype=torch.int32)
    manager.planner_num_cache_tokens = torch.empty(batch_size, dtype=torch.int32)
    manager.hbm_page_offsets = torch.arange(device_stride // 128, dtype=torch.int32)
    manager.dram_page_offsets = torch.arange(context_len // 128, dtype=torch.int32)
    manager.planner_hbm_block_table = torch.empty(
        (batch_size, device_stride // 128), dtype=torch.int32
    )
    manager.planner_dram_block_table = torch.empty(
        (batch_size, context_len // 128), dtype=torch.int32
    )
    manager.make_prefix_tail_metadata = lambda seq_lens: derive_prefix_tail_metadata(
        seq_lens, cache_capacity, sparse_context_len=cache_capacity
    )
    manager._plan_sparse_kv_cache = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("LIM plans must bypass the generic planner")
    )

    source_ids = torch.full((batch_size, cache_capacity), 7, dtype=torch.int32)
    plan = SimpleNamespace(
        source_ids=source_ids,
        device_slots=torch.arange(cache_capacity, dtype=torch.int32)
        .view(1, -1)
        .expand(batch_size, -1),
        topk_misses=torch.full((batch_size,), cache_capacity, dtype=torch.int32),
        miss_sources=torch.arange(cache_capacity, dtype=torch.int32)
        .view(1, -1)
        .expand(batch_size, -1),
        miss_slots=torch.arange(cache_capacity, dtype=torch.int32)
        .view(1, -1)
        .expand(batch_size, -1),
        miss_counts=torch.full((batch_size,), cache_capacity, dtype=torch.int32),
    )
    req_ids = torch.tensor([1, 0], dtype=torch.int32)
    seq_lens = torch.tensor([2049, 0], dtype=torch.int32)
    forward_batch = SimpleNamespace(
        batch_size=batch_size, req_pool_indices=req_ids, seq_lens=seq_lens
    )
    query_nope = torch.zeros((batch_size, 1, 2, 512), dtype=torch.bfloat16)
    query_rope = torch.zeros((batch_size, 1, 2, 64), dtype=torch.bfloat16)
    calls = []

    def fake_copy_sfa(*args):
        cache, topk_src, attention_out = args[2], args[6], args[-2]
        assert torch.count_nonzero(attention_out) == 0
        calls.append((cache.data_ptr(), args[14], topk_src.clone()))
        attention_out[0].fill_(len(calls))

    monkeypatch.setattr(
        torch.ops.npu,
        "fused_scatter_copy_sparse_flash_attention",
        fake_copy_sfa,
        raising=False,
    )

    outputs = []
    for layer_id in (0, 1):
        layer = SimpleNamespace(layer_id=layer_id)
        outputs.append(
            manager.copy_sfa_decode(
                layer,
                forward_batch,
                query_nope,
                query_rope,
                query_heads=2,
                scale_value=1.0,
                topk_src_ids=plan.source_ids,
                lim_plan=plan,
            ).clone()
        )

    assert calls[0][0] != calls[1][0]
    assert (calls[0][1], calls[1][1]) == (101, 202)
    assert torch.equal(calls[0][2], calls[1][2])
    assert torch.all(outputs[0][0] == 1)
    assert torch.all(outputs[1][0] == 2)
    assert torch.count_nonzero(outputs[0][1]) == 0
    assert torch.count_nonzero(outputs[1][1]) == 0

    source_ids[0, 0] = 19
    manager.copy_sfa_decode(
        SimpleNamespace(layer_id=1),
        forward_batch,
        query_nope,
        query_rope,
        query_heads=2,
        scale_value=1.0,
        topk_src_ids=source_ids,
        lim_plan=plan,
    )
    assert calls[-1][2][0, 0, 0] == 19
