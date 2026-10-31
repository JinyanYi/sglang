"""A5 C8 CopySFA consumes external LIM plans and canonical radix host pages."""

import torch

from sglang.kernels.ops.attention.npu_quant_sparse import (
    fused_quant_scatter_copy_sparse_flash_attention,
    load_quant_copy_sfa_operator,
)
from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.manager import (
    unidex_copy_inplace,
)
from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.mtp import (
    use_native_mtp_verify,
)

PACKED_ROW_BYTES = 656


class C8Cache:
    def __init__(self, manager, query_heads, compute_dtype):
        self.operator = load_quant_copy_sfa_operator()
        self.query = torch.empty(
            (manager.batch_capacity * 4, query_heads, 576),
            dtype=compute_dtype,
            device=manager.device,
        )
        self.query_ends = {
            q: torch.arange(
                q,
                q * (manager.batch_capacity + 1),
                q,
                dtype=torch.int32,
                device=manager.device,
            )
            for q in (1, 4)
        }

    def attention(self, manager, layer, batch, q, q_rope, plan):
        if plan is None:
            raise RuntimeError("A5 CopySFA requires a plan from the external A5 LIM")
        count = batch.batch_size
        width = plan.source_ids.shape[0] // count
        rows = count * width
        manager.lim_adapter.copy_sfa_metadata(manager, plan)
        heads = layer.tp_q_head_num
        query = self.query[:rows]
        query.zero_()
        query[:, :heads, :512].copy_(q[:rows].reshape(rows, -1, 512)[:, :heads])
        query[:, :heads, 512:].copy_(q_rope[:rows].reshape(rows, -1, 64)[:, :heads])
        layer_index = layer.layer_id - manager.start_layer
        output = manager.mtp_attn_out[:rows]
        fused_quant_scatter_copy_sparse_flash_attention(
            query,
            manager.device_kv_buffer[layer_index].view(-1, 128, 1, 656),
            manager.host_kv_buffer[layer_index].view(-1, 128, 1, 656),
            manager.dev_ptr_list[layer_index],
            manager.planner_hbm_block_table[:count],
            manager.planner_dram_block_table[:count],
            plan.device_slots.unsqueeze(1),
            plan.source_ids.unsqueeze(1),
            plan.topk_misses,
            self.query_ends[width][:count],
            manager.planner_actual_seq_lengths_kv[:count],
            manager.planner_num_cache_tokens[:count],
            plan.miss_sources.unsqueeze(1),
            plan.miss_slots.unsqueeze(1),
            plan.miss_counts,
            layer.scaling,
            output,
            operator=self.operator,
        )
        valid = (
            (plan.valid_rows & plan.sparse_rows)[:, None]
            .expand(count, width)
            .reshape(rows)
        )
        output.masked_fill_(~valid[:, None, None], 0)
        manager.last_decode_path = batch.sparse_kv_decode_path = "a5_c8_copy_sfa"
        return output[:, :heads].reshape(rows, -1).contiguous()


def forward_c8_offload(
    backend, q, k, layer, batch, save_kv_cache, q_rope, k_rope, topk_indices
):
    """Use Q1/Q4 CopySFA for long targets; prefill/short rows use vendor QSFA."""
    import torch_npu

    manager = backend.sparse_kv_manager
    stream = torch.npu.current_stream(backend.device)
    if save_kv_cache:
        manager.offload_v2(
            k.reshape(-1, 1, 512), k_rope.reshape(-1, 1, 64), layer, batch, stream
        )
    cpu_lengths = backend.forward_metadata.seq_lens_cpu_int
    native_q1 = (
        batch.forward_mode.is_decode()
        and batch.spec_info is None
        and (
            backend.graph_mode
            or (
                cpu_lengths is not None
                and all(
                    int(n) == 0 or int(n) >= manager.cache_capacity + 128
                    for n in cpu_lengths
                )
            )
        )
    )
    if use_native_mtp_verify(backend, batch, manager) or native_q1:
        result = manager.c8_cache.attention(
            manager,
            layer,
            batch,
            q,
            q_rope,
            backend.forward_metadata.sparse_kv_lim_plan,
        )
        return torch.nn.functional.pad(result, (0, 0, 0, q.shape[0] - result.shape[0]))
    if not (
        batch.forward_mode.is_extend_without_speculative()
        or batch.forward_mode.is_target_verify()
        or batch.forward_mode.is_decode()
    ):
        raise ValueError(
            "A5 offload supports target prefill, Q1 decode and Q4 verify; draft stays native"
        )
    if backend.graph_mode:
        raise RuntimeError("Short A5 target rows must use the eager QSFA fallback")
    manager.lim_adapter.reset_requests(batch.req_pool_indices)
    reqs_cpu = batch.req_pool_indices_cpu.tolist()
    if batch.forward_mode.is_target_verify():
        lengths = (
            batch.positions[: batch.batch_size * 4].view(batch.batch_size, 4)[:, -1] + 1
        )
        kv_lengths = torch.where(batch.seq_lens > 0, lengths, 0).to(torch.int32)
        # Short verification uses the eager vendor path, never graph replay.
        lengths_cpu = kv_lengths.cpu().tolist()
    else:
        kv_lengths = batch.seq_lens
        lengths_cpu = batch.seq_lens_cpu.tolist()
    pages_cpu = []
    for request, length in zip(reqs_cpu, lengths_cpu):
        if (
            length < 0
            or length > manager.max_context_len
            or (length > 0 and not 0 < request < manager.size)
        ):
            raise ValueError(
                "C8 prefill request/length exceeds registered host storage."
            )
        pages_cpu.append((int(length) + 127) // 128)
    total_pages = max(1, sum(pages_cpu))
    max_pages = max(1, max(pages_cpu, default=0))
    pages = torch.tensor(pages_cpu, dtype=torch.int64, device=q.device)
    starts = pages.cumsum(0) - pages
    rows = torch.repeat_interleave(
        batch.req_pool_indices.long(), pages * 128, output_size=sum(pages_cpu) * 128
    )
    offsets = torch.repeat_interleave(
        starts * 128, pages * 128, output_size=rows.numel()
    )
    positions = torch.arange(rows.numel(), device=q.device) - offsets
    lengths = torch.repeat_interleave(kv_lengths, pages * 128, output_size=rows.numel())
    packed = torch.zeros(
        (total_pages, 128, 1, PACKED_ROW_BYTES), dtype=torch.int8, device=q.device
    )
    layer_index = layer.layer_id - manager.start_layer
    unidex_copy_inplace(
        manager.host_kv_buffer[layer_index],
        packed.view(-1, 1, PACKED_ROW_BYTES),
        manager.host_source_indices(rows, positions).contiguous(),
        torch.arange(rows.numel(), dtype=torch.int64, device=q.device),
        (positions < lengths).contiguous(),
        2,
        1,
        block_dim=48,
        src_ptr=manager.dev_ptr_list[layer_index],
    )
    page_offsets = torch.arange(max_pages, device=q.device)
    table = torch.where(
        page_offsets[None] < pages[:, None], starts[:, None] + page_offsets[None], 0
    ).to(torch.int32)
    ends = backend.forward_metadata.actual_seq_lengths_q
    if ends is None:
        if batch.forward_mode.is_extend_without_speculative():
            ends = batch.extend_seq_lens.cumsum(0)
        else:
            width = 4 if batch.forward_mode.is_target_verify() else 1
            ends = manager.c8_cache.query_ends[width][: batch.batch_size]
    merged = torch.cat(
        (q.reshape(-1, q.shape[-2], 512), q_rope.reshape(-1, q_rope.shape[-2], 64)),
        dim=-1,
    ).contiguous()
    padded_heads = manager.c8_cache.query.shape[1]
    if merged.shape[1] != padded_heads:
        padded = torch.zeros(
            (merged.shape[0], padded_heads, 576),
            dtype=merged.dtype,
            device=merged.device,
        )
        padded[:, : merged.shape[1]].copy_(merged)
        merged = padded
    result = torch_npu.npu_kv_quant_sparse_flash_attention(
        query=merged,
        key=packed.view(torch.float8_e4m3fn),
        value=packed.view(torch.float8_e4m3fn),
        sparse_indices=topk_indices.reshape(-1, 1, manager.sparse_context_len),
        scale_value=layer.scaling,
        key_quant_mode=2,
        value_quant_mode=2,
        key_dequant_scale=None,
        value_dequant_scale=None,
        actual_seq_lengths_query=ends.to(device=q.device, dtype=torch.int32),
        actual_seq_lengths_kv=kv_lengths.to(torch.int32),
        block_table=table,
        sparse_block_size=1,
        layout_query="TND",
        layout_kv="PA_BSND",
        sparse_mode=3,
        attention_mode=2,
        quant_scale_repo_mode=1,
        tile_size=128,
        rope_head_dim=64,
    )
    result = result[0] if isinstance(result, tuple) else result
    return result[:, : layer.tp_q_head_num].contiguous()
