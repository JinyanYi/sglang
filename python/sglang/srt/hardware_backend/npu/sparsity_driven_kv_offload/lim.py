"""Q1 framework bridge from the DSA indexer to native LIM cache plans.

LIM selects TopK from a stable page-aligned prefix. Both experimental modes
append the same visible tail to attention; the default legacy selection policy
is unchanged. Runtime state belongs to this manager-owned adapter, while the
current plan travels on Ascend ForwardMetadata for IndexShare consumers.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class LIMSelection:
    source_ids: torch.Tensor
    device_slots: torch.Tensor
    topk_misses: torch.Tensor
    miss_sources: torch.Tensor
    miss_slots: torch.Tensor
    miss_counts: torch.Tensor
    prefix_lens: torch.Tensor
    sparse_rows: torch.Tensor
    valid_rows: torch.Tensor
    request_slots: torch.Tensor
    seq_lens: torch.Tensor


def _run_native_lim(*args) -> None:
    """Call the existing wheel wrapper only when the LIM route is active."""
    from sgl_kernel_npu.sparsity_driven_kv_offload import (
        fused_lightning_indexer_manage,
    )

    fused_lightning_indexer_manage(*args)


def prepare_lim_rows(seq_lens, request_slots, last_prefix, capacity, request_rows):
    """Derive native states without host scalar reads, including padding/reset."""
    valid = (request_slots > 0) & (request_slots < request_rows) & (seq_lens > 0)
    safe_slots = torch.where(valid, request_slots, request_rows).to(torch.int32)
    lengths = torch.where(valid, seq_lens, 0).to(torch.int32)
    prefix = ((lengths - 1).clamp(min=0) // 128) * 128
    sparse = valid & (prefix >= capacity)
    previous = last_prefix[safe_slots.long()]
    warm = sparse & (previous >= 0) & (prefix >= previous)
    state = torch.where(sparse, torch.where(warm, -1, -2), -3).to(torch.int32)
    # Native invalid rows take its seq_len < Q guard before touching pool state.
    next_prefix = torch.where(sparse, prefix, -1)
    return safe_slots, lengths, prefix, sparse, valid, state, next_prefix


class LIMDecodeAdapter:
    def __init__(self, manager):
        if manager.indexer_impl != "lim":
            raise ValueError("LIMDecodeAdapter requires indexer_impl='lim'.")
        self.capacity = manager.device_cache_size
        self.rows = manager.size
        self.batch_capacity = manager.batch_capacity
        self.top_k = manager.sparse_context_len
        self.source_capacity = (manager.max_context_len + 127) // 128 * 128
        device = manager.device
        self.source_ids = torch.empty(
            (self.batch_capacity, 1, self.top_k), dtype=torch.int32, device=device
        )
        self.device_slots = torch.empty_like(self.source_ids)
        self.topk_misses = torch.empty(
            self.batch_capacity, dtype=torch.int32, device=device
        )
        self.miss_sources = torch.empty(
            (self.batch_capacity, 32768), dtype=torch.int32, device=device
        )
        self.miss_slots = torch.empty_like(self.miss_sources)
        self.miss_counts = torch.empty_like(self.topk_misses)
        self.query_ends = torch.arange(
            1, self.batch_capacity + 1, dtype=torch.int32, device=device
        )
        self.query_scale = torch.empty(
            (self.batch_capacity, 64), dtype=torch.float32, device=device
        )
        self.block_table = torch.empty(
            (self.batch_capacity, self.source_capacity // 128),
            dtype=torch.int32,
            device=device,
        )
        self.last_prefix = torch.full(
            (manager.layer_num, self.rows + 1), -1, dtype=torch.int32, device=device
        )
        # One scale workspace is shared by all indexer layers. Native BF16 LIM
        # does not read its values, but its ABI requires the complete PA shape.
        # Its variable 4 bytes/index-token are charged to the indexer pool coeff.
        key = manager.paged_kv_cache.get_index_k_buffer(manager.start_layer)
        if key.dtype != torch.bfloat16 or manager.store_dtype != torch.bfloat16:
            raise ValueError(
                "LIM framework integration requires BF16 indexer and KV caches."
            )
        if key.ndim != 4 or key.shape[1:] != (128, 1, 128):
            raise ValueError("LIM requires PA index keys [pages,128,1,128].")
        self.key_scale = torch.empty(key.shape[:3], dtype=torch.float32, device=device)

    def reset_requests(self, request_slots):
        self.last_prefix[:, request_slots] = -1

    def select(
        self,
        manager,
        *,
        query,
        weights,
        key,
        block_table,
        forward_batch,
        layer_id,
    ):
        batch = query.shape[0]
        if (
            batch != forward_batch.req_pool_indices.numel()
            or batch > self.batch_capacity
        ):
            raise ValueError(
                "LIM Q1 requires one query row per padded request and batch <= "
                "graph capacity."
            )
        if (
            block_table.shape[0] != batch
            or block_table.shape[1] > self.block_table.shape[1]
        ):
            raise ValueError(
                "LIM index block-table geometry exceeds its fixed source capacity."
            )
        layer_index = layer_id - manager.start_layer
        slots, lengths, prefix, sparse, valid, state, next_prefix = prepare_lim_rows(
            forward_batch.seq_lens[:batch],
            forward_batch.req_pool_indices,
            self.last_prefix[layer_index],
            self.capacity,
            self.rows,
        )
        table = self.block_table[:batch]
        table.zero_()
        table[:, : block_table.shape[1]].copy_(block_table)
        sources = self.source_ids[:batch]
        # [:, :heads] would be strided for H32; view a contiguous prefix.
        scale = self.query_scale.view(-1)[: batch * query.shape[1]].view(
            batch, query.shape[1]
        )
        sources.fill_(-1)
        self.device_slots[:batch].fill_(-1)
        self.topk_misses[:batch].zero_()
        self.miss_sources[:batch].fill_(-1)
        self.miss_slots[:batch].fill_(-1)
        self.miss_counts[:batch].zero_()
        _run_native_lim(
            weights.contiguous(),
            scale,
            query.contiguous(),
            self.key_scale,
            key,
            table,
            self.query_ends[:batch],
            lengths,
            prefix,
            torch.where(sparse, self.capacity, 0).to(torch.int32),
            state,
            slots,
            manager.lim_slot_map[layer_index],
            sources,
            self.device_slots[:batch],
            self.topk_misses[:batch],
            self.miss_sources[:batch],
            self.miss_slots[:batch],
            self.miss_counts[:batch],
        )
        # The native invalid-query fast path can leave reused output rows
        # untouched during graph replay. Never publish a previous request's IDs.
        sources.masked_fill_(~valid[:, None, None], -1)
        self.last_prefix[layer_index].index_copy_(0, slots.long(), next_prefix)
        return LIMSelection(
            sources[:, 0],
            self.device_slots[:batch, 0],
            self.topk_misses[:batch],
            self.miss_sources[:batch],
            self.miss_slots[:batch],
            self.miss_counts[:batch],
            prefix,
            sparse,
            valid,
            slots,
            lengths,
        )
