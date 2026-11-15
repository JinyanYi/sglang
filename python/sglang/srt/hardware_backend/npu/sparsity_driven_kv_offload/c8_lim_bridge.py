"""Caller-side bridge to an independently supplied A5 LIM implementation.

No index retrieval, cache-management algorithm or native LIM is implemented here.
The 19-tensor ABI must return prefix-relative tail slots, as in the referenced
McZyWu port. The CopySFA page table maps those logical slots onto a circular tail.
"""

from dataclasses import dataclass

import torch

from sglang.kernels.ops.attention.npu_quant_sparse import (
    fused_quant_lightning_indexer_manage,
    load_quant_lim_operator,
)
from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.c8_metadata import (
    fused_lim_copy_sfa_metadata,
    fused_prepare_lim_metadata,
)
from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.c8_mtp import (
    validate_mtp_chain,
)


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

    @property
    def query_width(self):
        return self.source_ids.shape[0] // self.request_slots.numel()


class C8LIMBridge:
    """Own graph-stable call buffers; delegate all planning to external LIM."""

    is_native = True
    full_visible = False
    packed_c8 = True

    def __init__(self, manager):
        self.operator = load_quant_lim_operator()
        self.capacity = manager.cache_capacity
        self.rows = manager.size
        self.batch_capacity = manager.batch_capacity
        self.source_capacity = manager.max_context_len
        self.mtp_full_visible = True
        self.fused_pre_metadata = envs.SGLANG_NPU_FUSED_LIM_PRE_METADATA.get()
        self.fused_copy_metadata = envs.SGLANG_NPU_FUSED_LIM_COPY_SFA_METADATA.get()
        device = manager.device
        for name, spec in manager.budget.adapter_buffers.items():
            fill = -1 if name in ("last_prefix", "_metadata_prefix_scratch") else None
            setattr(self, name, spec.allocate(device, fill))
        self.query_ends = {
            q: torch.arange(
                q, q * (self.batch_capacity + 1), q, dtype=torch.int32, device=device
            )
            for q in (1, 4)
        }
        self.pre = {
            name: spec.allocate(device)
            for name, spec in manager.budget.pre_buffers.items()
        }
        manager.budget.validate_buffers(self, manager.budget.adapter_buffers)

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
        query_scale=None,
        key_scale=None,
        reference_indexer=None,
    ):
        return self._select(
            manager,
            query=query,
            weights=weights,
            key=key,
            block_table=block_table,
            forward_batch=forward_batch,
            layer_id=layer_id,
            query_scale=query_scale,
            key_scale=key_scale,
            width=1,
        )

    def select_mtp(
        self,
        manager,
        *,
        query,
        weights,
        key,
        block_table,
        forward_batch,
        layer_id,
        query_scale=None,
        key_scale=None,
    ):
        return self._select(
            manager,
            query=query,
            weights=weights,
            key=key,
            block_table=block_table,
            forward_batch=forward_batch,
            layer_id=layer_id,
            query_scale=query_scale,
            key_scale=key_scale,
            width=4,
        )

    def _select(
        self,
        manager,
        *,
        query,
        weights,
        key,
        block_table,
        forward_batch,
        layer_id,
        query_scale,
        key_scale,
        width,
    ):
        if query_scale is None or key_scale is None:
            raise ValueError("A5 LIM needs the actual FP32 query and key scales")
        batch = forward_batch.batch_size
        rows = batch * width
        if batch > self.batch_capacity or query.shape[0] < rows:
            raise ValueError("A5 LIM input exceeds its preallocated graph capacity")
        lengths = forward_batch.seq_lens[:batch]
        layer = layer_id - manager.start_layer
        table = self.block_table[:batch]
        if block_table.shape[0] != batch:
            raise ValueError("A5 LIM block table must contain one row per request")
        columns = min(block_table.shape[1], table.shape[1])
        pre = {name: tensor[:batch] for name, tensor in self.pre.items()}
        if self.fused_pre_metadata:
            fused_prepare_lim_metadata(
                seq_lens=lengths,
                positions=forward_batch.positions[:rows],
                locations=forward_batch.out_cache_loc[:rows],
                context_len=manager.logical_max_context_len,
                host_capacity=manager.host_token_capacity,
                request_slots=forward_batch.req_pool_indices,
                last_prefix=self.last_prefix[layer],
                block_table=block_table[:, :columns],
                out_block_table=table,
                capacity=self.capacity,
                request_rows=self.rows,
                full_visible=False,
                query_width=width,
                **pre,
            )
        else:
            lengths = forward_batch.positions[:rows].view(batch, width)[:, -1] + 1
            valid = (
                (forward_batch.req_pool_indices > 0)
                & (forward_batch.req_pool_indices < self.rows)
                & (lengths >= width)
                & (forward_batch.seq_lens[:batch] > 0)
            )
            from sglang.kernels.ops.kvcache import prepare_sparse_kv_publication_rows

            _, _, _, publication_valid = prepare_sparse_kv_publication_rows(
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                forward_batch.positions,
                forward_batch.out_cache_loc,
                tokens=query.shape[0],
                request_rows=self.rows,
                context_len=manager.logical_max_context_len,
                host_capacity=manager.host_token_capacity,
                query_width=width,
            )
            valid = valid & publication_valid[:rows].view(batch, width).all(dim=1)
            slots = torch.where(valid, forward_batch.req_pool_indices, self.rows).to(
                torch.int32
            )
            prefix = (lengths - width).clamp_min(0) // 128 * 128
            sparse = valid & (prefix >= self.capacity)
            previous = self.last_prefix[layer, slots.long()]
            warm = sparse & (previous >= 0) & (prefix >= previous)
            values = dict(
                safe_slots=slots,
                lengths=torch.where(valid, lengths, 0),
                prefixes=prefix,
                sparse_rows=sparse,
                valid_rows=valid,
                request_states=torch.where(sparse, torch.where(warm, -1, -2), -3),
                next_prefixes=torch.where(sparse, prefix, -1),
                cache_sizes=torch.where(sparse, self.capacity, 0),
            )
            for name, value in values.items():
                pre[name].copy_(value)
            table.zero_()
            table[:, :columns].copy_(block_table[:, :columns])
        if width == 4 and envs.SGLANG_ENABLE_ASYNC_ASSERT.get():
            validate_mtp_chain(
                forward_batch,
                forward_batch.positions[:rows].view(batch, width),
                pre["valid_rows"][:, None].expand(batch, width),
                self.rows,
            )
        fused_quant_lightning_indexer_manage(
            weights=weights[:rows].contiguous(),
            query_scale=query_scale[:rows].contiguous(),
            query=query[:rows].contiguous(),
            key_scale=key_scale,
            key=key,
            block_table=table,
            query_ends=self.query_ends[width][:batch],
            seq_lens=pre["lengths"],
            prefix_lens=pre["prefixes"],
            cache_sizes=pre["cache_sizes"],
            request_state=pre["request_states"],
            request_slots=pre["safe_slots"],
            slot_pool=manager.device_slot_map[layer],
            source_ids=self.sources[:rows],
            device_slots=self.slots[:rows],
            topk_misses=self.topk_misses[:rows],
            miss_sources=self.miss_sources[:batch],
            miss_slots=self.miss_slots[:batch],
            miss_counts=self.miss_counts[:batch],
            operator=self.operator,
        )
        self.last_prefix[layer].index_copy_(
            0, pre["safe_slots"].long(), pre["next_prefixes"]
        )
        return LIMSelection(
            self.sources[:rows, 0],
            self.slots[:rows, 0],
            self.topk_misses[:rows],
            self.miss_sources[:batch],
            self.miss_slots[:batch],
            self.miss_counts[:batch],
            pre["prefixes"],
            pre["sparse_rows"],
            pre["valid_rows"],
            pre["safe_slots"],
            pre["lengths"],
        )

    def copy_sfa_metadata(self, manager, plan):
        batch = plan.request_slots.numel()
        width = plan.source_ids.shape[0] // batch
        if not self.fused_copy_metadata:
            valid = plan.valid_rows & plan.sparse_rows
            routes = valid[:, None].expand(batch, width).reshape(-1)
            plan.source_ids.masked_fill_(
                ~plan.valid_rows[:, None].expand(batch, width).reshape(-1, 1), -1
            )
            plan.device_slots.masked_fill_(
                ~plan.valid_rows[:, None].expand(batch, width).reshape(-1, 1), -1
            )
            plan.topk_misses.masked_fill_(~routes, 0)
            plan.miss_counts.masked_fill_(~valid, 0)
            manager.planner_actual_seq_lengths_kv[:batch].copy_(
                torch.where(valid, plan.seq_lens, 0)
            )
            manager.planner_num_cache_tokens[:batch].copy_(
                torch.where(valid, self.capacity + plan.seq_lens - plan.prefix_lens, 0)
            )
            offsets = manager.hbm_page_offsets[None]
            cp = self.capacity // 128
            pages = manager.device_kv_stride // 128
            physical = plan.request_slots[:, None] * pages + torch.where(
                offsets < cp,
                offsets,
                cp + (plan.prefix_lens[:, None] // 128 + offsets - cp) % 2,
            )
            manager.planner_hbm_block_table[:batch].copy_(
                torch.where(valid[:, None], physical, 0)
            )
            manager.fill_host_block_table(
                plan.request_slots, valid, manager.planner_dram_block_table[:batch]
            )
            return
        fused_lim_copy_sfa_metadata(
            source_ids=plan.source_ids,
            device_slots=plan.device_slots,
            topk_misses=plan.topk_misses,
            miss_counts=plan.miss_counts,
            valid_rows=plan.valid_rows,
            sparse_rows=plan.sparse_rows,
            request_slots=plan.request_slots,
            actual_seq_lengths_kv=manager.planner_actual_seq_lengths_kv[:batch],
            num_cache_tokens=manager.planner_num_cache_tokens[:batch],
            hbm_block_table=manager.planner_hbm_block_table[:batch],
            dram_block_table=manager.planner_dram_block_table[:batch],
            capacity=self.capacity,
            host_context_len=manager.max_context_len,
            request_rows=self.rows,
            query_width=width,
            seq_lens=plan.seq_lens,
            host_block_table=self.block_table[:batch],
            packed_c8=True,
            prefix_lens=plan.prefix_lens,
        )
