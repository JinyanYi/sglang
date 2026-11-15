"""A5 packed-C8 policy on the fused manager's physical Host lifecycle."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from sgl_kernel_npu.sparsity_driven_kv_offload import unidex_copy_inplace
from sgl_kernel_npu.sparsity_driven_kv_offload.fused_shm import (
    create_fused_shm_tensor as create_shm_tensor,
)

from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.fused_manager import (
    FusedSparseKVCacheManager,
)
from sglang.srt.runtime_context import get_disagg, get_parallel

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class C8SparseKVCacheManager(FusedSparseKVCacheManager):
    """Retain Q1/Q4 ring residency while sharing native physical token IDs."""

    packed_c8 = True
    copy_sfa_enabled = True
    separate_mtp_tail = True
    tail_tokens = 256
    host_kv_is_paged = True
    indexer_impl = "lim"
    use_lru = False

    def __init__(
        self,
        req_to_token_pool,
        token_to_kv_pool_allocator,
        sparse_context_len,
        *,
        paged_kv_cache=None,
        budget_plan=None,
        compute_dtype=torch.bfloat16,
    ):
        from sglang.srt.environ import envs
        from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.c8 import (
            C8Cache,
        )
        from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.c8_lim_bridge import (
            C8LIMBridge,
        )

        self.req_to_token_pool = req_to_token_pool
        pool = (
            paged_kv_cache
            if paged_kv_cache is not None
            else token_to_kv_pool_allocator.get_kvcache()
        )
        self.paged_kv_cache = pool
        self.budget = budget_plan or getattr(pool, "sparse_kv_budget_plan", None)
        if not getattr(self.budget, "packed_c8", False):
            raise ValueError(
                "A5 C8 requires a separately reserved packed-C8 budget plan."
            )
        self.budget.bind_native_pool(pool)
        if self.budget.mf_shared != envs.SGLANG_NPU_A5_PD_MF.get():
            raise ValueError("A5 MF policy changed after sparse KV reservation.")
        if compute_dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("A5 CopySFA compute must use BF16/FP16.")
        if sparse_context_len != self.budget.top_k:
            raise ValueError("A5 C8 manager TopK differs from its reservation.")
        self.compute_dtype = compute_dtype
        if compute_dtype != self.budget.compute_dtype:
            raise ValueError(
                "A5 compute dtype differs from the persistent output plan."
            )
        self.actual_request_rows = req_to_token_pool.req_to_token.shape[0]
        self.logical_max_context_len = req_to_token_pool.req_to_token.shape[1]
        self.size = self.budget.request_rows
        if self.size < self.actual_request_rows:
            raise ValueError("A5 C8 reservation does not cover the request pool.")
        self.device = req_to_token_pool.device
        self.max_context_len = self.budget.context_len
        self.batch_capacity = self.budget.batch_capacity
        self.max_query_width = self.budget.max_query_width
        self.start_layer, self.layer_num = pool.start_layer, pool.layer_num
        self.head_num, self.head_dim = 1, 656
        self.kv_lora_rank, self.qk_rope_head_dim = 512, 64
        self.store_dtype = torch.int8
        self.sparse_context_len = sparse_context_len
        self.device_cache_size = self.cache_capacity = self.budget.cache_tokens
        self.device_kv_stride = self.cache_capacity + 256
        self.host_token_capacity = self.budget.host_token_capacity
        self.last_decode_path = "unselected"
        for name, spec in self.budget.manager_buffers.items():
            fill = (
                -1 if name in ("device_slot_map", "tail_starts", "tail_ends") else None
            )
            value = (
                [spec.allocate(self.device, fill) for _ in range(spec.count)]
                if name
                in ("device_kv_buffer", "device_slot_map", "tail_starts", "tail_ends")
                else spec.allocate(self.device, fill)
            )
            setattr(self, name, value)
        self.tail_workspace = tuple(
            s.allocate(self.device) for s in self.budget.tail_workspace_specs
        )
        self.tail_workspace[3].copy_(
            torch.arange(self.batch_capacity * 4, device=self.device)
        )
        self.hbm_page_offsets = torch.arange(
            self.device_kv_stride // 128, dtype=torch.int32, device=self.device
        )
        self.dram_page_offsets = torch.arange(
            self.max_context_len // 128, dtype=torch.int32, device=self.device
        )
        self.host_pool = self.mf_host_pool = None
        self.host_kv_buffer, self.host_ptr_list, self.dev_ptr_list = [], [], []
        if envs.SGLANG_NPU_A5_PD_MF.get():
            from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.a5_pd_pool import (
                A5PackedMFPool,
            )

            if get_disagg().disaggregation_mode != "decode":
                raise ValueError("A5 MF packed C8 storage belongs to P/D decode.")
            self.host_pool = self.mf_host_pool = A5PackedMFPool(
                tokens=self.host_token_capacity,
                layers=self.layer_num,
                device=self.device,
                group=get_parallel().attn_tp_group,
            )
        for layer in range(self.layer_num):
            if self.host_pool is not None:
                host = self.host_pool.packed[layer].view(self.budget.host_kv_spec.shape)
                host_ptr = dev_ptr = host.data_ptr()
            else:
                host, host_ptr, dev_ptr = create_shm_tensor(
                    shape=self.budget.host_kv_spec.shape,
                    dtype=torch.int8,
                    device_id=torch.npu.current_device(),
                    name=f"a5_c8_{id(pool)}_{layer}",
                )
            self.host_kv_buffer.append(host)
            self.host_ptr_list.append(host_ptr)
            self.dev_ptr_list.append(dev_ptr)
        self.budget.validate_buffers(self, self.budget.manager_buffers)
        self.lim_adapter = C8LIMBridge(self)
        self.c8_cache = C8Cache(self, self.budget.copy_sfa_query_heads, compute_dtype)
        self._install_req_alloc_hook(req_to_token_pool)
        logger.info(
            "A5 packed C8: C=%d tail=256 Q=%d managed HBM=%.2f MiB Host=%.2f MiB",
            self.cache_capacity,
            self.max_query_width,
            self.budget.managed_hbm_bytes / 1024**2,
            self.budget.host_kv_bytes / 1024**2,
        )

    def init_req(self, req):
        if not req.inflight_middle_chunks:
            self.reset_requests([req.kv.req_pool_idx])

    def reset_requests(self, req_ids):
        if not req_ids:
            return
        ids = torch.tensor(req_ids, device=self.device, dtype=torch.long)
        for starts, ends, slots in zip(
            self.tail_starts, self.tail_ends, self.device_slot_map
        ):
            starts.index_fill_(0, ids, -1)
            ends.index_fill_(0, ids, -1)
            slots.index_fill_(0, ids, -1)
        self.lim_adapter.reset_requests(ids)

    def move_host_kv(self, target, source):
        from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.a5_pd import (
            move_packed_kv,
        )

        return move_packed_kv(self, target, source)

    def fill_host_block_table(self, request_ids, valid_rows, output):
        positions = self.dram_page_offsets[None, :] * 128
        tokens = self.host_source_indices(request_ids[:, None], positions)
        valid = valid_rows[:, None] & (positions < self.logical_max_context_len)
        output.copy_(torch.where(valid, tokens // 128, 0).to(torch.int32))

    def offload_v2(
        self,
        k: torch.Tensor,
        k_rope: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        stream: torch.npu.Stream,
        *,
        defer_tail: bool = False,
    ):
        """Publish KV to paged host storage and return decode rows for CopySFA."""
        layer_idx = layer.layer_id - self.start_layer
        # k:         [num_new_tokens, nhead, kv_lora_rank]
        # k_rope:    [num_new_tokens, nhead, qk_rope_head_dim]
        # kv_device: [num_new_tokens, nhead, kv_lora_rank + qk_rope_head_dim]
        src_tensor = (
            self.paged_kv_cache._pack_dsa_fp8_kv_cache(k, k_rope)
            .view(torch.int8)
            .view(-1, 1, 656)
        )
        dst_tensor = self.host_kv_buffer[layer_idx]
        from sglang.kernels.ops.kvcache import prepare_sparse_kv_publication_rows
        from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.mtp import (
            sparse_query_width,
        )

        is_prefill = forward_batch.forward_mode.is_extend_without_speculative()
        compact_prefill = is_prefill and (
            sum(forward_batch.extend_seq_lens_cpu[: forward_batch.batch_size])
            == src_tensor.shape[0]
        )
        src_index, dst_index, token_positions, valid_mask = (
            prepare_sparse_kv_publication_rows(
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                None if is_prefill else forward_batch.positions,
                forward_batch.out_cache_loc,
                tokens=src_tensor.shape[0],
                request_rows=self.size,
                context_len=self.logical_max_context_len,
                host_capacity=self.host_token_capacity,
                query_width=1 if is_prefill else sparse_query_width(forward_batch),
                extend_seq_lens=forward_batch.extend_seq_lens if is_prefill else None,
                extend_prefix_lens=forward_batch.extend_prefix_lens
                if is_prefill
                else None,
                compact_prefill=compact_prefill,
            )
        )
        if dst_tensor is not None:
            assert src_tensor.shape[1:] == dst_tensor.shape[2:]
        assert src_index.shape == dst_index.shape == valid_mask.shape

        actual_stream = stream if stream is not None else torch.npu.current_stream()
        with torch.npu.stream(actual_stream):
            if self.host_pool is not None:
                if self.host_pool.rank == 0:
                    self.host_pool.copy_rows(
                        (src_tensor,),
                        (dst_tensor,),
                        src_index,
                        dst_index,
                        valid_mask,
                    )
                self.host_pool.publish()
            else:
                unidex_copy_inplace(
                    src_tensor,
                    dst_tensor,
                    src_index,
                    dst_index,
                    valid_mask,
                    1,  # src_tensor: [num_new_tokens, nhead, head_dim]
                    2,  # dst_tensor: [num_req, max_context_len, nhead, head_dim]
                    block_dim=48,
                    dst_ptr=self.dev_ptr_list[layer_idx],
                )

            if forward_batch.forward_mode.is_extend_without_speculative():
                self.tail_starts[layer_idx].index_fill_(
                    0, forward_batch.req_pool_indices.long(), -1
                )
                self.tail_ends[layer_idx].index_fill_(
                    0, forward_batch.req_pool_indices.long(), -1
                )
            elif not defer_tail and (
                forward_batch.forward_mode.is_target_verify()
                or forward_batch.forward_mode.is_decode()
            ):
                from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.npu_mtp_tail import (
                    stage_mtp_tail,
                )

                stage_mtp_tail(
                    current=src_tensor.view(-1, 656),
                    cache=self.device_kv_buffer[layer_idx],
                    host=dst_tensor,
                    host_address=self.dev_ptr_list[layer_idx],
                    req_to_token=self.req_to_token_pool.req_to_token,
                    requests=forward_batch.req_pool_indices,
                    positions=forward_batch.positions,
                    current_valid=valid_mask,
                    starts=self.tail_starts[layer_idx],
                    ends=self.tail_ends[layer_idx],
                    workspace=self.tail_workspace,
                    capacity=self.cache_capacity,
                    context_len=self.logical_max_context_len,
                    query_width=4
                    if forward_batch.forward_mode.is_target_verify()
                    else 1,
                )
            return src_tensor, token_positions, valid_mask
