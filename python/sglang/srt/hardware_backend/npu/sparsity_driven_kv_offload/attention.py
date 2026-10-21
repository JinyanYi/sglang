"""Ascend attention path backed by sparsity-driven KV offload."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch
import torch_npu

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.manager import (
    _wait_stream_event,
    normalize_batch_topk_indices,
)
from sglang.srt.layers.attention.dsa.utils import is_dsa_enable_prefill_cp

if TYPE_CHECKING:
    from sglang.srt.hardware_backend.npu.attention.ascend_backend import (
        AscendAttnBackend,
    )
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


def _get_sparse_kv_manager(backend: AscendAttnBackend):
    if backend.sparse_kv_manager is None:
        raise RuntimeError(
            "Sparsity-driven KV offload is disabled or was not initialized."
        )
    return backend.sparse_kv_manager


def _expand_dsa_sparse_indices(topk_indices: torch.Tensor) -> torch.Tensor:
    """Expand [T, K] to [T, 1, K] for NPU sparse attention."""
    if topk_indices.dim() == 2:
        return topk_indices.unsqueeze(-2)
    return topk_indices


def _copy_sfa_decode_eligible(backend, forward_batch, layer, q, topk_indices):
    manager = backend.sparse_kv_manager
    if not envs.SGLANG_NPU_ENABLE_COPY_SFA.get() or manager is None:
        return False
    if (
        not forward_batch.forward_mode.is_decode()
        or getattr(forward_batch, "spec_info", None) is not None
    ):
        return False
    if not manager.prefix_tail_enabled and not getattr(
        forward_batch, "sparse_kv_copy_sfa_capture", False
    ):
        seq_lens_cpu = backend.forward_metadata.seq_lens_cpu_int
        if seq_lens_cpu is None:
            return False
        live = [length for length in seq_lens_cpu if int(length) > 0]
        if not live or min(map(int, live)) < 2048:
            return False
    return (
        layer.tp_k_head_num == 1
        and backend.kv_lora_rank == 512
        and backend.qk_rope_head_dim == 64
        and q.dtype in (torch.bfloat16, torch.float16)
        and normalize_batch_topk_indices(topk_indices).shape[1] == 2048
        and manager.sparse_context_len == 2048
        and manager.cache_capacity >= manager.sparse_context_len
        and q.shape[0] == forward_batch.batch_size
    )


def forward_sparsity_driven_kv_offload(
    backend: AscendAttnBackend,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layer: RadixAttention,
    forward_batch: ForwardBatch,
    save_kv_cache: bool = True,
    q_rope: Optional[torch.Tensor] = None,
    k_rope: Optional[torch.Tensor] = None,
    topk_indices: Optional[torch.Tensor] = None,
):
    """Run sparse attention using host-offloaded compact MLA KV."""
    del v
    if q_rope is None or k_rope is None or topk_indices is None:
        raise ValueError(
            "Sparsity-driven KV offload requires q_rope, k_rope, and topk_indices."
        )

    is_prefill = forward_batch.forward_mode.is_extend_without_speculative()

    q_nope, q_pe = q, q_rope
    k_nope = k.view(-1, layer.tp_k_head_num, backend.kv_lora_rank).contiguous()
    k_pe = k_rope.view(-1, layer.tp_k_head_num, backend.qk_rope_head_dim).contiguous()
    sparse_kv_manager = _get_sparse_kv_manager(backend)
    stream = torch.npu.current_stream(backend.device)

    if save_kv_cache:
        sparse_kv_manager.offload_v2(k_nope, k_pe, layer, forward_batch, stream)

    if is_prefill:
        if backend.forward_metadata.actual_seq_lengths_q is not None:
            actual_seq_qlen = backend.forward_metadata.actual_seq_lengths_q
        else:
            actual_seq_qlen = torch.cumsum(forward_batch.extend_seq_lens, dim=0)
    elif backend.forward_metadata.actual_seq_lengths_q is None:
        if (
            forward_batch.forward_mode.is_draft_extend_v2()
            or forward_batch.forward_mode.is_target_verify()
        ):
            actual_seq_qlen = (
                torch.arange(
                    backend.speculative_num_draft_tokens,
                    backend.speculative_num_draft_tokens + q.shape[0],
                    backend.speculative_num_draft_tokens,
                    dtype=torch.int32,
                )
                .to(q.device)
                .to(torch.int32)
            )
        else:
            actual_seq_qlen = (
                torch.arange(1, q.shape[0] + 1).to(q.device).to(torch.int32)
            )
    else:
        actual_seq_qlen = backend.forward_metadata.actual_seq_lengths_q

    if backend.forward_metadata.actual_seq_lengths_kv is not None:
        actual_seq_lengths_kv = backend.forward_metadata.actual_seq_lengths_kv
    elif backend.forward_metadata.seq_lens_cpu_int is not None:
        actual_seq_lengths_kv = backend.forward_metadata.seq_lens_cpu_int
    else:
        actual_seq_lengths_kv = backend.forward_metadata.seq_lens

    if (
        is_prefill
        and is_dsa_enable_prefill_cp()
        and forward_batch.attn_cp_metadata is not None
    ):
        attn_out = backend.do_cp_balance_attn(
            q_nope,
            k_nope,
            q_pe,
            k_pe,
            topk_indices,
            layer,
            actual_seq_qlen,
            actual_seq_lengths_kv,
        )
    elif forward_batch.forward_mode.is_decode() and _copy_sfa_decode_eligible(
        backend, forward_batch, layer, q_nope, topk_indices
    ):
        if sparse_kv_manager.indexer_impl == "lim":
            lim_plan = backend.forward_metadata.sparse_kv_lim_plan
            if lim_plan is None:
                raise RuntimeError(
                    "LIM Copy-SFA decode reached attention without an indexer plan."
                )
            # IndexShare consumers use the anchor's selected token IDs, while
            # copy_sfa_decode still materializes their own layer's KV payload.
            topk_indices = lim_plan.source_ids
        sparse_kv_manager.last_decode_path = "copy_sfa"
        forward_batch.sparse_kv_decode_path = "copy_sfa"
        batch_size = forward_batch.batch_size
        padded_query_heads = q_nope.numel() // (batch_size * backend.kv_lora_rank)
        lim_plan = (
            backend.forward_metadata.sparse_kv_lim_plan
            if sparse_kv_manager.indexer_impl == "lim"
            else None
        )
        attn_out = sparse_kv_manager.copy_sfa_decode(
            layer,
            forward_batch,
            q_nope.view(batch_size, 1, padded_query_heads, backend.kv_lora_rank),
            q_pe.view(batch_size, 1, padded_query_heads, backend.qk_rope_head_dim),
            layer.tp_q_head_num,
            layer.scaling,
            topk_indices,
            lim_plan=lim_plan,
        )
    elif forward_batch.forward_mode.is_decode():
        sparse_kv_manager.last_decode_path = (
            "prefix_tail_reference"
            if sparse_kv_manager.prefix_tail_enabled
            else "legacy_reference"
        )
        forward_batch.sparse_kv_decode_path = sparse_kv_manager.last_decode_path
        batch_size = forward_batch.batch_size
        num_kv_heads = layer.tp_k_head_num
        num_query_heads = layer.tp_q_head_num
        nope_head_dim = backend.kv_lora_rank
        rope_head_dim = backend.qk_rope_head_dim

        topk_2d_input = normalize_batch_topk_indices(topk_indices)
        effective_topk_length = topk_2d_input.shape[1]
        selected_kv_length = sparse_kv_manager.selected_kv_width
        if effective_topk_length <= 0:
            raise RuntimeError("SFA BSND compact path expects a positive top-k length.")
        if effective_topk_length > selected_kv_length:
            raise RuntimeError(
                "DSA top-k length exceeds sparse KV device cache capacity: "
                f"topk_len={effective_topk_length}, "
                f"sparse_context_len={selected_kv_length}."
            )
        if effective_topk_length == selected_kv_length:
            topk_2d = topk_2d_input.contiguous()
        else:
            topk_2d = torch.full(
                (batch_size, selected_kv_length),
                -1,
                dtype=topk_2d_input.dtype,
                device=topk_2d_input.device,
            )
            topk_2d[:, :effective_topk_length] = topk_2d_input
            topk_2d = topk_2d.contiguous()

        assert num_kv_heads == 1, (
            "FIA_v2 MLA selected KV path expects KV_N == 1, "
            f"got num_kv_heads={num_kv_heads}"
        )

        padded_query_heads = q_nope.numel() // (batch_size * nope_head_dim)
        assert padded_query_heads >= num_query_heads, (
            "query head count mismatch: "
            f"padded_query_heads={padded_query_heads}, "
            f"num_query_heads={num_query_heads}"
        )

        # Materialize the compact top-k KV for the current attention step:
        # device-cache hits and host misses are copied into this buffer, the
        # device cache is refilled from it, and sparse attention consumes it
        # directly.
        if sparse_kv_manager.prefix_tail_enabled:
            selected_kv_buffer = sparse_kv_manager.selected_kv_buffer[
                layer.layer_id - sparse_kv_manager.start_layer
            ][:batch_size]
            selected_kv_buffer.zero_()
        else:
            selected_kv_buffer = torch.zeros(
                (
                    batch_size,
                    selected_kv_length,
                    num_kv_heads,
                    nope_head_dim + rope_head_dim,
                ),
                dtype=k.dtype,
                device=backend.device,
            )
        sparse_kv_manager.materialize_selected_kv(
            layer, forward_batch, topk_indices, selected_kv_buffer, stream
        )

        _wait_stream_event(stream, sparse_kv_manager.hit_done)
        _wait_stream_event(stream, sparse_kv_manager.miss_done)
        prefix_tail_metadata = None
        if sparse_kv_manager.prefix_tail_enabled:
            prefix_tail_metadata = sparse_kv_manager.make_prefix_tail_metadata(
                forward_batch.seq_lens[:batch_size]
            )
            sparse_kv_manager.materialize_dense_prefix(
                layer,
                forward_batch.req_pool_indices[:batch_size],
                forward_batch.seq_lens[:batch_size],
                prefix_tail_metadata,
                selected_kv_buffer,
                stream,
            )
            sparse_kv_manager.append_prefix_tail(
                layer,
                forward_batch.req_pool_indices[:batch_size],
                prefix_tail_metadata,
                selected_kv_buffer,
                stream,
            )

        selected_k_nope, selected_k_rope = selected_kv_buffer.split(
            [nope_head_dim, rope_head_dim], dim=-1
        )

        topk_valid = (topk_2d >= 0) & (
            topk_2d < sparse_kv_manager.logical_max_context_len
        )
        if forward_batch.seq_lens is not None:
            valid_rows = (forward_batch.seq_lens[:batch_size] > 0).view(batch_size, 1)
            topk_valid = topk_valid & valid_rows
        if prefix_tail_metadata is not None:
            topk_valid = topk_valid & (
                topk_2d < prefix_tail_metadata.indexer_seq_lens[:, None]
            )

        selected_topk_counts = topk_valid.sum(dim=1).to(
            device=q_nope.device, dtype=torch.int32
        )
        if prefix_tail_metadata is None:
            actual_seq_lengths_kv = selected_topk_counts.clamp(
                min=1, max=selected_kv_length
            ).contiguous()
        else:
            actual_seq_lengths_kv = (
                torch.where(
                    prefix_tail_metadata.sparse_rows,
                    selected_topk_counts + prefix_tail_metadata.tail_lens,
                    forward_batch.seq_lens[:batch_size].to(torch.int32),
                )
                .clamp(min=1, max=selected_kv_length)
                .contiguous()
            )
        actual_seq_lengths_query = torch.ones(
            batch_size, dtype=torch.int32, device=q_nope.device
        ).contiguous()

        compact_indices = (
            torch.arange(selected_kv_length, device=q_nope.device, dtype=torch.int32)
            .view(1, 1, 1, selected_kv_length)
            .expand(batch_size, 1, num_kv_heads, selected_kv_length)
            .clone()
        )
        compact_valid = topk_valid.view(batch_size, 1, 1, selected_kv_length).expand(
            batch_size, 1, num_kv_heads, selected_kv_length
        )
        sparse_indices = torch.where(
            compact_valid,
            compact_indices,
            torch.full_like(compact_indices, -1),
        ).contiguous()

        if prefix_tail_metadata is not None:
            tail_offsets = torch.arange(
                selected_kv_length - sparse_kv_manager.sparse_context_len,
                device=q_nope.device,
                dtype=torch.int32,
            )
            tail_valid = prefix_tail_metadata.sparse_rows[:, None] & (
                tail_offsets[None, :] < prefix_tail_metadata.tail_lens[:, None]
            )
            tail_slots = (
                (sparse_kv_manager.sparse_context_len + tail_offsets)
                .view(1, 1, 1, -1)
                .expand(batch_size, 1, num_kv_heads, -1)
            )
            sparse_indices[:, :, :, sparse_kv_manager.sparse_context_len :] = (
                torch.where(
                    tail_valid[:, None, None, :].expand_as(tail_slots),
                    tail_slots,
                    torch.full_like(tail_slots, -1),
                )
            )
            dense_offsets = torch.arange(
                selected_kv_length, device=q_nope.device, dtype=torch.int32
            )
            dense_slots = dense_offsets.view(1, 1, 1, -1).expand_as(sparse_indices)
            dense_valid = (
                dense_offsets[None, :] < forward_batch.seq_lens[:batch_size, None]
            )
            dense_indices = torch.where(
                dense_valid[:, None, None, :].expand_as(sparse_indices),
                dense_slots,
                torch.full_like(sparse_indices, -1),
            )
            sparse_indices = torch.where(
                (~prefix_tail_metadata.sparse_rows)[:, None, None, None],
                dense_indices,
                sparse_indices,
            )

        empty_rows = (topk_valid.sum(dim=1) == 0).view(batch_size, 1, 1)
        sparse_indices[:, :, :, 0] = torch.where(
            empty_rows.expand(batch_size, 1, num_kv_heads),
            torch.zeros(
                (batch_size, 1, num_kv_heads),
                dtype=torch.int32,
                device=q_nope.device,
            ),
            sparse_indices[:, :, :, 0],
        )

        q_nope_sfa = q_nope.view(
            batch_size, 1, padded_query_heads, nope_head_dim
        ).contiguous()
        q_rope_sfa = q_pe.view(
            batch_size, 1, padded_query_heads, rope_head_dim
        ).contiguous()
        k_nope_sfa = selected_k_nope.contiguous()
        k_rope_sfa = selected_k_rope.contiguous()

        assert q_nope_sfa.shape == (
            batch_size,
            1,
            padded_query_heads,
            nope_head_dim,
        )
        assert q_rope_sfa.shape == (
            batch_size,
            1,
            padded_query_heads,
            rope_head_dim,
        )
        assert k_nope_sfa.shape == (
            batch_size,
            selected_kv_length,
            num_kv_heads,
            nope_head_dim,
        )
        assert k_rope_sfa.shape == (
            batch_size,
            selected_kv_length,
            num_kv_heads,
            rope_head_dim,
        )

        ret = torch_npu.npu_sparse_flash_attention(
            q_nope_sfa,
            k_nope_sfa,
            k_nope_sfa,
            sparse_indices,
            layer.scaling,
            actual_seq_lengths_query=actual_seq_lengths_query,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            query_rope=q_rope_sfa,
            key_rope=k_rope_sfa,
            sparse_block_size=1,
            layout_query="BSND",
            layout_kv="BSND",
            sparse_mode=0,
            attention_mode=2,
            return_softmax_lse=False,
        )

        _wait_stream_event(stream, sparse_kv_manager.refill_done)
        _wait_stream_event(stream, sparse_kv_manager.slot_map_done)

        attn_out = ret[0] if isinstance(ret, tuple) else ret
        attn_out = attn_out[:, :, :num_query_heads, :].reshape(
            batch_size, num_query_heads * nope_head_dim
        )
    else:
        if is_prefill:
            k_nope_sfa, k_pe_sfa = sparse_kv_manager.get_forward_kv(
                layer, forward_batch, stream
            )
            forward_actual_seq_lengths_kv = torch.cumsum(forward_batch.seq_lens, dim=0)
        else:
            k_nope_sfa, k_pe_sfa = k_nope, k_pe
            forward_actual_seq_lengths_kv = actual_seq_lengths_kv

        topk_indices = _expand_dsa_sparse_indices(topk_indices)
        attn_out, _, _ = torch_npu.npu_sparse_flash_attention(
            query=q_nope,
            key=k_nope_sfa,
            value=k_nope_sfa,
            query_rope=q_pe,
            key_rope=k_pe_sfa,
            sparse_indices=topk_indices,
            scale_value=layer.scaling,
            actual_seq_lengths_query=actual_seq_qlen.to(
                device=q_nope.device, dtype=torch.int32
            ),
            actual_seq_lengths_kv=forward_actual_seq_lengths_kv.to(
                device=q_nope.device, dtype=torch.int32
            ),
            sparse_block_size=1,
            layout_query="TND",
            layout_kv="TND",
            sparse_mode=3,
            attention_mode=2,
            return_softmax_lse=False,
        )

    return attn_out
