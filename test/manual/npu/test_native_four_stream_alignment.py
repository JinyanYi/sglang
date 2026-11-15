"""Align native LIM/CopySFA with the original four-stream SGLang path."""

import json
import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

pytest.importorskip("torch_npu")

from sgl_kernel_npu.sparsity_driven_kv_offload import free_shm
from sgl_kernel_npu.sparsity_driven_kv_offload.fused_shm import free_fused_shm

from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload import (
    attention as four_stream_attention,
)
from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload import (
    manager as four_stream_manager,
)
from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.fused_attention import (
    forward_fused_sparse_kv_offload,
)
from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.fused_budget import (
    resolve_sparse_kv_budget,
)
from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.fused_manager import (
    FusedSparseKVCacheManager,
)
from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool, ReqToTokenPool
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.runtime_context import get_context, get_parallel

K = 2048
C = 8192
CONTEXT = 8704
HEADS = 8


class _OneLayerIndexPool(MLATokenToKVPool):
    def __init__(self, key):
        self.key = key
        self.kv_lora_rank, self.qk_rope_head_dim = 512, 64
        self.store_dtype = torch.bfloat16
        self.start_layer, self.layer_num, self.page_size = 0, 1, 128
        self.indexer_layer_ids = (0,)
        self.size = key.shape[0] * self.page_size - self.page_size

    def get_index_k_buffer(self, layer_id):
        assert layer_id == 0
        return self.key


def _model_config():
    return SimpleNamespace(
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        index_head_dim=128,
        hf_config=SimpleNamespace(
            architectures=["GlmMoeDsaForCausalLM"],
            index_head_dim=128,
            index_n_heads=32,
            index_topk=K,
        ),
    )


def _oracle(q, q_rope, selected, scale):
    selected = selected.float()
    scores = (q.float() @ selected[:, :512].T) + (q_rope.float() @ selected[:, 512:].T)
    return torch.softmax(scores * scale, dim=-1) @ selected[:, :512]


def _metrics(actual, expected):
    error = actual.float() - expected.float()
    return {
        "max_abs": error.abs().max().item(),
        "rms": error.square().mean().sqrt().item(),
        "relative_l2": (error.norm() / expected.float().norm().clamp_min(1e-12)).item(),
        "values": error.numel(),
    }


@pytest.mark.parametrize("selection_mode", ("shared", "independent"))
def test_native_four_stream_alignment_h8_q1_cold_and_warm(selection_mode):
    """Compare both attention chains with shared or independently selected KV."""
    assert torch.npu.is_available()
    torch.npu.set_device(0)
    device_id = torch.npu.current_device()
    device = torch.device("npu", device_id)
    generator = torch.Generator().manual_seed(20261115)

    pages = CONTEXT // 128
    physical_pages = pages + 1  # Page zero is reserved for graph padding.
    physical = (
        torch.arange(1, physical_pages, dtype=torch.long)[:, None] * 128
        + torch.arange(128, dtype=torch.long)[None, :]
    ).reshape(-1)
    fixed_topk = torch.cat(
        (torch.arange(K - 1, dtype=torch.long), torch.tensor([CONTEXT - 1]))
    )
    canonical = torch.randn(CONTEXT, 576, generator=generator).to(torch.bfloat16)

    key = torch.zeros(physical_pages, 128, 1, 128, dtype=torch.bfloat16, device=device)
    flat_key = key.view(-1, 1, 128)
    flat_key[:, 0, 0] = -8
    flat_key[physical[fixed_topk].to(device), 0, 0] = 4
    flat_key[int(physical[-1]), 0, 0] = 16
    index_pool = _OneLayerIndexPool(key)

    four_req = ReqToTokenPool(1, CONTEXT, "npu", False)
    fused_req = ReqToTokenPool(1, CONTEXT, "npu", False)
    fused_req.req_to_token[1, :CONTEXT].copy_(physical.to(device, torch.int32))
    budget = resolve_sparse_kv_budget(
        model_config=_model_config(),
        page_size=128,
        request_rows=2,
        graph_batch_capacity=2,
        context_len=CONTEXT,
        num_layers=1,
        top_k=K,
        kv_bytes_per_token=1152,
        cache_tokens=C,
        copy_sfa_query_heads=HEADS,
    )
    budget.bind_native_pool(index_pool)
    index_pool.sparse_kv_budget_plan = budget

    inputs = SimpleNamespace(
        batch_size=2,
        forward_mode=ForwardMode.DECODE,
        req_pool_indices=torch.tensor([1, 0], device=device, dtype=torch.int32),
        seq_lens=torch.tensor([CONTEXT, 0], device=device, dtype=torch.int32),
        seq_lens_cpu=torch.tensor([CONTEXT, 0], dtype=torch.int32),
        positions=torch.tensor([CONTEXT - 1, 0], device=device, dtype=torch.int64),
        out_cache_loc=torch.tensor(
            [physical[-1].item(), -1], device=device, dtype=torch.int64
        ),
        spec_info=None,
    )
    block_table = torch.zeros(2, pages, device=device, dtype=torch.int32)
    block_table[0].copy_(torch.arange(1, physical_pages, device=device))
    index_query = torch.zeros(2, 32, 128, dtype=torch.bfloat16, device=device)
    index_query[0, :, 0] = 1
    index_weights = torch.zeros(2, 32, dtype=torch.bfloat16, device=device)
    index_weights[0].fill_(1 / 32)
    q = torch.randn(2, HEADS, 512, generator=generator).to(device, torch.bfloat16)
    q_rope = torch.randn(2, HEADS, 64, generator=generator).to(device, torch.bfloat16)
    q[1].zero_()
    q_rope[1].zero_()
    current = torch.zeros(2, 1, 576, dtype=torch.bfloat16, device=device)
    current[0].copy_(canonical[-1])
    layer = SimpleNamespace(
        layer_id=0, tp_k_head_num=1, tp_q_head_num=HEADS, scaling=576**-0.5
    )
    four = fused = None
    captured = []

    original_create = four_stream_manager.create_shm_tensor

    def unique_create(shape, dtype, device_id=0, name=""):
        unique_name = f"{name}_native_alignment_{os.getpid()}_h8_q1_{selection_mode}"
        return original_create(shape, dtype, device_id=device_id, name=unique_name)

    try:
        with (
            get_context().override_server_args(disaggregation_mode="null"),
            get_parallel().override(attn_cp_size=1),
            patch.object(four_stream_manager, "create_shm_tensor", new=unique_create),
        ):
            four = four_stream_manager.SparseKVCacheManager(
                four_req, SimpleNamespace(get_kvcache=lambda: index_pool), K
            )
            fused = FusedSparseKVCacheManager(
                fused_req,
                SimpleNamespace(get_kvcache=lambda: index_pool),
                K,
                paged_kv_cache=index_pool,
            )

        four.host_kv_buffer[0][1, :, 0].copy_(canonical)
        fused.host_k_buffer[0].view(-1, 512).index_copy_(
            0, physical, canonical[:, :512]
        )
        fused.host_v_buffer[0].view(-1, 64).index_copy_(0, physical, canonical[:, 512:])
        assert torch.equal(four.host_kv_buffer[0][1, :, 0], canonical)
        assert torch.equal(
            fused.host_k_buffer[0].view(-1, 512)[physical], canonical[:, :512]
        )
        assert four.dev_ptr_list[0] != fused.dev_ptr_list[0]
        assert (
            four.device_kv_buffer[0].untyped_storage().data_ptr()
            != fused.device_kv_buffer[0].untyped_storage().data_ptr()
        )

        four_backend = SimpleNamespace(
            device=device,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            speculative_num_draft_tokens=1,
            sparse_kv_manager=four,
            forward_metadata=SimpleNamespace(
                actual_seq_lengths_q=None,
                actual_seq_lengths_kv=None,
                seq_lens_cpu_int=None,
                seq_lens=inputs.seq_lens,
            ),
        )
        fused_backend = SimpleNamespace(
            graph_mode=False,
            device=device,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            sparse_kv_manager=fused,
            forward_metadata=SimpleNamespace(sparse_kv_lim_plan=None),
        )
        real_sfa = four_stream_attention.torch_npu.npu_sparse_flash_attention

        def capture_selected(*args, **kwargs):
            captured.append((args[1].clone(), kwargs["key_rope"].clone()))
            return real_sfa(*args, **kwargs)

        cold_misses = None
        for phase in ("cold", "warm"):
            plan = fused.lim_adapter.select(
                fused,
                query=index_query,
                weights=index_weights,
                key=key,
                block_table=block_table,
                forward_batch=inputs,
                layer_id=0,
            )
            torch.npu.synchronize()
            ids = plan.source_ids[0].cpu().long()
            assert torch.equal(torch.sort(ids).values, fixed_topk)
            if selection_mode == "independent":
                native_topk = four_stream_attention.torch_npu.npu_lightning_indexer(
                    query=index_query,
                    key=key,
                    weights=index_weights,
                    actual_seq_lengths_query=torch.tensor(
                        [1, 2], dtype=torch.int32, device=device
                    ),
                    actual_seq_lengths_key=inputs.seq_lens,
                    block_table=block_table,
                    layout_query="TND",
                    layout_key="PA_BSND",
                    sparse_count=K,
                    sparse_mode=3,
                )[0].squeeze(1)
            else:
                native_topk = plan.source_ids
            # Native SFA represents padding with -1 token IDs. LIM keeps a
            # separate row-valid mask, so adapt only the invalid rows here.
            native_topk = native_topk.masked_fill(
                (inputs.seq_lens <= 0).view(-1, 1), -1
            )
            native_ids = native_topk[0].cpu().long()
            assert torch.equal(torch.sort(native_ids).values, fixed_topk)

            with patch.object(
                four_stream_attention.torch_npu,
                "npu_sparse_flash_attention",
                new=capture_selected,
            ):
                four_out = four_stream_attention.forward_sparsity_driven_kv_offload(
                    four_backend,
                    q,
                    current[..., :512],
                    None,
                    layer,
                    inputs,
                    q_rope=q_rope,
                    k_rope=current[..., 512:],
                    topk_indices=native_topk,
                )
            fused_backend.forward_metadata.sparse_kv_lim_plan = plan
            fused_out = forward_fused_sparse_kv_offload(
                fused_backend,
                q,
                current[..., :512],
                None,
                layer,
                inputs,
                q_rope=q_rope,
                k_rope=current[..., 512:],
                topk_indices=plan.source_ids,
            )
            torch.npu.synchronize()

            expected_selected = canonical[ids]
            actual_four = torch.cat(captured[-1], dim=-1)[0, :, 0].cpu()
            slots = plan.device_slots[0].long()
            actual_fused = torch.cat(
                (
                    fused.device_kv_buffer[0][1, slots, 0],
                    fused.device_rope_buffer[0][1, slots, 0],
                ),
                dim=-1,
            ).cpu()
            torch.testing.assert_close(
                actual_four, canonical[native_ids], rtol=0, atol=0
            )
            torch.testing.assert_close(actual_fused, expected_selected, rtol=0, atol=0)
            assert torch.cat(captured[-1], dim=-1)[1].count_nonzero().item() == 0

            expected = _oracle(
                q[0].cpu(), q_rope[0].cpu(), expected_selected, layer.scaling
            )
            actual_four_out = four_out.view(2, HEADS, 512)[0].cpu().float()
            actual_fused_out = fused_out.view(2, HEADS, 512)[0].cpu().float()
            # Only live rows are model outputs. Native SFA can emit tiny finite
            # residuals for its dummy query; record them separately. CopySFA
            # promises an exact masked zero for that row.
            native_padding = four_out.view(2, HEADS, 512)[1]
            assert native_padding.isfinite().all().item()
            assert fused_out.view(2, HEADS, 512)[1].count_nonzero().item() == 0

            misses = int(plan.miss_counts[0].cpu())
            if phase == "cold":
                assert misses == C
                cold_misses = misses
            else:
                assert misses < cold_misses
            print(
                "NATIVE_FOUR_STREAM_ALIGNMENT "
                + json.dumps(
                    {
                        "phase": phase,
                        "selection_mode": selection_mode,
                        "heads": HEADS,
                        "query_width": 1,
                        "topk": K,
                        "capacity": C,
                        "context": CONTEXT,
                        "fused_misses": misses,
                        "native_padding_max_abs": native_padding.abs().max().item(),
                        "four_vs_oracle": _metrics(actual_four_out, expected),
                        "fused_vs_oracle": _metrics(actual_fused_out, expected),
                        "fused_vs_four": _metrics(actual_fused_out, actual_four_out),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            torch.testing.assert_close(actual_four_out, expected, rtol=0.01, atol=0.002)
            torch.testing.assert_close(
                actual_fused_out, expected, rtol=0.01, atol=0.002
            )
            torch.testing.assert_close(
                actual_fused_out, actual_four_out, rtol=0.01, atol=0.002
            )
    finally:
        if four is not None or fused is not None:
            torch.npu.synchronize()
        free_shm(device_id)
        free_fused_shm(device_id)
