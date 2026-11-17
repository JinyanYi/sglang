"""Persistent allocation plan for A5 packed C8 with a separate ring tail."""

from __future__ import annotations

import torch

from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.fused_budget import (
    BufferSpec,
)


class C8BudgetPlan:
    """Use the same declared-buffer accounting as the BF16 fused backend."""

    packed_c8 = True
    page_size = 128
    tail_tokens = 256
    prefill_workspace_tokens = 0
    prefill_workspace_bytes = 0
    prefill_workspace_sizing_finalized = True
    scale_dtype = torch.float32
    scale_element_size = 4

    def __init__(
        self,
        *,
        request_rows,
        context_len,
        num_layers,
        top_k,
        copy_sfa_query_heads,
        max_query_width,
        batch_capacity,
        cache_tokens,
        compute_dtype=torch.bfloat16,
        mf_shared=False,
    ):
        if request_rows <= 1 or num_layers <= 0 or not 0 < context_len <= 262144:
            raise ValueError(
                "A5 C8 requires a live request row, positive layers and context <=262144."
            )
        if top_k != 2048 or max_query_width not in (1, 4):
            raise ValueError("A5 ring-tail integration supports K2048 and Q1/Q4.")
        if not 2048 <= cache_tokens <= 32384 or cache_tokens % 128:
            raise ValueError("A5 C8 requires page-aligned C in [2048,32384].")
        if max_query_width == 4 and cache_tokens < 8192:
            raise ValueError("A5 Q4 LIM requires C >=4*2048.")
        if copy_sfa_query_heads not in (8, 16, 32, 64, 128):
            raise ValueError("A5 C8 query heads must be 8/16/32/64/128.")
        if compute_dtype not in (torch.bfloat16, torch.float16):
            raise ValueError("A5 CopySFA compute must use BF16/FP16.")
        self.request_rows = request_rows
        # Request IDs retain the framework's null row, but payload storage does not.
        self.kv_request_rows = request_rows - 1
        self.context_len = (context_len + 127) // 128 * 128
        self.num_layers = num_layers
        self.top_k = top_k
        self.max_query_width = max_query_width
        self.batch_capacity = max(request_rows, batch_capacity)
        self.cache_tokens = cache_tokens
        self.copy_sfa_query_heads = copy_sfa_query_heads
        self.kv_dtype = torch.int8
        self.kv_cache_dim = 656
        self.compute_dtype = compute_dtype
        self.mf_shared = mf_shared
        self.mf_fence_bytes = torch.int32.itemsize if mf_shared else 0
        b, r, c = self.batch_capacity, request_rows, cache_tokens
        # Q4 buffers also serve the eager fallback and remain stable across
        # Q1/Q4 captures. Only one attention layer uses the shared workspaces.
        self.manager_buffers = {
            "device_kv_buffer": BufferSpec(
                (self.kv_request_rows, c + 256, 1, 656), torch.int8, num_layers
            ),
            "device_slot_map": BufferSpec(
                (r + 1, self.context_len), torch.int32, num_layers
            ),
            "tail_starts": BufferSpec((r + 1,), torch.int32, num_layers),
            "tail_ends": BufferSpec((r + 1,), torch.int32, num_layers),
            "planner_actual_seq_lengths_kv": BufferSpec((b,), torch.int32),
            "planner_num_cache_tokens": BufferSpec((b,), torch.int32),
            "planner_hbm_block_table": BufferSpec((b, (c + 256) // 128), torch.int32),
            "planner_dram_block_table": BufferSpec(
                (b, self.context_len // 128), torch.int32
            ),
            "mtp_attn_out": BufferSpec(
                (b * 4, copy_sfa_query_heads, 512), compute_dtype
            ),
            "hbm_page_offsets": BufferSpec(((c + 256) // 128,), torch.int32),
            "dram_page_offsets": BufferSpec((self.context_len // 128,), torch.int32),
        }
        self.adapter_buffers = {
            "last_prefix": BufferSpec((num_layers, r + 1), torch.int32),
            "block_table": BufferSpec((b, self.context_len // 128), torch.int32),
            "sources": BufferSpec((b * 4, 1, 2048), torch.int32),
            "slots": BufferSpec((b * 4, 1, 2048), torch.int32),
            "topk_misses": BufferSpec((b * 4,), torch.int32),
            "miss_sources": BufferSpec((b, 32768), torch.int32),
            "miss_slots": BufferSpec((b, 32768), torch.int32),
            "miss_counts": BufferSpec((b,), torch.int32),
            "_metadata_prefix_scratch": BufferSpec((r + 1,), torch.int32),
        }
        self.pre_buffers = {
            name: BufferSpec((b,), dtype)
            for name, dtype in (
                ("safe_slots", torch.int32),
                ("lengths", torch.int32),
                ("prefixes", torch.int32),
                ("sparse_rows", torch.bool),
                ("valid_rows", torch.bool),
                ("request_states", torch.int32),
                ("next_prefixes", torch.int32),
                ("cache_sizes", torch.int32),
            )
        }
        self.tail_workspace_specs = (
            BufferSpec((b * 128,), torch.int64),
            BufferSpec((b * 128,), torch.int64),
            BufferSpec((b * 128,), torch.bool),
            BufferSpec((b * 4,), torch.int64),
            BufferSpec((b * 4,), torch.int64),
            BufferSpec((b * 4,), torch.bool),
        )
        self.query_spec = BufferSpec((b * 4, copy_sfa_query_heads, 576), compute_dtype)
        # Two Q-end vectors in each of the LIM and attention adapters.
        self.endpoint_bytes = 4 * b * torch.int32.itemsize
        self.host_token_capacity = None
        self.host_kv_spec = None
        self.key_scale_spec = None

    @property
    def managed_hbm_bytes(self):
        specs = (
            *self.manager_buffers.values(),
            *self.adapter_buffers.values(),
            *self.pre_buffers.values(),
            *self.tail_workspace_specs,
            self.query_spec,
        )
        return (
            sum(spec.nbytes for spec in specs)
            + self.endpoint_bytes
            + self.mf_fence_bytes
        )

    @property
    def host_kv_bytes(self):
        if self.host_kv_spec is None:
            raise RuntimeError("Bind the native C8 pool before reading Host bytes.")
        return self.host_kv_spec.nbytes

    def bind_native_pool(self, pool):
        if not getattr(pool, "dsa_kv_cache_store_fp8", False) or pool.page_size != 128:
            raise ValueError("A5 C8 requires the native page128 FP8 MLA pool.")
        layer_ids = getattr(pool, "indexer_layer_ids", ())
        if not layer_ids:
            raise ValueError("A5 C8 requires an indexer K layer.")
        key = pool.get_index_k_buffer(layer_ids[0])
        if key.dtype != torch.float8_e4m3fn or tuple(key.shape[1:]) != (128, 1, 128):
            raise ValueError("A5 LIM requires native FP8 index-K [pages,128,1,128].")
        capacity = key.shape[0] * 128
        if (
            self.host_token_capacity is not None
            and self.host_token_capacity != capacity
        ):
            raise ValueError("A5 C8 plan is already bound to a different pool.")
        self.host_token_capacity = capacity
        self.host_kv_spec = BufferSpec(
            (1, capacity, 1, 656), torch.int8, self.num_layers
        )
        return self

    def disable_prefill_workspace(self):
        """C8 prefill uses the existing bounded eager fallback."""

    @staticmethod
    def validate_buffers(owner, specs):
        for name, spec in specs.items():
            spec.validate(getattr(owner, name), name)


def resolve_c8_budget(
    *,
    request_rows,
    context_len,
    num_layers,
    top_k,
    kv_bytes_per_token,
    cache_tokens=0,
    hbm_budget_bytes=0,
    copy_sfa_query_heads=128,
    max_query_width=1,
    graph_batch_capacity=0,
    model_config=None,
    page_size=128,
    mf_shared=False,
    **_,
):
    if kv_bytes_per_token != 656:
        raise ValueError("A5 packed KV must reserve all 656 bytes per token.")
    if page_size != 128:
        raise ValueError("A5 ring-tail LIM requires page size128.")
    plan = C8BudgetPlan(
        request_rows=request_rows,
        context_len=context_len,
        num_layers=num_layers,
        top_k=top_k,
        cache_tokens=cache_tokens or max_query_width * top_k,
        copy_sfa_query_heads=copy_sfa_query_heads,
        max_query_width=max_query_width,
        batch_capacity=graph_batch_capacity,
        compute_dtype=getattr(model_config, "dtype", torch.bfloat16),
        mf_shared=mf_shared,
    )
    if hbm_budget_bytes and plan.managed_hbm_bytes > hbm_budget_bytes:
        raise ValueError(
            f"A5 sparse KV needs {plan.managed_hbm_bytes} bytes, "
            f"exceeding its {hbm_budget_bytes}-byte HBM budget."
        )
    return plan
