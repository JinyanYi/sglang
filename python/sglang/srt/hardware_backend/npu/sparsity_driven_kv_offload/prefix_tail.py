"""Shared graph-friendly metadata for sparse-prefix plus resident-tail decode."""

from __future__ import annotations

from dataclasses import dataclass

import torch

PAGE_SIZE = 128
RESIDENT_TAIL_TOKENS = 256


@dataclass(frozen=True)
class PrefixTailMetadata:
    indexer_seq_lens: torch.Tensor
    prefix_lens: torch.Tensor
    tail_lens: torch.Tensor
    sparse_rows: torch.Tensor
    cache_tokens: torch.Tensor
    actual_kv_lens: torch.Tensor


def sparse_batch_capacity(
    request_capacity: int, graph_batch_capacity: int | None = None
) -> int:
    """Return scratch capacity for both request rows and padded graph batches."""
    if request_capacity < 0 or (
        graph_batch_capacity is not None and graph_batch_capacity < 0
    ):
        raise ValueError("Sparse KV capacities must be non-negative.")
    return max(request_capacity, graph_batch_capacity or 0)


def valid_sparse_request_rows(
    req_ids: torch.Tensor, seq_lens: torch.Tensor, request_capacity: int
) -> torch.Tensor:
    """Mask real requests using the pool size, independently of graph padding."""
    batch_size = req_ids.shape[0]
    return (req_ids >= 0) & (req_ids < request_capacity) & (seq_lens[:batch_size] > 0)


def derive_prefix_tail_metadata(
    seq_lens: torch.Tensor,
    cache_capacity: int,
    *,
    sparse_context_len: int,
    page_size: int = PAGE_SIZE,
    out: PrefixTailMetadata | None = None,
) -> PrefixTailMetadata:
    """Split long rows at a stable page boundary without host scalar reads."""
    if (
        cache_capacity <= 0
        or sparse_context_len <= 0
        or sparse_context_len > cache_capacity
        or page_size <= 0
        or cache_capacity % page_size
    ):
        raise ValueError("cache_capacity must be positive and page aligned")
    target_device = out.prefix_lens.device if out is not None else seq_lens.device
    lengths = seq_lens.to(device=target_device, dtype=torch.int32).clamp(min=0)
    aligned_prefix = ((lengths - 1).clamp(min=0) // page_size) * page_size
    sparse_rows = lengths > sparse_context_len
    prefix_lens = torch.where(sparse_rows, aligned_prefix, lengths)
    tail_lens = torch.where(sparse_rows, lengths - prefix_lens, 0)
    cache_tokens = torch.where(
        sparse_rows, torch.full_like(lengths, cache_capacity), torch.zeros_like(lengths)
    )
    actual_kv_lens = torch.where(sparse_rows, cache_tokens + tail_lens, lengths)
    if out is not None:
        out.indexer_seq_lens.copy_(prefix_lens)
        out.prefix_lens.copy_(prefix_lens)
        out.tail_lens.copy_(tail_lens)
        out.sparse_rows.copy_(sparse_rows)
        out.cache_tokens.copy_(cache_tokens)
        out.actual_kv_lens.copy_(actual_kv_lens)
        return out
    return PrefixTailMetadata(
        prefix_lens, prefix_lens, tail_lens, sparse_rows, cache_tokens, actual_kv_lens
    )


def resident_tail_write_mask(
    valid_mask: torch.Tensor,
    token_positions: torch.Tensor,
    visible_seq_lens: torch.Tensor,
    tail_capacity: int = RESIDENT_TAIL_TOKENS,
) -> torch.Tensor:
    """Keep only source tokens in each request's latest resident-tail window."""
    if tail_capacity <= 0:
        raise ValueError("tail_capacity must be positive")
    return valid_mask & (token_positions >= visible_seq_lens - tail_capacity)


def resident_tail_copy_indices(
    req_ids: torch.Tensor,
    token_positions: torch.Tensor,
    visible_seq_lens: torch.Tensor,
    valid_mask: torch.Tensor,
    device_kv_stride: int,
    cache_capacity: int,
    tail_capacity: int = RESIDENT_TAIL_TOKENS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build flattened HBM destinations and the collision-free tail write mask."""
    if device_kv_stride < cache_capacity + tail_capacity:
        raise ValueError("device_kv_stride must include the cache and resident tail")
    tail_mask = resident_tail_write_mask(
        valid_mask, token_positions, visible_seq_lens, tail_capacity
    )
    destination = (
        req_ids * device_kv_stride
        + cache_capacity
        + token_positions.remainder(tail_capacity)
    )
    return destination, tail_mask


def fill_prefix_tail_block_table(
    req_ids: torch.Tensor,
    valid_rows: torch.Tensor,
    prefix_lens: torch.Tensor,
    page_offsets: torch.Tensor,
    cache_capacity: int,
    tail_capacity: int,
    output: torch.Tensor,
) -> None:
    """Map the logical suffix pages onto the two-page physical tail ring."""
    if tail_capacity != RESIDENT_TAIL_TOKENS or tail_capacity % PAGE_SIZE:
        raise ValueError("the resident tail must contain exactly two KV pages")
    pages_per_row = cache_capacity // PAGE_SIZE + tail_capacity // PAGE_SIZE
    if output.shape != (req_ids.numel(), pages_per_row):
        raise ValueError("block table has an unexpected prefix-tail page shape")
    safe_req = torch.where(valid_rows, req_ids, torch.zeros_like(req_ids))
    row_bases = safe_req * pages_per_row
    torch.add(row_bases[:, None], page_offsets[None, :], out=output)
    cache_pages = cache_capacity // PAGE_SIZE
    ring_pages = tail_capacity // PAGE_SIZE
    page_parity = torch.div(prefix_lens, PAGE_SIZE, rounding_mode="floor") % ring_pages
    output[:, cache_pages] = row_bases + cache_pages + page_parity
    output[:, cache_pages + 1] = (
        row_bases + cache_pages + (page_parity + 1) % ring_pages
    )
