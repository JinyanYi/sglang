"""Stage a causal two-page MTP tail independently of historical LIM residency."""

import triton
import triton.language as tl
from sgl_kernel_npu.sparsity_driven_kv_offload import unidex_copy_inplace


@triton.jit
def _tail_metadata_kernel(
    req_to_token,
    requests,
    positions,
    starts,
    ends,
    history_src,
    history_dst,
    history_valid,
    current_dst,
    current_valid,
    REQUEST_ROWS: tl.constexpr,
    CONTEXT: tl.constexpr,
    REQUEST_STRIDE: tl.constexpr,
    C: tl.constexpr,
    Q: tl.constexpr,
):
    batch = tl.program_id(0)
    request = tl.load(requests + batch).to(tl.int32)
    first = tl.load(positions + batch * Q).to(tl.int32)
    last = tl.load(positions + batch * Q + Q - 1).to(tl.int32) + 1
    active = (request > 0) & (request < REQUEST_ROWS) & (first >= 0) & (last <= CONTEXT)
    safe = tl.where(active, request, 0)
    prefix = first // 128 * 128
    old_start = tl.load(starts + safe)
    old_end = tl.load(ends + safe)
    offset = tl.arange(0, 128)
    pos = prefix + offset
    restore = active & (pos < first) & ((pos < old_start) | (pos >= old_end))
    physical = tl.load(
        req_to_token + safe * REQUEST_STRIDE + pos, mask=restore, other=0
    ).to(tl.int64)
    dst = safe.to(tl.int64) * (C + 256) + C + pos % 256
    tl.store(history_src + batch * 128 + offset, physical)
    tl.store(history_dst + batch * 128 + offset, tl.where(restore, dst, 0))
    tl.store(history_valid + batch * 128 + offset, restore)
    q = tl.arange(0, Q)
    qpos = first + q
    qdst = safe.to(tl.int64) * (C + 256) + C + qpos % 256
    tl.store(current_dst + batch * Q + q, tl.where(active, qdst, 0))
    tl.store(current_valid + batch * Q + q, active)
    # The following data copies and consumers all use this ordered stream.
    tl.store(starts + safe, prefix, mask=active)
    tl.store(ends + safe, last, mask=active)


def stage_mtp_tail(
    *,
    current,
    cache,
    host,
    host_address,
    req_to_token,
    requests,
    positions,
    starts,
    ends,
    workspace,
    capacity,
    context_len,
    query_width=4,
):
    """Write current candidates locally; restore only missing stable tail rows."""
    batch = requests.numel()
    if query_width not in (1, 4):
        raise ValueError("Tail query width must be 1 or 4")
    if (
        positions.numel() < batch * query_width
        or current.shape[0] < batch * query_width
    ):
        raise ValueError(
            "Separate tail requires query_width current KV rows per request"
        )
    hs, hd, hv, cs, cd, cv = workspace
    _tail_metadata_kernel[(batch,)](
        req_to_token,
        requests,
        positions,
        starts,
        ends,
        hs,
        hd,
        hv,
        cd,
        cv,
        starts.numel() - 1,
        context_len,
        req_to_token.stride(0),
        capacity,
        query_width,
    )
    unidex_copy_inplace(
        host,
        cache,
        hs[: batch * 128],
        hd[: batch * 128],
        hv[: batch * 128],
        2,
        2,
        block_dim=48,
        src_ptr=host_address,
    )
    unidex_copy_inplace(
        current.view(-1, 1, current.shape[-1]),
        cache,
        cs[: batch * query_width],
        cd[: batch * query_width],
        cv[: batch * query_width],
        1,
        2,
        block_dim=48,
    )
