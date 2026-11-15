"""A5 target admission and optional chain diagnostics; draft KV stays native."""


def is_sparse_mtp_batch(forward_batch):
    mode = forward_batch.forward_mode
    return (
        mode.is_target_verify()
        or mode.is_draft_extend_v2()
        or (mode.is_decode() and getattr(forward_batch, "spec_info", None) is not None)
    )


def use_native_mtp_verify(backend, forward_batch, manager):
    """Keep the A5 Q4 fast path restricted to admitted long requests."""
    if not (
        getattr(manager, "packed_c8", False)
        and forward_batch.forward_mode.is_target_verify()
        and getattr(forward_batch.spec_info, "num_tokens_per_req", None) == 4
    ):
        return False
    if backend.graph_mode:
        return True
    lengths = forward_batch.seq_lens_cpu
    return lengths is not None and all(
        int(n) == 0 or int(n) >= manager.cache_capacity for n in lengths
    )


def validate_mtp_chain(forward_batch, positions, live, request_rows):
    """Retain optional eager validation for A5's contiguous Q4 chain."""
    from sglang.srt.environ import envs

    if not envs.SGLANG_ENABLE_ASYNC_ASSERT.get():
        return
    import torch

    from sglang.srt.utils.async_probe import maybe_assert_async

    if positions.device.type == "npu" and torch.npu.is_current_stream_capturing():
        raise RuntimeError("Disable graph capture for A5 MTP async validation.")
    requests = forward_batch.req_pool_indices
    active = forward_batch.seq_lens > 0
    valid = (requests > 0) & (requests < request_rows)
    contiguous = (positions[:, 1:] == positions[:, :-1] + 1).all(dim=1)
    maybe_assert_async(
        ((~active) | (valid & live.all(dim=1) & contiguous)).all(),
        "A5 native MTP requires complete contiguous Q4 live rows.",
    )
