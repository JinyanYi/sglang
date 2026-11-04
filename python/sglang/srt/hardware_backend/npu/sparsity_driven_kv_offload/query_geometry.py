"""Query geometry shared by the packed A5 chain-MTP callers."""

A5_MAX_QUERY_WIDTH = 14


def configured_a5_query_width(spec):
    """Bound persistent buffers by the configured chain, not TP token padding."""
    if (spec.speculative_algorithm or "NONE").upper() == "NONE":
        return 1
    width = spec.speculative_num_draft_tokens
    if (
        isinstance(width, bool)
        or not isinstance(width, int)
        or not 2 <= width <= A5_MAX_QUERY_WIDTH
        or spec.speculative_eagle_topk != 1
        or spec.speculative_num_steps != width - 1
    ):
        raise ValueError(
            "A5 LIM requires chain MTP with topk=1, draft tokens in [2,14] "
            "and steps=draft tokens-1."
        )
    return width


def sparse_query_width(forward_batch):
    """Read verify geometry explicitly; ordinary decode always has Q1."""
    if forward_batch.forward_mode.is_decode():
        return 1
    width = getattr(forward_batch.spec_info, "num_tokens_per_req", None)
    if (
        isinstance(width, bool)
        or not isinstance(width, int)
        or not 1 <= width <= A5_MAX_QUERY_WIDTH
    ):
        raise ValueError("A5 target verify requires its actual query width in [1,14].")
    return width
