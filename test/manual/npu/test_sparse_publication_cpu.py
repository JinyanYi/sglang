"""CPU interpreter checks for the shared A3/C8 publication metadata kernel."""

import importlib.util
import os
from pathlib import Path

import pytest
import torch

SOURCE = (
    Path(__file__).resolve().parents[3]
    / "python/sglang/kernels/ops/kvcache/npu_sparse_kv.py"
)


def load():
    spec = importlib.util.spec_from_file_location("publication_cpu", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "mode,width",
    [("decode", 1), ("verify", 4), ("verify", 14), ("compact", 1), ("padded", 1)],
)
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_shared_rows_match_independent_publication_oracle(
    monkeypatch, mode, width, dtype
):
    if os.environ.get("TRITON_INTERPRET") != "1":
        pytest.skip("CPU arithmetic requires TRITON_INTERPRET=1")
    module = load()
    prefill = mode in ("compact", "padded")
    requests = torch.tensor([1, 0, 5, 20, -1], dtype=dtype)
    lengths = torch.tensor([10, 5, 8, 9, 2], dtype=dtype)
    extended = [2, 0, 3, 1, 1]
    tokens = sum(extended) if mode == "compact" else (15 if prefill else 5 * width + 2)
    # Exercise strided inputs, TP suffix rows and invalid physical addresses.
    positions = (torch.arange(tokens * 2, dtype=dtype) + 5)[::2]
    location = (torch.arange(tokens * 2, dtype=dtype) - 1)[::2]
    location[3] = 100
    prefix = torch.tensor([3, 4], dtype=dtype)
    kwargs = dict(
        tokens=tokens,
        request_rows=20,
        context_len=20,
        host_capacity=50,
        query_width=width,
        extend_seq_lens=torch.tensor(extended, dtype=dtype) if prefill else None,
        extend_prefix_lens=prefix if prefill else None,
        compact_prefill=mode == "compact",
    )
    expected_position, expected_valid = [], []
    for token in range(tokens):
        if prefill:
            if mode == "compact":
                row = next(row for row in range(5) if token < sum(extended[: row + 1]))
                offset = token - sum(extended[:row])
            else:
                row, offset = divmod(token, tokens // 5)
            position = (int(prefix[row]) if row < prefix.numel() else 0) + offset
            live = (
                0 < int(requests[row]) < 20
                and offset < extended[row]
                and 0 <= position < 20
            )
        else:
            row = token // width
            position = int(positions[token]) if row < 5 else -1
            live = (
                row < 5
                and 0 < int(requests[row]) < 20
                and int(lengths[row]) > 0
                and 0 <= position < 20
            )
        expected_position.append(position)
        expected_valid.append(live and 0 <= int(location[token]) < 50)

    # Metadata preparation must not introduce a payload concatenation for C8.
    def unexpected_cat(*args, **kwargs):
        raise AssertionError("metadata-only publication concatenated a payload")

    with monkeypatch.context() as scope:
        scope.setattr(torch, "cat", unexpected_cat)
        actual = module.prepare_sparse_kv_publication_rows(
            requests, lengths, positions, location, **kwargs
        )
    expected = (
        torch.arange(tokens, dtype=torch.int64),
        location.long(),
        torch.tensor(expected_position, dtype=torch.int64),
        torch.tensor(expected_valid, dtype=torch.bool),
    )
    for output, reference in zip(actual, expected):
        torch.testing.assert_close(output, reference, rtol=0, atol=0)

    k, rope = torch.randn(tokens, 1, 8), torch.randn(tokens, 1, 4)
    kwargs.pop("tokens")
    adapted = module.prepare_sparse_kv_publication(
        k, rope, requests, lengths, positions, location, **kwargs
    )
    for output, reference in zip(adapted[:4], expected):
        torch.testing.assert_close(output, reference, rtol=0, atol=0)
    if prefill:
        assert adapted[4][0] is k and adapted[4][1] is rope
    else:
        torch.testing.assert_close(adapted[4], torch.cat((k, rope), -1), rtol=0, atol=0)
