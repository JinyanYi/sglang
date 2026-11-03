"""CPU contracts for synthetic draft-extend graph history lengths."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE = (
    Path(__file__).parents[3]
    / "python/sglang/srt/speculative/eagle_draft_extend_cuda_graph_runner.py"
)


def resolve(backend, width):
    # Exercise the real helper without importing CUDA/NPU runtime dependencies.
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    tree.body = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "resolve_draft_extend_seq_len_fill_value"
    ]
    namespace = {}
    exec(compile(tree, str(SOURCE), "exec"), namespace)
    return namespace["resolve_draft_extend_seq_len_fill_value"](backend, width)


@pytest.mark.parametrize(
    "device,fill,kpool,width,expected",
    [
        ("npu:0", 0, 1, 4, 4),
        ("npu:0", 0, 1, 1, 1),
        ("npu:0", 8, 1, 4, 8),
        ("npu:0", 0, 128, 4, 132),
        ("cuda:0", 0, 1, 4, 0),
        ("cuda:0", 0, 128, 4, 132),
    ],
)
@pytest.mark.parametrize("wrapped", [False, True])
def test_capture_history_covers_npu_query_block(
    device, fill, kpool, width, expected, wrapped
):
    full_backend = SimpleNamespace(device=device, dsa_index_kpool=kpool)
    backend = SimpleNamespace(get_cuda_graph_seq_len_fill_value=lambda: fill)
    if wrapped:
        backend.full_attn_backend = full_backend
    else:
        backend.device = device
        backend.dsa_index_kpool = kpool
    assert resolve(backend, width) == expected
