"""Exercise C8 eager host-metadata handling without loading NPU operators."""

import ast
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE = (
    Path(__file__).parents[3]
    / "python/sglang/srt/hardware_backend/npu/sparsity_driven_kv_offload/c8.py"
)


class Tensor:
    def __init__(self, values):
        self.values = values
        self.copies = 0

    def detach(self):
        return self

    def cpu(self):
        self.copies += 1
        return self

    def numel(self):
        return len(self.values)

    def tolist(self):
        return list(self.values)


def load(namespace=None):
    namespace = {} if namespace is None else namespace
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    exec(compile(tree, str(SOURCE), "exec"), namespace)
    return namespace


def batch(prefill=True, mirror=None):
    return SimpleNamespace(
        req_pool_indices=Tensor([3, 7]),
        req_pool_indices_cpu=mirror,
        seq_lens=Tensor([128, 129]),
        seq_lens_cpu=None,
        spec_info=None,
        forward_mode=SimpleNamespace(
            is_decode=lambda: False,
            is_target_verify=lambda: False,
            is_extend_without_speculative=lambda: prefill,
        ),
    )


def test_prefill_materializes_once_across_layers():
    resolve = load()["_eager_c8_request_ids"]
    current = batch()
    for _ in range(3):
        assert resolve(current) == [3, 7]
    assert current.req_pool_indices.copies == 1


def test_existing_host_mirror_avoids_device_copy():
    current = batch(mirror=Tensor([3, 7]))
    assert load()["_eager_c8_request_ids"](current) == [3, 7]
    assert current.req_pool_indices.copies == 0


def test_speculative_slots_do_not_acquire_a_stale_host_mirror():
    current = batch(prefill=False)
    resolve = load()["_eager_c8_request_ids"]
    assert resolve(current) == [3, 7]
    current.req_pool_indices.values = [9, 11]
    assert resolve(current) == [9, 11]
    assert current.req_pool_indices_cpu is None


def test_mismatched_mirror_is_rejected():
    with pytest.raises(ValueError, match="request counts"):
        load()["_eager_c8_request_ids"](batch(mirror=Tensor([3])))


@pytest.mark.parametrize("graph", [False, True])
def test_eager_fallback_uses_optional_metadata_but_graph_never_copies(
    monkeypatch, graph
):
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace())
    pages = []

    class StopAfterMetadata(Exception):
        pass

    def tensor(values, **kwargs):
        pages.extend(values)
        raise StopAfterMetadata

    namespace = load(
        {
            "torch": SimpleNamespace(
                npu=SimpleNamespace(current_stream=lambda device: None),
                tensor=tensor,
                int64="int64",
            ),
            "use_native_mtp_verify": lambda *args: False,
        }
    )
    manager = SimpleNamespace(
        lim_adapter=SimpleNamespace(reset_requests=lambda ids: None),
        max_context_len=45056,
        size=20,
    )
    backend = SimpleNamespace(
        sparse_kv_manager=manager,
        device="npu:0",
        graph_mode=graph,
        forward_metadata=SimpleNamespace(seq_lens_cpu_int=None),
    )
    current = batch()
    expected = RuntimeError if graph else StopAfterMetadata
    with pytest.raises(expected):
        namespace["forward_c8_offload"](
            backend,
            SimpleNamespace(device="npu:0"),
            None,
            None,
            current,
            False,
            None,
            None,
            None,
        )
    assert current.req_pool_indices.copies == (0 if graph else 1)
    assert pages == ([] if graph else [1, 2])
