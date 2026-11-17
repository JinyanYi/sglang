"""Check that framework calls forward tensors to the kernel package unchanged."""

import ast
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

SOURCE = (
    Path(__file__).parents[3]
    / "python/sglang/srt/hardware_backend/npu/sparsity_driven_kv_offload/c8_metadata.py"
)


@pytest.mark.parametrize(
    "name",
    ("fused_prepare_lim_metadata", "fused_lim_copy_sfa_metadata", "stage_mtp_tail"),
)
def test_metadata_adapter_passes_payload_and_result_unchanged(monkeypatch, name):
    calls, result = [], object()

    def operator(**kwargs):
        calls.append(kwargs)
        return result

    package = ModuleType("sgl_kernel_npu.sparsity_driven_kv_offload")
    package.c8_metadata = SimpleNamespace(**{name: operator})
    monkeypatch.setitem(sys.modules, package.__name__, package)
    spec = importlib.util.spec_from_file_location("c8_metadata_adapter", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    payload = dict(current=object(), positions=object(), query_width=4)
    assert getattr(module, name)(**payload) is result
    assert calls == [payload]
    assert calls[0]["current"] is payload["current"]
    assert calls[0]["positions"] is payload["positions"]


@pytest.mark.parametrize("fused", [False, True])
def test_compact_hbm_rows_keep_request_ids_and_host_pages(fused):
    source = SOURCE.with_name("c8_lim_bridge.py")
    tree = ast.parse(source.read_text(encoding="utf-8"))
    cls = next(
        node for node in tree.body if getattr(node, "name", None) == "C8LIMBridge"
    )
    calls = []
    namespace = {
        "torch": torch,
        "fused_lim_copy_sfa_metadata": lambda **kwargs: calls.append(kwargs),
    }
    exec(
        compile(ast.Module(body=[cls], type_ignores=[]), str(source), "exec"), namespace
    )
    bridge = object.__new__(namespace["C8LIMBridge"])
    bridge.capacity, bridge.rows = 8192, 4
    bridge.fused_copy_metadata = fused
    bridge.block_table = torch.tensor([[17, 19], [17, 23], [0, 0]], dtype=torch.int32)
    host = bridge.block_table.clone()

    def fill_host_table(requests, valid, output):
        torch.testing.assert_close(requests, torch.tensor([1, 3, 4], dtype=torch.int32))
        output.copy_(torch.where(valid[:, None], host, 0))

    manager = SimpleNamespace(
        hbm_page_offsets=torch.arange(66, dtype=torch.int32),
        device_kv_stride=8448,
        max_context_len=256,
        planner_actual_seq_lengths_kv=torch.empty(3, dtype=torch.int32),
        planner_num_cache_tokens=torch.empty(3, dtype=torch.int32),
        planner_hbm_block_table=torch.empty(3, 66, dtype=torch.int32),
        planner_dram_block_table=torch.empty(3, 2, dtype=torch.int32),
        fill_host_block_table=fill_host_table,
    )
    plan = SimpleNamespace(
        request_slots=torch.tensor([1, 3, 4], dtype=torch.int32),
        source_ids=torch.ones(12, 2048, dtype=torch.int32),
        device_slots=torch.ones(12, 2048, dtype=torch.int32),
        topk_misses=torch.ones(12, dtype=torch.int32),
        miss_counts=torch.ones(3, dtype=torch.int32),
        valid_rows=torch.tensor([True, True, False]),
        sparse_rows=torch.tensor([True, True, False]),
        seq_lens=torch.tensor([8449, 8449, 0], dtype=torch.int32),
        prefix_lens=torch.tensor([8320, 8320, 0], dtype=torch.int32),
    )
    bridge.copy_sfa_metadata(manager, plan)
    if fused:
        assert calls[0]["hbm_request_offset"] == 1
        assert calls[0]["request_slots"] is plan.request_slots
        assert calls[0]["host_block_table"].data_ptr() == bridge.block_table.data_ptr()
    else:
        table = manager.planner_hbm_block_table
        assert table[0, 0] == 0 and table[1, 0] == 2 * 66
        assert table[0, -2:].tolist() == [65, 64]
        assert table[1].max() < 3 * 66
        assert not table[2].count_nonzero()
        torch.testing.assert_close(manager.planner_dram_block_table, host)
        assert (plan.source_ids[8:] == -1).all()
