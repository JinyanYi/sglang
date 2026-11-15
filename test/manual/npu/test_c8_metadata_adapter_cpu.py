"""Check that framework calls forward tensors to the kernel package unchanged."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

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
