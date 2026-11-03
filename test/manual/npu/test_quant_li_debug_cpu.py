"""Check diagnostic ordering and capture safety without a device runtime."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE = (
    Path(__file__).parents[3]
    / "python/sglang/srt/hardware_backend/npu/quant_li_debug.py"
)


@pytest.fixture
def diagnostic(monkeypatch):
    spec = importlib.util.spec_from_file_location("quant_li_debug", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    events = []
    torch = SimpleNamespace(
        npu=SimpleNamespace(
            is_current_stream_capturing=lambda: False,
            synchronize=lambda: events.append("synchronize"),
        ),
        distributed=SimpleNamespace(is_initialized=lambda: False),
        is_tensor=lambda value: False,
        save=lambda payload, path: (events.append("save"), path.write_text("fixture")),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace())
    monkeypatch.setattr(module, "runtime_info", lambda *args: {})
    mode = SimpleNamespace(is_draft_extend_v2=lambda: True, is_decode=lambda: False)
    batch = SimpleNamespace(forward_mode=mode)
    backend = SimpleNamespace(is_draft_worker=True)
    return module, torch, events, batch, backend


def test_snapshot_finishes_before_native_call_and_only_once(diagnostic, tmp_path):
    module, _, events, batch, backend = diagnostic
    for _ in range(2):
        module.snapshot_draft_li({"sparse_count": 2048}, batch, backend, 0, tmp_path)
    assert events == ["synchronize", "save"]
    report = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert report["stage"] == "ready_to_call_native_li"
    assert report["inputs"]["sparse_count"] == 2048
    assert Path(report["fixture"]).exists()


def test_sync_failure_does_not_get_misattributed_to_li(diagnostic, tmp_path):
    module, torch, events, batch, backend = diagnostic

    def fail():
        raise RuntimeError("earlier async failure")

    torch.npu.synchronize = fail
    with pytest.raises(RuntimeError, match="earlier async failure"):
        module.snapshot_draft_li({}, batch, backend, 0, tmp_path)
    assert not events
    report = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert report["stage"] == "before_sync"
    assert not list(tmp_path.glob("*.pt"))


@pytest.mark.parametrize("capturing,target", [(True, False), (False, True)])
def test_no_host_io_in_capture_or_target_path(diagnostic, tmp_path, capturing, target):
    module, torch, events, batch, backend = diagnostic
    torch.npu.is_current_stream_capturing = lambda: capturing
    backend.is_draft_worker = not target
    module.snapshot_draft_li({}, batch, backend, 0, tmp_path)
    assert not events
    assert not list(tmp_path.iterdir())


def test_nonzero_rank_records_metadata_without_copying_kv(diagnostic, tmp_path):
    module, torch, events, batch, backend = diagnostic
    torch.distributed = SimpleNamespace(is_initialized=lambda: True, get_rank=lambda: 3)
    module.snapshot_draft_li({}, batch, backend, 0, tmp_path)
    assert events == ["synchronize"]
    assert next(tmp_path.glob("rank3-*.json"))
    assert not list(tmp_path.glob("*.pt"))
