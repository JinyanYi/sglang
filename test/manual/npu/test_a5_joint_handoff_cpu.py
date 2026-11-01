"""CPU checks for profile completeness/accounting and live-stream barriers."""

import importlib.util
import json
from decimal import Decimal
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[3] / "benchmark/hisparse/a5_copysfa_mtp"


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


analysis = load("analyze_profile")


def test_overlap_is_not_double_counted():
    assert (
        analysis.union_duration([(Decimal(0), Decimal(4)), (Decimal(3), Decimal(9))])
        == 9
    )


def test_article_windows_and_real_tokens_per_round(tmp_path):
    events = [{"name": "process_name", "pid": 2, "args": {"name": "Ascend Hardware"}}]
    for start in (0, 20):
        events.append(dict(name="step[VERIFY bs=16]", ph="X", pid=1, ts=start, dur=1))
        for name, offset, duration in (
            ("FusedQuantLightningIndexerManage", 0, 4),
            ("FusedQuantScatterCopySparseFlashAttention", 3, 6),
        ):
            events.append(
                dict(
                    name=name,
                    ph="X",
                    pid=2,
                    ts=start + offset,
                    dur=duration,
                    args={"Task Type": "MIX_AIC"},
                )
            )
    path = tmp_path / "trace_view.json"
    path.write_text(json.dumps(events))
    result = analysis.trace_report(
        path, 16, 2, [dict(start_us=0, end_us=10), dict(start_us=20, end_us=30)], 3, 1
    )
    assert result["valid"]
    assert result["component_kernel_counts"] == {"LIM": 2, "CopySFA": 2}
    assert result["article"]["target_78_layer_ms"] == pytest.approx(0.01)
    assert result["article"]["other_target_ms"] == pytest.approx(0.001)
    assert result["article"]["itl_ms"] == pytest.approx(0.02)
    assert result["article"]["tpot_ms"] == pytest.approx(0.02 / 3)
    assert result["article"]["system_tps"] == pytest.approx(16 / (0.02 / 3) * 1000)
    assert not analysis.trace_report(path, 16, 3)["valid"]
    assert not analysis.trace_report(path, 8, 2)["valid"]
    assert analysis.trace_report(path, 16, 2)["article"]["tpot_ms"] is None


def test_draft_markers_do_not_count_as_target_steps(tmp_path):
    path = tmp_path / "trace_view.json"
    path.write_text(
        json.dumps(
            [
                dict(name="process_name", pid=2, args={"name": "Ascend Hardware"}),
                dict(name="step[DRAFT bs=16]", ph="X", pid=1, ts=0, dur=1),
            ]
        )
    )
    assert not analysis.trace_report(path, 16, 1)["valid"]


def test_completed_stream_does_not_remain_live(monkeypatch):
    import threading

    workload = load("shared_prefix_workload")

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def __iter__(self):
            yield b'data: {"meta_info":{"completion_tokens":1,"prompt_tokens":1,"cached_tokens":1}}'

    monkeypatch.setattr(
        workload.urllib.request, "urlopen", lambda *args, **kwargs: Response()
    )
    active = workload.ActiveStreams()
    workload.stream_one(
        "http://example.invalid", [1], 1, 1, 0, threading.Lock(), active, 1
    )
    assert len(active) == 0
    assert active.peak == 1


def test_joint_gate_requires_graph_q4_and_rejects_skips(tmp_path):
    gate = load("verify_joint_junit")
    names = [
        f"test_real_lim_to_copysfa[{graph}-{width}]"
        for graph in ("False", "True")
        for width in (1, 4)
    ]
    path = tmp_path / "junit.xml"

    def write(selected, skipped=0):
        path.write_text(
            '<testsuites><testsuite tests="4" failures="0" errors="0" skipped="'
            + str(skipped)
            + '">'
            + "".join('<testcase name="' + name + '"/>' for name in selected)
            + "</testsuite></testsuites>"
        )

    write(names)
    assert gate.verify([path])["valid"]
    write(names[:-1])
    assert not gate.verify([path])["valid"]
    write(names, skipped=1)
    assert not gate.verify([path])["valid"]
