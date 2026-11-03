"""Synthetic graph traces keep target/draft and device/host timing distinct."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[3] / "benchmark/hisparse/a5_copysfa_mtp"


@pytest.fixture
def analysis(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    spec = importlib.util.spec_from_file_location(
        "article_report", ROOT / "article_report.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture_events(native=False):
    events = [{"name": "process_name", "pid": 2, "args": {"name": "Ascend Hardware"}}]

    def event(name, start, duration, task, model=7, lane=1, kind="AI_CORE"):
        return dict(
            name=name,
            ph="X",
            pid=2,
            tid=lane,
            ts=str(start),
            dur=duration,
            args={"Task Type": kind, "Task Id": task, "Model Id": model},
        )

    for start in (0, 1000):
        # Host markers intentionally have unrelated timestamps and durations.
        events.append(
            dict(
                name="step[VERIFY bs=16]",
                ph="X",
                pid=1,
                tid=1,
                ts=start + 9000,
                dur=7000,
            )
        )
        events.append(event("embedding", start, 1, 0))
        for index in range(21):
            events.append(
                event(
                    "LightningIndexer"
                    if native
                    else "FusedQuantLightningIndexerManage",
                    start + 1 + index / 10,
                    0.1,
                    index + 1,
                )
            )
        for index in range(78):
            events.append(
                event(
                    "SparseFlashAttention"
                    if native
                    else "FusedQuantScatterCopySparseFlashAttention",
                    start + 10 + index,
                    0.5,
                    index + 22,
                )
            )
        events.append(event("head", start + 100, 2, 100, kind="AI_VECTOR_CORE"))
        events.append(event("HcclTask", start + 2, 5, 0, lane=2, kind="COMMUNICATION"))
        for index in range(3):
            events.append(
                event("SparseFlashAttention", start + 200 + index, 20, index, model=8)
            )
    return events


def inspect(analysis, tmp_path, events, measured=None):
    path = tmp_path / "trace_view.json"
    path.write_text(json.dumps(events))
    return analysis.inspect_trace(path, 16, 2, 78, 3.0, measured, 4)


@pytest.mark.parametrize("native", [False, True])
def test_device_graph_windows_exclude_draft_and_ignore_host_enqueue(
    analysis, tmp_path, native
):
    row = inspect(analysis, tmp_path, fixture_events(native))
    assert row["status"] == "graph_proxy_validated"
    assert row["target_model_id"] == "7"
    assert row["device_itl_ms"] == pytest.approx(1)
    assert row["target_graph_ms"] == pytest.approx(0.102)
    assert row["pure_78_layer_ms"] is None
    key = "SFA" if native else "CopySFA"
    assert row["target_component_sum_ms"][key] == pytest.approx(0.039)
    assert row["per_round_component_counts"][key] == 78
    assert row["target_component_sum_ms"]["other"] == pytest.approx(0.008)
    assert row["article_assumption"]["tpot_ms"] == pytest.approx(1 / 3)
    assert row["article_assumption"]["balanced_system_tps_estimate"] == pytest.approx(
        192000
    )


def test_measured_acceptance_remains_separate_from_article_assumption(
    analysis, tmp_path
):
    row = inspect(analysis, tmp_path, fixture_events(), measured=2)
    assert row["measured_acceptance_formula"]["tpot_ms"] == 0.5
    assert row["article_assumption"]["tokens_per_round"] == 3


def test_incomplete_attention_count_is_not_silently_accepted(analysis, tmp_path):
    events = fixture_events()
    events.pop(next(i for i, row in enumerate(events) if "CopySparse" in row["name"]))
    row = inspect(analysis, tmp_path, events)
    assert row["status"] == "incomplete"
    assert "candidates" in row["errors"][0]


def test_ambiguous_target_models_are_rejected(analysis, tmp_path):
    events = fixture_events()
    copies = []
    for row in events:
        if row.get("args", {}).get("Model Id") == 7:
            copies.append(dict(row, args=dict(row["args"], **{"Model Id": 9})))
    row = inspect(analysis, tmp_path, events + copies)
    assert row["status"] == "incomplete"


def test_wrong_attention_distribution_across_rounds_is_rejected(analysis, tmp_path):
    events = fixture_events()
    # Overall count remains 156, but graph windows contain 77 and 79 calls.
    first = next(row for row in events if "CopySparse" in row["name"])
    first["ts"] = "1088"
    row = inspect(analysis, tmp_path, events)
    assert row["status"] == "incomplete"
    assert "attention count" in row["errors"][0]


def test_missing_stream_anchor_is_rejected(analysis, tmp_path):
    events = fixture_events()
    events = [
        row for row in events if not (row["name"] == "HcclTask" and row["ts"] == "1002")
    ]
    row = inspect(analysis, tmp_path, events)
    assert row["status"] == "incomplete"
    assert "first task occurs" in row["errors"][0]


def test_text_labels_shared_prefix_capacity_and_proxy_limits(analysis):
    result = dict(
        case_root="example",
        traces=[],
        expected_ranks=8,
        status="incomplete",
        dp_size=4,
        batch=16,
        assumed_tokens=3,
        unprofiled_performance={"shared_prefix_tokens": 39872},
    )
    text = analysis.text_report(result)
    assert "METHOD GAP" in text
    assert "not vLLM" in text
    assert "unresolved" in text
    assert "not measured acceptance" in text


def test_metadata_memcpy_is_not_called_kv_loading(analysis):
    assert analysis.component("MEMCPY_ASYNC") == "memcpy_unattributed"
    assert analysis.component("unidex_copy_21") == "kv_copy_candidates"


@pytest.mark.parametrize("stack,expected_exit", [(False, 0), (True, 2)])
def test_cli_writes_pasteable_report_and_keeps_stack_modes_separate(
    analysis, tmp_path, monkeypatch, stack, expected_exit
):
    profile = tmp_path / "04-profile/profile/rank0"
    profile.mkdir(parents=True)
    (profile / "trace_view.json").write_text(json.dumps(fixture_events()))
    capture = tmp_path / "04-profile/capture"
    capture.mkdir()
    (capture / "profile-request.json").write_text(
        json.dumps({"payload": {"with_stack": stack}})
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "article_report.py",
            str(tmp_path),
            "--expected-ranks",
            "1",
            "--dp-size",
            "1",
            "--min-steps",
            "2",
        ],
    )
    assert analysis.main() == expected_exit
    report = tmp_path / "06-article-report/REPORT.txt"
    assert report.exists()
    assert "ASSUMED_TPOT_ms" in report.read_text()
    result = json.loads((report.parent / "article-report.json").read_text())
    assert result["status"] == ("incomplete" if stack else "graph_proxy_validated")
