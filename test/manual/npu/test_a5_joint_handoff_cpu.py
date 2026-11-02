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
    dp4 = analysis.trace_report(
        path, 16, 2, [dict(start_us=0, end_us=10), dict(start_us=20, end_us=30)], 3, 4
    )
    assert dp4["article"]["system_tps"] == pytest.approx(
        result["article"]["system_tps"] * 4
    )
    assert dp4["article"]["system_tps_assumes_equal_dp_throughput"]
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


@pytest.mark.parametrize("dp_rank", [None, 2])
def test_completed_stream_does_not_remain_live(monkeypatch, dp_rank):
    import threading

    workload = load("shared_prefix_workload")

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def __iter__(self):
            yield b'data: {"meta_info":{"completion_tokens":1,"prompt_tokens":1,"cached_tokens":1}}'

    payloads = []

    def open_request(request, **kwargs):
        payloads.append(json.loads(request.data))
        return Response()

    monkeypatch.setattr(workload.urllib.request, "urlopen", open_request)
    active = workload.ActiveStreams()
    workload.stream_one(
        "http://example.invalid", [1], 1, 1, 0, threading.Lock(), active, 1, dp_rank
    )
    assert len(active) == 0
    assert active.peak == 1
    assert payloads[0].get("routed_dp_rank") == dp_rank


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


def test_runtime_gate_rejects_reduced_capacity_and_disabled_graph():
    checker = load("check_server_state")
    state = dict(
        tp_size=8,
        attn_dp_size=4,
        dp_size=1,
        ep_size=8,
        pp_size=1,
        attn_cp_size=1,
        effective_max_running_requests_per_dp=16,
        disable_radix_cache=False,
        kv_cache_dtype="fp8_e4m3",
        speculative_algorithm="EAGLE",
        speculative_num_steps=3,
        speculative_eagle_topk=1,
        speculative_num_draft_tokens=4,
        cuda_graph_config={"decode": {"backend": "full", "bs": [16]}},
        decode_cuda_graph_bs=[16],
    )
    info = {"internal_states": [dict(state) for _ in range(4)]}
    kwargs = dict(tp=8, dp=4, ep=8, batch=16, mtp=1, graph=1)
    assert checker.validate(info, **kwargs) == []
    info["internal_states"][2]["effective_max_running_requests_per_dp"] = 6
    assert any(
        "effective_max_running" in error for error in checker.validate(info, **kwargs)
    )
    info["internal_states"][2]["effective_max_running_requests_per_dp"] = 16
    info["internal_states"][2]["cuda_graph_config"] = {
        "decode": {"backend": "disabled"}
    }
    assert any("graph backend" in error for error in checker.validate(info, **kwargs))
    assert checker.validate({"internal_states": [state]}, **kwargs)


@pytest.mark.parametrize("profiled", [False, True])
def test_dp_workload_primes_and_routes_every_worker(tmp_path, monkeypatch, profiled):
    import sys
    from types import SimpleNamespace

    workload = load("shared_prefix_workload")
    calls = []
    captures = []
    monkeypatch.setattr(
        workload, "wait_for_profile_window", lambda *args: {"live_streams": 8}
    )
    monkeypatch.setattr(
        workload.subprocess,
        "run",
        lambda *args, **kwargs: captures.append((args, kwargs)),
    )

    def stream(url, ids, output, timeout, index, lock, active, target, dp_rank=None):
        calls.append((index, dp_rank))
        return dict(
            request=index,
            completion_tokens=output,
            cached_tokens=128,
            ttft_seconds=0.1,
            tpot_seconds=0.001,
            routed_dp_rank=dp_rank,
        )

    monkeypatch.setattr(workload, "stream_one", stream)
    monkeypatch.setattr(workload, "post", lambda *args: b"ok")
    tokenizer = SimpleNamespace(encode=lambda *args, **kwargs: [1, 2])
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=SimpleNamespace(
                from_pretrained=lambda *args, **kwargs: tokenizer
            )
        ),
    )

    def metrics(*args, **kwargs):
        raise RuntimeError("No metrics endpoint in CPU test")

    monkeypatch.setattr(workload.urllib.request, "urlopen", metrics)
    output = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "workload",
            "--model-path",
            "unused",
            "--input-tokens",
            "256",
            "--suffix-tokens",
            "128",
            "--num-prompts",
            "8",
            "--concurrency",
            "8",
            "--dp-size",
            "4",
            "--output",
            str(output),
        ],
    )
    if profiled:
        sys.argv.extend(["--profile-root", str(tmp_path)])
    assert workload.main() == 0
    assert calls[:4] == [(-1, rank) for rank in range(4)]
    assert sorted(calls[4:]) == [(index, index % 4) for index in range(8)]
    assert json.loads(output.read_text())["concurrency_per_dp"] == 2
    assert json.loads(output.read_text())["profiled"] is profiled
    assert len(captures) == int(profiled)
    if profiled:
        assert captures[0][1]["env"]["PROFILE_DIR"] == str(tmp_path / "profile")
        assert captures[0][1]["env"]["RUN_DIR"] == str(tmp_path / "capture")


def test_auto_profile_requires_continuous_live_window(monkeypatch):
    import threading
    from types import SimpleNamespace

    workload = load("shared_prefix_workload")
    active = workload.ActiveStreams()
    active.update((0, 1))
    active.progress.update({0: 64, 1: 64})
    clock = [0]

    def sleep(seconds):
        clock[0] += 1
        if clock[0] == 1:
            active.discard(1)
        elif clock[0] == 2:
            active.add(1)
            active.progress[1] = 64

    monkeypatch.setattr(workload.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(workload.time, "sleep", sleep)
    pending = SimpleNamespace(done=lambda: False)
    result = workload.wait_for_profile_window(
        [pending, pending], active, threading.Lock(), 2, 64, 2, 10
    )
    assert result["waited_seconds"] == 4
    assert result["stable_seconds"] == 2


def test_auto_profile_times_out_without_token_progress(monkeypatch):
    import threading
    from types import SimpleNamespace

    workload = load("shared_prefix_workload")
    active = workload.ActiveStreams()
    active.add(0)
    active.progress[0] = 1
    clock = [0]
    monkeypatch.setattr(workload.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        workload.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + 1)
    )
    with pytest.raises(TimeoutError, match="No stable profiling window"):
        workload.wait_for_profile_window(
            [SimpleNamespace(done=lambda: False)], active, threading.Lock(), 1, 64, 1, 2
        )


def test_auto_profile_preserves_request_failure():
    import threading
    from concurrent.futures import Future

    workload = load("shared_prefix_workload")
    failed = Future()
    failed.set_exception(ValueError("original request failure"))
    with pytest.raises(ValueError, match="original request failure"):
        workload.wait_for_profile_window(
            [failed], workload.ActiveStreams(), threading.Lock(), 1, 64, 5, 10
        )
