"""Diagnostic windows must weight prefix hits and discard graph-capture work."""

import importlib.util
import json
import logging
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

SOURCE = (
    Path(__file__).parents[3]
    / "python/sglang/srt/hardware_backend/npu/sparsity_driven_kv_offload/c8_stats.py"
)
FIELDS = (
    "calls",
    "requests",
    "queries",
    "prefix_selected",
    "prefix_misses",
    "tail_selected",
    "union_misses",
    "requests_gt512",
    "fallback_calls",
    "max_union_misses",
    "union_0",
    "union_1_128",
    "union_129_256",
    "union_257_512",
    "union_513_1024",
    "union_gt1024",
    "selected",
    "route_misses",
)


def recorder(monkeypatch, interval=1):
    module = ModuleType("sgl_kernel_npu.sparsity_driven_kv_offload.c8_metadata")
    module.C8_HIT_STAT_FIELDS = FIELDS
    module.record_c8_hit_statistics = lambda **kwargs: None
    monkeypatch.setitem(sys.modules, module.__name__, module)
    state = dict(capturing=False, synchronizations=0)

    def synchronize(device):
        state["synchronizations"] += 1

    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(
            is_current_stream_capturing=lambda: state["capturing"],
            synchronize=synchronize,
        ),
        raising=False,
    )
    spec = importlib.util.spec_from_file_location("test_c8_stats", SOURCE)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    manager = SimpleNamespace(
        device="cpu", start_layer=3, layer_num=2, cache_capacity=8192, batch_capacity=12
    )
    return loaded.C8HitRateRecorder(
        manager, interval=interval, split_query=False, fused_ring_tail=False
    ), state


def test_window_uses_weighted_prefix_hits_and_max_union(monkeypatch, caplog):
    stats, state = recorder(monkeypatch)
    stats.maybe_report()
    for layer, values in enumerate(
        (
            dict(
                calls=2,
                requests=2,
                prefix_selected=8,
                prefix_misses=4,
                tail_selected=9,
                max_union_misses=512,
                fallback_calls=1,
            ),
            dict(
                calls=1,
                requests=1,
                prefix_selected=2,
                prefix_misses=0,
                tail_selected=3,
                max_union_misses=513,
            ),
        )
    ):
        for name, value in values.items():
            stats.counters[layer, FIELDS.index(name)] = value
    with caplog.at_level(logging.INFO):
        stats.maybe_report()
    line = next(r.message for r in caplog.records if "A5_C8_CACHE_STATS" in r.message)
    report = json.loads(line.split("A5_C8_CACHE_STATS ", 1)[1])
    assert report["totals"]["prefix_hit_rate"] == pytest.approx(0.6)
    assert report["totals"]["max_union_misses"] == 513
    assert report["totals"]["fallback_call_fraction"] == pytest.approx(1 / 3)
    assert report["totals"]["tail_selected"] == 12
    assert [row["layer"] for row in report["layers"]] == [3, 4]
    assert not stats.counters.count_nonzero()
    assert state["synchronizations"] == 1


def test_capture_counts_are_discarded_before_reporting(monkeypatch, caplog):
    stats, state = recorder(monkeypatch)
    stats.maybe_report()
    state["capturing"] = True
    plan = SimpleNamespace(
        **{
            name: None
            for name in (
                "source_ids",
                "device_slots",
                "topk_misses",
                "miss_counts",
                "prefix_lens",
                "seq_lens",
                "valid_rows",
                "sparse_rows",
            )
        }
    )
    stats.record(0, plan)
    stats.counters.fill_(99)
    stats.maybe_report()
    assert state["synchronizations"] == 0
    state["capturing"] = False
    with caplog.at_level(logging.INFO):
        stats.maybe_report()
    assert not stats.counters.count_nonzero()
    assert not caplog.records


@pytest.mark.parametrize("interval", [0, 100000])
def test_reporting_window_cannot_overflow_device_counters(monkeypatch, interval):
    with pytest.raises(ValueError, match="INT32"):
        recorder(monkeypatch, interval)
