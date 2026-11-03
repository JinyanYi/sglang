#!/usr/bin/env python3
"""Validate no-stack A5 traces and export article-style latency accounting.

Kernel duration sums are not critical paths. Exact target-layer elapsed spans
require device windows, exported separately from MindStudio if not in a trace.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import re
import statistics
from collections import defaultdict
from decimal import Decimal
from pathlib import Path


def number(value):
    return Decimal(str(value))


def category(name):
    low = name.lower().replace("_", "")
    if "lightningindexermanage" in low:
        return "LIM"
    if "lightningindexer" in low:
        return "LI"
    if "copysparseflashattention" in low or "copysfa" in low:
        return "CopySFA"
    if "sparseflashattention" in low:
        return "SFA"
    if "unidexcopy" in low or "memcpy" in low:
        return "copy"
    if "dispatch" in low:
        return "dispatch_candidates"
    if "combine" in low:
        return "combine_candidates"
    return "other"


def union_duration(intervals):
    end = None
    total = Decimal(0)
    for start, stop in sorted(intervals):
        if end is None or start > end:
            total += stop - start
        elif stop > end:
            total += stop - end
        end = max(end or stop, stop)
    return total


def trace_report(
    path, batch, min_steps, windows=None, tokens_per_round=None, dp_size=1
):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        data = json.load(stream)
    events = data if isinstance(data, list) else data["traceEvents"]
    hardware = {
        e["pid"]
        for e in events
        if e.get("name") == "process_name"
        and "Ascend Hardware" in e.get("args", {}).get("name", "")
    }
    if not hardware:
        raise ValueError("Missing Ascend Hardware process: " + str(path))
    markers = []
    for event in events:
        match = re.match(r"step\[(DECODE|VERIFY) bs=(\d+)", event.get("name", ""))
        if match and event.get("ph") == "X":
            markers.append(
                dict(
                    mode=match[1],
                    batch=int(match[2]),
                    start_us=str(event["ts"]),
                    cpu_enqueue_duration_us=event["dur"],
                )
            )
    device = [
        e
        for e in events
        if e.get("pid") in hardware
        and e.get("ph") == "X"
        and (
            e.get("args", {}).get("Task Type")
            in (
                "AI_CORE",
                "AI_VECTOR",
                "MIX_AIC",
                "MIX_AIV",
                "AICORE",
                "AIV",
                "MEMCPY_ASYNC",
                "MEMCPY",
            )
            or "memcpy" in e.get("name", "").lower()
        )
    ]
    totals, counts, operators = (
        defaultdict(Decimal),
        defaultdict(int),
        defaultdict(Decimal),
    )
    intervals = []
    for event in device:
        duration = number(event["dur"])
        totals[category(event["name"])] += duration
        counts[category(event["name"])] += 1
        operators[event["name"]] += duration
        intervals.append((number(event["ts"]), number(event["ts"]) + duration))
    valid = (
        len(markers) >= min_steps
        and all(row["batch"] == batch for row in markers)
        and bool(device)
    )
    steps = len(markers)
    result = dict(
        trace=str(path),
        hardware_process_ids=sorted(hardware),
        valid=valid,
        target_step_count=steps,
        target_step_markers=markers,
        component_kernel_counts=dict(counts),
        component_duration_sum_ms={k: float(v / 1000) for k, v in totals.items()},
        component_sum_ms_per_target_step={
            k: float(v / 1000 / steps) for k, v in totals.items()
        }
        if steps
        else {},
        device_busy_union_ms=float(union_duration(intervals) / 1000),
        operators_sum_ms={
            k: float(v / 1000)
            for k, v in sorted(operators.items(), key=lambda row: -row[1])
        },
        scope="Whole trace. LIM/CopySFA are target-only; SFA/other may include draft work. Duration sums may overlap.",
        article=dict(
            target_78_layer_ms=None,
            other_target_ms=None,
            itl_ms=None,
            tpot_ms=None,
            per_dp_tps=None,
            system_tps=None,
        ),
        missing_article_inputs=[
            "validated device target-layer windows",
            "measured output tokens per scheduling round",
        ],
    )
    # Optional windows use exact device timestamps and must describe this trace.
    if windows:
        spans = sorted(
            (number(row["start_us"]), number(row["end_us"])) for row in windows
        )
        if (
            len(spans) != steps
            or any(stop <= start for start, stop in spans)
            or any(spans[i][1] > spans[i + 1][0] for i in range(len(spans) - 1))
        ):
            raise ValueError("Target windows must match steps and must not overlap")
        target_ms = statistics.fmean(
            float((stop - start) / 1000) for start, stop in spans
        )
        itl_ms = (
            statistics.fmean(
                float((spans[i + 1][0] - spans[i][0]) / 1000)
                for i in range(len(spans) - 1)
            )
            if len(spans) > 1
            else None
        )
        target_intervals = []
        for start, stop in spans:
            for event in device:
                a, b = number(event["ts"]), number(event["ts"]) + number(event["dur"])
                if category(event["name"]) != "other" and a >= start and b <= stop:
                    target_intervals.append((a, b))
        accounted_ms = float(union_duration(target_intervals) / 1000 / steps)
        result["article"].update(
            target_78_layer_ms=target_ms,
            other_target_ms=target_ms - accounted_ms,
            itl_ms=itl_ms,
        )
        result["missing_article_inputs"] = [
            "measured output tokens per scheduling round"
        ]
        if tokens_per_round and itl_ms:
            tpot = itl_ms / tokens_per_round
            result["article"].update(
                tpot_ms=tpot,
                per_dp_tps=batch / tpot * 1000,
                system_tps=batch / tpot * 1000 * dp_size,
                system_tps_assumes_equal_dp_throughput=dp_size > 1,
            )
            result["missing_article_inputs"] = []
        result["article"]["other_definition"] = (
            "Target elapsed minus union of named component device intervals, including gaps; not subtraction of overlapping kernel sums"
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-ranks", type=int, default=8)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--min-steps", type=int, default=20)
    parser.add_argument(
        "--target-windows",
        type=Path,
        help="CSV: trace,start_us,end_us; trace is the relative path under profile_dir",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--require-joint", action="store_true")
    mode.add_argument("--no-offload", action="store_true")
    parser.add_argument("--tokens-per-round", type=float)
    parser.add_argument("--dp-size", type=int, default=1)
    args = parser.parse_args()
    if args.tokens_per_round is not None and (
        not math.isfinite(args.tokens_per_round) or args.tokens_per_round <= 0
    ):
        parser.error("tokens-per-round must be a positive measured value")
    paths = sorted(args.profile_dir.rglob("trace_view.json"))
    paths += sorted(
        path
        for path in args.profile_dir.rglob("trace_view.json.gz")
        if not path.with_suffix("").exists()
    )
    windows = defaultdict(list)
    if args.target_windows:
        with args.target_windows.open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                windows[row["trace"]].append(row)
    reports = []
    for path in paths:
        try:
            reports.append(
                trace_report(
                    path,
                    args.batch,
                    args.min_steps,
                    windows.get(path.relative_to(args.profile_dir).as_posix()),
                    args.tokens_per_round,
                    args.dp_size,
                )
            )
        except Exception as exc:
            reports.append(dict(trace=str(path), valid=False, error=str(exc)))
    if args.require_joint:
        for row in reports:
            counts = row.get("component_kernel_counts", {})
            row["valid"] = (
                row["valid"]
                and counts.get("LIM", 0) > 0
                and counts.get("CopySFA", 0) > 0
            )
    if args.no_offload:
        for row in reports:
            counts = row.get("component_kernel_counts", {})
            row["valid"] = (
                row["valid"]
                and counts.get("LIM", 0) == 0
                and counts.get("CopySFA", 0) == 0
                and counts.get("LI", 0) > 0
                and counts.get("SFA", 0) > 0
            )
    result = dict(
        valid=len(reports) == args.expected_ranks
        and all(row["valid"] for row in reports),
        expected_rank_files=args.expected_ranks,
        actual_rank_files=len(reports),
        attention_dp_size=args.dp_size,
        batch_per_dp=args.batch,
        method="Offload.txt section5.1.4; ITL/tokens_per_round=TPOT; batch/TPOT*1000=per-DP TPS; sum independent DP, do not multiply by TP",
        logical_rank_mapping="Trace paths retained; verify rank/device identities with server logs",
        traces=reports,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            dict(
                valid=result["valid"],
                actual_rank_files=len(reports),
                expected_rank_files=args.expected_ranks,
                traces=[
                    {
                        key: row[key]
                        for key in (
                            "trace",
                            "valid",
                            "target_step_count",
                            "component_kernel_counts",
                            "component_sum_ms_per_target_step",
                            "article",
                            "missing_article_inputs",
                            "error",
                        )
                        if key in row
                    }
                    for row in reports
                ],
            ),
            indent=2,
        )
    )
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
