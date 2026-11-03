#!/usr/bin/env python3
"""Offline article-style NPU report with explicit graph and token assumptions."""

import argparse
import bisect
import gzip
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from analyze_profile import category, number, union_duration

COMPONENTS = (
    "LIM",
    "LI",
    "CopySFA",
    "SFA",
    "attention_vector_aux",
    "kv_copy_candidates",
    "memcpy_unattributed",
    "dispatch_candidates",
    "combine_candidates",
    "other",
)


def component(name, task_type=None):
    label = category(name)
    if label in {"CopySFA", "SFA"} and task_type is not None:
        # A5 exports vector helpers (or the vector side of a mixed kernel)
        # with attention-related names as well. They are not extra layers.
        if str(task_type).upper() not in {"AI_CORE", "MIX_AIC", "AICORE"}:
            return "attention_vector_aux"
    if label == "copy":
        return (
            "kv_copy_candidates"
            if "unidexcopy" in name.lower().replace("_", "")
            else "memcpy_unattributed"
        )
    return label


def read_json(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def is_compute(event):
    task = str(event.get("args", {}).get("Task Type", "")).upper()
    return (
        task
        in {
            "AI_CORE",
            "AI_VECTOR",
            "AI_VECTOR_CORE",
            "MIX_AIC",
            "MIX_AIV",
            "AICORE",
            "AIV",
            "MEMCPY_ASYNC",
            "MEMCPY",
            "SDMA_SQE",
            "COMMUNICATION",
        }
        or "memcpy" in event.get("name", "").lower()
    )


def signature(event):
    args = event["args"]
    return (
        str(args["Task Id"]),
        str(args.get("Subtask Id")),
        str(args.get("Task Type")),
        event["name"],
    )


def event_component(event):
    return component(event["name"], event.get("args", {}).get("Task Type", ""))


def attention_records(rows):
    grouped = defaultdict(list)
    for row in rows:
        if is_compute(row) and category(row["name"]) in {"CopySFA", "SFA"}:
            grouped[(row["name"], row.get("args", {}).get("Task Type"))].append(row)
    return [
        dict(
            name=name,
            task_type=task,
            count=len(items),
            duration_sum_ms=float(
                sum((number(item["dur"]) for item in items), number(0)) / 1000
            ),
            samples=[dict(tid=item["tid"], args=item["args"]) for item in items[:2]],
        )
        for (name, task), items in grouped.items()
    ]


def graph_rounds(events, steps):
    """Split each stream by a repeated task signature; never use CPU spans."""
    lanes = defaultdict(list)
    for event in events:
        if "Task Id" not in event.get("args", {}):
            raise ValueError("Graph device event lacks Task Id")
        lanes[str(event["tid"])].append(event)
    rounds = [[] for _ in range(steps)]
    anchors = {}
    for lane, rows in lanes.items():
        rows.sort(key=lambda event: number(event["ts"]))
        counts = Counter(signature(event) for event in rows)
        # Every stream must expose one complete occurrence of its first task
        # per target round. Partial/ambiguous captures require inspection.
        first = signature(rows[0])
        if counts[first] != steps:
            raise ValueError(
                f"stream {lane}: first task occurs {counts[first]} times, expected {steps}"
            )
        starts = [number(event["ts"]) for event in rows if signature(event) == first]
        if len(set(starts)) != steps:
            raise ValueError(f"stream {lane}: duplicate anchor timestamps")
        anchors[lane] = dict(signature=first, starts_us=[str(t) for t in starts])
        for event in rows:
            index = bisect.bisect_right(starts, number(event["ts"])) - 1
            rounds[index].append(event)
    return rounds, anchors


def inspect_trace(path, batch, min_steps, layers, assumed_tokens, measured_tokens, dp):
    data = read_json(path)
    events = data if isinstance(data, list) else data["traceEvents"]
    hardware = {
        event["pid"]
        for event in events
        if event.get("name") == "process_name"
        and "Ascend Hardware" in event.get("args", {}).get("name", "")
    }
    markers = []
    for event in events:
        match = re.match(r"step\[(DECODE|VERIFY) bs=(\d+)", event.get("name", ""))
        if match and event.get("ph") == "X":
            markers.append(dict(mode=match[1], batch=int(match[2])))
    device = [
        event
        for event in events
        if event.get("pid") in hardware and event.get("ph") == "X"
    ]
    models = defaultdict(list)
    for event in device:
        model = str(event.get("args", {}).get("Model Id", "unknown"))
        if model not in {"unknown", "-1", "4294967295"}:
            models[model].append(event)
    steps = len(markers)
    model_stats = {}
    for model, rows in models.items():
        counts = Counter(event_component(row) for row in rows if is_compute(row))
        model_stats[model] = dict(
            events=len(rows),
            counts=dict(counts),
            task_types=dict(
                Counter(row.get("args", {}).get("Task Type") for row in rows)
            ),
            top_names=Counter(row["name"] for row in rows).most_common(10),
            attention_records=attention_records(rows),
        )
    result = dict(
        trace=str(path),
        status="incomplete",
        target_step_count=steps,
        models=model_stats,
        device_schema_samples=[
            {key: row.get(key) for key in ("name", "pid", "tid", "args")}
            for row in device[:3]
        ],
        pure_78_layer_ms=None,
        scope="Target graph device span is a proxy; embedding/lm_head or metadata may be included.",
    )
    errors = []
    if not hardware:
        errors.append("Missing Ascend Hardware process")
    if steps < max(2, min_steps) or any(row["batch"] != batch for row in markers):
        errors.append(f"Expected at least {min_steps} target markers, all BS={batch}")
    candidates = [
        model
        for model, stats in model_stats.items()
        if steps > 0
        and stats["counts"].get("CopySFA", 0) + stats["counts"].get("SFA", 0)
        == layers * steps
    ]
    if len(candidates) != 1:
        errors.append(
            f"Expected one target model with {layers} attention calls/round; candidates={candidates}"
        )
    if errors:
        result["errors"] = errors
        return result
    model = candidates[0]
    result["target_model_id"] = model
    try:
        rounds, anchors = graph_rounds(models[model], steps)
        windows = []
        sums = defaultdict(list)
        residuals = []
        for index, rows in enumerate(rounds):
            compute = [row for row in rows if is_compute(row)]
            counts = Counter(event_component(row) for row in compute)
            if counts["CopySFA"] + counts["SFA"] != layers:
                raise ValueError(
                    f"round {index}: attention count != {layers}: {dict(counts)}"
                )
            if counts["LI"] + counts["LIM"] != 21:
                raise ValueError(f"round {index}: LI/LIM count != 21: {dict(counts)}")
            start = min(number(row["ts"]) for row in rows)
            stop = max(number(row["ts"]) + number(row["dur"]) for row in rows)
            if stop <= start or (windows and start < number(windows[-1]["end_us"])):
                raise ValueError(
                    f"round {index}: invalid or overlapping target graph windows"
                )
            named = []
            for label in COMPONENTS:
                selected = [row for row in compute if event_component(row) == label]
                sums[label].append(
                    float(
                        sum((number(row["dur"]) for row in selected), number(0)) / 1000
                    )
                )
                if label not in (
                    "other",
                    "memcpy_unattributed",
                    "attention_vector_aux",
                ):
                    named.extend(
                        (number(row["ts"]), number(row["ts"]) + number(row["dur"]))
                        for row in selected
                    )
            residuals.append(float((stop - start - union_duration(named)) / 1000))
            windows.append(
                dict(
                    start_us=str(start),
                    end_us=str(stop),
                    elapsed_ms=float((stop - start) / 1000),
                    counts=dict(counts),
                )
            )
        itl = statistics.fmean(
            float((number(right["start_us"]) - number(left["start_us"])) / 1000)
            for left, right in zip(windows, windows[1:])
        )
        if itl <= 0:
            raise ValueError("Nonpositive device ITL")
        result.update(
            status="graph_proxy_validated",
            windows=windows,
            stream_anchors=anchors,
            target_graph_ms=statistics.fmean(row["elapsed_ms"] for row in windows),
            device_itl_ms=itl,
            itl_intervals=len(windows) - 1,
            target_component_sum_ms={
                name: statistics.fmean(values) for name, values in sums.items()
            },
            target_graph_residual_ms=statistics.fmean(residuals),
            per_round_component_counts=windows[0]["counts"],
            component_scope="Selected target graph only; kernel sums may overlap. Residual subtracts the named article-candidate interval union, retaining unattributed MEMCPY in the residual.",
            article_assumption=dict(
                tokens_per_round=assumed_tokens,
                tpot_ms=itl / assumed_tokens,
                per_dp_tps=batch * assumed_tokens / itl * 1000,
                balanced_system_tps_estimate=dp * batch * assumed_tokens / itl * 1000,
            ),
        )
        if measured_tokens is not None:
            result["measured_acceptance_formula"] = dict(
                tokens_per_round=measured_tokens,
                tpot_ms=itl / measured_tokens,
                per_dp_tps=batch * measured_tokens / itl * 1000,
                acceptance_source="User-supplied; must refer to this profiling window",
            )
    except (ValueError, KeyError) as exc:
        result["errors"] = [str(exc)]
    return result


def text_report(result):
    lines = [
        "A5 ARTICLE-STYLE REPORT",
        f"case: {result['case_root']}",
        "comparison baseline: same-version SGLang without offload (not vLLM)",
        f"traces: {len(result['traces'])}/{result['expected_ranks']}; status: {result['status']}",
        f"attention DP={result['dp_size']}, TP per DP={result['expected_ranks'] // result['dp_size']}, BS per DP={result['batch']}",
        f"ARTICLE ASSUMPTION: {result['assumed_tokens']} output tokens/request/round; not measured acceptance.",
        "Pure 78-layer latency: unresolved; target graph span below may include embedding/lm_head/metadata.",
        "ITL: interval between consecutive target graph device starts; CPU enqueue durations are not used.",
        "TPS formula: BS/TPOT*1000 per DP; never sum all TP ranks as independent replicas.",
    ]
    perf = result.get("unprofiled_performance")
    if perf:
        lines.append("\nUNPROFILED CLIENT PERFORMANCE (separate from profiling)")
        for key in (
            "completed",
            "num_prompts",
            "input_tokens",
            "output_tokens_per_request",
            "concurrency",
            "shared_prefix_tokens",
            "output_tokens_per_second",
            "mean_ttft_ms",
            "mean_tpot_ms",
        ):
            lines.append(f"{key}: {perf.get(key)}")
        if perf.get("shared_prefix_tokens", 0) > 0:
            lines.append(
                "METHOD GAP: shared-prefix radix workload; not proof of BS*S independent-KV capacity as in article Table 7."
            )
    lines.append("\nPER-TRACE TARGET GRAPH RESULTS (duration in ms)")
    for trace in result["traces"]:
        lines.append(f"\ntrace: {trace['trace']}")
        if trace["status"] != "graph_proxy_validated":
            lines.append("INCOMPLETE: " + "; ".join(trace.get("errors", [])))
            lines.append(
                "device_schema_samples="
                + json.dumps(trace.get("device_schema_samples", []))
            )
            for model, stats in trace.get("models", {}).items():
                lines.append(
                    f"model={model} counts={stats['counts']} task_types={stats['task_types']}"
                )
                lines.append(f"top_names={stats['top_names']}")
                lines.append(
                    "attention_records="
                    + json.dumps(stats.get("attention_records", []))
                )
            continue
        lines.append(
            f"model={trace['target_model_id']} rounds={trace['target_step_count']} ITL_intervals={trace['itl_intervals']}"
        )
        lines.append(
            "calls/round: "
            + json.dumps(trace["per_round_component_counts"], sort_keys=True)
        )
        lines.append(
            "attention_records="
            + json.dumps(
                trace["models"][trace["target_model_id"]].get("attention_records", [])
            )
        )
        lines.append(
            f"target_graph_proxy_ms={trace['target_graph_ms']:.6f} device_ITL_ms={trace['device_itl_ms']:.6f}"
        )
        lines.append(
            "target_kernel_sums_ms: "
            + " ".join(
                f"{key}={value:.6f}"
                for key, value in trace["target_component_sum_ms"].items()
            )
        )
        lines.append(
            f"graph_residual_including_gaps_ms={trace['target_graph_residual_ms']:.6f}"
        )
        assumed = trace["article_assumption"]
        lines.append(
            f"ASSUMED_TPOT_ms={assumed['tpot_ms']:.6f} per_DP_TPS={assumed['per_dp_tps']:.3f} balanced_system_TPS_ESTIMATE={assumed['balanced_system_tps_estimate']:.3f}"
        )
    complete = [
        row for row in result["traces"] if row["status"] == "graph_proxy_validated"
    ]
    if complete:
        lines.append("\nRANK MEANS (TP ranks are not independent request replicas)")
        for key in ("target_graph_ms", "device_itl_ms", "target_graph_residual_ms"):
            values = [row[key] for row in complete]
            lines.append(
                f"{key}: mean={statistics.fmean(values):.6f} min={min(values):.6f} max={max(values):.6f}"
            )
        for key in COMPONENTS:
            values = [row["target_component_sum_ms"][key] for row in complete]
            lines.append(f"target_{key}_sum_ms_mean={statistics.fmean(values):.6f}")
    payload = result.get("profile_request_payload", {})
    lines.append("\nPROFILE SETTINGS: " + json.dumps(payload, sort_keys=True))
    if "with_stack" not in payload:
        lines.append("WARNING: no saved request receipt; stack setting is unverified.")
    for error in result.get("configuration_errors", []):
        lines.append("CONFIGURATION ERROR: " + error)
    lines.extend(
        [
            "\nBOUNDARIES",
            "No no-offload baseline measured by this command; no speedup claim.",
            "Model/stream task grouping is structural evidence, not pure layer-boundary instrumentation.",
            "Dispatch/combine categories are name-based candidates; kernel sums may overlap.",
            "Attention layer counts use AI_CORE/MIX_AIC primary tasks; vector attention-related records are retained separately, not divided by two or discarded.",
            "Unattributed MEMCPY is reported separately; it is not assumed to be historical-KV H2D loading.",
            "Measured acceptance for this exact profiling window is unavailable unless explicitly supplied.",
            "Weight/activation/KV/host memory attribution is not inferred from kernel durations.",
            "This command only reads saved files; it does not send requests or start profiling.",
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_root", type=Path)
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--expected-ranks", type=int, default=8)
    parser.add_argument("--dp-size", type=int, default=4)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--min-steps", type=int, default=20)
    parser.add_argument("--layers", type=int, default=78)
    parser.add_argument("--assumed-tokens", type=float, default=3.0)
    parser.add_argument("--measured-tokens-per-round", type=float)
    args = parser.parse_args()
    values = [
        args.expected_ranks,
        args.dp_size,
        args.batch,
        args.min_steps,
        args.layers,
        args.assumed_tokens,
    ]
    if args.measured_tokens_per_round is not None:
        values.append(args.measured_tokens_per_round)
    if (
        any(not math.isfinite(value) or value <= 0 for value in values)
        or args.expected_ranks % args.dp_size
    ):
        parser.error(
            "Positive finite sizes/token counts and ranks divisible by DP are required"
        )
    profile = args.profile_dir or args.case_root / "04-profile/profile"
    paths = sorted(profile.rglob("trace_view.json"))
    paths += sorted(
        path
        for path in profile.rglob("trace_view.json.gz")
        if not path.with_suffix("").exists()
    )
    result = dict(
        case_root=str(args.case_root),
        expected_ranks=args.expected_ranks,
        dp_size=args.dp_size,
        batch=args.batch,
        assumed_tokens=args.assumed_tokens,
        traces=[],
    )
    for path in paths:
        print(f"Reading {path}", flush=True)
        try:
            result["traces"].append(
                inspect_trace(
                    path,
                    args.batch,
                    args.min_steps,
                    args.layers,
                    args.assumed_tokens,
                    args.measured_tokens_per_round,
                    args.dp_size,
                )
            )
        except Exception as exc:
            result["traces"].append(
                dict(trace=str(path), status="incomplete", errors=[str(exc)])
            )
    perf = args.case_root / "03-perf/serving-perf.json"
    if perf.exists():
        result["unprofiled_performance"] = read_json(perf)
    receipt = args.case_root / "04-profile/capture/profile-request.json"
    if receipt.exists():
        result["profile_request_payload"] = read_json(receipt).get("payload", {})
    result["configuration_errors"] = []
    if result.get("profile_request_payload", {}).get("with_stack") is True:
        result["configuration_errors"].append(
            "with_stack=true: not the requested no-stack capture"
        )
    state_file = args.case_root / "04-profile/server-state.json"
    if state_file.exists():
        state = read_json(state_file)["summary"]
        result["server_configuration_summary"] = state
        for key, expected in (
            ("world_size", args.expected_ranks),
            ("attention_dp_size", args.dp_size),
            ("requested_batch_per_dp", args.batch),
        ):
            if state.get(key) != expected:
                result["configuration_errors"].append(
                    f"{key}={state.get(key)}, expected {expected}"
                )
    result["status"] = (
        "graph_proxy_validated"
        if len(paths) == args.expected_ranks
        and not result["configuration_errors"]
        and all(row["status"] == "graph_proxy_validated" for row in result["traces"])
        else "incomplete"
    )
    output = args.output_dir or args.case_root / "06-article-report"
    output.mkdir(parents=True, exist_ok=True)
    (output / "article-report.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    report = text_report(result)
    (output / "REPORT.txt").write_text(report, encoding="utf-8")
    print(report, end="")
    print(f"REPORT: {output / 'REPORT.txt'}", flush=True)
    return 0 if result["status"] == "graph_proxy_validated" else 2


if __name__ == "__main__":
    raise SystemExit(main())
