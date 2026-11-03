#!/usr/bin/env python3
"""Check resolved A5 topology and effective scheduler limits before workloads."""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path


def validate(info, *, tp, dp, ep, batch, mtp, graph, graph_batch=None):
    errors = []
    states = info.get("internal_states", [])
    if len(states) != dp:
        errors.append(f"Expected {dp} attention-DP states, got {len(states)}")
    for index, state in enumerate(states):
        expected = {
            "tp_size": tp,
            "attn_dp_size": dp,
            "dp_size": 1,
            "ep_size": ep,
            "pp_size": 1,
            "attn_cp_size": 1,
            "effective_max_running_requests_per_dp": batch,
            "disable_radix_cache": False,
            "kv_cache_dtype": "fp8_e4m3",
        }
        for key, value in expected.items():
            if state.get(key) != value:
                errors.append(
                    f"DP state {index}: {key}={state.get(key)!r}, expected {value!r}"
                )
        algorithm = state.get("speculative_algorithm")
        if mtp:
            if algorithm not in ("NEXTN", "EAGLE"):
                errors.append(f"DP state {index}: MTP algorithm={algorithm!r}")
            for key, value in (
                ("speculative_num_steps", 3),
                ("speculative_eagle_topk", 1),
                ("speculative_num_draft_tokens", 4),
            ):
                if state.get(key) != value:
                    errors.append(f"DP state {index}: {key} must be {value}")
        elif algorithm is not None:
            errors.append(
                f"DP state {index}: unexpected speculative algorithm {algorithm!r}"
            )
        decode = (state.get("cuda_graph_config") or {}).get("decode", {})
        backend = decode.get("backend")
        if backend is None or (backend != "disabled") != bool(graph):
            errors.append(
                f"DP state {index}: decode graph backend={backend!r}, GRAPH={graph}"
            )
        buckets = state.get("decode_cuda_graph_bs") or decode.get("bs") or []
        expected_bucket = graph_batch if graph_batch is not None else batch
        if graph and (expected_bucket < batch or expected_bucket not in buckets):
            errors.append(
                f"DP state {index}: graph buckets do not contain requested bucket {expected_bucket} covering live batch {batch}"
            )
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--tp", type=int, required=True)
    parser.add_argument("--dp", type=int, required=True)
    parser.add_argument("--ep", type=int, required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--mtp", type=int, choices=(0, 1), required=True)
    parser.add_argument("--graph", type=int, choices=(0, 1), required=True)
    parser.add_argument("--graph-batch", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(
        args.base_url.rstrip("/") + "/server_info", timeout=120
    ) as response:
        info = json.load(response)
    errors = validate(
        info,
        tp=args.tp,
        dp=args.dp,
        ep=args.ep,
        batch=args.batch,
        mtp=args.mtp,
        graph=args.graph,
        graph_batch=args.graph_batch,
    )
    summary = dict(
        status="fail" if errors else "pass",
        errors=errors,
        world_size=args.tp,
        attention_dp_size=args.dp,
        attention_tp_size=args.tp // args.dp,
        ep_size=args.ep,
        requested_batch_per_dp=args.batch,
        global_requests=args.batch * args.dp,
        mtp=args.mtp,
        graph=args.graph,
        graph_bucket=args.graph_batch or args.batch,
        boundary="Resolved configuration and scheduler capacity; not model accuracy or proof of graph replay",
    )
    args.output.write_text(
        json.dumps(dict(summary=summary, server_info=info), indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
