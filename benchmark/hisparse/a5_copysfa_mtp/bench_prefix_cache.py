#!/usr/bin/env python3
"""Prime every DP, then supply one fixed token-ID wave to native bench_serving."""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from pathlib import Path

from shared_prefix_workload import ActiveStreams, post, repeated, stream_one


def validate_result(result, count, input_tokens, output_tokens, expected_cached):
    """Reject partial completions and any request missing the cacheable prefix."""
    errors = []
    if result.get("completed") != count:
        errors.append(f"completed={result.get('completed')}, expected={count}")
    for field, expected in (
        ("input_lens", input_tokens),
        ("output_lens", output_tokens),
        ("cached_tokens", expected_cached),
    ):
        values = result.get(field, [])
        if len(values) != count:
            errors.append(f"{field}: got {len(values)} rows, expected={count}")
        for index, value in enumerate(values):
            if not isinstance(value, int) or value != expected:
                errors.append(f"{field}[{index}]={value}, expected={expected}")
    if any(result.get("errors", [])):
        errors.append("bench_serving reported request errors")
    return errors


def run_native_benchmark(args, token_ids):
    """Adapt only dataset/routing; preserve native timing, SSE parsing and metrics."""
    from sglang.benchmark import serving

    original_dataset = serving.get_dataset
    original_argv = sys.argv
    original_ci = os.environ.pop("SGLANG_IS_IN_CI", None)

    def dataset(bench_args, tokenizer, model_id):
        bench_args.dataset_name = "hisparse-full-prefix-single-wave"
        return [
            serving.DatasetRow(
                prompt=token_ids,
                prompt_len=len(token_ids),
                output_len=args.output_tokens,
                extra_request_body={"routed_dp_rank": index % args.dp_size},
            )
            for index in range(args.concurrency)
        ]

    # cli_main supplies the installed benchmark's complete argument defaults.
    # No warmup or flush is allowed between explicit priming and measurement.
    argv = [
        "sglang.benchmark.serving",
        "--backend",
        "sglang",
        "--base-url",
        args.base_url,
        "--model",
        args.model_path,
        "--tokenizer",
        args.model_path,
        "--dataset-name",
        "random",
        "--tokenize-prompt",
        "--num-prompts",
        str(args.concurrency),
        "--max-concurrency",
        str(args.concurrency),
        "--request-rate",
        "inf",
        "--random-input-len",
        str(args.input_tokens),
        "--random-output-len",
        str(args.output_tokens),
        "--random-range-ratio",
        "1",
        "--warmup-requests",
        "0",
        "--cache-report",
        "--output-details",
        "--output-file",
        str(args.output),
    ]
    args.output.with_name("bench-serving.argv.json").write_text(
        json.dumps(argv, indent=2) + "\n"
    )
    try:
        serving.get_dataset = dataset
        sys.argv = argv
        serving.cli_main()
    finally:
        serving.get_dataset = original_dataset
        sys.argv = original_argv
        if original_ci is not None:
            os.environ["SGLANG_IS_IN_CI"] = original_ci


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--input-tokens", type=int, default=40000)
    parser.add_argument("--output-tokens", type=int, default=1024)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--dp-size", type=int, default=4)
    parser.add_argument("--page-size", type=int, default=128)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.dp_size < 1 or args.concurrency < 1 or args.concurrency % args.dp_size:
        parser.error("concurrency must be positive and divisible by dp-size")
    if (
        args.page_size < 1
        or args.input_tokens <= args.page_size
        or args.output_tokens < 2
    ):
        parser.error("require input > page-size > 0 and output >= 2")
    if args.output.exists():
        parser.error("output already exists; use a fresh RUN_DIR")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    seed = tokenizer.encode(
        "Stable shared prefix: item 17 retains its owner and checksum. ",
        add_special_tokens=False,
    )
    token_ids = repeated(seed, args.input_tokens)
    expected_cached = (args.input_tokens - 1) // args.page_size * args.page_size
    (args.output.parent / "input_ids.json").write_text(json.dumps(token_ids) + "\n")
    lock = threading.Lock()
    active = ActiveStreams()
    base = args.base_url.rstrip("/")
    post(base + "/flush_cache", {}, args.timeout)

    def probe(rank):
        return stream_one(
            base + "/generate", token_ids, 1, args.timeout, -1, lock, active, 0, rank
        )

    prime = [probe(rank) for rank in range(args.dp_size)]
    probes = [probe(rank) for rank in range(args.dp_size)]
    manifest = dict(
        protocol="one measured wave; native bench_serving metrics; no profiler",
        num_prompts=args.concurrency,
        concurrency=args.concurrency,
        attention_dp_size=args.dp_size,
        requests_per_dp=args.concurrency // args.dp_size,
        input_tokens=args.input_tokens,
        output_tokens=args.output_tokens,
        page_size=args.page_size,
        expected_cached_tokens_per_request=expected_cached,
        required_recompute_tokens=args.input_tokens - expected_cached,
        kv_offload=os.environ.get("KV_OFFLOAD"),
        prime=prime,
        probes=probes,
        routing=[index % args.dp_size for index in range(args.concurrency)],
        timing_scope="Native whole-wave serving duration, including TTFT and drain",
        cache_scope="Every request must hit all page-aligned cacheable input tokens",
    )
    (args.output.parent / "workload.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    if any(row["cached_tokens"] != expected_cached for row in probes):
        raise RuntimeError(
            "DP cache probe failed; inspect workload.json before retrying"
        )
    print(
        f"SINGLE_WAVE requests={args.concurrency} per_dp={args.concurrency // args.dp_size} "
        f"expected_cached={expected_cached}/{args.input_tokens} profiler=False",
        flush=True,
    )
    run_native_benchmark(args, token_ids)
    result = json.loads(args.output.read_text().strip().splitlines()[-1])
    errors = validate_result(
        result, args.concurrency, args.input_tokens, args.output_tokens, expected_cached
    )
    validation = dict(valid=not errors, errors=errors)
    (args.output.parent / "cache-validation.json").write_text(
        json.dumps(validation, indent=2) + "\n"
    )
    # Values below are copied from native bench_serving without recomputation.
    print(
        "NATIVE_BENCH_SERVING_RESULT "
        + json.dumps(
            {
                key: result.get(key)
                for key in (
                    "completed",
                    "duration",
                    "output_throughput",
                    "mean_tpot_ms",
                    "median_tpot_ms",
                    "mean_ttft_ms",
                    "accept_length",
                )
            }
        ),
        flush=True,
    )
    print("CACHE_VALIDATION " + json.dumps(validation), flush=True)
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
