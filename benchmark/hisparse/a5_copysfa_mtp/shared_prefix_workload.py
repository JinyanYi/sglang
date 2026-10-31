#!/usr/bin/env python3
"""Measure full streaming outputs with an exact shared token-ID prefix."""

from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def post(url: str, payload: dict, timeout: int):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def repeated(values: list[int], size: int) -> list[int]:
    if not values:
        raise ValueError("Tokenizer produced an empty fixture")
    return (values * ((size + len(values) - 1) // len(values)))[:size]


def stream_one(
    url,
    ids,
    output_tokens,
    timeout,
    index,
    print_lock,
    active_requests,
    active_target,
):
    request = urllib.request.Request(
        url,
        data=json.dumps(
            {
                "input_ids": ids,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": output_tokens,
                    "ignore_eos": True,
                },
                "stream": True,
            }
        ).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    first = None
    completion = 0
    prompt = None
    cached = None
    finish = None
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw in response:
            line = raw.strip()
            if not line or line == b"data: [DONE]":
                continue
            data = json.loads(line[6:] if line.startswith(b"data: ") else line)
            meta = data.get("meta_info") or {}
            completion = max(completion, int(meta.get("completion_tokens") or 0))
            if completion and first is None:
                first = time.perf_counter()
                with print_lock:
                    print(
                        f"FIRST_STREAM_EVENT request={index} "
                        f"ttft_seconds={first - started:.6f}",
                        flush=True,
                    )
                    if index >= 0:
                        active_requests.add(index)
                        if len(active_requests) == active_target:
                            print(
                                f"ALL_STREAMS_ACTIVE count={active_target}",
                                flush=True,
                            )
            if meta.get("prompt_tokens") is not None:
                prompt = int(meta["prompt_tokens"])
            if meta.get("cached_tokens") is not None:
                cached = int(meta["cached_tokens"])
            finish = meta.get("finish_reason") or finish
    ended = time.perf_counter()
    if first is None or completion != output_tokens:
        raise RuntimeError(
            f"request {index} did not complete: first={first}, "
            f"tokens={completion}/{output_tokens}"
        )
    if prompt is not None and prompt != len(ids):
        raise RuntimeError(f"request {index} prompt={prompt}, expected={len(ids)}")
    return {
        "request": index,
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "cached_tokens": cached,
        "ttft_seconds": first - started,
        "latency_seconds": ended - started,
        "tpot_seconds": (ended - first) / max(1, completion - 1),
        "finish_reason": finish,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--input-tokens", type=int, default=65536)
    parser.add_argument("--output-tokens", type=int, default=1024)
    parser.add_argument("--suffix-tokens", type=int, default=128)
    parser.add_argument("--num-prompts", type=int, default=72)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--flush", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 < args.suffix_tokens < args.input_tokens:
        parser.error("suffix tokens must be in (0, input tokens)")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    prefix_seed = tokenizer.encode(
        "Stable shared prefix: item 17 retains its owner and checksum. ",
        add_special_tokens=False,
    )
    prefix = repeated(prefix_seed, args.input_tokens - args.suffix_tokens)

    def ids(index):
        suffix = tokenizer.encode(
            f" Request {index}: return a deterministic continuation. ",
            add_special_tokens=False,
        )
        return prefix + repeated(suffix, args.suffix_tokens)

    base = args.base_url.rstrip("/")
    if args.flush:
        post(base + "/flush_cache", {}, args.timeout)
    lock = threading.Lock()
    active_requests = set()
    active_target = min(args.num_prompts, args.concurrency)
    prime = stream_one(
        base + "/generate",
        ids(-1),
        1,
        args.timeout,
        -1,
        lock,
        active_requests,
        active_target,
    )
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(
                stream_one,
                base + "/generate",
                ids(index),
                args.output_tokens,
                args.timeout,
                index,
                lock,
                active_requests,
                active_target,
            )
            for index in range(args.num_prompts)
        ]
        rows = [future.result() for future in futures]
    misses = [
        row["request"]
        for row in rows
        if not isinstance(row["cached_tokens"], int) or row["cached_tokens"] <= 0
    ]
    if misses:
        raise RuntimeError(f"measured requests did not report radix hits: {misses}")
    elapsed = time.perf_counter() - started
    total = sum(row["completion_tokens"] for row in rows)
    result = {
        "input_tokens": args.input_tokens,
        "shared_prefix_tokens": len(prefix),
        "suffix_tokens": args.suffix_tokens,
        "output_tokens_per_request": args.output_tokens,
        "num_prompts": args.num_prompts,
        "concurrency": args.concurrency,
        "prime": prime,
        "completed": len(rows),
        "total_output_tokens": total,
        "elapsed_seconds": elapsed,
        "output_tokens_per_second": total / elapsed,
        "mean_ttft_ms": statistics.fmean(row["ttft_seconds"] for row in rows) * 1000,
        "mean_tpot_ms": statistics.fmean(row["tpot_seconds"] for row in rows) * 1000,
        "requests": rows,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "completed",
                    "total_output_tokens",
                    "elapsed_seconds",
                    "output_tokens_per_second",
                    "mean_ttft_ms",
                    "mean_tpot_ms",
                )
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
