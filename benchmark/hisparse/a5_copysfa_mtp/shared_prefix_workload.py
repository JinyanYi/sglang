#!/usr/bin/env python3
"""Measure full streaming outputs with an exact shared token-ID prefix."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
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


class ActiveStreams(set):
    peak = 0

    def __init__(self):
        super().__init__()
        self.progress = {}

    def add(self, item):
        super().add(item)
        self.peak = max(self.peak, len(self))

    def discard(self, item):
        super().discard(item)
        self.progress.pop(item, None)


def wait_for_profile_window(
    futures, active, lock, target, min_tokens, seconds, timeout
):
    """Require concurrent live streams with progress, then validate the real trace."""
    started = time.monotonic()
    stable_since = None
    while True:
        for future in futures:
            if future.done():
                future.result()  # Preserve the original request error, if any.
                raise RuntimeError("A request finished before the profiling window")
        now = time.monotonic()
        with lock:
            live = len(active)
            least_tokens = min((active.progress.get(i, 0) for i in active), default=0)
        if live == target and least_tokens >= min_tokens:
            if stable_since is None:
                stable_since = now
            if now - stable_since >= seconds:
                return dict(
                    live_streams=live,
                    least_completion_tokens=least_tokens,
                    stable_seconds=now - stable_since,
                    waited_seconds=now - started,
                    boundary="Client trigger only; per-rank full batch must pass trace validation",
                )
        else:
            stable_since = None
        if now - started >= timeout:
            raise TimeoutError(
                f"No stable profiling window: live={live}/{target}, min_tokens={least_tokens}/{min_tokens}"
            )
        time.sleep(0.1)


def stream_one(
    url,
    ids,
    output_tokens,
    timeout,
    index,
    print_lock,
    active_requests,
    active_target,
    dp_rank=None,
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
                **({"routed_dp_rank": dp_rank} if dp_rank is not None else {}),
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
    try:
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
                if index >= 0 and completion:
                    with print_lock:
                        active_requests.progress[index] = completion
    finally:
        with print_lock:
            active_requests.discard(index)
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
        "routed_dp_rank": dp_rank,
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
    parser.add_argument("--dp-size", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--flush", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile-root", type=Path)
    parser.add_argument("--profile-steps", type=int, default=20)
    parser.add_argument("--profile-warmup-tokens", type=int, default=64)
    parser.add_argument("--profile-warmup-seconds", type=float, default=5)
    parser.add_argument("--profile-window-timeout", type=float, default=900)
    args = parser.parse_args()
    if args.dp_size < 1 or args.concurrency < 1 or args.concurrency % args.dp_size:
        parser.error("concurrency must be positive and divisible by dp-size")
    if args.num_prompts < args.concurrency or args.num_prompts % args.dp_size:
        parser.error("num-prompts must cover concurrency and be divisible by dp-size")
    if not 0 < args.suffix_tokens < args.input_tokens:
        parser.error("suffix tokens must be in (0, input tokens)")
    if args.profile_root and (
        args.num_prompts != args.concurrency
        or args.profile_steps < 1
        or not 0 < args.profile_warmup_tokens < args.output_tokens
        or args.profile_warmup_seconds < 0
        or args.profile_window_timeout <= 0
    ):
        parser.error(
            "auto profiling requires one full wave, positive steps/timeout, and warmup below output length"
        )

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
    active_requests = ActiveStreams()
    active_target = min(args.num_prompts, args.concurrency)
    # Radix caches belong to individual attention-DP workers. Prime every
    # worker and route measured requests evenly to those exact workers.
    prime = [
        stream_one(
            base + "/generate",
            ids(-1),
            1,
            args.timeout,
            -1,
            lock,
            active_requests,
            active_target,
            rank if args.dp_size > 1 else None,
        )
        for rank in range(args.dp_size)
    ]

    def metrics_snapshot(name):
        try:
            with urllib.request.urlopen(base + "/metrics", timeout=30) as response:
                (args.output.parent / name).write_bytes(response.read())
        except Exception as exc:
            (args.output.parent / (name + ".error")).write_text(str(exc))

    metrics_snapshot("metrics-before.txt")
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
                index % args.dp_size if args.dp_size > 1 else None,
            )
            for index in range(args.num_prompts)
        ]
        if args.profile_root:
            args.profile_root.mkdir(parents=True, exist_ok=True)
            trigger = wait_for_profile_window(
                futures,
                active_requests,
                lock,
                active_target,
                args.profile_warmup_tokens,
                args.profile_warmup_seconds,
                args.profile_window_timeout,
            )
            (args.profile_root / "profile-trigger.json").write_text(
                json.dumps(trigger, indent=2) + "\n"
            )
            print("AUTO_PROFILE_WINDOW " + json.dumps(trigger), flush=True)
            # Reuse the normal no-stack request, automatic step stop, rank/batch
            # gate and timeout handling. The server is managed by the user.
            env = dict(
                os.environ,
                RUN_DIR=str(args.profile_root / "capture"),
                PROFILE_DIR=str(args.profile_root / "profile"),
                PROFILE_STEPS=str(args.profile_steps),
            )
            subprocess.run(
                ["bash", str(Path(__file__).with_name("60_profile.sh")), "start"],
                env=env,
                check=True,
            )
            print(
                f"AUTO_PROFILE_VALIDATED summary={args.profile_root / 'capture/profile-summary.json'}",
                flush=True,
            )
        rows = [future.result() for future in futures]
    misses = [
        row["request"]
        for row in rows
        if not isinstance(row["cached_tokens"], int) or row["cached_tokens"] <= 0
    ]
    if misses:
        raise RuntimeError(f"measured requests did not report radix hits: {misses}")
    elapsed = time.perf_counter() - started
    metrics_snapshot("metrics-after.txt")
    total = sum(row["completion_tokens"] for row in rows)
    result = {
        "input_tokens": args.input_tokens,
        "shared_prefix_tokens": len(prefix),
        "suffix_tokens": args.suffix_tokens,
        "output_tokens_per_request": args.output_tokens,
        "num_prompts": args.num_prompts,
        "concurrency": args.concurrency,
        "attention_dp_size": args.dp_size,
        "concurrency_per_dp": args.concurrency // args.dp_size,
        "routing": "fixed round robin across primed attention-DP workers",
        "peak_live_client_streams": active_requests.peak,
        "client_streams_are_not_proof_of_server_batch_size": True,
        "profiled": args.profile_root is not None,
        "timing_scope": "Includes profiler overhead; use 50_perf.sh for unprofiled performance"
        if args.profile_root
        else "Unprofiled client requests",
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
