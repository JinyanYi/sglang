#!/usr/bin/env python3
"""Send repeated real generation requests to a PP output-relay test server."""

import argparse
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def generate(url: str, case_id: int, max_new_tokens: int, timeout: int) -> dict:
    payload = {
        "text": (
            f"Case {case_id}: Write a short, original explanation of why "
            "the sky looks blue during the day."
        ),
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": max_new_tokens,
            "ignore_eos": True,
        },
    }
    request = urllib.request.Request(
        f"{url}/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.load(response)
            status = response.status
    except urllib.error.HTTPError as exc:
        body = exc.read(2000).decode(errors="replace")
        raise RuntimeError(f"request {case_id}: HTTP {exc.code}: {body}") from exc
    elapsed = time.monotonic() - started
    if status != 200 or not isinstance(result, dict):
        raise AssertionError(f"request {case_id}: bad response: {result!r}")
    meta = result.get("meta_info") or {}
    completed = meta.get("completion_tokens")
    if completed != max_new_tokens:
        raise AssertionError(
            f"request {case_id}: expected {max_new_tokens} tokens, "
            f"got {completed}; meta={meta!r}"
        )
    finish_reason = meta.get("finish_reason") or {}
    if finish_reason.get("type") == "abort":
        raise AssertionError(f"request {case_id}: aborted: {meta!r}")
    return {
        "case_id": case_id,
        "seconds": round(elapsed, 3),
        "completion_tokens": completed,
        "finish_reason": finish_reason,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sequential", type=int, default=8)
    parser.add_argument("--concurrent", type=int, default=12)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--request-timeout", type=int, default=90)
    args = parser.parse_args()
    if (
        min(
            args.sequential,
            args.concurrent,
            args.workers,
            args.max_new_tokens,
            args.request_timeout,
        )
        < 1
    ):
        parser.error("all counts and timeouts must be positive")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    total = args.sequential + args.concurrent
    with args.out.open("w", encoding="utf-8") as log:
        for case_id in range(args.sequential):
            item = generate(
                args.url, case_id, args.max_new_tokens, args.request_timeout
            )
            item["phase"] = "sequential"
            log.write(json.dumps(item) + "\n")
            log.flush()
            print(
                f"PASS sequential {case_id + 1}/{args.sequential}: {item}", flush=True
            )

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    generate,
                    args.url,
                    case_id,
                    args.max_new_tokens,
                    args.request_timeout,
                ): case_id
                for case_id in range(args.sequential, total)
            }
            for future in as_completed(futures):
                item = future.result()
                item["phase"] = "concurrent"
                log.write(json.dumps(item) + "\n")
                log.flush()
                print(
                    f"PASS concurrent {item['case_id'] - args.sequential + 1}/"
                    f"{args.concurrent}: {item}",
                    flush=True,
                )
    print(f"PASS all {total} real generation requests", flush=True)


if __name__ == "__main__":
    main()
