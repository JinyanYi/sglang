#!/usr/bin/env python3
"""Validate a known 64K needle answer and an observed hot radix hit."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

EXPECTED = "73194628"
MARKER = "<A5_HANDOFF_FILLER_73194628>"


def repeated(values: list[int], size: int) -> list[int]:
    if not values:
        raise ValueError("Tokenizer produced an empty filler")
    return (values * ((size + len(values) - 1) // len(values)))[:size]


def frozen_ids(model_path: str, length: int) -> tuple[list[int], dict]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    content = (
        f"The document identifier is {EXPECTED}. Remember it exactly.\n"
        f"{MARKER}\n"
        "Question: What is the document identifier? Reply with the identifier only."
    )
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    before, after = rendered.split(MARKER)
    before_ids = tokenizer.encode(before, add_special_tokens=False)
    after_ids = tokenizer.encode(after, add_special_tokens=False)
    filler_size = length - len(before_ids) - len(after_ids)
    if filler_size <= 0:
        raise ValueError("Chat template is too large for the requested fixture")
    filler_seed = tokenizer.encode(
        "The archive contains ordinary records that do not change the identifier. ",
        add_special_tokens=False,
    )
    ids = before_ids + repeated(filler_seed, filler_size) + after_ids
    assert len(ids) == length
    raw = json.dumps(ids, separators=(",", ":")).encode()
    return ids, {
        "expected": EXPECTED,
        "input_tokens": length,
        "before_tokens": len(before_ids),
        "filler_tokens": filler_size,
        "after_tokens": len(after_ids),
        "ids_json_sha256": hashlib.sha256(raw).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--input-tokens", type=int, default=65536)
    parser.add_argument("--counts", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if (
        not args.counts
        or args.counts[0] != 1
        or any(value < 1 for value in args.counts)
    ):
        parser.error("counts must start with cold count 1 and remain positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ids, fixture = frozen_ids(args.model_path, args.input_tokens)
    (args.output_dir / "needle_ids.json").write_text(json.dumps(ids) + "\n")
    (args.output_dir / "needle_fixture.json").write_text(
        json.dumps(fixture, indent=2) + "\n"
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    base = args.base_url.rstrip("/")

    def post(endpoint, payload):
        request = urllib.request.Request(
            base + endpoint,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with opener.open(request, timeout=args.timeout) as response:
            body = response.read()
            if endpoint == "/flush_cache":
                return body.decode(errors="replace")
            return json.loads(body)

    def request(index):
        started = time.monotonic()
        response = post(
            "/generate",
            {
                "input_ids": ids,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 16,
                    "ignore_eos": False,
                },
                "stream": False,
                "return_logprob": True,
                "logprob_start_len": -1,
            },
        )
        return {
            "index": index,
            "elapsed_seconds": time.monotonic() - started,
            "response": response,
        }

    flush = post("/flush_cache", {})
    (args.output_dir / "needle_flush.txt").write_text(flush + "\n")
    summary_rows = []
    cold_text = None
    for phase_index, count in enumerate(args.counts):
        label = "cold" if phase_index == 0 else f"hot{count}"
        with ThreadPoolExecutor(max_workers=count) as pool:
            rows = list(pool.map(request, range(count)))
        (args.output_dir / f"needle_{label}.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2) + "\n"
        )
        if cold_text is None:
            cold_text = rows[0]["response"].get("text", "")
        for row in rows:
            response = row["response"]
            meta = response.get("meta_info") or {}
            text = response.get("text", "")
            cached = meta.get("cached_tokens")
            summary_rows.append(
                {
                    "phase": label,
                    "request": row["index"],
                    "contains_nonce": EXPECTED in text,
                    "hot_cache_hit": phase_index == 0
                    or (isinstance(cached, (int, float)) and cached > 0),
                    "equals_cold_diagnostic": text == cold_text,
                    "text": text,
                    "cached_tokens": cached,
                    "completion_tokens": meta.get("completion_tokens"),
                    "spec_accept_length": meta.get("spec_accept_length"),
                }
            )
    passed = all(row["contains_nonce"] and row["hot_cache_hit"] for row in summary_rows)
    summary = {"passed": passed, "fixture": fixture, "results": summary_rows}
    (args.output_dir / "needle_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
