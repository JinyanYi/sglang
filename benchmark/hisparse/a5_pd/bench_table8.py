"""Measure the two A5 P/D cases using the native SGLang serving benchmark."""

import argparse
import concurrent.futures
import contextlib
import hashlib
import json
import math
import random
import re
import statistics
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "a5_copysfa_mtp"))
from bench_prefix_cache import run_native_benchmark, validate_result
from check_server_state import validate as validate_server
from shared_prefix_workload import post

DP = 8
INPUT_TOKENS = 104000
OUTPUT_TOKENS = 1024
PAGE_SIZE = 128
CACHE_TOKENS = (INPUT_TOKENS - 1) // PAGE_SIZE * PAGE_SIZE
COLD_PAGES = math.ceil(INPUT_TOKENS / PAGE_SIZE)
TRANSFER_METRIC = "sglang:kv_transfer_total_mb"
METRICS = (
    "mean_ttft_ms",
    "median_ttft_ms",
    "p99_ttft_ms",
    "mean_tpot_ms",
    "median_tpot_ms",
    "p99_tpot_ms",
    "output_throughput",
    "request_throughput",
)
DECODE_BATCH = re.compile(
    r"\bDP(\d+)\b.*Decode batch(?: \[\d+\])?, #running-req: (\d+).*npu graph: True"
)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def get(url):
    with urllib.request.urlopen(url, timeout=15) as response:
        return response.read()


def check_servers(args, out, bs):
    for role, url, batch in (
        ("prefill", args.prefill_url, 16),
        ("decode", args.decode_url, bs),
    ):
        info = json.loads(get(url + "/server_info"))
        write_json(out / f"{role}-server-info.json", info)
        errors = validate_server(
            info, tp=DP, dp=DP, ep=DP, batch=batch, mtp=1, graph=role == "decode"
        )
        for key, value in {
            "model_path": args.model_path,
            "context_length": 106496,
            "page_size": PAGE_SIZE,
            "disaggregation_mode": role,
            "disaggregation_transfer_backend": "ascend",
        }.items():
            if info.get(key) != value:
                errors.append(f"{key}={info.get(key)!r}, expected {value!r}")
        if errors:
            raise ValueError(f"{role}: " + "; ".join(errors))
    workers = json.loads(get(args.base_url + "/workers"))
    write_json(out / "router-workers.json", workers)
    actual = {
        (w["worker_type"], w["url"]) for w in workers["workers"] if w["is_healthy"]
    }
    expected = {
        (role, f"{url}@{rank}")
        for role, url in (("prefill", args.prefill_url), ("decode", args.decode_url))
        for rank in range(DP)
    }
    if actual != expected:
        raise ValueError(
            "Router must expose 8 healthy DP ranks per role; see router-workers.json"
        )


def load_fixture(args):
    path = args.root / "input_ids.json"
    if args.input_ids:
        ids = json.loads(args.input_ids.read_bytes())
        if path.exists() and json.loads(path.read_bytes()) != ids:
            raise ValueError(
                "The supplied input IDs differ from this run's frozen fixture"
            )
    elif path.exists():
        ids = json.loads(path.read_bytes())
    else:
        from transformers import AutoTokenizer

        from sglang.benchmark.datasets.common import get_available_tokens

        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path, trust_remote_code=True
        )
        special_ids = set(tokenizer.all_special_ids)
        available_ids = [
            token
            for token in get_available_tokens(tokenizer)
            if token not in special_ids
        ]
        ids = random.Random(42).choices(available_ids, k=INPUT_TOKENS)
    if (
        not isinstance(ids, list)
        or len(ids) != INPUT_TOKENS
        or any(type(token) is not int or token < 0 for token in ids)
    ):
        raise ValueError("Expected exactly 104000 nonnegative input token IDs")
    if not path.exists():
        write_json(path, ids)
    return ids


def transfer_totals(raw):
    values = {"count": 0.0, "sum": 0.0}
    for line in raw.decode().splitlines():
        if not line.startswith(TRANSFER_METRIC + "_"):
            continue
        sample, value = line.rsplit(None, 1)
        suffix = sample.split("{", 1)[0].removeprefix(TRANSFER_METRIC + "_")
        if suffix in values:
            values[suffix] += float(value)
    return int(values["count"]), values["sum"]


def transfer_snapshot(prefill_url, path, expected_count=None):
    deadline = time.monotonic() + 15
    while True:
        raw = get(prefill_url + "/metrics")
        totals = transfer_totals(raw)
        if (
            expected_count is None
            or totals[0] >= expected_count
            or time.monotonic() >= deadline
        ):
            path.write_bytes(raw)
            if expected_count is not None and totals[0] != expected_count:
                raise ValueError(
                    f"Expected transfer counter {expected_count}, got {totals[0]}; see {path}"
                )
            return totals
        time.sleep(0.5)


def prime(args, out, ids):
    for role, url in (("prefill", args.prefill_url), ("decode", args.decode_url)):
        response = post(url + "/flush_cache?timeout=60", {}, timeout=65)
        (out / f"{role}-flush.txt").write_bytes(response)
    before = transfer_snapshot(args.prefill_url, out / "cold-metrics-before.txt")
    body = {
        "input_ids": ids,
        "stream": False,
        "sampling_params": {"temperature": 0, "max_new_tokens": 4, "ignore_eos": True},
    }
    # One round-robin request per D rank populates the local radix prefix.
    with concurrent.futures.ThreadPoolExecutor(max_workers=DP) as executor:
        futures = [
            executor.submit(post, args.base_url + "/generate", body, 3600)
            for _ in range(DP)
        ]
        results = [json.loads(future.result()) for future in futures]
    metadata = [result["meta_info"] for result in results]
    write_json(out / "cold-prime.json", metadata)
    if any(
        m.get("prompt_tokens") != INPUT_TOKENS
        or m.get("completion_tokens") != 4
        or m.get("cached_tokens") != 0
        for m in metadata
    ):
        raise ValueError("Cold priming did not complete with empty P/D prefixes")
    after = transfer_snapshot(
        args.prefill_url, out / "cold-metrics-after.txt", before[0] + DP
    )
    total_mib = after[1] - before[1]
    if not math.isfinite(total_mib) or total_mib <= 0:
        raise ValueError("Cold priming did not record positive P/D KV payload")
    return total_mib / (DP * COLD_PAGES)


def capture_report(log, out):
    from PIL import Image, ImageDraw, ImageFont

    lines = log.read_text(encoding="utf-8").splitlines()
    start = next(
        i for i, line in enumerate(lines) if "Serving Benchmark Result" in line
    )
    end = next(
        i for i in range(start + 1, len(lines)) if lines[i] and set(lines[i]) == {"="}
    )
    report = "\n".join(lines[start : end + 1]) + "\n"
    (out / "bench-serving.txt").write_text(report, encoding="utf-8")
    try:
        font = ImageFont.truetype("DejaVuSansMono.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    left, top, right, bottom = ImageDraw.Draw(
        Image.new("RGB", (1, 1))
    ).multiline_textbbox((0, 0), report, font=font, spacing=5)
    image = Image.new("RGB", (right - left + 32, bottom - top + 32), "#111827")
    ImageDraw.Draw(image).multiline_text(
        (16 - left, 16 - top), report, font=font, fill="#f1f5f9", spacing=5
    )
    image.save(out / "bench-serving.png")


def run_wave(args, out, ids, log_path, bs, mib_per_page, formal):
    out.mkdir()
    before = transfer_snapshot(args.prefill_url, out / "metrics-before.txt")
    log_offset = log_path.stat().st_size
    raw = out / "bench-serving.jsonl"
    stdout = out / "bench-serving.stdout.log"
    try:
        with (
            stdout.open("w", encoding="utf-8") as output,
            contextlib.redirect_stdout(output),
        ):
            run_native_benchmark(args, ids, raw, route_dp=False)
    finally:
        print(stdout.read_text(encoding="utf-8"), end="", flush=True)
    if formal:
        capture_report(stdout, out)

    result = json.loads(raw.read_bytes().splitlines()[-1])
    errors = validate_result(
        result, args.concurrency, INPUT_TOKENS, OUTPUT_TOKENS, CACHE_TOKENS
    )
    after = transfer_snapshot(
        args.prefill_url, out / "metrics-after.txt", before[0] + args.concurrency
    )
    transfer_mib = after[1] - before[1]
    expected_mib = mib_per_page * args.concurrency
    if not math.isclose(transfer_mib, expected_mib, rel_tol=1e-5, abs_tol=1e-3):
        errors.append(
            f"P/D payload was {transfer_mib:.6f} MiB; expected {expected_mib:.6f} MiB for one tail page/request"
        )
    with log_path.open("rb") as log:
        log.seek(log_offset)
        decode_log = log.read().decode(errors="replace")
    (out / "decode.log").write_text(decode_log, encoding="utf-8")
    full_ranks = {
        int(rank)
        for rank, batch in DECODE_BATCH.findall(decode_log)
        if int(batch) == bs
    }
    if formal and full_ranks != set(range(DP)):
        errors.append(
            f"Missing full BS{bs} NPU graph samples for DP ranks {sorted(set(range(DP)) - full_ranks)}"
        )
    values = {key: result[key] for key in METRICS}
    if any(not math.isfinite(value) or value <= 0 for value in values.values()):
        errors.append("Nonpositive or nonfinite native benchmark metrics")
    report = dict(
        valid=not errors,
        errors=errors,
        expected_cached_tokens=CACHE_TOKENS,
        transfer_requests=after[0] - before[0],
        transfer_mib=transfer_mib,
        expected_tail_transfer_mib=expected_mib,
        full_batch_graph_ranks=sorted(full_ranks),
        metrics=values,
        accept_length=result.get("accept_length"),
    )
    write_json(out / "validation.json", report)
    if errors:
        raise RuntimeError(f"{out}: " + "; ".join(errors))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("native", "hisparse"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prefill-url", required=True)
    parser.add_argument("--decode-url", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--input-ids", type=Path)
    parser.add_argument("--warmup-rounds", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()
    if args.warmup_rounds < 1 or args.rounds < 1:
        parser.error("warmup and measured rounds must both be positive")
    for name in ("prefill_url", "decode_url", "base_url"):
        setattr(args, name, getattr(args, name).rstrip("/"))
    bs = 5 if args.mode == "native" else 16
    args.concurrency = bs * DP
    args.dp_size = DP
    args.input_tokens = INPUT_TOKENS
    args.output_tokens = OUTPUT_TOKENS

    server_dir = args.root / f"decode-{args.mode}"
    latest = args.root / f"decode-{args.mode}.latest"
    if latest.is_file():
        server_dir = args.root / latest.read_text(encoding="utf-8").strip()
    log_path = server_dir / "server.log"
    policy = (server_dir / "kv-policy.log").read_text(encoding="utf-8")
    expected_policy = (
        "KV_OFFLOAD=0 indexer=legacy copy_sfa=0 metadata_fusions=0"
        if args.mode == "native"
        else "KV_OFFLOAD=1 indexer=lim copy_sfa=1 metadata_fusions=1"
    )
    if expected_policy not in policy or not log_path.is_file():
        raise ValueError(
            f"Unexpected decode KV policy or missing server log in {server_dir}"
        )
    out = Path(tempfile.mkdtemp(prefix=f"bench-{args.mode}-", dir=args.root))
    print(f"run_dir={out}", flush=True)
    check_servers(args, out, bs)
    ids = load_fixture(args)
    mib_per_page = prime(args, out, ids)
    for index in range(1, args.warmup_rounds + 1):
        run_wave(
            args, out / f"warmup-{index:02}", ids, log_path, bs, mib_per_page, False
        )
    rounds = [
        run_wave(args, out / f"round-{index:02}", ids, log_path, bs, mib_per_page, True)
        for index in range(1, args.rounds + 1)
    ]
    summary = dict(
        mode=args.mode,
        decode_server_dir=str(server_dir),
        dp_per_role=DP,
        attention_tp=1,
        bs_per_dp=bs,
        concurrency=args.concurrency,
        input_tokens=INPUT_TOKENS,
        output_tokens=OUTPUT_TOKENS,
        fixture_sha256=hashlib.sha256(
            (args.root / "input_ids.json").read_bytes()
        ).hexdigest(),
        cold_pages_per_request=COLD_PAGES,
        cold_transfer_mib_per_page=mib_per_page,
        expected_hot_pages_per_request=1,
        rounds=rounds,
        median_of_round_metrics={
            key: statistics.median(row["metrics"][key] for row in rounds)
            for key in METRICS
        },
    )
    write_json(out / "summary.json", summary)
    print(json.dumps(summary["median_of_round_metrics"], indent=2))
    print(f"summary={out / 'summary.json'}")


if __name__ == "__main__":
    main()
