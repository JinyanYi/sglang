#!/usr/bin/env python3
"""Cold/hot P/D retrieval gate against a same-input native C8 reference."""

import argparse
import concurrent.futures
import hashlib
import json
import urllib.request
from pathlib import Path


def post(url, body, timeout=3600):
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def make_inputs(tokenizer, lengths):
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False)
    prefix = encode("The secret access code is 731954. Remember this code.\n")
    suffix = encode(
        "\nWhat is the secret access code stated at the beginning? "
        "Repeat that code on separate lines. Output only the code, without explanation.\n"
    )
    filler = encode("This record contains ordinary background information.\n")
    result = []
    for length in lengths:
        count = length - len(prefix) - len(suffix)
        if count < 0 or not filler:
            raise ValueError("Retrieval fixture is too short")
        result.append(prefix + (filler * (count // len(filler) + 1))[:count] + suffix)
    return result


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    reference = json.loads(args.reference.read_text()) if args.reference else None
    if reference is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path, trust_remote_code=True
        )
        inputs = make_inputs(tokenizer, args.lengths)
    else:
        inputs = reference["inputs"]
        if [len(ids) for ids in inputs] != args.lengths:
            raise ValueError("Reference input lengths differ")
        if (
            hashlib.sha256(json.dumps(inputs).encode()).hexdigest()
            != reference["fixture_sha256"]
        ):
            raise ValueError("Reference input token IDs have changed")
    report = dict(
        status="running",
        inputs=inputs,
        cases=[],
        fixture_sha256=hashlib.sha256(json.dumps(inputs).encode()).hexdigest(),
        reference=str(args.reference),
        concurrency=args.concurrency,
        scope="Retrieval and exact greedy output IDs; not a full model accuracy suite or performance run",
    )
    path = args.output / "summary.json"

    def save():
        path.write_text(json.dumps(report, indent=2) + "\n")

    save()
    for case, ids in enumerate(inputs):
        for url in (args.prefill_url, args.decode_url):
            post(url.rstrip("/") + "/flush_cache", {})
        record = dict(input_tokens=len(ids), cold=[], hot=[])
        report["cases"].append(record)
        for phase in ("cold", "hot"):
            body = dict(
                input_ids=ids,
                stream=False,
                sampling_params=dict(
                    temperature=0,
                    max_new_tokens=64,
                    ignore_eos=True,
                ),
            )
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=args.concurrency
            ) as executor:
                futures = [
                    executor.submit(post, args.base_url.rstrip("/") + "/generate", body)
                    for _ in range(args.concurrency)
                ]
                for future in futures:
                    response = future.result()
                    record[phase].append(response)
                    save()
                    output = response.get("output_ids")
                    meta = response.get("meta_info", {})
                    if (
                        not isinstance(output, list)
                        or len(output) != 64
                        or meta.get("completion_tokens") != 64
                        or meta.get("prompt_tokens") != len(ids)
                    ):
                        raise ValueError(
                            "Incomplete response, missing output IDs, or wrong token counts"
                        )
                    if "731954" not in response.get("text", ""):
                        raise ValueError(
                            "Retrieval answer does not contain the expected code"
                        )
                    oracle = (
                        reference["cases"][case]["cold"][0]["output_ids"]
                        if reference
                        else record["cold"][0]["output_ids"]
                    )
                    if output != oracle:
                        raise ValueError("Greedy output differs from native C8 oracle")
                    if phase == "hot":
                        expected = (len(ids) - 1) // 128 * 128
                        if meta.get("cached_tokens") != expected:
                            raise ValueError(
                                f"Hot prefix cached_tokens={meta.get('cached_tokens')}, expected={expected}"
                            )
            if args.require_retraction and not any(
                response.get("meta_info", {}).get("num_retractions", 0) > 0
                for response in record[phase]
            ):
                raise ValueError(
                    "Forced-retraction phase recorded no actual retractions"
                )
            print(
                f"PASS input={len(ids)} phase={phase} requests={len(record[phase])}",
                flush=True,
            )
        save()
    report["status"] = "pass"
    save()
    print(f"PASS summary={path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--prefill-url", required=True)
    parser.add_argument("--decode-url", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--lengths", type=int, nargs="+", default=[512, 4096, 40000])
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--require-retraction", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.concurrency <= 0:
        parser.error("concurrency must be positive")
    if (
        args.reference
        and json.loads(args.reference.read_text()).get("status") != "pass"
    ):
        parser.error("reference must be a completed passing native run")
    run(args)


if __name__ == "__main__":
    main()
