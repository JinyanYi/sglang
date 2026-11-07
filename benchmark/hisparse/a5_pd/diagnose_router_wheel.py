#!/usr/bin/env python3
"""Inspect the installed router and probe DP discovery using local HTTP fixtures."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def fingerprint():
    import sglang_router
    from sglang_router import sglang_router_rs

    binary = Path(sglang_router_rs.__file__).resolve()
    digest = hashlib.sha256()
    with binary.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    build_version = getattr(sglang_router_rs, "get_version_string", None)
    return {
        "python": sys.executable,
        "machine": platform.machine(),
        "distribution_version": importlib.metadata.version("sglang-router"),
        "package": sglang_router.__file__,
        "native_module": str(binary),
        "native_sha256": digest.hexdigest(),
        "native_build_version": build_version() if build_version else None,
    }


def get_json(url):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=2) as response:
        return json.load(response)


def mock_worker(role, dp_size, attention_dp, model):
    info = {
        "model_path": str(model),
        "served_model_name": "router-dp-diagnostic",
        "model_type": "gpt2",
        "architectures": ["GPT2LMHeadModel"],
        "is_generation": True,
        "tp_size": 8,
        "dp_size": dp_size,
        "attn_dp_size": attention_dp,
        "disaggregation_mode": role,
        "load_balance_method": "follow_bootstrap_room",
        "internal_states": [{} for _ in range(dp_size * attention_dp)],
    }
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            if self.path.split("?", 1)[0] not in (
                "/health",
                "/health_generate",
                "/server_info",
                "/get_server_info",
                "/model_info",
                "/get_model_info",
                "/v1/models",
            ):
                self.send_error(404)
                return
            body = json.dumps(info).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, info, requests


def probe(case, dp_size, attention_dp, model, output, timeout):
    directory = output / case
    directory.mkdir()
    servers = []
    process = None
    result = {"case": case, "expected_ranks_per_role": dp_size * attention_dp}
    try:
        endpoints = {}
        for role in ("prefill", "decode"):
            server, info, requests = mock_worker(role, dp_size, attention_dp, model)
            servers.append((server, requests))
            endpoints[role] = f"http://127.0.0.1:{server.server_port}"
            write_json(directory / f"{role}-server-info.json", info)
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
        command = [
            sys.executable,
            "-m",
            "sglang_router.launch_router",
            "--pd-disaggregation",
            "--dp-aware",
            "--policy",
            "round_robin",
            "--prefill",
            endpoints["prefill"],
            "8998",
            "--decode",
            endpoints["decode"],
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "info",
        ]
        write_json(directory / "command.json", command)
        environment = os.environ.copy()
        environment.update(
            no_proxy="localhost,127.0.0.1,::1",
            NO_PROXY="localhost,127.0.0.1,::1",
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            PYTHONUNBUFFERED="1",
        )
        expected = {
            (role, f"{url}@{rank}")
            for role, url in endpoints.items()
            for rank in range(dp_size * attention_dp)
        }
        deadline = time.monotonic() + timeout
        previous, stable_since = None, time.monotonic()
        payload = {}
        with (directory / "router.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, env=environment, stdout=log, stderr=log)
            while time.monotonic() < deadline and process.poll() is None:
                try:
                    payload = get_json(f"http://127.0.0.1:{port}/workers")
                    workers = payload["workers"]
                    healthy = {
                        (w["worker_type"], w["url"]) for w in workers if w["is_healthy"]
                    }
                    if healthy != previous:
                        previous, stable_since = healthy, time.monotonic()
                    both_roles = {role for role, _ in healthy} == {"prefill", "decode"}
                    if both_roles and time.monotonic() - stable_since >= 5:
                        break
                except (OSError, ValueError, KeyError):
                    pass
                time.sleep(0.25)
            write_json(directory / "router-workers.json", payload)
            healthy = {
                (w["worker_type"], w["url"])
                for w in payload.get("workers", [])
                if w["is_healthy"]
            }
            result.update(
                healthy_per_role={
                    role: sum(r == role for r, _ in healthy) for role in endpoints
                },
                matches_expected=healthy == expected and process.poll() is None,
                missing=sorted(expected - healthy),
                unexpected=sorted(healthy - expected),
                exit_before_cleanup=process.poll(),
            )
    finally:
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            result["exit_after_cleanup"] = process.returncode
        for server, requests in servers:
            write_json(directory / f"mock-{server.server_port}-requests.json", requests)
            server.shutdown()
            server.server_close()
    write_json(directory / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout", type=float, default=60)
    args = parser.parse_args()
    output = args.output or Path(tempfile.mkdtemp(prefix="router-wheel-diagnostics-"))
    output.mkdir(parents=True, exist_ok=True)
    if (output / "summary.json").exists():
        parser.error("output already contains a diagnostic summary")
    if args.timeout <= 0:
        parser.error("timeout must be positive")
    print(f"output={output}", flush=True)
    try:
        metadata = fingerprint()
    except Exception as error:
        write_json(
            output / "summary.json",
            {
                "status": "inconclusive",
                "error": f"{type(error).__name__}: {error}",
                "python": sys.executable,
            },
        )
        print(f"Cannot load installed router: {error}", flush=True)
        return 1
    write_json(output / "installed-router.json", metadata)
    print(json.dumps(metadata, indent=2), flush=True)
    model = output.resolve() / "tokenizer"
    model.mkdir()
    write_json(
        model / "tokenizer.json",
        {
            "version": "1.0",
            "added_tokens": [],
            "normalizer": None,
            "pre_tokenizer": None,
            "post_processor": None,
            "decoder": None,
            "truncation": None,
            "padding": None,
            "model": {"type": "WordLevel", "vocab": {"[UNK]": 0}, "unk_token": "[UNK]"},
        },
    )
    write_json(
        model / "tokenizer_config.json",
        {"tokenizer_class": "PreTrainedTokenizerFast", "unk_token": "[UNK]"},
    )
    write_json(model / "config.json", {"model_type": "gpt2"})
    results = []
    for name, dp, attention_dp in (
        ("dp1_control", 1, 1),
        ("legacy_dp8_control", 8, 1),
        ("attention_dp8", 1, 8),
    ):
        print(f"Probing {name}: dp_size={dp}, attn_dp_size={attention_dp}", flush=True)
        result = probe(name, dp, attention_dp, model, output, args.timeout)
        results.append(result)
        print(json.dumps(result), flush=True)
    controls_pass = all(result["matches_expected"] for result in results[:2])
    if all(result["matches_expected"] for result in results):
        status, code = "attention_dp_discovery_supported", 0
    elif (
        controls_pass
        and results[2]["exit_before_cleanup"] is None
        and results[2]["healthy_per_role"] == {"prefill": 1, "decode": 1}
    ):
        status, code = "attention_dp_discovery_mismatch", 2
    else:
        status, code = "inconclusive", 1
    report = dict(
        status=status,
        installed=metadata,
        cases=results,
        boundary="Installed-wheel rank discovery against local HTTP fixtures; no NPU execution or model requests",
    )
    write_json(output / "summary.json", report)
    print(f"status={status}\nsummary={output / 'summary.json'}", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
