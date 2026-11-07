"""Exercise installed-router discovery diagnostics with controlled subprocesses."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    "behavior, status, exitcode",
    [
        ("fixed", "attention_dp_discovery_supported", 0),
        ("legacy", "attention_dp_discovery_mismatch", 2),
        ("empty", "inconclusive", 1),
    ],
)
def test_discovery_diagnostic_against_subprocess(tmp_path, behavior, status, exitcode):
    package = tmp_path / "sglang_router"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "sglang_router_rs.py").write_text(
        'def get_version_string():\n    return "controlled-test-build"\n',
        encoding="utf-8",
    )
    distribution = tmp_path / "sglang_router-0.0.0.dist-info"
    distribution.mkdir()
    (distribution / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: sglang-router\nVersion: 0.0.0\n",
        encoding="utf-8",
    )
    (package / "launch_router.py").write_text(
        """import argparse
import json
import os
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

parser = argparse.ArgumentParser()
parser.add_argument("--prefill", nargs=2)
parser.add_argument("--decode")
parser.add_argument("--port", type=int)
args, _ = parser.parse_known_args()
workers = []
for role, url in [("prefill", args.prefill[0]), ("decode", args.decode)]:
    with urllib.request.urlopen(url + "/server_info") as response:
        info = json.load(response)
    count = info["dp_size"]
    if os.environ["CONTROLLED_ROUTER_BEHAVIOR"] == "fixed":
        count *= info["attn_dp_size"]
    if os.environ["CONTROLLED_ROUTER_BEHAVIOR"] != "empty":
        workers.extend(dict(worker_type=role, url=f"{url}@{rank}", is_healthy=True)
                       for rank in range(count))

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = json.dumps({"workers": workers}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *args):
        pass

ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
""",
        encoding="utf-8",
    )
    output = tmp_path / "evidence"
    environment = os.environ.copy()
    environment.update(
        PYTHONPATH=str(tmp_path),
        CONTROLLED_ROUTER_BEHAVIOR=behavior,
        no_proxy="localhost,127.0.0.1",
        NO_PROXY="localhost,127.0.0.1",
    )
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "benchmark/hisparse/a5_pd/diagnose_router_wheel.py"),
            "--output",
            str(output),
            "--timeout",
            "2" if behavior == "empty" else "15",
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=70,
    )
    assert result.returncode == exitcode, result.stdout + result.stderr
    report = json.loads((output / "summary.json").read_text())
    assert report["status"] == status
    assert report["installed"]["native_build_version"] == "controlled-test-build"
    assert len(report["installed"]["native_sha256"]) == 64
    assert len(report["cases"]) == 3
    for capture in output.glob("*/mock-*-requests.json"):
        assert "/generate" not in json.loads(capture.read_text())
