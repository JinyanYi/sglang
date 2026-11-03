"""Validate native-baseline launch construction without importing NPU packages."""

import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).parents[3]
HANDOFF = REPO / "benchmark/hisparse/a5_copysfa_mtp"
BASH = (
    r"C:\Program Files\Git\bin\bash.exe"
    if sys.platform == "win32"
    else shutil.which("bash")
)


def run_shell(script, **values):
    if not BASH or not Path(BASH).exists():
        pytest.skip("bash is unavailable")
    environment = dict(os.environ, **values)
    return subprocess.run(
        [BASH, "-c", script],
        cwd=REPO,
        env=environment,
        text=True,
        capture_output=True,
        timeout=15,
    )


@pytest.mark.parametrize("batch,bucket", [(7, 8), (16, 16)])
def test_native_launcher_clears_sparse_flags_and_preserves_mtp_graph(batch, bucket):
    result = run_shell(
        "bash benchmark/hisparse/a5_copysfa_mtp/41_launch_no_offload.sh --dry-run",
        BASELINE_PER_DP_BATCH_SIZE=str(batch),
        MODEL_PATH="/models/glm52",
        PYTHON_BIN="python3",
        KV_OFFLOAD="1",
        SGLANG_NPU_ENABLE_SPARSE_KV_OFFLOAD="1",
        SGLANG_NPU_ENABLE_COPY_SFA="1",
        SGLANG_NPU_SPARSE_KV_INDEXER="lim",
    )
    assert result.returncode == 0, result.stderr
    assert "KV_OFFLOAD=0 indexer=legacy copy_sfa=0 metadata_fusions=0" in result.stdout
    args = shlex.split(result.stdout.splitlines()[-1])
    for flag, expected in (
        ("--tp-size", "8"),
        ("--attn-dp-size", "4"),
        ("--ep-size", "8"),
        ("--max-running-requests", str(batch * 4)),
        ("--cuda-graph-bs-decode", str(bucket)),
        ("--speculative-num-steps", "3"),
        ("--speculative-num-draft-tokens", "4"),
        ("--kv-cache-dtype", "fp8_e4m3"),
        ("--context-length", "45056"),
    ):
        assert args[args.index(flag) + 1] == expected
    assert "--disable-cuda-graph" not in args
    assert "--disable-radix-cache" not in args


def test_offload_default_remains_joint():
    result = run_shell(
        "bash benchmark/hisparse/a5_copysfa_mtp/40_launch_server.sh --dry-run",
        KV_OFFLOAD="1",
        TP_SIZE="8",
        ATTN_DP_SIZE="4",
        EP_SIZE="8",
        PER_DP_BATCH_SIZE="16",
        MAX_RUNNING_REQUESTS="64",
        GRAPH_BATCH_SIZE="16",
        CONCURRENCY="64",
        PYTHON_BIN="python3",
        MTP="1",
        GRAPH="1",
    )
    assert result.returncode == 0, result.stderr
    assert "KV_OFFLOAD=1 indexer=lim copy_sfa=1 metadata_fusions=1" in result.stdout


def test_no_offload_policy_replaces_inherited_flags():
    result = run_shell(
        "source benchmark/hisparse/a5_copysfa_mtp/common.sh\n"
        "KV_OFFLOAD=0\nconfigure_kv_policy\nconfigure_profile_mode\n"
        'printf "%s %s %s %s %s" "$SGLANG_NPU_ENABLE_SPARSE_KV_OFFLOAD" '
        '"$SGLANG_NPU_ENABLE_COPY_SFA" "$SGLANG_NPU_SPARSE_KV_INDEXER" '
        '"${SGLANG_NPU_SPARSE_KV_CACHE_SIZE-unset}" "${PROFILE_MODE_ARGS[*]}"',
        SGLANG_NPU_ENABLE_SPARSE_KV_OFFLOAD="1",
        SGLANG_NPU_ENABLE_COPY_SFA="1",
        SGLANG_NPU_SPARSE_KV_INDEXER="lim",
        SGLANG_NPU_SPARSE_KV_CACHE_SIZE="12288",
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "0 0 legacy unset --no-offload"


def load(name):
    spec = importlib.util.spec_from_file_location(name, HANDOFF / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_live_seven_requests_can_use_graph_bucket_eight():
    checker = load("check_server_state")
    state = dict(
        tp_size=8,
        attn_dp_size=4,
        dp_size=1,
        ep_size=8,
        pp_size=1,
        attn_cp_size=1,
        effective_max_running_requests_per_dp=7,
        disable_radix_cache=False,
        kv_cache_dtype="fp8_e4m3",
        speculative_algorithm="NEXTN",
        speculative_num_steps=3,
        speculative_eagle_topk=1,
        speculative_num_draft_tokens=4,
        cuda_graph_config={"decode": {"backend": "full", "bs": [8]}},
    )
    info = {"internal_states": [dict(state) for _ in range(4)]}
    kwargs = dict(tp=8, dp=4, ep=8, batch=7, mtp=1, graph=1, graph_batch=8)
    assert checker.validate(info, **kwargs) == []
    info["internal_states"][0]["effective_max_running_requests_per_dp"] = 8
    assert checker.validate(info, **kwargs)


@pytest.mark.parametrize("lim", [False, True])
def test_native_profile_gate_rejects_joint_operators(tmp_path, monkeypatch, lim):
    profile = tmp_path / "profile"
    profile.mkdir()
    events = [{"name": "process_name", "pid": 2, "args": {"name": "Ascend Hardware"}}]
    for index in range(20):
        events.append(
            dict(name="step[VERIFY bs=7]", pid=1, ph="X", ts=index * 10, dur=1)
        )
        for name in ("QuantLightningIndexer", "SparseFlashAttention"):
            events.append(
                dict(
                    name=name,
                    pid=2,
                    ph="X",
                    ts=index * 10,
                    dur=1,
                    args={"Task Type": "MIX_AIC"},
                )
            )
    if lim:
        events.append(
            dict(
                name="FusedQuantLightningIndexerManage",
                pid=2,
                ph="X",
                ts=0,
                dur=1,
                args={"Task Type": "MIX_AIC"},
            )
        )
    (profile / "trace_view.json").write_text(json.dumps(events))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_profile.py",
            str(profile),
            "--output",
            str(tmp_path / "summary.json"),
            "--no-offload",
            "--expected-ranks",
            "1",
            "--batch",
            "7",
        ],
    )
    assert load("analyze_profile").main() == (1 if lim else 0)


def test_native_startup_requires_hbm_target_kv(tmp_path, monkeypatch):
    log = tmp_path / "server.log"
    prefix = "NPU MLA attention KV allocation: "
    log.write_text(
        prefix
        + "role=target offload=False CKV_bytes=1024 KPE_bytes=0\n"
        + prefix
        + "role=draft offload=False CKV_bytes=512 KPE_bytes=0\n"
    )
    monkeypatch.setattr(
        sys, "argv", ["check_startup_log.py", str(log), "--offload", "0"]
    )
    assert load("check_startup_log").main() == 0
    log.write_text(
        log.read_text().replace("role=target offload=False", "role=target offload=True")
    )
    with pytest.raises(RuntimeError, match="target allocation"):
        load("check_startup_log").main()
