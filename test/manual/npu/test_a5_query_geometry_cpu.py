"""Long-chain geometry, budgets and launch contracts without an NPU runtime."""

import ast
import importlib.util
import shlex
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_a5_no_offload_launcher_cpu import load, run_shell

SOURCE = (
    Path(__file__).parents[3]
    / "python/sglang/srt/hardware_backend/npu/sparsity_driven_kv_offload"
)


def plain(name):
    spec = importlib.util.spec_from_file_location(name, SOURCE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


geometry = plain("query_geometry")
budget = plain("budget")


@pytest.mark.parametrize("width", [2, 3, 4, 5, 6, 8, 14])
def test_chain_width_and_tp_padding_are_distinct(width):
    spec = SimpleNamespace(
        speculative_algorithm="NEXTN",
        speculative_num_steps=width - 1,
        speculative_eagle_topk=1,
        speculative_num_draft_tokens=width,
    )
    assert geometry.configured_a5_query_width(spec) == width
    batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_decode=lambda: False),
        spec_info=SimpleNamespace(num_tokens_per_req=width),
        # Extra TP suffix rows must not change the logical query width.
        positions=list(range(2 * width + 2)),
    )
    assert geometry.sparse_query_width(batch) == width
    batch.forward_mode = SimpleNamespace(is_decode=lambda: True)
    assert geometry.sparse_query_width(batch) == 1
    spec.speculative_algorithm = None
    assert geometry.configured_a5_query_width(spec) == 1


@pytest.mark.parametrize("width,topk,steps", [(15, 1, 14), (3, 2, 2), (5, 1, 3)])
def test_invalid_chain_is_rejected(width, topk, steps):
    with pytest.raises(ValueError, match="chain MTP"):
        geometry.configured_a5_query_width(
            SimpleNamespace(
                speculative_algorithm="NEXTN",
                speculative_num_steps=steps,
                speculative_eagle_topk=topk,
                speculative_num_draft_tokens=width,
            )
        )


@pytest.mark.parametrize("width", [1, 2, 3, 4, 5, 6, 8, 14])
def test_c8_budget_charges_compute_rows_and_checks_union_capacity(width):
    options = dict(
        request_rows=3,
        context_len=65536,
        num_layers=2,
        top_k=2048,
        kv_bytes_per_token=656,
        cache_tokens=width * 2048,
        indexer_impl="lim",
        copy_sfa=True,
        copy_sfa_query_heads=8,
        device_tail_tokens=256,
        packed_c8=True,
        mtp_full_visible=width > 1,
        max_query_width=width,
    )
    result = budget.resolve_sparse_kv_budget(**options)
    assert result.device_kv_bytes == 3 * 2 * (width * 2048 + 256) * 656
    assert result.copy_sfa_bytes >= 3 * width * 8 * (576 + 512) * 2
    if width > 1:
        with pytest.raises(ValueError, match="query width"):
            budget.resolve_sparse_kv_budget(
                **(options | {"cache_tokens": width * 2048 - 128})
            )


@pytest.mark.parametrize("width", [2, 3, 5, 6, 8, 14])
def test_launcher_and_runtime_checker_use_requested_chain(width):
    result = run_shell(
        "bash benchmark/hisparse/a5_copysfa_mtp/40_launch_server.sh --dry-run",
        MTP="1",
        MTP_STEPS=str(width - 1),
        MTP_DRAFT_TOKENS=str(width),
        MTP_TOPK="1",
        SPARSE_CACHE_SIZE=str(max(8192, width * 2048)),
        SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK="256",
        PYTHON_BIN="python3",
    )
    assert result.returncode == 0, result.stderr
    args = shlex.split(result.stdout.splitlines()[-1])
    assert args[args.index("--speculative-num-steps") + 1] == str(width - 1)
    assert args[args.index("--speculative-num-draft-tokens") + 1] == str(width)
    state = dict(
        tp_size=8,
        attn_dp_size=1,
        dp_size=1,
        ep_size=8,
        pp_size=1,
        attn_cp_size=1,
        effective_max_running_requests_per_dp=16,
        disable_radix_cache=False,
        kv_cache_dtype="fp8_e4m3",
        speculative_algorithm="NEXTN",
        speculative_eagle_topk=1,
        speculative_num_steps=width - 1,
        speculative_num_draft_tokens=width,
        cuda_graph_config={"decode": {"backend": "full", "bs": [16]}},
    )
    check = load("check_server_state").validate
    options = dict(tp=8, dp=1, ep=8, batch=16, mtp=1, graph=1)
    assert not check(
        {"internal_states": [state]},
        **options,
        mtp_steps=width - 1,
        mtp_draft_tokens=width,
    )
    assert check({"internal_states": [state]}, **options)


def test_a5_long_gate_does_not_enable_short_or_mixed_graphs():
    # Exercise the production gate with CPU tensors, avoiding package imports.
    import torch

    namespace = {"torch": torch}
    path = SOURCE / "mtp.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    tree.body = [
        node
        for node in tree.body
        if getattr(node, "name", "") == "use_native_mtp_verify"
    ]
    exec(compile(tree, str(path), "exec"), namespace)
    manager = SimpleNamespace(
        packed_c8=True,
        max_query_width=5,
        cache_capacity=12288,
        lim_adapter=SimpleNamespace(mtp_full_visible=True),
    )
    batch = SimpleNamespace(
        spec_info=SimpleNamespace(num_tokens_per_req=5),
        forward_mode=SimpleNamespace(is_target_verify=lambda: True),
        seq_lens_cpu=torch.tensor([40000, 40001]),
    )
    gate = namespace["use_native_mtp_verify"]
    backend = SimpleNamespace(graph_mode=False)
    assert gate(backend, batch, manager)
    batch.seq_lens_cpu[1] = 2000
    assert not gate(backend, batch, manager)
    batch.seq_lens_cpu[:] = 40000
    batch.spec_info.num_tokens_per_req = 6
    assert not gate(backend, batch, manager)


def test_joint_gate_requires_each_native_and_lim_query_case(tmp_path):
    gate = load("verify_joint_junit")
    widths = (1, 2, 3, 4, 5, 6, 8, 14)
    joint = [
        f"test_real_lim_to_copysfa[{graph}-{width}]"
        for graph in ("False", "True")
        for width in widths
    ]
    native = [
        f"test_general_query_{mode}_accuracy[q{width}]"
        for mode in ("eager", "graph")
        for width in widths
        if width not in (1, 4)
    ]
    path = tmp_path / "junit.xml"

    def write(names):
        path.write_text(
            f'<testsuites><testsuite tests="{len(names)}">'
            + "".join(f'<testcase name="{name}"/>' for name in names)
            + "</testsuite></testsuites>"
        )

    write(joint + native)
    assert gate.verify([path], widths)["valid"]
    write(joint + native[:-1])
    result = gate.verify([path], widths)
    assert not result["valid"]
    assert result["missing_native_query_cases"] == [native[-1]]
