"""Exercise production sparse-indexer routing without loading NPU libraries."""

import ast
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).parents[3] / "python/sglang/srt"
SOURCE = ROOT / "layers/attention/dsa/dsa_npu_indexer.py"
SPARSE = ROOT / "hardware_backend/npu/sparsity_driven_kv_offload"


@pytest.fixture
def dispatch(monkeypatch):
    # Load both real modules so a stale import cannot be hidden by a stub.
    for name in ("mtp", "c8_mtp"):
        qualified = f"sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.{name}"
        spec = importlib.util.spec_from_file_location(qualified, SPARSE / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        monkeypatch.setitem(sys.modules, qualified, module)

    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    (route,) = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and any(
            isinstance(value, ast.Constant) and value.value == "use_fused_sparse_kv"
            for value in ast.walk(node.test)
        )
    ]
    function = ast.parse("def dispatch():\n    return None\n").body[0]
    function.body.insert(0, route)
    return compile(
        ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
        str(SOURCE),
        "exec",
    )


def run_dispatch(
    dispatch,
    *,
    packed=True,
    mode="verify",
    width=4,
    graph=True,
    lengths=None,
    enabled=True,
    prefill=False,
    quant=None,
):
    calls = []
    plan = SimpleNamespace(source_ids=object())

    def select(kind, manager, **kwargs):
        calls.append((kind, kwargs))
        return plan

    manager = SimpleNamespace(
        packed_c8=packed,
        cache_capacity=8192,
        lim_adapter=SimpleNamespace(
            select=lambda *args, **kwargs: select("select", *args, **kwargs),
            select_mtp=lambda *args, **kwargs: select("mtp", *args, **kwargs),
        ),
    )
    backend = SimpleNamespace(
        use_fused_sparse_kv=enabled,
        graph_mode=graph,
        forward_metadata=SimpleNamespace(sparse_kv_lim_plan=None),
    )
    batch = SimpleNamespace(
        forward_mode=SimpleNamespace(
            is_decode=lambda: mode == "decode",
            is_target_verify=lambda: mode == "verify",
        ),
        spec_info=SimpleNamespace(num_tokens_per_req=width)
        if mode == "verify"
        else None,
        seq_lens_cpu=lengths,
    )
    query, query_scale, key_scale = object(), object(), object()
    scope = dict(
        manager=manager,
        backend=backend,
        is_prefill=prefill,
        self=SimpleNamespace(n_heads=32, head_dim=128),
        q=SimpleNamespace(view=lambda *shape: query),
        use_quant_indexer=packed if quant is None else quant,
        pool=SimpleNamespace(
            indexer_hadamard_128=object(),
            dtype="fp8",
            get_index_k_scale_buffer=lambda layer: key_scale,
        ),
        _quantize_npu_indexer_activation=lambda *args: (query, query_scale),
        weights=object(),
        past_key_states=object(),
        block_table=object(),
        forward_batch=batch,
        layer_id=7,
    )
    exec(dispatch, scope)
    result = scope["dispatch"]()
    if calls:
        assert result is plan.source_ids
        assert backend.forward_metadata.sparse_kv_lim_plan is plan
        assert calls[0][1]["forward_batch"] is batch
        assert calls[0][1]["query"] is query
        if packed:
            assert calls[0][1]["query_scale"] is query_scale
            assert calls[0][1]["key_scale"] is key_scale
        else:
            assert "query_scale" not in calls[0][1]
    else:
        assert result is None
        assert backend.forward_metadata.sparse_kv_lim_plan is None
    return [kind for kind, _ in calls]


@pytest.mark.parametrize(
    "case,expected",
    [
        ({"mode": "decode"}, ["select"]),
        ({}, ["mtp"]),
        ({"graph": False, "lengths": [8192, 0]}, ["mtp"]),
        ({"graph": False, "lengths": [8191]}, []),
        ({"width": 3}, []),
        ({"packed": False, "mode": "decode"}, ["select"]),
        ({"packed": False}, ["select"]),
        ({"packed": False, "width": 8, "graph": False}, ["select"]),
        ({"enabled": False}, []),
        ({"prefill": True}, []),
    ],
)
def test_sparse_indexer_routes_real_admission_helper(dispatch, case, expected):
    assert run_dispatch(dispatch, **case) == expected


def test_quantized_lim_requires_packed_c8_manager(dispatch):
    with pytest.raises(ValueError, match="A5 C8 bridge"):
        run_dispatch(dispatch, packed=False, quant=True)
