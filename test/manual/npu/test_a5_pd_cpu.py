"""CPU byte/ordering checks for the A5 MF adaptation; no NPU acceptance."""

import ast
import ctypes
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
SPARSE = ROOT / "python/sglang/srt/hardware_backend/npu/sparsity_driven_kv_offload"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SPARSE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


storage = load("a5_pd_pool")
operations = load("a5_pd")


class Group:
    world_size = 2

    def __init__(self, rank, messages):
        self.rank_in_group, self.messages, self.cursor = rank, messages, 0
        self.published = 0

    def broadcast_object(self, value):
        if self.rank_in_group == 0:
            self.messages.append(value)
        else:
            value = self.messages[self.cursor]
        self.cursor += 1
        return value

    def barrier(self):
        pass

    def broadcast(self, tensor, src):
        self.published += 1


class Offload:
    OffloadConfig = SimpleNamespace
    Scene = SimpleNamespace(SHARED="shared")

    def __init__(self):
        self.configs, self.allocations = [], []

    def initialize(self, config):
        self.configs.append(config)
        return 0

    def empty(self, shape, dtype, pin_memory):
        tensor = torch.empty(shape, dtype=dtype)
        self.allocations.append(tensor)
        return tensor

    def sparse_copy(self, src, dst, lengths, count, device):
        for source, target, length in zip(src.tolist(), dst.tolist(), lengths.tolist()):
            if length:
                ctypes.memmove(target, source, length)
        return 0


@pytest.fixture
def shared(monkeypatch):
    offload = Offload()
    module = ModuleType("memfabric_hybrid")
    module.offload = offload
    monkeypatch.setitem(sys.modules, "memfabric_hybrid", module)
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(current_device=lambda: 0, synchronize=lambda: None),
        raising=False,
    )
    messages = []
    owner = storage.A5PackedMFPool(
        tokens=384, layers=2, device="cpu", group=Group(0, messages)
    )
    peer = storage.A5PackedMFPool(
        tokens=384, layers=2, device="cpu", group=Group(1, messages)
    )
    return owner, peer, offload


def test_one_physical_owner_byte_alias_and_page_offsets(shared):
    owner, peer, offload = shared
    assert len(offload.allocations) == 2
    assert offload.configs[0].alloc_size > 0 and offload.configs[1].alloc_size == 0
    assert offload.configs[0].store_url == offload.configs[1].store_url
    ptrs, sizes, pages = owner.get_transfer_buf_infos()
    for i, (host, alias) in enumerate(zip(owner.packed, peer.packed)):
        assert host.data_ptr() == alias.data_ptr() == ptrs[i]
        assert ptrs[i] % (2 * 1024**2) == 0
        assert sizes[i] == 384 * 656 and pages[i] == 128 * 656
        host.view(-1, 656)[128].fill_(97)
        assert torch.equal(
            alias.view(-1, 656)[128], torch.full((656,), 97, dtype=torch.int8)
        )


def test_masked_cross_page_copies_and_errors(shared):
    owner, _, _ = shared
    host = owner.packed[0]
    source = torch.arange(3 * 656).remainder(127).to(torch.int8).view(3, 1, 656)
    old, new = torch.tensor([0, -1, 2]), torch.tensor([127, -1, 128])
    valid = torch.tensor([True, False, True])
    owner.copy_rows([source], [host], old, new, valid)
    got = torch.zeros_like(source)
    owner.copy_rows([host], [got], new, old, valid)
    torch.testing.assert_close(got[[0, 2]], source[[0, 2]], rtol=0, atol=0)
    assert not got[1].any()
    with pytest.raises(ValueError, match="widths and dtypes"):
        owner.copy_rows([source], [host[..., :655]], old, new, valid)
    owner.offload.sparse_copy = lambda *args: 17
    with pytest.raises(RuntimeError, match="17"):
        owner.copy_rows([source], [host], old, new, valid)


def make_pool(shared):
    owner, _, _ = shared
    manager = SimpleNamespace(
        mf_host_pool=owner,
        host_kv_buffer=owner.packed,
        device="cpu",
        resets=[],
    )
    manager.reset_requests = lambda ids: manager.resets.append(ids)
    pool = SimpleNamespace(
        sparse_kv_manager=manager,
        indexer_layer_id_to_slot={10: 0, 11: 1},
        start_layer=10,
        index_k_buffer=torch.arange(2 * 384 * 128)
        .remainder(127)
        .to(torch.uint8)
        .view(2, 3, 128, 1, 128),
        index_k_scale_buffer=torch.arange(2 * 384, dtype=torch.float32).view(
            2, 3, 128, 1
        ),
        device="cpu",
        cpu_offloading_chunk_size=2,
    )
    for i, host in enumerate(owner.packed):
        host.copy_(
            torch.arange(host.numel()).remainder(127).to(torch.int8).view_as(host) + i
        )
    return pool


def test_overlapping_move_snapshot_semantics(shared):
    pool = make_pool(shared)
    manager = pool.sparse_kv_manager
    src, dst = torch.tensor([127, 128, 129]), torch.tensor([128, 129, 127])
    expected = [t.view(-1, 1, 656)[src].clone() for t in manager.host_kv_buffer]
    operations.move_packed_kv(manager, dst, src)
    for host, want in zip(manager.host_kv_buffer, expected):
        torch.testing.assert_close(host.view(-1, 1, 656)[dst], want, rtol=0, atol=0)
    assert manager.mf_host_pool.group.published == 1


def test_retraction_restores_bytes_local_index_scales_and_invalidates_lim(shared):
    pool = make_pool(shared)
    src, dst = torch.tensor([127, 128, 255]), torch.tensor([12, 257, 258])
    snapshot = operations.snapshot_packed_kv(pool, src)
    expected = [[t.clone() for t in chunk] for chunk in snapshot[0]]
    manager = pool.sparse_kv_manager
    for host in manager.host_kv_buffer:
        host.zero_()
    pool.index_k_buffer.zero_()
    pool.index_k_scale_buffer.zero_()
    operations.restore_packed_kv(pool, snapshot, dst, 3)
    for layer, host in enumerate(manager.host_kv_buffer):
        for offset, destination in enumerate(dst):
            chunk, row = divmod(offset, 2)
            torch.testing.assert_close(
                host.view(-1, 1, 656)[destination],
                snapshot[layer][chunk][0][row],
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                pool.index_k_buffer[layer].view(-1, 1, 128)[destination],
                snapshot[layer][chunk][1][row],
                rtol=0,
                atol=0,
            )
            assert (
                pool.index_k_scale_buffer[layer].view(-1, 1)[destination]
                == snapshot[layer][chunk][2][row]
            )
    assert manager.resets == [[3]]
    assert torch.equal(expected[0][0], snapshot[0][0][0])
    # Non-owner must restore its local index/scale but cannot overwrite shared KV.
    manager.mf_host_pool.rank = 1
    for host in manager.host_kv_buffer:
        host.fill_(42)
    pool.index_k_buffer.zero_()
    operations.restore_packed_kv(pool, snapshot, dst, 4)
    assert all(torch.all(host == 42) for host in manager.host_kv_buffer)
    assert pool.index_k_buffer.any()
    assert manager.resets[-1] == [4]
    assert operations.snapshot_packed_kv(pool, src[:0]) == [[], []]


def test_packed_pd_exports_host_then_device_state(shared):
    # Execute the real class methods without importing serving-only dependencies.
    source = ROOT / "python/sglang/srt/hardware_backend/npu/memory_pool_npu.py"
    tree = ast.parse(source.read_text())
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "NPUMLATokenToKVPool"
    )
    names = {
        "_get_disagg_buffer_entries",
        "get_contiguous_buf_infos",
        "get_kv_layer_ids",
        "get_state_layer_ids",
    }
    methods = [
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    namespace = {"torch": torch}
    exec(
        compile(ast.Module(body=methods, type_ignores=[]), str(source), "exec"),
        namespace,
    )
    pool = make_pool(shared)
    (
        pool.layer_num,
        pool.dsa_kv_cache_store_fp8,
        pool.indexer_layer_ids,
        pool.dcp_size,
    ) = 2, True, (10, 11), 1
    for name in names:
        setattr(pool, name, namespace[name].__get__(pool))
    ptrs, sizes, pages = pool.get_contiguous_buf_infos()
    assert len(ptrs) == len(pool.get_kv_layer_ids()) == 6
    assert ptrs[:2] == [t.data_ptr() for t in pool.sparse_kv_manager.host_kv_buffer]
    assert pages == [128 * 656] * 2 + [128 * 128] * 2 + [128 * 4] * 2
    assert sizes[:2] == [384 * 656] * 2


@pytest.mark.parametrize(
    "mode,mf,draft,backend,expected",
    [
        ("prefill", True, False, "ascend", False),
        ("decode", True, True, "ascend", False),
        ("decode", True, False, "ascend", True),
        ("null", False, False, "ascend", True),
        ("decode", False, False, "ascend", ValueError),
        ("decode", True, False, "mooncake", ValueError),
    ],
)
def test_pd_policy_is_explicit_and_draft_stays_native(
    monkeypatch, mode, mf, draft, backend, expected
):
    class Flag:
        def __init__(self, value):
            self.value = value

        def get(self):
            return self.value

    source = SPARSE / "config.py"
    tree = ast.parse(source.read_text())
    function = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef)
        and n.name == "is_sparsity_driven_kv_offload_enabled"
    )
    utils = ModuleType("sglang.srt.hardware_backend.npu.utils")
    utils.is_npu_arch35 = lambda: True
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    namespace = dict(
        ModelConfig=object,
        envs=SimpleNamespace(
            SGLANG_NPU_A5_PD_MF=Flag(mf),
            SGLANG_NPU_ENABLE_COPY_SFA=Flag(True),
            SGLANG_NPU_ENABLE_SPARSE_KV_OFFLOAD=Flag(True),
        ),
        get_disagg=lambda: SimpleNamespace(
            disaggregation_mode=mode, disaggregation_transfer_backend=backend
        ),
        get_parallel=lambda: SimpleNamespace(
            pp_size=1, dcp_enabled=False, attn_cp_size=1
        ),
        get_schedule=lambda: SimpleNamespace(page_size=128, max_running_requests=64),
        get_spec=lambda: SimpleNamespace(
            speculative_algorithm="NEXTN", speculative_eagle_topk=1
        ),
        get_model=lambda: SimpleNamespace(kv_cache_dtype="fp8_e4m3"),
        get_sparse_kv_indexer_impl=lambda: "lim",
        attention_backends=lambda: ("ascend", "ascend"),
        is_npu=lambda: True,
        is_deepseek_dsa=lambda _: True,
        get_dsa_index_topk=lambda _: 2048,
    )
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    call = lambda: namespace[function.name](
        model_config=SimpleNamespace(hf_config={}),
        use_mla_backend=True,
        is_draft_worker=draft,
    )
    if expected is ValueError:
        with pytest.raises(ValueError, match="A5 C8 P/D"):
            call()
    else:
        assert call() is expected


@pytest.mark.parametrize("role", ["prefill", "decode"])
def test_pd_launcher_keeps_c8_mtp_graph_and_radix(role):
    bash = shutil.which("bash")
    if sys.platform == "win32":
        bash = r"C:\Program Files\Git\bin\bash.exe"
    if not bash:
        pytest.skip("bash unavailable")
    env = os.environ.copy()
    env.update(
        PREFILL_IP="192.0.2.1",
        DECODE_IP="192.0.2.2",
        MODEL_PATH="/weights/a5",
        FRAMEWORK_ROOT="/repo",
        KERNEL_ROOT="/kernel",
    )
    result = subprocess.run(
        [bash, str(ROOT / "benchmark/hisparse/a5_pd/launch.sh"), role, "--dry-run"],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert f"--disaggregation-mode {role}" in result.stdout
    assert "--kv-cache-dtype fp8_e4m3" in result.stdout
    assert "--speculative-num-draft-tokens 4" in result.stdout
    if role == "decode":
        assert "KV_OFFLOAD=1" in result.stdout and "GRAPH=1" in result.stdout
        assert "--disaggregation-decode-enable-radix-cache" in result.stdout
        assert "--disaggregation-decode-extra-slots 0" in result.stdout
    else:
        assert "KV_OFFLOAD=0" in result.stdout


@pytest.mark.parametrize("failure", [None, "cache", "tokens", "retraction"])
def test_pd_smoke_rejects_invalid_evidence(tmp_path, monkeypatch, failure):
    spec = importlib.util.spec_from_file_location(
        "a5_pd_smoke", ROOT / "benchmark/hisparse/a5_pd/smoke.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    inputs = [[11] * 512]
    response = dict(
        text="731954",
        output_ids=[7] * 64,
        meta_info=dict(
            prompt_tokens=512,
            completion_tokens=64,
            cached_tokens=384,
            num_retractions=1 if failure != "retraction" else 0,
        ),
    )
    reference = dict(
        status="pass",
        inputs=inputs,
        fixture_sha256=hashlib.sha256(json.dumps(inputs).encode()).hexdigest(),
        cases=[dict(cold=[response])],
    )
    ref = tmp_path / "reference.json"
    ref.write_text(json.dumps(reference))

    def post(url, body):
        if url.endswith("/flush_cache"):
            return {}
        result = json.loads(json.dumps(response))
        if failure == "cache":
            result["meta_info"]["cached_tokens"] = 0
        if failure == "tokens":
            result["output_ids"][10] = 8
        return result

    monkeypatch.setattr(module, "post", post)
    args = SimpleNamespace(
        output=tmp_path / "run",
        reference=ref,
        lengths=[512],
        concurrency=2,
        prefill_url="http://p",
        decode_url="http://d",
        base_url="http://router",
        require_retraction=True,
    )
    if failure:
        with pytest.raises(ValueError):
            module.run(args)
    else:
        module.run(args)
        assert (
            json.loads((args.output / "summary.json").read_text())["status"] == "pass"
        )
