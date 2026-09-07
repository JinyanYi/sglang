"""CPU regressions for K3 DSpark PP context collection and projection.

Load the model/worker methods without importing their accelerator backends so
these tests exercise the production control flow on a CPU-only host.
"""

import ast
import ctypes
import unittest
from collections import deque
from contextlib import nullcontext
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Optional
from unittest.mock import Mock

import torch
import torch.nn.functional as F
import numpy as np

from sglang.srt.speculative.dspark_components.dspark_pp import (
    DSparkPrefillLoadPlan,
    accumulate_context,
    context_feature_slice,
    dspark_prefill_load_scope,
    get_dspark_prefill_load_plan,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_ROOT = Path(__file__).resolve().parents[4]


def _methods(path, class_name, names, **namespace):
    tree = ast.parse((_ROOT / "python/sglang/srt" / path).read_text())
    nodes = tree.body
    if class_name is not None:
        nodes = next(
            n for n in nodes if isinstance(n, ast.ClassDef) and n.name == class_name
        ).body
    methods = [n for n in nodes if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(methods) == len(names)
    module = ast.parse("from __future__ import annotations")
    module.body.extend(methods)
    exec(compile(module, str(path), "exec"), namespace)
    return SimpleNamespace(**{name: namespace[name] for name in names})


class _Proxy:
    def __init__(self, tensors):
        self.tensors = tensors

    def __getitem__(self, key):
        return self.tensors[key]

    def __setitem__(self, key, value):
        self.tensors[key] = value


class TestDSparkPPProjection(unittest.TestCase):
    def test_capture_ownership_includes_boundary_and_empty_stages(self):
        layer_ids = [0, 1, 5, 7]
        slices = [
            context_feature_slice(layer_ids, i, i + 2, i == 6) for i in range(0, 8, 2)
        ]
        self.assertEqual([layer_ids[s] for s in slices], [[0], [1], [], [5, 7]])
        for ids in ([], [2, 1], [1, 1]):
            with self.assertRaises(ValueError):
                context_feature_slice(ids, 0, 4, True)

    def test_projection_matches_full_fc_including_padding_and_chunks(self):
        torch.manual_seed(17)
        layer_ids = [0, 1, 5, 7]
        for dtype in (torch.float32, torch.bfloat16):
            weight = torch.randn(32, 128).to(dtype)
            hidden = torch.randn(11, 128).to(dtype)
            expected = F.linear(hidden, weight).float()
            chunks = []
            for offset, length in ((0, 3), (3, 8)):
                accumulated = None
                for start in range(0, 8, 2):
                    features = context_feature_slice(
                        layer_ids, start, start + 2, start == 6
                    )
                    local = hidden[
                        offset : offset + length,
                        features.start * 32 : features.stop * 32,
                    ]
                    # DP padding is not part of the transferred prompt context.
                    if local.shape[1]:
                        local = F.pad(local, (0, 0, 0, 2))
                    local_weight = weight[
                        :,
                        features.start * 32 : features.stop * 32,
                    ]
                    accumulated = accumulate_context(
                        local, accumulated, local_weight, features, length
                    )
                self.assertEqual(accumulated.dtype, torch.float32)
                chunks.append(accumulated)
            actual = torch.cat(chunks)
            if dtype == torch.float32:
                torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
            else:
                relative_rms_error = (
                    actual - expected
                ).square().mean().sqrt() / expected.square().mean().sqrt()
                self.assertLess(relative_rms_error.item(), 0.004)

    def test_missing_capture_or_preceding_context_is_rejected(self):
        weight = torch.ones(4, 12)
        for hidden, acc, features in (
            (None, None, slice(0, 1)),
            (torch.ones(2, 8), None, slice(0, 1)),
            (torch.ones(1, 4), None, slice(0, 1)),
            (torch.ones(2, 4), None, slice(1, 2)),
            (None, torch.ones(3, 4), slice(1, 1)),
        ):
            with self.assertRaises(RuntimeError):
                accumulate_context(hidden, acc, weight, features, 2)
        self.assertIsNone(accumulate_context(None, None, weight, slice(0, 0), 2))

    def test_prefill_load_plan_scope_is_local_and_restored(self):
        plan = DSparkPrefillLoadPlan(
            feature_slice=slice(1, 3),
            num_context_features=5,
            load_kv_writer=False,
        )
        self.assertIsNone(get_dspark_prefill_load_plan())
        with dspark_prefill_load_scope(plan):
            self.assertIs(get_dspark_prefill_load_plan(), plan)
            self.assertEqual(plan.local_num_features, 2)
        self.assertIsNone(get_dspark_prefill_load_plan())

    def test_sparse_weight_loading_slices_fc_and_fused_qkv(self):
        methods = _methods(
            "models/dflash.py",
            "DFlashDraftModel",
            ["load_weights"],
            Optional=Optional,
            default_weight_loader=lambda param, weight: param.data.copy_(weight),
            _logical_linear_weight_shape=lambda *args, **kwargs: (),
        )

        fc = torch.nn.Parameter(torch.empty(4, 4))
        fc.dspark_feature_slice = slice(1, 2)
        qkv = torch.nn.Parameter(torch.empty(4, 3))
        qkv.dspark_kv_only = True
        qkv.checkpoint_q_size = 4
        qkv.checkpoint_kv_size = 2
        owner = SimpleNamespace(
            is_nemotron_35_draft=False,
            config=SimpleNamespace(hidden_size=4),
            num_context_features=3,
            named_parameters=lambda: [
                ("fc.weight", fc),
                ("layers.0.self_attn.qkv_proj.weight", qkv),
            ],
        )
        full_fc = torch.arange(48, dtype=torch.float32).reshape(4, 12)
        full_qkv = torch.arange(24, dtype=torch.float32).reshape(8, 3)
        methods.load_weights(
            owner,
            [
                ("fc.weight", full_fc),
                ("layers.0.self_attn.qkv_proj.weight", full_qkv),
            ],
        )
        torch.testing.assert_close(fc, full_fc[:, 4:8])
        torch.testing.assert_close(qkv, full_qkv[4:8])

    def test_sparse_weight_loading_skips_q_and_maps_split_kv(self):
        methods = _methods(
            "models/dflash.py",
            "DFlashDraftModel",
            ["load_weights"],
            Optional=Optional,
            default_weight_loader=lambda *args: None,
            _logical_linear_weight_shape=lambda *args, **kwargs: (),
        )
        loaded = {}
        qkv = torch.nn.Parameter(torch.empty(4, 3))
        qkv.dspark_kv_only = True
        qkv.weight_loader = lambda param, weight, shard: loaded.setdefault(
            shard, weight.clone()
        )
        owner = SimpleNamespace(
            is_nemotron_35_draft=False,
            config=SimpleNamespace(hidden_size=4),
            num_context_features=3,
            named_parameters=lambda: [
                ("layers.0.self_attn.qkv_proj.weight", qkv),
            ],
        )
        methods.load_weights(
            owner,
            [
                ("layers.0.self_attn.q_proj.weight", torch.full((4, 3), -1.0)),
                ("layers.0.self_attn.k_proj.weight", torch.full((2, 3), 1.0)),
                ("layers.0.self_attn.v_proj.weight", torch.full((2, 3), 2.0)),
            ],
        )
        self.assertEqual(set(loaded), {0, 1})
        torch.testing.assert_close(loaded[0], torch.ones(2, 3))
        torch.testing.assert_close(loaded[1], torch.full((2, 3), 2.0))

    def test_k3_boundary_uses_next_stages_weights(self):
        for use_attn_res in (False, True):
            layer_ids = [0, 1, 3, 5]
            captures = []
            proxy = None
            for start in (0, 2, 4):
                group = SimpleNamespace(
                    is_first_rank=start == 0, is_last_rank=start == 4, world_size=3
                )

                class Residual:
                    def __init__(self, hidden, blocks, block_residual):
                        self.block_residual = block_residual

                    def forward(self, hidden, residual, *args):
                        return hidden, None

                methods = _methods(
                    "models/kimi_k3.py",
                    "KimiK3LinearModel",
                    ["forward", "_dspark_capture_stream"],
                    torch=torch,
                    TYPE_CHECKING=False,
                    get_pp_group=lambda: group,
                    BumpAllocator=lambda **kw: None,
                    AttnResidual=Residual,
                    _cdiv=lambda a, b: (a + b - 1) // b,
                    envs=SimpleNamespace(
                        SGLANG_K3_SP_ATTN_RES=SimpleNamespace(get=lambda: False)
                    ),
                    get_global_expert_distribution_recorder=lambda: SimpleNamespace(
                        with_current_layer=lambda i: nullcontext()
                    ),
                    PPProxyTensors=_Proxy,
                    pack_aux_hidden_states=lambda h: torch.cat(h, dim=-1),
                    aggregate_stream=lambda hidden, bank, nvb, proj, norm: (
                        hidden + proj
                    ),
                )

                class Layer:
                    def __init__(self, index):
                        self.index = index
                        self.self_attention_res_proj = 100 * (index + 1)
                        self.self_attention_res_norm = None
                        self.prev_valid_blocks = 1

                    def __call__(self, *, hidden_states, **kw):
                        return hidden_states + self.index + 1, None, False

                owner = SimpleNamespace(
                    start_layer=start,
                    end_layer=start + 2,
                    pp_group=group,
                    config=SimpleNamespace(
                        attn_res_block_size=2 if use_attn_res else None
                    ),
                    _trim_padded_attn=False,
                    layers={i: Layer(i) for i in range(start, start + 2)},
                    dspark_layers_to_capture=layer_ids[
                        context_feature_slice(
                            layer_ids, start, start + 2, group.is_last_rank
                        )
                    ],
                    norm=lambda h: h,
                )
                if group.is_last_rank:
                    owner.output_attn_res_proj = 700
                    owner.output_attn_res_norm = None
                owner._dspark_capture_stream = MethodType(
                    methods._dspark_capture_stream, owner
                )
                output = methods.forward(
                    owner,
                    None,
                    torch.arange(2),
                    SimpleNamespace(),
                    inputs_embeds=torch.zeros(2, 1),
                    pp_proxy_tensors=proxy,
                )
                if group.is_last_rank:
                    captures.extend(output[1])
                else:
                    proxy = output
                    captures.append(proxy.tensors["dspark_aux_hidden_states"])
            actual = torch.cat(captures, dim=-1)
            expected = torch.tensor([1, 3, 10, 21], dtype=torch.float32)
            if use_attn_res:
                expected += torch.tensor([200, 300, 500, 700])
            torch.testing.assert_close(actual, expected.expand(2, -1))

    def test_only_last_stage_normalizes_and_removes_raw_captures(self):
        methods = _methods(
            "speculative/dspark_components/dspark_worker_v2.py",
            "DSparkWorkerV2",
            ["_accumulate_pp_context"],
            accumulate_context=accumulate_context,
        )
        fc = torch.nn.Linear(8, 4, bias=False)
        norm = Mock(side_effect=lambda h: h + 10)
        batch = SimpleNamespace(extend_lens=[2])
        worker = SimpleNamespace(
            _context_only_pp_rank=True,
            _context_features=slice(0, 1),
            draft_model=SimpleNamespace(fc=fc, hidden_norm=norm),
        )
        first = SimpleNamespace(
            pp_hidden_states_proxy_tensors=_Proxy(
                {"dspark_aux_hidden_states": torch.ones(2, 4)}
            )
        )
        methods._accumulate_pp_context(worker, batch, first, None)
        self.assertNotIn(
            "dspark_aux_hidden_states", first.pp_hidden_states_proxy_tensors.tensors
        )
        norm.assert_not_called()
        worker._context_only_pp_rank = False
        worker._context_features = slice(1, 2)
        last = SimpleNamespace(
            pp_hidden_states_proxy_tensors=None,
            logits_output=SimpleNamespace(hidden_states=torch.full((2, 4), 2.0)),
        )
        methods._accumulate_pp_context(
            worker, batch, last, first.pp_hidden_states_proxy_tensors
        )
        norm.assert_called_once()
        expected = (
            fc(torch.cat([torch.ones(2, 4), torch.full((2, 4), 2.0)], dim=-1)) + 10
        )
        torch.testing.assert_close(last.logits_output.hidden_states, expected)

    def test_intermediate_stage_returns_before_sampling_and_kv_injection(self):
        for idle in (False, True):
            target = Mock()
            output = SimpleNamespace()
            target.forward_batch_generation.return_value = output
            methods = _methods(
                "speculative/dspark_components/dspark_worker_v2.py",
                "DSparkWorkerV2",
                ["_forward_prefill"],
                get_parallel=lambda: SimpleNamespace(enable_dp_attention=False),
                CaptureHiddenMode=SimpleNamespace(FULL="full"),
            )
            worker = SimpleNamespace(
                _is_pp_prefill=True,
                _context_only_pp_rank=True,
                target_worker=target,
                _accumulate_pp_context=Mock(),
            )
            batch = SimpleNamespace(
                forward_mode=SimpleNamespace(is_idle=lambda: idle),
                seq_lens=torch.tensor([5]),
            )
            proxy = _Proxy({})
            result = methods._forward_prefill(worker, batch, None, proxy)
            self.assertIs(result, output)
            self.assertIs(result.new_seq_lens, batch.seq_lens)
            target.forward_batch_generation.assert_called_once_with(
                batch, capture_hidden_mode="full", pp_proxy_tensors=proxy
            )
            self.assertEqual(worker._accumulate_pp_context.call_count, 0 if idle else 1)

    def test_last_stage_writes_projected_context_to_shared_cache_location(self):
        events = []
        projected = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        cache_loc = torch.tensor([11, 13], dtype=torch.int32)
        output = SimpleNamespace(
            logits_output=SimpleNamespace(hidden_states=torch.zeros_like(projected)),
            next_token_ids=torch.tensor([5]),
        )

        class Injector:
            def inject_target_hidden(self, **kwargs):
                events.append(("inject", kwargs))

        def accumulate(batch, batch_output, pp_proxy_tensors):
            events.append(("accumulate", None))
            batch_output.logits_output.hidden_states = projected

        target = SimpleNamespace(
            forward_batch_generation=lambda *args, **kwargs: (
                events.append(("target", None)) or output
            )
        )
        methods = _methods(
            "speculative/dspark_components/dspark_worker_v2.py",
            "DSparkWorkerV2",
            ["_forward_prefill"],
            get_parallel=lambda: SimpleNamespace(enable_dp_attention=False),
            CaptureHiddenMode=SimpleNamespace(FULL="full"),
            SpecTpSyncSite=SimpleNamespace(DSPARK_TARGET="target"),
            is_pin_memory_available=lambda device: False,
            compute_position=lambda backend, prefix, context, total: (
                torch.arange(total),
                None,
            ),
            is_unified_kv_triton=lambda: False,
            make_next_draft_input=lambda **kwargs: kwargs,
            torch=torch,
        )
        worker = SimpleNamespace(
            _is_pp_prefill=True,
            _context_only_pp_rank=False,
            target_worker=target,
            _accumulate_pp_context=accumulate,
            _tp_sync=SimpleNamespace(sync=lambda *args: None),
            model_runner=SimpleNamespace(prefill_attention_backend_str="torch"),
            _kv_injector=Injector(),
        )
        batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_idle=lambda: False),
            extend_lens=[2],
            prefix_lens=[3],
            seq_lens=torch.tensor([5]),
            out_cache_loc=cache_loc,
        )
        result = methods._forward_prefill(worker, batch, None, _Proxy({}))

        self.assertIs(result, output)
        self.assertEqual(
            [event[0] for event in events], ["target", "accumulate", "inject"]
        )
        injected = events[-1][1]
        self.assertIs(injected["target_hidden"], projected)
        self.assertIs(injected["cache_loc"], cache_loc)
        self.assertTrue(injected["target_hidden_is_projected"])
        self.assertIsNone(output.logits_output.hidden_states)

    def test_dspark_ring_does_not_publish_eagle_fields(self):
        methods = _methods(
            "managers/scheduler_pp_mixin.py",
            "SchedulerPPMixin",
            ["_pp_prepare_tensor_dict"],
            add_auxiliary_output_to_pp_tensors=lambda *args: None,
        )
        owner = SimpleNamespace(spec_algorithm=SimpleNamespace(is_dspark=lambda: True))
        result = SimpleNamespace(
            next_token_ids=torch.tensor([2]),
            next_draft_input=object(),
            logits_output=None,
        )
        output = methods._pp_prepare_tensor_dict(
            owner, result, SimpleNamespace(return_logprob=False)
        )
        self.assertEqual(list(output), ["next_token_ids"])

    def test_only_last_prefill_stage_budgets_draft_kv(self):
        parallel = SimpleNamespace(pp_size=2, pp_rank=0, attn_dcp_size=2)
        methods = _methods(
            "model_executor/pool_configurator.py",
            None,
            ["_has_dflash_draft_pool", "_dflash_draft_cell_size"],
            get_parallel=lambda: parallel,
            get_disagg=lambda: SimpleNamespace(disaggregation_mode="prefill"),
            is_npu=lambda: True,
        )
        kvc = SimpleNamespace(
            is_draft_worker=False,
            spec_algorithm=SimpleNamespace(is_dflash_family=lambda: True),
            spec_aux_config=SimpleNamespace(dflash_draft_cell_size_per_token=128),
        )
        self.assertEqual(methods._dflash_draft_cell_size(kvc), 0)
        parallel.pp_rank = 1
        self.assertEqual(methods._dflash_draft_cell_size(kvc), 256)

    def test_original_topology_contacts_matching_rank_on_both_pp_stages(self):
        methods = _methods(
            "disaggregation/common/conn.py",
            "CommonKVManager",
            ["_resolve_rank_mapping"],
        )
        for rank in range(32):
            manager = SimpleNamespace(
                attn_tp_size=32,
                kv_args=SimpleNamespace(engine_rank=rank),
                is_mla_backend=False,
                is_hybrid_mla_backend=True,
                attn_cp_size=1,
                attn_cp_rank=0,
                pp_size=1,
                pp_rank=0,
            )
            info = SimpleNamespace(attn_tp_size=16, attn_cp_size=1, pp_size=2)
            methods._resolve_rank_mapping(manager, info)
            self.assertEqual(info.target_tp_ranks, [rank // 2])
            self.assertEqual(info.target_pp_ranks, [0, 1])
            self.assertEqual(info.required_prefill_response_num, 2)

    def test_ascend_mixed_mla_and_draft_transfers_exact_bytes(self):
        # Exercise real per-token offset construction with distinct K/V widths,
        # fragmented pages, head replication, and genuine 16 -> 32 head slicing.
        group = _methods(
            "disaggregation/common/utils.py",
            None,
            ["group_concurrent_contiguous"],
            np=np,
        )
        pairs = _methods(
            "disaggregation/utils.py", None, ["build_transfer_entry_pairs"], deque=deque
        )
        methods = _methods(
            "disaggregation/ascend/conn.py",
            "AscendKVManager",
            ["_send_hybrid_draft_kvcache"],
            np=np,
            group_concurrent_contiguous=group.group_concurrent_contiguous,
            build_transfer_entry_pairs=pairs.build_transfer_entry_pairs,
        )
        for total_heads in (4, 16, 32, 64):
            for page_size in (1, 4):
                for dst_rank in range(32):
                    src_heads, dst_heads = (
                        max(1, total_heads // 16),
                        max(1, total_heads // 32),
                    )
                    src_indices = np.array([1, 3, 4], dtype=np.int32)
                    dst_indices = np.array([4, 1, 2], dtype=np.int32)
                    arrays = []
                    dest = []
                    for heads, width in ((1, 5), (src_heads, 3), (src_heads, 2)):
                        arrays.append(
                            np.arange(
                                6 * page_size * heads * width, dtype=np.uint8
                            ).reshape(6, page_size, heads, width)
                        )
                        dest.append(
                            np.full(
                                (
                                    6,
                                    page_size,
                                    1 if len(dest) == 0 else dst_heads,
                                    width,
                                ),
                                255,
                                dtype=np.uint8,
                            )
                        )
                    src_ptrs = [a.ctypes.data for a in arrays]
                    dst_ptrs = [0] + [a.ctypes.data for a in dest]
                    blocks = []
                    manager = SimpleNamespace(
                        attn_tp_size=16,
                        kv_args=SimpleNamespace(
                            engine_rank=dst_rank // 2,
                            draft_total_kv_head_num=total_heads,
                            num_draft_kv_entries=2,
                            page_size=page_size,
                            kv_data_ptrs=src_ptrs,
                            kv_item_lens=[a[0].nbytes for a in arrays],
                            kv_layer_ids=[7, 8, 8],
                        ),
                        decode_kv_args_table={
                            "session": SimpleNamespace(dst_tp_rank=dst_rank)
                        },
                        _transfer_data=lambda session, b: blocks.extend(b) or 0,
                    )
                    methods._send_hybrid_draft_kvcache(
                        manager,
                        "session",
                        src_indices,
                        dst_ptrs,
                        dst_indices,
                        [3, 7, 8, 8],
                        32,
                    )
                    for src, dst, length in blocks:
                        # Assert bounds before simulating the DMA copy.
                        self.assertTrue(
                            any(
                                a.ctypes.data <= src
                                and src + length <= a.ctypes.data + a.nbytes
                                for a in arrays
                            )
                        )
                        self.assertTrue(
                            any(
                                a.ctypes.data <= dst
                                and dst + length <= a.ctypes.data + a.nbytes
                                for a in dest
                            )
                        )
                        ctypes.memmove(dst, src, length)
                    head_offset = (
                        dst_rank // max(1, 32 // total_heads) * dst_heads
                    ) - (dst_rank // 2 // max(1, 16 // total_heads) * src_heads)
                    for index, (src, dst) in enumerate(zip(arrays, dest)):
                        selected = (
                            src[src_indices]
                            if index == 0
                            else src[
                                src_indices, :, head_offset : head_offset + dst_heads
                            ]
                        )
                        np.testing.assert_array_equal(dst[dst_indices], selected)
                        np.testing.assert_array_equal(dst[[0, 3, 5]], 255)


if __name__ == "__main__":
    unittest.main()
