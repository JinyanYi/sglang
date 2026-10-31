"""CPU caller-contract regressions; native A5 execution is tested separately."""

import ast
import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "python/sglang/srt/hardware_backend/npu/sparsity_driven_kv_offload"


def load_plain(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def extract(path, names, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    tree.body = [node for node in tree.body if getattr(node, "name", None) in names]
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace


WRAPPER = load_plain(ROOT / "python/sglang/kernels/ops/attention/npu_quant_sparse.py")
BUDGET = load_plain(SOURCE / "budget.py")
MTP = load_plain(SOURCE / "mtp.py")


def plan_type(*args):
    names = (
        "source_ids",
        "device_slots",
        "topk_misses",
        "miss_sources",
        "miss_slots",
        "miss_counts",
        "prefix_lens",
        "sparse_rows",
        "valid_rows",
        "request_slots",
        "seq_lens",
    )
    return SimpleNamespace(**dict(zip(names, args)))


class TestA5CallerContract(unittest.TestCase):
    def bridge(self):
        self.calls = []

        def native(*args):
            self.calls.append(args)
            args[13].fill_(7)
            args[14].fill_(3)
            args[15].zero_()
            args[18].zero_()

        namespace = dict(
            torch=torch,
            load_quant_lim_operator=lambda: native,
            fused_quant_lightning_indexer_manage=WRAPPER.fused_quant_lightning_indexer_manage,
            envs=SimpleNamespace(
                SGLANG_NPU_FUSED_LIM_PRE_METADATA=SimpleNamespace(get=lambda: False),
                SGLANG_NPU_FUSED_LIM_COPY_SFA_METADATA=SimpleNamespace(
                    get=lambda: False
                ),
            ),
            LIMSelection=plan_type,
            prepare_mtp_rows=MTP.prepare_mtp_rows,
            validate_mtp_chain=lambda *args: None,
        )
        bridge_cls = extract(SOURCE / "c8_lim_bridge.py", {"C8LIMBridge"}, namespace)[
            "C8LIMBridge"
        ]
        manager = SimpleNamespace(
            cache_capacity=8192,
            size=4,
            batch_capacity=2,
            max_context_len=8704,
            logical_max_context_len=8704,
            device="cpu",
            layer_num=1,
            start_layer=0,
            device_slot_map=[torch.full((5, 8704), -1, dtype=torch.int32)],
        )
        return bridge_cls(manager), manager

    def select(self, width, requests=(1, 0)):
        bridge, manager = self.bridge()
        rows = len(requests) * width
        mode = SimpleNamespace(
            is_target_verify=lambda: width == 4, is_draft_extend_v2=lambda: False
        )
        positions = (torch.tensor([8445, 0])[:, None] + torch.arange(width)).flatten()
        batch = SimpleNamespace(
            batch_size=2,
            req_pool_indices=torch.tensor(requests),
            seq_lens=torch.tensor([8445, 0]),
            positions=positions,
            out_cache_loc=torch.arange(rows),
            forward_mode=mode,
            spec_info=SimpleNamespace(num_tokens_per_req=width),
        )
        query = torch.zeros(rows, 32, 128, dtype=torch.float8_e4m3fn)
        qs = torch.arange(rows * 32, dtype=torch.float32).view(rows, 32) + 1
        ks = torch.ones(68, 128, 1, dtype=torch.float32) * 3
        call = bridge.select_mtp if width == 4 else bridge.select
        plan = call(
            manager,
            query=query,
            weights=torch.ones(rows, 32, dtype=torch.bfloat16),
            key=torch.zeros(68, 128, 1, 128, dtype=torch.float8_e4m3fn),
            block_table=torch.arange(136, dtype=torch.int32).view(2, 68),
            forward_batch=batch,
            layer_id=0,
            query_scale=qs,
            key_scale=ks,
        )
        return bridge, manager, batch, plan, qs, ks

    def test_q4_uses_real_scales_and_masks_padding(self):
        bridge, manager, batch, plan, qs, ks = self.select(4)
        call = self.calls[-1]
        self.assertEqual(len(call), 19)
        torch.testing.assert_close(call[1], qs)
        torch.testing.assert_close(call[3], ks)
        self.assertEqual(call[6].tolist(), [4, 8])
        self.assertEqual(plan.seq_lens.tolist(), [8449, 0])
        self.assertEqual(plan.prefix_lens.tolist(), [8320, 0])
        self.assertEqual(call[10].tolist(), [-2, -3])
        self.assertTrue((plan.source_ids[4:] == -1).all())
        self.assertTrue((plan.device_slots[4:] == -1).all())
        self.assertEqual(bridge.last_prefix[0, 1].item(), 8320)
        self.assertEqual(bridge.last_prefix[0, 4].item(), -1)

    def test_q1_has_per_request_query_ends(self):
        _, _, _, plan, _, _ = self.select(1)
        self.assertEqual(self.calls[-1][6].tolist(), [1, 2])
        self.assertEqual(plan.source_ids.shape, (2, 2048))

    def test_post_metadata_preserves_real_lengths_and_rotates_odd_tail_page(self):
        bridge, manager, _, plan, _, _ = self.select(4)
        manager.device_kv_stride = 8448
        manager.hbm_page_offsets = torch.arange(66, dtype=torch.int32)
        manager.planner_actual_seq_lengths_kv = torch.empty(2, dtype=torch.int32)
        manager.planner_num_cache_tokens = torch.empty(2, dtype=torch.int32)
        manager.planner_hbm_block_table = torch.empty(2, 66, dtype=torch.int32)
        manager.planner_dram_block_table = torch.empty(2, 68, dtype=torch.int32)

        def physical_table(requests, valid, out):
            # Different request rows may share the same radix physical prefix.
            out.copy_(torch.where(valid[:, None], torch.arange(68)[None] + 17, 0))

        manager.fill_host_block_table = physical_table
        bridge.copy_sfa_metadata(manager, plan)
        self.assertEqual(manager.planner_actual_seq_lengths_kv.tolist(), [8449, 0])
        self.assertEqual(manager.planner_num_cache_tokens.tolist(), [8321, 0])
        self.assertEqual(manager.planner_hbm_block_table[0, -2:].tolist(), [131, 130])
        self.assertTrue((manager.planner_hbm_block_table[1] == 0).all())
        self.assertEqual(manager.planner_dram_block_table[0, 0].item(), 17)

    def test_compute_workspace_is_not_priced_as_one_byte_c8(self):
        options = dict(
            request_rows=3,
            context_len=65536,
            num_layers=1,
            top_k=2048,
            kv_bytes_per_token=656,
            cache_tokens=16384,
            copy_sfa=True,
            indexer_impl="lim",
            copy_sfa_query_heads=8,
            mtp_full_visible=True,
            device_tail_tokens=256,
            packed_c8=True,
        )
        result = BUDGET.resolve_sparse_kv_budget(**options)
        self.assertEqual(result.device_kv_bytes, 3 * (16384 + 256) * 656)
        self.assertGreaterEqual(result.copy_sfa_bytes, 3 * 4 * 8 * (576 + 512) * 2)
        for name, value in (
            ("cache_tokens", 32512),
            ("copy_sfa_query_heads", 4),
            ("context_len", 262272),
            ("device_tail_tokens", 0),
        ):
            with self.subTest(name=name), self.assertRaises(ValueError):
                BUDGET.resolve_sparse_kv_budget(**(options | {name: value}))

    def test_missing_scales_rejected_before_lim_launch(self):
        bridge, manager = self.bridge()
        with self.assertRaisesRegex(ValueError, "actual FP32"):
            bridge.select(
                manager,
                query=torch.empty(2, 32, 128),
                weights=None,
                key=None,
                block_table=None,
                forward_batch=None,
                layer_id=0,
            )
        self.assertFalse(self.calls)

    def test_copy_sfa_merged_query_keeps_bf16_with_int8_kv(self):
        received = []
        namespace = dict(
            torch=torch,
            load_quant_copy_sfa_operator=lambda: object(),
            fused_quant_scatter_copy_sparse_flash_attention=lambda *a, **kw: (
                received.append(a),
                a[-1].fill_(2),
            ),
        )
        cache_cls = extract(SOURCE / "c8.py", {"C8Cache"}, namespace)["C8Cache"]
        manager = SimpleNamespace(
            batch_capacity=2,
            device="cpu",
            start_layer=0,
            lim_adapter=SimpleNamespace(copy_sfa_metadata=lambda *a: None),
            device_kv_buffer=[torch.empty(2, 128, 1, 656, dtype=torch.int8)],
            host_kv_buffer=[torch.empty(2, 128, 1, 656, dtype=torch.int8)],
            dev_ptr_list=[12345],
            planner_hbm_block_table=torch.ones(2, 1, dtype=torch.int32),
            planner_dram_block_table=torch.ones(2, 1, dtype=torch.int32),
            planner_actual_seq_lengths_kv=torch.ones(2, dtype=torch.int32),
            planner_num_cache_tokens=torch.ones(2, dtype=torch.int32),
            mtp_attn_out=torch.empty(8, 8, 512, dtype=torch.bfloat16),
        )
        cache = cache_cls(manager, 8, torch.bfloat16)
        _, _, batch, plan, _, _ = self.select(4)
        output = cache.attention(
            manager,
            SimpleNamespace(layer_id=0, tp_q_head_num=4, scaling=0.1),
            batch,
            torch.ones(8, 4, 512, dtype=torch.bfloat16),
            torch.ones(8, 4, 64, dtype=torch.bfloat16) * 3,
            plan,
        )
        self.assertEqual(received[0][0].dtype, torch.bfloat16)
        self.assertEqual(received[0][0].shape, (8, 8, 576))
        self.assertEqual(received[0][1].dtype, torch.int8)
        self.assertEqual(received[0][3], 12345)
        self.assertEqual(received[0][9].tolist(), [4, 8])
        self.assertTrue((received[0][0][:, 4:] == 0).all())
        self.assertTrue((output[4:] == 0).all())


if __name__ == "__main__":
    unittest.main()
