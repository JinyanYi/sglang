"""CPU regressions for NPU DSA indexer pool layer ownership."""

from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.mem_cache import kv_cache_configurator as kcc
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, stage="base-a", runner_config="cpu")


class TestNPUIndexerLayerLayout(CustomTestCase):
    def test_pd_compaction_requires_pp1_dcp1_target(self):
        memory = SimpleNamespace(
            enable_hierarchical_cache=False,
            enable_hisparse=False,
            enable_unified_cache_external_linker=False,
        )
        for role in ("prefill", "decode"):
            for pp_size, dcp_size, draft, expected in (
                (1, 1, False, True),
                (2, 1, False, False),
                (1, 2, False, False),
                (2, 2, False, False),
                (1, 1, True, False),
            ):
                with (
                    self.subTest(role=role, pp=pp_size, dcp=dcp_size, draft=draft),
                    patch.object(kcc, "_is_npu", True),
                    patch.object(kcc, "get_memory", return_value=memory),
                    patch.object(
                        kcc,
                        "get_disagg",
                        return_value=SimpleNamespace(disaggregation_mode=role),
                    ),
                    patch.object(
                        kcc,
                        "get_parallel",
                        return_value=SimpleNamespace(
                            pp_size=pp_size, attn_dcp_size=dcp_size
                        ),
                    ),
                ):
                    self.assertEqual(
                        kcc._should_elide_dsa_index_k(is_draft_worker=draft), expected
                    )

    def test_pd_compaction_preserves_cache_compatibility_exclusions(self):
        for excluded in (
            "enable_hierarchical_cache",
            "enable_hisparse",
            "enable_unified_cache_external_linker",
        ):
            memory = SimpleNamespace(
                enable_hierarchical_cache=False,
                enable_hisparse=False,
                enable_unified_cache_external_linker=False,
            )
            setattr(memory, excluded, True)
            with (
                self.subTest(excluded=excluded),
                patch.object(kcc, "_is_npu", True),
                patch.object(kcc, "get_memory", return_value=memory),
                patch.object(
                    kcc,
                    "get_disagg",
                    return_value=SimpleNamespace(disaggregation_mode="decode"),
                ),
                patch.object(
                    kcc,
                    "get_parallel",
                    return_value=SimpleNamespace(pp_size=1, attn_dcp_size=1),
                ),
            ):
                self.assertFalse(kcc._should_elide_dsa_index_k(is_draft_worker=False))

    def test_non_arch35_dsa_pool_uses_only_active_indexer_layers(self):
        """Shared layers must not consume indexer slots on non-arch35 NPU."""
        configurator = kcc.KVCacheConfigurator.__new__(kcc.KVCacheConfigurator)
        configurator.modelslim_cache_layout = None
        configurator.device = "cpu"
        configurator.is_draft_worker = False
        configurator.kv_cache_dtype = torch.bfloat16
        configurator.layer_info = SimpleNamespace(
            start_layer=10,
            end_layer=14,
            num_effective_layers=4,
        )
        configurator.model_config = SimpleNamespace(
            hf_config=object(),
            kv_lora_rank=16,
            qk_rope_head_dim=8,
            index_head_dim=32,
        )
        pool = object()

        with (
            patch.object(
                kcc,
                "get_parallel",
                return_value=SimpleNamespace(attn_dcp_size=2),
            ),
            patch.object(
                kcc,
                "get_schedule",
                return_value=SimpleNamespace(page_size=16),
            ),
            patch.object(
                kcc,
                "get_exec",
                return_value=SimpleNamespace(
                    features=SimpleNamespace(enable_memory_saver=False)
                ),
            ),
            patch.object(kcc, "_should_elide_dsa_index_k", return_value=True),
            patch.object(
                kcc,
                "dsa_layer_skips_topk",
                side_effect=lambda _config, layer_id: layer_id in (11, 13),
            ),
            patch(
                "sglang.srt.hardware_backend.npu.utils.is_npu_arch35",
                return_value=False,
            ),
            patch(
                "sglang.srt.hardware_backend.npu.memory_pool_npu.NPUMLATokenToKVPool",
                return_value=pool,
            ) as pool_class,
        ):
            result = configurator._build_ascend_mla_kv_pool(
                max_total_num_tokens=64,
                is_dsa_model=True,
            )

        self.assertIs(result, pool)
        self.assertEqual(pool_class.call_args.args, (64,))
        self.assertEqual(pool_class.call_args.kwargs["index_size"], 128)
        self.assertEqual(pool_class.call_args.kwargs["indexer_layer_ids"], (10, 12))


if __name__ == "__main__":
    import unittest

    unittest.main()
