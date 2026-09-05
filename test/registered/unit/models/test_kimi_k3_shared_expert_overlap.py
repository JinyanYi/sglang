from types import SimpleNamespace
from unittest.mock import patch

import torch

import sglang.srt.models.kimi_k3 as kimi_k3
from sglang.test.ci.ci_register import register_cpu_ci


register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _ForwardMode:
    def is_extend(self):
        return False

    def is_target_verify(self):
        return True


def _fake_moe(events, *, shared_attn_tp_comm):
    moe = SimpleNamespace()
    moe.shared_experts = lambda x: events.append("shared_compute") or x + 1
    moe._shared_experts_attn_tp_comm = shared_attn_tp_comm
    moe._sbo_shared_overlap = False
    moe._shared_experts_tp1 = not shared_attn_tp_comm
    moe.use_latent_moe = True
    moe._use_mega_moe = False
    moe.tp_size = 1
    moe._ep_front = (
        lambda hidden: events.append("front") or (object(), hidden.clone())
    )
    moe._ep_front_overlap = lambda hidden: None
    moe.experts = (
        lambda hidden, topk: events.append("routed") or hidden + 10
    )
    moe._reduce_latent = lambda hidden: events.append("latent") or hidden
    moe.routed_expert_up_proj = (
        lambda hidden: (events.append("routed_up") or hidden, None)
    )
    moe._forward_shared_experts = (
        lambda hidden: events.append("shared_sync") or hidden + 1
    )
    return moe


def test_k3_npu_shared_mlp_overlaps_deepep_but_collectives_stay_on_main():
    events = []
    hidden = torch.arange(8, dtype=torch.float32).view(2, 4)
    gathered = torch.empty(4, 4)
    moe = _fake_moe(events, shared_attn_tp_comm=True)
    forward_batch = SimpleNamespace(forward_mode=_ForwardMode())

    def all_gather(output, value):
        events.append("all_gather")
        output.copy_(value.repeat(2, 1))

    def launch_shared(value, forward_func):
        events.append("launch_shared")
        return forward_func(value)

    def reduce_scatter(output, value):
        events.append("reduce_scatter")
        output.copy_(value[: output.shape[0]])

    with (
        patch.object(kimi_k3, "_is_npu", True),
        patch.object(
            kimi_k3.envs.SGLANG_NPU_USE_MULTI_STREAM, "get", return_value=True
        ),
        patch.object(
            kimi_k3,
            "get_moe_a2a_backend",
            return_value=SimpleNamespace(is_deepep=lambda: True),
        ),
        patch.object(
            kimi_k3,
            "get_parallel",
            return_value=SimpleNamespace(attn_tp_group=object()),
        ),
        patch.object(kimi_k3, "get_local_dp_buffer", return_value=gathered),
        patch.object(
            kimi_k3,
            "attn_tp_all_gather_into_tensor",
            side_effect=all_gather,
        ),
        patch.object(
            kimi_k3, "process_shared_expert", side_effect=launch_shared, create=True
        ),
        patch.object(
            kimi_k3,
            "wait_share_stream",
            side_effect=lambda: events.append("wait_shared"),
            create=True,
        ),
        patch.object(
            kimi_k3,
            "attn_tp_reduce_scatter_tensor",
            side_effect=reduce_scatter,
        ),
    ):
        output = kimi_k3.KimiK3MoE._forward_unfused(
            moe,
            hidden,
            prefix_sum=None,
            forward_batch=forward_batch,
        )

    assert events == [
        "front",
        "all_gather",
        "launch_shared",
        "shared_compute",
        "routed",
        "latent",
        "routed_up",
        "wait_shared",
        "reduce_scatter",
    ]
    assert torch.equal(output, (hidden + 10) + (hidden + 1))


def test_k3_npu_shared_overlap_disabled_keeps_synchronous_path():
    events = []
    hidden = torch.ones(2, 4)
    moe = _fake_moe(events, shared_attn_tp_comm=False)
    forward_batch = SimpleNamespace(forward_mode=_ForwardMode())

    with (
        patch.object(kimi_k3, "_is_npu", True),
        patch.object(
            kimi_k3.envs.SGLANG_NPU_USE_MULTI_STREAM, "get", return_value=False
        ),
        patch.object(
            kimi_k3,
            "get_moe_a2a_backend",
            return_value=SimpleNamespace(is_deepep=lambda: True),
        ),
        patch.object(
            kimi_k3,
            "process_shared_expert",
            side_effect=AssertionError("side stream must remain disabled"),
            create=True,
        ),
    ):
        output = kimi_k3.KimiK3MoE._forward_unfused(
            moe,
            hidden,
            prefix_sum=None,
            forward_batch=forward_batch,
        )

    assert events == [
        "front",
        "shared_sync",
        "routed",
        "latent",
        "routed_up",
    ]
    assert torch.equal(output, (hidden + 10) + (hidden + 1))
