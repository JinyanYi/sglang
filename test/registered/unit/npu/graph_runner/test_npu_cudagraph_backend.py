"""CPU unit tests for the NPU CUDA-graph backend's Python orchestration."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.hardware_backend.npu.graph_runner import npu_cudagraph_backend as mod
from sglang.srt.model_executor.runner.shape_key import ShapeKey
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _make_backend():
    device_module = SimpleNamespace(
        current_device=mock.Mock(return_value=3),
        synchronize=mock.Mock(),
        set_device=mock.Mock(),
        Event=mock.Mock(),
    )
    runner = SimpleNamespace(device_module=device_module, enable_torch_compile=False)
    with (
        get_parallel().override(tp_group=SimpleNamespace(barrier=mock.Mock())),
        mock.patch.object(mod.TorchMemorySaverAdapter, "create", return_value=None),
    ):
        return mod.NPUCudaGraphBackend(runner), runner


class TestNPUCudaGraphBackend(unittest.TestCase):
    def test_copy_sfa_legacy_gate_rejects_mixed_context_and_missing_lengths(self):
        backend, _ = _make_backend()
        shape_key = ShapeKey(size=4)
        backend._graphs[shape_key] = mock.Mock()
        backend._outputs[shape_key] = "output"
        decode_batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_decode=lambda: True),
            seq_lens_cpu=torch.tensor([0, 128, 2048, 4096]),
        )

        with (
            mock.patch.object(
                mod.envs.SGLANG_NPU_ENABLE_COPY_SFA, "get", return_value=True
            ),
            mock.patch.object(
                mod.envs.SGLANG_NPU_SPARSE_KV_POLICY, "get", return_value="legacy"
            ),
        ):
            self.assertFalse(backend.can_run(decode_batch, shape_key))
            decode_batch.seq_lens_cpu = None
            self.assertFalse(backend.can_run(decode_batch, shape_key))

    def test_prefix_tail_replays_captured_mixed_context_and_missing_length_batches(
        self,
    ):
        backend, _ = _make_backend()
        shape_key = ShapeKey(size=4)
        backend._graphs[shape_key] = mock.Mock()
        backend._outputs[shape_key] = "output"
        decode_batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_decode=lambda: True),
            seq_lens_cpu=torch.tensor([0, 128, 2048, 4096]),
        )

        with (
            mock.patch.object(
                mod.envs.SGLANG_NPU_ENABLE_COPY_SFA, "get", return_value=True
            ),
            mock.patch.object(
                mod.envs.SGLANG_NPU_SPARSE_KV_POLICY,
                "get",
                return_value="prefix_tail",
            ),
        ):
            self.assertTrue(backend.can_run(decode_batch, shape_key))
            decode_batch.seq_lens_cpu = None
            self.assertTrue(backend.can_run(decode_batch, shape_key))

    def test_replay_update_converts_legacy_sequence_lengths_to_int32_tensor(self):
        backend, runner = _make_backend()
        graph = mock.Mock()
        shape_key = ShapeKey(size=2)
        backend._graphs[shape_key] = graph
        backend._outputs[shape_key] = "output"

        output = backend.replay_with_input_update(
            shape_key,
            seq_lens=[7, 9],
            attr_name="seq_lens",
            attr_type=torch.empty(0),
        )

        runner.device_module.set_device.assert_called_once_with(3)
        graph.replay.assert_called_once_with()
        update_input = graph.update.call_args.kwargs["cpu_update_input"]
        self.assertEqual(len(update_input), 1)
        self.assertEqual(set(update_input[0]), {"seq_lens"})
        self.assertEqual(update_input[0]["seq_lens"].dtype, torch.int32)
        self.assertTrue(torch.equal(update_input[0]["seq_lens"], torch.tensor([7, 9])))
        self.assertEqual(output, "output")


if __name__ == "__main__":
    unittest.main()
