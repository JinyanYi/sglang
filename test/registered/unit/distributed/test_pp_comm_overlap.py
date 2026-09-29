import unittest
from collections import defaultdict, deque
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, call

import torch

from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class FakeStream:
    def __init__(self, stream_id):
        self.cuda_stream = stream_id


class FakeEvent:
    def __init__(self):
        self.recorded_stream = None

    def record(self, stream):
        self.recorded_stream = stream


def _make_scheduler(**attrs):
    scheduler = object.__new__(SchedulerPPMixin)
    scheduler.__dict__.update(attrs)
    return scheduler


class TestPPCommOverlap(CustomTestCase):
    def test_graph_proxy_send_records_forward_reuse_fence(self):
        comm_stream = FakeStream(4)
        work = Mock()
        works = [SimpleNamespace(work=work)]
        scheduler = _make_scheduler(
            pp_comm_stream=comm_stream,
            pp_comm_stream_ctx=nullcontext(),
            pp_send_done_event=None,
            device_module=SimpleNamespace(Event=FakeEvent),
        )

        scheduler._pp_commit_comm_work(works, fence_next_forward=True)

        work.wait.assert_called_once_with()
        self.assertEqual(works, [])
        self.assertIs(scheduler.pp_send_done_event.recorded_stream, comm_stream)

    def test_no_fence_event_without_comm_stream(self):
        scheduler = _make_scheduler(
            pp_comm_stream=None,
            pp_comm_stream_ctx=nullcontext(),
            pp_send_done_event=None,
        )

        scheduler._pp_commit_comm_work([SimpleNamespace(work=Mock())], True)

        self.assertIsNone(scheduler.pp_send_done_event)

    def test_forward_waits_for_graph_send_and_proxy_receive(self):
        schedule_stream = FakeStream(1)
        send_done_event = object()
        recv_event = object()
        forward_stream = Mock()
        scheduler = _make_scheduler(
            schedule_stream=schedule_stream,
            forward_stream=forward_stream,
            pp_send_done_event=send_done_event,
            pp_proxy_recv_event=recv_event,
        )

        scheduler._pp_wait_forward_dependencies()

        forward_stream.wait_stream.assert_called_once_with(schedule_stream)
        self.assertEqual(
            forward_stream.wait_event.call_args_list,
            [call(send_done_event), call(recv_event)],
        )
        self.assertIsNone(scheduler.pp_send_done_event)
        self.assertIsNone(scheduler.pp_proxy_recv_event)

    def test_inbox_returns_original_receive_event(self):
        recv_event = object()
        tensor_dict = {"__msg_type__": "output", "value": torch.arange(2)}
        scheduler = _make_scheduler(
            _pp_tensor_dict_inbox=defaultdict(
                deque, {"output": deque([(tensor_dict, recv_event)])}
            ),
        )

        received, event = scheduler._pp_recv_typed_dict("output")

        self.assertIs(received, tensor_dict)
        self.assertIs(event, recv_event)

    def test_cpu_output_relay_bypasses_tp_all_gather_only_for_outputs(self):
        tp_group = object()
        pp_group = Mock()
        pp_group.send_tensor_dict.return_value = []
        scheduler = _make_scheduler(
            pp_output_via_cpu=True,
            attn_tp_group=tp_group,
            pp_group=pp_group,
            pp_comm_stream_ctx=nullcontext(),
        )

        scheduler._pp_send_dict_to_next_stage(
            {"next_token_ids": torch.arange(2)}, msg_type="output"
        )
        self.assertIsNone(
            pp_group.send_tensor_dict.call_args.kwargs["all_gather_group"]
        )

        scheduler._pp_send_dict_to_next_stage(
            {"hidden_states": torch.arange(2)}, msg_type="proxy"
        )
        self.assertIs(
            pp_group.send_tensor_dict.call_args.kwargs["all_gather_group"], tp_group
        )

    def test_cpu_output_relay_receive_bypasses_tp_all_gather(self):
        recv = Mock(return_value=({}, None))
        scheduler = _make_scheduler(
            pp_output_via_cpu=True,
            attn_tp_group=object(),
            _pp_recv_typed_dict=recv,
        )

        scheduler._pp_recv_dict_from_prev_stage()

        recv.assert_called_once_with(expected_kind="output", all_gather_group=None)

    @unittest.skipUnless(
        hasattr(torch, "npu") and torch.npu.is_available(), "requires NPU"
    )
    def test_cpu_output_relay_preserves_bytes_and_cpu_forward_buffer(self):
        device = torch.device("npu:0")
        torch.npu.set_device(device)
        token_ids = torch.tensor([3, 17, 4096], dtype=torch.int64, device=device)
        original = PPProxyTensors(
            {"next_token_ids": token_ids, "cpu_tensor": torch.tensor([23])}
        )
        event = torch.npu.Event()
        event.record(torch.npu.current_stream())
        pp_group = Mock(is_last_rank=True, is_first_rank=False)
        pp_group.send_tensor_dict.return_value = []
        scheduler = _make_scheduler(
            pp_output_via_cpu=True,
            pp_group=pp_group,
            attn_tp_group=object(),
            pp_comm_stream_ctx=nullcontext(),
            device_module=torch.npu,
            device=device,
            _pp_spec_relay=False,
            future_map=Mock(),
        )
        batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_prebuilt=lambda: False),
            return_logprob=False,
            spec_algorithm=SimpleNamespace(is_dspark=lambda: False),
            req_pool_indices=None,
            input_ids=None,
        )

        scheduler._pp_send_output_to_next_stage(
            0, [batch], deque([(event, original)]), None
        )
        relayed = pp_group.send_tensor_dict.call_args.kwargs["tensor_dict"]
        self.assertEqual(relayed["__pp_output_device_keys__"], ["next_token_ids"])
        self.assertEqual(relayed["next_token_ids"].device.type, "cpu")
        self.assertIs(relayed["cpu_tensor"], original["cpu_tensor"])
        self.assertTrue(torch.equal(relayed["next_token_ids"], token_ids.cpu()))
        self.assertEqual(original["next_token_ids"].device.type, "npu")

        local_result = scheduler._pp_prep_batch_result(
            batch, SimpleNamespace(can_run_cuda_graph=False), PPProxyTensors(relayed)
        )
        self.assertEqual(local_result.next_token_ids.device.type, "npu")
        self.assertTrue(torch.equal(local_result.next_token_ids.cpu(), token_ids.cpu()))
        self.assertEqual(relayed["next_token_ids"].device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
