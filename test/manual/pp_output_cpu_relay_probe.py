#!/usr/bin/env python3
"""Model-free NPU PP output-ring probe for SGLANG_PP_OUTPUT_VIA_CPU.

Run with torchrun on 3 or more NPUs. This exercises the real scheduler output
send/receive methods and GroupCoordinator dictionary P2P, without a model.
"""

import argparse
import inspect
import os
import subprocess
import sys
from collections import deque
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def pattern(torch, origin, iteration, count):
    return (
        torch.arange(count, dtype=torch.int64) * 1009
        + origin * 10000019
        + iteration * 100003
    )


def check_bytes(torch, actual, expected, label):
    actual_bytes = actual.detach().cpu().contiguous().view(torch.uint8)
    expected_bytes = expected.contiguous().view(torch.uint8)
    if not torch.equal(actual_bytes, expected_bytes):
        raise AssertionError(f"{label}: received bytes differ")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--sizes", default="8,4096")
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()
    sizes = [int(value) for value in args.sizes.split(",")]
    if (
        args.iterations < 1
        or args.timeout < 1
        or any(size < 8 or size % 8 for size in sizes)
    ):
        parser.error(
            "iterations/timeout must be positive; sizes must be multiples of 8"
        )
    if os.getenv("SGLANG_PP_OUTPUT_VIA_CPU") != "1":
        parser.error("set SGLANG_PP_OUTPUT_VIA_CPU=1")
    if os.getenv("SGLANG_PP_SKIP_PURE_CHUNKED_OUTPUT_COMM", "0") == "1":
        parser.error("disable SGLANG_PP_SKIP_PURE_CHUNKED_OUTPUT_COMM")

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "python"))

    import torch
    import torch.distributed as dist
    import torch_npu  # noqa: F401

    import sglang.srt.managers.scheduler_pp_mixin as pp_mixin
    from sglang.srt.distributed.parallel_state import GroupCoordinator
    from sglang.srt.environ import envs
    from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin
    from sglang.srt.model_executor.forward_batch_info import PPProxyTensors

    source = Path(inspect.getfile(SchedulerPPMixin)).resolve()
    expected_source = (
        repo_root / "python/sglang/srt/managers/scheduler_pp_mixin.py"
    ).resolve()
    if source != expected_source:
        raise RuntimeError(f"imported {source}, expected {expected_source}")
    if not envs.SGLANG_PP_OUTPUT_VIA_CPU.get():
        raise RuntimeError("SGLang did not read the CPU relay environment switch")

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size <= 2:
        parser.error("this probe targets PP > 2; use at least 3 ranks")
    torch.set_num_threads(1)
    torch.npu.set_device(local_rank)
    device = torch.device(f"npu:{local_rank}")
    dist.init_process_group("gloo", timeout=timedelta(seconds=args.timeout))

    coordinator = GroupCoordinator.__new__(GroupCoordinator)
    coordinator.rank = rank
    coordinator.local_rank = local_rank
    coordinator.rank_in_group = rank
    coordinator.world_size = world_size
    coordinator.ranks = list(range(world_size))
    coordinator.device = device
    coordinator.cpu_group = dist.group.WORLD
    coordinator.device_group = dist.group.WORLD  # All tested payloads must be CPU.

    scheduler = object.__new__(SchedulerPPMixin)
    scheduler.__dict__.update(
        pp_group=coordinator,
        attn_tp_group=None,
        device_module=torch.npu,
        device=device,
        spec_algorithm=SimpleNamespace(is_none=lambda: True),
        future_map=SimpleNamespace(stash=lambda *a, **kw: None),
        tp_worker=SimpleNamespace(model_runner=SimpleNamespace(sampling_observer=None)),
    )
    with patch.object(
        pp_mixin,
        "get_parallel",
        return_value=SimpleNamespace(pp_size=world_size, pp_async_batch_depth=0),
    ):
        scheduler.init_pp_loop_state()
    if not scheduler.pp_output_via_cpu:
        raise RuntimeError("PP scheduler did not enable CPU output relay")
    batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_prebuilt=lambda: False),
        return_logprob=False,
        spec_algorithm=SimpleNamespace(is_dspark=lambda: False),
        req_pool_indices=None,
        input_ids=None,
    )
    metadata = SimpleNamespace(can_run_cuda_graph=False)
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unavailable"
    print(
        f"START rank={rank} pp={world_size} commit={commit} scheduler={source}",
        flush=True,
    )

    try:
        for size in sizes:
            for iteration in range(args.iterations):
                expected_send = pattern(torch, rank, iteration, size // 8)
                if coordinator.is_last_rank:
                    device_tensor = expected_send.to(device)
                    event = torch.npu.Event()
                    event.record(torch.npu.current_stream())
                    queue = deque(
                        [(event, PPProxyTensors({"next_token_ids": device_tensor}))]
                    )
                    outputs = None
                else:
                    queue = deque()
                    outputs = PPProxyTensors(
                        {
                            "next_token_ids": expected_send,
                            "__pp_output_device_keys__": ["next_token_ids"],
                        }
                    )

                # Align a saturated ring: every stage calls send before recv.
                dist.barrier()
                works = scheduler._pp_send_output_to_next_stage(
                    0, [batch], queue, outputs
                )
                received, _ = scheduler._pp_recv_dict_from_prev_stage()
                for work in works:
                    work.work.wait()

                origin = (rank - 1) % world_size
                expected = pattern(torch, origin, iteration, size // 8)
                if received["__pp_output_device_keys__"] != ["next_token_ids"]:
                    raise AssertionError("missing device-origin metadata")
                if received["next_token_ids"].device.type != "cpu":
                    raise AssertionError("PP output did not arrive on CPU")
                check_bytes(torch, received["next_token_ids"], expected, "wire")
                local = scheduler._pp_prep_batch_result(
                    batch, metadata, PPProxyTensors(received)
                )
                if local.next_token_ids.device.type != "npu":
                    raise AssertionError("local consumption did not restore NPU tensor")
                check_bytes(torch, local.next_token_ids, expected, "local")
                if received["next_token_ids"].device.type != "cpu":
                    raise AssertionError("forwarding buffer moved off CPU")
                dist.barrier()
            print(
                f"DATA_PASS rank={rank} size={size} iterations={args.iterations}",
                flush=True,
            )
        print(f"PASS rank={rank}", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
