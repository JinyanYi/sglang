#!/usr/bin/env python3
"""Small real MF -> A5 AICore readback, eager/graph and TP sharing probe."""

import os

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401
from sgl_kernel_npu.sparsity_driven_kv_offload import unidex_copy_inplace

from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.a5_pd_pool import (
    A5PackedMFPool,
)


class ProbeGroup:
    def __init__(self):
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.rank_in_group = int(os.environ.get("RANK", "0"))
        if self.world_size > 1:
            dist.init_process_group("hccl")
            self.cpu = dist.new_group(backend="gloo")

    def broadcast_object(self, value):
        if self.world_size > 1:
            items = [value]
            dist.broadcast_object_list(items, src=0, group=self.cpu)
            return items[0]
        return value

    def barrier(self):
        if self.world_size > 1:
            dist.barrier(group=self.cpu)

    def broadcast(self, tensor, src):
        if self.world_size > 1:
            dist.broadcast(tensor, src=src)


def main():
    local = int(os.environ.get("LOCAL_RANK", "0"))
    torch.npu.set_device(local)
    device = f"npu:{local}"
    group = ProbeGroup()
    pool = A5PackedMFPool(tokens=384, layers=2, device=device, group=group)
    source = torch.arange(4 * 656, dtype=torch.int32).remainder(127).to(torch.int8)
    source = source.reshape(4, 1, 656).to(device)
    result = torch.empty_like(source)
    indices = torch.tensor([1, 127, 128, 255], dtype=torch.int64, device=device)
    rows = torch.arange(4, dtype=torch.int64, device=device)
    valid = torch.ones(4, dtype=torch.bool, device=device)
    for layer, host in enumerate(pool.packed):

        def enqueue():
            if pool.rank == 0:
                pool.copy_rows([source], [host], rows, indices, valid)
            pool.publish()
            unidex_copy_inplace(
                host.view(-1, 1, 656),
                result,
                indices,
                rows,
                valid,
                1,
                1,
                block_dim=48,
                src_ptr=host.data_ptr(),
            )

        enqueue()
        torch.npu.synchronize()
        torch.testing.assert_close(result.cpu(), source.cpu(), rtol=0, atol=0)
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            enqueue()
        for replay in range(3):
            source.fill_(layer * 10 + replay)
            graph.replay()
            torch.npu.synchronize()
            torch.testing.assert_close(result.cpu(), source.cpu(), rtol=0, atol=0)
    print(
        f"PASS rank={group.rank_in_group}: MF write + AICore read, eager/graph, byte exact",
        flush=True,
    )
    if group.world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
