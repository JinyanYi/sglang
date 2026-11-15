"""Validate the external LIM tail ABI before combining it with CopySFA."""

import pytest
import torch

pytest.importorskip("torch_npu")


def test_external_lim_uses_prefix_relative_tail_slots():
    import acl

    from sglang.kernels.ops.attention.npu_quant_sparse import load_quant_lim_operator

    if acl.rt.get_device_info(0, 601) != (3510, 0):
        pytest.skip("The external FP8 LIM contract requires Ascend950")
    torch.npu.set_device(0)
    operator = load_quant_lim_operator()
    prefix, capacity, width = 8320, 8192, 4
    final_length = prefix + 129
    source_capacity = (final_length + 127) // 128 * 128
    pages = source_capacity // 128
    device = "npu"
    query = (
        torch.ones(width, 32, 128, dtype=torch.float32)
        .to(torch.float8_e4m3fn)
        .to(device)
    )
    key_cpu = torch.zeros(source_capacity, 128)
    # Tail scores strictly exceed every prefix score, guaranteeing selection.
    key_cpu[prefix:final_length] = 1
    key = key_cpu.to(torch.float8_e4m3fn).view(pages, 128, 1, 128).to(device)
    weights = torch.ones(width, 32, dtype=torch.bfloat16, device=device)
    query_scale = torch.ones(width, 32, dtype=torch.float32, device=device)
    key_scale = torch.ones(pages, 128, 1, dtype=torch.float32, device=device)
    table = torch.arange(pages, dtype=torch.int32, device=device).view(1, pages)
    ends = torch.tensor([width], dtype=torch.int32, device=device)
    lengths = torch.tensor([final_length], dtype=torch.int32, device=device)
    prefixes = torch.tensor([prefix], dtype=torch.int32, device=device)
    capacities = torch.tensor([capacity], dtype=torch.int32, device=device)
    state = torch.tensor([-2], dtype=torch.int32, device=device)
    request = torch.tensor([1], dtype=torch.int32, device=device)
    pool = torch.full((3, source_capacity), -1, dtype=torch.int32, device=device)
    sources = torch.full((width, 1, 2048), -1, dtype=torch.int32, device=device)
    destinations = torch.full_like(sources, -1)
    route_misses = torch.empty(width, dtype=torch.int32, device=device)
    miss_src = torch.empty(1, 32768, dtype=torch.int32, device=device)
    miss_dst = torch.empty_like(miss_src)
    misses = torch.empty(1, dtype=torch.int32, device=device)
    operator(
        weights,
        query_scale,
        query,
        key_scale,
        key,
        table,
        ends,
        lengths,
        prefixes,
        capacities,
        state,
        request,
        pool,
        sources,
        destinations,
        route_misses,
        miss_src,
        miss_dst,
        misses,
    )
    torch.npu.synchronize()
    source_cpu, slot_cpu = sources.cpu(), destinations.cpu()
    for route in range(width):
        ids = source_cpu[route, 0]
        slots = slot_cpu[route, 0]
        visible = final_length - (width - 1 - route)
        assert bool(((ids >= 0) & (ids < visible)).all())
        tail = ids >= prefix
        assert int(tail.sum()) == visible - prefix
        assert torch.equal(slots[tail], capacity + ids[tail] - prefix), (
            "External LIM returns a different tail slot layout. This bridge expects "
            "C+source-prefix; do not combine absolute circular slots with its rotated page table."
        )
    assert misses.item() == capacity
