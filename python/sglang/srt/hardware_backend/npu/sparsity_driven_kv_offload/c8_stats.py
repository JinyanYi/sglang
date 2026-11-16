"""Opt-in, windowed A5 CopySFA diagnostics outside the attention timing path."""

import json
import logging

import torch
from sgl_kernel_npu.sparsity_driven_kv_offload.c8_metadata import (
    C8_HIT_STAT_FIELDS,
    record_c8_hit_statistics,
)

logger = logging.getLogger(__name__)


class C8HitRateRecorder:
    def __init__(self, manager, *, interval, split_query, fused_ring_tail):
        # Bound every per-layer window, including the largest request miss list.
        if interval < 1 or interval * manager.batch_capacity * 32768 >= 2**31:
            raise ValueError(
                "A5 C8 statistics interval exceeds the INT32 window capacity"
            )
        self.interval = interval
        self.device = manager.device
        self.start_layer = manager.start_layer
        self.capacity = manager.cache_capacity
        self.split_query = split_query
        self.fused_ring_tail = fused_ring_tail
        self.counters = torch.zeros(
            (manager.layer_num, len(C8_HIT_STAT_FIELDS)),
            dtype=torch.int32,
            device=self.device,
        )
        self._steps = 0
        self._reset_after_capture = True

    def record(self, layer_index, plan):
        if torch.npu.is_current_stream_capturing():
            self._reset_after_capture = True
        record_c8_hit_statistics(
            source_ids=plan.source_ids,
            device_slots=plan.device_slots,
            topk_misses=plan.topk_misses,
            miss_counts=plan.miss_counts,
            prefix_lens=plan.prefix_lens,
            seq_lens=plan.seq_lens,
            valid_rows=plan.valid_rows,
            sparse_rows=plan.sparse_rows,
            counters=self.counters,
            layer_index=layer_index,
        )

    def maybe_report(self, *, in_capture=False):
        if in_capture or torch.npu.is_current_stream_capturing():
            return
        if self._reset_after_capture:
            self.counters.zero_()
            self._reset_after_capture = False
            self._steps = 0
            return
        self._steps += 1
        if self._steps < self.interval:
            return
        self._steps = 0
        # Deliberately synchronize only in diagnostics, never inside capture.
        torch.npu.synchronize(self.device)
        rows = self.counters.cpu().tolist()
        self.counters.zero_()
        layers = []
        totals = {name: 0 for name in C8_HIT_STAT_FIELDS}
        for index, row in enumerate(rows):
            values = dict(zip(C8_HIT_STAT_FIELDS, row))
            if not values["calls"]:
                continue
            for name, value in values.items():
                if name == "max_union_misses":
                    totals[name] = max(totals[name], value)
                else:
                    totals[name] += value
            values["layer"] = self.start_layer + index
            values["prefix_hit_rate"] = (
                1 - values["prefix_misses"] / values["prefix_selected"]
                if values["prefix_selected"]
                else None
            )
            layers.append(values)
        if not totals["calls"]:
            return
        totals["prefix_hit_rate"] = (
            1 - totals["prefix_misses"] / totals["prefix_selected"]
            if totals["prefix_selected"]
            else None
        )
        totals["fallback_call_fraction"] = totals["fallback_calls"] / totals["calls"]
        logger.info(
            "A5_C8_CACHE_STATS %s",
            json.dumps(
                dict(
                    capacity=self.capacity,
                    split_query=self.split_query,
                    fused_ring_tail=self.fused_ring_tail,
                    totals=totals,
                    layers=layers,
                ),
                separators=(",", ":"),
            ),
        )
