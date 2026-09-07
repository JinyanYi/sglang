"""Context projection for K3 pipeline-parallel DSpark prefill."""

from bisect import bisect_left
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterator, Optional

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class DSparkPrefillLoadPlan:
    """The draft parameters required by one PD-prefill PP stage."""

    feature_slice: slice
    num_context_features: int
    load_kv_writer: bool

    def __post_init__(self) -> None:
        start, stop = self.feature_slice.start, self.feature_slice.stop
        if (
            start is None
            or stop is None
            or self.feature_slice.step not in (None, 1)
            or not 0 <= start <= stop <= self.num_context_features
        ):
            raise ValueError("Invalid DSpark prefill context feature slice.")

    @property
    def local_num_features(self) -> int:
        return self.feature_slice.stop - self.feature_slice.start


_LOAD_PLAN: ContextVar[Optional[DSparkPrefillLoadPlan]] = ContextVar(
    "dspark_prefill_load_plan", default=None
)


@contextmanager
def dspark_prefill_load_scope(
    plan: Optional[DSparkPrefillLoadPlan],
) -> Iterator[None]:
    token = _LOAD_PLAN.set(plan)
    try:
        yield
    finally:
        _LOAD_PLAN.reset(token)


def get_dspark_prefill_load_plan() -> Optional[DSparkPrefillLoadPlan]:
    return _LOAD_PLAN.get()


def context_feature_slice(
    layer_ids: list[int], start_layer: int, end_layer: int, is_last_rank: bool
) -> slice:
    if not layer_ids or layer_ids != sorted(set(layer_ids)):
        raise ValueError("DSpark PP requires sorted, unique target layer ids.")
    # K3 captures the mixture computed by the NEXT consumer. The first layer
    # on this stage owns the preceding stage's boundary capture.
    return slice(
        bisect_left(layer_ids, max(0, start_layer - 1)),
        bisect_left(layer_ids, end_layer if is_last_rank else end_layer - 1),
    )


def accumulate_context(
    hidden: Optional[torch.Tensor],
    accumulated: Optional[torch.Tensor],
    weight: torch.Tensor,
    features: slice,
    num_tokens: int,
) -> Optional[torch.Tensor]:
    """Sum FC column-block products; normalize only after the final stage.

    The wire carries [tokens, hidden_size], never the concatenated captures.
    Accumulate in FP32 to avoid rounding the running sum at every PP hop.
    """
    hidden_size = weight.shape[0]
    if features.start > 0 and accumulated is None:
        raise RuntimeError("Missing DSpark context from the preceding PP stage.")
    if accumulated is not None and accumulated.shape != (num_tokens, hidden_size):
        raise RuntimeError("DSpark PP context has an unexpected token/feature shape.")
    if features.start == features.stop:
        return accumulated
    width = (features.stop - features.start) * hidden_size
    if hidden is None or hidden.shape[0] < num_tokens or hidden.shape[1] != width:
        raise RuntimeError("Missing or incorrectly shaped local DSpark PP captures.")
    if weight.shape[1] == width:
        local_weight = weight
    else:
        local_weight = weight[
            :, features.start * hidden_size : features.stop * hidden_size
        ]
        if local_weight.shape[1] != width:
            raise RuntimeError("DSpark PP context projection has an unexpected width.")
    partial = F.linear(
        hidden[:num_tokens],
        local_weight,
    ).float()
    return partial if accumulated is None else accumulated + partial
