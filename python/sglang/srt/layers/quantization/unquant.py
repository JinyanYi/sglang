from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional

logger = logging.getLogger(__name__)

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from sglang.srt.environ import envs
from sglang.srt.layers.amx_utils import (
    CPUQuantMethod,
    _amx_process_weight_after_loading,
)
from sglang.srt.layers.moe import (
    MoeRunner,
    MoeRunnerBackend,
    MoeRunnerConfig,
    get_deepep_mode,
    get_moe_a2a_backend,
    get_moe_runner_backend,
)
from sglang.srt.layers.moe.moe_runner.triton import TritonMoeQuantInfo
from sglang.srt.layers.quantization.base_config import (
    FusedMoEMethodBase,
    LinearMethodBase,
    QuantizeMethodBase,
)
from sglang.srt.layers.utils import MultiPlatformOp, copy_or_rebind_param
from sglang.srt.utils import (
    cpu_has_amx_support,
    get_bool_env_var,
    is_cpu,
    is_hip,
    is_npu,
    set_weight_attrs,
    use_intel_amx_backend,
    use_intel_xpu_backend,
)

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        DispatchOutput,
        StandardDispatchOutput,
    )


_is_cpu_amx_available = cpu_has_amx_support()
_is_hip = is_hip()
_is_cpu = is_cpu()
_is_npu = is_npu()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip

if _use_aiter:
    from aiter.ops.shuffle import shuffle_weight
    from aiter.tuned_gemm import tgemm

if _is_npu:
    from sglang.srt.hardware_backend.npu.utils import npu_format_cast


class UnquantizedEmbeddingMethod(QuantizeMethodBase):
    """Unquantized method for embeddings."""

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        """Create weights for embedding layer."""
        weight = Parameter(
            torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return F.linear(x, layer.weight, bias)

    def embedding(self, layer: torch.nn.Module, input_: torch.Tensor) -> torch.Tensor:
        return F.embedding(input_, layer.weight)


class UnquantizedLinearMethod(LinearMethodBase):
    """Linear method without quantization."""

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        weight = Parameter(
            torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if _is_cpu and _is_cpu_amx_available:
            _amx_process_weight_after_loading(layer, ["weight"])

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if use_intel_amx_backend(layer):
            x_shapes = x.shape
            if len(x_shapes) == 3:
                x = x.view(-1, x.shape[-1])
            output = torch.ops.sgl_kernel.weight_packed_linear(
                x,
                layer.weight,
                bias,
                True,  # is_vnni
            )
            if len(x_shapes) == 3:
                output = output.view(x_shapes[0], x_shapes[1], -1)
            return output

        elif _use_aiter and type(layer.weight.data) is torch.Tensor:
            return tgemm.mm(x, layer.weight, bias, otype=x.dtype)

        return F.linear(x, layer.weight, bias)


def _moe_debug_native_dump(
    layer: "torch.nn.Module",
    x: "torch.Tensor",
    topk_output: tuple,
    lid: int,
) -> None:
    """
    Debug helper (GPU only): run a pure-Python expert loop, dump per-expert
    intermediate values (gate_up → act → down) and aggregate stats to disk.
    Results are saved as gpu_expert_*_L<lid>.pt in the working directory.
    This function does NOT affect the actual forward pass output.
    """
    topk_weights, topk_ids, _ = topk_output
    num_experts = layer.num_experts

    cnts = topk_ids.new_zeros((topk_ids.shape[0], num_experts))
    cnts.scatter_(1, topk_ids.to(torch.int64), 1)
    tokens_per_expert = cnts.sum(dim=0)
    idxs = topk_ids.view(-1).argsort()
    sorted_tokens = x[idxs // topk_ids.shape[1]]
    tpe_np = tokens_per_expert.cpu().numpy()

    all_gmm1, all_act, all_gmm2 = [], [], []
    start_idx = 0
    for i, num_tok in enumerate(tpe_np):
        end_idx = start_idx + num_tok
        if num_tok == 0:
            start_idx = end_idx
            continue
        tok = sorted_tokens[start_idx:end_idx]
        w13 = layer.w13_weight[i]
        w2 = layer.w2_weight[i]

        gate_up = F.linear(tok, w13)
        gate, up = gate_up.chunk(2, dim=-1)
        act_out = F.silu(gate) * up
        down_out = F.linear(act_out, w2)

        all_gmm1.append(gate_up)
        all_act.append(act_out)
        all_gmm2.append(down_out)

        logger.info(
            "[DBG GPU EXPERT] L%03d E%03d ntok=%d "
            "gate_up norm=%.5f mean=%.6f std=%.6f  "
            "act norm=%.5f mean=%.6f std=%.6f  "
            "down norm=%.5f mean=%.6f std=%.6f",
            lid, i, num_tok,
            gate_up.norm().item(), gate_up.mean().item(), gate_up.std().item(),
            act_out.norm().item(), act_out.mean().item(), act_out.std().item(),
            down_out.norm().item(), down_out.mean().item(), down_out.std().item(),
        )
        start_idx = end_idx

    if all_gmm1:
        cat_gmm1 = torch.cat(all_gmm1, dim=0)
        cat_act = torch.cat(all_act, dim=0)
        cat_gmm2 = torch.cat(all_gmm2, dim=0)
        logger.info(
            "[DBG GPU EXPERT] L%03d ALL gmm1(gate_up) norm=%.5f mean=%.6f std=%.6f sum=%.6f",
            lid, cat_gmm1.norm().item(), cat_gmm1.mean().item(), cat_gmm1.std().item(), cat_gmm1.sum().item(),
        )
        logger.info(
            "[DBG GPU EXPERT] L%03d ALL act norm=%.5f mean=%.6f std=%.6f sum=%.6f",
            lid, cat_act.norm().item(), cat_act.mean().item(), cat_act.std().item(), cat_act.sum().item(),
        )
        logger.info(
            "[DBG GPU EXPERT] L%03d ALL gmm2(down) norm=%.5f mean=%.6f std=%.6f sum=%.6f",
            lid, cat_gmm2.norm().item(), cat_gmm2.mean().item(), cat_gmm2.std().item(), cat_gmm2.sum().item(),
        )
        torch.save(
            {
                "dispatched_x": sorted_tokens.cpu(),
                "topk_ids": topk_ids.cpu(),
                "topk_weights": topk_weights.float().cpu(),
                "gmm1_out": cat_gmm1.cpu(),
                "act_out": cat_act.cpu(),
                "gmm2_out": cat_gmm2.cpu(),
            },
            f"gpu_expert_dump_L{lid:03d}.pt",
        )
        logger.info("[DBG GPU EXPERT] L%03d saved gpu_expert_dump_L%03d.pt", lid, lid)


class UnquantizedFusedMoEMethod(FusedMoEMethodBase, MultiPlatformOp):
    """MoE method without quantization."""

    def __init__(
        self,
        use_triton_kernels: bool = False,
        use_flashinfer_trtllm_moe: bool = False,
        use_deep_gemm: bool = False,
    ):
        super().__init__()
        self.use_flashinfer_cutlass = get_moe_runner_backend().is_flashinfer_cutlass()
        self.use_triton_kernels = use_triton_kernels
        self.with_bias = False
        self.use_flashinfer_trtllm_moe = use_flashinfer_trtllm_moe
        self.use_deep_gemm = use_deep_gemm
        self._cache_permute_indices = dict({})

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        with_bias: bool = False,
        **extra_weight_attrs,
    ):
        self.with_bias = with_bias

        # Fused gate_up_proj (column parallel)
        w13_up_dim = (
            2 * intermediate_size_per_partition
            if layer.moe_runner_config.is_gated
            else intermediate_size_per_partition
        )
        w13_weight_n, w13_weight_k = (w13_up_dim, hidden_size)
        if self.use_triton_kernels:
            w13_weight_n, w13_weight_k = w13_weight_k, w13_weight_n
        w13_weight = torch.nn.Parameter(
            torch.empty(num_experts, w13_weight_n, w13_weight_k, dtype=params_dtype),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        if self.with_bias:
            w13_weight_bias = torch.nn.Parameter(
                torch.empty(num_experts, w13_up_dim, dtype=torch.float32),
                requires_grad=False,
            )
            layer.register_parameter("w13_weight_bias", w13_weight_bias)
            set_weight_attrs(w13_weight_bias, extra_weight_attrs)

        # down_proj (row parallel)
        w2_weight_n, w2_weight_k = (
            hidden_size,
            intermediate_size_per_partition,
        )
        if self.use_triton_kernels:
            w2_weight_n, w2_weight_k = w2_weight_k, w2_weight_n
        w2_weight = torch.nn.Parameter(
            torch.empty(num_experts, w2_weight_n, w2_weight_k, dtype=params_dtype),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        if self.with_bias:
            w2_weight_bias = torch.nn.Parameter(
                torch.empty(num_experts, hidden_size, dtype=torch.float32),
                requires_grad=False,
            )
            layer.register_parameter("w2_weight_bias", w2_weight_bias)
            set_weight_attrs(w2_weight_bias, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        _should_use_aiter_moe = (
            _use_aiter
            and (
                get_moe_runner_backend().is_auto()
                or get_moe_runner_backend().is_aiter()
            )
            and self._aiter_ck_moe_supported(layer)
        )
        if _should_use_aiter_moe:
            copy_or_rebind_param(
                layer, "w13_weight", shuffle_weight(layer.w13_weight.data, (16, 16))
            )
            torch.cuda.empty_cache()
            copy_or_rebind_param(
                layer, "w2_weight", shuffle_weight(layer.w2_weight.data, (16, 16))
            )
            torch.cuda.empty_cache()

        # Pack weight for get better performance on CPU
        if _is_cpu and _is_cpu_amx_available:
            _amx_process_weight_after_loading(layer, ["w13_weight", "w2_weight"])
            if hasattr(layer, "w13_weight_bias"):
                layer.w13_weight_bias = Parameter(
                    layer.w13_weight_bias.float(), requires_grad=False
                )
            if hasattr(layer, "w2_weight_bias"):
                layer.w2_weight_bias = Parameter(
                    layer.w2_weight_bias.float(), requires_grad=False
                )

        if (
            self.use_deep_gemm
            and layer.w13_weight.dtype == torch.bfloat16
            and get_moe_a2a_backend().is_deepep()
            and get_deepep_mode().enable_low_latency()
            and not _is_npu
            and not _is_hip
            and hasattr(layer, "dispatcher")
        ):
            layer.dispatcher.set_quant_config({"dispatcher_output_dtype": "bf16"})

        # Reorder rows of W1 for fused gated activation
        if self.use_flashinfer_trtllm_moe:
            from flashinfer.fused_moe.core import (
                _maybe_get_cached_w3_w1_permute_indices,
                convert_to_block_layout,
                get_w2_permute_indices_with_cache,
            )

            # w1 and w3 have been swapped, so we don't need do that here
            epilogue_tile_m = 128
            block_k = 128
            old_shape_w13 = layer.w13_weight.data[0].shape
            old_shape_w2 = layer.w2_weight.data[0].shape
            new_shape_w13 = None
            new_shape_w2 = None
            for i in range(layer.num_local_experts):
                permute_indices = _maybe_get_cached_w3_w1_permute_indices(
                    self._cache_permute_indices,
                    layer.w13_weight.data[i].view(torch.uint8),
                    epilogue_tile_m,
                    is_gated_act_gemm=layer.moe_runner_config.is_gated,
                )
                tmp_weights1 = (
                    layer.w13_weight.data[i]
                    .clone()
                    .view(torch.uint8)[permute_indices.to(layer.w13_weight.data.device)]
                    .contiguous()
                )

                permute_indices = get_w2_permute_indices_with_cache(
                    self._cache_permute_indices,
                    layer.w2_weight.data[i].view(torch.uint8),
                    epilogue_tile_m,
                )
                tmp_weights2 = (
                    layer.w2_weight.data[i]
                    .clone()
                    .view(torch.uint8)[permute_indices.to(layer.w2_weight.data.device)]
                    .contiguous()
                )

                tmp_weights1 = convert_to_block_layout(
                    tmp_weights1.view(torch.uint8), block_k
                )
                tmp_weights2 = convert_to_block_layout(
                    tmp_weights2.view(torch.uint8), block_k
                )

                new_shape_w13 = tmp_weights1.view(torch.bfloat16).shape
                new_shape_w2 = tmp_weights2.view(torch.bfloat16).shape
                layer.w13_weight.data[i] = (
                    tmp_weights1.view(torch.bfloat16)
                    .contiguous()
                    .reshape(old_shape_w13)
                )
                layer.w2_weight.data[i] = (
                    tmp_weights2.view(torch.bfloat16).contiguous().reshape(old_shape_w2)
                )

            layer.w13_weight.data = layer.w13_weight.data.reshape(
                layer.num_local_experts, *new_shape_w13
            )
            layer.w2_weight.data = layer.w2_weight.data.reshape(
                layer.num_local_experts, *new_shape_w2
            )

        if _is_npu:
            for weight_name in ["w13_weight", "w2_weight"]:
                weight = getattr(layer, weight_name)
                origin_weight = weight.data.transpose(1, 2)
                new_weight = origin_weight.contiguous()
                origin_weight.untyped_storage().resize_(0)
                weight.data = npu_format_cast(new_weight)

        return

    def maybe_restore_flashinfer_trtllm_bf16_weight_shape_for_load(
        self,
        layer: torch.nn.Module,
        param: torch.nn.Parameter,
        weight_name: str,
    ) -> None:
        """Restore canonical BF16 MoE load shapes before hot weight copy.

        The flashinfer TRT-LLM BF16 postprocess reshapes expert weights into
        block layout. During weight update, checkpoint tensors are in
        canonical layout and need a temporary shape restore for copy.
        """
        if not get_moe_runner_backend().is_flashinfer_trtllm_routed():
            return

        expected_shape = None
        if weight_name.endswith(".experts.w13_weight"):
            w13_rows = (
                2 * layer.intermediate_size_per_partition
                if layer.moe_runner_config.is_gated
                else layer.intermediate_size_per_partition
            )
            expected_shape = (layer.num_local_experts, w13_rows, layer.hidden_size)
        elif weight_name.endswith(".experts.w2_weight"):
            expected_shape = (
                layer.num_local_experts,
                layer.hidden_size,
                layer.intermediate_size_per_partition,
            )

        if expected_shape is None or tuple(param.data.shape) == expected_shape:
            return

        expected_numel = expected_shape[0] * expected_shape[1] * expected_shape[2]
        if param.data.numel() != expected_numel:
            raise RuntimeError(
                f"Cannot restore flashinfer TRT-LLM BF16 MoE weight shape for {weight_name}: "
                f"current shape={tuple(param.data.shape)}, expected shape={expected_shape}."
            )

        param.data = param.data.reshape(expected_shape)

    def _aiter_ck_moe_supported(self, layer) -> bool:
        # aiter CK fused-MoE requires intermediate_size_per_partition to be 128-aligned
        # (GemmSpec=Default; otherwise CK raises "not support this GEMM problem").
        return layer.intermediate_size_per_partition % 128 == 0

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config
        if self.use_flashinfer_trtllm_moe:
            backend = (
                MoeRunnerBackend.FLASHINFER_TRTLLM_ROUTED
                if get_moe_runner_backend().is_flashinfer_trtllm_routed()
                else MoeRunnerBackend.FLASHINFER_TRTLLM
            )
        elif self.use_flashinfer_cutlass:
            import sglang.srt.layers.moe.moe_runner.flashinfer_cutlass  # noqa: F401

            backend = MoeRunnerBackend.FLASHINFER_CUTLASS
        elif self.use_deep_gemm:
            backend = MoeRunnerBackend.DEEP_GEMM
        elif self.use_triton_kernels:
            backend = MoeRunnerBackend.TRITON_KERNELS
        else:
            backend = MoeRunnerBackend.TRITON
        self.runner = MoeRunner(backend, moe_runner_config)

        # aiter CK fused-MoE only supports 128-aligned shapes; otherwise use triton.
        self._aiter_runner: Optional[MoeRunner] = None
        if (
            _use_aiter
            and (
                get_moe_runner_backend().is_auto()
                or get_moe_runner_backend().is_aiter()
            )
            and get_moe_a2a_backend().supports_aiter()
        ):
            if self._aiter_ck_moe_supported(layer):
                self._aiter_runner = MoeRunner(
                    MoeRunnerBackend.AITER, moe_runner_config
                )
            elif get_moe_runner_backend().is_aiter():
                raise ValueError(
                    "moe_runner_backend=aiter is not supported for "
                    f"intermediate_size_per_partition={layer.intermediate_size_per_partition}; "
                    "use --moe-runner-backend triton."
                )
            else:
                logger.warning_once(
                    "aiter CK fused-MoE does not support "
                    f"intermediate_size_per_partition={layer.intermediate_size_per_partition}; "
                    "using triton MoE runner."
                )

    @property
    def load_up_proj_weight_first(self) -> bool:
        # FlashInfer CUTLASS kernel assumes [Up, Gate] Proj as W13
        return self.use_flashinfer_cutlass

    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        return self.forward(
            layer=layer,
            dispatch_output=dispatch_output,
        )

    def forward_cuda(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        x = dispatch_output.hidden_states

        backend = self.runner.runner_backend
        if backend.is_triton_kernels():
            from sglang.srt.layers.moe.moe_runner.triton_kernels import (
                TritonKernelsQuantInfo,
            )

            quant_info = TritonKernelsQuantInfo(
                w13_weight=layer.w13_weight,
                w2_weight=layer.w2_weight,
                w13_bias=getattr(layer, "w13_weight_bias", None),
                w2_bias=getattr(layer, "w2_weight_bias", None),
            )
            return self.runner.run(dispatch_output, quant_info)
        elif self.runner.runner_backend.is_deep_gemm():
            w13_weight = layer.w13_weight
            w2_weight = layer.w2_weight
            from sglang.srt.layers.moe.moe_runner.deep_gemm import DeepGemmMoeQuantInfo

            # Only use_fp8=False when SGLANG_DEEPEP_BF16_DISPATCH is true,
            # otherwise use_fp8=True for FP8 dispatch path
            use_fp8 = not envs.SGLANG_DEEPEP_BF16_DISPATCH.get()
            quant_info = DeepGemmMoeQuantInfo(
                w13_weight=w13_weight,
                w2_weight=w2_weight,
                use_fp8=use_fp8,
            )
            return self.runner.run(dispatch_output, quant_info)
        elif self.use_flashinfer_cutlass:
            from sglang.srt.layers.moe.moe_runner.flashinfer_cutlass import (
                FlashInferCutlassMoeQuantInfo,
            )

            quant_info = FlashInferCutlassMoeQuantInfo(
                quant_type="bf16",
                w13_weight=layer.w13_weight,
                w2_weight=layer.w2_weight,
                output_dtype=x.dtype,
                moe_ep_size=layer.moe_ep_size,
                moe_ep_rank=layer.moe_ep_rank,
                moe_tp_size=layer.moe_tp_size,
                moe_tp_rank=layer.moe_tp_rank,
                apply_routed_scaling_factor=not layer.should_fuse_routed_scaling_factor_in_topk,
            )
            return self.runner.run(dispatch_output, quant_info)
        elif self.use_flashinfer_trtllm_moe:
            from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
                FlashInferTrtllmBf16MoeQuantInfo,
            )

            quant_info = FlashInferTrtllmBf16MoeQuantInfo(
                gemm1_weights=layer.w13_weight,
                gemm2_weights=layer.w2_weight,
                global_num_experts=layer.num_experts,
                local_expert_offset=layer.moe_ep_rank * layer.num_local_experts,
            )
            return self.runner.run(dispatch_output, quant_info)
        else:
            if self._aiter_runner is not None:
                from sglang.srt.layers.moe.moe_runner.aiter import (
                    AiterMoeQuantInfo,
                )

                quant_info = AiterMoeQuantInfo(
                    w13_weight=layer.w13_weight,
                    w2_weight=layer.w2_weight,
                    expert_mask=layer.dispatcher.expert_mask_gpu,
                )
                return self._aiter_runner.run(dispatch_output, quant_info)

            quant_info = TritonMoeQuantInfo(
                w13_weight=layer.w13_weight,
                w2_weight=layer.w2_weight,
                b13=getattr(layer, "w13_weight_bias", None),
                b2=getattr(layer, "w2_weight_bias", None),
            )
            _lid = getattr(layer, "layer_id", -1)
            from sglang.srt.model_executor.runner import get_is_capture_mode
            from sglang.srt.distributed import get_tensor_model_parallel_rank

            if _lid == 1 and not get_is_capture_mode() and get_tensor_model_parallel_rank() == 0:
                _moe_debug_native_dump(
                    layer,
                    dispatch_output.hidden_states,
                    dispatch_output.topk_output,
                    _lid,
                )
            return self.runner.run(dispatch_output, quant_info)

    def forward_cpu(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output

        moe_runner_config = self.moe_runner_config

        assert (
            moe_runner_config.activation == "silu"
        ), f"activation = {moe_runner_config.activation} is not supported."

        if use_intel_amx_backend(layer):
            from sglang.srt.layers.moe.topk import apply_topk_weights_cpu

            topk_weights, topk_ids, _ = topk_output
            x, topk_weights = apply_topk_weights_cpu(
                moe_runner_config.apply_router_weight_on_input, topk_weights, x
            )
            output = torch.ops.sgl_kernel.fused_experts_cpu(
                x,
                layer.w13_weight,
                layer.w2_weight,
                topk_weights,
                topk_ids,
                False,  # inplace # See [Note] inplace should be False in fused_experts.
                CPUQuantMethod.UNQUANT,
                None,  # w1_scale
                None,  # w2_scale
                None,  # w1_zp
                None,  # w2_zp
                None,  # block_size
                getattr(layer, "w13_weight_bias", None),
                getattr(layer, "w2_weight_bias", None),
                layer.moe_runner_config.gemm1_alpha,
                layer.moe_runner_config.gemm1_clamp_limit,
                True,  # is_vnni
            )
            return StandardCombineInput(hidden_states=output)
        else:
            from sglang.srt.layers.moe.fused_moe_native import moe_forward_native

            output = moe_forward_native(
                layer,
                x,
                topk_output,
                moe_runner_config,
            )
            return StandardCombineInput(hidden_states=output)

    def get_triton_quant_info(self, layer: torch.nn.Module) -> TritonMoeQuantInfo:
        return TritonMoeQuantInfo(
            w13_weight=layer.w13_weight,
            w2_weight=layer.w2_weight,
            b13=getattr(layer, "w13_weight_bias", None),
            b2=getattr(layer, "w2_weight_bias", None),
        )

    def forward_xpu(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output

        moe_runner_config = self.moe_runner_config
        assert moe_runner_config.activation in [
            "silu",
            "gelu",
            "relu2",  # Nemotron-H (NemotronHForCausalLM) uses squared-ReLU.
        ], f"activation = {moe_runner_config.activation} is not supported."

        backend = self.runner.runner_backend
        if use_intel_xpu_backend():
            # sgl-kernel-xpu path
            from sgl_kernel import fused_experts

            topk_weights, topk_ids, _ = topk_output
            if moe_runner_config.apply_router_weight_on_input:
                x = x * topk_weights.to(x.dtype)
                topk_weights = torch.ones_like(topk_weights)
            output = fused_experts(
                x,
                layer.w13_weight,
                layer.w2_weight,
                topk_weights,
                topk_ids,
                b1=getattr(layer, "w13_weight_bias", None),
                b2=getattr(layer, "w2_weight_bias", None),
                activation=moe_runner_config.activation,
                gemm1_alpha=moe_runner_config.gemm1_alpha,
                gemm1_limit=moe_runner_config.gemm1_clamp_limit,
            )
            return StandardCombineInput(hidden_states=output)
        else:
            assert backend.is_triton()
            assert (
                moe_runner_config.activation == "silu"
            ), f"activation = {moe_runner_config.activation} is not supported \
            for Triton PATH, please set ENV SGLANG_USE_SGL_XPU=1."

            quant_info = self.get_triton_quant_info(layer)
            return self.runner.run(dispatch_output, quant_info)

    def forward_npu(
        self,
        layer: torch.nn.Module,
        dispatch_output: DispatchOutput,
    ) -> CombineInput:

        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput
        from sglang.srt.layers.moe.token_dispatcher.base import DispatchOutputChecker

        if DispatchOutputChecker.format_is_deepep(dispatch_output):
            return self._forward_npu_deepep(layer, dispatch_output)

        # x.shape = [B*S, H]
        x = dispatch_output.hidden_states
        # topk_weights.shape = [B*S, K]; topk_ids.shape = [B*S, K]
        topk_weights, topk_ids, _ = dispatch_output.topk_output

        original_dtype = x.dtype
        num_tokens = x.shape[0]
        topk_weights = topk_weights.to(x.dtype)
        topk_ids = topk_ids.to(torch.int32)
        num_experts = layer.num_experts
        top_k = layer.top_k or topk_ids.shape[1]  # in case layer.top_k is not set

        _lid = getattr(layer, "layer_id", -1)
        from sglang.srt.model_executor.runner import get_is_capture_mode
        from sglang.srt.distributed import get_tensor_model_parallel_rank

        _dbg = _lid == 1 and not get_is_capture_mode() and get_tensor_model_parallel_rank() == 0

        hidden_states, expanded_row_idx, expert_tokens, _ = (
            torch.ops.npu.npu_moe_init_routing_v2(
                x,
                topk_ids,
                active_num=num_tokens * top_k,
                expert_num=num_experts,
                expert_tokens_num_type=1,
                expert_tokens_num_flag=True,
                active_expert_range=[0, num_experts],
                quant_mode=-1,
            )
        )
        expert_tokens = expert_tokens.to(torch.int64)

        if _dbg:
            _t = hidden_states.float()
            logger.info(
                "[DBG NPU EXPERT] L%03d after_routing shape=%s norm=%.5f mean=%.6f std=%.6f sum=%.6f",
                _lid, list(_t.shape), _t.norm().item(), _t.mean().item(), _t.std().item(), _t.sum().item(),
            )
            logger.info(
                "[DBG NPU EXPERT] L%03d expert_tokens(cumsum)=%s",
                _lid, expert_tokens.tolist(),
            )
            torch.save(
                {"dispatched_x": _t.cpu(), "expert_tokens": expert_tokens.cpu(),
                 "topk_ids": topk_ids.cpu(), "topk_weights": topk_weights.float().cpu()},
                f"npu_expert_routing_L{_lid:03d}.pt",
            )

        w13_bias = [layer.w13_weight_bias] if self.with_bias else None
        w2_bias = [layer.w2_weight_bias] if self.with_bias else None

        # gmm1: gate_up_proj
        hidden_states = torch.ops.npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=[layer.w13_weight],
            bias=w13_bias,
            split_item=2,
            group_list_type=1,
            group_type=0,
            group_list=expert_tokens,
            output_dtype=original_dtype,
        )[0]

        if _dbg:
            _t = hidden_states.float()
            logger.info(
                "[DBG NPU EXPERT] L%03d after_gmm1(gate_up) shape=%s norm=%.5f mean=%.6f std=%.6f sum=%.6f",
                _lid, list(_t.shape), _t.norm().item(), _t.mean().item(), _t.std().item(), _t.sum().item(),
            )
            torch.save({"gmm1_out": _t.cpu()}, f"npu_expert_gmm1_L{_lid:03d}.pt")

        # act_fn:
        if self.moe_runner_config.activation == "npu_swiglu_oai":
            from sgl_kernel_npu.activation.swiglu_oai import swiglu_oai

            hidden_states = swiglu_oai(layer, hidden_states)
        elif self.moe_runner_config.activation == "silu":
            if self.moe_runner_config.gemm1_clamp_limit is not None:
                from sgl_kernel_npu.activation.swiglu_quant import swiglu_quant

                hidden_states, _ = swiglu_quant(
                    hidden_states,
                    group_list=expert_tokens,
                    group_list_type=1,
                    need_quant=False,
                    do_limit=True,
                    limit=self.moe_runner_config.gemm1_clamp_limit,
                )
            else:
                hidden_states = torch.ops.npu.npu_swiglu(hidden_states)
        else:
            from sglang.srt.layers.activation import GeluAndMul

            hidden_states = GeluAndMul()(hidden_states)

        if _dbg:
            _t = hidden_states.float()
            logger.info(
                "[DBG NPU EXPERT] L%03d after_swiglu(act) shape=%s norm=%.5f mean=%.6f std=%.6f sum=%.6f",
                _lid, list(_t.shape), _t.norm().item(), _t.mean().item(), _t.std().item(), _t.sum().item(),
            )
            torch.save({"act_out": _t.cpu()}, f"npu_expert_act_L{_lid:03d}.pt")

        # gmm2: down_proj
        hidden_states = torch.ops.npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=[layer.w2_weight],
            bias=w2_bias,
            split_item=2,
            group_list_type=1,
            group_type=0,
            group_list=expert_tokens,
            output_dtype=original_dtype,
        )[0]

        if _dbg:
            _t = hidden_states.float()
            logger.info(
                "[DBG NPU EXPERT] L%03d after_gmm2(down) shape=%s norm=%.5f mean=%.6f std=%.6f sum=%.6f",
                _lid, list(_t.shape), _t.norm().item(), _t.mean().item(), _t.std().item(), _t.sum().item(),
            )
            torch.save({"gmm2_out": _t.cpu()}, f"npu_expert_gmm2_L{_lid:03d}.pt")

        final_hidden_states = torch.ops.npu.npu_moe_finalize_routing(
            hidden_states,
            skip1=None,
            skip2=None,
            bias=None,
            scales=topk_weights,
            expanded_src_to_dst_row=expanded_row_idx,
            export_for_source_row=topk_ids,
            drop_pad_mode=2,
        )

        if _dbg:
            _t = final_hidden_states.float()
            logger.info(
                "[DBG NPU EXPERT] L%03d after_finalize shape=%s norm=%.5f mean=%.6f std=%.6f sum=%.6f",
                _lid, list(_t.shape), _t.norm().item(), _t.mean().item(), _t.std().item(), _t.sum().item(),
            )
            torch.save({"finalize_out": _t.cpu()}, f"npu_expert_finalize_L{_lid:03d}.pt")

        return StandardCombineInput(hidden_states=final_hidden_states)

    def _forward_npu_deepep(
        self,
        layer: torch.nn.Module,
        dispatch_output: DispatchOutput,
    ) -> CombineInput:
        from sglang.srt.hardware_backend.npu.quantization.fused_moe_method_npu import (
            npu_fused_moe_without_routing_weights_bf16,
        )
        from sglang.srt.layers.moe.token_dispatcher import (
            DeepEPLLCombineInput,
            DeepEPNormalCombineInput,
        )
        from sglang.srt.layers.moe.token_dispatcher.base import DispatchOutputChecker

        # NOTE: Ascend's Dispatch & Combine does not support FP16
        output_dtype = torch.bfloat16
        group_list_type = 1

        if DispatchOutputChecker.format_is_deepep_normal(dispatch_output):
            hidden_states, _, _, _, num_recv_tokens_per_expert = dispatch_output
            group_list = torch.tensor(
                num_recv_tokens_per_expert,
                dtype=torch.int64,
                device=hidden_states.device,
            )
            combine_cls = DeepEPNormalCombineInput
        else:
            hidden_states, _, _, _, group_list, _ = dispatch_output
            group_list = group_list.to(torch.int64)
            combine_cls = DeepEPLLCombineInput

        hidden_states = npu_fused_moe_without_routing_weights_bf16(
            layer, hidden_states, group_list_type, group_list, output_dtype
        )
        return combine_cls(
            hidden_states=hidden_states,
            topk_ids=dispatch_output.topk_ids,
            topk_weights=dispatch_output.topk_weights,
        )

    def forward_tpu(self, *args, **kwargs) -> CombineInput:
        raise NotImplementedError("The TPU backend currently does not support MoE.")

    def forward_musa(self, *args, **kwargs) -> CombineInput:
        return self.forward_cuda(*args, **kwargs)

    forward_native = forward_cpu
