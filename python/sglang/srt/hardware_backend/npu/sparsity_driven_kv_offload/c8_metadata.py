"""Thin calls to the companion kernel package's A5 C8 operators."""

from sgl_kernel_npu.sparsity_driven_kv_offload import c8_metadata as kernels


def fused_prepare_lim_metadata(**kwargs):
    return kernels.fused_prepare_lim_metadata(**kwargs)


def fused_lim_copy_sfa_metadata(**kwargs):
    return kernels.fused_lim_copy_sfa_metadata(**kwargs)


def stage_mtp_tail(**kwargs):
    return kernels.stage_mtp_tail(**kwargs)
