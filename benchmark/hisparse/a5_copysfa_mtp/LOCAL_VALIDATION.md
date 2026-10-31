# Preparation evidence (2026-10-31)

This pair was prepared on Windows without CANN or A5 hardware. It is ready for
manual validation, not a claim that A5 model serving has passed.

## Sources and scope

- Framework base: `4d1fefd6b79f214f908359e032f96ba3a64137ee`.
- Kernel base: `db77f6fd2db26b2e60a02c63f821121045482835`.
- Both bases are from `fix/mtp3-draft-hbm-tail-20261031` in the personal forks.
- McZyWu A5 reference: framework `801ffd28efb8f5673c53aa96634fc53137ffd73f`
  and kernel `503f8a2f432b57fb56fc3403833192f35469e1dd`, branch
  `feat/hisparse-lim-copysfa-c8-a5-10-26`.
- CopySFA payload reference: vllm-ascend PR #18490,
  `1b180deb840484c2fe33cbdd7abb68a615085172`. The kernel directory retains
  per-file provenance and licenses.

The target uses packed 656-byte C8 KV, physical radix host page tables and its
own C-token HBM pool plus a 256-token causal tail. Draft native HBM ownership
and accepted-token relocation come from the requested base. The new caller
bridge consumes an external 19-Tensor A5 LIM; no A5 LIM kernel is included.
The optional external DSO must not register a second CopySFA operator.

## Completed local checks

Python 3.12.14 with CPU PyTorch was used for:

```bash
python -m pytest -q \
  test/manual/npu/test_a5_copy_sfa_contract.py \
  test/manual/npu/test_sparse_kv_offload_cache.py \
  test/manual/npu/test_a5_copy_sfa_metadata.py \
  test/manual/npu/test_a5_external_lim_contract.py
# exit 0: 27 passed, 2 skipped, 19 subtests passed

python -m pytest -q \
  tests/test_fused_quant_scatter_copy_sparse_flash_attention_cpu.py \
  tests/test_fused_quant_scatter_copy_sparse_flash_attention.py \
  tests/test_fused_quant_scatter_copy_sparse_flash_attention_a5.py
# in the kernel checkout; exit 0: 8 passed, 2 skipped
```

The four skipped modules require `torch_npu`. CPU checks cover the caller ABI,
real scale forwarding, Q4 endpoints, invalid-row isolation, prefix-relative
tail page rotation, budgeting, full-byte accepted relocation, independent
packed attention fixtures and imported-source hashes. They do not execute
the native CopySFA or Triton kernels. Framework pytest also reported the
environment's unknown `asyncio_mode` config warning.

Repository-pinned framework isort 7.0.0, Ruff 0.15.1 formatting and
`F401,F821,F823,UP037` checks pass. Kernel isort 5.13.2, Black 24.10.0,
clang-format 18.1.8 for owned C++ files and supplemental Ruff F401 pass.
Codespell 2.4.1 and `git diff --check` pass in both repositories. All 11
handoff shell scripts pass `bash -n`; Python files parse; launch dry-run passes.
The vendored kernel payload is excluded from formatting and its 36 recorded
hashes pass. Generic pre-commit orchestration was not run.

A broader framework Ruff check additionally reports seven existing E402,
E712 and E741 findings outside the changed lines; these are outside the
repository's selected lint rules. The Linux runtime draft-policy test cannot
collect on this Windows environment (`ModuleNotFoundError: resource`). Its
Linux execution remains in `20_test_cpu.sh`.

Raw local logs are under the sibling preparation artifact directory
`../evidence/` outside both Git checkouts: `framework_pytest.log`,
`kernel_pytest.log`, `kernel_quality.txt` and framework quality receipts.

## Required A5 checks

Follow [README.md](README.md) in order: native build, standalone CopySFA
numerical and replay tests, metadata/tail tests, actual external-LIM tail ABI,
draft/target allocation-role logs, 64K cold/hot semantic radix checks, then
unprofiled performance and a separate steady-decode trace.

Native compilation, queued-launch/workspace ownership under graph replay,
A5 accuracy, MTP acceptance, radix correctness and end-to-end performance
remain unverified until those stages run on A5.
