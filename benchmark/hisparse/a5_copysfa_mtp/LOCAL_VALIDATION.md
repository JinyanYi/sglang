# Preparation evidence (2026-10-31)

This pair was prepared on Windows and subsequently checked in a private A3
container on host 88 with CANN 9.2. A5 hardware execution remains manual; no
A5 model-serving, numerical-parity or latency result is claimed here.

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

## A3 device checks on 2026-10-31

The private container used Python 3.12.15, Torch 2.13.0+cpu, Torch-NPU 2.13.0,
Triton 3.6.0 and CANN 9.2.0. The full Ascend910_9382 kernel build exited 0.
All suites below exited 0 with zero skipped tests:

- Native A3 CopySFA combined, dynamic TND and resident refresh: 42 passed.
- Native LIM/MTP full-visible, CopySFA integration and vendor parity: 32 passed.
- Framework CPU caller/cache contracts: 27 passed.
- Packed metadata and full 656-byte transport: 20 passed, including
  C8192/16384/32384, Q1/Q4 and eager/graph.
- Extended packed graph transport: 6 cases, each 128 changed-input replay
  rounds, passed. These cover acceptance, rollback, request reuse, shared
  immutable prefix pages and private mutable tails.
- Linux runtime allocation/draft policy: 34 passed. The first run exposed a
  test-only `__new__` fixture missing `separate_mtp_tail`; setting its normal
  constructor default fixed the fixture. The original failure log is retained.
- Kernel CPU golden/provenance and benchmark fixtures: 17 passed locally.
- Original-reference registered host header and native byte transport: 2
  passed on A3, each with 16 changed-data rounds, including graph replay.

Additional probes passed: original CANN serialization versus the port is now
byte-exact at 80 bytes; all 38 comparison fixture constructions passed a CPU
source/slot/payload/causality audit; the production owned-launch helper passed
64 queued ephemeral-input launches and 64 graph replays with an A3 copy kernel.
None of these probes executes A5 attention.

The final explicit batch-launch helper additionally passed 192 queued eager
calls and 192 graph replays with an A3 byte-copy kernel, including two streams.
The redirected callback executed 197 times including warmups and captures.
This checks the production launch config and owner submission; it does not
test A5 attention or establish A5 concurrent-kernel safety.

The serialization test found a real port bug: the earlier direct-launch tiling
layout occupied 96 bytes instead of the original 80. Kernel commit
`be616002f22b22552e3ccb6b68b94b59acb84c3f` fixes nested padding and adds the
original-CANN byte regression. Pull this commit or a later branch head before
building on A5. The original vendor payload is unchanged.

The original PR18490 OPP also cross-built and installed successfully in a
private vendor directory on A3. Its registered-memory wrapper resolved both
ACLNN symbols to that private library. Generic Torch copies are unsuitable
for its borrowed NPU header; the shipped transport test uses the native byte
copy operator over the actual registered host pointer.

The offline source/preparation/OPP/binding pipeline from kernel `5d8b826`
completed without manual source edits; the final reference then passed all
19 CPU/registered-transport tests. The selected-Python follow-up was checked
against the actual interpreter, Torch and Torch-NPU environment.

## Ascend950 cross-build and host contract

The full Ascend950PR_9599 wheel cross-build with the corrected tiling header
exited 0. Its library loaded on A3 for Meta-only checks: 11 passed, zero skips.
No native A5 kernel was executed on A3.

Kernel `995130e5068d2e1c6426d0346c09c97fc4c0d9b5` adds explicit batch scheduling
to all four CopySFA host-stub entries, matching the original OPP. CANN 9.2's
legacy generated inline wrapper did not preserve an implementation-level
schedule attribute; the final launch instead supplies the public runtime
configuration directly. Other operators retain their existing launch policy.

This last change affected host code only. The host-stub object, static archive
and shared library were rebuilt using standard CMake Makefile `/fast` targets.
The already compiled device object remained byte-identical before and after,
SHA256 `a40dac34d6fe7272398f22eddb75f176f18cb28a0bc0ebd28438942e3d2bb262`.
All four stub relocations reference the batch handler; its definition is
present in the rebuilt host object. The repackaged wheel contains exactly the
same SO as the source package. Final load/Meta checks again passed 11/11 with
zero skips, and packaging exited 0:

- SO SHA256: `8d4c820d178165e84016791e806cc24c0bbbdf846216e06deffa16496b48b171`.
- Wheel SHA256: `f231c47c45bb88c75b3e62ce8cfdee4ebcc29fbe910b2706091240b98b02eeff`.

Retained intermediate failures are preparation issues, not hidden passes:
ordinary incremental CANN preprocessing hit `ld.lld: ...cpp.o: unknown file
type`; a private patch tar with mtime=0 initially left a stale host object and
an undefined handler at load; repeated Python packaging required the normal
`setup.py clean --all` step to clear staged providers. The final successful
object, relocation, load and wheel-hash checks above supersede those attempts.
Build changed device code normally from a fresh checkout/build directory;
the host-only shortcut is not a device-code validation procedure.

Final static checks pass with the repository-pinned formatters/lint versions,
codespell and `git diff --check`. All 12 handoff shell scripts pass `bash -n`.
All four MTP/GRAPH launch combinations passed dry-run checks, and six positive
and negative startup-role fixtures passed. The Linux draft-policy fixture fix
and the standalone comparison runner are included in the paired framework
branch. Generic pre-commit orchestration remains unrun.

Raw evidence is preserved outside the repositories at
`/home/y00970600/copysfa-vllm-align-20261031/logs/` on host 88 and locally under
`artifacts/copysfa-vllm-align-88-20261031/remote_snapshot_01/logs/` in the Windows
workspace. Relevant subdirectories are `tests-a3`, `runtime-policy`,
`runtime-policy-first-failure`, `final-cpu-reference` and
`tiling-layout-regression-01`. Standalone probe logs are
`owned-launch-probe-03.log` and `reference-view-probe-04.log`; each has an
adjacent exit-code file. `fixture-matrix-audit.json` records the 38 CPU cases.
Final additions are `batch-launch-probe`, `build-a5-02`, `build-a5-batch-fast`,
`build-a5-batch-host-final`, `package-final`, `a5-meta-final` and
`final-delivery`. Original-reference build/manifests are under the task root's
`original-a5-final/`. Failed preparation logs are retained separately.

The earlier build failure caused by CRLF was limited to the Windows-exported
private source packet. The Git blob for `build.sh` has LF endings; Linux Git
checkouts do not require a repository-wide line-ending change.

## Required A5 checks

Follow [README.md](README.md) in order: native build, standalone CopySFA
numerical and replay tests, metadata/tail tests, actual external-LIM tail ABI,
draft/target allocation-role logs, 64K cold/hot semantic radix checks, then
unprofiled performance and a separate steady-decode trace.

A5 queued-launch/workspace behavior, numerical accuracy, MTP acceptance,
model radix correctness and end-to-end performance remain unverified until
those stages run on A5. A3 metadata tests do not establish A5 equivalence.
The standalone original-versus-port comparison has 38 cases and treats
accuracy as a hard gate; a default latency regression above 20% is a warning.
