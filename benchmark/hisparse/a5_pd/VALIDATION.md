# Candidate validation, 2026-11-04

Validated on the local Windows CPU host only. There is no A5 device attached.

Command, from the framework root with an isolated CPU torch installation:

```bash
python -m pytest -q \
  test/manual/npu/test_a5_pd_cpu.py \
  test/manual/npu/test_a5_prefix_bench_cpu.py \
  test/manual/npu/test_a5_no_offload_launcher_cpu.py \
  test/manual/npu/test_a5_joint_handoff_cpu.py
```

Result: **47 passed, zero skipped/failures/errors, exit0**. The 17 new tests
cover real CPU tensor bytes with a simulated MF external API: owner/peer GVA
aliasing, 2MiB allocation alignment, 656-byte page offsets, masked cross-page
copy/error propagation, overlapping accepted-token moves, retraction snapshots,
owner-only shared restore, per-rank index/scales, LIM invalidation, transfer
entry ordering, P/D/draft policy, native/graph/MTP/radix launcher configuration,
and rejection of incorrect greedy output/cache/retraction evidence. The other
30 checks retain the colocated warmup/native-benchmark and handoff contracts.
One pre-existing pytest warning reports an unknown `asyncio_mode` option.

Also passed on the changed Python/shell files:

- Python compilation, repository-selected Ruff checks and Ruff formatting.
- isort import ordering.
- Bash `-n` and real `--dry-run` P/D launcher execution.
- `git diff --check` in both repositories.

The local JUnit evidence is retained outside the checkout at
`../a5-pd-cpu.xml`. It does not contain hardware timing or numerical A5 results.

Not executed: A5 MF GVA/AICore readback, TP sharing and graph publication,
LIM+CopySFA with MF storage on A5, inter-node HBM-to-DRAM transfer, full-model
P/D greedy/cache/retraction gates, and performance/profiling. These gates and
the exact commands are in README.md. The existing numerical oracle and its
tolerances were reused without changes. API presence or HTTP readiness alone
must not be reported as P/D acceptance.

No operator source/binding changes are made in the paired kernel branch.
No A5 build is claimed; rebuilding may be required if the deployment changes
the installed CANN/torch ABI. Exact A5 MF/HDK/firmware/LCNE compatibility remains
a provider/runtime prerequisite, not a result of the CPU checks.
