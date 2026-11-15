# A5 C8 ring-tail integration

This branch integrates the running A5 backend with the framework structure in
SGLang PR #47383. Its companion kernel branch starts from sgl-kernel-npu PR #909.
Both repositories use `integrate/a5-c8-ring-pr47383-pr909-20261115`.

## Source pair

- Framework PR base: `0ea5775c1f2a0a19f5ed4985a494836d56063f21`.
- Kernel PR base: `8130d51342b95d349cc75b3e637cb5ba4983f2ab`.
- A5 framework donor: `b9f2971b977b61bad78965c6776485538b0f3051`.
- A5 kernel donor: `c7dc9ece6fc06020ddad1f9c73313ec210e59096`.

No A3 experiments outside these two PRs are included. The donor branches remain
the reference for device behavior; this integration has not passed A5 acceptance.

## Preserved A5 contract

The target uses FP8 index query/key with real FP32 scales, opaque 656-byte C8 KV,
Q1 decode and Q4 chain verification. Draft KV remains native. Historical C slots
and the separate 256-token ring retain their existing residency protocol.
LIM emits prefix-relative logical tail slots. The two final HBM page-table
entries rotate those slots onto `C + position % 256`; CopySFA does not remap them
a second time. Native physical token IDs continue to address Host KV across
radix aliases, accepted-token relocation, retraction and MF P/D transfer.

The C8 budget declares persistent manager/adapter/query/tail buffers before
allocation, binds Host capacity to the native index pool's physical pages, and
reserves the actual per-layer index scales. BF16 fused attention keeps the PR's
existing manager and default metadata policy.

## Fusion stages

Prefill, decode and target verification share the latest framework publication
kernel through `prepare_sparse_kv_publication_rows`. It computes source and
destination rows, positions, valid masks, physical bounds and TP suffix padding
in one metadata launch. The metadata-only entry retains the opaque C8 payload;
it does not concatenate unused BF16 KV. The public A3 payload adapter keeps its
existing prefill tuple and decode concatenation behavior. Draft KV stays native.

C8 pre/post LIM metadata lives in `c8_metadata.py`, with admission and optional
chain diagnostics in `c8_mtp.py`. The A3 manager, budget, MTP module and public
pre/post metadata retain the latest PR implementations. C8 pre-metadata derives
its stable page prefix from actual positions; post-metadata rotates HBM pages
and clears invalid source/slot rows in one launch. Accepted-token moves and
snapshots remain A5-local and preserve all 656 bytes and the FP8 index scales.

Two native candidate flags are opt-in and default to false:

- `SGLANG_NPU_A5_C8_SPLIT_QUERY=1` requests the v2 split-Q interface.
- `SGLANG_NPU_A5_C8_FUSED_RING_TAIL=1` additionally defers ring restoration and
  current-KV writes into CopySFA. It requires split query.

The installed wheel must advertise schema_version 1, packed_row_bytes 656 and the
requested capability. Unsupported query head/stride layouts retain the old
packing and external staging. A missing capability fails at startup when a new
flag is explicitly enabled. Retaining the fallback query/tail workspaces keeps
the memory reservation valid for either route.

These candidate flags do not establish A5 numerical correctness or speed. Kernel
build paths may expose different capabilities; inspect the companion build notes
and installed capability result before enabling them.

This checkpoint uses the legacy `build.sh -a kernels Ascend950` route, as invoked
by `10_build_kernel.sh`, with `SGL_KERNEL_A5_C8_BACKEND=original`. The standalone
modern/refactor 3510 build currently omits the original OPP and 19-tensor LIM
interface and cannot boot this C8 backend. Its zero fusion capabilities do not
mean that it is an alternative complete A5 wheel.

Both architectures compile and register the shared `unidex_copy` and private
fused SHM helpers. The generic SHM allocator and slot-map operator remain A3-only.

## Validation boundary

CPU checks cover persistent-buffer sizing, physical Host snapshot/retraction,
pre/post metadata arithmetic, v2 argument order and fallback ownership. They do
not exercise the Ascend launcher ABI or execute attention arithmetic. The Ascend
Triton 3.2 CPU interpreter requires a test-only correction for scalar false store
masks; the fixture preserves production kernel semantics without changing the
device implementation.

The update has CPU regression coverage for publication arithmetic, C8
budget/dispatch/lifecycle and framework routes, plus A3 eager and changed-input
graph checks for publication and C8 integer metadata. These checks validate the
shared metadata on A3, not Ascend950 quantized LIM or attention execution. The
older A5/A3 step profiles supplied by the user are diagnostic references, not a
matched performance comparison or acceptance target.

Before deployment, build the paired wheel with an A5-compatible CANN toolchain,
then validate original LIM and C8 attention eagerly and under changed-input graph
replay. Cover Q1/Q4, page crossings, invalid/padded rows, acceptance/rejection
rewrites, request reuse and shared radix prefixes. Compare v2 split-only and
internal-ring modes independently against the retained path. MF byte transfer,
P/D correctness and matched unprofiled serving measurements are separate gates.
Historical donor test/benchmark results are not acceptance for this branch.
