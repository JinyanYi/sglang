# Offline article-style text report

Run after the saved no-stack capture finishes. No NPU runtime packages, model
loading, network requests, service restart, or new profiling are required.

```bash
cd /mnt/share/y00970600/personal/sglang
git pull --ff-only origin feat/a5-copysfa-mtp-radix-20261031
bash benchmark/hisparse/a5_copysfa_mtp/66_article_report.sh \
  /mnt/share/y00970600/a5-joint-runs/20261103T064132Z-e2e-mtp3-graph
```

The command prints progress while reading eight traces sequentially, then a
compact report suitable for pasting into chat. It writes `REPORT.txt` and
`article-report.json` under the case's `06-article-report` directory. Defaults
are eight ranks, attention DP4, per-DP BS16, twenty target rounds, and 78 layers.
Override these only when the saved experiment differs. A `--profile-dir` option
also supports profiling data outside the standard case layout.

The report follows the local `Offload.txt` sections 5.1.4 and 5.1.5 and Table 7:

- Select a unique graph model containing exactly 78 attention calls per target
  round. Draft models must not be counted as target layers.
- Split every model stream at its repeated first task signature. Reject partial
  stream captures, overlapping windows, and rounds without 78 attention calls
  and 21 LI/LIM calls. Preserve model IDs and stream anchors in the JSON.
- Report the elapsed first-to-last device task span of the selected target
  graph, and consecutive device-start intervals as ITL. CPU enqueue spans are
  not device elapsed times.
- Label the graph span as a **proxy**, not exact 78-layer execution: embedding,
  output head, or metadata may be inside the same captured graph. Exact layer
  boundaries need separate validated instrumentation or manual attribution.
- Sum component kernel durations only within that target graph. Include
  `AI_VECTOR_CORE`, which the older broad summary filter omitted. Duration sums
  for attention-related vector records are separate from primary AIC attention
  tasks: the primary `AI_CORE`/`MIX_AIC` count establishes layer count. Export
  raw names, task types, counts and sample task IDs to audit this distinction.
  Do not divide a doubled raw record count by two. Mixed engine types are also
  part of the stream anchor signature.
  Duration sums
  may overlap. Residual time subtracts the union of named article-candidate
  intervals and retains gaps. Dispatch/combine remain name-based candidates;
  generic MEMCPY is not automatically historical-KV loading.
- Calculate a clearly labeled article-assumption column using 3.0 output
  tokens per request per round. `TPOT=ITL/3` and
  `TPS_per_DP=BS/TPOT_ms*1000`. Each rank's `TPS_per_DP*DP` is only a balanced-DP
  estimate; TP ranks are not independent replicas and must not all be summed.
- Keep the unprofiled client metrics separate. A shared-prefix radix workload
  is not evidence for independent full-context KV capacity proportional to
  `BS*S`. Capacity comparisons need separately controlled residency workloads.
- The formal comparison baseline is the same SGLang version with offload
  disabled. This offline analysis does not run that baseline or claim speedup.

If same-window measured output tokens per request per scheduling round are
available, pass `--measured-tokens-per-round VALUE`. This creates a separate
formula result; it does not replace the article assumption. Neither a short
needle request's acceptance nor a whole-workload gauge establishes that value
for the profiling window.

Exit 0 means the graph proxy was structurally validated on the expected traces,
not that pure-layer boundaries, model accuracy, or no-offload comparison were
validated. Exit 2 means incomplete analysis; `REPORT.txt` still includes model
counts, task types, and a few device schema samples. Paste that report so the
parser can be adjusted without transferring full traces. A saved receipt with
`with_stack=true` is rejected; an absent receipt is explicitly marked unverified.
