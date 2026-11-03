# Same-version SGLang native A5 baseline

The primary result is measured client output tokens divided by elapsed time
from `50_perf.sh`. Profiling and the article's assumed-acceptance formulas are
secondary diagnostics and must not replace measured serving throughput.

This launcher uses the same framework, installed kernel wheel, GLM-5.2
W4A4C8/MXFP4 model, FP8 KV, DP4/attention-TP2/EP8, MTP3, radix cache, DeepEP
auto/hybrid, page size, context limit, prefill settings and graph backend as
the completed offload case. It explicitly sets offload and CopySFA to zero,
selects the legacy/native indexer, and clears both LIM metadata switches.
The native FP8 LI dispatcher fix remains active. Loading a wheel that also
contains LIM/CopySFA does not mean those operators are executed.

Stop the previous owned service and verify that its workers released the cards.
Then run in the existing A5 container:

```bash
cd /mnt/share/y00970600/personal/sglang
git pull --ff-only origin feat/a5-copysfa-mtp-radix-20261031
export BASELINE_PER_DP_BATCH_SIZE=7
export MEM_FRACTION_STATIC=0.80
bash benchmark/hisparse/a5_copysfa_mtp/41_launch_no_offload.sh
```

`--dry-run` prints the resolved command without importing an NPU package or
starting the service. No rebuild is required for these shell/Python changes.
Actual A5 startup, memory capacity and throughput still require manual testing.

The default seven live requests per DP follows Table 7's reference row. It is
not a measured capacity maximum for this machine or this shared-prefix workload.
The graph bucket is eight because TP2 draft decode requires an even token
bucket. Total live concurrency is **28**, not 32; throughput uses real completed
output tokens. The request admission limit remains seven per DP. DeepEP capacity
stays at 64 as in the offload run, covering the padded graph bucket's Q4 inputs.

After readiness, run the actual performance test from another terminal:

```bash
cd /mnt/share/y00970600/personal/sglang
export BASELINE_PER_DP_BATCH_SIZE=7
source benchmark/hisparse/a5_copysfa_mtp/article_no_offload.env.example
export no_proxy="localhost,127.0.0.1,::1${no_proxy:+,$no_proxy}"
export NO_PROXY="$no_proxy"
RUN_DIR="$RUN_ROOT/$(date -u +%Y%m%dT%H%M%SZ)-native-b7-serving-perf" \
  bash benchmark/hisparse/a5_copysfa_mtp/50_perf.sh
```

This uses the same 256 requests, 40,000 input tokens, 1,024 output tokens and
shared-prefix generation/radix priming as the earlier offload run. The JSON
records completed count, output-token count, elapsed seconds, actual output
tokens/s, TTFT and TPOT. Preserve the command, server log and JSON. A failed
request or unexpected scheduler admission limit is not a valid performance run.

Also run a **same-concurrency comparison** at per-DP BS16 (global 64), if the
native service can support it: stop the BS7 instance, set
`BASELINE_PER_DP_BATCH_SIZE=16` in both terminals and repeat launch and testing.
Do not run the two instances on the same cards concurrently. If BS16 fits this
shared-prefix workload, do not claim that BS7 was the native capacity ceiling.
If it fails, retain the actual failure and memory evidence rather than silently
reducing concurrency. The allocator's token capacity must be inspected before
calling any setting the maximum for independent 40K contexts.

The earlier offload experiment and this directly comparable client test share
39,872 input-prefix tokens. Independent-context capacity is a separate workload;
these results do not prove the article's independent `BS*S` KV residency.
For a controlled optional KV cap use `BASELINE_MAX_TOTAL_TOKENS`; otherwise the
launcher clears an inherited `MAX_TOTAL_TOKENS` and lets the allocator budget it.

Optional no-stack profiling uses the same native environment:

```bash
AUTO_PROFILE=1 STEADY_NUM_PROMPTS="$CONCURRENCY" \
  bash benchmark/hisparse/a5_copysfa_mtp/55_steady_decode.sh
```

`KV_OFFLOAD=0` selects a native LI/SFA trace gate: no LIM or CopySFA records may
be present. It does not weaken the joint gate used for offload runs. For the
offline text report, pass `--batch 7` (live requests) rather than graph bucket 8:

```bash
bash benchmark/hisparse/a5_copysfa_mtp/66_article_report.sh \
  /path/to/native-case --batch 7
```

Native target/draft allocation logs can be checked with
`KV_OFFLOAD=0 SERVER_LOG=/path/to/server.log bash .../42_check_startup_log.sh`.
Both models must show `offload=False` and positive HBM CKV bytes. The launcher
does not change the system/container environment or stop other services.
