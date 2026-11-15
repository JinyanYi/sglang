"""Validate cache gates and dataset-only adaptation of native bench_serving."""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).parents[3] / "benchmark/hisparse/a5_copysfa_mtp"


def load(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    spec = importlib.util.spec_from_file_location(
        "bench_prefix_cache", ROOT / "bench_prefix_cache.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("count", [28, 64])
@pytest.mark.parametrize("rounds,bad_warmup", [(1, False), (2, False), (1, True)])
def test_one_wave_native_metrics_and_full_cache_gate(
    tmp_path, monkeypatch, capsys, count, rounds, bad_warmup
):
    bench = load(monkeypatch)
    probes = []
    flushes = []
    waves = []
    tokenizer = SimpleNamespace(encode=lambda *a, **kw: [11, 17, 23])
    transformers = ModuleType("transformers")
    transformers.AutoTokenizer = SimpleNamespace(
        from_pretrained=lambda *a, **kw: tokenizer
    )
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setattr(bench, "post", lambda *a: flushes.append(a))

    def probe(url, ids, output, timeout, index, lock, active, target, rank):
        probes.append((rank, list(ids)))
        return dict(cached_tokens=39936, prompt_tokens=40000, completion_tokens=1)

    monkeypatch.setattr(bench, "stream_one", probe)
    serving = ModuleType("sglang.benchmark.serving")

    def original_dataset(*a):
        return None

    serving.get_dataset = original_dataset
    serving.DatasetRow = lambda **kw: SimpleNamespace(**kw)
    result = dict(
        completed=count,
        duration=12.3,
        output_throughput=123.456,
        mean_tpot_ms=7.89,
        input_lens=[40000] * count,
        output_lens=[1024] * count,
        cached_tokens=[39936] * count,
        errors=[""] * count,
    )
    output = tmp_path / "bench-serving.jsonl"

    def cli():
        argv = sys.argv
        assert "--profile" not in argv and "--flush-cache" not in argv
        assert argv[argv.index("--warmup-requests") + 1] == "0"
        assert argv[argv.index("--num-prompts") + 1] == str(count)
        assert argv[argv.index("--max-concurrency") + 1] == str(count)
        import os

        assert "SGLANG_IS_IN_CI" not in os.environ
        args = SimpleNamespace(dataset_name="random")
        rows = serving.get_dataset(args, tokenizer, "model")
        assert len(rows) == count
        assert all(row.prompt == probes[0][1] for row in rows)
        assert [row.extra_request_body["routed_dp_rank"] for row in rows] == [
            i % 4 for i in range(count)
        ]
        wave_output = Path(argv[argv.index("--output-file") + 1])
        is_warmup = wave_output.name.startswith("decode-warmup-")
        wave_result = dict(result)
        if is_warmup:
            wave_result["output_throughput"] = 1.0
            if bad_warmup:
                wave_result["cached_tokens"] = [39808] + [39936] * (count - 1)
        waves.append(wave_output)
        wave_output.write_text(json.dumps(wave_result) + "\n")

    serving.cli_main = cli
    package = ModuleType("sglang.benchmark")
    package.serving = serving
    monkeypatch.setitem(sys.modules, "sglang.benchmark", package)
    monkeypatch.setenv("SGLANG_IS_IN_CI", "1")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench",
            "--model-path",
            "model",
            "--concurrency",
            str(count),
            "--decode-warmup-rounds",
            str(rounds),
            "--output",
            str(output),
        ],
    )
    if bad_warmup:
        with pytest.raises(RuntimeError, match="decode_warmup_1 failed validation"):
            bench.main()
        assert len(waves) == 1
        assert not output.exists()
        assert not json.loads(
            (tmp_path / "decode-warmup-1.validation.json").read_text()
        )["valid"]
        assert serving.get_dataset is original_dataset
        return
    assert bench.main() == 0
    assert len(waves) == rounds + 1
    assert waves[-1] == output
    assert len(flushes) == 1
    assert flushes[0][0].endswith("/flush_cache")
    assert [rank for rank, ids in probes] == list(range(4)) * 2
    assert serving.get_dataset is original_dataset
    assert json.loads(output.read_text()) == result
    assert json.loads((tmp_path / "cache-validation.json").read_text())["valid"]
    manifest = json.loads((tmp_path / "workload.json").read_text())
    assert len(manifest["decode_warmup_results"]) == rounds
    assert manifest["measurement"]["output_throughput"] == 123.456
    assert all(
        row["output_throughput"] == 1 for row in manifest["decode_warmup_results"]
    )
    assert not manifest["flush_between_decode_warmup_and_measurement"]
    console = capsys.readouterr().out
    measured_line = next(
        line
        for line in console.splitlines()
        if line.startswith("NATIVE_BENCH_SERVING_RESULT")
    )
    assert '"output_throughput": 123.456' in measured_line


def test_decode_warmup_cannot_be_disabled(tmp_path, monkeypatch):
    bench = load(monkeypatch)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench",
            "--model-path",
            "model",
            "--concurrency",
            "28",
            "--decode-warmup-rounds",
            "0",
            "--output",
            str(tmp_path / "result.jsonl"),
        ],
    )
    with pytest.raises(SystemExit) as exc:
        bench.main()
    assert exc.value.code == 2


@pytest.mark.parametrize(
    "field,value",
    [("cached_tokens", 39808), ("output_lens", 12), ("input_lens", 39999)],
)
def test_one_bad_request_invalidates_whole_run(monkeypatch, field, value):
    bench = load(monkeypatch)
    result = dict(
        completed=2,
        input_lens=[40000] * 2,
        output_lens=[1024] * 2,
        cached_tokens=[39936] * 2,
    )
    result[field][1] = value
    assert bench.validate_result(result, 2, 40000, 1024, 39936)
