#!/usr/bin/env python3
"""单条测试 AIME26 最后 3 道难题，流式输出，temperature=0 便于 NPU/GPU 对比。

用法:
  # 先启动服务: bash accuracy_run.sh  (port 9909)
  python3 test_aime26_last3.py           # 默认跑 Q28/Q29/Q30
  python3 test_aime26_last3.py 28        # 只跑 Q28
  python3 test_aime26_last3.py 28 30     # 跑 Q28 和 Q30
  python3 test_aime26_last3.py --repeat 3 28  # 同一题重复 3 次，检查 temp=0 是否稳定

参数对齐 qwen3.6/eval.sh:
  - temperature=0, max_tokens=131072
  - extra_body.chat_template_kwargs.enable_thinking=true
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Iterable

# 与 accuracy_run.sh 一致
URL = "http://127.0.0.1:9999/v1/chat/completions"
MODEL = "/home/weights/Qwen3.6-35B-A3B"
MAX_TOKENS = 131072
TIMEOUT = 3600

DATASET = os.path.join(
    os.path.dirname(__file__), "..", "test_files", "aime26", "aime2026.jsonl"
)

PROMPT_TEMPLATE = (
    "Solve the following math problem step by step. "
    "Put your answer inside \\boxed{{}}.\n\n"
    "{question}\n\n"
    "Remember to put your answer inside \\boxed{{}}."
)


@dataclass(frozen=True)
class Question:
    idx: int
    problem: str
    answer: str


def load_dataset() -> dict[int, Question]:
    questions: dict[int, Question] = {}
    with open(DATASET, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            qid = int(row["id"])
            questions[qid] = Question(
                idx=qid,
                problem=row["problem"].strip(),
                answer=str(row["answer"]),
            )
    return questions


def build_prompt(problem: str) -> str:
    return PROMPT_TEMPLATE.format(question=problem)


def extract_answer(pred_str: str) -> str:
    """尽量复用 evalscope 的 boxed 提取逻辑。"""
    try:
        from evalscope.metrics.math_parser import extract_answer as evalscope_extract
        from evalscope.benchmarks.aime.math_normalize import normalize_answer

        extracted = evalscope_extract(pred_str)
        return normalize_answer(extracted)
    except Exception:
        pass

    if "boxed" not in pred_str:
        nums = re.findall(r"-?\d+", pred_str.replace(",", ""))
        return nums[-1] if nums else ""

    ans = pred_str.split("boxed")[-1]
    if not ans:
        return ""
    if ans[0] == "{":
        stack = 1
        buf = []
        for ch in ans[1:]:
            if ch == "{":
                stack += 1
                buf.append(ch)
            elif ch == "}":
                stack -= 1
                if stack == 0:
                    break
                buf.append(ch)
            else:
                buf.append(ch)
        return "".join(buf).strip()
    return ans.split("$")[0].strip()


def grade_answer(pred: str, target: str) -> bool:
    try:
        from evalscope.benchmarks.aime.grader import grade_answer as evalscope_grade

        return bool(evalscope_grade(pred, target))
    except Exception:
        return pred.strip() == target.strip()


def ask_stream(prompt: str) -> tuple[str, int]:
    payload = json.dumps(
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": MAX_TOKENS,
            "stream": True,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
        }
    ).encode()

    req = urllib.request.Request(
        URL,
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    parts: list[str] = []
    token_count = 0

    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue

            choice = chunk.get("choices", [{}])[0]
            delta = choice.get("delta", {})
            reasoning = delta.get("reasoning_content") or ""
            content = delta.get("content") or ""

            if reasoning:
                sys.stdout.write(reasoning)
                sys.stdout.flush()
                parts.append(reasoning)
            if content:
                sys.stdout.write(content)
                sys.stdout.flush()
                parts.append(content)

            usage = chunk.get("usage")
            if usage:
                token_count = usage.get("completion_tokens", 0)

    return "".join(parts), token_count


def run_once(q: Question, run_id: int | None = None) -> bool:
    header = f"Q{q.idx}  target={q.answer}"
    if run_id is not None:
        header += f"  run={run_id}"
    print(f"\n{'=' * 70}")
    print(header)
    print(f"{'=' * 70}\n")

    try:
        text, toks = ask_stream(build_prompt(q.problem))
    except urllib.error.URLError as e:
        print(f"\nERROR: 无法连接 {URL}: {e}", file=sys.stderr)
        print("请先执行: bash accuracy_run.sh", file=sys.stderr)
        return False
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        return False

    pred_raw = extract_answer(text)
    correct = grade_answer(pred_raw, q.answer)
    hit_cap = toks >= MAX_TOKENS

    print(f"\n\n{'=' * 70}")
    print(
        f"tokens={toks}  extracted={pred_raw!r}  target={q.answer!r}  "
        f"correct={'✓' if correct else '✗'}  {'⚠ 打满 max_tokens!' if hit_cap else ''}"
    )
    print("=" * 70)
    return correct


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="每道题重复次数，用于 temp=0 下检查输出是否稳定",
    )
    parser.add_argument(
        "indices",
        nargs="*",
        type=int,
        help="题号，默认 28 29 30（AIME26 最后 3 题）",
    )
    return parser.parse_args(list(argv))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    indices = args.indices or [28, 29, 30]

    questions = load_dataset()
    unknown = [i for i in indices if i not in questions]
    if unknown:
        print(f"未知题号 {unknown}，可选: {sorted(questions)}", file=sys.stderr)
        return 1

    results: list[tuple[int, bool]] = []
    for idx in indices:
        q = questions[idx]
        for run_id in range(1, args.repeat + 1):
            ok = run_once(q, run_id if args.repeat > 1 else None)
            results.append((idx, ok))

    print(f"\n{'#' * 70}")
    print("汇总")
    for idx in indices:
        runs = [ok for qid, ok in results if qid == idx]
        passed = sum(runs)
        total = len(runs)
        status = "✓" if passed == total else "✗"
        print(f"  Q{idx}: {passed}/{total} {status}")
    print("#" * 70)
    return 0 if all(ok for _, ok in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
