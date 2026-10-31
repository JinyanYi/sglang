#!/usr/bin/env python3
"""Validate target/draft NPU KV allocation roles in a completed startup log."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

PATTERN = re.compile(
    r"NPU MLA attention KV allocation: role=(draft|target) offload=(True|False) "
    r"CKV_bytes=(\d+) KPE_bytes=(\d+)"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--mtp", type=int, choices=(0, 1), default=1)
    args = parser.parse_args()
    text = args.log.read_text(errors="replace")
    records = [
        (role, offload == "True", int(ckv), int(kpe), match.group(0))
        for match in PATTERN.finditer(text)
        for role, offload, ckv, kpe in [match.groups()]
    ]
    drafts = [item for item in records if item[0] == "draft"]
    targets = [item for item in records if item[0] == "target"]
    # C8 embeds RoPE in each CKV record; its separate KPE allocation is empty.
    if args.mtp and not drafts:
        raise RuntimeError("MTP=1 must allocate draft KV")
    if not args.mtp and drafts:
        raise RuntimeError("MTP=0 must not allocate draft KV")
    if not all(not offload and ckv > 0 for _, offload, ckv, _, _ in drafts):
        raise RuntimeError(
            "Every draft allocation must have offload=False and positive CKV bytes"
        )
    if not targets or not all(offload for _, offload, _, _, _ in targets):
        raise RuntimeError("Every target allocation must have offload=True")
    for record in records:
        print(record[-1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
