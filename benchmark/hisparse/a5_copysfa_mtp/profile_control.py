#!/usr/bin/env python3
"""Invoke SGLang's explicit profiler endpoints without changing server launch."""

from __future__ import annotations

import argparse
import json
import urllib.request


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("start", "stop"))
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--num-steps", type=int, default=20)
    args = parser.parse_args()
    endpoint = "/start_profile" if args.action == "start" else "/stop_profile"
    payload = {}
    if args.action == "start":
        if not args.output_dir:
            parser.error("--output-dir is required for start")
        payload = {
            "output_dir": args.output_dir,
            "activities": ["CPU", "GPU"],
            "record_shapes": True,
            "with_stack": False,
            "detailed_annotations": True,
            "profile_id": "a5-copysfa-mtp",
            "num_steps": args.num_steps,
        }
    request = urllib.request.Request(
        args.base_url.rstrip("/") + endpoint,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=900) as response:
        print(response.read().decode(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
