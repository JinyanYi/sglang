#!/usr/bin/env python3
"""Reject skipped A5 cases or an absent graph/Q4/fusion joint test."""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path


def verify(paths, widths=(1, 4)):
    totals = dict(tests=0, failures=0, errors=0, skipped=0)
    names = set()
    for path in paths:
        root = ET.parse(path).getroot()
        for suite in root.iter("testsuite"):
            for key in totals:
                totals[key] += int(suite.get(key, 0))
        names.update(case.get("name") for case in root.iter("testcase"))
    required = {
        f"test_real_lim_to_copysfa[{graph}-{width}]"
        for graph in ("False", "True")
        for width in widths
    }
    missing = sorted(required - names)
    native_required = {
        f"test_general_query_{mode}_accuracy[q{width}]"
        for width in widths
        if width not in (1, 4)
        for mode in ("eager", "graph")
    }
    missing_native = sorted(native_required - names)
    valid = (
        totals["tests"] > 0
        and not missing
        and not missing_native
        and all(totals[key] == 0 for key in ("failures", "errors", "skipped"))
    )
    return dict(
        valid=valid,
        totals=totals,
        missing_joint_cases=missing,
        missing_native_query_cases=missing_native,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("xml", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--query-widths", type=int, nargs="+", default=[1, 4])
    args = parser.parse_args()
    result = verify(args.xml, args.query_widths)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
