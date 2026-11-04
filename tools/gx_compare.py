#!/usr/bin/env python3
"""Compare two GX captures (greedy-exactness). Exit non-zero on any diff.

Usage: gx_compare.py gx-7.0.json gx-7.2.json
"""
import json
import sys


def main():
    left = json.load(open(sys.argv[1]))
    right = json.load(open(sys.argv[2]))
    failures = 0
    for a, b in zip(left["rows"], right["rows"]):
        if a["full"] == b["full"]:
            print(f"  {a['name']}: IDENTICAL ({len(a['full'])} chars)")
            continue
        common = 0
        for x, y in zip(a["full"], b["full"]):
            if x != y:
                break
            common += 1
        failures += 1
        print(f"  {a['name']}: DIVERGES at char {common}\n"
              f"    left : {a['full'][common:common + 60]!r}\n"
              f"    right: {b['full'][common:common + 60]!r}")
    print("GX VERDICT:", "PASS (identical)" if failures == 0
          else f"FAIL ({failures} rows differ)")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
