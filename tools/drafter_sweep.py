#!/usr/bin/env python3
"""20-position parity sweep summary for the DFlash2 drafter (M1c, Phase E2).

Level 1 (BF16 vs upstream): runs the reference harness
`tools/qwen27b/dflash_reference.py` (in Gufo-mtp-perf, not modified here) on
every `pos-<N>` dir of the BF16 sweep with `--head reference`, then checks each
`report.json`. A position passes iff the harness reports `passed: true`
(every stage relative_rmse <= 1e-4, packed logits max_abs <= 1e-3, candidate
sets equal, full-forward total variation <= 1e-4).

Level 2 (Q6_K vs BF16): compares the native-head greedy proposals of the two
sweeps (`greedy-native.json`) elementwise. Passes iff >= 95% of the proposals
are identical (133 of 140 for 20 positions x 7 tokens).

The harness exits non-zero when `passed` is false but still writes
`report.json`; that is read as "report written, check passed". A missing
report is an error.

Outputs `--out` + `.md` and `--out` + `.json`, and prints the markdown table
and summary line to stdout.

Example:
  python3 tools/drafter_sweep.py \\
    --sweep-bf16 /tmp/opencode/sweep-bf16 --sweep-q6k /tmp/opencode/sweep-q6k \\
    --draft-bf16 /home/bruno/AI/GGUF/mtp/Flint-DFlash2-BF16.gguf \\
    --draft-q6k /home/bruno/AI/GGUF/mtp/Flint-DFlash2-Q6_K.gguf \\
    --target /home/bruno/AI/GGUF/flint-UD-Q4_K_XL.gguf \\
    --upstream /tmp/opencode/dflash-upstream/dflash/model.py \\
    --harness /home/bruno/AI/Gufo-mtp-perf/tools/qwen27b/dflash_reference.py \\
    --harness-cwd /home/bruno/AI/Gufo-mtp-perf \\
    --out /tmp/opencode/sweep-summary
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

POS_RE = re.compile(r"^pos-(\d+)$")
LEVEL1_MIN_PERCENT = 95  # Level 2 threshold: identical * 100 >= 95 * total
GREEDY_FILE = "greedy-native.json"


def discover_positions(sweep_dir):
    """Returns (position, path) pairs for pos-<N> dirs, sorted numerically."""
    root = Path(sweep_dir)
    if not root.is_dir():
        sys.exit(f"error: sweep dir not found: {root}")
    found = []
    for entry in root.iterdir():
        match = POS_RE.match(entry.name)
        if match and entry.is_dir():
            found.append((int(match.group(1)), entry))
    if not found:
        sys.exit(f"error: no pos-<N> dirs in {root}")
    return sorted(found)


def run_harness(args, trace_dir):
    """Runs the reference harness for one trace; returns the parsed report."""
    report_path = trace_dir / "report.json"
    if report_path.exists() and not args.force:
        print(f"skip (report exists): {trace_dir}", file=sys.stderr)
    else:
        command = [
            "nix", "develop", "-c", "python3", str(args.harness),
            "--upstream", str(args.upstream),
            "--config", str(trace_dir / "config.json"),
            "--target", str(args.target),
            "--draft", str(args.draft_bf16),
            "--trace", str(trace_dir),
            "--output", str(report_path),
        ]
        print(f"run: {trace_dir}", file=sys.stderr)
        # Non-zero exit means "check passed"; the report is authoritative.
        # stderr is kept only for diagnosis when no report is written.
        result = subprocess.run(
            command, cwd=args.harness_cwd, stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE, text=True, check=False,
        )
        if not report_path.exists():
            tail = "\n".join(result.stderr.strip().splitlines()[-20:])
            sys.exit(
                f"error: harness wrote no report for {trace_dir} "
                f"(exit {result.returncode})\n{tail}"
            )
    with open(report_path, encoding="utf-8") as handle:
        return json.load(handle)


def worst_stage(report):
    """Returns (\"mode.stage\", relative_rmse) with the largest relative_rmse."""
    worst_name, worst_value = None, -1.0
    for mode, stages in report["stages"].items():
        for stage, metrics in stages.items():
            value = metrics["relative_rmse"]
            if value > worst_value:
                worst_name, worst_value = f"{mode}.{stage}", value
    if worst_name is None:
        sys.exit("error: report has no stage metrics")
    return worst_name, worst_value


def load_greedy(sweep_dir):
    """Loads the native-head greedy proposals as {position: [ids]}."""
    path = Path(sweep_dir) / GREEDY_FILE
    if not path.exists():
        sys.exit(f"error: missing {path}")
    with open(path, encoding="utf-8") as handle:
        return {int(key): value for key, value in json.load(handle).items()}


def level2(bf16_greedy, q6k_greedy, positions):
    """Counts identical greedy proposals per position between the two quants."""
    per_position = []
    identical_total = 0
    total = 0
    for position in positions:
        bf16 = bf16_greedy.get(position)
        q6k = q6k_greedy.get(position)
        if bf16 is None or q6k is None:
            sys.exit(f"error: position {position} missing from greedy-native.json")
        if len(bf16) != len(q6k):
            sys.exit(
                f"error: position {position} greedy lengths differ "
                f"({len(bf16)} vs {len(q6k)})"
            )
        identical = sum(1 for a, b in zip(bf16, q6k) if a == b)
        identical_total += identical
        total += len(bf16)
        per_position.append({"position": position, "identical": identical,
                             "total": len(bf16)})
    passed = identical_total * 100 >= LEVEL1_MIN_PERCENT * total
    fraction = identical_total / total if total else 0.0
    return {
        "passed": passed,
        "identical": identical_total,
        "total": total,
        "fraction": fraction,
        "per_position": [{"position": p["position"], "identical": p["identical"]}
                         for p in per_position],
    }, per_position


def write_outputs(out_prefix, level1, level2_result, per_position_greedy):
    """Writes the markdown and JSON summaries and returns the markdown text."""
    lines = [
        "| position | BF16 worst stage | value | Level 1 | Q6_K==BF16 greedy (of 7) |",
        "|---|---|---|---|---|",
    ]
    for position, greedy in zip(level1["positions"], per_position_greedy):
        verdict = "PASS" if position["passed"] else "FAIL"
        lines.append(
            f"| {position['position']} | {position['worst_stage']} | "
            f"{position['worst_relative_rmse']:.3e} | {verdict} | "
            f"{greedy['identical']} |"
        )
    summary = (
        f"Level 1: {level1['positions_passed']}/{len(level1['positions'])} "
        f"positions pass. Level 2: {level2_result['identical']}/"
        f"{level2_result['total']} proposals identical "
        f"({level2_result['fraction'] * 100:.1f}%)."
    )
    markdown = "\n".join(lines) + "\n\n" + summary + "\n"

    out_path = Path(out_prefix)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Path(f"{out_prefix}.md").write_text(markdown, encoding="utf-8")
    payload = {"level1": level1, "level2": level2_result}
    with open(f"{out_prefix}.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    return markdown


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sweep-bf16", required=True, type=Path,
                        help="BF16 sweep dir (pos-<N> dirs, greedy-native.json)")
    parser.add_argument("--sweep-q6k", required=True, type=Path,
                        help="Q6_K sweep dir (greedy-native.json)")
    parser.add_argument("--draft-bf16", required=True, type=Path,
                        help="BF16 drafter GGUF (used for the harness)")
    parser.add_argument("--draft-q6k", required=True, type=Path,
                        help="Q6_K drafter GGUF (recorded for reference; the "
                             "Level 2 check reads the Q6_K greedy file only)")
    parser.add_argument("--target", required=True, type=Path,
                        help="target GGUF (Q4_K_XL)")
    parser.add_argument("--upstream", required=True, type=Path,
                        help="upstream dflash/model.py")
    parser.add_argument("--harness", required=True, type=Path,
                        help="dflash_reference.py")
    parser.add_argument("--harness-cwd", required=True, type=Path,
                        help="working directory for the harness (Gufo-mtp-perf)")
    parser.add_argument("--out", required=True, type=Path,
                        help="output prefix; writes <out>.md and <out>.json")
    parser.add_argument("--force", action="store_true",
                        help="rerun the harness even when report.json exists")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    for name in ("draft_bf16", "draft_q6k", "target", "upstream", "harness"):
        path = getattr(args, name)
        if not path.exists():
            sys.exit(f"error: --{name.replace('_', '-')} not found: {path}")
    if not args.harness_cwd.is_dir():
        sys.exit(f"error: --harness-cwd not found: {args.harness_cwd}")

    positions = discover_positions(args.sweep_bf16)
    position_results = []
    for number, trace_dir in positions:
        report = run_harness(args, trace_dir)
        name, value = worst_stage(report)
        position_results.append({
            "position": number,
            "passed": bool(report["passed"]),
            "worst_stage": name,
            "worst_relative_rmse": float(value),
            "selector_max_probability_error":
                float(report["selector"]["max_probability_error"]),
            "max_total_variation":
                float(report["full_forward"]["max_total_variation"]),
        })

    positions_passed = sum(1 for p in position_results if p["passed"])
    level1_result = {
        "passed": positions_passed == len(position_results),
        "positions_passed": positions_passed,
        "positions": position_results,
    }

    numbers = [number for number, _ in positions]
    level2_result, per_position_greedy = level2(
        load_greedy(args.sweep_bf16), load_greedy(args.sweep_q6k), numbers)

    markdown = write_outputs(str(args.out), level1_result, level2_result,
                             per_position_greedy)
    print(markdown, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
