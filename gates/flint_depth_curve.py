#!/usr/bin/env python3
"""Flint 40B AR depth curve — decode/prefill vs prompt depth (F-J4 protocol).

Same measurement protocol as tools/flint/fj4_bench.py (prefill = prompt_tokens /
TTFT, decode = (completion-1)/(T_last - T_first), thermal context per run), but
sweeps prompt depth instead of the three fixed F-J4 cells. Prompts are the
repo's own F-J4 prose fixture (tools/flint/fixtures/fj4/long.txt) repeated to
depth, so the prompt family matches the accepted F-J4 numbers.

Writes one JSONL record per run and prints the mean table.
"""

import argparse
import glob
import json
import pathlib
import statistics
import time
import urllib.request

FIXTURES = pathlib.Path(__file__).resolve().parent / "fixtures" / "fj4"
LONG = (FIXTURES / "long.txt").read_text()
MSG = (FIXTURES / "msg.txt").read_text()
# long.txt is 110,098 chars for 23,586 prompt tokens (BENCHMARKS.md F-J4 long).
CHARS_PER_TOKEN = len(LONG) / 23586


def read_sys(path_glob):
    hits = glob.glob(path_glob)
    if not hits:
        return "?"
    try:
        return (pathlib.Path(hits[0]).read_text()).strip()
    except OSError:
        return "?"


def read_div(path_glob, div):
    raw = read_sys(path_glob)
    try:
        return int(raw) / div
    except ValueError:
        return -1.0


def k10temp_tctl():
    for name_path in glob.glob("/sys/class/hwmon/hwmon*/name"):
        try:
            if pathlib.Path(name_path).read_text().strip() != "k10temp":
                continue
        except OSError:
            continue
        tctl = read_div(str(pathlib.Path(name_path).parent / "temp1_input"), 1000.0)
        if tctl >= 0:
            return tctl
    return -1.0


def thermal():
    edge = read_div("/sys/class/drm/card1/device/hwmon/hwmon*/temp1_input", 1000.0)
    power = read_div("/sys/class/drm/card1/device/hwmon/hwmon*/power1_average", 1e6)
    sclk = read_sys("/sys/class/drm/card1/device/pp_dpm_sclk")
    cur = "?"
    for line in sclk.splitlines():
        if "*" in line:
            cur = line.split(":")[1].strip().split("Mhz")[0].strip()
    return {
        "edge_c": edge,
        "tctl_c": k10temp_tctl(),
        "power_w": power,
        "sclk_mhz": cur,
        "dpm": read_sys("/sys/class/drm/card1/device/power_dpm_force_performance_level"),
    }


def text_for_tokens(n_tokens):
    """long.txt repeated to ~n_tokens, trimmed to a char count."""
    target = int(n_tokens * CHARS_PER_TOKEN)
    reps = target // len(LONG) + 1
    return (LONG * reps)[:target]


def stream_once(url, model, text, max_tokens, no_cache_prompt=False):
    payload = {
        "model": model, "messages": [{"role": "user", "content": text}],
        "max_tokens": max_tokens, "temperature": 0.0, "top_p": 1.0,
        "stream": True, "stream_options": {"include_usage": True},
    }
    if no_cache_prompt:
        payload["cache_prompt"] = False
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first = None
    last = None
    usage = None
    with urllib.request.urlopen(request, timeout=2400) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            chunk = json.loads(body)
            if chunk.get("usage") is not None:
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                delta = choice.get("delta", {})
                if (delta.get("content") or delta.get("reasoning_content")
                        or delta.get("tool_calls")):
                    now = time.perf_counter()
                    if first is None:
                        first = now
                    last = now
    return t0, first, last, usage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8016/v1/chat/completions")
    ap.add_argument("--model", default="flint-q6")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--depths", default="512,2048,8192,32768,65280,130816")
    ap.add_argument("--out", default="/tmp/opencode/flint-depth.jsonl")
    ap.add_argument("--summary", default=None,
                    help="summary JSON path (default: alongside --out)")
    ap.add_argument("--label", default="flint-40b UD-Q6_K_XL AR",
                    help="label recorded in the summary cell")
    ap.add_argument("--fixtures-prompt", default=None,
                    help="override the long.txt path")
    ap.add_argument("--no-cache-prompt", action="store_true",
                    help="send cache_prompt=false (force cold prefill)")
    args = ap.parse_args()

    global LONG, MSG, CHARS_PER_TOKEN
    if args.fixtures_prompt:
        LONG = pathlib.Path(args.fixtures_prompt).read_text()
        CHARS_PER_TOKEN = len(LONG) / 23586

    depths = [int(d) for d in args.depths.split(",") if d.strip()]
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("")

    print(f"warm-up: thermal {thermal()}", flush=True)
    t0, first, last, usage = stream_once(args.url, args.model, MSG, 8)
    print(f"warm-up ok: ttft {first - t0:.2f}s usage {usage}", flush=True)

    rows = []
    for depth in depths:
        text = text_for_tokens(depth)
        samples = []
        for run in range(args.runs):
            before = thermal()
            t0, first, last, usage = stream_once(
                args.url, args.model, text, args.max_tokens, args.no_cache_prompt)
            after = thermal()
            ttft = first - t0
            p_tok = usage["prompt_tokens"]
            c_tok = usage["completion_tokens"]
            prefill = p_tok / ttft if ttft > 0 else 0.0
            decode = (c_tok - 1) / (last - first) if last and last > first and c_tok > 1 else 0.0
            gufo = (usage or {}).get("gufo", {}) or {}
            rec = {
                "target_depth": depth, "run": run, "prompt_tokens": p_tok,
                "completion_tokens": c_tok, "ttft_s": round(ttft, 3),
                "prefill_tps": round(prefill, 1), "decode_tps": round(decode, 2),
                "cache_hit": gufo.get("cache_hit"),
                "cache_miss_reason": gufo.get("cache_miss_reason"),
                "prefill_tokens": gufo.get("prefill_tokens"),
                "prefill_chunks": gufo.get("prefill_chunks"),
                "before": before, "after": after, "ts": time.time(),
            }
            samples.append(rec)
            with out.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(f"  depth {p_tok:>6} run {run}: prefill {prefill:8.1f} t/s  "
                  f"decode {decode:6.2f} t/s  ttft {ttft:7.2f}s  "
                  f"edge {after['edge_c']:.0f}C tctl {after['tctl_c']:.0f}C "
                  f"{after['power_w']:.0f}W sclk {after['sclk_mhz']}MHz", flush=True)
        rows.append({
            "depth": round(statistics.mean([s["prompt_tokens"] for s in samples])),
            "prefill": round(statistics.mean([s["prefill_tps"] for s in samples]), 1),
            "decode": round(statistics.mean([s["decode_tps"] for s in samples]), 2),
            "decode_sd": round(statistics.pstdev([s["decode_tps"] for s in samples]), 2),
            "n": len(samples),
        })

    print("\n=== depth curve (means) ===")
    print(f"{'prompt tok':>10} {'prefill t/s':>12} {'decode t/s':>11} {'sd':>6}")
    for r in rows:
        print(f"{r['depth']:>10} {r['prefill']:>12.1f} {r['decode']:>11.2f} {r['decode_sd']:>6.2f}")
    summary_path = (pathlib.Path(args.summary) if args.summary
                    else out.with_name(out.stem + "-summary.json"))
    summary_path.write_text(
        json.dumps({"cell": args.label, "runs": args.runs,
                    "max_tokens": args.max_tokens, "port": args.url,
                    "rows": rows}, indent=2))
    print("\nwrote", out, "and", summary_path)


if __name__ == "__main__":
    main()
