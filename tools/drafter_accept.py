#!/usr/bin/env python3
"""Acceptance measurement for the DFlash2 drafter (M1c).

Subcommands:
  oracle   Run the shared prompt workload through llama.cpp's DFlash
           speculative decoding (llama-server) and record acceptance and
           mean length per prompt and per category.
  compare  Join the oracle report with the extension's free-running and
           streamed reports by id and write a markdown + JSON comparison.

Input is the token-id dump written by `flint_drafter_parity --dump-prompts`
(so both engines see identical prompt ids). llama-server is started here
unless --url is given, and is always killed on exit.

Mean length source: llama-server prints one line per finished request,
`draft acceptance = A (accepted / generated), mean len = L`. llama.cpp
b11804 defines L = 1 + accepted / verif_steps (server-context.cpp:772), where
verif_steps is the number of verification cycles. Cycles are therefore
derived from the accepted count and L (see derive_cycles), and checked
against the printed 2 decimals. Blocks are not assumed to be 7 long.

Example:
  python3 tools/drafter_accept.py oracle \
    --dump /tmp/opencode/m1c/prompts-dump.jsonl --results-dir /tmp/opencode/m1c
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

SERVER_BIN = "/home/bruno/AI/llama/llama-b11804-rocm/llama-server"
TARGET_GGUF = "/home/bruno/AI/GGUF/flint-UD-Q4_K_XL.gguf"
DRAFT_GGUF = "/home/bruno/AI/GGUF/mtp/Flint-DFlash2-BF16.gguf"
DEFAULT_DUMP = "/tmp/opencode/m1c/prompts-dump.jsonl"
DEFAULT_RESULTS = "/tmp/opencode/m1c"
DEFAULT_PORT = 8098
DEFAULT_P_MINS = ["0", "0.5"]

LOG_RE = re.compile(
    r"draft acceptance = ([0-9.]+) \(\s*(\d+) accepted / \s*(\d+) generated\), "
    r"mean len = \s*([0-9.]+)"
)


def read_dump(path):
    """Parses the prompt dump, one JSON object per non-empty line."""
    records = []
    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise SystemExit(f"{path}:{number}: {error}") from error
    if not records:
        raise SystemExit(f"{path}: no prompts")
    return records


def http_json(url, payload=None, timeout=600):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data else {},
    )
    with urllib.request.urlopen(request, timeout=timeout) as reply:
        return json.loads(reply.read().decode("utf-8"))


def server_healthy(base_url):
    try:
        http_json(base_url + "/health", timeout=2)
        return True
    except (urllib.error.URLError, OSError, ValueError):
        return False


def start_server(p_min, port, log_path, args):
    """Starts llama-server with DFlash speculation and waits for /health."""
    base_url = f"http://127.0.0.1:{port}"
    if server_healthy(base_url):
        raise SystemExit(
            f"port {port} already answers /health; stop the running server first"
        )
    command = [
        args.server_bin,
        "-m", args.target,
        "-md", args.draft,
        "--spec-type", "draft-dflash",
        "--spec-draft-n-max", "7",
        "--spec-draft-p-min", p_min,
        "-fa", "on", "-np", "1", "--temp", "0", "-ngl", "99", "-c", "8192",
        "--host", "127.0.0.1", "--port", str(port), "--no-webui",
    ]
    log = open(log_path, "w", encoding="utf-8")
    process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=log)
    deadline = time.monotonic() + args.startup_timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            log.close()
            raise SystemExit(f"llama-server exited early; see {log_path}")
        if server_healthy(base_url):
            return process, log, base_url
        time.sleep(1)
    stop_server(process, log)
    raise SystemExit(f"llama-server not healthy after {args.startup_timeout}s")


def stop_server(process, log):
    """Always-run teardown: terminate, then kill if it does not exit."""
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    log.close()


def complete(base_url, record, timeout):
    payload = {
        "prompt": record["prompt_ids"],
        "n_predict": record["max_tokens"],
        "temperature": 0,
        "top_k": 1,
        "cache_prompt": False,
        "ignore_eos": not record["stop_on_eos"],
        "return_tokens": True,
    }
    return http_json(base_url + "/completion", payload, timeout=timeout)


def check_tokens(response, record):
    """Fails loudly if the server does not return generated ids."""
    tokens = response.get("tokens")
    if not isinstance(tokens, list) or not all(isinstance(t, int) for t in tokens):
        raise SystemExit(
            f"{record['id']}: response has no 'tokens' id list "
            f"(keys: {sorted(response)}). return_tokens unsupported; stop."
        )
    return tokens


def parse_log_lines(log_path):
    """Returns the (accepted, generated, mean_len) of each finished request."""
    rows = []
    with open(log_path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = LOG_RE.search(line)
            if match:
                rows.append(
                    {
                        "log_acceptance": float(match.group(1)),
                        "log_accepted": int(match.group(2)),
                        "log_generated": int(match.group(3)),
                        "log_mean_len": float(match.group(4)),
                    }
                )
    return rows


def derive_cycles(accepted, predicted_n, mean_len):
    """Integer verification cycles behind a llama-server 'mean len' value.

    llama.cpp b11804 (server-context.cpp:772) defines
        mean_len = 1 + accepted / verif_steps
    so cycles = accepted / (mean_len - 1). The emitted count is also
    predicted_n = 1 + accepted + cycles (one bonus token per cycle plus the
    first token), which gives a primary derivation. Both candidates must
    reproduce the printed mean len to its 2 decimals, otherwise we stop.
    """
    def reproduces(cycles):
        return cycles >= 1 and round(1 + accepted / cycles, 2) == round(mean_len, 2)

    primary = predicted_n - accepted - 1
    if reproduces(primary):
        return primary
    if mean_len > 1:
        fallback = round(accepted / (mean_len - 1))
        if reproduces(fallback):
            return fallback
    raise SystemExit(
        f"cannot derive cycles: accepted={accepted} predicted_n={predicted_n} "
        f"mean_len={mean_len} (primary cycles={primary})"
    )


def run_pass(base_url, records, log_path, timeout):
    """Runs every prompt in order. Returns one result row per prompt."""
    results = []
    for index, record in enumerate(records):
        response = complete(base_url, record, timeout)
        if index == 0:
            # The first response is the schema check: keep it verbatim so the
            # field names the script relies on can be verified by hand.
            with open(log_path + ".first-response.json", "w",
                      encoding="utf-8") as handle:
                json.dump(response, handle, indent=2)
            print(f"first response keys: {sorted(response)}", flush=True)
            print(f"first response timings: {response.get('timings')}", flush=True)
        tokens = check_tokens(response, record)
        timings = response.get("timings", {})
        results.append(
            {
                "id": record["id"],
                "category": record["category"],
                "kind": record["kind"],
                "prompt_tokens": len(record["prompt_ids"]),
                "max_tokens": record["max_tokens"],
                "stop_on_eos": record["stop_on_eos"],
                "tokens": tokens,
                "drafted": int(timings.get("draft_n", 0)),
                "accepted": int(timings.get("draft_n_accepted", 0)),
                "emitted": int(timings.get("predicted_n", len(tokens))),
                "stop_type": response.get("stop_type"),
                "timings": timings,
            }
        )
        print(
            f"{record['id']}: drafted={results[-1]['drafted']} "
            f"accepted={results[-1]['accepted']} emitted={results[-1]['emitted']}",
            flush=True,
        )
    return results


def attach_cycles(results, log_path):
    """Matches log lines to requests (one slot, sequential) and derives cycles."""
    log_rows = parse_log_lines(log_path)
    if len(log_rows) != len(results):
        raise SystemExit(
            f"{log_path}: {len(log_rows)} 'draft acceptance' lines for "
            f"{len(results)} requests; cannot attribute cycles"
        )
    for row, log_row in zip(results, log_rows):
        row.update(log_row)
        row["cycles"] = derive_cycles(
            row["accepted"], row["emitted"], log_row["log_mean_len"]
        )
        if log_row["log_accepted"] != row["accepted"] or (
            log_row["log_generated"] != row["drafted"]
        ):
            raise SystemExit(
                f"{row['id']}: log acceptance {log_row['log_accepted']}/"
                f"{log_row['log_generated']} disagrees with timings "
                f"{row['accepted']}/{row['drafted']}"
            )


def pooled(rows):
    """Pooled ratios: sums first, then divide (not an average of averages)."""
    drafted = sum(r["drafted"] for r in rows)
    accepted = sum(r["accepted"] for r in rows)
    emitted = sum(r["emitted"] for r in rows)
    cycles = sum(r["cycles"] for r in rows)
    return {
        "prompts": len(rows),
        "drafted": drafted,
        "accepted": accepted,
        "emitted": emitted,
        "cycles": cycles,
        "acceptance": accepted / drafted if drafted else 0.0,
        # llama.cpp definition: 1 + accepted / verification cycles.
        "mean_len": 1 + accepted / cycles if cycles else 0.0,
    }


def summarize(results):
    categories = sorted({r["category"] for r in results})
    summary = {"all": pooled(results)}
    for category in categories:
        summary[category] = pooled([r for r in results if r["category"] == category])
    return summary


def write_outputs(results, summary, path):
    with open(path, "w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row) + "\n")
        handle.write(json.dumps({"summary": summary}) + "\n")


def run_oracle(args):
    records = read_dump(args.dump)
    os.makedirs(args.results_dir, exist_ok=True)
    for p_min in args.p_min:
        tag = p_min.replace(".", "")
        out_path = os.path.join(args.results_dir, f"oracle-p{tag}.jsonl")
        log_path = os.path.join(args.results_dir, f"oracle-p{tag}.server.log")
        print(f"== p_min {p_min} ==", flush=True)
        if args.url:
            if not args.log:
                raise SystemExit("--url needs --log to attribute cycles")
            base_url, process, log = args.url.rstrip("/"), None, None
            log_path = args.log
        else:
            process, log, base_url = start_server(p_min, args.port, log_path, args)
        try:
            results = run_pass(base_url, records, log_path, args.timeout)
        finally:
            if process is not None:
                stop_server(process, log)
        try:
            attach_cycles(results, log_path)
        except SystemExit:
            # Keep the generated ids and timings even when cycles cannot be
            # derived; the server run is the expensive part.
            with open(out_path + ".raw.json", "w", encoding="utf-8") as handle:
                json.dump(results, handle)
            raise
        summary = summarize(results)
        write_outputs(results, summary, out_path)
        print(f"wrote {out_path}")
        for name, row in summary.items():
            print(
                f"  {name:<12} prompts={row['prompts']:>2} "
                f"acceptance={row['acceptance'] * 100:6.2f}% "
                f"mean_len={row['mean_len']:.3f} "
                f"(accepted {row['accepted']}/{row['drafted']}, "
                f"emitted {row['emitted']}/{row['cycles']} cycles)"
            )


CATEGORY_ORDER = ["code", "json", "prose", "continuation"]
COUNT_FIELDS = ("drafted", "accepted", "emitted", "cycles")


def load_report(path, label, extra_fields=()):
    """One JSON object per prompt line, keyed by id. Summary lines are skipped."""
    rows = {}
    required = ("id", "category") + COUNT_FIELDS + tuple(extra_fields)
    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as error:
                raise SystemExit(f"{label} {path}:{number}: {error}") from error
            if "summary" in obj and "id" not in obj:
                continue
            missing = [f for f in required if f not in obj]
            if missing:
                raise SystemExit(f"{label} {path}:{number}: missing fields {missing}")
            if obj["id"] in rows:
                raise SystemExit(f"{label} {path}: duplicate id {obj['id']!r}")
            rows[obj["id"]] = obj
    if not rows:
        raise SystemExit(f"{label} {path}: no prompt rows")
    return rows


def join_reports(oracle, free, stream):
    """Joins the three reports by id. Fails if any id set differs."""
    ids_o, ids_f, ids_s = set(oracle), set(free), set(stream)
    if not (ids_o == ids_f == ids_s):
        union = ids_o | ids_f | ids_s
        raise SystemExit(
            "id sets differ; missing ids per input: "
            f"oracle={sorted(union - ids_o)} "
            f"ours-free={sorted(union - ids_f)} "
            f"ours-stream={sorted(union - ids_s)}"
        )
    joined = []
    for pid in oracle:
        o, f, s = oracle[pid], free[pid], stream[pid]
        if not (o["category"] == f["category"] == s["category"]):
            raise SystemExit(
                f"{pid}: category mismatch {o['category']!r}/"
                f"{f['category']!r}/{s['category']!r}"
            )
        joined.append((o, f, s))
    return joined


def compare_table(joined):
    """Per-category and pooled ('all') rows. Divergence is a mean over prompts."""
    categories = sorted({t[0]["category"] for t in joined},
                        key=lambda c: (CATEGORY_ORDER.index(c)
                                       if c in CATEGORY_ORDER else len(CATEGORY_ORDER), c))
    table = []
    for name in categories + ["all"]:
        sel = joined if name == "all" else [t for t in joined if t[0]["category"] == name]
        oracle = pooled([t[0] for t in sel])
        free = pooled([t[1] for t in sel])
        stream = pooled([t[2] for t in sel])
        divergence = sum(float(t[2]["stream_divergence"]) for t in sel) / len(sel)
        table.append(
            {
                "category": name,
                "prompts": len(sel),
                "oracle_p0": oracle,
                "ours_free": free,
                "ours_stream": stream,
                "delta_mean_len": stream["mean_len"] - oracle["mean_len"],
                "stream_divergence_mean": divergence,
            }
        )
    return table


def render_markdown(table):
    header = (
        "| category | oracle p0 acc | oracle p0 mean len | ours free acc | "
        "ours free mean len | ours stream acc | ours stream mean len | "
        "delta mean len (stream - oracle) | stream divergence |"
    )
    lines = [header, "|" + "---|" * 9]
    for row in table:
        o, f, s = row["oracle_p0"], row["ours_free"], row["ours_stream"]
        lines.append(
            f"| {row['category']} "
            f"| {o['acceptance'] * 100:.2f}% | {o['mean_len']:.3f} "
            f"| {f['acceptance'] * 100:.2f}% | {f['mean_len']:.3f} "
            f"| {s['acceptance'] * 100:.2f}% | {s['mean_len']:.3f} "
            f"| {row['delta_mean_len']:+.3f} "
            f"| {row['stream_divergence_mean']:.4f} |"
        )
    lines.append("")
    lines.append(
        "Acceptance = sum accepted / sum drafted; mean len = 1 + sum accepted / "
        "sum cycles (pooled). Stream divergence = mean over prompts of the "
        "per-prompt stream_divergence value."
    )
    return "\n".join(lines) + "\n"


def run_compare(args):
    oracle = load_report(args.oracle, "oracle")
    free = load_report(args.ours_free, "ours-free")
    stream = load_report(args.ours_stream, "ours-stream", ("stream_divergence",))
    table = compare_table(join_reports(oracle, free, stream))
    markdown = render_markdown(table)
    print(markdown, end="")
    with open(args.out + ".md", "w", encoding="utf-8") as handle:
        handle.write(markdown)
    payload = {
        "inputs": {
            "oracle": args.oracle,
            "ours_free": args.ours_free,
            "ours_stream": args.ours_stream,
        },
        "stream_divergence_basis": "mean over prompts",
        "rows": table,
    }
    with open(args.out + ".json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    print(f"wrote {args.out}.md and {args.out}.json")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    oracle = sub.add_parser("oracle", help="llama.cpp DFlash baseline")
    oracle.add_argument("--dump", default=DEFAULT_DUMP,
                        help="prompt ids from --dump-prompts")
    oracle.add_argument("--results-dir", default=os.environ.get(
        "RESULTS_DIR", DEFAULT_RESULTS))
    oracle.add_argument("--p-min", nargs="+", default=DEFAULT_P_MINS,
                        help="one server run per value (default: 0 0.5)")
    oracle.add_argument("--url", help="use a running llama-server instead")
    oracle.add_argument("--log", help="server log for --url runs")
    oracle.add_argument("--port", type=int, default=DEFAULT_PORT)
    oracle.add_argument("--server-bin", default=SERVER_BIN)
    oracle.add_argument("--target", default=TARGET_GGUF)
    oracle.add_argument("--draft", default=DRAFT_GGUF)
    oracle.add_argument("--startup-timeout", type=int, default=600)
    oracle.add_argument("--timeout", type=int, default=900,
                        help="per /completion request, seconds")
    oracle.set_defaults(func=run_oracle)
    compare = sub.add_parser("compare", help="join oracle and ours reports by id")
    compare.add_argument("--oracle", required=True,
                         help="oracle JSONL report (one prompt per line)")
    compare.add_argument("--ours-free", required=True,
                         help="free-running flint_drafter_parity report")
    compare.add_argument("--ours-stream", required=True,
                         help="streamed flint_drafter_parity report")
    compare.add_argument("--out", required=True,
                         help="writes PREFIX.md and PREFIX.json")
    compare.set_defaults(func=run_compare)
    return parser


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
