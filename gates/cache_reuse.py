#!/usr/bin/env python3
"""Continuation-cache prefix reuse gate (kAppendRows layout, Stage 2).

Starts `gufo serve llm` with a RAM cache, then checks:
  r1  long user prompt; capture its answer and prompt tokens
  r2  identical request; cache=memory, cached_tokens > 0, answer identical
  r3  multi-turn follow-up: [user LONG, assistant <r1 reasoning+content>,
      user "Now continue."]; status 200, cache reuse, cached_tokens >= 0.5 *
      r1 prompt tokens, non-empty content

Why the answer is reasoning_content + content: Flint's pinned chat template
has no thinking switch (no enable_thinking conditional; the assistant prompt
is always rendered as <|assistant|>), so the model always decides on its own
whether to reason. Every request therefore sends reasoning_budget_tokens=0,
which forces the close tag before any reasoning token. The answer is
reasoning_content + content so equality still compares the whole output if a
budget ever leaks reasoning. Content must be non-empty independently.
The r3 assistant turn replays both fields, so the rendered prompt matches
r1's cached prefix.
  r1  cache_snapshot_bytes consistent with valid rows: between one token's
      rows and prompt+completion tokens of per-layer K+V rows, plus one logits
      row and the fixed metadata. Row bytes come from the target GGUF
      (block_count, attention.head_count_kv, attention.key_length). If the GGUF
      cannot be parsed, fall back to the weaker bound
      cache_snapshot_bytes < 0.25 * 822684216 (old single-blob size at 8k).

Why r3 is not a same-turn suffix: the adapter captures continuation state only
at request boundaries (PrefillCheckpointPositions is not implemented), so a
prompt extended inside the same user turn can never hit a checkpoint at or
below its common prefix. Partial mid-prompt reuse needs PrefillCheckpointPositions
(deferred). A multi-turn follow-up reuses r1's full prompt-end checkpoint.

Stdlib only. Exit 0 on success, 1 on any failed assertion.
"""

import argparse
import json
import os
import signal
import struct
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

OLD_BLOB_BYTES = 822684216  # Single-blob snapshot at -c 8192 (0.3.1 baseline).
PARAGRAPH = (
    "The river bent slowly past the old mill, carrying leaves and small "
    "branches toward the sea. Along its banks, reeds swayed in the wind while "
    "herons waited motionless for fish to pass. A narrow footpath followed the "
    "water for several miles, past orchards, stone walls, and a village church "
    "whose bell rang at noon. Travelers often stopped there to rest and to "
    "watch the light change over the hills. "
)
QUESTION = "\n\nSummarise the passage above in one sentence."
FOLLOWUP = "Now continue."


# ---------------------------------------------------------------- GGUF ------

_SCALAR = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?",
           10: "Q", 11: "q", 12: "d"}


def read_gguf_geometry(path):
    """Return (num_layers, per_layer_kv_heads, head_dim, vocab) or None."""
    try:
        with open(path, "rb") as fh:
            def take(fmt):
                size = struct.calcsize(fmt)
                return struct.unpack("<" + fmt, fh.read(size))[0]

            def read_string():
                n = take("Q")
                return fh.read(n).decode("utf-8", "replace")

            def read_value(vtype):
                # Returns the scalar, or for arrays only the element count
                # (the tokenizer vocabulary is large and only its length is
                # needed).
                if vtype in _SCALAR:
                    return take(_SCALAR[vtype])
                if vtype == 8:
                    return read_string()
                if vtype == 9:
                    elem = take("I")
                    count = take("Q")
                    if elem == 8:
                        for _ in range(count):
                            read_string()
                    elif elem in _SCALAR:
                        fh.seek(struct.calcsize(_SCALAR[elem]) * count, 1)
                    else:
                        raise ValueError("unsupported gguf array element")
                    return count
                raise ValueError("unsupported gguf value type")

            if fh.read(4) != b"GGUF":
                return None
            version = take("I")
            if version < 2:
                return None
            take("Q")  # tensor count
            kv_count = take("Q")
            wanted = {}
            for _ in range(kv_count):
                key = read_string()
                vtype = take("I")
                value = read_value(vtype)
                wanted[key] = value
            arch = wanted.get("general.architecture")
            if not isinstance(arch, str):
                return None

            def get(name):
                return wanted.get(f"{arch}.{name}")

            layers = get("block_count")
            kv = get("attention.head_count_kv")
            heads = get("attention.head_count")
            key_len = get("attention.key_length")
            emb = get("embedding_length")
            vocab = get("vocab_size") or wanted.get("tokenizer.ggml.tokens")
            if key_len is None and emb and heads:
                key_len = emb // heads
            if not (layers and kv and key_len):
                return None
            return int(layers), kv, int(key_len), int(vocab or 0)
    except (OSError, ValueError, struct.error):
        return None


# ------------------------------------------------------------- server -------

class Server:
    def __init__(self, args, log_path):
        cmd = [args.gufo, "serve", "llm", "-m", args.target, "-c",
               str(args.context), "--cache-ram-bytes", str(args.cache_bytes),
               # No --think flag: the pinned template has no thinking switch,
               # so it has no effect here. Reasoning is bounded per request
               # with reasoning_budget_tokens=0 instead.
               "-t", "0", "-s", "1", "--log-progress", "-p", str(args.port)]
        if args.dflash_model:
            cmd += ["--speculative", "dflash2", "--dflash-model",
                    args.dflash_model]
        self.cmd = cmd
        self.log_path = log_path
        self.lines = []
        self.lock = threading.Lock()
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True,
                                     bufsize=1)
        self.file = open(log_path, "w")
        self.reader = threading.Thread(target=self._tee, daemon=True)
        self.reader.start()

    def _tee(self):
        for line in self.proc.stdout:
            with self.lock:
                self.lines.append(line)
                self.file.write(line)
                self.file.flush()
            sys.stderr.write(line)
            sys.stderr.flush()

    def has(self, needle):
        with self.lock:
            return any(needle in line for line in self.lines)

    def completed_lines(self):
        with self.lock:
            return [l for l in self.lines
                    if "event=completed" in l and "cached_tokens=" in l]

    def tail(self, n=40):
        with self.lock:
            return "".join(self.lines[-n:])

    def stop(self):
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                print("warning: gufo ignored SIGTERM for 120 s; SIGKILL",
                      file=sys.stderr)
                self.proc.kill()
                self.proc.wait()
        self.reader.join(timeout=10)
        self.file.close()


def wait_ready(server, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if server.proc.poll() is not None:
            raise RuntimeError("gufo exited during startup\n" + server.tail())
        if server.has("event=load_completed") and server.has("event=listening"):
            return
        time.sleep(0.5)
    raise RuntimeError("startup timed out after %d s\n%s"
                       % (timeout, server.tail()))


def served_model(port):
    try:
        with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/v1/models", timeout=10) as resp:
            data = json.load(resp).get("data", [])
            if data and data[0].get("id"):
                return data[0]["id"]
    except (OSError, ValueError):
        pass
    return "flint"


def chat(port, model, messages, max_tokens):
    body = json.dumps({
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "seed": 1,
        # 0 forces </think> before any reasoning token (inference_backend.cpp).
        "reasoning_budget_tokens": 0,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=900) as resp:
            status, payload = resp.status, json.load(resp)
    except urllib.error.HTTPError as err:
        status, payload = err.code, err.read().decode("utf-8", "replace")
    return status, payload


def message_of(payload):
    try:
        return payload["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return {}


def answer_of(message):
    """reasoning_content + content: the whole visible output of one turn."""
    return (message.get("reasoning_content") or "") + (message.get("content") or "")


def usage_of(payload):
    return payload.get("usage", {}) if isinstance(payload, dict) else {}


def check(ok, message, failures):
    print(("PASS " if ok else "FAIL ") + message)
    if not ok:
        failures.append(message)


# --------------------------------------------------------------- main -------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gufo", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--dflash-model")
    ap.add_argument("--port", type=int, default=8094)
    ap.add_argument("--context", type=int, default=8192)
    ap.add_argument("--cache-bytes", type=int, default=8589934592)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--prompt-chars", type=int, default=6000)
    ap.add_argument("--keep-log")
    args = ap.parse_args()

    log_path = args.keep_log or os.path.join(
        "/tmp/opencode", "cache_reuse_%d.log" % int(time.time()))
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    print(f"serve log: {log_path}")
    geometry = read_gguf_geometry(args.target)
    if geometry is None:
        print("note: GGUF geometry unreadable; using the weak snapshot bound")
    else:
        print("gguf geometry: layers=%d kv_heads=%s head_dim=%d vocab=%d"
              % (geometry[0], geometry[1], geometry[2], geometry[3]))

    server = Server(args, log_path)
    failures = []
    results = {}
    try:
        wait_ready(server, 300)
        model = served_model(args.port)
        print(f"served model: {model}")

        prompt = (PARAGRAPH * (args.prompt_chars // len(PARAGRAPH) + 1))
        prompt = prompt[:args.prompt_chars] + QUESTION
        user = {"role": "user", "content": prompt}

        def run(name, messages):
            status, payload = chat(args.port, model, messages, args.max_tokens)
            usage = usage_of(payload)
            cached = (usage.get("prompt_tokens_details") or {}).get(
                "cached_tokens", 0)
            gufo = usage.get("gufo") or {}
            message = message_of(payload)
            results[name] = {
                "status": status,
                "message": message,
                "content": message.get("content") or "",
                "answer": answer_of(message),
                "cached": cached,
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "snapshot": gufo.get("cache_snapshot_bytes"),
                "restore_ms": gufo.get("cache_restore_ms"),
            }
            print(f"{name}: status={status} prompt_tokens="
                  f"{results[name]['prompt_tokens']} cached_tokens={cached} "
                  f"cache_snapshot_bytes={results[name]['snapshot']} "
                  f"cache_restore_ms={results[name]['restore_ms']}")
            print(f"{name} content={results[name]['content'][:120]!r} "
                  f"reasoning_len={len(message.get('reasoning_content') or '')}")
            if status != 200 or not results[name]["content"]:
                print(f"{name} raw payload: {str(payload)[:1500]}")

        run("r1", [user])
        run("r2", [user])
        r1 = results["r1"]
        # Replay both fields so the rendered prefix equals r1's cached prompt
        # plus its generated tokens.
        assistant = {"role": "assistant", "content": r1["content"]}
        if r1["message"].get("reasoning_content"):
            assistant["reasoning_content"] = r1["message"]["reasoning_content"]
        run("r3", [user, assistant, {"role": "user", "content": FOLLOWUP}])
        r2, r3 = results["r2"], results["r3"]
        for name, res in results.items():
            check(res["status"] == 200, f"{name} status 200", failures)
        # Completion lines are ordered r1, r2, r3; r2 is the second-to-last
        # of the three (the only completions this script issues).
        completed = server.completed_lines()
        r2_line = completed[-2] if len(completed) >= 2 else ""
        print("r2 completed: " + r2_line.strip())
        check("cache=memory" in r2_line,
              "r2 request-completed line reports cache=memory", failures)
        check(r2["cached"] > 0, f"r2 cached_tokens > 0 (got {r2['cached']})",
              failures)
        check(bool(r1["content"]), "r1 content non-empty", failures)
        check(r1["answer"] != "" and r1["answer"] == r2["answer"],
              "r2 answer (reasoning+content) identical to r1", failures)
        floor = 0.5 * r1["prompt_tokens"]
        print(f"r3 reuse: cached_tokens={r3['cached']} "
              f"r1_prompt_tokens={r1['prompt_tokens']} floor={floor:.0f}")
        check(r3["cached"] > 0 and r3["cached"] >= floor,
              f"r3 cached_tokens {r3['cached']} >= 0.5 * r1 prompt tokens",
              failures)
        check(bool(r3["content"]), "r3 content non-empty", failures)
        r3_line = completed[-1] if completed else ""
        print("r3 completed: " + r3_line.strip())
        check("cache=memory" in r3_line,
              "r3 request-completed line reports cache=memory", failures)

        snap = r1["snapshot"]
        if not snap:
            check(False, "r1 cache_snapshot_bytes reported", failures)
        elif geometry:
            layers, kv, head_dim, vocab = geometry
            # Per token: every layer's K and V row, fp16 (num_kv*head_dim*2 each).
            row_total = layers * kv * head_dim * 2 * 2
            lo = row_total  # At least one token's rows.
            hi = (r1["prompt_tokens"] + r1["completion_tokens"]) * row_total \
                + vocab * 4 + 4096
            print(f"snapshot expected rows-only range: [{lo}, {hi}] B "
                  f"(row_total={row_total} B/token)")
            check(lo <= snap <= hi,
                  f"r1 cache_snapshot_bytes {snap} within valid-row bounds",
                  failures)
            check(snap < 0.25 * OLD_BLOB_BYTES,
                  f"r1 cache_snapshot_bytes {snap} far below old blob",
                  failures)
        else:
            bound = 0.25 * OLD_BLOB_BYTES
            check(snap < bound,
                  f"r1 cache_snapshot_bytes {snap} < {bound:.0f} (weak bound)",
                  failures)
    except RuntimeError as err:
        print(f"setup failure: {err}", file=sys.stderr)
        failures.append(str(err))
    finally:
        server.stop()
        print("summary:")
        for name in ("r1", "r2", "r3"):
            if name in results:
                res = results[name]
                print(f"  {name} status={res['status']} "
                      f"cached_tokens={res['cached']} "
                      f"cache_snapshot_bytes={res['snapshot']} "
                      f"cache_restore_ms={res['restore_ms']}")
        print(f"serve log: {log_path}")

    if failures:
        print(f"FAILED ({len(failures)} assertion(s))")
        sys.exit(1)
    print("PASSED")


if __name__ == "__main__":
    main()
