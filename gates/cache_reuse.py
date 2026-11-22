#!/usr/bin/env python3
"""Continuation-cache prefix reuse gate (kAppendRows layout, Stage 2).

Starts `gufo serve llm` with a RAM cache, then checks:
  r1  long user prompt; capture its answer and prompt tokens
  r2  identical request; cache=memory, cached_tokens > 0, answer identical
  r3  multi-turn follow-up: [user LONG, assistant <r1 reasoning+content>,
      user "Now continue."]; status 200, cache reuse, cached_tokens >= 0.5 *
      r1 prompt tokens, non-empty content

Spec-on runs (--dflash-model) reuse the same reuse assertions. They do not
validate the drafter: serve reports draft_proposed=0 because draft counters are
not copied on the SelectNext/Advance path (see DEFERRED.md), so the spec-on leg
is "spec on (cache reuse only; draft counters unavailable)".

Why the answer is reasoning_content + content: Flint's pinned chat template
has no thinking switch (no enable_thinking conditional; the assistant prompt
is always rendered as <|assistant|>), so the model always decides on its own
whether to reason. Every request therefore sends reasoning_budget_tokens=0,
which forces the close tag before any reasoning token. The answer is
reasoning_content + content so equality still compares the whole output if a
budget ever leaks reasoning. Content must be non-empty independently.
The r3 assistant turn replays both fields, so the rendered prompt matches
r1's cached prefix.
  r1  cache_snapshot_bytes is exact (Stage 3 derivation, verified on the logs):
          cache_snapshot_bytes == prompt_tokens * row_total
                                  + cache_captures * (vocab*4 + 64)
      row_total = layers * 2 * kv_heads * head_dim * 2 (fp16 K+V per token,
      from the target GGUF). The reported figure is a SUM over captures: each
      in-pass checkpoint is a prefix of the prompt, so its row chunks are
      shared with the final checkpoint; each additional capture adds only its
      private payload (logits row + metadata). Missing cache_captures on the
      completed line fails the check; it is never skipped.
      That form assumes every layer keeps every row. Sliding-window layers
      (attention.sliding_window_pattern, True = window) keep a ring of the
      window, so the r1 check is window-aware (the ring layout, PR #5):
          global layers   rows = prompt_tokens
          window layers   rows in [rows_low, rows_high], where
            rows_low  = prompt - FirstChunkRow(prompt)   one full final window
            rows_high = captures * (window + 256)        each in-pass capture
                                                         publishes at most one
                                                         window plus a chunk,
                                                         and captures share no
                                                         window chunks
          FirstChunkRow(p) = (p - window) - (p - window) % 256 (clamped at 0)
          cache_snapshot_bytes in
            [global_layers*global_rows*2*row_layer + rows_low*swa_layers*2*row_layer
               + captures*(vocab*4+64),
             same with rows_high]
      row_layer = kv_heads * head_dim * 2 (one layer's K or V row, fp16).
      The 24k run (23,790 tokens, 5 captures, window 513, 14 global / 35 window
      layers) gives global 682,106,880 B and window 53,760,000 .. 275,609,600 B
      (rows_low = 23,790 - FirstChunkRow(23,790) = 23,790 - 23,040 = 750).
      The GGUF-unparseable fallback is the weak bound
      cache_snapshot_bytes < 0.25 * 822684216 (old single-blob size at 8k).

  r3  (--max-r3-chunk-bytes) a follow-up turn's capture is an exact function
      of the token counts (no equality is assumed beyond the derivation below):
          tail = r1_prompt_tokens % rows_per_chunk
          new  = r3_prompt_tokens - r1_prompt_tokens
          rows = tail + new
          r3 cache_snapshot_bytes == rows * row_total
                                     + r3 cache_captures * (vocab*4 + 64)
      Derivation: shareable chunks from r1 are [0, r1 - r1 % 256). The r1
      partial tail chunk is stored as a tail payload, not a shareable chunk,
      so the follow-up capture rewrites everything from that chunk boundary
      onward, i.e. the tail plus the new tokens. Each capture adds its
      private payload (logits + metadata). An earlier bound of
      256 * row_total + captures * private only holds when tail + new <= 256,
      which is prompt-length dependent; it is replaced, not loosened.
      Verified on logs: A/B/C/D 238 + 35 = 273 rows; E 213 + 35 = 248 rows.
      r3 must produce exactly one capture (all runs do); any other count fails
      as a changed request shape rather than falling back to a loose bound.

Why r3 is not a same-turn suffix: the adapter captures continuation state only
at request boundaries (PrefillCheckpointPositions is not implemented), so a
prompt extended inside the same user turn can never hit a checkpoint at or
below its common prefix. Partial mid-prompt reuse needs PrefillCheckpointPositions
(deferred). A multi-turn follow-up reuses r1's full prompt-end checkpoint.

Stage 3 identity checks (disk cache; no text grep, no byte patch):
  A1  the serve log has event=cache_identity with the expected state_abi and
      payload_layout (--expect-identity).
  A2  --foreign-context N: restart on the same disk dir with -c N and the same
      prompt; require 200, no cache=disk, and context_tokens=N in the identity.
  C1  --manifest-compare DIR_A DIR_B: manifests store only a 32-byte SHA-256
      digest of the identity text, so the two dirs' digests must differ while
      their component counts agree. Parsed from the manifest header
      (GUFOMNF2 | version | checkpoint | lineage | digest | ...).
  C2  the same run as A2 (a different identity does not reuse).
  A manifest from a v1 build would also fail structural validation: one
  private blob against 2 private + 98 append-row components. Identity
  matching is checked first and is what C1+C2 prove. Byte-flipping a manifest
  is not used: the trailing CRC rejects it before any identity compare, so
  it proves checksum validation, not identity matching.

Stdlib only. Exit 0 on success, 1 on any failed assertion.
"""

import argparse
import json
import os
import re
import shutil
import signal
import struct
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

OLD_BLOB_BYTES = 822684216
ROWS_PER_CHUNK = 256  # continuation_adapter.cpp kRowsPerChunk  # Single-blob snapshot at -c 8192 (0.3.1 baseline).
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


def first_chunk_row(position, window):
    """Lowest row a window layer retains for a capture at `position`.

    The window keeps [position - window, position), and rows are stored in
    256-row chunks, so the retained range starts at the chunk holding
    position - window. Clamped at 0 for prompts shorter than the window.
    """
    start = max(0, position - window)
    return start - start % ROWS_PER_CHUNK


def read_gguf_geometry(path):
    """Return the target geometry, or None when the GGUF is unreadable.

    Result: (num_layers, kv_heads, head_dim, vocab, sliding_window, swa_pattern)
    where swa_pattern is a list of bools (True = sliding-window layer, as in
    src/runtime/config.cpp) or None when the key is absent.
    """
    try:
        with open(path, "rb") as fh:
            def take(fmt):
                size = struct.calcsize(fmt)
                return struct.unpack("<" + fmt, fh.read(size))[0]

            def read_string():
                n = take("Q")
                return fh.read(n).decode("utf-8", "replace")

            def read_value(vtype):
                # Returns the scalar; for arrays only the element count, except
                # bool arrays, which are returned in full (the SWA pattern).
                # The tokenizer vocabulary is large and only its length is
                # needed.
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
                    elif elem == 7:
                        return [b != 0 for b in fh.read(count)]
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
            # kv may be a per-layer array in some GGUFs; the gate needs one
            # positive int, so anything else degrades to the weak bound.
            if isinstance(kv, list):
                kv = kv[0] if kv and all(v == kv[0] for v in kv) else None
            if not all(isinstance(v, int) and not isinstance(v, bool) and v > 0
                       for v in (layers, kv, key_len)):
                return None
            if not isinstance(vocab, int) or vocab <= 0:
                return None
            window = get("attention.sliding_window") or 0
            pattern = get("attention.sliding_window_pattern")
            if not (isinstance(pattern, list) and len(pattern) == layers):
                pattern = None
            return (int(layers), int(kv), int(key_len), int(vocab),
                    int(window), pattern)
    except (OSError, ValueError, TypeError, struct.error):
        return None


# ------------------------------------------------------------- server -------

class Server:
    def __init__(self, args, log_path, disk_dir=None):
        cmd = [args.gufo, "serve", "llm", "-m", args.target, "-c",
               str(args.context), "--cache-ram-bytes", str(args.cache_bytes),
               # No --think flag: the pinned template has no thinking switch,
               # so it has no effect here. Reasoning is bounded per request
               # with reasoning_budget_tokens=0 instead.
               "-t", "0", "-s", "1", "--log-progress", "-p", str(args.port)]
        if disk_dir:
            cmd += ["--cache-disk", disk_dir,
                    "--cache-disk-bytes", str(args.cache_disk_bytes)]
        if args.dflash_model:
            cmd += ["--speculative", "dflash2", "--dflash-model",
                    args.dflash_model]
        self.cmd = cmd
        self.log_path = log_path
        self.lines = []
        self.lock = threading.Lock()
        env = dict(os.environ)
        for item in args.env or []:
            key, _, value = item.partition("=")
            env[key] = value
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True,
                                     bufsize=1, env=env)
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

    def lines_with(self, needle):
        with self.lock:
            return [line for line in self.lines if needle in line]

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
        return self.proc.returncode


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


def parse_restore_ms(line):
    match = re.search(r"cache_restore_ms=([0-9.]+)", line)
    return float(match.group(1)) if match else None


def check(ok, message, failures):
    print(("PASS " if ok else "FAIL ") + message)
    if not ok:
        failures.append(message)


# ------------------------------------------------------------ helpers -------

def long_prompt(chars, head=""):
    body = (PARAGRAPH * (chars // len(PARAGRAPH) + 1))[:chars]
    return head + body + QUESTION


def ledger_of(line):
    match = re.search(r"cache_global_ledger_bytes=([0-9]+)", line)
    return int(match.group(1)) if match else None


def captures_of(line):
    match = re.search(r"cache_captures=([0-9]+)", line)
    return int(match.group(1)) if match else None


def stop_server(server, label):
    rc = server.stop()
    print(f"{label}: gufo exit code {rc}")
    return rc


def identity_line(server):
    lines = server.lines_with("event=cache_identity")
    return lines[-1].strip() if lines else ""


def field(line, name):
    match = re.search(r"\b%s=(\S+)" % re.escape(name), line)
    return match.group(1) if match else None


def run_request(server, port, model, name, messages, max_tokens, results):
    status, payload = chat(port, model, messages, max_tokens)
    usage = usage_of(payload)
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
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
    res = results[name]
    print(f"{name}: status={status} prompt_tokens={res['prompt_tokens']} "
          f"cached_tokens={cached} cache_snapshot_bytes={res['snapshot']} "
          f"cache_restore_ms={res['restore_ms']}")
    print(f"{name} content={res['content'][:120]!r} "
          f"reasoning_len={len(message.get('reasoning_content') or '')}")
    if status != 200 or not res["content"]:
        print(f"{name} raw payload: {str(payload)[:1500]}")
    return res


def completed_chat(server, index, timeout=10.0):
    """The index-th (0-based) chat-completion completed line on this server.

    Serve's request=<id> is its own counter (r1 = /v1/models, r2 = first chat),
    not the gate's name, so chat completions are selected by order. The
    completed line is written after the HTTP response is sent, so poll until
    the line appears rather than reading once.
    """
    deadline = time.monotonic() + timeout
    while True:
        chats = [l for l in server.completed_lines()
                 if "path=/v1/chat/completions" in l]
        if index < len(chats):
            return chats[index]
        if time.monotonic() >= deadline:
            return ""
        time.sleep(0.1)


# ------------------------------------------------------ manifest parsing -----

MANIFEST_MAGIC = b"GUFOMNF2"


def read_manifests(root):
    """Parse each manifest header: (path, digest, component_count).

    Layout (src/cache/disk.cpp EncodeManifest): magic(8) version(4)
    checkpoint(8) lineage(8) compatibility(32, raw) purpose(1) rank(4)
    input_len(4) input(n) token_count(4) tokens(4 each) component_count(4).
    The trailing 8-byte checksum is not validated here.
    """
    found = []
    for dirpath, _, files in os.walk(os.path.join(root, "v2", "manifests")):
        for name in files:
            path = os.path.join(dirpath, name)
            with open(path, "rb") as fh:
                data = fh.read()
            if data[:8] != MANIFEST_MAGIC:
                continue
            digest = data[28:60]
            pos = 60 + 1 + 4  # compatibility end, purpose(1), rank(4)
            input_len = struct.unpack_from("<I", data, pos)[0]
            pos += 4 + input_len
            token_count = struct.unpack_from("<I", data, pos)[0]
            pos += 4 + 4 * token_count
            components = struct.unpack_from("<I", data, pos)[0]
            found.append((path, digest, components))
    return found


# --------------------------------------------------------------- main -------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gufo", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--dflash-model")
    ap.add_argument("--port", type=int, default=8094)
    ap.add_argument("--context", type=int, default=8192)
    ap.add_argument("--cache-bytes", "--ram-bytes", dest="cache_bytes",
                    type=int, default=8589934592,
                    help="--cache-ram-bytes value (alias --ram-bytes)")
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--prompt-chars", type=int, default=6000)
    ap.add_argument("--keep-log")
    ap.add_argument("--env", action="append", default=[], metavar="KEY=VALUE",
                    help="environment variable for the gufo serve child "
                         "(repeatable), e.g. FLINT_SPARSE_ATTN=1")
    # Stage 3 additions. Each is opt-in; without them the script is unchanged.
    ap.add_argument("--cache-disk", metavar="DIR",
                    help="enable the disk continuation cache in DIR")
    ap.add_argument("--cache-disk-bytes", type=int, default=8589934592)
    ap.add_argument("--restart", action="store_true",
                    help="capture in one process, SIGTERM, reuse in a fresh "
                         "process on the same --cache-disk (requires it)")
    ap.add_argument("--stress", type=int, metavar="N",
                    help="N distinct long prefixes, then a repeat of #1; "
                         "disk cache off")
    ap.add_argument("--expect-identity", metavar="STATE_ABI:LAYOUT",
                    help="require the event=cache_identity line to carry "
                         "these state_abi and payload_layout values")
    ap.add_argument("--foreign-context", type=int, metavar="N",
                    help="A2/C2: restart on --cache-disk with -c N and the "
                         "same prompt; require 200, no cache=disk, and a "
                         "different context_tokens in the identity line")
    ap.add_argument("--manifest-compare", nargs=2, metavar=("DIR_A", "DIR_B"),
                    help="C1: require the two dirs' manifest digests to "
                         "differ with equal component counts")
    ap.add_argument("--min-r2-cached", type=int, default=0,
                    help="require r2 cached_tokens >= N (deep-prompt runs)")
    ap.add_argument("--min-r3-cached", type=int, default=0,
                    help="require r3 cached_tokens >= N (deep-prompt runs)")
    ap.add_argument("--max-r3-chunk-bytes", action="store_true",
                    help="require 0 < r3 capture <= rows_per_chunk * row_total "
                         "+ r3 cache_captures * (vocab*4 + 64) (follow-up turn "
                         "appends at most one new chunk per component)")
    args = ap.parse_args()
    if args.restart and not args.cache_disk:
        ap.error("--restart requires --cache-disk")
    if args.stress and args.cache_disk:
        ap.error("--stress runs with the disk cache off")
    if args.foreign_context and not args.cache_disk:
        ap.error("--foreign-context requires --cache-disk")

    log_path = args.keep_log or os.path.join(
        "/tmp/opencode", "cache_reuse_%d.log" % int(time.time()))
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    leg = ("spec on (cache reuse only; draft counters unavailable)"
           if args.dflash_model else "spec off")
    print(f"leg: {leg}")
    print(f"serve log: {log_path}")

    failures = []
    if args.manifest_compare:
        return manifest_compare(args, failures)
    if args.stress:
        return stress(args, log_path, failures)
    if args.foreign_context:
        return foreign_context(args, log_path, failures)
    return reuse_run(args, log_path, failures)


def reuse_run(args, log_path, failures):
    geometry = read_gguf_geometry(args.target)
    if geometry is None:
        print("note: GGUF geometry unreadable; using the weak snapshot bound")
    else:
        print("gguf geometry: layers=%d kv_heads=%s head_dim=%d vocab=%d"
              % (geometry[0], geometry[1], geometry[2], geometry[3]))

    results = {}
    server = Server(args, log_path, args.cache_disk)
    try:
        wait_ready(server, 300)
        model = served_model(args.port)
        print(f"served model: {model}")
        user = {"role": "user", "content": long_prompt(args.prompt_chars)}
        run_request(server, args.port, model, "r1", [user], args.max_tokens,
                    results)
        # Read r1's completed line now: after a restart it lives in the
        # phase-1 log, which the restarted server does not hold.
        results["r1"]["line"] = completed_chat(server, 0)
        r2_source = "memory"
        if args.restart:
            # Phase 1 ends here: the capture is on disk; stop cleanly.
            rc = stop_server(server, "phase1")
            check(rc == 0, f"phase 1 clean shutdown (exit {rc})", failures)
            server = Server(args, log_path + ".phase2.log", args.cache_disk)
            wait_ready(server, 300)
            model = served_model(args.port)
            r2_source = "disk"
        ident = identity_line(server)
        print("identity: " + ident)
        if args.expect_identity:
            abi, layout = args.expect_identity.split(":", 1)
            check(field(ident, "state_abi") == abi
                  and field(ident, "payload_layout") == layout,
                  f"A1 identity line reports state_abi={abi} "
                  f"payload_layout={layout}", failures)
        run_request(server, args.port, model, "r2", [user], args.max_tokens,
                    results)
        r1 = results["r1"]
        # Replay both fields so the rendered prefix equals r1's cached prompt
        # plus its generated tokens.
        assistant = {"role": "assistant", "content": r1["content"]}
        if r1["message"].get("reasoning_content"):
            assistant["reasoning_content"] = r1["message"]["reasoning_content"]
        run_request(server, args.port, model, "r3",
                    [user, assistant, {"role": "user", "content": FOLLOWUP}],
                    args.max_tokens, results)
        r2, r3 = results["r2"], results["r3"]
        for name, res in results.items():
            check(res["status"] == 200, f"{name} status 200", failures)
        r2_line = completed_chat(server, 0 if args.restart else 1)
        print("r2 completed: " + r2_line.strip())
        check(bool(r2_line) and f"cache={r2_source}" in r2_line,
              f"r2 request-completed line reports cache={r2_source}", failures)
        check(r2["cached"] > 0, f"r2 cached_tokens > 0 (got {r2['cached']})",
              failures)
        if args.min_r2_cached:
            check(r2["cached"] >= args.min_r2_cached,
                  f"r2 cached_tokens {r2['cached']} >= {args.min_r2_cached}",
                  failures)
        # cache=memory and cached_tokens>0 also hold when r2 is served from the
        # live session without a restore, so require a measured restore.
        r2_restore = parse_restore_ms(r2_line)
        print(f"r2 cache_restore_ms (log) = {r2_restore}")
        check(r2_restore is not None and r2_restore > 0,
              f"r2 restore happened (cache_restore_ms={r2_restore})", failures)
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
        r3_line = completed_chat(server, 1 if args.restart else 2)
        print("r3 completed: " + r3_line.strip())
        # After a disk restore r3 may be served from the restored session
        # (memory) or from a disk checkpoint; either is a real reuse.
        check(bool(r3_line) and ("cache=memory" in r3_line
                                 or "cache=disk" in r3_line),
              "r3 request-completed line reports a cache hit", failures)
        if args.min_r3_cached:
            check(r3["cached"] >= args.min_r3_cached,
                  f"r3 cached_tokens {r3['cached']} >= {args.min_r3_cached}",
                  failures)

        snap = r1["snapshot"]
        if not snap:
            check(False, "r1 cache_snapshot_bytes reported", failures)
        elif geometry:
            layers, kv, head_dim, vocab, window, pattern = geometry
            # Per token: one layer's K and V row, fp16 (num_kv*head_dim*2 each).
            layer_bytes = 2 * kv * head_dim * 2
            row_total = layers * layer_bytes
            captures = captures_of(r1.get("line", ""))
            if window and pattern is None:
                check(False, "GGUF has a sliding window but no readable "
                      "attention.sliding_window_pattern", failures)
            elif captures is None:
                check(False, "r1 completed line reports cache_captures "
                      "(needed for the capture-sum size check)", failures)
            else:
                swa_layers = sum(pattern) if window else 0
                global_layers = layers - swa_layers
                prompt = r1["prompt_tokens"]
                global_rows = prompt
                rows_low = prompt - first_chunk_row(prompt, window)
                rows_high = captures * (window + ROWS_PER_CHUNK)
                private = captures * (vocab * 4 + 64)
                fixed = global_rows * global_layers * layer_bytes + private
                low = fixed + rows_low * swa_layers * layer_bytes
                high = fixed + rows_high * swa_layers * layer_bytes
                print(f"r1 snapshot bounds: global {global_layers} layers x "
                      f"{global_rows} rows; window {swa_layers} layers "
                      f"(window {window}) x [{rows_low}, {rows_high}] rows; "
                      f"captures={captures} private={private}")
                print(f"r1 snapshot range [{low}, {high}] B, measured {snap} B")
                check(low <= snap <= high,
                      f"r1 cache_snapshot_bytes {snap} within window-aware "
                      f"bounds [{low}, {high}]", failures)
            # OLD_BLOB_BYTES was measured at -c 8192 with a 1223-token prompt;
            # it is not a bound for other contexts or prompt sizes.
            if args.context == 8192 and args.prompt_chars <= 6000:
                check(snap < 0.25 * OLD_BLOB_BYTES,
                      f"r1 cache_snapshot_bytes {snap} far below old blob",
                      failures)
            # r3's capture is chunk-shared (a tail of the 256-row chunk), so it
            # is not an exact function of its prompt; only bound it.
            r3_snap = r3["snapshot"]
            r1_tokens = r1["prompt_tokens"]
            print(f"r3 cache_snapshot_bytes = {r3_snap}")
            check(bool(r3_snap) and 0 < r3_snap < snap,
                  f"r3 cache_snapshot_bytes {r3_snap} in (0, r1 snapshot)",
                  failures)
            if args.max_r3_chunk_bytes:
                r3_captures = captures_of(r3_line)
                if r3_captures is None or not r3_snap:
                    check(False, "r3 completed line reports cache_captures and "
                          "cache_snapshot_bytes (needed for the chunk bound)",
                          failures)
                elif r3_captures != 1:
                    check(False, f"r3 cache_captures={r3_captures}, expected 1: "
                          "the request shape changed (follow-up turn is no "
                          "longer one capture); the exact r3 bound does not "
                          "apply", failures)
                else:
                    tail = r1_tokens % ROWS_PER_CHUNK
                    new = r3["prompt_tokens"] - r1_tokens
                    rows = tail + new
                    # The exact form assumes each window layer's r3 capture
                    # starts at or below the r1 chunk boundary, so the tail
                    # is rewritten and nothing older is. If the window's first
                    # chunk for the r3 prompt lies above that boundary, the
                    # request shape changed and the exact form does not apply.
                    boundary = ROWS_PER_CHUNK * (r1_tokens // ROWS_PER_CHUNK)
                    r3_first = (first_chunk_row(r3["prompt_tokens"], window)
                                if window else 0)
                    if r3_first > boundary:
                        check(False, f"r3 window first chunk {r3_first} > r1 "
                              f"chunk boundary {boundary}: the request shape "
                              "changed; the exact r3 bound does not apply",
                              failures)
                    else:
                        expected_r3 = rows * row_total + 1 * (vocab * 4 + 64)
                        print(f"r3 exact = (r1 % 256 = {tail} + new {new}) = "
                              f"{rows} rows * {row_total} + 1*(vocab*4+64) = "
                              f"{expected_r3}; r3 capture {r3_snap}")
                        check(r3_snap == expected_r3,
                              f"r3 cache_snapshot_bytes {r3_snap} == exact "
                              f"derived {expected_r3}", failures)
        else:
            bound = 0.25 * OLD_BLOB_BYTES
            check(snap < bound,
                  f"r1 cache_snapshot_bytes {snap} < {bound:.0f} (weak bound)",
                  failures)
    except RuntimeError as err:
        print(f"setup failure: {err}", file=sys.stderr)
        failures.append(str(err))
    finally:
        rc = stop_server(server, "final")
        check(rc == 0, f"clean shutdown (exit {rc})", failures)
        print(f"serve log: {log_path}")
    return finish(failures)


def finish(failures):
    if failures:
        print(f"FAILED ({len(failures)} assertion(s))")
        sys.exit(1)
    print("PASSED")


def foreign_context(args, log_path, failures):
    """A2/C2: same disk dir, different -c. The identity digest changes, so the
    cache must not match. Dir is first captured at args.context."""
    results = {}
    server = Server(args, log_path, args.cache_disk)
    try:
        wait_ready(server, 300)
        model = served_model(args.port)
        user = {"role": "user", "content": long_prompt(args.prompt_chars)}
        run_request(server, args.port, model, "seed", [user], args.max_tokens,
                    results)
    except RuntimeError as err:
        failures.append(str(err))
    finally:
        check(stop_server(server, "seed") == 0, "seed clean shutdown",
              failures)
    expected_ctx = args.foreign_context
    args.context = expected_ctx
    server = Server(args, log_path + ".foreign.log", args.cache_disk)
    try:
        wait_ready(server, 300)
        model = served_model(args.port)
        ident = identity_line(server)
        print("foreign identity: " + ident)
        check(field(ident, "context_tokens") == str(expected_ctx),
              f"A2 identity line reports context_tokens={expected_ctx}", failures)
        res = run_request(server, args.port, model, "foreign", [user],
                          args.max_tokens, results)
        line = completed_chat(server, 0)
        print("foreign completed: " + line.strip())
        check(res["status"] == 200, "A2 foreign request status 200", failures)
        check(bool(line), "A2 foreign request-completed line present", failures)
        check("cache=disk" not in line,
              "A2 foreign context does not reuse the disk cache", failures)
    except RuntimeError as err:
        failures.append(str(err))
    finally:
        rc = stop_server(server, "foreign")
        check(rc == 0, f"A2 clean shutdown (exit {rc})", failures)
        print(f"serve log: {log_path}.foreign.log")
    finish(failures)


def manifest_compare(args, failures):
    """C1: two dirs captured with different -c. The digests must differ while
    the component counts agree, so the difference is the identity alone."""
    a_dir, b_dir = args.manifest_compare
    a = read_manifests(a_dir)
    b = read_manifests(b_dir)
    print(f"{a_dir}: {len(a)} manifest(s); {b_dir}: {len(b)} manifest(s)")
    check(bool(a) and bool(b), "both dirs contain v2 manifests", failures)
    if a and b:
        da = {d for _, d, _ in a}
        db = {d for _, d, _ in b}
        ca = sorted({c for _, _, c in a})
        cb = sorted({c for _, _, c in b})
        print("digest A: " + ", ".join(x.hex() for x in sorted(da)))
        print("digest B: " + ", ".join(x.hex() for x in sorted(db)))
        print(f"component counts A={ca} B={cb}")
        check(da.isdisjoint(db),
              "C1 compatibility digest differs between the two identities",
              failures)
        check(ca == cb and len(ca) == 1,
              "C1 component counts identical across the two dirs", failures)
    finish(failures)


def stress(args, log_path, failures):
    """Distinct long prefixes under a small RAM budget, then a repeat of #1.

    Pass criteria (per the Stage 3 plan): every request 200 with a non-empty
    answer; at least 3 of the 7 report cache_captures >= 1 (pruning must not
    wedge captures); clean shutdown. No event=state_lending assertion: that
    line needs the set_shed hook, which this path does not install.
    The ledger is printed per request, not bounded: it includes host backing
    reservations, not just checkpoints.
    """
    server = Server(args, log_path)
    results = {}
    captured = 0
    total = 0
    try:
        wait_ready(server, 300)
        model = served_model(args.port)
        prompts = [long_prompt(args.prompt_chars, head=f"Document {i}. ")
                   for i in range(1, args.stress + 1)]
        sequence = [("p%d" % i, p) for i, p in enumerate(prompts, 1)]
        sequence.append(("p1-repeat", prompts[0]))
        for index, (name, prompt) in enumerate(sequence):
            user = {"role": "user", "content": prompt}
            res = run_request(server, args.port, model, name, [user],
                              args.max_tokens, results)
            total += 1
            check(res["status"] == 200 and bool(res["content"]),
                  f"{name} status 200 with non-empty answer", failures)
            line = completed_chat(server, index)
            captures = captures_of(line)
            if captures is not None and captures >= 1:
                captured += 1
            print(f"{name} ledger={ledger_of(line)} captures={captures} "
                  f"cache={field(line, 'cache')} "
                  f"cache_reuse={field(line, 'cache_reuse')}")
        check(captured >= 3,
              f"at least 3 of {total} requests report cache_captures >= 1 "
              f"(got {captured})", failures)
        # event=state_lending is logged only when the resources' set_shed
        # hook is installed (component_text_cache.cpp, cache Impl destructor).
        # That hook is not installed on this RAM-only path, so its absence
        # proves nothing and is not asserted. The ledger series is the
        # evidence here instead.
    except RuntimeError as err:
        print(f"setup failure: {err}", file=sys.stderr)
        failures.append(str(err))
    finally:
        rc = stop_server(server, "stress")
        check(rc == 0, f"clean shutdown (exit {rc})", failures)
        print(f"serve log: {log_path}")
    finish(failures)


if __name__ == "__main__":
    main()
