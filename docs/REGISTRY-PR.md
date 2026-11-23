# Registry PR: flint 0.4.0 — **opened as gufo-org/gufo-extension#27**

Status: PR <https://github.com/gufo-org/gufo-extension/pull/27> is open for
review. Revision `cf9f7aa02991049647abe9487979681e8fd99e2b` (tag `v0.4.0`),
digest `sha256:3b8e9e867fe187bb7e6fa2a18c3d7a483c8c45d16afc4e9667e685653ff07557`
from a clean clone at that revision. `registry.py check` and
`registry.py verify flint 0.4.0` both pass. The rest of this file is the draft
that was posted; keep it in sync with the branch if the review asks for changes.

The public registry is `github.com/gufo-org/gufo-extension` (`index.json`,
`extensions/`, `REVIEW.md`). A release enters only after a Gufo maintainer
reviews its source at the listed commit and approves the PR.

## Before opening (done)

- [x] Commit 0.4.0. Record its full SHA as `revision`.
- [x] Tag `v0.4.0` on that commit.
- [x] Compute the digest with the registry's tool, from a clean clone at that
      revision: `python3 tools/registry.py digest <clone>`.
- [x] Confirm the Gufo build used for validation: v0.36.0
      `4a7d8c6d178fc6979e5ae38c07b40e1fe89dd1ce`, extension API **0.3.42**.
- [x] The entry's `api_version` stays `>= 0.3.0`.
- [x] `python3 tools/registry.py check` and
      `python3 tools/registry.py verify flint 0.4.0` pass.

## Manifest diff (`index.json`, `extensions.flint`)

Add a release at the top of `releases`. The 0.3.1 entry and older ones stay.
The values below are the ones the branch uses.

```diff
     "releases": [
+      {
+        "api_version": ">= 0.3.0",
+        "digest": "sha256:3b8e9e867fe187bb7e6fa2a18c3d7a483c8c45d16afc4e9667e685653ff07557",
+        "notes": "Native tool calls (OpenAI tools/tool_choice, flint-glm dialect, buffered and streamed) and a per-layer row continuation cache: each layer's K/V is append rows, so a capture copies only the valid tokens and a follow-up turn only the new tail; per-session memory is unchanged from 0.3.1. Prefill is split-invariant and checkpoint-aligned; the drafter attends to history injected after a restore (8k repeat 41.5 -> 54.0 t/s, follow-up 43.5 -> 68.6); the GGUF mapping is released after upload (load RSS 26.3 -> 2.4 GB). Cache identity is flint-rocm-session-v2/flint-kv-rows-v2, so 0.3.1 disk caches are a miss by design. Greedy output is deterministic per build and spec-on equals spec-off; greedy output can differ from 0.3.1 because prefill now runs every pass on the prefill kernels and the K-quant WMMA dot uses explicit FMAs. The 8 GiB retained-snapshot cap is unchanged. Built and validated on Gufo v0.36.0 (API 0.3.42); source builds on any 0.3.x host. -c 262144: one session loads in ~10 s; two are admitted and use 78,646 MiB of device.",
+        "platforms": [
+          "linux"
+        ],
+        "revision": "cf9f7aa02991049647abe9487979681e8fd99e2b",
+        "version": "0.4.0"
+      },
       {
         "api_version": ">= 0.3.0",
         "digest": "sha256:a112bc6e7abc0f3094664301465eaf5e157391f7d8cb01828c87f20ebf2f13cd",
```

No other registry fields change: `source`, `homepage`, `license`, `maintainers`,
`description` and `platforms` stay as they are. Update `description` only if the
review asks for it.

## PR title

`flint: add 0.4.0 (tool calls, per-layer row cache, 0.36.0 validation)`

## PR body

```markdown
## Extension
- Name: `flint` (Flint 40B MoE text generation, GGUF `flint` architecture)
- Version: `0.4.0`
- Source: https://github.com/brunobarbosaux-ai/flint-gufo-extension
- Revision: `cf9f7aa02991049647abe9487979681e8fd99e2b`
- Tag: `v0.4.0`
- License: MIT (`LICENSE`, `THIRD_PARTY_NOTICES.md`)
- Maintainer: @brunobarbosaux-ai
- Platforms: linux (gfx1151, ROCm 7.2.1)
- Gufo build: v0.36.0 `4a7d8c6` (extension API 0.3.42); `api_version >= 0.3.0`

## What is new since 0.3.1
- **Native tool calls.** OpenAI-style `tools`, assistant `tool_calls`, `tool`
  messages and `tool_choice` (`auto`, `required`, `none`, named function),
  buffered and streamed. The prompt renders the template's `# Tools` and
  `<|observation|>` blocks byte for byte against 21 golden conversations; a
  `flint-glm` dialect supplies the grammar and scans calls back, including one
  still being written.
- **Per-layer row continuation cache.** Each layer's K and V are one
  `kAppendRows` component, so a capture copies only the valid tokens and a
  follow-up turn only the new tail. `payload_layout` is
  `flint-kv-rows-v2`; prefill is split-invariant with even-row checkpoints.
- **Drafter history after a restore** (#8). The drafter now attends to the
  history injected since a restore frontier instead of to none. At 8,074 tokens,
  repeat 41.5 -> 54.0 t/s and follow-up 43.5 -> 68.6; at 23,790 tokens
  49.2 -> 57.1 for the follow-up.
- **GGUF mapping released after upload** (#9). Load RSS 26,347 -> 2,447 MiB.
  The 24 GB of pages become evictable, so a cold load pays a disk read.
- **Correctness fixes** that changed greedy output vs 0.3.1: prefill runs every
  pass on the prefill kernels; the K-quant WMMA dot uses explicit FMAs; the
  sliding-window key tiles sit on the absolute `kWmmaKeys` grid.

## Evidence (see `docs/QUALITY.md`, section "Native tool calls and the per-layer row cache (0.4.0)")
- Tool template 21/21 byte-identical; tool scanner covers complete, typed,
  malformed and unfinished calls and streamed arguments only grow.
- GX greedy byte-identity spec-on vs spec-off: 3/3 at depth 0, 8k, 32k.
- Cache reuse, 24k, RAM and disk: exact bounds `r1 2,390,377,280` /
  `r3 30,203,008`, restore 607 ms on 0.36.0, identity line
  `flint-rocm-session-v2:flint-kv-rows-v2`.
- Cache-on vs cache-off: identical output on all three legs at 8,074 and
  23,790 tokens, with and without the drafter.
- `prefill_split`: 14 even splits, 0 differing, now runs in plain `ctest`.
- Memory: 2/2, claims identical to the 0.3.1 build.
- Suite: `engine_snapshot`, `target_adapter`, `ops`, `dflash_ops`,
  `dflash_forward`, `prefill_split` pass on 0.36.0.

## Known limits (stated in the docs, not hidden)
- 262144 is a **load** check; no 262144-token request has been run, so no
  throughput claim is made at that depth. Long-context requests are exercised
  to 130k.
- Two sessions at 262144 load in 10-187 s depending on page cache; one session
  is ~10 s. The spread comes from the released GGUF mapping (a deliberate
  trade-off for the RSS drop).
- The 8 GiB retained-snapshot cap is unchanged, so 130k prefix reuse is
  unavailable by design. Gufo 0.36.0 contains the linear-time fix; raising it
  needs its own A/B and is scheduled for 0.5.0.
- Sampled mode (`temp > 0`) does not guarantee identical output across spec
  on/off, and temp 1.0 is 3-6% slower with speculation. Speculation is off by
  default.
- Concurrency is serialized (one request at a time).
- Two concurrent requests that declare the same tool name with different
  schemas can mislead the tool scanner.
- Base-session VRAM undercount of about 600-666 MiB (pre-existing, in
  `DEFERRED.md`).

## Review notes
- The diff since 0.3.1 (`cb7303d5`) is: native tool calls (`src/tool_dialect*`,
  `src/chat_template.cpp`, `tests/tool_*`, `tests/golden/`), the per-layer row
  cache (`src/gpu/engine.*`, `continuation_adapter.*`, `runner.cpp`,
  `tests/prefill_split_test.cpp`, `gates/cache_reuse.py`, `tools/spec_gate.cpp`),
  the DFlash2 history fix (`src/spec/`), and the engine load changes
  (`src/gpu/engine.cpp`). Run `git diff cb7303d5..cf9f7aa --stat` to confirm.
- No prebuilt binaries are vendored. The model and drafter are user-supplied
  GGUFs.
- No network access, no telemetry, no writes outside the given model paths.
```

## Notes for the reviewer checklist

- **Network / processes / files:** the extension reads only the model and
  drafter paths given by the user. No network access, no telemetry.
- **Prebuilt blobs:** none. Vendored tiers are source (see `THIRD_PARTY_NOTICES.md`).
- **Metadata:** `extension.json` providers `flint` and keys `gguf:flint` describe
  what the code serves; `speculative_defaults` stays in `provider.cpp`.
- **Weight licensing:** weights are not bundled (`README.md`, Weights section).
- **Cache invalidation:** layout `v2` is a deliberate break from 0.3.1; the
  registry notes must state it so users do not report a false regression.
- **0.36.0 API additions unused:** shared restore, beside-decode restore,
  session-state declarations and `Idle()` are candidates for 0.5.0.
