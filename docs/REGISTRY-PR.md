# Registry PR: flint 0.5.0 — **opened as gufo-org/gufo-extension#41**

Status: PR <https://github.com/gufo-org/gufo-extension/pull/41> is open for
review. Revision `dcccec1bec9941973c6e6e0e66273b8db1953c89` (tag `v0.5.0`),
digest `sha256:8508f149c1f80429cdf584c84737a38892bc6589ee1b4bd92f900ecf704ebac6`
from a clean clone at that revision. `registry.py check` and
`registry.py verify flint 0.5.0` both pass. The rest of this file is the draft
that was posted; keep it in sync with the branch if the review asks for changes.

The public registry is `github.com/gufo-org/gufo-extension` (`index.json`,
`extensions/`, `REVIEW.md`). A release enters only after a Gufo maintainer
reviews its source at the listed commit and approves the PR. 0.4.0 entered the
same way (PR #27, merged).

## Before opening (done)

- [x] Commit 0.5.0. Record its full SHA as `revision`.
- [x] Tag `v0.5.0` on that commit.
- [x] Compute the digest with the registry's tool, from a clean clone at that
      revision: `python3 tools/registry.py digest <clone>`.
- [x] Confirm the Gufo build used for validation: v0.36.0
      `4a7d8c6d178fc6979e5ae38c07b40e1fe89dd1ce`, extension API **0.3.42**.
- [x] The entry's `api_version` stays `>= 0.3.0`.
- [x] `python3 tools/registry.py check` and
      `python3 tools/registry.py verify flint 0.5.0` pass.

Note: the digest covers every file in the tree, so the tagged commit must not
contain the digest string. This file records it in a later commit, as it did
for 0.4.0.

## Manifest diff (`index.json`, `extensions.flint`)

Add a release at the top of `releases`. The 0.4.0 entry and older ones stay.
The values below are the ones the branch uses.

```diff
     "releases": [
+      {
+        "api_version": ">= 0.3.0",
+        "digest": "sha256:8508f149c1f80429cdf584c84737a38892bc6589ee1b4bd92f900ecf704ebac6",
+        "notes": "Sliding-window layers become a ring in the continuation cache (payload_layout flint-kv-rows-window-v1; state_abi unchanged), so a 130k frontier retains 4.0 GB instead of 12.9 GB: an identical repeat restores all 128,981 tokens from RAM in 51 ms and, after a disk restart, 128,980 tokens from disk in 1.4 s; the 8 GiB retained-snapshot cap is unchanged and no longer binds. Cache identity changes, so 0.4.0 disk caches are a miss by design and re-prefill. Weights load through a pinned staging buffer on four threads (#11): under page-cache pressure a 131072 load drops from 133 to 64 s for 0.3-0.5 s on a healthy node (8.0-8.5 -> 8.5-9.0 s). At 262144 the per-session request state falls from 26.0 GiB to about 8.7 GiB, so four sessions load in 9.5 s using 60.0 GiB and eight fit on this 128 GB box. Matrix-core decode attention (#7) is compiled but OFF by default; the shipped build behaves like 0.4.0 and flipping it is gated on documented prerequisites. Also: drafter_parity starts each prompt from Reset(), and Model::Load rejects an odd prefill_tile instead of producing wrong logits. Built and validated on Gufo v0.36.0 (API 0.3.42); source builds on any 0.3.x host.",
+        "platforms": [
+          "linux"
+        ],
+        "revision": "dcccec1bec9941973c6e6e0e66273b8db1953c89",
+        "version": "0.5.0"
+      },
```

## PR body (draft)

- Name: `flint` (Flint 40B MoE text generation, GGUF `flint` architecture)
- Version: `0.5.0`
- Source: https://github.com/brunobarbosaux-ai/flint-gufo-extension
- Revision: `dcccec1bec9941973c6e6e0e66273b8db1953c89`
- Tag: `v0.5.0`
- License: MIT (`LICENSE`, `THIRD_PARTY_NOTICES.md`)
- Maintainer: @brunobarbosaux-ai
- Platforms: linux (gfx1151, ROCm 7.2.1)
- Gufo build: v0.36.0 `4a7d8c6` (extension API 0.3.42); `api_version >= 0.3.0`

### What is new since 0.4.0

- **Ring KV window.** Sliding-window layers keep only the rows a pass can still
  read (`kWindowRows`), so the continuation cache no longer stores full-row K/V
  for them. `payload_layout` becomes `flint-kv-rows-window-v1`; `state_abi`
  stays `flint-rocm-session-v2`. At 130k the frontier snapshot drops from
  12,946,504,512 B (full rows) to **3,995,190,656 B**, which fits the unchanged
  8 GiB cap together with its follow-up (3,776,601,216 B).
- **Staged weight copy (#11).** The GGUF is copied into a pinned host buffer
  and uploaded from there on four threads instead of copying straight out of
  the mapping. Under page-cache pressure one 131072 load goes 133 s -> 64 s; on
  a healthy node it costs 0.3-0.5 s (8.0-8.5 -> 8.5-9.0 s).
- **Matrix-core decode attention (#7), off by default.** A WMMA path for the
  global layers' decode attention is compiled but selected only when
  `FLINT_MATRIX_ATTENTION=1` is given at configure time; the default build
  behaves like 0.4.0, and the `-matrix` layout suffix blocks cross-restore in
  both directions.
- **Correctness fixes.** `tools/drafter_parity` starts each prompt from
  `Reset()` (a deep rewind is illegal with a ring), and `Model::Load` now
  rejects an odd `prefill_tile` where it previously produced silently wrong
  logits.

### Evidence (see `docs/QUALITY.md`, section "Ring KV window and steadier loads (0.5.0)")

- Suite: 15/15 `flint.*` tests pass on the release commit, built from a clean
  clone.
- GX greedy byte-identity, spec-on vs spec-off: 3/3 at depths 0, 8k, 32k, 131k.
- Null drafter: 72/72 byte-identical; settings matrix 8/8.
- Cache reuse: 24k RAM/spec/sparse PASSED (restore 12.5-19.7 ms); 24k disk
  restart PASSED; 60k RAM PASSED (restore 26 ms); **60k disk restart PASSED**
  (0.4.0 was cap-refused); **130k disk restart PASSED** (128,980 tokens from
  disk in 1,394 ms, where 0.4.0 was a miss).
- Memory second-slot delta 2/2 with claims unchanged from 0.4.0.
- 262144: 4 sessions 9.5 s / 60.0 GiB, 8 sessions 14.0 s / 94.9 GiB; 131072
  loads 8.0 s (1 session) and 8.0-8.5 s (2 sessions), warm cache.

### Known limits (stated in the docs, not hidden)

- Load time still depends on the page cache. The staged copy halves the
  pressured case but does not remove the spread; quote load times only with a
  warmed cache. The 2026-11-17 hang has not reproduced in 24 loads.
- 262144 is a **load** check; no 262144-token request has been run through
  serve, so no throughput claim is made at that depth. Requests are exercised
  to 130k.
- Matrix attention is inert by default; enabling it is gated on `DEFERRED.md`
  prerequisites (setter vs layout, V-zeroing unproven, width-invariance limits).
- Sampled mode (`temp > 0`) does not guarantee identical output across spec
  on/off, and temp 1.0 is 3-6% slower with speculation. Speculation is off by
  default.
- Concurrency is serialized (one request at a time).
- Two concurrent requests that declare the same tool name with different
  schemas can mislead the tool scanner.
- Base-session VRAM undercount of about 600-666 MiB (pre-existing, in
  `DEFERRED.md`).

### Review notes

- The diff since 0.4.0 (`cf9f7aa`) is: the ring cache
  (`src/gpu/engine.*`, `continuation_adapter.*`, `src/cache/`), the staged copy
  (`src/gpu/device_model.cpp`), the matrix attention kernels and build switch
  (`src/gpu/kernels/`, `extension.cmake`), the window-aware cache gate
  (`gates/cache_reuse.py`), `tests/ring_snapshot_test.cpp`, `tests/ops_test.cpp`,
  `tests/target_adapter_test.cpp`, `tools/attn_bench.cpp`, plus docs. Run
  `git diff cf9f7aa..dcccec1bec9941973c6e6e0e66273b8db1953c89 --stat` to confirm.
- No prebuilt binaries are vendored. The model and drafter are user-supplied
  GGUFs.
- No network access, no telemetry, no writes outside the given model paths.

## Notes for the reviewer checklist

- **Network / processes / files:** the extension reads only the model and
  drafter paths given by the user. No network access, no telemetry.
- **Prebuilt blobs:** none. Vendored tiers are source (see `THIRD_PARTY_NOTICES.md`).
- **Metadata:** `extension.json` providers `flint` and keys `gguf:flint` describe
  what the code serves; `speculative_defaults` stays in `provider.cpp`.
- **Weight licensing:** weights are not bundled (`README.md`, Weights section).
- **Cache invalidation:** layout `flint-kv-rows-window-v1` is a deliberate break
  from 0.4.0's `flint-kv-rows-v2`; the registry notes state it so users do not
  report a false regression.
- **Matrix attention:** compiled but off; the default build selects the same
  scalar decode kernel as 0.4.0, so this option does not change decode numerics.
  (The ring changes what the cache retains, not the attention arithmetic; greedy
  output stayed byte-identical in the #11 A/B, and spec-on equals spec-off at
  every depth tested, but no 0.4.0-vs-0.5.0 byte comparison at depth was run.)
