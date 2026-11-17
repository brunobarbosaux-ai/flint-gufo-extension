# Registry PR: flint 0.3.0 (draft, not opened)

The public registry is `github.com/gufo-org/gufo-extension` (`index.json`,
`REVIEW.md`). A release enters only after a Gufo maintainer reviews its source at
the listed commit and approves the PR. This file is the draft. Do not open the
PR until the 0.3.0 commit exists and the digest is computed.

## Before opening

- [ ] Commit 0.3.0 (orchestrator). Record its full SHA as `revision`.
- [ ] Compute the digest with the registry's `tools/registry.py` (checked out
      from the registry repo; not available in this tree). It must match the tree
      at `revision`.
- [ ] Confirm the Gufo build revision used for validation: `c40862ff` (v0.32.0).
- [ ] The entry's `api_version` is `>= 0.2.15`. `speculative_defaults` was added
      in API revision 0.2.15 (commit `49380168`). Do not claim an older floor.

## Manifest diff (`index.json`, `extensions.flint`)

Add a release at the top of `releases`. The existing entries stay. Replace
`<COMMIT>` and `<DIGEST>` with the values from the steps above.

```diff
   "flint": {
     "description": "Flint 40B MoE text generation",
     "homepage": "https://github.com/brunobarbosaux-ai/flint-gufo-extension",
     "license": "MIT",
     "maintainers": [
       "@brunobarbosaux-ai"
     ],
     "releases": [
+      {
+        "api_version": ">= 0.2.15",
+        "digest": "sha256:<DIGEST>",
+        "notes": "DFlash2 speculative decoding (opt-in, FLINT_SPEC=dflash): drafter-proposed tokens verified in one batched pass; greedy output byte-identical to spec-off; decode speedup 1.24-1.35 at depth 0, temp 0. Declares speculative_defaults (max 7, min 1, adaptive).",
+        "platforms": [
+          "linux"
+        ],
+        "revision": "<COMMIT>",
+        "version": "0.3.0"
+      },
       {
         "api_version": ">= 0.2.7",
         "digest": "sha256:129994a14f8e0c02ba6301a729494b9779cf0b075b39c881323f89d31decdd0c",
```

No other registry fields change: `source`, `homepage`, `license`, `maintainers`
and `description` are already correct for this repo.

## PR title

`flint: add 0.3.0 (DFlash2 speculative decoding)`

## PR body

```markdown
## Extension
- Name: `flint` (Flint 40B MoE text generation, GGUF `flint` architecture)
- Version: `0.3.0`
- Source: https://github.com/brunobarbosaux-ai/flint-gufo-extension
- Revision: `<COMMIT>`
- License: MIT (`LICENSE`, `THIRD_PARTY_NOTICES.md`)
- Maintainer: @brunobarbosaux-ai
- Platforms: linux (gfx1151, ROCm 7.2.1)
- Gufo build: `c40862ff` (v0.32.0); `api_version >= 0.2.15`

## What is new since 0.2.0
- DFlash2 speculative decoding, **opt-in** (`FLINT_SPEC=dflash`, default off).
  A drafter proposes tokens and the target verifies them in one batched pass.
- Provider declares `speculative_defaults` (dflash2: max 7, min 1, adaptive;
  policies fixed and adaptive).
- Drafter file selected with `FLINT_DRAFT` (default `Flint-DFlash2-Q6_K.gguf`).

## Evidence (see `docs/QUALITY.md`, section "DFlash2 speculative decoding (0.3.0)")
- Greedy byte-identity, spec-on vs spec-off: 12/12 at depths 0, 8k, 32k, 131k.
- Null drafter (no-op speculation): 72/72 byte-identical.
- Settings matrix incl. grammar (budget-cut outputs not evaluated): passing.
- Drafter quants Q6_K, Q4_K_M, BF16: greedy 3/3 each; decode speedup 1.24-1.35
  at depth 0, temp 0.
- Spec-off decode unchanged: 70.96 / 63.56 / 51.05 / 28.18 t/s at 0 / 8k / 32k /
  131k.

## Known limits (stated in the docs, not hidden)
- Sampled mode (temp > 0): a fixed seed does not guarantee identical output with
  speculation on. Speculation is off by default for this reason.
- Temp 1.0 adaptive is about 3-6% slower than spec-off in the M4 measurements.
- Base-session VRAM undercount of about 600-666 MiB (pre-existing, in
  `DEFERRED.md`); speculative sessions add about 1.2 GB.
- Concurrency is serialized (one request at a time).

## Review notes
- Reviewer: the diff since 0.2.0 (`efacf0e`) is the speculative cycle
  (`src/spec/`, `src/gpu/dflash/`), the drafter, the gate tools (`tools/`), the
  `speculative_defaults` wiring in `provider.cpp`, and docs. Run
  `git diff efacf0e..<COMMIT> --stat` to confirm the list before opening the PR.
- No prebuilt binaries are vendored; the drafter is a user-supplied GGUF.
```

## Notes for the reviewer checklist

- **Network / processes / files:** the extension reads only the model and
  drafter paths given by the user. No network access, no telemetry.
- **Prebuilt blobs:** none. Vendored tiers are source (see `THIRD_PARTY_NOTICES.md`).
- **Metadata:** `extension.json` providers `flint` and keys `gguf:flint` describe
  what the code serves. `speculative_defaults` is set in `provider.cpp`.
- **Weight licensing:** weights are not bundled (`README.md`, Weights section).
