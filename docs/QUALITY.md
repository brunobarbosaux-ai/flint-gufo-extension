# Flint extension quality report

Scope: what is validated, with what evidence, and what remains open. Failed or
inconclusive checks stay visible until they pass.

## Environment

- GPU: AMD Strix Halo (gfx1151), ROCm 7.2.1
- Gufo: main `6cc8a42ca84438a0566ee498a315f35c6190dc8e`, v0.31.0 (extension API series 0.2)
- Build: `gpu-test` preset (`RelWithDebInfo`, HIP on, tests on), option-aligned
  per-tier flags matching the reference fork's builds (fast-math on the vendored
  GEMM/mmq/w8a8 tiers; Flint's own kernels stay `-O3`)
- Extension revision: `efacf0e` (option alignment) with provenance `6dc4ce0`
- Oracle for all parity gates: pinned `llama.cpp` b11804 ROCm build, CPU mode,
  `flint-UD-Q4_K_XL.gguf`, fixed seed — the baseline does not drift the way a
  rebuilt reference binary would

## Passing checks

| Check | What it covers | Evidence |
| --- | --- | --- |
| `flint.config` | Architecture config parsing from GGUF metadata (in-memory containers) | `tests/config_test.cpp` |
| `flint.probe` | Artifact probe: `general.architecture == "flint"` detection, negative cases | `tests/probe_test.cpp` |
| `flint.ops` | Vendored HIP operators vs host reference: `RmsNormRows` (grouped), `SiluScale` | `tests/ops_test.cpp`, gfx1151 device |
| Registration | `gufo diagnose --section extensions` lists `flint` with provider available; in-tree models coexist in the same binary | re-verified 2026-11-07 on the option-aligned build |
| Link isolation | No symbol collisions with in-tree `qwen38_flash_next`, `gufo_qwen_text`, `gufo_core` (namespaces moved under `gufo::vendor::*`) | link of `gufo` with all extensions included |
| G5 (distribution) | T1.0 p0.95 k20 word distribution vs the pinned llama.cpp oracle, 200 rows/arm on the option-aligned binary: same support, same ordering, TV=0.0300 (<= 0.25), no missing rows | `evidence/g5-verdict-fresh.log`, 2026-11-07 |
| G8 (concurrency determinism) | Identical greedy output across 1..8 concurrent sessions (1508-char stream, 1 distinct) | re-run 2026-11-07 on the option-aligned binary: `evidence/g8-final-fresh.log` |
| GX (greedy determinism) | Two independent greedy captures bit-identical per binary — the extension and the reference fork each self-check PASS on all five prompts (no run-to-run drift) | `evidence/gx-self-final.log` (2026-11-07); cross-binary record in `evidence/gx-determinism-and-parity.log` |
| Build reproducibility | The option-aligned binary reproduces the pre-reboot gate run bit-for-bit: every G4 delta matches the 2026-11-06 run to 4 decimals against the independently re-recorded oracle baseline | `evidence/fresh-run-digests.sha256`, `evidence/g4-ours-vs-oracle-fresh.log` |
| Nix flake | `nix flake check` passes; the flake pins Gufo `6cc8a42` and builds+registers the extension (provider available on gfx1151) | 2026-11-07; runtime closure: 178 store paths, MIT top-level license, ROCm 10.1 stack dominates |
| Component-cache reuse | Warm-path output byte-identical to cold path for the same prompt ("What is 2+2?" -> `4`); serve logs show `reuse=ram`, `cached_tokens=1208`, `prefill_tokens=0` at ~2.8x lower latency | captured 2026-11-05 (transient script lost in the Nov 7 reboot; re-queued for `evidence/`) |
| Reasoning split | Flint's XML framing (`<think`/`</think`, the literals its chat template uses) is registered as both `ReasoningBudgetTags` and `ReasoningDelimiters`. The host splits the block: `reasoning_content` holds the thought, `content` the answer, and `reasoning_tokens` counts the block (7/61 tokens across probes) | captured 2026-11-05 (transient script lost in the Nov 7 reboot; re-queued for `evidence/`) |
| Reasoning budget | `reasoning_budget_tokens=16` forces the close at 16 reasoning tokens then emits the answer; requires `TokenizeControl` to resolve the special-token pair, so it is overridden with `parse_special_tokens` | captured 2026-11-05 (transient script lost in the Nov 7 reboot; re-queued for `evidence/`) |
| Depth curve (full) | Serial cold-prefill over HTTP at 131072 ctx, six depths x 3 runs: 8183 tok -> prefill 488.5 t/s, decode 64.2 t/s; 32748 tok -> 1137.3 t/s, 51.1 t/s; 65265 tok -> 1133.0 t/s, 40.2 t/s; 130808 tok -> 1035.8 t/s, 27.59 t/s (sd 0.24). Shallow depths are TTFT-dominated by the monolithic snapshot copy (see gate notes) | `evidence/depth-q4-summary.json`, `evidence/depth-q4-deep-summary.json`, raw rows in `evidence/depth-q4*.jsonl` |
| API boundary | Series-0.2 component-cache-only continuation; every include lands on the audited `gufo/ext/SURFACE` (the only exception, `gufo/ext/generated/flint_constants.hpp`, is host-generated from `extension.json`); no private `gufo/` includes | `grep` over the tree vs `include/gufo/ext/SURFACE` |
| Sampling defaults | One function sets both `TextRunnerDescriptor::sampling_defaults` and `TextProvider::sampling_defaults` (temp 0.6, top_p 0.95, top_k 20) | `runner.cpp` `FlintSamplingDefaults`, `provider.cpp` registration |

## Gate notes

### G4 (near-tie logit parity vs the oracle), 2026-11-07

Baseline: fresh oracle capture (pinned llama.cpp b11804, CPU, greedy + top-20
logprobs per token, 8 prompt classes at 512-token depth). Two serve arms were
scored against it with the harness's rank+delta verifier.

**Extension (option-aligned binary): 6 of 8 rows pass.** Every verified fork is
a rank-#2 near-tie, deltas 0.0169–0.2075. Two rows fail:

- `ascii` — rank #2, delta 0.5947, just over the harness's 0.5 envelope. The
  envelope was calibrated on the UD-Q6_K_XL stack; the harness itself documents
  that coarser artifacts need a wider bound. On Q4 the oracle's top-2 at that
  position is a genuine near-tie ("Maybe initials" vs "Maybe first letters").
  Resolved by the Q4 envelope calibration below: under the calibrated bound
  this row passes (7/8).
- `whitespace` — the oracle's own token stream for this row fails the
  harness's rebuild check (`full.strip() != recorded full`), so the fork point
  cannot be verified. Oracle-side harness limitation, not a serve-side defect.

**Envelope calibration (Q4).** The 0.5 delta / rank-3 envelope in the gate
harness was calibrated on the UD-Q6_K_XL stack. To decide the correct Q4
envelope we measured the pinned oracle's *own* prefill-vs-decode
disagreement: 397 near-tie positions re-scored through the batched-prefill
path and compared against the decode capture (pipeline validated — the
position-0 top-1 reproduces the capture on all 8 rows; 3339/3342 capture
positions align). Results:

- the oracle flips its own top-1 on **15.9%** of near-tie positions
  (63/397); every flip sits at a decode top1-top2 gap < 0.478 — none are
  structural disagreements;
- self-flip deltas: 89% within 0.5, tail to **1.335** (P95 0.709);
- the oracle's own pick fell as far as **rank 4** (once).

So the committed envelope (0.5 / rank 3) is *stricter than the oracle is
consistent with itself* on Q4. Proposed Q4 envelope: **delta <= 0.8 (P97 of
self-flips), rank <= 4** (covers the full observed rank range). Applied via
the harness overrides (`g4-compare --max-delta 0.8 --max-rank 4`):

- extension: **7/8** — `ascii` (rank 2, delta 0.5947) passes; only the
  oracle-side `whitespace` unverifiable row remains;
- reference fork: **6/8** — both its divergences (`whitespace` rank 4 delta
  0.5665, `code`) fall inside; only `tool_xml` unverifiable remains.

The constant change itself belongs in the Strycast harness (to be proposed
with this data); the calibration run and both re-scored logs are recorded
here. Evidence: `evidence/g4-envelope-calibration.json`,
`evidence/g4-compare-ours-q4env.log`, `evidence/g4-compare-fork-q4env.log`.

**Reference fork (live serve arm, same session): 5 of 8 rows pass.** The fork
passes `ascii` (delta 0.0178) but fails `whitespace` at rank **#4** (delta
0.5665) and has two unverifiable rows (`code`, `tool_xml`). Two conclusions:

1. On the Q4 artifact the reference implementation itself does not pass 8/8
   against the independent oracle — G4 on Q4 measures near-tie sensitivity, and
   both binaries sit in the same regime (6/8 vs 5/8).
2. The extension's standing against the oracle is at least as strong as the
   reference's, on the same session, same oracle, same artifact.

Determinism cross-check: the deltas reproduce the 2026-11-06 run exactly (same
build flags, same oracle captures), confirming the option-aligned build is
bit-stable across machine reboots and the oracle baseline is reproducible.

Evidence: `evidence/g4-ours-vs-oracle-fresh.log`,
`evidence/g4-fork-vs-oracle-fresh.log`, raw JSON digests in
`evidence/fresh-run-digests.sha256`.

### GX cross-binary note

Two independent captures per binary are bit-identical (GX self-check PASS for
both the extension and the reference fork), but cross-binary bit-parity is not
achievable: across three differently-built extension binaries the fork-vs-ours
divergence profile changed every time while self-determinism held. Bit-parity
across independently built binaries is therefore declared out of scope; the
gates that carry the quality claim are G4-vs-oracle, G5 distribution and G8.
Evidence: `evidence/gx-determinism-and-parity.log` (2026-11-06 record),
`evidence/gx-self-final.log` (2026-11-07).

### Toolchain caveat (Nix build)

The Nix-built binary (clang-24 / ROCm 10.1 from nixpkgs) registers and serves
correctly but runs ~3x slower than the CMake ROCm 7.2.1 build and diverges on
greedy tokens — different device codegen. The Nix build serves the
"reproducible build + registration" purpose; the 7.2.1 CMake build is the
correct one for gate parity and quality claims. This matches the Strycast
warning that other compilers can break bit-exact checks.

### Snapshot cost

Flint exposes only a monolithic session snapshot, so the component-cache
adapter copies the live KV (proportional to the frontier) on each capture.
Measured `cache_snapshot_ms` ~15.2 s at the 131072-token depth (13 GB payload);
shallow-depth TTFT in the curve is dominated by the same copy, so the curve
reports prefill/decode throughput and treats TTFT as a benchmark cost, not a
correctness signal. No per-row KV access exists in the Flint engine API to make
this incremental.

### Weight placement (v0.1.1, 2026-11-09)

Flint's weight loader allocated every tensor with a bare `hipMalloc`. On APU
machines the device heap is a cap below physical unified memory
(`ttm.pages_limit`), so a small heap cap made the first large tensor fail with
`device allocation failed` and the model never loaded. The loader now follows
the Flash-Next canonical pattern
(`src/models/qwen38_flash_next/kernels/rocm/device_model.cpp`):

1. a measuring pass sums the bytes every allocation requests;
2. `gufo::hip::WeightAllocator` (SDK `experimental/managed_alloc.hpp`) decides
   `kMalloc` vs `kManaged` once, from the heap free at construction;
3. the upload pass allocates through `weights.Allocate()`, which still falls
   back per-call to managed memory on `hipErrorOutOfMemory` — so a wrong
   one-time decision on an unusual machine degrades to slower paging, not a
   load failure.

`hipFree` remains correct for both allocation kinds on ROCm (confirmed against
Gufo's own models).

Validation on the development machine (Ryzen AI Max+ 395, 124 GiB LPDDR5X,
device heap 112 GiB — this machine cannot reproduce the small-heap condition,
so the `kManaged` branch is code-reviewed against the reference
implementation but not yet observed under a constrained heap):

- builds and installs through `gufo extension install` in the official
  runtime image (Gufo 0.32.0, API 0.2.16);
- greedy output identical to the previous module (no regression);
- GX self-determinism: 5 prompts x 2 greedy captures, all bit-identical
  (the `[ModelLoad]` wall-clock line is excluded from the comparison);
- throughput unchanged: pp512 ~2.1-2.2k tok/s, tg128 ~69 tok/s at depth 0.

### Demand-sized session contexts (v0.2.0, 2026-11-09)

Sessions were created with a hardcoded 8192-token context
(`kDriverContext`), so every prompt, chat and bench request was capped at
8192 total tokens regardless of the model's 262144 native context. The cap
was the extension's own constant, not a driver or hardware limit.

Sessions are now sized to what the request needs (`PickContext`):

- prompt + generation for one-shot prompts, recreated when a later chat
  turn needs more room;
- clamped to the model's native `context_length` from GGUF config;
- clamped to device memory: `Model::FreeDeviceBytes()` (hipMemGetInfo) is
  queried at session creation and the request is halved until
  `Model::SessionBytes` fits within 75% of free — long contexts degrade
  gracefully on small-heap machines instead of failing.

KV cost is linear: 49 layers x 2 x (4 KV heads x 128 head_dim) x 2 B =
~98 KiB/token, i.e. 0.77 GiB at 8k, 3.1 GiB at 32k, 24.6 GiB at 262k.

Measured depth curve (runtime image, depth + 512-token prompt + 128
generated, 1 repetition):

| depth | pp512 tok/s | tg128 tok/s |
| --: | --: | --: |
| 0 | 2252 | 69.5 |
| 8192 | 1988 | 60.6 |
| 32768 | 1421 | 47.9 |
| 131072 | 641 | 25.2 |
| 261503 | 383 | 15.8 |

Previously depths beyond ~6.6k were rejected outright. Short-prompt greedy
output is unchanged (no regression). The 261503 depth is the maximum that
fits the 262144 native context alongside the 512-token prompt and 128
generated tokens; degradation is smooth across the whole range with no
cliff.

## Open / not yet validated

- [ ] The `kManaged` weight-placement branch needs confirmation on a machine
      with a small device-heap cap (BIOS-reserved GPU memory or a low
      `ttm.pages_limit`). The per-call OOM fallback bounds the risk, but the
      one-time placement decision has not been observed under a constrained
      heap.

- [ ] Land the Q4 envelope proposal in the Strycast harness (`g4-compare`
      constants: delta <= 0.8, rank <= 4) with the calibration data — the
      measurement is done (see G4 gate notes); this is a harness change, not
      an extension change.
- [ ] Re-capture the component-cache-reuse and reasoning-probe evidence into
      `evidence/` (scripts lost in the 2026-11-07 reboot; claims date to
      2026-11-05 runs on the same source revision).
- [ ] Tokenizer parity gates (G1) against the Strycast in-tree reference.
- [ ] Install into a fresh prefix (runtime closure inventory is done: 178
      paths, MIT top-level, see Nix row).

## Artifacts

Evidence logs live in `docs/evidence/` (digests, sha256):

| File | Digest |
| --- | --- |
| `g4-ours-vs-oracle-fresh.log` | see `fresh-run-digests.sha256` |
| `g4-fork-vs-oracle-fresh.log` | see `fresh-run-digests.sha256` |
| `g5-verdict-fresh.log` | see `fresh-run-digests.sha256` |
| `g8-final-fresh.log` | see `fresh-run-digests.sha256` |
| `gx-self-final.log` | see `fresh-run-digests.sha256` |
| `g4-envelope-calibration.json` | see `fresh-run-digests.sha256` |
| `g4-compare-ours-q4env.log` | see `fresh-run-digests.sha256` |
| `g4-compare-fork-q4env.log` | see `fresh-run-digests.sha256` |
| `fresh-run-digests.sha256` | covers all raw JSONs + logs of the 2026-11-07 run |
| `g8-full.log` | `26fe837c42db3d12f4e60d02863e907daeba8028650832e1bae5cb932bf3dc2e` |
| `g5-verdict.log` | `f14aa5d3cea2af37688d4bcc3662f9512ce58fec55c797f1928cb75445d3abd3` |
| `g4-ours-vs-baseline.log` | `bfd04eac7ba59d11c2f199beb546f805e055fcd49bd53b3d4b58846f644c1e8e` |
| `g4-reference-self-parity.log` | `d6ab16e477bf724338078b89cedfe1a2e10a4adc71db0c488ea00d3d887dcaad` |
| `gx-determinism-and-parity.log` | `d42306d04ba65e1f6804fd3c04885755b781ea79ff48c74a37b2d0b50a6820ef` |
| `depth-q4.jsonl` | `fb7ed2be4bb91630b7bfee93a5aeb727cd63ea9fe4ef6e616561edaef05fb204` |
| `depth-q4-deep.jsonl` | `f5b23a1e6dc5f5e0bd743a1b1f925da26b2a0ee6bd56411312eb09ed21e5d3e9` |
| `depth-q4-summary.json` | `1a0808f28cef08e7d656d77ee2a0eb78f5260c6a96b9eb2f0591dbb1728184b1` |
| `depth-q4-deep-summary.json` | `0ac13c27ab8b486ef87afbd1a12637152e74481b2d2922b0092fa8437cebbf47` |

Raw capture JSONs (`g4-oracle-fresh.json` 12 MB, `g5-*-fresh.json`,
`gx-ours-final-*.json`) stay out of the repository; sha256 digests for the
2026-11-07 run are committed in `fresh-run-digests.sha256`.
