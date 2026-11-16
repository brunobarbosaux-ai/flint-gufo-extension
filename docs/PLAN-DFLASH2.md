# Build plan: DFlash2 speculative decoding for Flint

Status: approved to build (2026-11-15). Decision history and measurements live in
[`ROADMAP-DFLASH2.md`](ROADMAP-DFLASH2.md); this document is the execution plan.

The extension is **generic**: it serves whatever the end user asks for, on any
content, at any temperature. Nothing below may assume greedy decoding, coding
traffic, or a particular sampling configuration.

## 1. What we are shipping, honestly

| depth | plain decode | greedy (temp 0) | sampled (temp >= 0.6) |
| --: | --: | --: | --: |
| 0 | 71 t/s | ~1.4-1.5x | ~1.1-1.2x |
| 8k | 63 t/s | ~1.3-1.4x | ~1.05-1.15x |
| 32k | 50 t/s | ~1.1-1.2x | ~1.0x |
| 131k | 28 t/s | ~1.0x | ~1.0x |

Best content is code and JSON; prose is ~0.2x lower in every column. The greedy
column is an upper bound — it is the one measured on the oracle (51.1%
acceptance). The sampled column is estimated from distribution overlap and must
be **measured** in M3, because it is the common case for a generic extension.

The 2x gate this project started with is not reachable on this engine, and the
reasons are measured rather than assumed: verify rows genuinely route to
different experts (3.8 GB per cycle of the 5.4 GB verify delta at 8k is routed
expert reads), and global-layer KV is re-read per row. We build anyway, with a
bounded win, because the remaining cost is implementation and the risk is now
known.

**Kill criterion:** M1a (adoption probe) must load `Flint-DFlash2-BF16.gguf`
and emit draft tokens inside half a day, and M1b must match the oracle on 20/20
fixed positions, first on BF16 and then on Q6_K. If both checkpoints are not met
inside 3 working days, stop the drafter and ship the prompt-lookup fallback
(§3.7). Everything before M1a is a prerequisite that stands on its own merits.

## 2. Current state

Done and committed:

- `Session::AppendAll` verifies 1..8 rows **bit-identically to single-token
  decode** at 0 / 8k / 32k / 131k (`FLINT_CHECK_ROWS=1`, 256 positions each,
  256/256 exact, zero argmax flips). This is the precondition for everything in
  §3 and it did not hold before `kVecExpertRows` 32 -> 48.
- Multi-row attention (split `s` owns key blocks `b % splits == s`): width-8
  verify 192.2 -> 153.2 ms at 131k, plain decode 38.7 -> 36.0 ms there.
- Verify cost ratio at width 8: **2.21 / 2.48 / 3.16 / 4.25** at 0 / 8k / 32k /
  131k.
- Drafter quant chosen: **Q6_K** (336 MB; BF16 811 MB, Q4_K_M 240 MB). All five
  quants are in `/home/bruno/AI/GGUF/mtp/`.
- Oracle acceptance (llama.cpp b11804, `--spec-type draft-dflash`,
  `--draft-p-min 0.5`, 12 greedy prompts, 192 tokens): **51.1%**
  (1506/2947) — code 50.0%, JSON 62.8%, prose 41.9%; 3.69 drafts and 2.89
  accepted tokens per cycle. At `p_min=0`: 33.5%.
- Bench tooling: `FLINT_BENCH_VERIFY`, `FLINT_CHECK_ROWS` (1 invariance, 2
  reversed order, 3 tail dependence), `FLINT_PROFILE`, `FLINT_NO_GRAPH`,
  `FLINT_DEBUG_ATTN`.

Not built — the rest of the deliverable:

- **drafter forward pass: vendored, not written.** `src/models/qwen/hip/dflash/`
  is already generic over the drafter GGUF — `QwenDFlashConfig`
  (`src/models/qwen/dflash_weights.hpp:17`) reads `target_layer_ids`,
  `block_size`, `mask_token_id`, conv kernel/group, selector rank/top-k, SWA and
  layer count from the file — and its loader dequantizes Q6_K
  (`dflash/model.cpp:70`). The work is a config mapping and an adoption test.
- draft backend, target adapter and the speculative cycle
- target hidden-state capture in the engine
- any integration with the decode loop

## 3. How the cycle integrates without a core hook

### 3.1 The gap

`ext::TextModelRunner` is **single-token**: core's serve loop calls `SelectNext`
then `Advance` once per token. There is **no extension-facing speculative hook**
in `src/ext/` — verified 2026-11-15; the only speculative symbols on the ext
surface are `TextSpeculativeConfig` / `TextSpeculativeDefaults` (settings, no
factory).

Qwen does speculation because it is in-tree: `src/models/qwen/drivers.cpp`
constructs `SpeculativeVerifier` and `QwenDFlashGpuDraftBackend` and drives the
loop itself. The interfaces under `src/core/speculative/` are present in the SDK
include tree, but they live on a core path, not on `gufo/ext/`, and they are
aimed at in-tree multi-request serving with distribution-exact `q`. For a
generic, self-contained extension they are the wrong dependency: implementing
`IDraftBackend` means honouring ~25 virtuals (snapshot and persistent payload
semantics, `ProposeSampled` exactness, distinct-session batch rules) to use
about six, and the loop still has to be driven from the runner.

### 3.2 The design: batched forward, lazy per-row sampling

The speculative part is the batched **forward**, not the sampling. Per cycle:

1. **Prime.** Ask the drafter for up to `k` tokens from the accepted prefix.
2. **Verify once.** `AppendAll([anchor, drafts])` — one pass, logits and the five
   hidden taps for every row, then `Rewind` to the anchor.
3. **Accept lazily, one row per `SelectNext`.** Sample row *i* with the live
   `sampler`. If it equals draft *i*, the token is returned and the next call
   continues at row *i+1*. On the first mismatch, return the sampled token, drop
   the remaining rows (`Rewind`) and resume ordinary single-token decoding.
4. **`Advance` is a no-op during a cycle** — the KV already holds those
   positions. Position and `CheckpointPosition` are advanced as rows are
   consumed.

Sampling therefore happens exactly once per emitted token, in order, through the
same `sampling::SamplerState` core would have used, with its state intact. No
part of the sampler is reimplemented or bypassed.

### 3.3 Why this is correct for any sampler

At each position the emitted token is a draw from the user's own constrained
distribution: either the draft matched that draw, or the draw itself is emitted
and the cycle ends. Temperature, top-k, top-p, min-p, repeat and frequency
penalties, and any bound grammar or reasoning budget apply exactly as in
non-speculative decoding, because they are applied by the same sampler object.

This is only valid because row *i*'s logits are bit-identical to a single-token
decode at that position (§2, first item). If they were not, sampling row *i* from
a verify pass would not be the draw core would have made, and the scheme would
be silently wrong. This is the property Step 0.5 bought.

### 3.4 Expected acceptance

Acceptance is `P(the target's own draw equals the draft)`, i.e. distribution
overlap — not the `min(1, p/q)` rate proper speculative sampling achieves:

| settings | tokens/cycle (est.) | where it lands |
| :-- | --: | :-- |
| temperature 0 | ~2.9 (measured, 51.1%) | the 1.4-1.5x column |
| temperature 0.6-0.8 | ~1.6-1.9 | ~1.1-1.2x |
| high temperature / flat | ~1.1 | controller drafts 1-2, near-neutral |

Rejection sampling with the draft's `q` would raise the middle row, but it needs
the *transformed* target distribution (after penalties and constraint masking)
and `SamplerState` exposes only `Sample`, not that distribution. Reconstructing
it inside the extension would couple us to host internals and drift. Keeping the
sampler a black box is the right trade for a generic extension; the adaptive
controller is what keeps the high-temperature case from becoming a slowdown.

### 3.5 Limits to document, not to fix

- With a fixed seed, output is **not byte-identical** to non-speculative decoding
  (different RNG realization), though the distribution is identical. Greedy is
  exact and byte-identical.
- `FLINT_SPEC=off` must exist and be the default in the shipped config until the
  M5 gates pass, and a documented way to disable speculation per deployment is
  required for anyone needing seed-exact reproduction.

### 3.6 Upstream hook (optional, not a replacement)

A virtual on `TextModelRunner` (or a capability in `TextRunnerDescriptor`) would
let a provider own the loop the way Qwen does, and would make proper rejection
sampling possible. Worth proposing upstream, but the design here does not depend
on it and nothing is throwaway if it lands: the drafter, the adapter and the
cycle all stay; only the `runner.cpp` glue would be rewired.

### 3.7 If the drafter does not converge

The cycle, the target adapter, the hidden capture and the runner integration are
drafter-agnostic — the drafter is one implementation of `Propose`. If M1 fails
its checkpoints, the same code ships with a **prompt-lookup drafter**: an n-gram
match over the accepted prefix that proposes the tokens which followed the last
occurrence of the current suffix. No model, no extra memory, ~100 lines, and it
keeps every part of the investment already committed. Acceptance is lower on
open-ended text and higher on rewrite/edit traffic, and the same adaptive
controller prices it.

## 4. Reference assets

| asset | path | use |
| :-- | :-- | :-- |
| Qwen DFlash2 stack | `/home/bruno/AI/Gufo/src/models/qwen/hip/dflash/` — `model.cpp` 415 L, `backend.cpp` 977 L, `executor.cpp` 1327 L, `batch.cpp` 607 L, `dflash.hpp` 485 L | port target; same drafter arch |
| Qwen verifier | `src/models/qwen/hip/qwen_speculative_verifier.{hpp,cpp}` | verify-pass shape, row acceptance |
| layer capture | `src/models/qwen/hip/arena.cpp` `SetTargetLayerCapture`, `kMaxTargetLayerTaps` | how taps are wired |
| llama.cpp | `/tmp/opencode/llamacpp-ref/speculative.cpp`; upstream `src/models/dflash.cpp`, `dflash2.cpp` | lattice, target layers, `p_min` |
| drafter | `/home/bruno/AI/GGUF/mtp/Flint-DFlash2-Q6_K.gguf` | 81 tensors (32 F32, 49 Q6_K), 5 blocks, hidden 2048, block 8, SWA 2048, fc 2048->10240, no token embedding, `target_layers=[6,16,27,37,47]` |
| generic drafter | `src/models/qwen/dflash_weights.hpp:17` (`QwenDFlashConfig`, file-driven), `src/models/qwen/hip/dflash/model.cpp:70` (Q6_K dequant) | the drafter is config-driven: vendor it, do not rewrite it |
| oracle | `/home/bruno/AI/llama/llama-b11804-rocm/` | acceptance/parity reference |

Layer-id mapping trap: llama.cpp lists the taps as layer **inputs**
`[6,16,27,37,47]`; the tap is the residual **output** of layers
`[5,15,26,36,46]`.

## 5. Architecture

```
runner.cpp ──(lazy rows)── SpecCycle (src/spec/cycle.{hpp,cpp})
                              ├── TargetAdapter  (src/spec/target_adapter.{hpp,cpp})
                              │     └── gpu::Session::Append / AppendAll / Logits / Rewind
                              ├── DflashDrafter  (src/drafter/backend.{hpp,cpp})
                              │     └── DflashForward (src/drafter/forward.{hpp,cpp})
                              │           └── DflashModel (src/drafter/model.{hpp,cpp})
                              └── sampling::SamplerState (host-owned, one Sample per token)
engine.cpp ── hidden capture ring: SetHiddenCapture(ids) / Hidden(row, tap)
```

Nothing under `src/core/` is used. The extension depends only on `gufo/ext/*`,
the sampling and GGUF headers already in use, and its own engine.

New files, all under the extension's `src/` (vendored code namespaced
`gufo::vendor::*` where it is copied from in-tree sources):

| file | contents | ~LOC |
| :-- | :-- | --: |
| `src/drafter/model.{hpp,cpp}` | **vendored** `dflash` loader: 81 tensors, Q6_K dequant, F32 promotion, plus the config mapping | ~450 vendored + 150 written |
| `src/drafter/forward.{hpp,cpp}` | **vendored**: encoder, 5 blocks (dynamic grouped conv, non-causal SWA 2048, gated MLP), selector lattice | ~950 vendored |
| `src/drafter/backend.{hpp,cpp}` | internal drafter: context priming from the five taps, KV ring, `Propose(prefix, pos, max)`, `AcceptFeedback`; no `IDraftBackend` | 250 |
| `src/spec/target_adapter.{hpp,cpp}` | internal adapter over `gpu::Session`: prompt append, batched verify, row logits, hidden rows, rewind; no `ISpeculativeTargetExecutor` | 200 |
| `src/spec/cycle.{hpp,cpp}` | draft -> `AppendAll` -> lazy row acceptance -> `Rewind`; adaptive length from measured per-row cost; stats | 450 |
| engine changes | hidden capture ring (~80 MB at 2048 rows x 5 taps x 2048 F32) | 150 |

Memory: drafter Q6_K 336 MB + feature ring 80 MB + draft KV ring 40 MB, against
a 112 GiB unified heap. Written code is ~1,050 LOC; ~1,400 LOC are vendored from
the in-tree Qwen DFlash2 stack and carry `THIRD_PARTY_NOTICES.md` attribution.

## 6. Milestones

### M0 — Hidden capture (0.5 d)  **[done]**
- `Session::SetHiddenCapture(std::span<const uint32_t> layer_output_ids)`,
  `CapturedHidden(row, tap)` (host scratch) and `CapturedHiddenDevice(row, tap)`
  (in-place read for the drafter). Capture happens after the residual add of the
  named layers; tap order is the caller's. Ring: 2048 rows x taps x hidden F32.
- The copy is a kernel that reads the position from **device memory**, not a
  host-baked offset: the captured step graph replays at any depth, and changing
  the tap set invalidates the graphs.
- Measured (`FLINT_CHECK_HIDDEN`, `flint-UD-Q4_K_XL`): rows **bit-identical** to
  a single-token decode at every width 1..8 on all three paths (eager, graph
  capture, graph replay); wraparound verified at position 2100; distinct taps
  hold distinct residuals; cost **0.3-1.7%** of a decode step (noise band), and
  capture off launches nothing. `tg1` with capture off stays at 71.8 t/s.
- Files: `src/gpu/engine.{hpp,cpp}`, `src/gpu/kernels/kernels.{hpp,hip.cpp}`,
  `drivers.cpp` (`FLINT_CHECK_HIDDEN`).

### M1a — Adoption probe (0.5 d)  **[checkpoint 1 done]**
- `tools/drafter_probe.cpp` (target `flint_drafter_probe`, host tier, no GPU):
  opens the drafter through `gufo/ext/gguf_reader.hpp`, re-derives the whole
  loader contract — every metadata key, the diffusion topology rules, the tap
  convention, and the 81-tensor inventory with shapes — and exits non-zero on
  any mismatch.
- **Checkpoint 1 (artifact contract): passed on Q6_K, BF16, Q4_K_M and Q8_0.**
  Confirmed from the file itself: hidden 2048, intermediate 6144, heads 32 /
  kv 8 / head_dim 128, 5 blocks, block 8, conv kernel 2 / group 16, selector
  rank 256 / top-k 16, SWA 2048, context 262144, mask token 150019, vocabulary
  150144 (inferred from the selector, since the drafter has no head), taps
  `[6,16,27,37,47]` -> outputs `[5,15,26,36,46]`. 81 tensors: 32 vectors +
  49 matrices, 315.3 MiB of storage; byte-identical inventory in all four
  quants.

Two findings the probe produced that change the vendoring, not the plan:

1. **The export carries no rotary-dimension metadata** (`dflash.rope.*` is
   absent for all five quants). The in-tree loader *requires* it; llama.cpp
   falls back to `head_dim`. The vendored loader must adopt the fallback, or
   every Flint drafter is rejected before the first tensor is read.
2. **Q5_K_M is unusable as-is**: it stores Q5_K, which the in-tree
   `IsSupportedMatrixType` does not accept (F32/BF16/Q8_0/Q4_K/Q6_K only).
   Either drop Q5_K_M from the candidate set or add Q5_K, which the extension's
   MMQ tier already handles for MoE.

Checkpoint 2 is the forward itself, and the probe shows it is a **vendoring
job with four units**, not a 50-line shim:

| unit | size | state |
| :-- | --: | :-- |
| `src/families/qwen/kernels/dflash_kernels.{hpp,hip}` | 1,232 L | **done** — `src/gpu/dflash/`, compiles and passes `flint.dflash_ops` without weights |
| `dflash_weights.{hpp,cpp}` | ~400 L | needs shims for `QwenTensorRef` / `QwenLayerWeights` / `ModelConfig`; add the rotary fallback and Q5_K |
| forward + block workspace | ~600-700 L | `ForwardBlockBatch` / `BlockWorkspace` are methods of the executor; extract, do not copy the class |
| target embedding + head | ~40 L | the drafter borrows the target's `token_embd` and `output`; the engine exposes neither today |
| generic `Launch*` helpers | — | RMSNorm / RoPE / GEMM / embedding / residual-add: map onto the extension's vendored kernels instead of vendoring a second set |

### M1b unit 1 — draft kernels vendored  **[done]**
- `src/gpu/dflash/dflash_kernels.{hpp,hip.cpp}` are the in-tree files with
  **only three edits**: the include guard, the namespace
  (`gufo::hip::kernels` -> `gufo::models::flint::gpu::dflash`) and the include of
  `gufo/ext/sampling.hpp` in place of `src/core/sampling.hpp`. Every kernel body
  is byte-identical, so a stage-by-stage parity failure is a parity failure and
  not a re-derivation. The Q8_0 block layout comes from the `quant_ops.hpp` the
  extension already vendors (byte-identical to the in-tree one), imported with
  two `using` declarations because the enclosing-namespace lookup that served
  the in-tree file no longer applies.
- Compiled into `gufo_flint_hip` on the same tier as Flint's own kernels (-O3,
  no fast-math), not the fast-math GEMM tier.
- Gate: `flint.dflash_ops` (`tests/dflash_ops_test.cpp`), every launcher against
  a host reference on hand-checkable shapes — grouped dynamic conv for both
  attention and FFN sides, SiLU-mul, BF16->Q8_0 against a host quantizer
  (exact scale and every quantized byte), and the non-causal attention
  dispatcher compared to both a host reference and its scalar/wave routes
  (including a head dim that is not a multiple of four, where the wave route is
  invalid and the dispatcher must fall back). The selector is checked on the
  greedy path with vocabulary-wide transition codebooks and a fixture whose
  winner is *not* the logits argmax, so a kernel that ignored the codebooks
  fails.
- Two things the test surfaced that a weights-based test would have hidden:
  the wave attention route reads each key row as `float4` and is only valid for
  a head dim that is a multiple of four (the dispatcher already gates on it),
  and a host-compiled `hip_bfloat16` is a bare 16-bit field with no
  conversions — the test owns the BF16 conversion rather than depending on
  which header branch is active.

### M1b — Parity against the oracle (2 d, boxed)  **[kills or confirms the track]**
- Instrument the oracle to dump the encoder output, block-0 output and lattice
  scores for one fixed prompt, then compare stage by stage. End-to-end comparison
  only says "wrong"; intermediates say where.
- Level 1: BF16 drafter, 20 fixed positions, draft ids **identical**.
- Level 2: Q6_K drafter, 20/20 identical — this isolates quantization order from
  math, which is the most likely false alarm.
- Level 3: greedy acceptance on the 12-prompt set within ±3 points of 51.1%.
- Acceptance: levels 1 and 2 pass. Not met within 3 working days of M1a ->
  prompt-lookup fallback (§3.7).
- Start with the four vendoring units above, in that order: kernels first (they
  compile and unit-test without weights), then the loader with the rotary
  fallback, then the extracted forward, then the engine's tied-weight
  accessors.

### M2 — Target adapter (1-1.5 d)
- Prompt append (batched), `VerifyRows([anchor, drafts])` returning per-row
  logits and per-row 5-tap hidden rows, `Rewind`, and position tracking.
- Acceptance: `VerifyRows` rows are **bit-identical** to `Append` one token at a
  time (the `FLINT_CHECK_ROWS` property, asserted from inside the adapter);
  hidden rows equal the captured values; rewind leaves identical state.

### M3 — Cycle + runner integration (2 d)
- `SpecCycle` per §3.2, including `Advance` no-op, position bookkeeping, EOS
  mid-block (drop remaining rows), and `PreviewFirstToken` **must not** start a
  cycle.
- `FLINT_SPEC=dflash|off|null`: `null` proposes deliberately wrong drafts, so the
  cycle runs but nothing is ever accepted.
- Acceptance:
  - **GX gate** — greedy spec-on is byte-identical to spec-off, 12 prompts.
  - **null-drafter gate** — `FLINT_SPEC=null` reproduces spec-off **byte for
    byte with any sampler and seed**, because exactly one `Sample` is consumed
    per token. This is the strongest available test that sampler state and RNG
    order are preserved, and it is the one that protects non-greedy users.
  - measured tokens/cycle and speedup at temperature 0, 0.6 and 1.0 on the
    12-prompt set; the table in §1 is replaced with measurements.
- If either gate fails, do not proceed: the cycle is not distribution-correct.

### M4 — Draft policy (1 d)
- Adaptive length priced by measured per-row cost at the current depth (the
  Strycast pattern), plus `fixed`; honour `max_draft_tokens` /
  `min_draft_tokens` / `dflash_policy`.
- Acceptance: adaptive >= fixed at every depth, at temperature 0 **and** 0.7;
  never slower than spec-off at temperature 1.0; stats (`draft_rounds`,
  `draft_tokens`, `draft_accepted_tokens`) match the oracle within noise.

### M5 — Gates (2-3 d)
| gate | criterion |
| :-- | :-- |
| GX | greedy spec on == spec off, byte-identical, 12 prompts x 4 depths |
| null drafter | byte-identical to spec-off, temp 0 / 0.7 / 1.0, fixed seeds |
| settings honour | top-k, top-p, min-p, repeat/frequency penalty, grammar: no constraint violations, same settings as spec-off |
| drafter quant | Q6_K primary; Q4_K_M and BF16 sanity, each validated for output (per-quant lesson) |
| depth curve | 0 / 8k / 32k / 131k, both policies, temp 0 and 0.7 |
| no regression | spec off == current numbers (71/63/50/28 t/s) |
| cost | verify width 8 ratio unchanged at 2.21 / 2.48 / 3.16 / 4.25 |
| memory | drafter + ring accounted in `ResourceClaim` |
| concurrency | > 1 request: speculation off initially, documented |

### M6 — Release (1 d)
- `docs/QUALITY.md` evidence, roadmap update, `extension.json` -> 0.3.0 +
  `speculative_defaults` (`dflash2`, max 7, min 1, policies fixed+adaptive),
  registry PR, llama-swap entry, issue #1 reply with the measured table.
- Ship with speculation **opt-in** (`FLINT_SPEC=dflash`) until the sampled-mode
  numbers are known-good in the field; document the seed-reproducibility caveat.

**Total: ~10-12 working days** (~1,050 LOC written, ~1,400 vendored). M1 is the
only real technical unknown, and its first checkpoint is half a day away.

## 7. Risks

| risk | impact | mitigation |
| :-- | :-- | :-- |
| Drafter forward mismatches the oracle (lattice, dynamic conv, SWA masking) | track stalls | M1a adoption probe first, half a day; M1b compares encoder / block-0 / lattice intermediates instead of end to end; BF16 before Q6_K so quant order is not mistaken for a bug; 3-day box with a defined fallback (§3.7) |
| Sampler state or RNG order disturbed by lazy acceptance | wrong distribution for every non-greedy user | null-drafter gate (byte-identical, any seed), one `Sample` per token by construction, `PreviewFirstToken` must not cycle |
| Row logits drift from single-token decode | silent, subtle, content-dependent | M2 asserts bit-identity from inside the adapter; re-run `FLINT_CHECK_ROWS` after any engine change |
| Adaptive policy over-drafts at temperature and depth | slower than plain | price per-row cost; `min_draft_tokens` floor; M4 requires temp-1.0 never slower |
| Per-quant drafter regression (the Q6_K_XL lesson) | garbage output | M5 validates output per drafter quant, not just load/bench |
| Hidden capture perturbs decode | silent quality issue | D2D copies only on 5 taps, measured < 2%, GX/null gates catch the rest |
| Upstream hook (B) lands and differs | rework | only `runner.cpp` glue is design-specific; drafter, adapter and cycle are unaffected |

## 8. Out of scope

- Reaching 2x; expert-grouped verify; batched speculation across concurrent
  requests; int4 attention/`lm_head` requant (independent engine track with
  better ROI); distribution-exact rejection sampling (needs the transformed
  target distribution, see §3.4).
- Prompt-lookup drafting is not a separate track any more: it is the fallback
  drafter for a failed M1 (§3.7) and reuses the whole cycle.
