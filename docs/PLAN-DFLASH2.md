# Build plan: DFlash2 speculative decoding for Flint

Status: approved to build (2026-11-15). Decision history and measurements live in
[`ROADMAP-DFLASH2.md`](ROADMAP-DFLASH2.md); this document is the execution plan.

## 1. What we are shipping, honestly

| depth | plain decode | projected with DFlash2 | content |
| --: | --: | --: | :-- |
| 0 | 71 t/s | **~1.5x** | code/JSON; prose ~1.2x |
| 8k | 63 t/s | ~1.4x | code/JSON; prose ~1.1x |
| 32k | 50 t/s | ~1.2x | code/JSON only |
| 131k | 28 t/s | ~1.0x | neutral |

The 2x gate this project started with is not reachable on this engine, and the
reasons are measured rather than assumed: verify rows genuinely route to
different experts (3.8 GB per cycle of the 5.4 GB verify delta at 8k is routed
expert reads), and global-layer KV is re-read per row. We build anyway, with a
bounded win, because the remaining cost is implementation and the risk is now
known.

**Kill criterion:** if M1 (drafter parity) has not converged in 3 working days,
stop and close the track. Everything before M1 is a prerequisite that stands on
its own merits.

## 2. Current state

Done and committed:

- `Session::AppendAll` verifies 1..8 rows **bit-identically to single-token
  decode** at 0 / 8k / 32k / 131k (`FLINT_CHECK_ROWS=1`, 256 positions each,
  256/256 exact, zero argmax flips). This is the precondition for a lossless
  gate and it did not hold before `kVecExpertRows` 32 -> 48.
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

Not built — the entire deliverable:

- drafter GGUF loader and forward pass
- `IDraftBackend` and `ISpeculativeTargetExecutor` implementations
- target hidden-state capture in the engine
- any integration with the decode loop

## 3. The integration gap (read this first)

`ext::TextModelRunner` is **single-token**: core's serve loop calls `SelectNext`
then `Advance` once per token. There is **no extension-facing speculative hook**
in `src/ext/` — verified 2026-11-15: the only speculative symbols on the ext
surface are `TextSpeculativeConfig` / `TextSpeculativeDefaults` (settings, no
factory).

Qwen does speculation because it is in-tree: `src/models/qwen/drivers.cpp`
constructs `SpeculativeVerifier` and `QwenDFlashGpuDraftBackend` and drives the
loop itself. The interfaces are public
(`build/gpu-test/sdk/public-include/src/core/speculative/{draft_backend,
speculative_verifier,prompt_lookup}.hpp`), so an extension can implement them —
it just has nowhere to hand them over.

Two ways forward:

**A — runner-internal loop with a token buffer (recommended).**
`SelectNext` runs one draft/verify cycle, rewinds to the accepted prefix, buffers
the accepted tokens and returns the first. `Advance` pops the buffer without a
forward pass (the KV already holds those tokens). No core change. Constraints:
greedy only; must keep `position()`, `CheckpointPosition` and EOS handling
consistent; must no-op `PreviewFirstToken` correctly.

**B — upstream a hook into gufo.** A virtual on `TextModelRunner` (or a
capability struct in `TextRunnerDescriptor`) that lets a provider own the
speculative loop, matching what Qwen already does internally. Correct long term;
adds review latency and is not under our control.

Plan: build **A** behind `FLINT_SPEC=dflash`, and open the B proposal upstream in
parallel. A is abandoned in favour of B the moment core exposes a hook.

## 4. Reference assets

| asset | path | use |
| :-- | :-- | :-- |
| Qwen DFlash2 stack | `/home/bruno/AI/Gufo/src/models/qwen/hip/dflash/` — `model.cpp` 415 L, `backend.cpp` 977 L, `executor.cpp` 1327 L, `batch.cpp` 607 L, `dflash.hpp` 485 L | port target; same drafter arch |
| Qwen verifier | `src/models/qwen/hip/qwen_speculative_verifier.{hpp,cpp}` | `ISpeculativeTargetExecutor` shape |
| layer capture | `src/models/qwen/hip/arena.cpp` `SetTargetLayerCapture`, `kMaxTargetLayerTaps` | how taps are wired |
| llama.cpp | `/tmp/opencode/llamacpp-ref/speculative.cpp`; upstream `src/models/dflash.cpp`, `dflash2.cpp` | lattice, target layers, `p_min` |
| drafter | `/home/bruno/AI/GGUF/mtp/Flint-DFlash2-Q6_K.gguf` | 81 tensors (32 F32, 49 Q6_K), 5 blocks, hidden 2048, block 8, SWA 2048, fc 2048->10240, no token embedding, `target_layers=[6,16,27,37,47]` |
| oracle | `/home/bruno/AI/llama/llama-b11804-rocm/` | acceptance/parity reference |

Layer-id mapping trap: llama.cpp lists the taps as layer **inputs**
`[6,16,27,37,47]`. `IDraftBackend::TargetHiddenLayerIds` wants zero-based layer
**outputs**, i.e. `[5,15,26,36,46]`.

## 5. Architecture

```
runner.cpp ──(buffered tokens)── SpecCycle (src/spec/cycle.{hpp,cpp})
                                   ├── TargetExecutor  (src/spec/executor.{hpp,cpp})
                                   │     └── gpu::Session::Append / AppendAll / Rewind
                                   ├── DflashBackend   (src/drafter/backend.{hpp,cpp})
                                   │     └── DflashForward (src/drafter/forward.{hpp,cpp})
                                   │           └── DflashModel (src/drafter/model.{hpp,cpp})
                                   └── speculative::SpeculativeVerifier (core, public)
engine.cpp ── hidden capture ring: SetHiddenCapture(ids) / Hidden(row, tap)
```

New files, all under the extension's `src/` (vendored code namespaced
`gufo::vendor::*` where it is copied from in-tree sources):

| file | contents | ~LOC |
| :-- | :-- | --: |
| `src/drafter/model.{hpp,cpp}` | `dflash` GGUF load, 81 tensors, F32 promotion | 400 |
| `src/drafter/forward.{hpp,cpp}` | encoder, 5 blocks (dynamic grouped conv, non-causal SWA 2048, gated MLP), selector lattice | 900 |
| `src/drafter/backend.{hpp,cpp}` | `IDraftBackend`: `Name`, `RequiresTargetHiddenStates=true`, `TargetHiddenLayerIds`, `PrimeTargetContext`, `AppendTargetContext`, `UpdateTargetHidden`, `AcceptFeedback`, `Propose`, `ProposeSampled`, `ProposeBatch`, snapshot hooks | 500 |
| `src/spec/executor.{hpp,cpp}` | `ISpeculativeTargetExecutor` over `gpu::Session` | 350 |
| `src/spec/cycle.{hpp,cpp}` | greedy cycle: draft -> `AppendAll` -> prefix accept -> `Rewind`; adaptive length; stats | 300 |
| engine changes | hidden capture ring (~80 MB at 2048 rows x 5 taps x 2048 F32) | 150 |

Memory: drafter Q6_K 336 MB + feature ring 80 MB + draft KV ring 40 MB, against
a 112 GiB unified heap.

## 6. Milestones

### M0 — Hidden capture (0.5 d)
- `Session::SetHiddenCapture(std::span<const uint32_t> layer_output_ids)` and
  `Session::Hidden(row, tap)`; capture happens after the residual add of the
  named layers, D2D into a ring buffer sized 2048 rows (SWA-2048 is all the
  drafter reads).
- Acceptance: capture off costs 0; on costs **< 2%** of a decode step; rows are
  identical across widths (use the M3 GX gate as the real check).
- Files: `src/gpu/engine.{hpp,cpp}`.

### M1 — Drafter loader + forward, parity against llama.cpp (2-3 d)  **[kills or confirms the track]**
- Load the GGUF; run the encoder + 5 blocks + lattice on a fixed
  (token, 5-tap) sequence.
- Parity harness: same prompt and same target, compare our draft tokens and
  selector scores against the oracle. Two levels, cheapest first:
  1. acceptance on the 12-prompt set must land within ±3 points of 51.1%;
  2. draft tokens must be **identical** to the oracle's on 20 fixed positions
     (add a `--dump-drafts` build of the oracle if needed).
- Acceptance: level 2 passes on all 20 positions.
- Failure action: if not converged in 3 days, stop per the kill criterion.

### M2 — Target executor (1-2 d)
- `ISpeculativeTargetExecutor` over `Session`: `ForwardPromptBatch`,
  `ForwardPromptSuffix`, `ForwardToken`, `SaveState`/`RestoreState` (snapshot
  API already exists), `CopyLastLogits`, `CopyVerificationLogits(row)`,
  `SetPromptHiddenCapture`/`GetPromptHiddenStates`.
- Acceptance: replaying a 2k prompt through the executor produces logits
  **bit-identical** to plain `Append`; `CopyVerificationLogits(k)` equals
  `Logits(k)`.

### M3 — Greedy loop + runner integration (1-2 d)
- `SpecCycle`: draft up to 7 -> `AppendAll([anchor, drafts])` -> walk rows ->
  `Rewind` to accepted prefix -> emit accepted tokens; buffer the extras.
- `runner.cpp`: `SelectNext` fills the buffer, `Advance` drains it; guard every
  invariant (position, checkpoints, EOS, `PreviewFirstToken`).
- Enabled only when `FLINT_SPEC=dflash` **and** sampling is greedy.
- Acceptance: **GX gate** — 12 greedy prompts byte-identical to plain decode.
  Any mismatch means the cycle is not lossless; do not proceed.

### M4 — Draft policy (1 d)
- Adaptive length priced by measured per-row cost (the Strycast pattern), plus
  `fixed`; honour `max_draft_tokens` / `min_draft_tokens` / `dflash_policy`.
- Acceptance: adaptive >= fixed at every depth in the sweep; stats
  (`draft_rounds`, `draft_tokens`, `draft_accepted_tokens`) match the oracle's
  within noise.

### M5 — Gates (2-3 d)
| gate | criterion |
| :-- | :-- |
| GX | spec on == spec off, byte-identical, 12 prompts x 4 depths |
| drafter quant | Q6_K primary; Q4_K_M and BF16 sanity, each validated for output (per-quant lesson) |
| depth curve | 0 / 8k / 32k / 131k, both policies |
| no regression | spec off == current numbers (71/63/50/28 t/s) |
| cost | verify width 8 ratio unchanged at 2.21 / 2.48 / 3.16 / 4.25 |
| memory | drafter + ring accounted in `ResourceClaim` |
| concurrency | > 1 request: speculation off initially, documented |

### M6 — Release (1 d)
- `docs/QUALITY.md` evidence, roadmap update, `extension.json` -> 0.3.0 +
  `speculative_defaults` (`dflash2`, max 7, min 1, policies fixed+adaptive),
  registry PR, llama-swap entry, issue #1 reply with the final table.

**Total: 9-13 working days**, of which M1 is the only real technical unknown.

## 7. Risks

| risk | impact | mitigation |
| :-- | :-- | :-- |
| Drafter forward mismatches the oracle (lattice, dynamic conv, SWA masking) | track dies | M1 parity gate before any integration; llama.cpp source is the spec; 3-day box |
| Buffered-token integration breaks a runner invariant | corruption | greedy-only, feature flag, GX gate, fall back to single-token on any mismatch |
| Hidden capture perturbs decode | silent quality issue | D2D copies only on 5 taps, measured < 2%, GX catches the rest |
| Adaptive policy over-drafts at depth | slower than plain | price per-row cost by depth; `min_draft_tokens` floor |
| Per-quant drafter regression (the Q6_K_XL lesson) | garbage output | M5 validates output per drafter quant, not just load/bench |
| Upstream hook (B) lands and differs from A | rework | A is thin: `SpecCycle` is the seam, only `runner.cpp` glue is A-specific |

## 8. Out of scope

- Reaching 2x; expert-grouped verify; prompt-lookup drafting (separate track,
  shares the cycle); batched speculation across concurrent requests; int4
  attention/`lm_head` requant (independent engine track with better ROI).
