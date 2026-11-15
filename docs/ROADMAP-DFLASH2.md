# Roadmap: DFlash2 speculative decoding for Flint

Plan for addressing the performance-improvement opportunity in issue #1
(DFlash2 drafter, int4 quant, rec cross-check). Ordered so cheap decision
points come before expensive work. NPU / XDNA2 work is excluded throughout
(deliberate scope rule: extension is GPU-only; the fork's NPU pipelines and
their fabric-clock requirements stay out).

Status markers: `[ ]` open, `[~]` in progress, `[x]` done.

## Context

- Decode baseline (extension, runtime image, UD-Q4_K_XL): 69.5 t/s at depth 0,
  25.2 t/s at 131k depth. Fork reports 72.6-73.6 t/s plain — baseline confirmed.
- Bandwidth math: ~2.1B active params at Q4 ≈ 1.2 GiB read/token → 69 t/s is
  ~83 GB/s, roughly a third of the ~256 GB/s LPDDR5X ceiling. ~3x headroom
  exists in kernel efficiency before speculation even enters.
- DFlash2 host machinery (draft policy, verifier contract, `draft_backend.hpp`,
  `speculative_verifier.hpp`) is already upstream in gufo 0.32.0 — the
  extension declares/implements the interface, it does not rebuild DFlash2.
- The fork has no Flint drafter either ("no MTP head and no drafter yet"), but
  its GLM-5.3 Flash integration (DFlash2 + prompt-lookup drafts, adaptive
  length, no MTP head) is the exact blueprint for a model like Flint.
- Drafter: `incoai/Flint-DFlash2-GGUF` — inspected 2026-11-14: dedicated
  `dflash` arch (not Flint-arch), hidden-state-conditioned, drafts 8 tokens
  per pass with selector confidences. Details in Phase 1.
- Standing lessons: per-quant validation is mandatory (Q6_K_XL loads, benches
  fine, and generates garbage — issue open); speed claims only on the
  runtime-image toolchain (Nix build ~3x slower, diverges); acceptance is
  sharply content-dependent (Laguna: 79-81% code/JSON vs 35% prose; 27B ~61%).
- Multi-row verification (`AppendAll`, `Logits(row)`, `Rewind`,
  `kMaxVerifyRows = 5`) already exists in the extension engine.
- **Phase 1 result (2026-11-14): DFlash2 is not worth integrating for Flint as
  it stands.** Measured on the b11804 oracle (which runs this drafter against a
  Flint target): mean accepted length 4.2-4.6 on code/JSON, 3.2 on prose;
  verify pass costs 2.3-2.7 single-token decodes; net 1.31x mean, 1.54x best,
  0.81x worst. The MoE verify wall, not acceptance, is the limit.

## Phase 0 — Quick wins from the fork (drafter-independent)

- [x] `FLINT_SPARSE_ATTN` block-sparse attention — **already wired** in the
      extension (shipped with the fork lineage); Phase 0 turned into
      validation: prefill at 131k depth 641 -> 1,681 tok/s (2.6x), decode
      unchanged, bit-identical greedy output at 4k depth, coherent at 32k
      (approximate beyond ~19k, documented in QUALITY.md).
- [ ] (moved to Phase 3) Prompt-lookup drafting — needs per-row logits to
      verify a draft block, i.e. the same multi-row verification foundation
      as DFlash2. Not a standalone quick win.
- [x] Document sparse-attention results in QUALITY.md — done 2026-11-14.
- [ ] Decide whether sparse ships as v0.3.0 with prompt-lookup deferred, or
      waits for the full v0.4.0 (decision pending).

## Phase 1 — Drafter study (go/no-go decision gate)  **[status: DONE — gate FAILED at 1.31x]**

The original plan assumed a standalone acceptance loop. Step 1 disproved that:
the drafter is hidden-state-conditioned, so acceptance cannot be measured
without the drafter forward and a target hidden-state export.

**Resolution (2026-11-14):** no extension code was needed after all. The pinned
parity oracle, llama.cpp b11804 ROCm, already implements the `dflash`
architecture (`libllama.so` carries `selector_hidden`, `conv_group_size`,
`target_layers`; `--spec-type draft-dflash` auto-detects DFlash2 from the
GGUF), and sets `res->t_layer_inp[il]` from its own Flint graph. Acceptance and
mean accepted length are properties of the (target, drafter) pair, not of the
engine — so the gate could be measured on the reference implementation
directly, with the extension still untouched.

- [x] Fetch `incoai/Flint-DFlash2-GGUF`; inspect architecture. **Done
      2026-11-14** (BF16 drafter at `/home/bruno/AI/GGUF/mtp/`, quants
      Q4_K_M/Q5_K_M/Q6_K/Q8_0 also present). Findings:
  - dedicated `dflash` architecture, 403M params, 5 blocks, hidden 2048,
    32 q / 8 kv heads, head_dim 128, non-causal block attention with
    sliding window 2048, 2-tap conv layers (`conv_kernel_size=2`,
    `conv_group_size=16`);
  - `dflash.block_size = 8` (drafts 8 tokens per pass);
  - `fc.weight` (2048 -> 10240) projects the **target hidden state**; there
    is **no token-embedding tensor** — the drafter does not embed tokens
    and cannot run standalone;
  - `selector_hidden` (2048 -> 256) plus rank-256 factorized
    `selector_predecessor`/`selector_successor` over the 150,144-token
    vocabulary produce the per-token selector confidences (the fork's >=0.7
    running-product cut applies to these);
  - tokenizer is Flint's own (bos 150000, eos 150001) — no vocab remap;
  - license: apache-2.0 (drafter), base model RecursiveSI/flint.
- [x] Build the acceptance harness — **not needed**; llama.cpp b11804 is the
      harness. Twelve prompts (4 code, 4 JSON, 4 prose), greedy
      (`temperature 0`, `top_k 1`), single slot, `-c 8192`, natural stop,
      `n_predict 192`:
      `llama-server -m flint-UD-Q4_K_XL.gguf -md mtp/Flint-DFlash2-BF16.gguf
      --spec-type draft-dflash --spec-draft-n-max 7 --spec-draft-p-min P
      -fa on -np 1 --temp 0`
- [x] Measure acceptance on REAL workloads. **Done 2026-11-14.**

  Baseline arm (same binary, no drafter): **54.3-54.6 tok/s**, flat across
  categories.

  | category | base t/s | spec t/s (p_min 0) | spec t/s (p_min 0.5) | best speedup | mean len | token acc | verify ratio |
  |---|---|---|---|---|---|---|---|
  | code | 54.6 | 73.7 | 71.2 | 1.30x | 4.23 | 50% | ~2.7 |
  | JSON | 54.3 | 71.7 | **83.9** | **1.54x** | 4.61 | 63% | ~2.6 |
  | prose | 54.3 | 56.1 | 58.3 | 1.07x | 3.16 | 42% | ~2.3 |
  | all 12 | 54.4 | — | — | **1.31x mean** | — | 51% | ~2.6 |

  Per-prompt spread: 1.57x (short code) down to 0.81x (rambling prose).
  `mean len` is accepted tokens per verify step (1..8); the server reports it
  as "mean len" in `slot print_timing`.
- [x] DECISION POINT. Measured verify cost ratio is **2.3-2.7x** a single-token
      decode, and speedup collapses to `mean_len / verify_ratio`.
      - Verdict: **FAIL.** 1.31x mean, 1.54x best category. The roadmap's own
        rule (<1.5x -> stop) applies; prose is a net loss.
      - Structural cause: Flint is MoE, 6 of 256 experts per token. An 8-row
        verify pass touches up to ~7x the expert weights of a single row, so
        the verify pass costs ~2.6 decodes. Reaching 2x would need
        `mean_len >= 5.2` at this ratio — i.e. ~85% token acceptance on
        ordinary text. Measured best is 4.6 (JSON).
      - This is not an implementation artifact: the reference implementation
        already reaches the acceptance the model supports (mean len 4-5 on
        code/JSON is healthy); the loss is entirely on the verify side, and
        llama.cpp cannot avoid it either.
- [ ] Open tiebreaker (cheap, no drafter work): the ratio was measured on
      llama.cpp, whose single-row decode is 54.4 t/s vs the extension's 69.5.
      If an 8-row `AppendAll` verify on our kernels costs <= 2.0 single-row
      steps, code/JSON would land at 2.1-2.3x and the gate would pass. One
      bench (time `AppendAll` at 1..8 rows, same prompt, same `-c`) settles
      it. Run it before closing the track; do not build the drafter first.
- [ ] If the bench confirms ~2.6x: close issue #1 with this data, keep Phase 0
      wins, stop.

## SDK contract — what the host already owns (gufo 0.32.0, inspected 2026-11-14)

Read from the runtime image (`/lib/gufo/sdk`). This is the single biggest
scope correction since the roadmap was written: gufo core already contains the
draft/verify loop, adaptive draft length and lossless rejection sampling. The
extension does **not** build a speculative decoder; it implements two adapters
plus one model.

- `gufo/ext/text.hpp`: `TextSpeculativeBackend{kDisabled, kDFlash, kDSpark,
  kMtp}`, `TextSpeculativeConfig{draft_model_path, max_draft_tokens = 7,
  min_draft_tokens = 1, dflash_policy: fixed|adaptive}`, stats
  `draft_rounds / draft_tokens / draft_accepted_tokens`.
- `src/core/speculative/speculative_verifier.hpp`: `SpeculativeOptions`
  (`target_acceptance_rate = 0.70`, `enable_adaptive_draft_length`,
  `use_batched_verification`), `VerifySampledToken` (rejection sampling),
  `SpeculativeStats`.
- `ISpeculativeTargetExecutor` — the **target adapter** we implement over the
  engine: `ForwardPromptBatch`, `ForwardPromptSuffix`, `ForwardToken`,
  `SaveState`/`RestoreState`, `FinishVerification`, `SetPromptHiddenCapture(
  enabled, target_layer_ids)`, `GetPromptHiddenStates`, `CopyLastHidden`,
  `CopyLastLogits`, `CopyVerificationLogits(row)`,
  `ForwardVerificationChunk(candidates, start_pos, capture_hidden,
  capture_logits)` (default loops `ForwardToken`; override to batch).
- `IDraftBackend` — the **drafter adapter** we implement: `Name`,
  `RequiresTargetHiddenStates`, `TargetHiddenLayerIds`, `PrimeTargetContext`,
  `AppendTargetContext`, `UpdateTargetHidden`, `AcceptFeedback`,
  `Propose`/`ProposeSampled`/`ProposeBatch`, and snapshot hooks
  (`SnapshotPayloadBytes`, `SerializePersistent`).

Mapping to the extension:

| SDK method | Extension primitive |
|---|---|
| `ForwardToken`, `ForwardPromptSuffix` | `Session::Append` |
| `ForwardVerificationChunk` | `Session::AppendAll` (batched rows) |
| `SaveState` / `RestoreState` | `Session::Rewind` / re-append |
| `CopyVerificationLogits(row)` | `Session::Logits(row)` |
| `SetPromptHiddenCapture` / `GetPromptHiddenStates` | new: per-layer residual capture |
| `PrimeTargetContext` / `AppendTargetContext` | drafter KV injection |

Two contract details that fail silently if missed: the SDK asks for zero-based
layer **outputs**, while llama.cpp's `dflash.target_layers = [6, 16, 27, 37,
47]` are layer **inputs** (i.e. outputs `[5, 15, 26, 36, 46]`), and the drafter
attention is SWA-2048, so only the last ~2048 feature rows need to exist
(`~80 MB`), not the whole prompt.

## Step 0 — Verify-cost bench  **[DONE — numbers superseded by Step 0.5]**

Built at gufo 0.32.0 (`gpu-test`, `RelWithDebInfo`, gfx1151, BUILD_TESTING=OFF,
`GUFO_EXTENSIONS=flint`) with `Session::kMaxVerifyRows` raised 5 -> 8 to match
the engine's `kVecBatch = 8` batched path. `FLINT_BENCH_VERIFY=1` times
`AppendAll` at every width after a prefill, rewinding between steps, on varied
real text (not the bench seed cycle).

| width | ms/step @0 | vs w1 | ms/step @8k | vs w1 | @32k | vs w1 | @131k | vs w1 |
| --: | --: | --: | --: | --: | --: | --: | --: | --: |
| 1 | 13.98 | 1.00 | 16.09 | 1.00 | 20.61 | 1.00 | 38.69 | 1.00 |
| 4 | 21.34 | 1.53 | 26.66 | 1.66 | 42.48 | 2.06 | 101.06 | 2.61 |
| 5 | 25.14 | 1.80 | 31.51 | 1.96 | 51.14 | 2.48 | 123.09 | 3.18 |
| 6 | 31.48 | 2.25 | 38.75 | 2.41 | 62.12 | 3.01 | 149.96 | 3.88 |
| 7 | 33.50 | 2.40 | 41.84 | 2.60 | 68.85 | 3.34 | 171.34 | 4.43 |
| 8 | 35.58 | 2.55 | 45.14 | 2.81 | 76.79 | 3.73 | 198.33 | **5.13** |

Width-1 confirms the shipped depth curve (71.6 t/s at 0, 25.8 t/s at 131k),
so the bench is measuring the production decode path.

**Verdict: the gate fails at every depth.** Projected speedup =
`mean_len / ratio`, with mean len from the Phase 1 study (code 4.23, JSON 4.61,
prose 3.16):

| depth | 2x needs mean len | code | JSON | prose |
| --: | --: | --: | --: | --: |
| 0 | >= 5.10 | 1.66x | 1.81x | 1.24x |
| 8k | >= 5.62 | 1.51x | 1.64x | 1.12x |
| 32k | >= 7.46 | 1.13x | 1.24x | 0.85x |
| 131k | >= 10.3 | 0.82x | 0.90x | 0.62x |

Best measured mean len is 4.61 (JSON), and those are upper bounds: acceptance
itself also degrades with depth. Past ~16k the verify pass costs more than the
tokens it returns, which is disqualifying for a model whose selling point is
262k context.

**Structural cause, and the only real fix.** Each verify row re-reads the KV
cache independently, so the cost grows linearly with width *and* depth: MoE
expert reads put a ~2.5x floor at depth 0, and global-layer attention adds the
rest (2.55 -> 5.13 between 0 and 131k). No drafting scheme escapes this —
`ngram`/prompt-lookup drafts even wider blocks and hits the same wall harder.
The fix is a multi-row attention kernel that reads each KV block once for all
rows (and expert-grouped verify), which is an engine project, not a drafter
project, and it is the prerequisite for *any* speculation on this engine.

- [x] `FLINT_BENCH_VERIFY` bench in `RunFlintBench`.
- [x] `kMaxVerifyRows` 5 -> 8.
- [x] Depth sweep 0 / 8k / 32k / 131k.
- [x] Decision: ratio > 2.3 at every depth -> **stop the DFlash2 track**.
- [ ] File the multi-row attention idea as its own work item (it also makes
      prompt-lookup drafting viable, and it is measurable with this bench).
- [ ] Note: hipGraph capture is unavailable in this environment
      (`hipEventSynchronize ... capturing stream`), so all widths ran on the
      eager step path — the same path the shipped 69.5 t/s figure uses.
- [ ] Oracle n-gram probe skipped as moot: n-gram drafts are 48-64 tokens wide,
      where the ratio is far worse than 2.5x.

## Step 0.5 — Row invariance  **[DONE 2026-11-15: fixed, and it moved the gate]**

Greedy speculation is only lossless if the target's argmax does not depend on
how many rows the pass that produced it carried: the verifier compares draft
tokens against row *i* of a batched pass, while production decode produces that
row one token at a time. `FLINT_CHECK_ROWS=1` measures it directly — at a fixed
cache position, compare the argmax of `Logits(0)` from `Append` against row 0 of
`AppendAll(w)` for w = 1..8, 256 positions per depth.

**It failed, with a clean boundary at width 6:**

| width | flips | bit-identical | median max \|dlogit\| |
| --: | --: | --: | --: |
| 1-5 | 0/256 | 256/256 | 0 |
| 6-8 | 5/256 | 0/256 | 0.578 |

Two probes localised it. Reversing the width order changed nothing (not state
leak), and `FLINT_CHECK_ROWS=3` — same width, different trailing rows — left
row 0 bit-identical at every width, so the batch is causally correct: later
drafts cannot contaminate earlier rows.

The cause was one constant. `kVecExpertRows = 32` splits the MoE in two: for
`n * expert_used <= 32` (n <= 5 at 6 routed experts) the gather path runs, above
it the routed path with per-expert tile lists, padding and a per-layer host
round-trip. DFlash2's verify width is 8, so the widths that matter were on the
routed path — different kernels, different rounding, and a router whose top-6
selection can flip on a near tie, which is what the 0.578 median delta and the
2.07 outlier were.

**Fix: `kVecExpertRows` 32 -> 48** (8 verify rows x 6 experts; the vector MoE
dispatch is capped at `MMVQ_MAX_BATCH_SIZE = 8`). Effect, all depths, all
widths: 256/256 bit-identical to decode, zero flips. It also removed the
host round-trip from the verify step:

| width | ratio @0 | @8k | @32k | @131k |
| --: | --: | --: | --: | --: |
| 5 | 1.81 | 1.93 | 2.38 | 3.08 |
| 7 | 2.06 | 2.27 | 2.99 | 4.13 |
| 8 | **2.18** | **2.44** | **3.35** | **4.75** |
| 8 (before) | 2.55 | 2.81 | 3.73 | 5.13 |

The 5 -> 6 step in the old curve (1.80 -> 2.25) was the path switch, not the
extra row; the curve is now smooth. No change to plain decode or prefill
(71.6-72.2 tok/s, 2395-2405 tok/s), because n = 1 never crosses either
threshold.

**Re-projected speedup.** With the oracle's measured schedule (3.69 drafts per
cycle, 1.89 accepted, so 2.89 tokens per verify) and a Q6_K drafter (+2.1 ms
per cycle), against each depth's own plain decode:

| depth | plain | verify@~4.7 rows + draft | tokens/cycle | projected |
| --: | --: | --: | --: | --: |
| 0 | 71.9 | 26.0 ms | 2.89 | **1.54x** |
| 8k | 63.0 | 32.2 ms | 2.89 | 1.42x |
| 32k | 49.5 | 47.6 ms | 2.89 | 1.22x |
| 131k | 26.7 | 112 ms | 2.89 | 0.97x |

Before this fix the same arithmetic gave 1.07x / 1.00x / 0.77x / 0.61x.

So the track is no longer blocked on correctness, and the projected gain roughly
doubled — but the gate (2x) is still missed at every depth, and the reason is now
purely the depth term: at 131k a width-8 verify pays 178 ms against 37.5 ms for
one row, ~140 ms of which is the same global KV re-read eight times (26.3 GB at
~188 GB/s). Widths 1-8 are otherwise row-invariant and cheap.

- [x] Row-invariance check (`FLINT_CHECK_ROWS=1|2|3`), 256 positions x 4 depths.
- [x] Cause: MoE path switch at `kVecExpertRows`; not a mask or state fault.
- [x] Fix: `kVecExpertRows` 48; bit-identical widths, verify 10-14% cheaper.
- [ ] Multi-row attention: read each global KV block once for all verify rows.
      That is the only remaining term between 1.5x and 2x+ at long context.
- [ ] Re-run the quant sweep against a fixed token stream once acceptance is
      measured on our engine rather than llama.cpp's (see Step 1.5 note).

## Step 1.5 — Why cross-quant acceptance numbers are not comparable

llama.cpp greedy output is not batch-invariant: with speculation off, `-ub 512`,
`-ub 8` and `-ub 4` produced different text on 12/12 prompts. Drafters change
the verify batch shape, so each quant generated slightly different text and its
acceptance figure describes a different workload. Aggregate placement stands
(all quants 49-56%), but no per-quant winner can be read from it. Q6_K is the
pick on size alone (336 MB vs 811 MB BF16, 2.4x less drafter traffic per cycle).

## Implementation plan (recorded, not pursued)

**Step 0 did not pass (2026-11-14).** The table stays as the cost model: it
applies unchanged if a multi-row attention kernel ever brings the ratio under
2.0, which is the precondition for starting Step 1.

| # | Step | Effort | Notes |
|---|---|---|---|
| 1 | Target adapter | 2-3 days | `ISpeculativeTargetExecutor` over `Append`/`AppendAll`/`Rewind`; per-layer residual capture; off-by-one validation vs llama.cpp |
| 2 | Drafter module | 1-2 weeks | `dflash` loader (81 tensors, BF16->F32 at load), encoder, 5 blocks, dynamic conv, non-causal SWA attention, selector lattice; differential test vs llama.cpp `dflash2_lattice` |
| 3 | Wire `IDraftBackend` + provider defaults | 2-4 days | `RequiresTargetHiddenStates`, layer ids `{5,15,26,36,46}`, `PrimeTargetContext` / `AcceptFeedback`, policy `adaptive` |
| 4 | Gates | 3-5 days | GX lossless greedy, G8, per-quant drafter, depth curve, no-regression with speculation off |
| 5 | Release | 1-2 days | v0.3.0, `QUALITY.md` evidence, registry PR, llama-swap entry |

Memory envelope: drafter weights 1.6 GB (F32) + feature rows 80 MB + drafter
KV ring 40 MB — negligible against the 112 GiB heap.

## Phase 2 — Drafter module  **[on hold — gated on Step 0]**

**Corrected by the llama.cpp reference (2026-11-14).** The drafter is an
EAGLE-style model with its own KV cache, not a standalone block model. What the
extension would actually need:

- [ ] **Target feature export, 5 layers not 1**: `dflash.target_layers =
      [6, 16, 27, 37, 47]` in the drafter GGUF. The `fc` input is the
      concatenation of those five layers' *inputs* (the residual stream
      entering layer `i`, `inpL`), in that order: 5 x 2048 = 10,240 -> 2048.
      A single pre-`lm_head` row does not satisfy the contract. llama.cpp
      keeps them as device rows (`llama_set_embeddings_layer_inp_dev`) so the
      drafter reads them without a host round-trip.
- [ ] **Dual-mode drafter graph**: an *embd* batch runs the injection path
      (`fc` -> encoder RMSNorm -> per-layer K/V projection -> `cpy_k`/`cpy_v`
      into the drafter KV cache); a *token* batch runs block diffusion over
      `[anchor, MASK x 7]` with non-causal, cache-aware attention.
- [ ] **Dynamic grouped conv, not a fixed 2-tap**: `conv_proj` predicts
      per-position, per-group coefficients (kernel 2, group 16) from the
      normalized hidden state; they are added to a static `conv_base` and
      applied before attention (side 0) and after the attn/FFN output
      (side 1). Every layer carries its own.
- [ ] **Selector lattice, not scalar confidences**: top-16 candidates per
      block position, rank-256 `selector_predecessor`/`selector_successor`
      embeddings gated by `selector_hidden`, walked on the host by argmax
      (`--spec-draft-p-min` truncates on the selector probability). DFlash2
      never feeds those logits to a sampler.
- [ ] Drafter shares the target's token embedding and `lm_head` (it has
      neither), so a full-vocab drafter cannot run without the target loaded.
- [ ] Gate: drafter-only agreement check against the drafter's own reference
      (per-quant paranoia; Q6 lesson).

## Phase 3 — Draft/verify loop  **[core-owned, not extension work]**

**Updated 2026-11-14:** this phase largely disappears. gufo core owns the loop,
adaptive length, prefix acceptance and rejection sampling (see the SDK section).
The extension supplies rows and logits; `kMaxVerifyRows` is the only engine
constraint to revisit. What remains is wiring and measurement:

- Draft length is capped at `block_size - 1 = 7`, not 8 — the selector only
  scores positions 1..7.
- Verification is a single batched forward of `[anchor + drafts]` followed by a
  prefix-acceptance walk, `Rewind` to the accepted prefix; `draft_tokens` and
  `draft_accepted_tokens` come from core stats.
- `p_min` truncation (0.5 was the best point here) trades mean length for
  verify width: it lifted JSON 1.32x -> 1.54x and prose 1.03x -> 1.07x while
  costing code 1.35x -> 1.30x.
- [x] Multi-row logits (`AppendAll` / `Logits(row)`) — present.
- [ ] Prompt-lookup drafting (moved here from Phase 0): the same
      append/verify/rewind machinery; the fork implements it in its CLI
      driver, not in serve. This is the only variant that still makes sense
      without a drafter model, and it shares the same verify-cost wall.
- [ ] Gates: GX with speculation on (drafted greedy text must equal plain
      greedy text except at near-ties), G8 concurrency with speculation,
      no-regression on plain decode.
- [ ] Tool-call / grammar turns: keep verification on-GPU (fork lesson:
      59 -> 90+ t/s on 27B).
- [ ] Gates: GX with speculation on (drafted greedy text must equal plain
      greedy text except at near-ties), G8 concurrency with speculation,
      no-regression on plain decode.

## Phase 4 — Validation + release

- [ ] Full gate battery on UD-Q4_K_XL: G4/G5/G8/GX, depth curve with
      speculation on/off, acceptance x speedup table across content types.
- [ ] Speed claims on the runtime-image toolchain only.
- [ ] v0.4.0 + registry PR with the acceptance study attached as evidence.
- [ ] Update the llama-swap `flint-ext` entry with
      `--speculative dflash2 --dflash-model ...` once validated.

## Parallel track — int4 quant (independent, interleavable)

- [x] `RecursiveSI/flint-int4-gguf`: format checked — it is **Q4_1**, which
      the extension's kernels do not cover (Q4_K/Q5_K/Q6_K/Q5_1/Q8_0/IQ4_XS
      only). **Does not run; track parked.** Unsloth GGUFs
      (UD-Q4_K_XL, UD-Q6_K_XL) remain the supported checkpoints.
- [ ] Revisit only if a Q4_1 (or other supported-format) int4 artifact
      appears, or if a Q4_1 kernel is worth writing.
- [ ] Compounds with DFlash2: smaller target weights -> cheaper verify rows.

## Parallel track — rec cross-check

- [ ] Read `recursive-oss/rec` (Apple engine) for kernel schedules,
      expert-dispatch ordering, KV layout ideas. Low cost, reference reading.

## Explicit non-goals

- NPU / XDNA2 anything (fabric-clock hold requirements, machine-hang risk,
  out of extension scope).
- MTP (Flint has no MTP head; prompt lookup is the only drafter-less option).
- Cross-binary bit-parity (declared out of scope in QUALITY.md).

## Reference notes (fork `mikealanni/gufo`, audited 2026-11-14)

- The fork has **no DFlash2 / draft-model speculation for Flint**: its
  Flint speculation is lossless prompt-lookup only, on by default in
  `gufo prompt`, explicitly not in serve; its README states "there is no
  draft model for Flint". Speeds: plain/lookup 72.6/74.9 code,
  72.6/72.3 prose, 72.6/73.3 JSON.
- Its Flint engine has the verification scaffolding we also have
  (`AppendAll`, `Logits(row)`, `Rewind`, `kMaxVerifyRows = 5`), so the
  incoai drafter is new ground for every Gufo lineage.
- Its prompt-lookup loop lives in the CLI driver, not in serve — the
  reference to read when implementing ours.

## Effort shape (revised 2026-11-14, after Step 0)

Phase 0: done (validation only). Phase 1: **done** — 1.31x mean, 1.54x best,
verify ratio 2.3-2.7x on the oracle. Step 0: **done** — on our own kernels the
verify ratio is 2.55x at depth 0 and 5.13x at 131k, so the DFlash2 track stops
here; no drafter module was written. What would still move the number is a
multi-row attention kernel (share each KV read across verify rows), which is
worth doing on its own merits because it gates every form of speculation,
including prompt lookup.

Reference value kept from the study: `src/models/dflash.cpp` +
`dflash2.cpp` + `common/speculative.cpp` in llama.cpp are a complete, working
implementation of exactly this drafter against a Flint target, including the
`target_layers` contract and the host-side selector walk. Reviving the track
later does not start from a blank page, and `FLINT_BENCH_VERIFY=1` re-measures
the gate in one run.
