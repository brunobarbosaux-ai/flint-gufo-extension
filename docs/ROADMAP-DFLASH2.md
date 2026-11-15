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

## Phase 2 — Drafter loading and forward in the extension  **[on hold — gated on the verify bench]**

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

## Phase 3 — Draft/verify loop  **[on hold; scaffolding exists]**

**Updated 2026-11-14:** the multi-row verification scaffolding already
exists in the extension — `Session::AppendAll`, `Logits(row)`, `Rewind`,
`kMaxVerifyRows = 5` — shipped with the fork lineage. llama.cpp clarifies the
loop's shape:

- Draft length is capped at `block_size - 1 = 7` (8 with
  `sample_from_anchor`), not 8 — the selector only scores positions 1..7.
- Verification is a single batched forward of `[anchor + drafts]` followed by a
  prefix-acceptance walk on logits, `Rewind` to the accepted prefix; per-token
  acceptance and `mean len` are the two numbers to log.
- `p_min` truncation (0.5 was the best point here) trades mean length for
  verify width: it lifted JSON 1.32x -> 1.54x and prose 1.03x -> 1.07x while
  costing code 1.35x -> 1.30x.
- [ ] Verify rows beyond 5 if the 8-token block is kept whole (raise
      `kMaxVerifyRows` or cap draft length at 5).
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

## Effort shape (revised 2026-11-14, after the study)

Phase 0: done (validation only). Phase 1: **done** — and it decided the
project: 1.31x mean, 1.54x best, verify ratio 2.3-2.7x. Phase 2/3 are on hold
behind one cheap bench (8-row `AppendAll` cost on our own kernels). If that
bench lands at ~2.6x, the track closes with the study as the deliverable and
nothing was sunk into the drafter module. Phase 4: a day.

Reference value kept from the study: `src/models/dflash.cpp` +
`dflash2.cpp` + `common/speculative.cpp` in llama.cpp are a complete, working
implementation of exactly this drafter against a Flint target, including the
`target_layers` contract and the host-side selector walk. Reviving the track
later does not start from a blank page.
