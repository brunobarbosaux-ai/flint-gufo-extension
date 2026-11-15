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

## Phase 1 — Drafter study (go/no-go decision gate)  **[status: inspection done, measurement pending]**

The original plan assumed a standalone acceptance loop. Step 1 disproved that:
the drafter is hidden-state-conditioned, so acceptance cannot be measured
without the drafter forward and a target hidden-state export. Phase 1 now
shares its harness with Phase 2.

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
- [ ] Build the acceptance harness (requires Phase 2's drafter forward and
      the hidden-state export).
- [ ] Measure acceptance on REAL workloads (own chat/code/JSON traffic), not
      filler prompts.
- [ ] DECISION POINT: projected speedup =
      `acceptance x draft_length / (1 + verify_cost_ratio)`.
      - If realistic acceptance cannot beat ~1.5x on our text: stop, close
        issue #1 with the data, keep Phase 0 wins.
      - If >= 2x looks reachable: proceed to full integration.

## Phase 2 — Drafter loading and forward in the extension  **[current work]**

- [ ] **Target hidden-state export**: Flint's engine produces logits only;
      the drafter's `fc` needs the pre-`lm_head` 2048-float row
      (`Session::LastHidden()` or equivalent).
- [ ] Drafter loader: the drafter is `dflash` arch (81 tensors), not
      Flint-arch, so a small dedicated loader is needed (BF16/F32 only — no
      quantized kernels). Reuse Flint's tokenizer.
- [ ] Drafter forward on our own kernels: RMSNorm, BF16 matmuls, the 2-tap
      conv layers, non-causal block attention, factorized selector logits.
      Self-contained module; establish correctness against a CPU reference
      before GPU tuning.
- [ ] Wire the SDK draft-backend interface on the provider
      (`draft_backend.hpp`, `speculative_verifier.hpp`).
- [ ] Gate: drafter-only agreement check against the drafter's own reference
      (per-quant paranoia; Q6 lesson).

## Phase 3 — Draft/verify loop  **[current work, shares Phase 2]**

**Updated 2026-11-14:** the multi-row verification scaffolding already
exists in the extension — `Session::AppendAll`, `Logits(row)`, `Rewind`,
`kMaxVerifyRows = 5` — shipped with the fork lineage. The remaining work is
smaller than originally planned:

- [ ] Verify rows beyond 5 if the 8-token block is kept whole (fork's 27B
      drafts 7; either raise `kMaxVerifyRows` or cap draft length at 5).
- [ ] Implement the draft/verify loop: drafter drafts a block, target
      verifies each row via `AppendAll`, prefix acceptance with adaptive
      length, optional >=0.7 selector-confidence cut for greedy.
      Reference: the fork's prompt-lookup loop in its CLI driver.
- [x] Multi-row logits (`AppendAll` / `Logits(row)`) — present.
- [ ] Prompt-lookup drafting (moved here from Phase 0): the same
      append/verify/rewind machinery; the fork implements it in its CLI
      driver, not in serve.
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

## Effort shape (revised 2026-11-14)

Phase 0: done (validation only). Phase 1: inspection done; measurement
folds into Phase 2. Phase 2: the current bulk (hidden export + dflash
drafter module). Phase 3: smaller than planned — the multi-row verification
scaffolding already exists; the loop logic remains. Phase 4: a day.
If the acceptance number is bad, total sunk cost is the drafter module,
which is reusable for any future Flint drafter.
