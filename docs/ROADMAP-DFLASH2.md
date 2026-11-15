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
- Drafter candidate: `incoai/Flint-DFlash2-GGUF` (presumably Flint-arch — our
  loader already reads Flint tensors).
- Standing lessons: per-quant validation is mandatory (Q6_K_XL loads, benches
  fine, and generates garbage — issue open); speed claims only on the
  runtime-image toolchain (Nix build ~3x slower, diverges); acceptance is
  sharply content-dependent (Laguna: 79-81% code/JSON vs 35% prose; 27B ~61%).

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

## Phase 1 — Acceptance-rate study (go/no-go decision gate)

- [ ] Fetch `incoai/Flint-DFlash2-GGUF`; inspect architecture (Flint-arch?
      head size? vocab coverage? quant?).
- [ ] Build a throwaway acceptance harness: load drafter alongside the
      target; drafts can initially be scored through the existing single-row
      path (correctness of the harness matters, not its speed).
- [ ] Measure acceptance on REAL workloads (own chat/code/JSON traffic), not
      filler prompts.
- [ ] DECISION POINT: projected speedup =
      `acceptance x draft_length / (1 + verify_cost_ratio)`.
      - If realistic acceptance cannot beat ~1.5x on our text: stop, close
        issue #1 with the data, keep Phase 0 wins.
      - If >= 2x looks reachable: proceed to Phase 2.

## Phase 2 — Drafter loading in the extension

- [ ] Second model instance: drafter is a small Flint-arch GGUF; reuse the
      existing loader, config parser and tokenizer.
- [ ] Drafter forward on our own kernels (mmq MoE path covers it).
- [ ] Wire the SDK draft-backend interface on the provider
      (`draft_backend.hpp`, `speculative_verifier.hpp`).
- [ ] Gate: drafter-only agreement check against the drafter's own reference
      (per-quant paranoia; Q6 lesson).

## Phase 3 — Multi-row verification on the Flint engine

The real engineering. Decode is currently 1 row; DFlash2 verifies up to 7
draft tokens per step.

- [ ] Batched decode rows: attention with per-row positions, routed experts
      with M>1 (mmq MoE kernels already take M — prefill proves it), batched
      LM head.
- [ ] Verifier contract: prefix acceptance against target logits, adaptive
      draft length, optionally the fork's >=0.7 running-confidence cut for
      greedy (+8-9% on prose/math with identical output).
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

- [ ] `RecursiveSI/flint-int4-gguf`: check the format against kernel
      coverage first (kernel port vs validation run).
- [ ] If supported: full per-quant gate battery (GX + G4 + coherence smoke —
      Q6 lesson), then a second llama-swap entry.
- [ ] Compounds with DFlash2: smaller target weights -> cheaper verify rows.

## Parallel track — rec cross-check

- [ ] Read `recursive-oss/rec` (Apple engine) for kernel schedules,
      expert-dispatch ordering, KV layout ideas. Low cost, reference reading.

## Explicit non-goals

- NPU / XDNA2 anything (fabric-clock hold requirements, machine-hang risk,
  out of extension scope).
- MTP (Flint has no MTP head; prompt lookup is the only drafter-less option).
- Cross-binary bit-parity (declared out of scope in QUALITY.md).

## Effort shape

Phase 0: ~a day each. Phase 1: 1-2 days and it decides the project.
Phases 2-3: the bulk (~a week of careful work with gates). Phase 4: a day.
If Phase 1's number is bad, total sunk cost is two days and Phase 0's wins
are kept regardless.
