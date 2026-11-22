# State

## Current workstream

**Continuation cache → per-layer `kAppendRows` components** (`docs/PLAN-KV-ROWS.md`).
Stage 2 (adapter rewrite) COMPLETE (`c559211`, review fixes `e1d0f6f`).
Stage 3 (disk cache + eviction stress) COMPLETE (`b2893cc`, `4cc90dc`).
Stage 4 (position assertion + 130k A/B) COMPLETE (`9ceaa4d`). Cap kept on 0.3.x because
#1319 is unreleased; raise when the minimum supported Gufo includes it. Stage 5 next.

Shipped and idle:
- 0.3.1 (retained-snapshot cap) — registry PR gufo-org/gufo-extension#24 open, CI PASS.
- 0.4.0 (tool calls + perf) — waits for sebt3's PR #3 rebase.

## Decisions in force

- Follow `docs/PLAN-KV-ROWS.md` stage order; never skip a gate.
- Never loosen gates (1e-4 stage, 5e-6 selector, 95% Level 2).
- Never modify `/home/bruno/AI/Gufo-api03` (reference tree) or Gufo/Strycast repos from
  this workstream; extension work only in this repo.
- One GPU model process at a time: `pkill -x flint_spec_gate; pkill -x gufo` first;
  never SIGKILL a GPU process mid-load.
- `kAppendRows` does **not** reduce first-capture bytes; the 8 GiB cap stays until
  measured otherwise (PLAN-KV-ROWS §7 Q1).
- Speculation is opt-in via `--speculative`/`--dflash-model`; no env vars on user paths.
- `flint` stays its own registry name (`strycast_` means CI provenance).
- Preserve contributor authorship (merge/rebase, never squash).

## Environment

- Extension repo: `/home/bruno/Projects/flint-gufo-extension`
- Gufo 0.3 reference tree: `/home/bruno/AI/Gufo-api03` (API 0.3.27)
- Build dir: `/home/bruno/AI/Gufo-api03/build/host-test` (targets `gufo`, `flint_spec_gate`)
- Target model: `/home/bruno/AI/GGUF/flint-UD-Q4_K_XL.gguf`
- Drafters: `/home/bruno/AI/GGUF/mtp/Flint-DFlash2-{BF16,Q6_K}.gguf`
- Oracle (disposable): `/tmp/opencode/llamacpp-oracle/llama-b11804`

## Open questions

- gufo#1319 (O(n²) `ReserveBacking` fix) **MERGED** 2026-11-21 (`93a1d230`), not in a
  release yet (v0.35.0 predates it). The cap-removed 130k A/B is now runnable: build a
  **disposable clone** of Gufo at our 0.3.27 reference plus `93a1d230` under `/tmp/opencode`
  (never modify `/home/bruno/AI/Gufo-api03`), point `GUFO_EXTENSION_DIRS` at this repo,
  and measure with the 8 GiB cap temporarily removed. Result decides PLAN-KV-ROWS §7 Q1.
- gufo#1317: keep open until Stage 5 numbers are posted (before/after capture size,
  restore time, follow-up-turn cost).
