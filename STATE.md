# State

## Current workstream

**Continuation cache → per-layer `kAppendRows` components** (`docs/PLAN-KV-ROWS.md`).
Stage 2 (adapter rewrite) COMPLETE (`c559211`, review fixes `e1d0f6f`).
Stage 3 (disk cache + eviction stress) COMPLETE (`b2893cc`, `4cc90dc`).
Stage 4 (position assertion + 130k A/B) COMPLETE (`9ceaa4d`).
Stage 5 (long-context validation) COMPLETE. All five stages done (`47cab02`, `dff1a01`
for the gate; `9ceaa4d` for the runner check).
Open follow-ups: post the #1317 before/after comment; raise the 8 GiB cap now that
Gufo 0.36.0 contains #1319 (`93a1d230`) — scheduled for 0.5.0 with the ring's v3 layout;
full battery at the 0.4.0 release gate.

Shipped and idle:
- 0.3.1 (retained-snapshot cap) — **merged** into gufo-org/gufo-extension
  (`ea39c22`, 2026-11-22); registry entry `extensions/flint.json` pins
  `cb7303d5`, digest `sha256:a112bc6e…`; `registry.py verify flint 0.3.1` passes;
  post-merge `Registry checks` and `Website catalogue` both green.

Release in progress:
- 0.4.0 (tool calls + per-layer row cache + #6 + #8 + #9) — validated on Gufo
  v0.36.0 (API 0.3.42). Docs and `extension.json` pinned; battery and registry PR
  pending.
- 0.4.1 is reserved for hotfixes only.
- 0.5.0 (ring + matrix attention) — waits for sebt3's rebases (#5, #7). The ring
  takes `payload_layout` v3, so 0.4.0 disk caches stop restoring at 0.5.0.

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
- Gufo reference tree: `/tmp/opencode/gufo036` (v0.36.0, tag `4a7d8c6`, API 0.3.42);
  `/home/bruno/AI/Gufo-api03` is the older 0.35.0 tree and stays untouched
- Build dir: `/tmp/opencode/gufo036/build/host-036` (targets `gufo`, `flint_spec_gate`)
- Target model: `/home/bruno/AI/GGUF/flint-UD-Q4_K_XL.gguf`
- Drafters: `/home/bruno/AI/GGUF/mtp/Flint-DFlash2-{BF16,Q6_K}.gguf`
- Oracle (disposable): `/tmp/opencode/llamacpp-oracle/llama-b11804`

## Open questions

- gufo#1319 (O(n²) `ReserveBacking` fix) **MERGED** 2026-11-21 (`93a1d230`) and
  **released in Gufo 0.36.0**. The cap-removed 130k A/B is runnable on the 0.36
  tree (`/tmp/opencode/gufo036`, never modify `/home/bruno/AI/Gufo-api03`), point
  `GUFO_EXTENSION_DIRS` at this repo, and measure with the 8 GiB cap temporarily
  removed. Result decides PLAN-KV-ROWS §7 Q1; scheduled for 0.5.0.
- 0.36.0 moved the 262144 admission budget to 87.3 GiB: three sessions are now
  admitted (the old build refused the third) but the third loads in ~112 s and
  leaves ~5.8 GiB host available. Revisit before advertising more than two.
- 0.36.0 262144 load time is not stable at two sessions (10-187 s over five
  runs), consistent with the released GGUF mapping becoming evictable. Not a
  0.4.0 blocker; re-measure with a warmed page cache before quoting a number.
- gufo#1317: keep open until Stage 5 numbers are posted (before/after capture size,
  restore time, follow-up-turn cost).
