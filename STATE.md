# State

## Current workstream

**Continuation cache → per-layer `kAppendRows` components** (`docs/PLAN-KV-ROWS.md`).
Stage 2 (adapter rewrite) COMPLETE (`c559211`, review fixes `e1d0f6f`).
Stage 3 (disk cache + eviction stress) COMPLETE (`b2893cc`, `4cc90dc`).
Stage 4 (position assertion + 130k A/B) COMPLETE (`9ceaa4d`).
Stage 5 (long-context validation) COMPLETE. All five stages done (`47cab02`, `dff1a01`
for the gate; `9ceaa4d` for the runner check).
Open follow-ups: post the #1317 before/after comment; raise the 8 GiB cap now that
Gufo 0.36.0 contains #1319 (`93a1d230`) — scheduled for 0.5.0 with the ring's
`flint-kv-rows-window-v1` payload layout;
full battery at the 0.4.0 release gate.

Shipped and idle:
- 0.3.1 (retained-snapshot cap) — **merged** into gufo-org/gufo-extension
  (`ea39c22`, 2026-11-22); registry entry `extensions/flint.json` pins
  `cb7303d5`, digest `sha256:a112bc6e…`; `registry.py verify flint 0.3.1` passes;
  post-merge `Registry checks` and `Website catalogue` both green.

Merged for 0.5.0:
- **Ring KV (#5) merged 2026-11-26 as `c47ce23`** (merge commit, sebt3's ten
  commits keep their authorship). Head verified before merge (`968b720`): own
  battery green — `ring_snapshot` with the `rewind ok` leg, `prefill_split`
  14/14 (42.4 s), `tile4136` (41.4 s), `engine_snapshot` (28.1 s), gx 3/3 at
  depth 0 and 32k, 24k disk gate `r1 959,071,040 B` restore 272 ms, 130k
  `r1 3,995,190,656 B` restoring fully in ~52 ms.
- Post-merge fixes landing on `main`: `tools/drafter_parity.cpp` `Rewind(0)` to
  `Reset()` (the ring makes a deep rewind illegal), and `prefill_tile` parity
  validation in `Model::Load` (odd tiles were silently wrong on `main`).
- **Matrix-core decode attention (#7) merged 2026-11-26 as `c9ca909`** (merge commit,
  sebt3's six commits keep their authorship; head `fbdadd0`). Matrix attention ships
  inert: the default build behaves like `main`. Cross-restore is blocked in both
  directions by the payload layout suffix `-matrix`.
- Follow-ups on `main`, not yet pushed:
  - `370855a`: `FLINT_MATRIX_ATTENTION` from the environment overrides a disagreeing
    cached value (verified by configure, five cases).
  - `bb2557e`: eight-row (all-live wave) case in both gate legs.
  - `2ed4c8b`: narrows two doc claims.
  - `91cdf29`: records the flip prerequisites in `DEFERRED.md`.

Released, awaiting registry review:
- 0.4.0 (tool calls + per-layer row cache + #6 + #8 + #9) — validated on Gufo
  v0.36.0 (API 0.3.42), tagged `v0.4.0` at `cf9f7aa`; registry PR
  gufo-org/gufo-extension#27 open for review (digest `sha256:3b8e9e86…`).
  GitHub Releases: `v0.4.0` (latest), plus `v0.3.0` and `v0.3.1` back-filled
  from their registry revisions `b262cb6a` and `cb7303d5`.
- 0.4.1 is reserved for hotfixes only.
- 0.5.0 (ring + staged weight copy; matrix attention ships with the option OFF, as merged in
  `c9ca909`; flipping the default is a later release gated on the `DEFERRED.md`
  prerequisites). The ring
  renames `payload_layout` from `flint-kv-rows-v2` to `flint-kv-rows-window-v1`
  and keeps `state_abi` at `flint-rocm-session-v2`; the identity string changes,
  so 0.4.0 disk caches stop restoring and are re-prefilled.
- **Staged weight copy (#11) merged 2026-11-27 as `cbd1075`** (merge commit,
  sebt3's two commits keep authorship; head `fe46d2b`). Weights load through a
  pinned staging buffer with a four-thread copy. Verified on the merge tree
  (`main` `6cb866d` + `fe46d2b`, tree identical to what shipped): `flint.ops`
  2.05 s, `flint.target_adapter` 90.1 s, gx 3/3 byte-identical at depth 0 and
  32k, memory gate PASS both modes (spec-off claimed 2,143,724,468 / measured
  2,074,083,328 B; spec-on 2,563,154,868 / 2,493,513,728 B), 24k disk gate
  PASSED on `flint-rocm-session-v2:flint-kv-rows-window-v1`, 130k
  `r1 3,995,190,656 B` restoring in 50.9 ms. Greedy output byte-identical to
  `main` on two prompts (48 and 160 tokens).
  - A/B 2026-11-26 (`.flint-work/pr11-pressure-ab.py`, three arms, three
    interleaved reps, sessions 2 × 131072): healthy node `main` 8.0/8.0/8.5 s,
    staged one-thread 9.5/9.5/10.0 s, staged four-thread 8.5/8.5/9.0 s; under
    pressure (`posix_fadvise(DONTNEED)` plus a 44 GiB anonymous hog)
    132.6/132.6/134.6, 138.1/138.1/138.6 and 64.0/64.5/64.5 s. The four-thread
    head is 2.05× the direct mapping under pressure and within 0.5 s healthy;
    the one-thread commit alone is slower than `main` on both, so a bisect that
    lands on `c8e059b` reads as a regression (historical only).
  - Caveat: `workingset_refault_file` stayed at 0 in every run, so the effect is
    measured but not attributed to file refaults from that instrumentation. The
    pressure condition itself is unambiguous (8 s healthy against 133 s under it).

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
- Work root (**persistent**; this box reboots often and `/tmp` is wiped):
  `/home/bruno/Projects/.flint-work`
  - Gufo tree: `.flint-work/gufo036` (v0.36.0, tag `4a7d8c6`, API 0.3.42);
    `/home/bruno/AI/Gufo-api03` is the older 0.35.0 tree and stays untouched
  - Build dirs: `.flint-work/gufo036/build/host-{036,pr5d,...}` (targets `gufo`,
    `flint_spec_gate`, the `flint_*_test` binaries)
  - Extension worktrees: `.flint-work/wt-pr5d`, `wt-main-fix`, `wt-pr7`, ...
  - `main` build: `.flint-work/gufo036/build/host-main-fix` — rebuilt per change;
    it predates the #11 merge, so rebuild before the 0.5.0 gate
  - Battery: `.flint-work/battery-pr5d.sh` (writes logs to `.flint-work/battery-<dir>/`)
  - **Cleanup debt:** delete `.flint-work` once 0.5.0 ships and no branch is under
    verification; it holds a full clone plus build trees (~30 GB).
- Target model: `/home/bruno/AI/GGUF/flint-UD-Q4_K_XL.gguf`
- Drafters: `/home/bruno/AI/GGUF/mtp/Flint-DFlash2-{BF16,Q6_K}.gguf`
- Oracle (disposable, `/tmp` is fine): `/tmp/opencode/llamacpp-oracle/llama-b11804`

## Open questions

- gufo#1319 (O(n²) `ReserveBacking` fix) **MERGED** 2026-11-21 (`93a1d230`) and
  **released in Gufo 0.36.0**. The cap-removed 130k A/B is runnable on the 0.36
  tree (`.flint-work/gufo036`, never modify `/home/bruno/AI/Gufo-api03`), point
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
- Ring (#5) rebased onto `main` at `3630602` (`verify/pr5-rebased`, scratch): at
  128,981 tokens the retained snapshot is 3,995,190,656 B and both the repeat and
  the follow-up restore **100%** in ~52 ms, so the 8 GiB cap no longer binds at
  130k. Re-measure on the merged tree before planning the cap raise; with the ring
  it may be unnecessary.
