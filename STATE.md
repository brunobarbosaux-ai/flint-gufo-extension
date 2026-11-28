# State

## Current workstream

**0.5.0 released, awaiting registry review.** Tagged `v0.5.0` at `dcccec1`,
GitHub Release published, registry PR gufo-org/gufo-extension#41 open — a Gufo
maintainer's review is the only open step. Scope as shipped: ring KV (#5,
`payload_layout flint-kv-rows-window-v1`), `drafter_parity` `Reset()` + odd
`prefill_tile` rejection, staged weight copy (#11), matrix attention (#7) inert
(option OFF). The 8 GiB retained-snapshot cap shipped unchanged and no longer
binds at 130k. Evidence: `docs/QUALITY.md` (0.5.0); checklist:
`PROGRESS.md`.
Next, if picked up: matrix attention's `DEFERRED.md` prerequisites (M1/M2,
L1-L4) — a separate release, not a hotfix. The other open threads (600-666 MiB
VRAM undercount, MMQ arena release, 262144-token request, mid-prompt checkpoints)
are listed in `PROGRESS.md` § "After 0.5.0".

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
- **Matrix-core decode attention (#7) merged 2026-11-26 as `d64cc43`** (merge commit,
  sebt3's six commits keep their authorship; head `fbdadd0`). Matrix attention ships
  inert: the default build behaves like `main`. Cross-restore is blocked in both
  directions by the payload layout suffix `-matrix`.
- Follow-ups on `main` after the #7 merge (all pushed, `d64cc43`..`6cb866d`):
  - `258839f`: `FLINT_MATRIX_ATTENTION` from the environment overrides a disagreeing
    cached value (verified by configure, five cases).
  - `f8ac4e8`: eight-row (all-live wave) case in both gate legs.
  - `a8ad54d`: narrows two doc claims.
  - `903a443`: records the flip prerequisites in `DEFERRED.md`.
  - `20b33be`: the decode bench fails on a tolerance or invariance breach.
  - `6cb866d`: records the #7 battery and the staged-copy A/B.

Released:
- 0.4.0 (tool calls + per-layer row cache + #6 + #8 + #9) — validated on Gufo
  v0.36.0 (API 0.3.42), tagged `v0.4.0` at `cf9f7aa`; registry PR
  gufo-org/gufo-extension#27 **merged and published** (digest `sha256:3b8e9e86…`,
  entry visible in `index.json`).
  GitHub Releases: `v0.4.0` (latest), plus `v0.3.0` and `v0.3.1` back-filled
  from their registry revisions `b262cb6a` and `cb7303d5`.
- 0.4.1 is reserved for hotfixes only.
- **0.5.0 released.** Ring + staged weight copy; matrix attention ships with the
  option OFF, as merged in `d64cc43`; flipping the default is a later release
  gated on the `DEFERRED.md` prerequisites. Tag `v0.5.0` at `dcccec1`, digest
  `sha256:8508f149…` from a clean clone at the tag, GitHub Release
  `v0.5.0 — Ring KV window and steadier loads`, registry PR
  gufo-org/gufo-extension#41 open for review. The ring renames `payload_layout`
  from `flint-kv-rows-v2` to `flint-kv-rows-window-v1` and keeps `state_abi` at
  `flint-rocm-session-v2`; the identity changes, so 0.4.0 disk caches are a
  measured miss (`cache_miss_reason=no_checkpoint`).
  - Release gate on the clean-clone build: 15/15 tests, gx 3/3 at 0/8k/32k/131k
    spec off+on, null 72/72, settings 8/8, reuse 24k RAM/spec/sparse/disk, 60k
    RAM/disk restart (0.4.0 was cap-refused), 130k RAM/disk restart (128,980
    tokens from disk in 1.4 s where 0.4.0 was a miss). Capacity at 262144:
    4 sessions 9.5 s / 60.0 GiB, 8 sessions 14.0 s / 94.9 GiB. Greedy output
    byte-identical to 0.4.0 on three prompts. Full table: `docs/QUALITY.md`
    (0.5.0).
- **Staged weight copy (#11) merged 2026-11-26 as `cbd1075`** (merge commit,
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
    interleaved reps, one session at 131072 — the script's `-s 2` is `--seed`,
    every log reports `sessions=1`; sessions are `-j`): healthy node `main`
    8.0/8.0/8.5 s, staged one-thread 9.5/9.5/10.0 s, staged four-thread
    8.5/8.5/9.0 s; under pressure (`posix_fadvise(DONTNEED)` plus a 44 GiB
    anonymous hog) 132.6/132.6/134.6, 138.1/138.1/138.6 and 64.0/64.5/64.5 s.
    The four-thread head is 2.05× the direct mapping under pressure and within
    0.5 s healthy (the weight upload happens once, so the result is
    session-independent); the one-thread commit alone is slower than `main` on
    both, so a bisect that lands on `c8e059b` reads as a regression (historical
    only).
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
  - `main` build: `.flint-work/gufo036/build/host-main-fix` — rebuilt per change.
    The release binaries are `.flint-work/gufo036/build/rel050` (0.5.0, built
    with `-DGUFO_EXTENSION_DIRS` pointing at the clean clone
    `.flint-work/rel050`) and `.flint-work/gufo036/build/rel040` (0.4.0
    `cf9f7aa`, kept for the cross-version greedy and cache checks).
  - Battery: `.flint-work/battery-rel050.sh` (0.5.0 release legs, logs in
    `.flint-work/battery-rel050/`); the per-PR scripts (`battery-pr5d.sh`,
    `battery-pr7.sh`, `battery-pr11m.sh`) are the historical runs
  - **Cleanup debt:** delete `.flint-work` once registry PR #41 is merged and no
    branch is under verification. Until then the two release build trees
    (`gufo036/build/rel040` and `gufo036/build/rel050`, ~20 GB) stay as
    reproduction trees for the review; everything else there is disposable.
- Target model: `/home/bruno/AI/GGUF/flint-UD-Q4_K_XL.gguf`
- Drafters: `/home/bruno/AI/GGUF/mtp/Flint-DFlash2-{BF16,Q6_K}.gguf`
- Oracle (disposable, `/tmp` is fine): `/tmp/opencode/llamacpp-oracle/llama-b11804`

## Open questions

- 262144 capacity is answered for 0.5.0: with the ring a session reserves about
  8.7 GiB, so on this box four sessions load in 9.5 s (60.0 GiB) and eight fit
  (94.9 GiB, 22.6 GiB host left). The old "advertise two" was a count against
  the full-row layout. What is not answered is **load latency under pressure**
  (one outlier at 7 sessions, 41 s) and **a 262144-token request**, which has
  never been run through serve. Never quote a multi-session load time without a
  warmed page cache.
- gufo#1319 (O(n²) `ReserveBacking` fix) **MERGED** 2026-11-21 (`93a1d230`) and
  **released in Gufo 0.36.0**. The cap-removed 130k A/B is no longer needed for
  0.5.0: on the merged ring tree a 130k frontier retains `3,995,190,656 B` and
  restores all 128,981 tokens in 50.9 ms, so the 8 GiB cap does not bind there.
  Raising it stays a separate, measured change (PLAN-KV-ROWS §7 Q1 closed as
  "unchanged in 0.5.0").
- gufo#1317 is **closed** (2026-11-21) after the maintainer's root cause and our
  before/after numbers were posted. Remaining upstream hygiene (capture off the
  request thread, clean refusal, non-zero default `stall_timeout`, bounded
  waits) lives in Gufo core and is listed in `DEFERRED.md`.
- Ring (#5) merged as `c47ce23`. Measured on the merged tree: at 128,981 tokens
  the retained snapshot is 3,995,190,656 B and both the repeat and the
  follow-up restore **100%** in ~52 ms, so the 8 GiB cap does not bind at 130k.
  The cap raise is dropped from 0.5.0 on that evidence (see `DEFERRED.md`).
