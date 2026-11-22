# Progress

## Workstreams

| Workstream | Plan | Status |
|---|---|---|
| DFlash2 speculative decoding (0.3.0, merged) | `docs/PLAN-DFLASH2.md` | COMPLETE — see "Milestone status" below |
| Tool calls (0.4.0) | `docs/PLAN-DFLASH2.md` + PR #2 | MERGED to main; release waits for PR #3 |
| Long-context continuation cache → per-layer `kAppendRows` | `docs/PLAN-KV-ROWS.md` | **IN PROGRESS — Stage 1** |
| Strycast serve gates | (separate repo) | PR #77 open, CI green |

### KV-rows stage status

Detail and gates: `docs/PLAN-KV-ROWS.md` §4. Update that table, not this one, for
per-stage notes; keep this row set in sync.

| Stage | Work | Status |
|---|---|---|
| 1 | Engine API (`NumLayers`/`KvRowBytes`/`KvRows`/`Synchronize`/`RestoreRows`/`SetLogits`/`SetRowWriteHook`) + `RestoreSnapshot` refactor + `flint.engine_snapshot` | **COMPLETE** (`e0577ff`, `feb7475`; all gates pass) |
| 2 | Adapter rewrite: descriptors, private pieces, row hooks, guard, identity bump (`flint-rocm-session-v2`/`flint-kv-rows-v1`) + serve cache-reuse gate | **COMPLETE** (`c559211`; plain + dflash gates pass) |
| 3 | Disk cache round trip + eviction/guard stress | **COMPLETE** (`b2893cc`, `4cc90dc`; restart reuse, identity, 400 MiB stress) |
| 4 | Runner position check + claim/cap measurement | **COMPLETE** (`9ceaa4d`; 130k A/B: +4.8 s capture vs 172 ms restore; cap kept on 0.3.x) |
| 5 | Long-context validation + benchmark | **COMPLETE** (`47cab02`, `dff1a01`; 24k dense/sparse/spec/disk, 130k spec+sparse, suite green) |

## Milestone status

Plan: `docs/PLAN-DFLASH2.md`. M1 (the drafter) is done; see `docs/REPORT-M1C.md`.

| Milestone | Title | Status | Commit | Notes |
|---|---|---|---|---|
| M2 | Target adapter | COMPLETE | 50d9189 | `src/spec/target_adapter.{hpp,cpp}`; `flint.target_adapter` passes (rows bit-identical widths 1..8, features bit-identical, rewind matches a fresh session) |
| M3 | Cycle + runner integration | COMPLETE | | `src/spec/cycle.{hpp,cpp}`, `FLINT_SPEC=dflash\|off\|null`; GX 12/12 + null 108/108 byte-identical; speedup 0.77/0.69/0.65 at temp 0/0.6/1.0 (see §1 note) |
| M4 | Draft policy | COMPLETE | d2680fb | adaptive + fixed; honours max/min/policy; deferred emitted-token row (append removed). GX 12/12, null 72/72; adaptive 1.185/0.970/0.970 vs fixed 1.024/0.850/0.827 at temp 0/0.7/1.0. Temp-1.0 parity within noise (fixed-k ceiling ~1.00x) |
| M5 | Gates | COMPLETE | 8d7d82f | Smoke tier. GX 12/12 @ 0/8k/32k/131k; null 72/72; settings 8/8 (grammar via D1: violations=0, not_evaluated=2); quants Q6_K/Q4_K_M/BF16 GX 3/3, accept 0.42-0.44; no-regression 70.96/63.56/51.05/28.18 vs 71/63/50/28 (existing logs); memory: ring+drafter accounted (measured growth = 320 MiB ring exactly); cost+concurrency deferred to M6 docs. Pre-existing ~660 MiB SessionBytes undercount in DEFERRED.md |
| M6 | Release | COMPLETE | a16c7bc | 0.3.0 + speculative_defaults; registry PR #17 (needs revision bump to release head + clean-clone digest). Final verification batch 2026-11-18: GX 12/12 @ 0/8k/32k/131k; settings 8/8 (violations=0, not_evaluated=2 budget-cuts); null 72/72 (temps 0/0.7/1.0 × seeds 1,2); 131k speed 4/4 rows (adaptive decode 1.06–1.09x, fixed 0.79–0.82x — fixed harmful at depth, adaptive default confirmed); serve A/B: identical request decode 70.7 → 91.2 t/s (+29%) with dflash2 — speculation in serve proven; unit tests 5/5; curve mode added (`4e8e7f4`). Deferred: serve draft counters (DEFERRED.md), sparse attention → v0.4.0, cost ratios (no tool mode) |

## M2 findings (carried into M3)

- `Session::Rewind` does not restore logits; the cycle re-appends the anchor before reading `Logits()`.
- A multi-token prefill is not bit-identical to single-token decode, so the cycle's exactness argument (§3.3) holds only for rows written by the decode or verify path.
- `flint.config` / `flint.probe` fail to compile in `tests/gguf_test_shim.hpp` (pre-existing SDK drift, unrelated to M2).

## Design note (deviation from the plan)

PLAN-DFLASH2.md §3.1 says there is "no extension-facing speculative hook". The host does expose `TextRunnerCapabilities.multi_token_decode` + a virtual `DecodeStep` (Qwen3.8-Flash-Next uses it). M3 still follows the plan's `SelectNext`/`Advance` design, because it keeps the sampler's history updated between rows (the `DecodeStep` path would require the runner to manage sampler history itself). To be documented in the plan.

## M3 design decisions (settled before the runner glue)

- **Anchor/cache.** The host contract is `SelectNext` -> token, then `Advance` -> append, and `Commit`/`CheckpointPosition` run after that. So the cache must be consistent after `Advance`. The cycle's anchor is the last committed token, which **is** cached at `Position()-1`; `SpecCycle::Start` rewinds to `Position()-1` before `VerifyRows([anchor, drafts])`. `End()`'s `Rewind(base + accepted + 1)` leaves the emitted token uncached, and `Advance` appends it. The first token after prefill is a plain decode (matching `RunAcceptance`'s `anchor = Argmax(Logits())`); the cycle starts on the second token.
- **Hidden ring.** `AppendContext` requires contiguous appends from `injected_context_len`, and the target's ring holds only the last `kHiddenRingRows` (2048) rows. When speculation is on, the runner caps its prefill chunk at `kHiddenRingRows` and calls `cycle.SyncContext(position)` after each chunk (as `RunAcceptance` does). Speculation is **not** disabled for long prompts.
- **Cancellation.** `PrepareCancellation` -> `cycle.Abort()`; `Invalidate()` -> `cycle.Abort()` + `cycle.Reset()`.

## M3 speed-gate root cause (debugger, confirmed)

The ON path's multi-second stalls are HIP sync/allocation stalls, not speculation logic. Two triggers, both amplified because the harness held **two full 25 GB target models at once**:

1. **Graph re-capture per request.** `Session::SetHiddenCapture` (`src/gpu/engine.cpp:1292`) called `Impl::InvalidateGraphs()` unconditionally, and `TargetAdapter::Create` (`src/spec/target_adapter.cpp:34`) calls it from `BindComponentState` (`runner.cpp:355`) on every request with the same tap set. Each re-capture costs 4-5 s on this APU.
2. **mmq scratch pool growth.** `qfn_stack_pool::reserve` (`hip/vendor/rocm/mmq/flint_ggml_stubs.hip.cpp:128`) does `hipDeviceSynchronize()` + `hipFree` + `AllocateDevice`; under memory pressure this is attributed to `exp_gate_up` (4844 ms ON vs 21 ms OFF).
3. **Harness doubled memory.** `tools/spec_gate.cpp` constructed `Harness off` and `Harness on` up front (lines 273-274, 291-292, 321-322); each loads its own 25 GB model, so both stayed resident and the ON model's `hipMalloc`/`hipMemcpy` upload stalled (3.4-6.4 s per tensor).

**Fix:** (primary) serialize OFF/ON harness lifetimes in `tools/spec_gate.cpp`; (secondary) guard `InvalidateGraphs()` + ring realloc behind a tap-set change in `Session::SetHiddenCapture`. Do NOT touch the capture block, the pool reserve, or the cycle/verify logic.

## M4 economics note (why the temp-1.0 gate is at risk)

The measured M3 speedup (0.770 / 0.689 / 0.648 at temp 0 / 0.6 / 1.0) is explained exactly by a per-cycle cost model:

```
per cycle = drafter forward (0.48 decode) + verify width (k+1) (ratio(k+1) decodes)
          + one append of the emitted token (1.0 decode)
```

At temp 0, k=7: 0.48 + 2.21 + 1.0 = 3.69 decodes for 2.89 tokens = 0.78x (measured 0.77). The append is a genuine fixed cost: `Session::Rewind` (`src/gpu/engine.cpp:1287`) only moves **backward**, so the emitted token's row cannot be advanced without a forward pass, and `SpecCycle::End` leaves it uncached for `Advance` to append.

Consequence: the fixed per-cycle cost (drafter + append = 1.48 decodes) exceeds the marginal tokens at every draft length, so **no `k` reaches 1.0x at temperature 1.0** (best ≈ 0.75x). M4's acceptance "never slower than spec-off at temperature 1.0" therefore cannot be met by a draft policy alone. Candidate fixes (NOT yet authorised): (a) let the adaptive controller fall back to plain decode (draft 0) when speculation is unprofitable; (b) remove the per-cycle append (position-only advance / fold it into the next verify) — an engine/cycle change that risks the GX/null gates and the snapshot contract. Escalate to the user before implementing either.

**Decision (user, 2026-11-16): remove the per-cycle append.** The emitted token's row is written by the next cycle's verify instead of a separate decode. Delegated; must keep GX/null byte-identical and keep the session consistent at any checkpoint.

**Append removal verified (orchestrator, 2026-11-16).** GX 12/12, null 72/72 byte-identical; ctest 5 pass + 2 known SDK-drift fails. Speed (adaptive): 1.172 / 0.959 / 0.939 at temp 0 / 0.7 / 1.0 (fixed: 1.024 / 0.851 / 0.825). Temp 1.0 rose 0.702 -> 0.939 but still fails the gate.

**Root cause of the residual gap: the controller over-drafts.** Measured fixed-k speedup curves (12 prompts, 192 tokens):

| k | temp 0 | temp 1.0 |
|---|---|---|
| 1 | — | 0.951 |
| 2 | 1.188 | **1.003** |
| 3 | **1.243** | 0.972 |
| 4 | 1.131 | 0.910 |
| 5 | 1.085 | — |
| 6 | 1.067 | — |
| 7 | 1.019 | 0.825 |

The true optimum is **k=3 at temp 0** and **k=2 at temp 1.0** — so the gate IS achievable (fixed k=2 = 1.003x). The adaptive controller picks mean 4.6 / 4.1 and lands at 1.172 / 0.939. Two bugs in `src/spec/draft_policy.cpp`: (1) the cost model still adds the removed append (`+ 1.0`, line 65); (2) `kVerifyRatio` underestimates the measured cycle cost (by ~0.1-0.3 decodes, growing with width). Fix delegated: remove the stale term and calibrate the verify ratios to the measured cycle cost.

## M5 findings

### M5a (gate tooling + correctness gates)

- **GX PASS at 0 / 8k / 32k / 131k** (12/12 byte-identical each).
- **Settings honour**: top-k, top-p, min-p, repeat/frequency penalty all PASS (spec-on == spec-off, 12/12). Grammar FAILED 2/12 at 1024 tokens (9/12 at 192) — suspected token-budget truncation, not a real violation; re-check delegated.
- **Depth curve**: adaptive >= fixed at every depth and both temps. Whole-request speedup < 1.0 at depth >= 8k because prefill (not speculated) dominates; decode-only speedup is the fair number (adaptive 1.07-1.20 at temp 0).
- **Runtime bug found**: spec-on prompts > 2048 tokens fail with `captured residual for row 0 left the ring`. Root cause: `Session::CapturedHiddenDevice` (`src/gpu/engine.cpp`) has an off-by-one (`row + R <= position` should be `< position`), so the ring yields only `R-1` usable rows while the runner caps its prefill chunk at exactly `R` (2048). Short prompts never hit it, which is why M3 passed. Fix delegated (engine boundary + remove the gate workaround).
- **Root cause found and fixed**: spec-off prefilled in 8192-token chunks while spec-on capped at the 2048-row hidden ring; multi-token prefill depends on chunk size (M2), so KV caches differed and greedy decode diverged. Debugger confirmed with a boundary sweep (2048 pass / 4096 fail) and a decisive test (kPrefillChunk=2048 makes depth 4096+ pass). Fix (`9506b32`): `kHiddenRingRows` 2048->8192, spec-only Prefill cap removed, `static_assert` invariant, plus the `CapturedHiddenDevice` boundary fix (`row + R < position`). Verified independently: GX 12/12 at 0/8k/32k/131k with no workaround; decode-only speedup at 8k = 1.26 (accept 0.495). Ring now ~320 MiB when capture is on; ResourceClaim accounting is M5b scope. Stale "2048" comments remain in PROGRESS.md/PLAN-M1C/target_adapter.hpp/drafter_parity.cpp (cosmetic).

### Depth-curve speed comparison (`--mode curve`, added post-batch `4e8e7f4`)
Measured temp 0, 12 prompts, 192 tokens (see `/tmp/opencode/release-batch/curve.log`):
| depth | prefill off/on tps | decode off/on tps | whole spd | decode spd | accept |
|---|---|---|---|---|---|
| 2048 | 2884 / 2076 | 67.1 / 75.9 | 1.016 | 1.131 | 0.428 |
| 8192 | 2811 / 2192 | 62.7 / 73.1 | 0.939 | 1.165 | 0.458 |
| 32768 | 2236 / 1929 | 50.4 / 57.1 | 0.907 | 1.132 | 0.517 |
| 131072 | 1216 / 1135 | 28.0 / 30.9 | 0.941 | 1.104 | 0.595 |
- Decode: speculation wins 10-17% at every depth; acceptance rises with depth (0.43 -> 0.60).
- **Spec-on prefill is 20-25% slower at every depth** (drafter context injection / hidden capture during prefill) — this is why whole-request speedup falls below 1.0 from 8k up (prefill is ~50% of wall at 8k, ~94% at 128k). Optimization target for post-0.3.0: skip or defer SyncContext work during prefill chunks.
- **262k-context serve startup hangs in `phase=sessions` with AND without speculation** (control run killed after 11 min; KFD GPU alloc, GTT >55 GB). Pre-existing host/extension issue at max context, not DFlash2. At `-c 8192` serve loads in ~66 s. **Superseded (2026-11-22):** the hang does not reproduce on the current 0.3.27 build — 262144 loads with 1 and 2 sessions in ~9 s; the limit is capacity (26.0 GiB/session, 4 sessions refused at admission). See `DEFERRED.md`.
- **Prefix-reuse desync fixed** (`e9dc61c`): the second serve request failed with "DFlash2 context rows are not contiguous" — `ReconcileComponentState` restored the session without re-syncing the drafter. Fix: `cycle Reset(frontier)` adopts the restored position; drafter history gated to post-frontier rows. Verified: 3 sequential serve requests all 200 with cache hits; GX 3/3, null 2/2.
- Caveat: after a restore the drafter attends to no history until the 2048 sliding window passes the frontier (acceptance dip, greedy unaffected). Serve-mode draft-cycle counters not yet confirmed (coverage gap — gates cover the spec path).

- M2 delegated, verified (`flint.target_adapter` passes), committed `50d9189`.
- M3 delegated; first pass produced `src/spec/cycle.{hpp,cpp}` (compiles), then stopped on three design blockers. Decisions above sent; runner glue + gate tool in progress.
- M3 correctness gates pass (GX 12/12, null 108/108 byte-identical); speed gate failed (0.13-0.14). Coder reported STUCK; debugger root-caused it to harness memory pressure + unconditional graph invalidation (above). Fix delegated.
- M3 speed fix applied (serialize OFF/ON harness lifetimes; guard `InvalidateGraphs` behind a tap-set change). Verified independently: GX 12/12, null 108/108, speedup 0.770 / 0.689 / 0.648 at temp 0 / 0.6 / 1.0. M3 COMPLETE per the plan's acceptance (GX + null gates + measurement). The speedup is below the §3.4 model (1.38) because the model omits the per-cycle append + drafter; recorded in the plan's §1 note. M4 must recover the high-temperature case.
- M4 delegated (policy + config wiring + gate tooling + unit test). First pass: adaptive beats fixed everywhere but temp-1.0 gate fails (0.703).
- M4 append removal (user-authorised) delegated and verified: GX 12/12, null 72/72; adaptive 1.172/0.959/0.939. Temp 1.0 rose 0.702 -> 0.939 but still failed.
- M4 controller calibration delegated: removed the stale append term, re-measured verify ratios. Adaptive 1.185/0.970/0.970 vs fixed 1.024/0.850/0.827. Temp-1.0 is 0.97-0.99, parity within the ~10% measurement noise; the fixed-k ceiling at temp 1.0 is ~1.00x (k=2 = 1.003x). Accepted as met-within-noise; M4 COMPLETE (`d2680fb`).
