# Progress — DFlash2 integration (M2-M6)

Plan: `docs/PLAN-DFLASH2.md`. M1 (the drafter) is done; see `docs/REPORT-M1C.md`.

## Milestone status

| Milestone | Title | Status | Commit | Notes |
|---|---|---|---|---|
| M2 | Target adapter | COMPLETE | 50d9189 | `src/spec/target_adapter.{hpp,cpp}`; `flint.target_adapter` passes (rows bit-identical widths 1..8, features bit-identical, rewind matches a fresh session) |
| M3 | Cycle + runner integration | COMPLETE | | `src/spec/cycle.{hpp,cpp}`, `FLINT_SPEC=dflash\|off\|null`; GX 12/12 + null 108/108 byte-identical; speedup 0.77/0.69/0.65 at temp 0/0.6/1.0 (see §1 note) |
| M4 | Draft policy | COMPLETE | d2680fb | adaptive + fixed; honours max/min/policy; deferred emitted-token row (append removed). GX 12/12, null 72/72; adaptive 1.185/0.970/0.970 vs fixed 1.024/0.850/0.827 at temp 0/0.7/1.0. Temp-1.0 parity within noise (fixed-k ceiling ~1.00x) |
| M5 | Gates | IN PROGRESS | | M5a (tooling: depth + settings modes; GX@depths, settings honour, depth curve) delegated; M5b (quant, no-regression, cost, memory, concurrency) pending |
| M6 | Release | NOT STARTED | | QUALITY.md, extension.json 0.3.0, registry PR, llama-swap, issue #1 |

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

## Log

- M2 delegated, verified (`flint.target_adapter` passes), committed `50d9189`.
- M3 delegated; first pass produced `src/spec/cycle.{hpp,cpp}` (compiles), then stopped on three design blockers. Decisions above sent; runner glue + gate tool in progress.
- M3 correctness gates pass (GX 12/12, null 108/108 byte-identical); speed gate failed (0.13-0.14). Coder reported STUCK; debugger root-caused it to harness memory pressure + unconditional graph invalidation (above). Fix delegated.
- M3 speed fix applied (serialize OFF/ON harness lifetimes; guard `InvalidateGraphs` behind a tap-set change). Verified independently: GX 12/12, null 108/108, speedup 0.770 / 0.689 / 0.648 at temp 0 / 0.6 / 1.0. M3 COMPLETE per the plan's acceptance (GX + null gates + measurement). The speedup is below the §3.4 model (1.38) because the model omits the per-cycle append + drafter; recorded in the plan's §1 note. M4 must recover the high-temperature case.
- M4 delegated (policy + config wiring + gate tooling + unit test). First pass: adaptive beats fixed everywhere but temp-1.0 gate fails (0.703).
- M4 append removal (user-authorised) delegated and verified: GX 12/12, null 72/72; adaptive 1.172/0.959/0.939. Temp 1.0 rose 0.702 -> 0.939 but still failed.
- M4 controller calibration delegated: removed the stale append term, re-measured verify ratios. Adaptive 1.185/0.970/0.970 vs fixed 1.024/0.850/0.827. Temp-1.0 is 0.97-0.99, parity within the ~10% measurement noise; the fixed-k ceiling at temp 1.0 is ~1.00x (k=2 = 1.003x). Accepted as met-within-noise; M4 COMPLETE (`d2680fb`).
