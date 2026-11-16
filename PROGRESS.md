# Progress — DFlash2 integration (M2-M6)

Plan: `docs/PLAN-DFLASH2.md`. M1 (the drafter) is done; see `docs/REPORT-M1C.md`.

## Milestone status

| Milestone | Title | Status | Commit | Notes |
|---|---|---|---|---|
| M2 | Target adapter | COMPLETE | 50d9189 | `src/spec/target_adapter.{hpp,cpp}`; `flint.target_adapter` passes (rows bit-identical widths 1..8, features bit-identical, rewind matches a fresh session) |
| M3 | Cycle + runner integration | COMPLETE | | `src/spec/cycle.{hpp,cpp}`, `FLINT_SPEC=dflash\|off\|null`; GX 12/12 + null 108/108 byte-identical; speedup 0.77/0.69/0.65 at temp 0/0.6/1.0 (see §1 note) |
| M4 | Draft policy | NOT STARTED | | adaptive length priced by measured per-row cost |
| M5 | Gates | NOT STARTED | | GX, null drafter, settings honour, quant, depth curve, no regression, cost, memory, concurrency |
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

## Log

- M2 delegated, verified (`flint.target_adapter` passes), committed `50d9189`.
- M3 delegated; first pass produced `src/spec/cycle.{hpp,cpp}` (compiles), then stopped on three design blockers. Decisions above sent; runner glue + gate tool in progress.
- M3 correctness gates pass (GX 12/12, null 108/108 byte-identical); speed gate failed (0.13-0.14). Coder reported STUCK; debugger root-caused it to harness memory pressure + unconditional graph invalidation (above). Fix delegated.
- M3 speed fix applied (serialize OFF/ON harness lifetimes; guard `InvalidateGraphs` behind a tap-set change). Verified independently: GX 12/12, null 108/108, speedup 0.770 / 0.689 / 0.648 at temp 0 / 0.6 / 1.0. M3 COMPLETE per the plan's acceptance (GX + null gates + measurement). The speedup is below the §3.4 model (1.38) because the model omits the per-cycle append + drafter; recorded in the plan's §1 note. M4 must recover the high-temperature case.
