# Deferred items

### Base-session VRAM undercount in `Model::SessionBytes` (pre-existing)
- `Session::Init` (`src/gpu/engine.cpp`) allocates per-session scratch pools, step-graph workspaces and allocator slack totalling ~600–666 MiB more than `SessionBytes` accounts for. Measured with `flint_spec_gate --mode memory` (second-session delta): spec-off claim 1457 MiB vs measured 2123 MiB; spec-on claim 1873 MiB vs measured 2443 MiB. Identical in both modes, so it predates DFlash2 and is not an M5 regression.
- The M5 memory gate criterion (hidden ring + drafter accounted in `ResourceClaim`) is met: the spec-on per-session claim grows 436 MiB over spec-off and the measured growth is exactly the 320 MiB hidden ring.
- Action before release: audit `Session::Init` allocations against `SessionBytes` and add the missing terms. Evidence: `/tmp/opencode/m5-smoke/memory2.log`.

### Serve startup hangs at 262144-token context (pre-existing, no speculation involved)
- `gufo serve -c 262144` never completes the `sessions` load phase (11+ min, killed), with AND without `--speculative`. Blocked in KFD GPU memory allocation; GTT climbs past 55 GB while the session cap is 2. Verified boundary (2026-11-17): 196608 loads (~62 s), 262144 hangs; boundary between them not bisected.
- Practical ceiling today: **196608 (192k)**. Documented in the README; the model's native 262144 remains unsupported at serve startup.
- Action: audit the host session pre-allocation path (queue budget / `ResourceClaim`-driven sizing) at 256k; likely fix is capping pre-allocated sessions by free VRAM.

### Serve draft counters report zero (root-caused 2026-11-18; not a logging bug)
- Serve drives Flint via `SelectNext()`/`Advance()` (per PLAN-DFLASH2: sampler-history correctness), so the counter-copying `DecodeStep` hook (`runner.cpp:612`) is never called; `Descriptor().multi_token_decode` is false and the host's `StepDecode` path never reads `SpecCycle::stats()`.
- Speculation itself demonstrably runs in serve: decode 74.3 t/s at 8k ≈ spec-on (74.7) vs spec-off (63.6). A same-request A/B tps comparison (serve with vs without `--speculative dflash2`) is the accepted proof for 0.3.0 — no code change.
- Fix options deferred to v0.4.0: (A) flip `multi_token_decode` capability — extension-only but changes the serve scheduling path (needs full re-verification); (B) host-side counter channel on `TextRunnerState` — touches Gufo core, out of extension scope.

### MMQ scratch arena never reclaimed (noted during `fb616ad` root-cause)
- `qfn_stack_pool::reserve` allocates a 256 MiB arena per (device, stream); each session owns a private stream, and `flint_mmq_release_stream` (`flint_mmq.h:10`) is declared but never called — arenas are never returned. The claim now models this honestly, but a long-lived server with rotating sessions could accumulate arenas. Action: call `flint_mmq_release_stream` on session teardown, or share one arena under the device lock.

### Long-context first-request stall (0.3.1 caps the claim; real fix = kAppendRows, gufo#1317)
- At `-c 130000` with the default RAM cache, the host captured a full private-state snapshot **synchronously on the request thread** (~15.7 GB) before answering: first request 14.2 s vs 0.47 s with the capture refused. Prefill itself was healthy; `--stall-timeout-ms` defaults to 0, so on slower machines the stall logs nothing and looks like a hang.
- Upstream root cause (Gufo maintainer, gufo-org/gufo#1317): `ResourceCharge::ReserveBacking` compared each backing block with every earlier one, so one 15.7 GB frontier (~240k blocks) made the first capture's allocation take ~20 s; fixed with a sort in gufo-org/gufo#1319 (12 ms). Our 0.3.1 cap is therefore a **mitigation, not the fix**.
- Flint-side mitigation (`0.3.1`): the retained-snapshot capacity/ceiling claim is capped at 8 GiB, so a snapshot larger than the cap is refused cheaply instead of captured. Verified: 130k first request 0.47 s; 8k keeps prefix retention (`cached_tokens=8` on the second request).
- **Structural fix (ours, pending)**: declare K/V as per-layer `kAppendRows` components (row hooks, `rows_per_chunk = 256`, `kPrivateState` only for small fixed state), as recommended in gufo#1317. Gains: captures copy only valid tokens, follow-up turns copy only new tokens (chunk sharing), many frontiers fit the cache, backing segments stay ~128 MiB instead of growing to the whole budget. Revisit the 8 GiB cap afterwards. Execution plan and stage status: `docs/PLAN-KV-ROWS.md`.
- Load-only verification caveat: the 192k "verified" number was serve **startup**, never a long-context request.
- Remaining upstream hygiene (separate from #1319): capture off the request thread; clean refusal instead of `std::bad_alloc`; non-zero default `stall_timeout`; timeouts on the unbounded waits.

### Mid-prompt (partial-prefix) checkpoint reuse is not implemented
- Measured during the KV-rows Stage 2 gate (2026-11-21): with a 1223-token prompt, an
  identical second request reuses fully (`cached_tokens=1223`, restore 3-4 ms), but a
  request whose prompt extends the first one *inside the same user turn* misses
  (`cache_miss_reason=no_compatible_boundary`, `common_prefix_tokens=1221`,
  `nearest_checkpoint_tokens=0`). Identical result on the 0.3.1 single-blob adapter, so
  this is not a KV-rows regression.
- Cause: Flint prefills the prompt in one unsplit pass and the adapter captures only at
  request boundaries, so the only checkpoint is at the end of the first prompt; anything
  shorter has nothing at or below its common prefix. The extension leaves
  `Adapter::PrefillCheckpointPositions` unimplemented, and the host's boundary policy is
  in Gufo (`component_text_cache.cpp`).
- Effect: chat follow-ups reuse the full previous turn (the workload that matters), but
  "user edited the tail of a long prompt" pays full prefill. Not a gate: the Stage 2 gate
  tests the multi-turn follow-up instead.
- Action if ever needed: implement `PrefillCheckpointPositions` (in-pass checkpoints at
  the prefill chunk boundaries) and have the host capture them. Separate from the KV-rows
  work; no partial-prefix claim should be made in docs until then.

### Thinking cannot be disabled from the request (template has no switch)
- Flint's pinned template has no `enable_thinking` conditional
  (`tests/golden/chat_template.jinja`), and `src/chat_template.cpp` renders
  `<|assistant|>` unconditionally (`(void)options`). The host's `--think`,
  `thinking`, `chat_template_kwargs.enable_thinking` and `reasoning_effort` therefore do
  not change the prompt for this extension; the model always decides.
- Consequence for tests/tools: to get an immediate answer, set the host's reasoning
  budget (`"reasoning_budget_tokens": 0`), which forces `</think>`. `gates/cache_reuse.py`
  uses that and compares `reasoning_content + content`.
- Not a bug against the pinned template; revisit only if the model gains a
  non-thinking mode.

### Small RAM budgets retain nothing on the row cache (accounting note)
- Measured (Stage 3, `--cache-ram-bytes 419430400`): the ledger sits at ~374 MB with a
  123 MB checkpoint, so the backing reservation consumes nearly the whole budget and every
  captured checkpoint is pruned before the next request. `cache_captures=1` on all seven
  requests, ledger flat (no growth, no refused captures), and a repeat of the first prefix
  is a miss. This is a valid capture/prune stress, not a retention test.
- To exercise retention under pressure, use a budget of at least ~1.5 GiB (one checkpoint
  plus backing). Revisit when the backing sizing is understood; the effective checkpoint
  share of the budget is far below the nominal value.
- `event=state_lending sheds=N` is emitted only when the resources' `set_shed` hook is
  installed, which this path does not do; its absence is not a pin signal. Borrow health
  is judged from the flat ledger plus continued captures.
