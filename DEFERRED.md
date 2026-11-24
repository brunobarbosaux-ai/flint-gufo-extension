# Deferred items

### Base-session VRAM undercount in `Model::SessionBytes` (pre-existing)
- `Session::Init` (`src/gpu/engine.cpp`) allocates per-session scratch pools, step-graph workspaces and allocator slack totalling ~600–666 MiB more than `SessionBytes` accounts for. Measured with `flint_spec_gate --mode memory` (second-session delta): spec-off claim 1457 MiB vs measured 2123 MiB; spec-on claim 1873 MiB vs measured 2443 MiB. Identical in both modes, so it predates DFlash2 and is not an M5 regression.
- The M5 memory gate criterion (hidden ring + drafter accounted in `ResourceClaim`) is met: the spec-on per-session claim grows 436 MiB over spec-off and the measured growth is exactly the 320 MiB hidden ring.
- Action before release: audit `Session::Init` allocations against `SessionBytes` and add the missing terms. Evidence: `/tmp/opencode/m5-smoke/memory2.log`.

### Serve at 262144-token context: loads, but advertise two sessions (capacity, not a hang)
- History: on 2026-11-17 `gufo serve -c 262144` never completed the `sessions` load phase (11+ min, killed; blocked in KFD GPU allocation, GTT climbing past 55 GB, with AND without `--speculative`); 196608 loaded in ~62 s and the boundary was not bisected.
- **That hang no longer reproduces** (2026-11-22 on the 0.35.0 build with the row-component cache; re-checked 2026-11-23 on Gufo 0.36.0): 1 session at 262144 reaches `listening` in ~10 s, spec off. The original hang was never bisected, so the cause is not pinned; only the current behaviour is verified.
- **Load time is not stable at two sessions on 0.36.0**: five runs measured 10, 36, 56, 138 and 187 s. This matches the trade-off of releasing the GGUF mapping after upload (0.4.0): the artifact's pages become evictable, and a host that cannot keep 24 GB cached pays a disk read on the next load. One session stays near 10 s. Do not quote a two-session load time without a warmed page cache.
- The remaining limit is device capacity, and the host refuses cleanly instead of hanging. A 262144-token session reserves 27,921,916,852 B (26.0 GiB) of request state. On 0.36.0 the admission budget is 93,764,722,688 B (87.3 GiB), so **three** sessions are admitted (the old build refused the third) but the third loads in ~112 s and leaves ~5.8 GiB host available; four fail at admission: "at most 3 sessions fit at this context". The budget is free memory minus the host reserve at that moment, not the device size; measured `gpu_device_used_mib` is 51.0 GiB for 1 session, 76.8 GiB for 2 and 102.6 GiB for 3, of which ~25 GiB is weights and non-claim allocation outside the request-state reservation.
- Not verified: **262144-token requests**. Long-context requests are exercised to 130k (Stage 4/5: dense, sparse, speculative, disk restart — `docs/PLAN-KV-ROWS.md`); a 262144 prefill has not been run through serve, so no throughput claim should be made at that depth.
- With the ring-buffer PR (sebt3 `feat/ring-kv`, #5) the same box loads 2 sessions at 262144 using 43.6 GiB and **4 sessions using 61.5 GiB** (10 s). When it lands this item collapses to "four sessions at the native context".

### Serve draft counters report zero (root-caused 2026-11-18; not a logging bug)
- Serve drives Flint via `SelectNext()`/`Advance()` (per PLAN-DFLASH2: sampler-history correctness), so the counter-copying `DecodeStep` hook (`runner.cpp:612`) is never called; `Descriptor().multi_token_decode` is false and the host's `StepDecode` path never reads `SpecCycle::stats()`.
- Speculation itself demonstrably runs in serve: decode 74.3 t/s at 8k ≈ spec-on (74.7) vs spec-off (63.6). A same-request A/B tps comparison (serve with vs without `--speculative dflash2`) is the accepted proof for 0.3.0 — no code change.
- Fix options deferred to 0.5.0 (was v0.4.0): (A) flip `multi_token_decode` capability — extension-only but changes the serve scheduling path (needs full re-verification); (B) host-side counter channel on `TextRunnerState` — touches Gufo core, out of extension scope.

### MMQ scratch arena never reclaimed (noted during `fb616ad` root-cause)
- `qfn_stack_pool::reserve` allocates a 256 MiB arena per (device, stream); each session owns a private stream, and `flint_mmq_release_stream` (`flint_mmq.h:10`) is declared but never called — arenas are never returned. The claim now models this honestly, but a long-lived server with rotating sessions could accumulate arenas. Action: call `flint_mmq_release_stream` on session teardown, or share one arena under the device lock.

### Long-context first-request stall (0.3.1 caps the claim; real fix = kAppendRows, gufo#1317)
- At `-c 130000` with the default RAM cache, the host captured a full private-state snapshot **synchronously on the request thread** (~15.7 GB) before answering: first request 14.2 s vs 0.47 s with the capture refused. Prefill itself was healthy; `--stall-timeout-ms` defaults to 0, so on slower machines the stall logs nothing and looks like a hang.
- Upstream root cause (Gufo maintainer, gufo-org/gufo#1317): `ResourceCharge::ReserveBacking` compared each backing block with every earlier one, so one 15.7 GB frontier (~240k blocks) made the first capture's allocation take ~20 s; fixed with a sort in gufo-org/gufo#1319 (12 ms). Our 0.3.1 cap is therefore a **mitigation, not the fix**.
- Flint-side mitigation (`0.3.1`): the retained-snapshot capacity/ceiling claim is capped at 8 GiB, so a snapshot larger than the cap is refused cheaply instead of captured. Verified: 130k first request 0.47 s; 8k keeps prefix retention (`cached_tokens=8` on the second request).
- **Structural fix (done, `docs/PLAN-KV-ROWS.md`)**: K/V are per-layer `kAppendRows` components (row hooks, `rows_per_chunk = 256`, `kPrivateState` only for small fixed state), as recommended in gufo#1317. Gains: captures copy only valid tokens, follow-up turns copy only new tokens (chunk sharing), many frontiers fit the cache (backing-segment sizing not separately measured). Execution plan and stage status: `docs/PLAN-KV-ROWS.md`; the 8 GiB cap is now a version guard (see below).
- Depth caveat: long-context requests are now exercised to **130k** (Stage 4/5: dense, block-sparse, speculative, disk restart — `docs/PLAN-KV-ROWS.md`), but 262144 is verified at serve **startup** only (see the 262144 item above), so no throughput claim exists at that depth.
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

### 8 GiB snapshot cap: version guard satisfied, raise moves to 0.5.0
- Measured 2026-11-21 with the merged linear-time fix (gufo#1319, `93a1d230`) in a
  disposable tree, cap removed: capture of a 128,981-token frontier costs **+4.8 s** on a
  104 s prefill (`cache_snapshot_bytes=12,946,504,512`), and a follow-up request restores
  in **172 ms** instead of re-prefilling for ~104 s.
- Gufo **0.36.0 contains `93a1d230`**, so the version guard's condition is met. The cap
  stays in 0.4.0 because raising it changes admission and memory behaviour and needs its
  own 130k A/B on the 0.36 release build.
- Measured consequences on 0.4.0/0.36.0 (release battery): the 8 GiB cap is 8,581,545,984 B
  of RAM cache, about 85,600 rows. At 24k two frontiers fit and RAM/disk reuse is exact.
  At 60k an identical repeat restores in 82 ms and a follow-up still captures; after a disk
  restart the restored 59,459-row frontier is re-captured (5,967,430,208 B), which fills
  the cap, and the follow-up capture is refused (`cache_snapshot_bytes=0`; the disk layer
  also logs `persistence_refused reason=min_step`). At 130k one partial 63,488-row
  checkpoint is retained: an identical repeat restores 63,488 of 128,981 tokens in 85 ms
  and prefills 65,493; after a restart that state is a miss. Answers stayed identical in
  every leg.
- Action in 0.5.0: re-measure the 130k A/B on the merged ring tree before raising or
  removing the cap (`kFlintMaxRetainedSnapshotBytes` in `runner.cpp`). The ring's
  `flint-kv-rows-window-v1` layout retains 3,995,190,656 B at 128,981 tokens and restores
  the full prefix in 52 ms, so the cap may no longer bind and the raise may be
  unnecessary. Until the ring lands, 130k prefix reuse stays partial by design.

### Ring length must carry the masked-tile slack (PR #5, required before merge)
- slimsami's analysis: aligned SWA key tiles start at
  `floor((pos_min - window + 1) / kWmmaKeys) * kWmmaKeys`, up to `kWmmaKeys - 1` keys
  below the window. Those keys are masked, but a row's last bit can still depend on the
  content of keys its wave partner masks (<= 1 ulp, hardware WMMA). In a ring such a slot
  may already be overwritten, and what was written there depends on how the prompt was
  split, so split invariance needs the aligned tiles to stay inside rows the ring holds.
- A pass touches at most `tile + window + kWmmaKeys - 2` rows, i.e. `tile + window + 30`
  with `kWmmaKeys = 32`.
- `RingLength` (`src/gpu/engine.cpp`) is currently `round64(sliding_window + tile)`. With
  the default tile 4096 that leaves 63 rows of slack (`round64(4609) = 4672`), which is why
  the rebased run showed 0 differing. A `prefill_tile` with `(sliding_window + tile) % 64`
  in 35..63 leaves fewer than 30 and splits may differ by an ulp.
- Fix: `need = sliding_window + tile + kWmmaKeys - 1`, still rounded to 64 and still
  rejected when it exceeds capacity. `kWmmaKeys` is a `constexpr` local to
  `src/gpu/kernels/kernels.hip.cpp`; expose it through `kernels.hpp` or name the term in
  the engine.
- Regression test: run `flint.prefill_split` with `prefill_tile = 4136` (23 rows of slack
  before the fix) and require 0 differing after it.
- Also agreed: assert or comment in `KvRows()` that a window layer must be read through
  `TransferRows`. Do **not** assert `RingLength == sliding_window + tile` in
  `prefill_split_test`: comparing the last `sliding_window` rows is the contract, and the
  extra ring rows are slack that is never read unmasked.

### Drafter context floor invariant (PR #8 follow-up)
- `history_begin` is the first row actually present in the drafter context, and the
  kernels never read below it. A restore sets floor = length = frontier
  (`ResetContext(frontier)`); a gap would be `ResetContext(p)`, with any future
  `SkipContextTo` a thin wrapper over that rather than a second floor; only restoring
  drafter rows may lower it.
- No gap case exists today: `CopyHiddenRows` serves all 8,192 hidden-ring rows and
  `static_assert(kHiddenRingRows >= kPrefillChunk)` keeps a pass inside it. The ring PR
  does not touch the drafter, so nothing composes yet.
- Revisit if `kPrefillChunk` ever grows past `kHiddenRingRows`: that `static_assert` is
  where `ResetContext(first + skip)` becomes necessary.
