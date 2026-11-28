# Deferred items

### Base-session VRAM undercount in `Model::SessionBytes` (pre-existing, still open after 0.5.0)
- `Session::Init` (`src/gpu/engine.cpp`) allocates per-session scratch pools, step-graph workspaces and allocator slack totalling ~600–666 MiB more than `SessionBytes` accounts for. Measured with `flint_spec_gate --mode memory` (second-session delta): spec-off claim 1457 MiB vs measured 2123 MiB; spec-on claim 1873 MiB vs measured 2443 MiB. Identical in both modes, so it predates DFlash2 and is not an M5 regression.
- The M5 memory gate criterion (hidden ring + drafter accounted in `ResourceClaim`) is met: the spec-on per-session claim grows 436 MiB over spec-off and the measured growth is exactly the 320 MiB hidden ring.
- Action (open, not a release gate): audit `Session::Init` allocations against
  `SessionBytes` and add the missing terms. Every admission/claim number quoted
  in the docs is this underestimate plus the accounted terms; the 0.5.0 memory
  gate measures the delta, not the absolute. Original evidence log
  (`/tmp/opencode/m5-smoke/memory2.log`) was lost with the 2026-11-07 reboot —
  re-measure with `flint_spec_gate --mode memory` before changing the code.

### Serve at 262144-token context: eight sessions now load; the spread, not capacity, is the limit
- **Status after 0.5.0:** the staged weight copy shipped (#11, `cbd1075`) and issue #10
  is closed. What remains open here is (a) the load-time spread under memory pressure,
  mitigated 2.05x but not removed, and (b) a 262144-token **request**, which has never
  been run through serve.
- History: on 2026-11-17 `gufo serve -c 262144` never completed the `sessions` load phase (11+ min, killed; blocked in KFD GPU allocation, GTT climbing past 55 GB, with AND without `--speculative`); 196608 loaded in ~62 s and the boundary was not bisected.
- **That hang no longer reproduces** (2026-11-22 on the 0.35.0 build with the row-component cache; re-checked 2026-11-23 on Gufo 0.36.0): 1 session at 262144 reaches `listening` in ~10 s, spec off. The original hang was never bisected, so the cause is not pinned; only the current behaviour is verified.
- **Load time is not stable at two sessions on 0.36.0**: five runs measured 10, 36, 56, 138 and 187 s. The spread is the pageable `hipMemcpy` out of the memory-mapped artifact (`src/gpu/device_model.cpp:97` for the stacked experts and `:206` in `Copy()`): while part of the file is not resident the kernel evicts the pages the copy is reading and faults them back in, so the upload crawls. Releasing the mapping after upload (0.4.0's #9) is the *enabler* — those pages are evictable at all, where before they stayed pinned for the process lifetime — not the cause, and #9 is not a stall fix (issue #10, sebt3). Measured on `main`, where both arms include #9: direct mapping 32.9/35.6/51.4 s with 65–139k file refaults/s; a pinned 64 MiB staging buffer 24.6/25.2/25.6 s with 21–46k/s. That A/B did **not** reproduce the old hang, so this is the shape of the spread, not a diagnosis of it. One session stays near 10 s. Do not quote a two-session load time without a warmed page cache.
- **Merged in 0.5.0** as `cbd1075` (PR #11, staged copy on four threads). Our interleaved A/B on 2026-11-26, three arms and three reps, **one session** at 131072 (the pressure script passed `-s 2`, which is `--seed`; every load log reports `sessions=1`, and the flag for sessions is `-j`): healthy node `main` 8.0-8.5 s, staged one-thread 9.5-10.0 s, staged four-thread 8.5-9.0 s; under pressure (`posix_fadvise(DONTNEED)` on the GGUF plus a 44 GiB anonymous hog) 132.6/132.6/134.6 s, 138.1/138.1/138.6 s and 64.0/64.5/64.5 s. So it buys 2.05x under pressure for 0.3-0.5 s on a healthy node; the one-thread commit alone is slower than `main` and is history only. The mitigation is session-independent (it changes the weight upload, which happens once), and the one-session framing is what the numbers support. Note our rig never moved `workingset_refault_file` (0 in all nine loads) yet reproduced the slowdown, so the mitigation is wider than the refault mechanism — the counter is not a reliable monitor here. The 0.4.0 release note is corrected.
- The **hang** itself stays a known unknown: six loads on sebt3's node and eighteen on ours on `main`, none stalled (ours degraded to 133 s and finished). No reproduction exists to test against; reopen issue #10 with the memory state if one reappears.
- The remaining limit at 262144 is **load time under memory pressure**, not admission. On the
  0.4.0 full-row layout a session reserved 27,921,916,852 B (26.0 GiB) of request state, so
  with the 87.3 GiB admission budget three were admitted (the third in ~112 s, leaving ~5.8 GiB
  host available) and the fourth was refused: "at most 3 sessions fit at this context".
  With the ring (`flint-kv-rows-window-v1`, 0.5.0) a session reserves about **8.7 GiB**;
  measured on the release build (`-j`, warm cache, this 128 GB box): 4 sessions load in 9.5 s
  using 60.0 GiB (`gpu_device_used_mib=61,491`), 5 in 12.0 s / 68.8 GiB, 6 in 12.5 s /
  77.5 GiB, 8 in 14.0 s / 94.9 GiB with 22.6 GiB host left; 7 measured once at 41.0 s, the one
  outlier and in the same pressure regime the staged copy addresses. The old "advertise two"
  was a capacity statement; on this box eight now fit, though latency and host headroom still
  argue for fewer sessions. Admission remains free memory minus the host reserve at that
  moment, not the device size.
- Not verified: **262144-token requests**. Long-context requests are exercised to 130k (Stage 4/5: dense, sparse, speculative, disk restart — `docs/PLAN-KV-ROWS.md`); a 262144 prefill has not been run through serve, so no throughput claim should be made at that depth.
- Measured on the merged ring tree (`c47ce23` and later): the item collapsed as predicted — four
  sessions at 262144 use 60.0 GiB and load in 9.5 s, eight use 94.9 GiB and load in 14.0 s (see
  the capacity bullet above). What remains is the load-time spread, not the ceiling.

### Serve draft counters report zero (root-caused 2026-11-18; not a logging bug)
- Serve drives Flint via `SelectNext()`/`Advance()` (per PLAN-DFLASH2: sampler-history correctness), so the counter-copying `DecodeStep` hook (`runner.cpp:612`) is never called; `Descriptor().multi_token_decode` is false and the host's `StepDecode` path never reads `SpecCycle::stats()`.
- Speculation itself demonstrably runs in serve: decode 74.3 t/s at 8k ≈ spec-on (74.7) vs spec-off (63.6). A same-request A/B tps comparison (serve with vs without `--speculative dflash2`) is the accepted proof for 0.3.0 — no code change.
- Fix options (still open; 0.5.0 shipped without them): (A) flip
  `multi_token_decode` capability — extension-only but changes the serve
  scheduling path (needs full re-verification); (B) host-side counter channel on
  `TextRunnerState` — touches Gufo core, out of extension scope.

### MMQ scratch arena never reclaimed (noted during `fb616ad` root-cause)
- `qfn_stack_pool::reserve` allocates a 256 MiB arena per (device, stream); each session owns a private stream, and `flint_mmq_release_stream` (`flint_mmq.h:10`) is declared but never called — arenas are never returned. The claim now models this honestly, but a long-lived server with rotating sessions could accumulate arenas. Action: call `flint_mmq_release_stream` on session teardown, or share one arena under the device lock.

### Long-context first-request stall (0.3.1 caps the claim; real fix = kAppendRows, gufo#1317)
- At `-c 130000` with the default RAM cache, the host captured a full private-state snapshot **synchronously on the request thread** (~15.7 GB) before answering: first request 14.2 s vs 0.47 s with the capture refused. Prefill itself was healthy; `--stall-timeout-ms` defaults to 0, so on slower machines the stall logs nothing and looks like a hang.
- Upstream root cause (Gufo maintainer, gufo-org/gufo#1317; closed 2026-11-21):
  `ResourceCharge::ReserveBacking` compared each backing block with every earlier one,
  so one 15.7 GB frontier (~240k blocks) made the first capture's allocation take
  ~20 s; fixed with a sort in gufo-org/gufo#1319 (12 ms). Our 0.3.1 cap is therefore a
  **mitigation, not the fix**.
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

### 8 GiB snapshot cap: kept, and no longer binding at 130k (0.5.0)
- Measured 2026-11-21 with the merged linear-time fix (gufo#1319, `93a1d230`) in a
  disposable tree, cap removed: capture of a 128,981-token frontier costs **+4.8 s** on a
  104 s prefill (`cache_snapshot_bytes=12,946,504,512`), and a follow-up request restores
  in **172 ms** instead of re-prefilling for ~104 s.
- Gufo **0.36.0 contains `93a1d230`**, so the version guard's condition is met. The cap
  stays in 0.4.0 because raising it changes admission and memory behaviour and needs its
  own 130k A/B on the 0.36 release build.
- Measured consequences on 0.4.0/0.36.0 (release battery, full-row layout `flint-kv-rows-v2`): the 8 GiB cap is 8,581,545,984 B
  of RAM cache, about 85,600 rows. At 24k two frontiers fit and RAM/disk reuse is exact.
  At 60k an identical repeat restores in 82 ms and a follow-up still captures; after a disk
  restart the restored 59,459-row frontier is re-captured (5,967,430,208 B), which fills
  the cap, and the follow-up capture is refused (`cache_snapshot_bytes=0`; the disk layer
  also logs `persistence_refused reason=min_step`). At 130k one partial 63,488-row
  checkpoint is retained: an identical repeat restores 63,488 of 128,981 tokens in 85 ms
  and prefills 65,493; after a restart that state is a miss. Answers stayed identical in
  every leg.
- Measured on the 0.5.0 release build (ring, `flint-kv-rows-window-v1`; clean clone
  `58bc87c`, warm cache), the cap itself is unchanged at 8,581,545,984 B:
  - 24k: snapshot `959,071,040 B`, restore 12.5 ms.
  - 60k: snapshot `1,921,088,320 B` (0.4.0: 5,967,430,208 B), so after a disk restart the
    re-capture `1,746,911,808 B` also fits and the follow-up capture that 0.4.0 refused now
    succeeds (`r3` captures `1,750,424,128 B`).
  - 130k: the frontier retains `3,995,190,656 B` and the follow-up `3,776,601,216 B`, so
    both fit; an identical repeat restores all 128,981 tokens from RAM in 50.9 ms and, after
    a disk restart, 128,980 tokens from disk in 1,394 ms.
- Decision in 0.5.0: **the cap stays at 8 GiB and the raise is not needed.** With the ring
  nothing measured approaches it, and it is still a cheap guard against a full-row layout.
  Raising it is a separate, measured change if a future layout brings the bytes back.

### Ring length must carry the masked-tile slack (PR #5) — **FIXED**
- **FIXED** (pre-merge, `1bf1538`, with `968b720` for the one-row-short append):
  `RingLength` is now `round64(sliding_window + tile + kAttnKeyTile - 1)`,
  `kAttnKeyTile` is exposed through `src/gpu/kernels/kernels.hpp`, the
  `KvRows()` comment says a ring layer is not position-indexed and must be read
  through `TransferRows()`, and `flint.prefill_split_tile4136` (tile 4136, 23
  rows of slack before the fix) is in the suite and passes on the 0.5.0 release
  build. The analysis below is kept as the record of why the term exists.
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

### `prefill_tile` must be even (pre-existing, found by sebt3 while reviewing #5)
- A prefill pass with an odd row count leaves its last row wrong. On `main` (`8d70c09`, no
  ring) tile 4157 first differs at row 4156 and tile 4159 at row 4158 — the last row of
  the first pass — while 4096, 4158, 4162 and 4160 all give 0 differing. At 4159 the
  per-split max |diff| values are the same on `main` and on the ring, so the ring does not
  cause it.
- Default 4096 hides it; nothing validates the option, so a user-set odd tile silently
  produces rows that disagree with an unpaged prefill.
- **FIXED** on `main` (`c10c70c`): `ValidateEngineOptions` rejects `prefill_tile < 2` and
  odd values at the top of `Model::Load`, with the measurement in the message and the
  unpinned mechanism stated as such. It lives in `src/gpu/engine_options.cpp` (host tier)
  so `flint.engine_options` covers it without a GPU. Nothing in the extension set
  `prefill_tile`, so no caller changes behaviour.
- Still open: the mechanism is not pinned. If a future pass shape (verify width, a
  different tile) can end on an odd row for another reason, the same symptom returns
  without this guard firing. `flint.prefill_split_tile4136` remains the near-boundary leg.

### Rewind guard and `tools/drafter_parity.cpp` under the ring (PR #5)
- `Session::Rewind` flags `ring_broken` when a rewind exceeds
  `ring_len - sliding_window + 1` (4160 at the default tile). The strict bound also pays
  for the masked aligned key tiles: `ring_len - sliding_window - kAttnKeyTile + 2` = 4129.
- Production rewinds are <= 7 rows, so the tighter guard costs nothing and fails loudly
  instead of serving silently wrong rows.
- **FIXED** on `main` (`e3913e7`): `tools/drafter_parity.cpp` starts each prompt from
  `Reset()` instead of `Rewind(0)`, so the ring is overwritten as the prompt is written and
  no deep rewind is attempted. Verified: the tool runs all 14 prompts, 58 s, exit 0.

### Matrix-core attention: prerequisites before it can be the default (#7)
- As merged, `FLINT_MATRIX_ATTENTION` is OFF. A default build dispatches the scalar path, keeps `kGlobalSplits` at 32 and the payload layout unchanged, so it is behaviourally `main`. The matrix build is opt-in, one build directory per mode (`docs/ATTENTION-WMMA.md`).
- M1: `src/gpu/kernels/kernels.hip.cpp:1889` `SetMatrixAttention` mutates `g_matrix_attention` at runtime, while `FlintPayloadLayout()` in `runner.cpp:53` keys off the compile-time `MatrixAttentionBuild()`. A default build that ever called the setter on a serving path would write matrix-kernel rows under the un-suffixed layout string. Today only tests and the bench call it. Fix before flipping: assert the two agree on the serving path, or make the setter test-only.
- M2: the V-zeroing in `AttentionSliceWmmaKernel` (`kernels.hip.cpp:1305-1310`, split-KV staging of V) is documented as required, but `0 * finite == 0`, so the mechanism must be non-finite V, stale ring rows, or a hardware quirk. None is proven. The gate fills V with finite values only. Fix before flipping: add an offset whose masked V region is Inf/NaN and assert row invariance still holds, or drop the zeroing and its claim.
- M3: doc claims narrowed in `docs/ATTENTION-WMMA.md` (drafter identity verified at 8k/32k only; one-row decode is within 2 % to 29k and ~8 % faster at 131k). Re-measure at 131k before claiming more.
- L1: `AttentionSliceWmmaKernel` is compiled in every build; `#ifdef FLINT_MATRIX_ATTENTION` would make "inert" literal (compile time and binary size only).
- L2: `flint_attn_bench` is not guarded by `if(TARGET gufo_flint_hip)` in `extension.cmake`, so a no-HIP configure cannot link it (same as the other tools, not a regression).
- L3: the key bound in the matrix kernel is computed as `int(qpos) - int(kt)` (`kernels.hip.cpp:1419`); fine below 2^31 context, an unsigned-safe form would remove the latent overflow.
- L4: the matrix kernel runs at the same ~176 VGPRs/thread as the reference kernel at `__launch_bounds__(256)`; worth a `-Rpass` check on gfx1151 before the mode becomes default.
