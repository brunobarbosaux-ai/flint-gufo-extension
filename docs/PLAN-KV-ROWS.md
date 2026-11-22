# Build plan: continuation cache as per-layer `kAppendRows` components

Status: **approved to start Stage 1** (2026-11-21). Execution plan; measurements and
decision history live in the §8 log at the bottom (append-only — never rewrite an entry).

Tracking: gufo-org/gufo#1317 (recommendation), gufo-org/gufo#1319 (upstream O(n²) fix).
Extension side: `DEFERRED.md` § "Long-context first-request stall".

Scope warning: this touches **what a checkpoint is**. Nothing in this plan may loosen a
gate. If bit-exactness fails, stop and investigate — do not adjust a threshold.

---

## 1. Goal and non-goals

### Why

Flint models the session as one `kPrivateState` blob: `state_bytes = 24 + SnapshotBytes(context)`
(`continuation_adapter.cpp:95-103`). At 130k that is a ~15.7 GB capture, copied
synchronously on the request thread, and every follow-up turn copies the whole frontier
again. Consequences measured in 0.3.1:

- first request at `-c 130000`: 14.2 s vs 0.47 s with capture refused (workaround: the
  8 GiB retained-snapshot cap, `cb7303d`);
- a large frontier pins the backing segment, so most of the 32 GiB cache budget is
  consumed even with no request in flight;
- multi-turn chat pays the full frontier copy per turn.

The cache already supports the right shape: `ComponentKind::kAppendRows` with per-token
`row_bytes`, `rows_per_chunk = 256`, plus `CopyRowsOut`/`CopyRowsIn`/`BorrowRowsView` and
the `MutationGuard` hooks. All present at **API 0.3.27** — no newer API is required.

### Goal

K and V become one `kAppendRows` component per layer each; `kPrivateState` keeps only the
small fixed state (metadata + last logits row). Capture copies only valid tokens; a
follow-up turn copies only the new chunks; backing segments stay ~128 MiB.

### Non-goals (this plan)

- Removing the 8 GiB cap. `kAppendRows` does **not** reduce the first-capture bytes; the
  copy is still synchronous. The cap stays until capture leaves the request thread (§7).
- Moving capture off the request thread, changing host stall-timeout defaults, or the
  `bad_alloc` refusal — all upstream, tracked separately.
- Drafter/hidden-ring state: not cached, not changed here.
- Any change to append numerics, sampling, or the speculative cycle.

---

## 2. Locked decisions

| # | Decision | Rationale |
|---|---|---|
| D1 | Engine API first (Stage 1), adapter rewrite after | `k_cache`/`v_cache` are private; the adapter cannot see rows. Engine work is independently testable. |
| D2 | `row_bytes = num_kv * head_dim * sizeof(__half)`, token-major | Exactly `k_cache[il]`'s layout (`engine.cpp:480-488`). `num_kv`, **not** `num_heads`. Head-major/FP32 gather is the reference's other path and does not apply. |
| D3 | One component per layer for K, one per V; `layout_version = 1` | Matches in-tree models; no inter-layer stride exists. |
| D4 | `rows_per_chunk = 256` | Cache-wide convention; tail chunk is the remainder. |
| D5 | `kPrivateState` = metadata (fixed, versioned) + last logits row (`vocab*4`) | Small, fixed, and not row-addressable. Mirrors the reference's `kMetadata`/`kLogits`. |
| D6 | Row copies are synchronous `hipMemcpy` around a stream sync; `BorrowRowsView` returns the device pointer | Device buffers live for the session lifetime, so borrows are stable until the guard drains. No stream API needed in the engine header. |
| D7 | Session position is set in `Validate`, never `BeginRestore` | A failed/partial restore must leave the slot invalid, not half-restored. |
| D8 | Engine row-write hook fires once per `Append`/`AppendAll` for `[position, position+n)` before any write | The only way to cover every write path (runner, spec target adapter) with `MutationGuard::BeforeOverwrite`. Over-reporting is safe (guard drops pins early); under-reporting corrupts the cache. |
| D9 | Keep `SaveSnapshot`/`RestoreSnapshot` for tests; refactor `RestoreSnapshot` to call the new `RestoreRows` | One implementation of the sparse `kmm` rebuild (`engine.cpp:1594-1612`) — the trap that silently breaks sparse attention. |
| D10 | Bump `state_abi` → `flint-rocm-session-v2`, `payload_layout` → `flint-kv-rows-v1` in the **same commit** as the adapter layout change (Stage 2) | A v1 single-blob disk cache must not load into a v2 multi-component adapter; the identity is the only host-visible discriminator. |
| D11 | 8 GiB cap stays until §7 is answered by measurement | Cheap refusal guard; removal is a separate, measured change. |
| D12 | `Reset()` is the canonical "empty for reuse"; `RestoreRows(0)` additionally zeroes logits and key bounds and exists only for restore paths | `Invalidate` uses `Reset()`; the two paths must not be confused (review finding, Stage 1). |

---

## 3. Target component layout

| id | component | kind | `row_bytes` | `rows_per_chunk` | `state_bytes` |
|---|---|---|---|---|---|
| 1 | `kMetadata` | `kPrivateState` | 0 | 0 | 64 (fixed struct, zero-padded) |
| 2 | `kLogits` | `kPrivateState` | 0 | 0 | `vocab_size * 4` |
| `3 + 2*l` | `kKey(l)` | `kAppendRows` | `num_kv*head_dim*2` | 256 | 0 |
| `4 + 2*l` | `kValue(l)` | `kAppendRows` | `num_kv*head_dim*2` | 256 | 0 |

`Metadata` (host struct, `static_assert(sizeof == 64)`): `uint32 version`, `uint32 position`,
`uint32 logits_valid`, `uint32 reserved`, rest zero. `version` gates the private payload;
the artifact/ABI fingerprint is the runner's job, not this component's.

`Positions()` returns every descriptor at the session position. There is no window
component and no capacity field — `context` bounds `valid_rows`.

---

## 4. Stages

| Stage | Work | Status | Gate | Commit |
|---|---|---|---|---|
| **1** | Engine API: `NumLayers`, `KvRowBytes`, `KvRows`, `Synchronize`, `RestoreRows`, `SetLogits`, `SetRowWriteHook`; refactor `RestoreSnapshot` onto `RestoreRows`; new `flint.engine_snapshot` test | **COMPLETE** | `flint.engine_snapshot` (both legs), `flint.ops`, `flint.dflash_ops`, `flint.dflash_forward`, `flint.target_adapter`; `--mode gx` 3/3 | `e0577ff` + review fixes `feb7475` |
| **2** | Adapter rewrite: descriptors (metadata/logits private, K/V per layer as `kAppendRows`), private piece capture/restore, row hooks, `MutationGuard` integration, **identity bump** (`flint-rocm-session-v2`, `flint-kv-rows-v1`) | **COMPLETE** | in-process cache: r2 `cache=memory`, `cached_tokens=1223`, restore 2.9 ms, answer identical; exact captured bytes; spec-on leg passes | `c559211` + review fixes `e1d0f6f` |
| **3** | Disk cache round trip + guard/eviction stress (multi-session, prune under a small budget) | **COMPLETE** | restart `cache=disk` (1223 tok, 45 ms; 9001 tok, 244 ms), answers identical; identity mismatch → no reuse; 7/7 under 400 MiB with flat ledger | `b2893cc` (identity log) + `4cc90dc` (gate) |
| **4** | Runner: position check in `ReconcileComponentState`, claim/cap decision by measurement | NOT STARTED | `--mode memory`; 130k latency A/B | — |
| **5** | Long-context validation: 130k capture, follow-up turns, disk cache, spec on/off, sparse on/off | NOT STARTED | full suite + benchmark vs 0.3.1 baseline | — |

### Stage 1 — engine API (detail)

Files: `src/gpu/engine.hpp`, `src/gpu/engine.cpp`, `tests/engine_snapshot_test.cpp` (+ `extension.cmake` registration).
**No changes to `continuation_adapter.*`, `runner.cpp`, or the cap.**

Additions to `Session` (public):

```cpp
std::uint32_t NumLayers() const noexcept;
std::size_t   KvRowBytes() const noexcept;   // num_kv*head_dim*sizeof(__half)
const void*   KvRows(std::uint32_t layer, bool value) const noexcept;  // device ptr
void*         KvRows(std::uint32_t layer, bool value) noexcept;
bool          Synchronize(std::string* error_msg) const;
// Rows must already be in place. Sets position, logit_rows=1, and rebuilds the
// sparse key bounds exactly as RestoreSnapshot does. Leaves logits untouched.
bool          RestoreRows(std::uint32_t position, std::string* error_msg);
bool          SetLogits(std::span<const float> row, std::string* error_msg);
// Called with [first,end) before Append/AppendAll write rows; empty clears.
void          SetRowWriteHook(std::function<void(std::uint32_t, std::uint32_t)>);
```

Behaviour:

- `KvRows`: bounds-check `layer < num_layers`; return `nullptr` when out of range. The
  non-const overload is the restore target.
- `RestoreRows`: `position <= capacity`; `position == 0` empties the session
  (`h_logits` zeroed, `logit_rows` stays 1 — `Logits()` indexes `logit_rows - 1`, so 0
  would underflow — and `kmm` cleared for non-SWA layers only, because SWA layers have no
  `kmm` buffer). For `position > 0`: syncs the stream, replicates `RestoreSnapshot`'s
  steps (`h_pos[0]=0` + 4-byte `pos` upload, `PoolKeyBlocks` over `[0, position)` for
  non-SWA layers when `sparse`, final sync), and only then sets `position` and
  `logit_rows = 1`, so a failed restore leaves the session untouched (D7). It must not
  write `h_logits`.
- `SetLogits`: require `row.size() == vocab_size`; `memcpy` into `h_logits`;
  `logit_rows = 1`.
- `SetRowWriteHook`: invoked once in `Append`/`AppendAll`, after the capacity check and
  before the first `Forward`, with `[position, position+tokens.size())`. Never invoked by
  `Rewind`. No hook installed ⇒ no behaviour change.
- `RestoreSnapshot`: replace its tail (logits copy is separate) with `RestoreRows`; keep
  the payload header/version/size checks and the `position > 0` requirement.

Test `flint.engine_snapshot` as built (GPU, skips with exit 0 without artifact/device):
one loaded model, two legs sharing it — dense at context 512/prefix 64, sparse at
`FLINT_SPARSE_ATTN=1`/`FLINT_SPARSE_TOPK=1`, context 4096/prefix 2560 (past the
2304-token sparse-engage threshold, asserted in the test). Per leg: snapshot round trip
into a *dirty* session, split-path restore (`KvRows` copies + `RestoreRows` +
`SetLogits`), `RestoreRows(0)`, single-call hook accounting, hook ordering (the row is still
unwritten when the hook runs), and a multi-chunk 5000-token append reporting one call.
Continuation is checked with a 16-token append before single-token decode: widths 1–8
take the vector path and never read the key bounds, so only `n > 8` proves the `kmm`
rebuild (verified against a mutant that skips it).

### Stage 2 — adapter rewrite (detail)

Replace the single descriptor with the layout in §3 and make the adapter functional in
one commit (descriptor-only stages cannot be gated: the cache always captures every
declared component, so throwing row hooks make every capture fail).

- **Descriptors**: `kMetadata` (64 B), `kLogits` (`vocab*4`), per-layer `kKey`/`kValue`
  (`row_bytes = num_kv*head_dim*2`, `rows_per_chunk = 256`). Keep a component table
  parallel to the descriptors so `layer`/`part` are recoverable from an id.
- **Private pieces**: `CapturePrivatePiece`/`LoadPrivatePiece` handle metadata and
  logits only, at arbitrary offsets; stage into `metadata`/`logits` byte buffers and
  record coverage. Metadata carries `version`, `position`, `logits_valid`.
- **Rows**: `CopyRowsOut`/`CopyRowsIn` validate range/size, `Synchronize`, `hipMemcpy`
  at `KvRows(layer, is_value) + first*row_bytes`, record coverage, return `Settled`.
  `BorrowRowsView` returns the device span for `kAppendRows` only. Accept tail ranges
  (`end - first` not a multiple of 256).
- **Restore**: `BeginRestore` requires one position per descriptor, all at the same
  `valid_rows`, guards every append component over `[0, capacity)`, synchronizes, and
  records `expected`/coverage. `Validate` checks coverage per component
  (`valid_rows*row_bytes` for rows, `state_bytes` for private), parses metadata
  (`version`, `position == expected`), then `RestoreRows(position)` and
  `SetLogits(logits)` — position is set only here (D7).
- **Guard**: `CreateSlot` installs the engine row-write hook that calls
  `guard.BeforeOverwrite(id, first, end)` for every append component; the slot destructor
  clears the hook and calls `BeforeRelease(id, 0, capacity)`; `Invalidate` calls
  `BeforeRelease` + `Reset()` + `AfterReset()`.
- **Identity** (moved here from the old Stage 4): bump `kFlintStateAbi` →
  `flint-rocm-session-v2` and both `payload_layout` strings → `flint-kv-rows-v1` in the
  **same commit** as the layout change. A v1 disk cache must never be readable as the new
  layout, and the identity is the only thing that distinguishes them for the host.

### Stage 3 — disk cache + eviction stress (detail)

Disk write/read round trip at 8k and 32k, old-cache rejection with the bumped identity,
repeated prefix reuse across a multi-turn session (guard drains, no pin leaks), and a
small `--cache-ram-bytes` budget so checkpoint pruning runs.

### Stage 4 — runner claim (detail)

`ReconcileComponentState` additionally asserts `flint.session().Position() == position`.
`ResourceClaim` unchanged except for the cap decision from §7; do not add
`prompt_state_bytes_per_token` unless the host would otherwise under-charge (avoid
double counting with `per_request_state_bytes`).

---

## 5. Gates and commands

```bash
# build (one GPU process at a time)
pkill -x flint_spec_gate; pkill -x gufo
cd /home/bruno/AI/Gufo-api03 && nix develop -c cmake --build build/host-test \
  --target gufo flint_spec_gate --parallel 16

# unit tests, one at a time (full -R flint hangs)
ctest --test-dir build/host-test -R '^flint.ops$' --output-on-failure
# same for flint.dflash_ops, flint.dflash_forward, flint.target_adapter,
# flint.engine_snapshot (new in Stage 1)

# spec gates
flint_spec_gate --target <gguf> --prompts gates/fixtures/... --mode gx
flint_spec_gate --target <gguf> --mode memory
```

Never loosen: stage 1e-4, selector 5e-6, Level 2 95%. A failure is a finding.

---

## 6. Traps

1. `row_bytes = num_kv * head_dim * sizeof(__half)` — token-major. `num_heads` or
   head-major silently corrupts attention.
2. `component.layer` indexes `k_cache[layer]`/`v_cache[layer]`; there is no stride.
3. Sparse: `RestoreRows` must rebuild `kmm` exactly as `RestoreSnapshot`; otherwise
   sparse decode reads wrong key blocks while looking correct at low depth.
4. Position in `Validate`, not `BeginRestore`.
5. `BeforeRelease` on slot destruction and `AfterReset` on invalidate, or the guard leaks
   pins and the cache stalls.
6. Tail chunks are not multiples of 256 — accept any `first..end` range.
7. `BorrowRowsView` exposes device memory: never reset the session while a borrow is
   pinned; `Rewind` is fine (rows below the frontier are stable).
8. Restore must set `logit_rows = 1`, or the next decode samples stale logits.
9. Bump both identity strings; a v1 disk cache must be rejected, not misread.
10. Do not delete `SaveSnapshot`/`SnapshotBytes` — tests and `spec_gate` tooling still
    use them; only the adapter stops.
11. **`CopyRowsIn` must accept a device source.** With `BorrowRowsView` returning device
    memory, the cache passes that same span back to `CopyRowsIn` when the source slot is
    still live. A hard-coded `hipMemcpyHostToDevice` silently "works" on this APU via host
    mapping and corrupts on a discrete GPU — it is the Stage 3 multi-session path.
    Use `hipMemcpyDefault` (or a pointer-attribute branch).
12. **`State::failed` must be atomic.** Non-overlapping loads are allowed on independent
    streams; a plain `bool` written by two threads is a data race.
13. **`CopyRowsOut` bounds by capacity, not `Position()`.** Preservation after a rewind
    asks for rows that are stable but outside the live frontier.
14. **Gate assertions must prove the copy.** `cache_snapshot_bytes` is a *computed* size
    and `cache=memory` can be served from the live session; assert `cache_restore_ms > 0`
    and the exact captured byte count, or the gate passes without a restore.
15. **`cache_snapshot_bytes` is a sum over captures, not one checkpoint.** The host adds
    every published snapshot (`text_model_runner.cpp:590,777,814`). In-pass checkpoints are
    prompt prefixes, so their chunks are shared and each extra capture adds only
    `vocab*4 + 64` (logits + metadata). Exact form:
    `rows*row_total + cache_captures*(vocab*4 + 64)`. Verified at 1223/7268/9001 tokens.
16. **A follow-up turn captures at most one new chunk per component.** Its new bytes are
    the tail remainder after the last 256-row boundary plus the private payload of each
    capture: `r3 <= rows_per_chunk*row_total + captures*(vocab*4 + 64)`. A ratio-of-r1
    bound is prompt-length dependent (2.4% at 8880 tokens looked fine but failed a 2%
    rule); use the absolute chunk bound.

---

## 7. Open questions (answer with measurements, not opinion)

- **Q1 — the 8 GiB cap.** Remove/raise only after Stage 5 measures 130k first-request
  latency with the row components in place. Expectation from the scoping: unchanged
  first-capture latency, much cheaper follow-up turns. If latency is unchanged and
  failures are clean, the cap can become the host budget instead of a hard 8 GiB.
- **Q2 — retained-snapshot budget.** With many frontiers now fitting, re-measure the
  cache budget needed for N concurrent prefixes at 32k/130k. Record the number in
  `DEFERRED.md`.
- **Q3 — `prompt_state_bytes_per_token`.** Only if the host under-charges prompt KV once
  rows are visible; verify against `--mode memory` before setting it.

---

## 8. History log (append-only)

### 2026-11-21 — scoping complete, Stage 1 approved
- Scoping (explorer, read-only) confirmed the whole contract exists at API 0.3.27;
  nothing newer is needed. Critical path is the engine, not the adapter.
- Key correction recorded: `kAppendRows` does not reduce the first-capture bytes
  (~15.7 GB at 130k, still synchronous). Wins are chunk sharing on follow-up turns,
  many frontiers per budget, and ~128 MiB backing segments.
- 8 GiB cap retained for now (D11).
- Reply posted on gufo#1317 accepting the maintainer's root cause and committing to this
  plan: <https://github.com/gufo-org/gufo/issues/1317#issuecomment-7069663371>
- Docs corrected in `a9d87d9`: 192k was a load-only check; root cause is
  `ReserveBacking` O(n²) (their #1319), our cap is a mitigation.
- #1319 status at the time: OPEN, MERGEABLE, BLOCKED on review (+39/−3, 2 files).

### 2026-11-21 — Stage 1 COMPLETE (engine row API)
- Commits: `e0577ff` (API + test) and `feb7475` (review fixes). Pushed to `main`.
- Added: `NumLayers`, `KvRowBytes`, `KvRows` (const/non-const), `Synchronize`,
  `RestoreRows`, `SetLogits`, `SetRowWriteHook`; `RestoreSnapshot` now runs on
  `RestoreRows` + a direct logits memcpy. No adapter/runner/cap change.
- Gates (orchestrator-run): `flint.engine_snapshot` 18.1 s both legs; `flint.ops`,
  `flint.dflash_ops`, `flint.dflash_forward`, `flint.target_adapter` pass; `--mode gx`
  3/3 byte-identical (Q6_K drafter, 128 tokens).
- Review findings fixed: `RestoreRows(0)` no longer memsets SWA layers' null `kmm`;
  `position`/`logit_rows` are assigned only after the device work succeeds;
  `RestoreSnapshot` no longer allocates per restore; hook ordering and multi-chunk
  coverage added.
- Review finding that mattered most: the first sparse leg was inert (512 context, below
  the 2304-token engage threshold), so a broken key-bound rebuild would have passed. The
  leg now runs at 4096/2560 with top-k 1 and continues with a 16-token append, because
  widths ≤ 8 take the vector path and never read `kmm`.
- Known low-priority leftovers: one redundant sync in `RestoreSnapshot` (harmless), the
  `KvRows` const overload returning a non-const pointer internally (public API correct),
  and the pre-existing `AttnCore` unused-variable warning.

### 2026-11-21 — Stage 2 re-scoped (adapter rewrite in one commit; identity bump moved)
- Reason: the cache captures every declared component on every capture, so a stage that
  declares `kAppendRows` descriptors but leaves `CopyRowsOut`/`CopyRowsIn` throwing is
  ungatable — no capture can succeed. The former Stage 2 + Stage 3 are therefore merged
  into one "adapter rewrite" stage with the row hooks and guard included.
- The identity bump moved from the old Stage 4 into Stage 2: the component layout changes
  in Stage 2, so the host-visible identity must change with it or a
  `flint-rocm-session-snapshot-v1` disk cache could be read as the new layout.
- Stage 3 is now disk cache + eviction/guard stress; Stage 4 is the runner position check
  and the claim/cap measurement; Stage 5 is long-context validation.
- Expected headline metric for the Stage 2 gate, from the 0.3.1 baseline at 8k
  (`/tmp/opencode/v2_8k.log`): `cache_snapshot_bytes` 822,684,216 B for a 5-token prefix
  (one full-capacity blob). With valid-rows-only components it should be ~0.5 MB and
  scale with the number of valid rows; `cached_tokens` on the second request must stay 5.

### 2026-11-21 — Stage 2 implemented (adapter rewrite), gate green
- Commit `c559211`. Layout as specified in §3: metadata 64 B, logits `vocab*4`,
  K/V per layer with `row_bytes = num_kv*head_dim*2 = 1024`; 49 layers × 2 =
  98 append-row components, so one token costs 100,352 B across all of them.
- Measured (Q4_K_XL, 1223-token prompt, `gates/cache_reuse.py`):
  - r1 `cache_snapshot_bytes = 123,331,136` = 1223 × 100352 + 600576 + 64 — exactly
    the valid-row layout (all K/V components + logits + metadata); the old blob was
    822,684,216 B at the same prompt.
  - r2 (identical request): `cache=memory`, `cached_tokens=1223`, restore 2.9–3.1 ms
    vs 177 ms for the old blob, answer bit-identical.
  - r3 (multi-turn follow-up): `cached_tokens=1223`, non-empty answer, restore 3.1 ms.
  - dflash2 run: same numbers (restore 4.1 ms), so the restore path survives
    `ReconcileComponentState` + drafter-cycle reset. Serve still logs `draft_proposed=0`
    (known counter gap in DEFERRED.md), so the run does not measure drafter throughput.
- Gate-definition correction: the original r3 (same user turn, appended suffix) can never
  hit a checkpoint — the adapter captures at request boundaries only. Recorded in
  DEFERRED.md; r3 is now the multi-turn follow-up, which is the workload the plan targets.
- Template finding: Flint's pinned chat template has no thinking switch, so the host's
  thinking flags do not change the prompt; the gate forces `</think>` with
  `reasoning_budget_tokens: 0` and compares `reasoning_content + content`. Recorded in
  DEFERRED.md.

### 2026-11-21 — Stage 2 review fixes (`e1d0f6f`)
- **H1 (silent corruption path)**: `CopyRowsIn` had a hard-coded
  `hipMemcpyHostToDevice`. With `BorrowRowsView` returning device memory, the host hands
  that same span back when the source slot is live; on this APU it "works" via host
  mapping and would corrupt on a discrete GPU. Now `hipMemcpyDefault`. This is exactly the
  Stage 3 multi-session case — added as trap 11.
- **H2**: `State::failed` is atomic (concurrent non-overlapping load pieces).
- **H3/H4 (gate false-pass)**: `cache_snapshot_bytes` is a *computed* size and
  `cache=memory` can be served from the live session, so the gate now requires
  `cache_restore_ms > 0` and the exact r1 size
  (`prompt_tokens*row_total + vocab*4 + 64`). r3 is only bounded: its capture is
  chunk-shared (23,482,368 B = a 234-row tail after reusing 1024 rows).
- Bounds/validation: `CopyRowsOut` bounded by `Capacity()` (preservation after a rewind),
  `CreateSlot` asserts `Capacity() == context_`, `Validate` requires
  `logits_valid == (position > 0)`, `As()` compares the adapter pointer, `KvRows()`
  nullptr checks.
- Spec-on leg relabelled: serve reports `draft_proposed=0` because counters are not copied
  on the `SelectNext`/`Advance` path, so the leg proves cache reuse with speculation
  loaded, not drafter throughput.
- Re-verified by the orchestrator: `flint.engine_snapshot`, `flint.target_adapter`,
  `flint.ops` pass; plain gate r2 `cached_tokens=1223`, restore 2.9 ms; spec-on gate
  restore 4.1 ms; both PASSED.

### 2026-11-21 — Stage 3 complete (disk cache + eviction)
- Commits: `b2893cc` (one-line `event=cache_identity` diagnostic) and `4cc90dc` (gate).
- Measured (`gates/cache_reuse.py`, Q4_K_XL):
  - **Disk round trip across a restart**, 1223 tokens at 8k: restarted process
    `cache=disk`, `cached_tokens=1223`, restore 37–46 ms, answer identical to the
    capturing process. At 8880 tokens / 32k: `cache=disk`, `cached_tokens=8880`,
    restore 235 ms, answer identical.
  - **Capture-sum identity**: r1 `894,128,960 = 8880×100352 + 5×600,640` — exact at
    1223 (1 capture), 7268 (4), 8880/9001 (5). `cache_snapshot_bytes` sums captures, and
    extra in-pass captures cost only their private payload.
  - **Follow-up bound**: r3 copies at most one tail chunk per component
    (`≤ 256×row_total + captures×600,640` = 26,290,752 B); measured 21,774,912 B at
    8880 tokens (211 tail rows) and 8,227,392 B at 9001.
  - **Identity**: manifests store `SHA-256(identity)` only. Same dir with `-c 16384`
    → `cache=miss`, full prefill, digest differs from the 8k dir while both manifests have
    100 components. A v1 manifest additionally fails structural validation.
  - **Eviction**, 400 MiB budget, 6 distinct prefixes + repeat: 7/7 status 200, all with
    `cache_captures=1`, ledger flat at ~374 MB, clean exit. At this budget the backing
    reservation leaves no checkpoint headroom, so nothing is retained (documented in
    DEFERRED).
- Correction recorded: a ratio-of-r1 bound for the follow-up capture is prompt-length
  dependent (2.44% at 8880 tokens vs a 2% guess); replaced with the absolute chunk bound
  (trap 16).
