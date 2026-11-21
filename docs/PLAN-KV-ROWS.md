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
| D10 | Bump `state_abi` → `flint-rocm-session-v2`, `payload_layout` → `flint-kv-rows-v1` in Stage 4 | A v1 single-blob disk cache must not load into a v2 multi-component adapter. |
| D11 | 8 GiB cap stays until §7 is answered by measurement | Cheap refusal guard; removal is a separate, measured change. |

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
| **1** | Engine API: `NumLayers`, `KvRowBytes`, `KvRows`, `Synchronize`, `RestoreRows`, `SetLogits`, `SetRowWriteHook`; refactor `RestoreSnapshot` onto `RestoreRows`; new `flint.engine_snapshot` test | IN PROGRESS | `flint.ops`, `flint.dflash_ops`, `flint.dflash_forward`, `flint.target_adapter`, `flint.engine_snapshot`; spec-off/on smoke | — |
| **2** | Adapter descriptors + private components: metadata/logits capture/restore, `Positions`, `BeginRestore`, `Validate`, `Invalidate`; rows still throw | NOT STARTED | bit-exact logits after restore; prefix reuse | — |
| **3** | Row hooks + guard: `CopyRowsOut`/`CopyRowsIn`/`BorrowRowsView`, engine row-write hook, `BeforeOverwrite`/`BeforeRelease`/`AfterReset` | NOT STARTED | K/V round-trip bit-exact; disk cache write/read; `flint_spec_gate --mode gx` | — |
| **4** | Identity bump + runner: `flint-rocm-session-v2`, `flint-kv-rows-v1`, position check in `ReconcileComponentState`, cap decision by measurement | NOT STARTED | old cache rejected; new round-trip; `--mode memory`; 130k latency A/B | — |
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
- `RestoreRows`: `position <= capacity`; `position == 0` resets position and clears
  `logit_rows`; otherwise syncs the stream, replicates `RestoreSnapshot`'s steps
  (`h_pos[0]=0` + 4-byte `pos` upload, `position`, `logit_rows=1`, `PoolKeyBlocks` over
  `[0, position)` for non-SWA layers when `sparse`, final sync). Must not write
  `h_logits`.
- `SetLogits`: require `row.size() == vocab_size`; `memcpy` into `h_logits`;
  `logit_rows = 1`.
- `SetRowWriteHook`: invoked once in `Append`/`AppendAll`, after the capacity check and
  before the first `Forward`, with `[position, position+tokens.size())`. Never invoked by
  `Rewind`. No hook installed ⇒ no behaviour change.
- `RestoreSnapshot`: replace its tail (logits copy is separate) with `RestoreRows`; keep
  the payload header/version/size checks and the `position > 0` requirement.

Test `flint.engine_snapshot` (GPU, skips with exit 0 without artifact/device, same
pattern as `target_adapter_test`): append a prefix at `-c 512`; snapshot; append more;
restore; assert position, logits bits, and that the next append is bit-identical to a
fresh session that appended the same prefix. Repeat with `RestoreRows` + `SetLogits`
directly (no snapshot) to prove the split path. Run once with sparse on and once off
(`FLINT_SPARSE_ATTN`).

### Stage 2 — adapter descriptors (detail)

Replace the single descriptor with the layout in §3; `CapturePrivatePiece`/
`LoadPrivatePiece` handle only `kPrivateState` (metadata + logits, zero-fill beyond
valid bytes). `BeginRestore` validates one position per descriptor, records `expected`
and coverage, calls `Synchronize`. `Validate` checks coverage per component, parses
metadata, calls `RestoreRows(expected_position)` then `SetLogits`. `Invalidate` calls
`guard.BeforeRelease` over `[0, capacity)` then `Reset` then `AfterReset`. Row hooks
still return an error. `SnapshotBytes`/`PayloadHeader` use is removed from the adapter.

### Stage 3 — rows + guard (detail)

`CopyRowsOut`/`CopyRowsIn` validate range and size, sync, `hipMemcpy` at
`KvRows(layer, is_value) + first*row_bytes`, record coverage, return `Settled`.
`BorrowRowsView` returns `{KvRows + first*row_bytes, (end-first)*row_bytes}` for
`kAppendRows` only. `CreateSlot` installs the hook: for each `kAppendRows` component,
`guard.BeforeOverwrite(id, first, end)`. The slot destructor clears the hook and calls
`BeforeRelease` for every append component over `[0, capacity)`.

### Stage 4 — identity + claim (detail)

`runner.cpp`: `kFlintStateAbi` → `flint-rocm-session-v2`, `ComponentIdentity.payload_layout`
→ `flint-kv-rows-v1`. `ReconcileComponentState` additionally asserts
`flint.session().Position() == position`. `ResourceClaim` unchanged except for the cap
decision from §7; do not add `prompt_state_bytes_per_token` unless the host would
otherwise under-charge (avoid double counting with `per_request_state_bytes`).

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
