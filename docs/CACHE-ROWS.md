# Continuation cache: row components instead of one private blob

Status: design, not implemented. Branch `feat/swa-ring-kv`.

## Problem (measured 2026-11-16, 4 sessions x 262144, cache RAM 8 GiB)

`FlintContinuationAdapter` declares one `kPrivateState` component whose
`state_bytes` is the largest possible snapshot (~7.5 GB at 262144 tokens).

- Every captured checkpoint costs ~5 s (`cache_snapshot_ms` ~5000) and 7.5 GB of
  cache, whatever the real prefix length (37k tokens measured).
- One checkpoint fills an 8 GiB cache: with 4 sessions only one conversation is
  cached at a time.
- A private state is exact: a request whose common prefix is shorter than the
  last checkpoint gets `no_compatible_boundary` and prefills everything again
  (20 s TTFT at 37k tokens, against 0.45 s when it continues the checkpoint).

## Target

Checkpoints sized by what they hold, shared between checkpoints, reusable from
any earlier checkpoint boundary.

| Component | Kind | Count | `row_bytes` | `rows_per_chunk` |
|---|---|---|---|---|
| K of a global layer | `kAppendRows` | 1 per global layer | `kv_row * 2` (f16) | 256 |
| V of a global layer | `kAppendRows` | 1 per global layer | `kv_row * 2` | 256 |
| K of a sliding-window layer | `kWindowRows` | 1 per SWA layer | `kv_row * 2` | 256 |
| V of a sliding-window layer | `kWindowRows` | 1 per SWA layer | `kv_row * 2` | 256 |
| last logits | `kPrivateState` | 1 | - (`state_bytes = vocab * 4`) | - |

`window_rows` of a `kWindowRows` component = the model's `sliding_window`.
`kv_row = num_kv * head_dim`. The reference layout is gufo's
`qwen38_flash_next/continuation_layout.cpp` (per-layer K and V components,
`rows_per_chunk = 256`, private logits).

Why this fixes the three points above:

- Rows are chunked (256) and shared between checkpoints: a checkpoint at n costs
  the new chunks plus a ~0.6 MB private state, not the whole KV.
- A window component keeps only the chunks from the one containing
  `n - window_rows`.
- Any earlier checkpoint is a valid restore point: rows `[0, p)` come from the
  shared chunks, the logits from that checkpoint's private state. Checkpoints
  sit on the prefill boundaries (`PlanPrefill`, `prefill_tile` = 4096) plus the
  prompt end, so a prefix that diverges at 37002 restores from 36864 and
  prefills 138 tokens.

## Ring compatibility

The window ring holds `window + prefill_tile` rows (`RingLength`); gufo needs at
least `window_rows + rows_per_chunk - 1`. `prefill_tile` (4096) >> 255, so the
current ring already satisfies the contract. Row numbers are absolute; the ring
maps `row % ring_len` (`CopyRows` already does this for `SaveSnapshot`).
Before an eviction of a ring row, `MutationGuard::BeforeOverwrite(id, first, end)`
must be called (checkpoints may borrow it): this is the part that does not
exist yet, since today nothing borrows rows.

## Engine API to add (`src/gpu/engine.hpp`, `Session`)

- `CopyRowsToHost(layer, is_value, first, end, dst, stream)` and
  `CopyRowsToDevice(...)`: absolute row range, ring-aware, async on the cache
  stream (`CopyRows` + `hipMemcpyAsync`).
- `Logits()` is already readable; add `SetLogits(span)` and
  `SetPosition(position)` (position, `logit_rows = 1`, `ring_broken = false`,
  rebuild of the block-sparse key bounds when `sparse`, as the tail of
  `RestoreSnapshot` does).
- Keep `SaveSnapshot`/`RestoreSnapshot` and `FLINT_SNAPSHOT_SELFTEST`: they are
  the bit-exact oracle for the new path.

## Adapter (`continuation_adapter.*`)

Replace the single component by the table above, built by a
`ContinuationLayout` (pure host, unit-testable without a GPU). `Positions()`
reports `position` for every row component and for the private one.
`BeginRestore` records the expected positions and guards ranges;
`CopyRowsIn` accepts only retained rows and records coverage; `Validate`
checks complete coverage, then `SetLogits` + `SetPosition`. `BorrowRowsView`
stays empty (rows live in device memory).

## Tests

1. Host-only (CPU build, `engine_stub.cpp`): layout geometry (ids, row sizes,
   window rows, chunk arithmetic) and `Positions`/`PlanPrefill` invariants.
2. GPU, bit-exact: extend `FLINT_SNAPSHOT_SELFTEST` so a restore through the
   row components equals `SaveSnapshot`/`RestoreSnapshot` logits, across a
   wrapped ring and from a checkpoint shorter than the live position.
3. Serving: the 37k-token script (`T1` then two continuations, then a request
   that diverges at 37002): expect `cache_hit`, a small `cache_snapshot_ms`,
   restore from 36864; 4 concurrent conversations all cached with an 8 GiB cache.

## Out of scope / later

- A DFlash2 drafter (issue #1) adds its own KV as more components of the same
  layout; the layout is built so that components can be appended.
- Disk persistence (`persistent_encoding`) is declared true today; keep it
  working or turn it off explicitly if the row layout makes it non-trivial.
