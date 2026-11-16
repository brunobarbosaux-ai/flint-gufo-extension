# Progress — DFlash2 integration (M2-M6)

Plan: `docs/PLAN-DFLASH2.md`. M1 (the drafter) is done; see `docs/REPORT-M1C.md`.

## Milestone status

| Milestone | Title | Status | Commit | Notes |
|---|---|---|---|---|
| M2 | Target adapter | COMPLETE | 50d9189 | `src/spec/target_adapter.{hpp,cpp}`; `flint.target_adapter` passes (rows bit-identical widths 1..8, features bit-identical, rewind matches a fresh session) |
| M3 | Cycle + runner integration | IN PROGRESS | | `src/spec/cycle.{hpp,cpp}`, `FLINT_SPEC=dflash\|off\|null`, GX + null-drafter gates |
| M4 | Draft policy | NOT STARTED | | adaptive length priced by measured per-row cost |
| M5 | Gates | NOT STARTED | | GX, null drafter, settings honour, quant, depth curve, no regression, cost, memory, concurrency |
| M6 | Release | NOT STARTED | | QUALITY.md, extension.json 0.3.0, registry PR, llama-swap, issue #1 |

## M2 findings (carried into M3)

- `Session::Rewind` does not restore logits; the cycle re-appends the anchor before reading `Logits()`.
- A multi-token prefill is not bit-identical to single-token decode, so the cycle's exactness argument (§3.3) holds only for rows written by the decode or verify path.
- `flint.config` / `flint.probe` fail to compile in `tests/gguf_test_shim.hpp` (pre-existing SDK drift, unrelated to M2).

## Design note (deviation from the plan)

PLAN-DFLASH2.md §3.1 says there is "no extension-facing speculative hook". The host does expose `TextRunnerCapabilities.multi_token_decode` + a virtual `DecodeStep` (Qwen3.8-Flash-Next uses it). M3 still follows the plan's `SelectNext`/`Advance` design, because it keeps the sampler's history updated between rows (the `DecodeStep` path would require the runner to manage sampler history itself). To be documented in the plan.

## Log

- M2 delegated, verified (`flint.target_adapter` passes), committed `50d9189`.
- M3 delegated.
