# Flint extension quality report

Scope: what is validated, with what evidence, and what remains open. Failed or
inconclusive checks stay visible until they pass.

## Environment

- GPU: AMD Strix Halo (gfx1151), ROCm 7.2.1
- Gufo: main `186a507da77bcbb3e8f0cfbd49f688f88535e9e2`
- Build: `gpu-test` preset (`RelWithDebInfo`, HIP on, tests on)

## Passing checks

| Check | What it covers | Evidence |
| --- | --- | --- |
| `flint.config` | Architecture config parsing from GGUF metadata (in-memory containers) | `tests/config_test.cpp` |
| `flint.probe` | Artifact probe: `general.architecture == "flint"` detection, negative cases | `tests/probe_test.cpp` |
| `flint.ops` | Vendored HIP operators vs host reference: `RmsNormRows` (grouped), `SiluScale` | `tests/ops_test.cpp`, gfx1151 device |
| Registration | `gufo diagnose --section extensions` lists `flint` with provider available; in-tree models coexist in the same binary | manual run, 2026-11-04 |
| Link isolation | No symbol collisions with in-tree `qwen38_flash_next`, `gufo_qwen_text`, `gufo_core` (namespaces moved under `gufo::vendor::*`) | link of `gufo` with all extensions included |

## Open / not yet validated

- [ ] Tokenizer parity gates (G1) against the Strycast in-tree reference — pending
      gate run.
- [ ] Full parity/quality gates (G4/G5/G8/GX: logits parity, generation quality,
      MoE routing, MTP) — pending; harness is Strycast `run_gates.sh`, to be
      vendored into this repository.
- [ ] End-to-end prompt/chat/bench drivers on real weights.
- [ ] HTTP serving contract (`gufo serve` with the flint runner).
- [ ] Session isolation, streaming, cancellation/recovery.
- [ ] Install into a fresh prefix + runtime closure inventory.
- [ ] `api_version` floor `0.1.19` re-derivation from upstream change history
      (established from a shallow Gufo clone; see README).

## Artifacts

Validation artifact and gate outputs to be recorded here as gates run, with
immutable digests.
