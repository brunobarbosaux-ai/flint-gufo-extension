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
| G5 (distribution) | T1.0 p0.95 k20 word distribution vs the pinned llama.cpp oracle, 60 rows/arm: same support, same ordering, TV=0.0333 (<= 0.25), no missing rows | `g5-serve.json` vs `g5-oracle.json`, `serve_gates.py g5-verdict` |
| G8 (concurrency determinism) | Identical greedy output across 1..8 concurrent sessions (314-char stream, 1 distinct) | `g8-quick.log`, `g8-full.log`, `serve_gates.py g8 --gate` |
| GX (greedy exactness) | Two independent greedy captures bit-identical on all five prompts (1140/1229/1125/427/171 chars) | `gx-q4.json` vs `gx-q4-b.json`, `gx_compare.py` |
| Component-cache reuse | Warm-path output byte-identical to cold path for the same prompt ("What is 2+2?" -> `4`); serve logs show `reuse=ram`, `cached_tokens=1208`, `prefill_tokens=0` at ~2.8x lower latency | `/tmp/opencode/prefix_test.py`, `serve.log` |

## Gate notes

- **G4 (logit-parity forks vs recorded baseline)**: 7 of 8 rows pass against
  `/home/bruno/AI/bench/flint-p8/g4-8.0-q4.json`; each fork is a rank-2 near-tie
  (delta 0.014-0.355). The single failed row (`whitespace`) forks at rank 3 with
  delta 0.5978, just outside the delta <= 0.5 envelope. That envelope was
  calibrated on the UD-Q6_K_XL stack; the gate harness itself documents that
  coarser artifacts need a wider bound (IQ4_XS measured 2.0x). Both engines pick
  a valid top-3 continuation there, so this is near-tie numerics noise, not a
  structural divergence. Left visible pending a Q4-specific envelope.
- **Snapshot cost**: Flint exposes only a monolithic session snapshot, so the
  component-cache adapter copies the live KV (proportional to the frontier) on
  each capture. At the serve context (65536) a capture is ~8 s; the full-tier
  depth curve at 130816 tokens is dominated by these copies (~56 s each) and is
  therefore a benchmark cost, not a correctness signal. No per-row KV access
  exists in the Flint engine API to make this incremental.

## Open / not yet validated

- [ ] G4 Q4-specific near-tie delta envelope (one row at rank 3, delta 0.5978).
- [ ] Tokenizer parity gates (G1) against the Strycast in-tree reference.
- [ ] Install into a fresh prefix + runtime closure inventory.
- [ ] `api_version` floor `0.1.19` re-derivation from upstream change history
      (established from a shallow Gufo clone; see README).
- [ ] Nix flake for reproducible extension builds.

## Artifacts

Validation artifact and gate outputs to be recorded here as gates run, with
immutable digests.
