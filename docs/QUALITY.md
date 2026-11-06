# Flint extension quality report

Scope: what is validated, with what evidence, and what remains open. Failed or
inconclusive checks stay visible until they pass.

## Environment

- GPU: AMD Strix Halo (gfx1151), ROCm 7.2.1
- Gufo: main `6cc8a42ca84438a0566ee498a315f35c6190dc8e`, v0.31.0 (extension API series 0.2)
- Build: `gpu-test` preset (`RelWithDebInfo`, HIP on, tests on)

## Passing checks

| Check | What it covers | Evidence |
| --- | --- | --- |
| `flint.config` | Architecture config parsing from GGUF metadata (in-memory containers) | `tests/config_test.cpp` |
| `flint.probe` | Artifact probe: `general.architecture == "flint"` detection, negative cases | `tests/probe_test.cpp` |
| `flint.ops` | Vendored HIP operators vs host reference: `RmsNormRows` (grouped), `SiluScale` | `tests/ops_test.cpp`, gfx1151 device |
| Registration | `gufo diagnose --section extensions` lists `flint` with provider available; in-tree models coexist in the same binary | manual run, 2026-11-04 |
| Link isolation | No symbol collisions with in-tree `qwen38_flash_next`, `gufo_qwen_text`, `gufo_core` (namespaces moved under `gufo::vendor::*`) | link of `gufo` with all extensions included |
| G5 (distribution) | T1.0 p0.95 k20 word distribution vs the pinned llama.cpp oracle, 200 rows/arm on the final binary: same support, same ordering, TV=0.0650 (<= 0.25), no missing rows | `evidence/g5-verdict.log`, 2026-11-05 |
| G8 (concurrency determinism) | Identical greedy output across 1..8 concurrent sessions (314-char stream, 1 distinct) | re-run 2026-11-05 on the final binary: `evidence/g8-full.log` |
| GX (greedy determinism) | Two independent greedy captures bit-identical per binary — the extension and the reference fork each self-check PASS on all five prompts (no run-to-run drift) | `evidence/gx-determinism-and-parity.log` |
| Component-cache reuse | Warm-path output byte-identical to cold path for the same prompt ("What is 2+2?" -> `4`); serve logs show `reuse=ram`, `cached_tokens=1208`, `prefill_tokens=0` at ~2.8x lower latency | `/tmp/opencode/prefix_test.py`, `serve.log` |
| Reasoning split | Flint's XML framing (`<think`/`</think`, the literals its chat template uses) is registered as both `ReasoningBudgetTags` and `ReasoningDelimiters`. The host splits the block: `reasoning_content` holds the thought, `content` the answer, and `reasoning_tokens` counts the block (7/61 tokens across probes) | `/tmp/opencode/reason_full.sh`, `/tmp/opencode/reason_probe.sh` |
| Reasoning budget | `reasoning_budget_tokens=16` forces the close at 16 reasoning tokens then emits the answer; requires `TokenizeControl` to resolve the special-token pair, so it is overridden with `parse_special_tokens` | `/tmp/opencode/reason_budget.sh` |
| Depth curve (full) | Serial cold-prefill over HTTP at 131072 ctx, six depths x 3 runs: 8183 tok -> prefill 488.5 t/s, decode 64.2 t/s; 32748 tok -> 1137.3 t/s, 51.1 t/s; 65265 tok -> 1133.0 t/s, 40.2 t/s; 130808 tok -> 1035.8 t/s, 27.59 t/s (sd 0.24). Shallow depths are TTFT-dominated by the monolithic snapshot copy (see gate notes) | `evidence/depth-q4-summary.json`, `evidence/depth-q4-deep-summary.json`, raw rows in `evidence/depth-q4*.jsonl` |
| API boundary | Series-0.2 component-cache-only continuation; every include lands on the audited `gufo/ext/SURFACE` (the only exception, `gufo/ext/generated/flint_constants.hpp`, is host-generated from `extension.json`); no private `gufo/` includes | `grep` over the tree vs `include/gufo/ext/SURFACE` |
| Sampling defaults | One function sets both `TextRunnerDescriptor::sampling_defaults` and `TextProvider::sampling_defaults` (temp 0.6, top_p 0.95, top_k 20) | `runner.cpp` `FlintSamplingDefaults`, `provider.cpp` registration |

## Gate notes

- **G4 (logit-parity vs recorded baseline)**: 7 of 8 rows pass against the
  Oct 30 recording (`/home/bruno/AI/bench/flint-p8/g4-8.0-q4.json`); each fork
  is a rank-2 near-tie (delta 0.014-0.355). The single failing row
  (`whitespace`, rank 3, delta 0.5978) is a stale-baseline artifact, not an
  extension defect: rerunning the reference fork binary itself against the same
  recording fails the row identically — same rank, same delta 0.5978,
  byte-identical continuation text — because the reference was rebuilt on
  Oct 31, after the baseline was captured. The rebuilt reference also drifts
  further from the recording than the extension does on `tool_xml`. The row
  stays visible; refreshing the baseline from the current reference clears it.
  Evidence: `evidence/g4-ours-vs-baseline.log`,
  `evidence/g4-reference-self-parity.log`.
- **Snapshot cost**: Flint exposes only a monolithic session snapshot, so the
  component-cache adapter copies the live KV (proportional to the frontier) on
  each capture. Measured `cache_snapshot_ms` ~15.2 s at the 131072-token depth
  (13 GB payload); shallow-depth TTFT in the curve is dominated by the same
  copy, so the curve reports prefill/decode throughput and treats TTFT as a
  benchmark cost, not a correctness signal. No per-row KV access exists in the
  Flint engine API to make this incremental.

## Open / not yet validated

- [ ] Re-record the G4/GX baselines from the current reference build (the
      Oct 30 records predate the reference's Oct 31 rebuild; see gate notes).
- [ ] Tokenizer parity gates (G1) against the Strycast in-tree reference.
- [ ] Install into a fresh prefix + runtime closure inventory.
- [ ] Nix flake for reproducible extension builds.

## Artifacts

Evidence logs live in `docs/evidence/` (digests, sha256):

| File | Digest |
| --- | --- |
| `g8-full.log` | `26fe837c42db3d12f4e60d02863e907daeba8028650832e1bae5cb932bf3dc2e` |
| `g5-verdict.log` | `f14aa5d3cea2af37688d4bcc3662f9512ce58fec55c797f1928cb75445d3abd3` |
| `g4-ours-vs-baseline.log` | `bfd04eac7ba59d11c2f199beb546f805e055fcd49bd53b3d4b58846f644c1e8e` |
| `g4-reference-self-parity.log` | `d6ab16e477bf724338078b89cedfe1a2e10a4adc71db0c488ea00d3d887dcaad` |
| `gx-determinism-and-parity.log` | `d42306d04ba65e1f6804fd3c04885755b781ea79ff48c74a37b2d0b50a6820ef` |
| `depth-q4.jsonl` | `fb7ed2be4bb91630b7bfee93a5aeb727cd63ea9fe4ef6e616561edaef05fb204` |
| `depth-q4-deep.jsonl` | `f5b23a1e6dc5f5e0bd743a1b1f925da26b2a0ee6bd56411312eb09ed21e5d3e9` |
| `depth-q4-summary.json` | `1a0808f28cef08e7d656d77ee2a0eb78f5260c6a96b9eb2f0591dbb1728184b1` |
| `depth-q4-deep-summary.json` | `0ac13c27ab8b486ef87afbd1a12637152e74481b2d2922b0092fa8437cebbf47` |

Raw capture JSONs (`g4-serve.json` 11 MB, `g5-serve.json`, `g5-oracle.json`,
`gx-q4.json`) stay out of the repository; digests available on request.
