# Matrix-core decode attention for the global layers

`AttentionSliceWmmaKernel` (`src/gpu/kernels/kernels.hip.cpp`) replaces the
scalar slice kernels for a global layer (no window, `head_dim` 128, a group of 8
query heads per KV head) when the pass carries 1 to 8 rows: plain decode and the
verification pass of speculative decoding.

It is **off by default** and chosen at build time:

```sh
FLINT_MATRIX_ATTENTION=1 gufo extension install <source>
```

(or `-DFLINT_MATRIX_ATTENTION=ON` in a CMake build). Such a build writes its
continuation cache as `flint-kv-rows-window-v1-matrix`, so it never restores
rows a default build computed, nor the reverse. A default build runs the scalar
kernels with 32 key slices.

## Why

Plain decode attention was already bound by memory, not by the arithmetic: at 131k
the scalar kernel reads the KV at ~190 GB/s (the matrix-core kernel: ~200). What
the scalar kernels could not do is carry several rows: each row re-read the KV,
and the score pass is a scalar dot product per key, so a width-8 verify pass cost
8.2 ms per global layer at 131k against 1.4 ms for one row. That is what held
speculation below 1x past ~8k of context.

## Design

* One block per (kv head, key slice, group of 8 rows), 8 waves, 64 slices per
  layer (`kGlobalSplits`). Slice `s` owns the 32-key blocks `b % 64 == s`, a
  partition that depends on neither the position nor the batch width.
* The 8 waves share every K/V tile, staged once in LDS: a width-8 pass reads the
  global KV once (1.69 ms at 131k, 0.50 ms at 32k).
* A wave owns one row: its 8 query heads fill columns 0-7 of the 16x16 tile (the
  other columns idle: the pass is memory-bound). It computes S^T = K Q^T in
  registers, so the softmax needs no cross-lane traffic beyond one max, and
  accumulates O^T = V^T P^T. The partial (numerator, natural-log maximum,
  denominator) is the format `AttentionMergeKernel` already merges, with the sink.
* The query is rounded to F16 and so are the probabilities, as in the prefill
  kernel: ~1e-4 relative difference to the scalar kernels.

## Row invariance, and the V leak

A row must come out bit-identical whether it travels alone or in a block, or the
verification pass is not the decode pass. Measured on this hardware, the V of a
key that a row masks (weight exactly 0) still moves that row's output by an ulp
of the accumulator; K of the same key does not. A shared tile holds, past a row's
position, whatever another row needs or whatever the stale cache holds, so rows
would differ with the shape of the batch.

The kernel therefore zeroes, when a wave reads V, the keys past its own row's
position (`boundary` tiles only; a tile wholly past the row is skipped). A wave
then sees exactly what a lone row sees. Staging keys one pair further, tried
first, fixed a synthetic test and failed on the real engine, whose cache holds
stale rows.

`flint_attn_bench` (`tools/attn_bench.cpp`) runs the check on synthetic KV (64
rows per depth against a pair and a lone row) and the timings below. The server
runs none of it.

## Measured (global layer, F16 KV, 4 KV heads, 268 MB at 131k)

| depth | rows | scalar | matrix core |
|---|---|---|---|
| 8k | 8 | 0.48 ms | 0.15 ms |
| 32k | 8 | 2.04 ms | 0.50 ms |
| 131k | 1 | 1.43 ms | 1.31 ms |
| 131k | 8 | 8.18 ms | 1.69 ms |

End to end, 4 sessions x 262144, drafter on, T=1, decode tok/s:

| context | scalar kernels | matrix-core kernel |
|---|---|---|
| 2k | 88 | 98 |
| 9k | 62 (drafter off: slower with it) | 75 |
| 36k | 46 | 69 |
| 128k | 25 | 43.5 |

Greedy output with the drafter is identical to the output without it, on the
real engine.

## Limits (to be read before merging)

* **Not bit-exact against the scalar kernels.** The query and the probabilities
  are rounded to F16, so outputs differ by ~1e-4 relative (max 1.8e-4 at 8k,
  ~1e-4 at 32k). Greedy outputs therefore change against the scalar path; a
  default build runs the scalar path and matches `main`.
  What does hold is row invariance (a row in a block of 64 equals the row alone)
  and drafter-on == drafter-off with this kernel.
* A matrix-core build runs `kGlobalSplits` 64 instead of 32 for the global
  layers. That alone changes the greedy output at ~28k tokens, even on the
  scalar path.
* Measured on gfx1151 at 0 / ~6k / ~29k tokens, plain one-row decode is the same
  as the scalar kernel (within 2 %); the gain is for the multi-row verification
  pass of speculative decoding.
