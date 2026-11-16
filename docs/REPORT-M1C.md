# M1c report — acceptance against the oracle

## Summary

M1c closed the acceptance question against the llama.cpp oracle. **Verdict A**:
the extension's forced-stream acceptance matches the oracle at `p_min 0`
(pooled mean len 3.05 vs 3.04; every category within 0.04), so the gap the
study saw was the workload, not a bug. The 20-position parity sweep found and
fixed two real bugs — the harness config omitted the sliding window, and the
traced context skipped the rows the ring overwrites — and Level 1 now passes
20/20. Level 2 (Q6_K vs BF16 greedy) is 125/140 (89.3%), below the 95% gate: a
real quantisation effect on the selector codebooks at long context, not
near-ties. Projected speedup at depth 0 is 1.5x on code/JSON and 1.2x on prose.

## What was done

| phase | status | commit | notes |
|---|---|---|---|
| A selector batch test | done | `5bfc314` | guard-tail failure confirmed when allocation shrunk: yes |
| B0 config facts | done | `fc9eeb3` | suspects found: `ConfigJson` `layer_types` (harness-only) |
| B1-B3 workload + oracle | done | `fc9eeb3` | 12 chat + 2 raw prompts; oracle p0 and p0.5 |
| C acceptance on shared workload | done | `3d78abb` | verdict A |
| D diagnostics | skipped (verdict A) | | |
| E 20-position sweep | done | `faad940` | Level 1 20/20; Level 2 125/140 |
| F gate script | done | `8e2b1df` | `tools/drafter_gate.sh` |
| G docs | done | this commit | this report |

## Drafter configuration facts (B0)

| field | extension | source of truth | value there | match |
|---|---|---|---|---|
| `target_layer_ids` | `[5,15,26,36,46]` | GGUF `dflash.target_layers` `[6,16,27,37,47]` (1-based) | `[5,15,26,36,46]` | yes |
| `layer_types` | `"full_attention"` (before E) | GGUF `dflash.attention.sliding_window_pattern` `[1,1,1,1,1]` | `sliding_attention` | **no** — fixed in E |
| `input_embedding_scale` | absent, so 1.0 | HF `dflash_config` | 1.0 | yes |
| `mask_token_id` | 150019 | GGUF `tokenizer.ggml.mask_token_id` | 150019 | yes |
| `rope_theta` | 10000 | GGUF `dflash.rope.freq_base` | 10000 | yes |
| `rms_norm_eps` | 1e-6 | GGUF `dflash.attention.layer_norm_rms_epsilon` | 1e-6 | yes |
| `head_dim` | 128 | GGUF `dflash.attention.key_length` | 128 | yes |
| `block_size` | 8 | GGUF `dflash.block_size` | 8 | yes |

The one mismatch was harness-only: the extension's attention kernel always
applies the 2048 window, which is correct, but `ConfigJson` wrote
`full_attention`, so the reference harness rebuilt the draft model unwindowed
past 2048 rows. E writes `sliding_attention` plus the top-level
`use_sliding_window`/`sliding_window` that `Qwen3Config` needs to honour it.

## Acceptance on the shared workload (BF16 drafter)

| category | oracle p0 acc | oracle p0 mean len | oracle p0.5 acc | oracle p0.5 mean len | ours free acc / mean len | ours stream acc / mean len | delta mean len (stream - oracle p0) | stream divergence |
|---|---|---|---|---|---|---|---|---|
| code | 33.93% | 3.346 | 55.51% | 3.356 | 30.18% / 3.113 | 33.50% / 3.345 | -0.002 | 0.0221 |
| json | 34.14% | 3.342 | 52.87% | 3.453 | 31.75% / 3.223 | 33.98% / 3.379 | +0.037 | 0.0156 |
| prose | 24.50% | 2.693 | 43.30% | 2.558 | 21.15% / 2.481 | 23.71% / 2.660 | -0.033 | 0.0352 |
| continuation | 27.58% | 2.928 | 48.67% | 3.148 | 23.02% / 2.612 | 28.12% / 2.968 | +0.040 | 0.0577 |
| pooled chat | 29.47% | 3.041 | 49.89% | 3.095 | 25.77% / 2.804 | 29.31% / 3.051 | +0.010 | 0.0291 |

Acceptance = sum accepted / sum drafted; mean len = 1 + sum accepted / sum
cycles (pooled). Stream divergence = mean over prompts of the per-prompt
`stream_divergence` (how often the target's own argmax disagrees with the
oracle's stream).

Per-row acceptance curve (ours, stream, pooled chat): row 0 69.2%, row 1
45.6%, row 2 30.4%, row 3 22.8%, row 4 16.8%, row 5 13.4%, row 6 10.8%.

Q6_K vs BF16 (stream): acceptance +0.12 points pooled (29.43% vs 29.31%), mean
len +0.009 (3.060 vs 3.051). Per category the mean-len delta is at most +0.06
(json).

## Verdict

**A (workload, not a bug).** C5 thresholds: pooled mean len >= oracle - 0.25,
every chat category >= oracle - 0.40, pooled acceptance within 3 points.
Forced stream, BF16, vs oracle p0: pooled mean len 3.051 vs 3.041 (+0.010);
code -0.002, json +0.037, prose -0.033, all inside -0.40; pooled acceptance
29.31% vs 29.47% (0.16 points). Phase D was skipped.

## Diagnostics (if D ran)

Skipped: verdict A.

## Parity sweep (E)

| position | BF16 worst stage | value | Level 1 | Q6_K==BF16 greedy (of 7) |
|---|---|---|---|---|
| 64 | packed.layer.1.ffn_down | 4.989e-06 | PASS | 6 |
| 128 | packed.layer.1.v | 6.117e-06 | PASS | 7 |
| 256 | packed.layer.4.v | 4.240e-06 | PASS | 5 |
| 512 | packed.layer.0.k | 7.169e-06 | PASS | 7 |
| 1000 | packed.layer.0.k | 1.415e-05 | PASS | 7 |
| 1024 | packed.layer.0.k | 1.400e-05 | PASS | 1 |
| 1500 | packed.layer.0.k | 2.355e-05 | PASS | 7 |
| 2000 | packed.layer.0.k | 2.838e-05 | PASS | 7 |
| 2046 | packed.layer.0.k | 2.824e-05 | PASS | 7 |
| 2047 | packed.layer.0.k | 2.796e-05 | PASS | 7 |
| 2048 | packed.layer.0.k | 2.829e-05 | PASS | 7 |
| 2049 | packed.layer.0.k | 2.800e-05 | PASS | 3 |
| 2050 | packed.layer.0.k | 2.837e-05 | PASS | 7 |
| 2100 | packed.layer.0.k | 2.935e-05 | PASS | 7 |
| 2560 | packed.layer.0.k | 2.969e-05 | PASS | 7 |
| 3000 | packed.layer.0.k | 4.988e-05 | PASS | 6 |
| 3500 | packed.layer.0.k | 5.061e-05 | PASS | 6 |
| 4000 | packed.layer.0.k | 5.615e-05 | PASS | 7 |
| 4096 | packed.layer.0.k | 5.672e-05 | PASS | 7 |
| 4500 | packed.layer.0.k | 6.027e-05 | PASS | 7 |

Level 1: 20/20 positions pass. Level 2: 125/140 proposals identical (89.3%).

Level 2 is below the 95% gate and is reported as a failure. The disagreements
are real, not near-ties: the Q6_K selector codebooks (256x150144) are quantised,
and at long context the selector distribution flattens enough to flip the
argmax (pos-1024 row 3: BF16 top-1 0.93 vs Q6_K 0.39/0.38; pos-64 row 6: 0.97
vs a different token). Only three of the fifteen differing rows are near-ties
(top-1 minus top-2 below 0.05). Forced-stream acceptance is unaffected
(Level 3), because the target verifies every proposal.

## Projected speedup

`mean_len / verify ratio`, using the ours-stream BF16 mean len (code 3.345,
json 3.379, prose 2.660) and the measured width-8 verify ratios 2.21 / 2.48 /
3.16 / 4.25 at 0 / 8k / 32k / 131k:

| depth | verify ratio | code | json | prose |
|---|---|---|---|---|
| 0 | 2.21 | 1.51x | 1.53x | 1.20x |
| 8k | 2.48 | 1.35x | 1.36x | 1.07x |
| 32k | 3.16 | 1.06x | 1.07x | 0.84x |
| 131k | 4.25 | 0.79x | 0.79x | 0.63x |

Past 32k the verify ratio outruns the mean len, so a fixed 7-proposal block is
a net slowdown; the adaptive controller must draft fewer rows there (M4).

## Gate

`tools/drafter_gate.sh full` -> `summary.md`:

```
# Drafter gate, tier `full`

Results: `/tmp/opencode/drafter-gate`

- Level 1 (BF16 vs upstream): PASS, 20/20 positions pass every stage gate
- Level 2 (Q6_K vs BF16 greedy): FAIL, 125/140 (89.3%) greedy proposals identical
- Level 3 (acceptance vs oracle): PASS, pooled mean len BF16 3.051 / Q6_K 3.060 vs oracle 3.041

Overall: FAIL
```

## Open issues and deviations from the plan

- **Level 2 fails (89.3% < 95%).** Reported as a failure; no threshold was
  changed. It is a Q6_K selector-codebook quantisation effect at long context,
  not a wiring bug: Level 1 (BF16 vs upstream) passes 20/20 and Level 3
  (acceptance) passes for both quants. The shipped artifact is Q6_K, so this is
  a property of the artifact, not of the extension.
- **`p_min 0.5` not reproduced.** The extension's greedy selector confidences
  are always 1.0 (`TestSelectorGreedy`), so llama.cpp's per-proposal `p_min`
  truncation cannot be reproduced without kernel changes. The comparison is at
  `p_min 0`; the oracle's `p_min 0.5` numbers are recorded for reference only.
- **Level 2 head route.** The plan says native head; the sweep runs
  `--head reference` for Level 1 and the gate copies `greedy.json` to
  `greedy-native.json`. The head route does not change the selector's greedy
  proposals (byte-identical for both quants), so the two are the same file.
- **The study's "mean len 4.23 code" is not reproduced.** The oracle's mean len
  on this workload is ~3.35 code at both `p_min` values; the 4.23 figure came
  from a different workload.

## What's next

M2 (target adapter) and M3 (runner cycle) from `docs/PLAN-DFLASH2.md`. The
measured mean len (3.05 pooled, 3.35 code) gives a projected 1.5x on code/JSON
at depth 0 and 1.35x at 8k, falling below 1x past 32k. That is above the 1.3x
go/no-go bar on code/JSON at shallow depth and below it on prose and at long
context, so M4's adaptive draft length is required before the win is real at
depth. The runner cycle (M3) is the remaining piece of Level 3: it exercises
the sampler-as-black-box commit path rather than this driver's argmax.
