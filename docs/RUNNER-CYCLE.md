# DFlash2 runner cycle (M2/M3)

The drafter itself (`src/gpu/dflash/`) and the layer-residual capture
(`Session::SetHiddenCapture`) are the extension's; this document covers the
cycle that drives them in `FlintTextRunner::DecodeStep`.

Enable with `--speculative dflash2 --dflash-model <Flint-DFlash2-*.gguf>`.

## Cycle

* Each adapter slot owns a `DflashForward` next to its `Session`, and the
  session records the drafter's taps (`SetHiddenCapture`). The drafter borrows
  the target's embedding and head (`Model::Tied`).
* After every target pass the runner copies the consumed rows from the capture
  ring (`Session::CopyHiddenRows`, at most two device copies) and appends them to
  the drafter's context (`AppendContext`). Only the last `HiddenRingRows() - 1`
  rows of a pass are available (a row is served while `row + ring > position`);
  earlier rows are skipped with `SkipContextTo`.
* One `DecodeStep` is one block: the drafter proposes k tokens after the anchor,
  one verification pass (`AppendAll`) runs the anchor and the proposals, and the
  rows are walked with the request's sampler (on a tentative copy; only the draw
  state is published): a proposal is kept when the sample equals it, otherwise
  the sample is the next token. Each emitted token is a draw from the target
  distribution given its prefix, so output is unchanged: greedy output is
  identical with and without the drafter, on the same build.
* The token that ends a block is sampled but not consumed (`pending`): it is the
  anchor of the next block, so the verification pass consumes it with the
  proposals. `Rewind` + `Session::KeepLogitRow` leave the session at the last
  consumed row, so a checkpoint taken at that point is exact.
* After a continuation-cache restore the drafter keeps no history before the
  restored frontier (`ResetContext` + `SkipContextTo`); its window is shrunk to
  the rows that exist. Acceptance after a restore is within a few points of cold.
* `RuntimeCapabilities::multi_token_decode` must be true, otherwise the
  scheduler drives `Advance` one token at a time and never calls `DecodeStep`.

## Defaults and switches

* Block length: 3 proposals greedy, 2 when sampling (the cost of a verified row
  is ~2.5 ms against ~14 ms for a plain step, and each extra row is accepted less
  often). `FLINT_DRAFT_TOKENS` fixes it.
* No speculation for a request whose prompt or position exceeds 8192 tokens
  (`FLINT_DRAFT_MAX_CONTEXT`): the verification pass reads the KV once per row
  and costs more than it saves at depth.
* `FLINT_DRAFT_REJECTION=1`: rejection sampling against the drafter's own
  distribution; measured no better than the match rule, which is rejection
  sampling for a deterministic draft.
* `FLINT_DRAFT_TIMING=1`: mean ms per round (draft, verification, tail).

## Measured (4 sessions x 262144, UD-Q4_K_XL, drafter Q4_K_M, 300 tokens)

| | no drafter | with drafter |
|---|---|---|
| T=0, mean of 4 prompts | 67 tok/s | 86 |
| T=1, mean of 4 prompts | 69 | 79 |

A round costs ~4.4 ms of drafting and ~20 ms of verification for 3 rows;
proposal acceptance is 40-65 %. Cache restore, cold prefill and four
simultaneous conversations all keep speculation on and give identical greedy
output. Global throughput does not add up across sessions: gufo runs them one
after the other (one stream 83 tok/s, four 84 tok/s in total).
