# M1c: close the acceptance gap against the oracle

> **Status: complete (2026-11-15).** All phases A-G done; verdict A. Results in
> `docs/REPORT-M1C.md`. The phase table below is the execution record.

This plan can be executed by a model that has not seen the earlier conversation.
Everything it needs is in this file or in files it names. Read all of it before
starting, including §0, "Rules".

## Why this milestone exists

M1b proved the drafter's block forward is numerically correct. On the BF16
drafter, every stage agrees with the pinned upstream PyTorch reference to
4.7e-6 relative RMSE, against a 1e-4 gate. The offline acceptance it then
measured is much lower than the oracle's:

| | oracle (llama.cpp b11804) | extension (M1b, `--accept`) |
|---|---|---|
| headline | 51.1% acceptance at `--draft-p-min 0.5` (1506/2947) | 30.7% on C++ source, 17.6% on markdown |
| drafts per cycle | 3.69 (`p_min` truncates blocks) | always 7 |
| `p_min 0` acceptance | 33.5% | (only `p_min 0` was measured) |
| mean len (accepted + 1 per cycle) | code 4.23, JSON 4.61, prose 3.16 | code-like 3.15, prose-like 2.23 |

`p_min` changes how many tokens are drafted, so the headline 51.1% is not a
metric we can compare against. **Mean len** (tokens emitted per verify cycle)
is: `p_min` can only shorten a block, never add accepted tokens. On mean len we
are about **1 token per cycle behind**. That difference decides the project's
outcome. At the extension's 2.21 verify-cost ratio, the oracle's mean len
projects to about 1.9x on code; ours projects to about 1.4x.

There are two candidate explanations, and they call for opposite actions:

- **A. Different workload.** The oracle ran 12 chat prompts through the chat
  template (the data DFlash drafters are trained on) and stopped naturally at
  192 tokens. M1b continued two of our own files from the middle, with no
  template, for 512 tokens. If this is the cause, nothing is broken.
- **B. A drafter-input bug the parity test cannot see.** The parity harness
  rebuilds the drafter from a `config.json` the extension writes, and reads
  target features the extension captured. Wrong tap layers, a feature/anchor
  offset, or a wrong `mask_token_id` / embedding scale / sliding-window flag
  would be wrong on both sides, so parity still passes and acceptance silently
  drops.

M1c separates A from B on a shared, identical workload, then hardens the
parity and acceptance gates so later work (M2/M3, the runner cycle) cannot
regress either one unnoticed.

## Deliverable

When every phase is done or explicitly stopped, write
`docs/REPORT-M1C.md` (template in §9) and post the same summary in chat. The
summary covers: what was done, the measured results, the A/B verdict, and
what to do next.

---

## 0. Rules (read before touching anything)

1. **Extension only.** Edit files under
   `/home/bruno/Projects/flint-gufo-extension` only. Never modify the Gufo tree
   (`/home/bruno/AI/Gufo`), Strycast, or `/home/bruno/AI/Gufo-mtp-perf`. If the
   reference harness `tools/qwen27b/dflash_reference.py` in Gufo-mtp-perf needs
   a change, stop and ask the user. Do not fork it silently.
2. **Do not loosen a gate to make it pass.** The 1e-4 stage gate, the 5e-6
   selector gate and the thresholds defined in this plan are fixed. If one
   fails, investigate and report. Never edit a threshold.
3. **One GPU model process at a time.** The llama.cpp oracle server and the
   extension's tools each load a 25 GB target. Stop the server (and check with
   `pgrep -f llama-server`) before running an extension tool, and the other way
   round.
4. **Read before editing.** Follow the existing code style: comments explain
   why, `[[nodiscard]]`, `std::string* error_msg` error reporting, no
   exceptions in library code (the tools may throw, as `drafter_parity.cpp`
   already does).
5. **Keep tests green.** After every C++ change, run the test command in §1.
   `flint.config` and `flint.probe` report "Not Run" in this build. That is
   expected and not a failure.
6. **Commit per phase**, using a conventional-commit subject, for example
   `feat(dflash): ...`, `test(dflash): ...` or `docs(plan): ...`. The body
   gives the measured numbers. Push to `origin main` after each phase whose
   tests pass. Never commit trace directories or model files.
7. **Timeboxes are hard.** If a phase runs past its timebox, stop. Write what
   you have into the report and ask the user how to continue.
8. **Ask before** downloading or building anything large (more than ~1 GB, or
   a llama.cpp build), and before any destructive git operation.
9. Ignore the nix warnings `ignoring untrusted flake configuration setting`.
   They are harmless.

## 1. Environment and commands

| what | where |
|---|---|
| extension repo | `/home/bruno/Projects/flint-gufo-extension` |
| Gufo tree (build host) | `/home/bruno/AI/Gufo` |
| build dir | `/home/bruno/AI/Gufo/build/host-test` (already configured with this extension) |
| tool binaries | `/home/bruno/AI/Gufo/build/host-test/extensions/build/flint/build/` |
| target model | `/home/bruno/AI/GGUF/flint-UD-Q4_K_XL.gguf` (Q8_0 `token_embd` / `output`) |
| drafters | `/home/bruno/AI/GGUF/mtp/Flint-DFlash2-{BF16,Q6_K}.gguf` |
| oracle binary | `/home/bruno/AI/llama/llama-b11804-rocm/llama-server` |
| reference harness | `/home/bruno/AI/Gufo-mtp-perf/tools/qwen27b/dflash_reference.py` (run inside `nix develop` in that directory: torch lives there) |
| upstream DFlash model | `/tmp/opencode/dflash-upstream/dflash/model.py` |
| scratch space | `/tmp/opencode/` |

If `/tmp/opencode/dflash-upstream` is missing, recreate it:

```sh
cd /tmp/opencode && git clone --quiet --filter=blob:none https://github.com/z-lab/dflash dflash-upstream
cd dflash-upstream && git checkout --quiet 07ebd93db9f472af339b644bb70221ad8428328a
sha256sum dflash/model.py   # must be f55b7fe0a4c0b3073e0f9cdce547cce29f4b8e2168c4d2818760007c43b7651e
```

Build and test. Run these from `/home/bruno/AI/Gufo`. Builds take minutes, so
use long timeouts.

```sh
# reconfigure only after editing extension.json / extension.cmake
nix develop -c cmake build/host-test
nix develop -c cmake --build build/host-test --target flint_drafter_parity flint_dflash_ops_test flint_dflash_forward_test flint_ops_test --parallel 16
nix develop -c ctest --test-dir build/host-test -R "flint" 2>&1 | grep -E "Passed|Failed|Not Run"
```

Existing parity run, for reference (this is what M1b ran):

```sh
B=/home/bruno/AI/Gufo/build/host-test/extensions/build/flint/build
cd /home/bruno/AI/Gufo && nix develop -c $B/flint_drafter_parity \
  /home/bruno/AI/GGUF/flint-UD-Q4_K_XL.gguf /home/bruno/AI/GGUF/mtp/Flint-DFlash2-BF16.gguf \
  --out /tmp/opencode/trace-bf16-ref --head reference
cd /home/bruno/AI/Gufo-mtp-perf && nix develop -c python3 tools/qwen27b/dflash_reference.py \
  --upstream /tmp/opencode/dflash-upstream/dflash/model.py \
  --config /tmp/opencode/trace-bf16-ref/config.json \
  --target /home/bruno/AI/GGUF/flint-UD-Q4_K_XL.gguf \
  --draft /home/bruno/AI/GGUF/mtp/Flint-DFlash2-BF16.gguf \
  --trace /tmp/opencode/trace-bf16-ref --output /tmp/opencode/trace-bf16-ref/report.json
```

Files you will work in:

- `tools/drafter_parity.cpp`: the parity and acceptance driver
  (`flint_drafter_parity`). `MeasureAcceptance()` is the M1b acceptance loop;
  `ConfigJson()` writes the harness config.
- `src/gpu/dflash/dflash_forward.{hpp,cpp}`: `DflashForward` (`Create`,
  `AppendContext`, `Run`, `ContextLength`).
- `src/gpu/dflash/dflash_kernels.{hpp,hip.cpp}`: vendored kernels, including
  `LaunchDFlashSelectorBatch` and `DFlashSelectorScratchElements`.
- `src/gpu/dflash/dflash_weights.cpp`: the loader. GGUF `dflash.target_layers`
  `[6,16,27,37,47]` are layer inputs; the loader subtracts one to get the
  output taps `[5,15,26,36,46]`.
- `src/gpu/engine.hpp`: `Session` (`Append`, `AppendAll`, `Logits(row)`,
  `Rewind`, `SetHiddenCapture`, `CapturedHidden(row, tap)`). The hidden
  capture ring keeps only the last `kHiddenRingRows = 2048` positions.
- `src/chat_template.hpp`: `RenderPrompt(messages, {enable_thinking})`, Flint's
  chat template. It lives in library `gufo_flint`, which
  `flint_drafter_parity` already links.
- `tests/dflash_ops_test.cpp`: kernel tests. `TestSelectorGreedy()` covers
  only the single-row `LaunchDFlashSelectorStep`.
- `docs/PLAN-DFLASH2.md`, `docs/ROADMAP-DFLASH2.md`: the project plan and
  history. Update them in Phase G.

---

## Phase order and decision points

```
A  selector batch regression test            (0.5 d)  always          done (5bfc314)
B  facts + shared workload + oracle baseline (1 d)    always          done (fc9eeb3)
C  extension on the same workload, compare   (1 d)    always   -> A   done (3d78abb), verdict A
D  input diagnostics                          (1.5 d)  ONLY if B      skipped (verdict A)
E  20-position parity sweep (Levels 1 and 2)  (1 d)    always          done (faad940)
F  one-command gate script                    (0.5 d)  always          done (8e2b1df)
G  docs + final report                        (0.5 d)  always          done (27d91f2)
```

Why this order: A is cheap and removes a GPU-hang risk before many long runs.
B and C answer the question that decides the project. D runs only when it is
needed. E depends on a fact B establishes (whether the drafter uses a sliding
window), and on D's fixes if D ran. F wraps what A–E built, and G records it.

---

## Phase A — Selector batch regression test (timebox 0.5 d)

**Problem.** M1b found that the selector's partial top-k scratch was sized for
one row, while `LaunchDFlashSelectorBatch` gives every proposal row its own
`DFlashSelectorScratchElements(vocab)` slice. At the real vocabulary that is an
out-of-bounds write, which shows up as a GPU memory fault or hang. The fix is
in `dflash_forward.cpp` (allocation is `rows * selector_scratch`), but no test
covers the batch entry point.

**Task.** Add `TestSelectorBatch()` to `tests/dflash_ops_test.cpp` and call it
from `main()` next to `TestSelectorGreedy()`.

- Reuse the `TestSelectorGreedy` fixture style: `kVocab = 5000` (three partial
  blocks), small rank and top-k (4/4), BF16-exact codebook values from the
  `ExactBf16` / `Bf16Value` helpers.
- **Two greedy sequences** (temperature 0) with counts 3 and 2, so 5 proposal
  rows in total. Give each row its own logits (`5 * kVocab` floats, a distinct
  planted top-k per row, `logits_stride = 0`).
- Token buffer layout, from the header comment: each sequence's tokens are its
  **anchor followed by its proposal slots**, so the buffer holds
  `rows + sequences = 7` entries. Put anchors at index 0 (seq 0) and index 4
  (seq 1).
- Read the kernel (`dflash_selector_finalize_batch_kernel` and the launcher in
  `dflash_kernels.hip.cpp`) and confirm the predecessor rule: step 0 uses the
  anchor, and each later step uses the previous chosen token. Then write a
  host reference that walks the same chain, using a transition score
  `sum_r pred[prev][r] * projected[row][r] * succ[cand][r]` over each row's
  top-k candidates. `projected_hidden` is per row, `rows * rank` floats.
- Allocate `partial_scores` and `partial_ids` as exactly
  `rows * DFlashSelectorScratchElements(kVocab)` elements, **plus a guard tail**
  of 1024 elements filled with a sentinel (for example `NaN`-payload floats /
  `0xDEADBEEF` ids). After the launch, check the guard is untouched.
- Check per row: the chosen token equals the host reference, the confidence is
  1, the candidate set equals the host top-k, and the probabilities are one-hot.
- Make sure the fixture actually exercises the transitions: at least one row's
  winner must differ from its logits argmax (the existing test enforces this
  for its single row).

**Done when** `flint.dflash_ops` passes. Then temporarily shrink the
allocation to one row's scratch and confirm the test fails (the guard or a row
mismatch catches it). Revert, and say in the commit body that you did this.

Commit: `test(dflash): cover the batched selector and its per-row scratch`.

---

## Phase B — Facts, shared workload, oracle baseline (timebox 1 d)

### B0. Establish the drafter's true configuration (about 30 min, no GPU)

`config.json` from `ConfigJson()` hard-codes several values. Compare each one
with a source the extension did not write. Record every value and its source
in the report.

| field | extension writes | check against |
|---|---|---|
| `target_layer_ids` | `[5,15,26,36,46]` | HF checkpoint config; llama.cpp `dflash.target_layers` usage |
| `layer_types` | all `"full_attention"` | HF checkpoint config. **Important:** upstream applies the sliding window only to layers typed `"sliding_attention"` (`dflash/model.py`, `Qwen3DFlashAttention.__init__`), while the extension's attention kernel always applies the 2048 window. If the truth is full attention, the extension is wrong past 2048 context; if sliding, `ConfigJson` is wrong. Either way only the >2048 region is affected, which Phase E tests. |
| `input_embedding_scale` | absent, so 1.0 | HF checkpoint config (`dflash_config`) |
| `mask_token_id` | from GGUF `tokenizer.ggml.mask_token_id` | HF config |
| `rope_theta`, `rms_norm_eps`, `head_dim`, `block_size` | from GGUF | HF config |

Sources, in order:

1. The HF model card / `config.json` of the drafter's source checkpoint. The
   GGUF metadata says the base is `RecursiveSI/flint` and the author is
   `incoai`, so look for `incoai/Flint-DFlash2` (and the GGUF repo
   `incoai/Flint-DFlash2-GGUF`). Use `webfetch` on
   `https://huggingface.co/<repo>/raw/main/config.json`. If no non-GGUF
   checkpoint is public, record that.
2. llama.cpp at tag `b11804`: `src/models/dflash2.cpp` (or `dflash.cpp`),
   `common/speculative.cpp`, and the `dflash` branch of
   `convert_hf_to_gguf.py`. Fetch raw files from
   `https://raw.githubusercontent.com/ggml-org/llama.cpp/b11804/<path>`. If the
   paths differ, search the repo tree at that tag. Record:
   - which hidden state feeds the drafter: `t_layer_inp[il]` for
     `il ∈ target_layers` is the input of layer `il`, which is the output of
     layer `il-1`. Confirm the extension's off-by-one.
   - whether the drafter masks with a sliding window, and how;
   - how `--spec-draft-p-min` is applied: per proposal or cumulative, on which
     probability, at which temperature;
   - how llama.cpp counts `draft_n` and `draft_n_accepted`.

If any value disagrees with what the extension uses, **do not fix it yet.**
List it in the report as "suspect #n". Phase C measures whether it matters,
and Phase D fixes it.

### B1. A fixed shared prompt set

Create `tools/fixtures/accept/prompts.jsonl`, one JSON object per line:

```json
{"id": "code-1", "category": "code", "kind": "chat", "text": "...", "max_tokens": 192}
```

- **12 chat prompts**: 4 `code`, 4 `json`, 4 `prose`. The study's original
  prompts were not saved, so write new ones and keep them fixed from now on.
  Make them realistic, distinct and self-contained. JSON prompts should ask
  for structured JSON output (for example "Return a JSON object describing
  ..."). `max_tokens` is 192, with natural stop, matching the study.
- **2 raw continuation prompts** (`kind: "raw"`, `category: "continuation"`,
  `"file"` instead of `"text"`, `"prefill": 1024`, `"max_tokens": 512`). Point
  them at the two M1b files, as paths relative to the repo:
  `src/gpu/dflash/dflash_forward.cpp` and `docs/PLAN-DFLASH2.md`. Pin each with
  `"git_rev": "7a9d6d3"` and read it with `git show 7a9d6d3:<path>`, so the
  content cannot drift as these files change.

### B2. The prompt token ids are the contract

Both engines must see **byte-identical prompt token ids**. Do not let
llama.cpp apply its own template.

Add a mode to `flint_drafter_parity`:

```
flint_drafter_parity TARGET.gguf DRAFT.gguf --prompts tools/fixtures/accept/prompts.jsonl --dump-prompts OUT.jsonl
```

It needs only the tokenizer, but loading the model is acceptable if that is
simpler. For each prompt it writes `{"id", "category", "kind", "prompt_ids":
[...], "max_tokens", "stop_on_eos": bool}`:

- `chat`: `RenderPrompt({{"user", text, ""}}, {.enable_thinking = false})`,
  encoded with `TokenizerOptions{.parse_special_tokens = true}`. This is the
  same as `drivers.cpp` lines ~178-185; copy that code exactly. `stop_on_eos`
  is true.
- `raw`: encode the file text without special-token parsing and keep exactly
  the first `prefill` ids. `stop_on_eos` is false.

The tool needs a small JSONL reader. The prompt file is flat
(strings/ints/bools), so a minimal hand parser is fine. Check first whether
the Gufo SDK exposes a JSON library under `gufo/ext/*` or vcpkg (`grep -rn
"nlohmann\|simdjson" /home/bruno/AI/Gufo/include /home/bruno/AI/Gufo/vcpkg.json`)
and use it if allowed for extensions.

### B3. Oracle baseline on the shared workload

Write `tools/drafter_accept.py` with an `oracle` subcommand. It:

1. Starts `llama-server` itself, or uses a running one if `--url` is given:

   ```sh
   /home/bruno/AI/llama/llama-b11804-rocm/llama-server \
     -m /home/bruno/AI/GGUF/flint-UD-Q4_K_XL.gguf \
     -md /home/bruno/AI/GGUF/mtp/Flint-DFlash2-BF16.gguf \
     --spec-type draft-dflash --spec-draft-n-max 7 --spec-draft-p-min P \
     -fa on -np 1 --temp 0 -ngl 99 -c 8192 --host 127.0.0.1 --port 8098 --no-webui
   ```

   It captures stderr to a log, waits for `/health`, and always kills the
   server on exit (`try/finally`).
2. For each prompt in the dump, POSTs to `/completion` with
   `{"prompt": prompt_ids, "n_predict": max_tokens, "temperature": 0,
   "top_k": 1, "cache_prompt": false, "ignore_eos": not stop_on_eos,
   "return_tokens": true}`.
   - **Inspect the first response by hand** and record which fields exist.
     The generated token ids must come back as ids (`tokens`). If
     `return_tokens` is not supported, stop and ask; do not re-tokenize text.
   - Record `timings.draft_n`, `timings.draft_n_accepted`,
     `timings.predicted_n` and the generated ids.
   - Mean len: if the server log prints it per request (`slot print_timing` /
     `mean len` / `draft acceptance`), parse it from the log. Otherwise compute
     `predicted_n / cycles`, using the cycle count the log or the response
     exposes. State which source you used. Never guess cycles as
     `draft_n / 7` without confirming that blocks are always 7 long.
3. Runs once at `p_min 0` and once at `p_min 0.5`, and writes
   `oracle-p0.jsonl` / `oracle-p05.jsonl` to `$RESULTS_DIR` (default
   `/tmp/opencode/m1c`), one line per prompt, plus a per-category summary:
   acceptance `Σaccepted/Σdrafted`, and mean len as `Σemitted/Σcycles`
   (pooled, not an average of averages).

**Sanity check.** The oracle's chat results should be in the neighbourhood of
the study (p0 ≈ 33%, p0.5 ≈ 51%; mean len ≈ 4.2 code, 4.6 JSON, 3.2 prose).
The prompts are new, so expect ±5 points. If they are far off (for example
p0.5 below 40%), check the request first (template applied twice, thinking
on, `top_k`) before you trust the number.

Commit B (tools + fixture + `extension.*` changes, not results):
`feat(tools): shared acceptance workload and oracle baseline (M1c B)`.

---

## Phase C — Extension on the same workload; verdict (timebox 1 d)

### C1. `ResetContext()` on the drafter

Running many prompts in one process needs the drafter's injected context to be
reset between prompts. `AppendContext` requires
`position == injected_context_len`. Add
`void DflashForward::ResetContext() noexcept` that sets the injected length
back to 0. Read `AppendContext` and `Run` first and confirm that nothing else
holds per-sequence state: ring contents beyond `ContextLength()` must never be
read. Document this in the header. The runner (M3) needs the same call when a
session rewinds before the drafter's context.

Extend `flint.dflash_forward` with one check: inject context, `Run`, then
`ResetContext`, re-inject the same context, `Run` again. The second
proposals must equal the first.

### C2. `--prompts` acceptance mode

```
flint_drafter_parity TARGET.gguf DRAFT.gguf --temperature 0 \
  --prompts-ids OUT_OF_B2.jsonl --report ours.jsonl [--stream ORACLE.jsonl]
```

Load the target and the drafter once. For each prompt: `session->Rewind(0)`,
`forward->ResetContext()`, prefill the prompt ids, then run the same cycle
`MeasureAcceptance()` runs, with these changes:

- **Prefill in chunks of at most 1024 tokens**, injecting the captured rows
  into the drafter after each chunk. The hidden ring holds only 2048 rows, so a
  long prompt must be fed incrementally. Refactor `MeasureAcceptance` so the
  per-prompt loop is a function that both the old `--accept` path and the new
  path call. Do not duplicate it.
- **Stop rule:** stop after `max_tokens` emitted tokens or, when
  `stop_on_eos`, when an emitted token is EOS
  (`tokenizer.IsEos`-style check; see `hip/vendor/qwen/tokenizer.hpp`). Count
  a cycle's emitted tokens only up to and including the stop token, as
  upstream `dflash_generate` does.
- **`--stream FILE` (forced stream):** if given, the continuation for prompt
  `id` is the oracle's generated ids from that file, not the target's argmax.
  Acceptance in a cycle is the number of leading drafts that equal the stream's
  next tokens; the token after them is the stream's next token, not the
  target's argmax. Verify still calls `AppendAll`, because the target must
  produce the hidden taps for the accepted rows. Record how often the target's
  own argmax disagrees with the stream (`stream_divergence`): it shows how
  similar the two engines' greedy texts are.
- **`p_min` emulation (optional, only if B0 showed a clean definition).** If
  llama.cpp's `p_min` uses a per-proposal probability the selector can report
  at temperature 0, add `--p-min X`: truncate each block at the first proposal
  below X before verify, and verify only `anchor + kept`. If the extension's
  greedy `confidences` are always 1.0 (`TestSelectorGreedy` asserts this), the
  definition cannot be reproduced without kernel changes, so **skip this** and
  write "p_min 0.5 not reproduced" in the report. `p_min 0` is the primary
  comparison either way.

Report one JSON line per prompt: `id, category, prompt_len, emitted, cycles,
drafted, accepted, acceptance, mean_len, per_row_accepted[7],
stream_divergence (if --stream)`.

### C3. Runs

Run the BF16 drafter. Then run Q6_K on the same prompts (forced-stream run
only), so the Q6_K cost is measured on the new workload too.

1. Free-running (`ours-free.jsonl`).
2. Forced stream from `oracle-p0.jsonl` (`ours-stream.jsonl`). **This is the
   primary comparison**: same prompt ids, same continuation text, so acceptance
   differs only through the drafter and its inputs.

### C4. `drafter_accept.py compare`

Join the reports by `id` and print a markdown table per category and pooled:
oracle p0 acceptance / mean len, ours-free, ours-stream, delta, stream
divergence. Write `compare.md` and `compare.json`.

### C5. Verdict

Compare `ours-stream` with `oracle-p0`, per category and pooled, on **mean
len** first and acceptance second.

- **Verdict A (workload, no bug):** pooled mean len ≥ oracle − 0.25,
  **and** every chat category ≥ oracle − 0.40, **and** pooled acceptance
  within 3 points. Skip Phase D.
- **Verdict B (input bug likely):** any of the above fails. Run Phase D.
- If the `continuation` prompts pass but chat fails, or the other way round,
  record that in the report. It points at template/BOS handling (chat-only
  failure) or at long-context/window handling (continuation-only failure).

Commit C: `feat(dflash): measure acceptance on the oracle's workload (M1c C)`.
Put the verdict table in the commit body.

---

## Phase D — Input diagnostics (only on verdict B; timebox 1.5 d)

Work in this order and stop at the first fix that closes the gap. After each
candidate fix, rerun **C3 step 2 (forced stream)** and the BF16 parity gate
(§1 command). A fix that raises acceptance but breaks parity is wrong.

1. **Suspects from B0.** Apply each config discrepancy found in B0 (in the
   loader or the forward; `ConfigJson` must then write the corrected value
   too) and measure each one separately.
2. **Tap sensitivity.** Add `--force-taps a,b,c,d,e` to the driver. It
   overrides the artifact's taps; the current `--target-layer` check fails on
   a mismatch, so either keep that flag as is and add the new one, or
   document the change. Measure forced-stream acceptance with all taps shifted
   by −1 and by +1, using the free-running mode on the 4 code prompts only (to
   keep it cheap). The correct taps should be clearly best. If a shifted set
   wins by more than 1 point of mean len, the off-by-one is wrong: fix the
   loader and add a comment that cites the llama.cpp source line from B0.
3. **Feature/anchor alignment.** Upstream (`dflash_generate` in
   `dflash/model.py`) drafts the block at `start` (anchor position) with
   context = target hidden of positions `0..start-1`. After a verify, it
   appends the hidden of the `produced` committed rows, which includes the
   anchor's row. Check that `MeasureAcceptance` injects exactly these rows at
   exactly these positions, with no off-by-one after a bonus token. Add a debug
   assertion that `forward.ContextLength() == position` right before every
   `Run`. Measure a deliberate off-by-one variant to see whether acceptance is
   sensitive to it.
4. **Capture point.** Confirm in `engine.cpp` (`CaptureHidden` call site,
   around the `tap_layers` loop) that the captured row is the layer's
   **output residual stream before the final norm**, which matches upstream
   `output_hidden_states[layer_id + 1]`. A capture after a norm, or of a
   partial residual, is a bug.
5. **Last resort: feature dump against llama.cpp.** This needs a llama.cpp
   build with an eval callback (`llama-eval-callback` is not in the shipped
   b11804 binaries). **Ask the user before building it.** If approved, dump
   the drafter's `fc` input for one prompt and compare it row by row with the
   extension's `target_features.f32`. Expect agreement at the target-quant
   floor (~1e-2 relative). A wrong layer or a shifted row is off by orders of
   magnitude.

If the timebox ends without closing the gap: stop and report what was tried,
with numbers for each attempt, and recommend next steps. Do not start E–F on
a broken drafter without the user's approval.

Commit each real fix separately: `fix(dflash): <what>`, with acceptance before
and after in the body.

---

## Phase E — 20-position parity sweep: Levels 1 and 2 (timebox 1 d)

M1b checked parity at one position (64). Real sessions pass through the
2048-row sliding window and the context-ring wraparound, and both were
untested.

### E1. Driver: multi-position dump

Add `--positions 64,256,1024,...` to the dump mode. Load once, prefill once
up to the largest position in **chunks of at most 1024**, and at each listed
position:

- keep **host copies of all captured rows so far**, because the harness needs
  `target_features` for every context row and the ring keeps only 2048. Copy
  each chunk's rows out right after appending it.
- create (or `ResetContext` + re-inject) the drafter with exactly the rows
  `0..position-1`, trace one block, and write `OUT/pos-<N>/` (stages,
  `trace.json`, `config.json`, `target_features.f32`).
- record the greedy proposals at temperature 0 (a separate untraced `Run`) in
  `OUT/greedy.json` as `{position: [ids]}`.

Positions (20): `64, 128, 256, 512, 1000, 1024, 1500, 2000, 2046, 2047,
2048, 2049, 2050, 2100, 2560, 3000, 3500, 4000, 4096, 4500`. Use a real
prompt: concatenate the 12 chat prompts' texts and the two raw files until
there are ≥ 4600 tokens, and record the source in `trace.json`.

Before running anything past 2048, make sure `config.json`'s `layer_types`
matches the truth established in B0, and that the extension's attention
window matches it as well. Otherwise every position past 2048 fails for a
known reason, not an unknown one.

### E2. Level definitions (these replace M1b's Level 1 and 2 in the docs)

| level | check | pass |
|---|---|---|
| 1 (BF16 vs upstream) | harness on all 20 `pos-*` dirs, `--head reference`, temperature 0.8 | every position `passed: true`: every stage ≤ 1e-4, candidate sets equal, selector probabilities ≤ 5e-6 |
| 2 (Q6_K vs BF16) | `greedy.json` from both quants, native head | ≥ 95% of the 140 greedy proposals identical (≥ 133 of 140) |
| 3 (acceptance) | Phase C forced stream | the C5 verdict-A thresholds, for both BF16 and Q6_K |

Write `tools/drafter_sweep.py`, which runs the harness on every `pos-*` dir
and tabulates worst stage, its value and pass/fail per position, and the Q6_K
vs BF16 greedy agreement.

If Level 1 fails only at positions ≥ 2048: that is the sliding window or the
ring. Read `LaunchDFlashNonCausalAttention*` (window arithmetic: upstream
masks `|q - k| >= window` for non-causal sliding layers) and the ring
indexing in `AppendContext`, then fix and rerun. If it fails at a near-tie in
the selector (candidate sets differ by one token whose probabilities are
within 1e-5 of the k-th), record it as a near-tie with the numbers. Do not
treat it as a failure, and do not adjust any threshold.

Commit E: `feat(dflash): 20-position parity sweep and level gates (M1c E)`.

---

## Phase F — One-command gate (timebox 0.5 d)

Write `tools/drafter_gate.sh`, in the style of `tools/run_gates.sh`
(`set -uo pipefail`, environment overrides, logs in `$RESULTS_DIR`):

- Environment variables: `BUILD_DIR` (default the build dir above),
  `TARGET` (Q4_K_XL), `DRAFT_BF16`, `DRAFT_Q6K`, `ORACLE_BIN_DIR`,
  `UPSTREAM_DIR` (default `/tmp/opencode/dflash-upstream`), `HARNESS`
  (Gufo-mtp-perf path), `RESULTS_DIR` (default `/tmp/opencode/drafter-gate`).
- Tiers:
  - `parity`: verify or clone the upstream at `07ebd93` and check its sha256,
    then run E1 + `drafter_sweep.py` for BF16 and Q6_K.
  - `accept`: dump prompts, then the oracle at p0 and p0.5 (skippable with
    `SKIP_ORACLE=1`, which reuses cached oracle JSONL in `$RESULTS_DIR`), then
    ours free and forced-stream for BF16 and Q6_K, then `compare`.
  - `full`: both tiers.
- Finish with `summary.md` + `summary.json`: one line per level with PASS/FAIL
  and the key numbers. The exit code is non-zero if any level fails.
- Enforce rule 3: the oracle server is stopped before an extension tool runs.

Document it in `tools/README.md` (a short section; follow that file's style).

Run `tools/drafter_gate.sh full` once from a clean `$RESULTS_DIR`. The report
quotes its `summary.md`.

Commit F: `feat(tools): one-command drafter gate (M1c F)`.

---

## Phase G — Docs and the final report (timebox 0.5 d)

1. `docs/PLAN-DFLASH2.md`:
   - **Correct the 51.1% interpretation** wherever the doc compares to it
     (§1/§2, §3.4, "M1b level 3"). It is a `p_min 0.5` figure with 3.69
     drafts/cycle; the comparable metrics are mean len and `p_min 0`
     acceptance. Remove the claim that the extension's first-proposal rate
     "sits in the oracle's band".
   - Replace M1b Levels 1–3 with the E2 definitions and their results.
   - Update §3.4's expected-speedup table with the measured mean len ×
     the verify ratios (2.21 / 2.48 / 3.16 / 4.25 at 0 / 8k / 32k / 131k).
     Speedup ≈ `mean_len / ratio`.
   - Add an "M1c" section: what was found, which fixes, and where the gate
     lives.
2. `docs/ROADMAP-DFLASH2.md`: update the acceptance rows of the status table
   the same way.
3. Write `docs/REPORT-M1C.md` from the §9 template, and post its contents in
   chat as the closing message.

Commit G: `docs(plan): M1c results and the corrected acceptance baseline`.

---

## 9. Report template (`docs/REPORT-M1C.md`)

Fill in every field. Use "not run" / "skipped (reason)" where that applies;
never leave a placeholder.

```markdown
# M1c report — acceptance against the oracle

## Summary
<3-5 sentences: the verdict (A or B), whether any bug was found/fixed, the
headline mean len vs oracle, and the projected speedup at depth 0.>

## What was done
| phase | status | commit | notes |
|---|---|---|---|
| A selector batch test | done/stopped | <sha> | guard-tail failure confirmed when allocation shrunk: yes/no |
| B0 config facts | ... | | suspects found: <list or none> |
| B1-B3 workload + oracle | ... | | |
| C acceptance on shared workload | ... | | |
| D diagnostics | ran / skipped (verdict A) | | |
| E 20-position sweep | ... | | |
| F gate script | ... | | |
| G docs | ... | | |

## Drafter configuration facts (B0)
| field | extension | source of truth | value there | match |

## Acceptance on the shared workload (BF16 drafter)
| category | oracle p0 acc | oracle p0 mean len | oracle p0.5 acc | oracle p0.5 mean len | ours free acc / mean len | ours stream acc / mean len | Δ mean len (stream − oracle p0) | stream divergence |
| code | | | | | | | | |
| json | | | | | | | | |
| prose | | | | | | | | |
| continuation | | | | | | | | |
| pooled chat | | | | | | | | |

Per-row acceptance curve (ours, stream, pooled chat): row 0..6.

Q6_K vs BF16 (stream): acceptance Δ, mean len Δ.

## Verdict
A or B, with the thresholds from C5 and the numbers that decided it.

## Diagnostics (if D ran)
| attempt | change | mean len before | after | parity still passes | kept |

## Parity sweep (E)
| position | BF16 worst stage | value | Level 1 | Q6_K==BF16 greedy (of 7) |
Level 1: x/20 positions pass. Level 2: y/140 proposals identical (z%).

## Projected speedup
| depth | verify ratio | code | json | prose |
(mean_len / ratio, using ours-stream BF16 or Q6_K mean len — say which)

## Gate
`tools/drafter_gate.sh full` -> summary.md quoted here.

## Open issues and deviations from the plan
<anything skipped, any rule that had to be bent and why, any near-ties>

## What's next
<Recommended next step, normally M2 (target adapter) + M3 (runner cycle)
from docs/PLAN-DFLASH2.md, with the updated go/no-go reasoning based on the
measured mean len. If the projected speedup is < 1.3x on chat code/JSON at
depth 0, say so plainly and recommend whether to continue, re-scope (for
example adaptive draft length / p_min policy in M4), or stop.>
```
