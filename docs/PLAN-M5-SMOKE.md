# M5 smoke tier plan

This plan closes M5 (gates) of `docs/PLAN-DFLASH2.md` cheaply. It reuses the
data we already have and runs on the GPU only where the existing data cannot
answer the question. Any LLM or person picking this up should be able to
follow it step by step.

**Status:** waiting for user approval. Do not start until approved.

## 0. Ground rules (read first)

1. **Smoke tier is the default.** A check runs the smallest input that can
   answer its question: few prompts, short outputs, shallow depth. A full
   suite runs only if a smoke check fails or looks suspicious, and the user
   approves it. Full suites run unattended in one batch before release, not
   interactively.
2. **Reuse existing data.** Before running anything, check whether the logs
   already answer the question (section 1). Re-run only when code that could
   change the result has changed since the data was produced.
3. **Never loosen a gate.** Do not edit thresholds (1e-4 stage gate, 5e-6
   selector gate, 95% Level 2) or gate criteria. If a check fails, record the
   numbers and report. Changing a criterion needs explicit user approval
   (see decision D1).
4. **One GPU model process at a time.** Run
   `pkill -x flint_spec_gate; pkill -x llama-server; sleep 2` before every GPU
   command.
5. **Scope.** Edit only `/home/bruno/Projects/flint-gufo-extension`. Never
   modify the Gufo tree (`/home/bruno/AI/Gufo`), Strycast, or Gufo-mtp-perf.
6. **Timebox.** Each step has a wall-clock budget. If a GPU command runs more
   than 2x its budget, kill it and report. Don't wait it out.

### Commands

```sh
# Build (from the Gufo tree, which hosts the extension build)
cd /home/bruno/AI/Gufo && nix develop -c cmake --build build/host-test --target flint_spec_gate --parallel 16

# Shorthands used below
B=/home/bruno/AI/Gufo/build/host-test/extensions/build/flint/build/flint_spec_gate
T=/home/bruno/AI/GGUF/flint-UD-Q4_K_XL.gguf
P=/home/bruno/Projects/flint-gufo-extension/tools/fixtures/accept/prompts.jsonl
D=/home/bruno/AI/GGUF/mtp            # drafter quants: Flint-DFlash2-{Q6_K,Q4_K_M,BF16}.gguf
L=/tmp/opencode/m5-smoke             # log dir (mkdir -p $L)
```

The gate takes the drafter with `--dflash-model <path>`. The gate's default is
Q6_K.

## 1. What existing data already answers

State at the start: `HEAD` includes `9506b32` (prefill-chunking fix and
ring-boundary fix). M5a logs are in `/tmp/opencode/m5a/`.

| gate | existing evidence | still valid? | action |
| :-- | :-- | :-- | :-- |
| GX (12 prompts × 4 depths) | 12/12 at 0/8k/32k/131k after `9506b32` (coder run; orchestrator re-ran 8k and 131k) | yes | **none**. Re-run a 3-prompt guard only after a code change (step 5). |
| null drafter | 72/72 byte-identical (`/tmp/opencode/null_m4_verify.log`) | yes. `9506b32` doesn't touch the null path's output logic | **none** |
| settings: top-k/top-p/min-p/repeat/frequency | all PASS 12/12 (`m5a/settings3.log`) | yes. Sampler path unchanged since | **none** |
| settings: grammar | identical spec-on/off **12/12**; 2/12 outputs fail JSON validation, *in both modes*. Sample `code-2` is a long `"answer"` string cut at the 1024-token budget | speculation is honouring the grammar (identity holds). The open question is only truncation vs real violation | **step 2** (one small run) |
| depth curve | both policies, temp 0/0.7, all 4 depths (`m5a/speed_*.log`); adaptive ≥ fixed everywhere | measured before `9506b32`. Spec-on depth >0 numbers may shift slightly (acceptance at 8k rose to 0.495) | **none for M5**. Note it as pre-fix; refresh in the release batch |
| no regression (spec-off 71/63/50/28 t/s) | spec-off decode t/s in `m5a/speed_*.log`: **70.96 / 63.56 / 51.05 / 28.18** | yes. `9506b32` leaves spec-off prefill at 8192 chunks, and the ring is allocated only when capture is on, so the spec-off path is unchanged | **PASS from existing data.** Record it. No run. |
| cost (verify width-8 ratio 2.21/2.48/3.16/4.25) | M4 policy validated end-to-end | n/a | **deferred to the M6 docs** (user agreed) |
| drafter quant | only Q6_K ever exercised | no data for Q4_K_M/BF16 | **step 3** |
| memory (`ResourceClaim`) | `runner.cpp` `ResourceClaim()` reports `per_request_state_bytes = SessionBytes(...)`, which covers the KV cache and prefill scratch only. The hidden ring (~320 MiB with spec on) and the drafter state/weights are **not counted** | real bug | **step 4** (code fix plus a seconds-long check) |
| concurrency | `SupportedPlans()` returns `kSerial`, `physical_width = 1`, so requests are already serialized | yes | **docs only.** Fold into M6 |

## 2. Grammar: truncation or real violation (budget ~15 min)

**Question:** are the 2 failing grammar outputs cut off by the token budget, or
did they finish and still break the grammar? Identity spec-on == spec-off
already passes, so speculation is not at fault either way.

**2a. Gate-tool change (delegate to `api-coder`; touches only `tools/spec_gate.cpp`):**
- Add `--settings <name-substring>` so `--mode settings` can run only the
  matching settings, e.g. `--settings grammar`.
- For grammar settings, print **every** violating prompt id, not just the
  first. For each one, print `ended_by=budget` (output reached `--max-tokens`
  with no stop) or `ended_by=stop`, plus the last ~80 characters of the
  output.
- Do **not** change the PASS/FAIL logic in this step.

**2b. Run:**
```sh
mkdir -p $L; pkill -x flint_spec_gate; pkill -x llama-server; sleep 2
$B --target $T --prompts $P --mode settings --settings grammar --max-tokens 1024 2>&1 | tee $L/grammar.log
```

**2c. Interpret:**
- Every violation `ended_by=budget` and the tail is a valid JSON prefix →
  **truncation.** Go to decision D1.
- Any violation `ended_by=stop` → **real violation** in the grammar path.
  Since spec-off fails identically, it predates this work and lives in the
  host's grammar/constraint code, not in DFlash2. Report it to the user with
  the output. Don't fix it under M5.

**Decision D1 (needs user approval; do not apply silently):** change the
grammar check to "a violation is an output that ended by stop and fails
validation. A budget-cut output is counted as `not_evaluated` and reported
separately." Alternative without a criterion change: re-run only the
truncated prompts with a larger `--max-tokens` (e.g. 4096) by passing a
prompts file that contains just those ids.

## 3. Drafter quant sanity (budget ~20 min total)

**Question:** does each quant load, keep greedy output correct, and give
sensible acceptance? In greedy mode every emitted token is the target's
argmax, so a weak quant can only cost speed, not correctness. A 3-prompt GX
run is enough as the correctness check.

For each `Q` in `Q6_K Q4_K_M BF16`:
```sh
pkill -x flint_spec_gate; pkill -x llama-server; sleep 2
$B --target $T --dflash-model $D/Flint-DFlash2-$Q.gguf --prompts $P --mode gx --prompts-limit 3 --max-tokens 128 2>&1 | tail -2 | tee $L/quant_gx_$Q.log
pkill -x flint_spec_gate; pkill -x llama-server; sleep 2
$B --target $T --dflash-model $D/Flint-DFlash2-$Q.gguf --prompts $P --mode speed --prompts-limit 3 --max-tokens 128 --temps 0 --policies adaptive 2>&1 | grep '^speed' | tee $L/quant_speed_$Q.log
```

**Pass:** each quant loads and GX shows `3/3 byte-identical`. **Report**
`accept_rate`, `tokens_per_cycle` and `decode_speedup` per quant in a table.
The plan has no numeric acceptance threshold here, so don't invent one. Flag
a quant whose `accept_rate` is under half of Q6_K's as suspicious and ask the
user before running anything bigger.

## 4. Memory accounting fix (budget ~45 min including build)

**Question:** does `ResourceClaim` cover everything a spec-on session
allocates?

**4a. Fix (delegate to `api-coder`):**
- In `runner.cpp` `ResourceClaim()`, when speculation is enabled
  (speculation on, i.e. `--speculative dflash2`), add the per-session hidden ring
  (`kHiddenRingRows × taps × hidden × sizeof(float)`, ≈ 320 MiB) and the
  drafter's per-session state (context ring K/V and scratch) to
  `per_request_state_bytes`. Add the drafter weights to
  `resident_weights_bytes`.
- Derive every term from the existing constants/accessors (`kHiddenRingRows`,
  `kMaxHiddenTaps` or the actual tap count, config `hidden_size`, drafter
  `history_capacity`). No hardcoded byte counts. If the drafter has no bytes
  accessor, add a small one in `src/gpu/dflash/dflash_forward.{hpp,cpp}`
  (e.g. `AllocatedBytes()`), mirroring `Session::AllocatedBytes()`.
- Spec-off claim must be unchanged.

**4b. Check (seconds of GPU time):**
- Add `--mode memory` to the gate tool. For spec-off and spec-on, create the
  runner and one slot, measure `hipMemGetInfo` free bytes before and after,
  and print the claim alongside the measured delta.
- **Pass:** for each mode, `claimed resident + per_request ≥ measured delta`,
  and spec-off's claim equals its pre-fix value.

## 5. Post-change GX guard (budget ~5 min)

Run only if steps 2a/4 changed code (they will). Steps 2a and 4 don't touch
the decode path, so this is a build-sanity guard, not a re-validation:
```sh
pkill -x flint_spec_gate; pkill -x llama-server; sleep 2
$B --target $T --prompts $P --mode gx --prompts-limit 3 --max-tokens 128 2>&1 | tail -1 | tee $L/gx_guard.log
```
Plus the unit tests, run one at a time because the full `-R flint` run has
hung past 20 minutes:
```sh
cd /home/bruno/AI/Gufo
for t in target_adapter draft_policy ops dflash_ops dflash_forward; do nix develop -c ctest --test-dir build/host-test -R "^flint.$t\$" --timeout 600 2>&1 | tail -1; done
```
`flint.config` and `flint.probe` don't build (pre-existing SDK drift in
`tests/gguf_test_shim.hpp`). Leave them.

**Escalate to a deeper run only if** the guard fails, or if a later change
touches `engine.cpp`, `runner.cpp` `Prefill`/`SelectNext`/`Advance`, or
`src/spec/cycle.cpp`. In that case add `--depth 4096` (it crosses the old
2048 boundary) before considering 8k+.

## 6. Close M5 (budget ~15 min)

1. Update `PROGRESS.md`: M5 row → COMPLETE, with one line per gate citing
   either the existing log or the smoke log in `$L`. Mark the depth curve as
   "measured pre-`9506b32`; refreshed in the release batch". Mark cost and
   concurrency as "deferred to M6 docs".
2. Commit per step with conventional subjects, e.g.
   `feat(gate): settings filter and grammar end-reason`,
   `fix(runner): account hidden ring and drafter in ResourceClaim`,
   `docs: M5 COMPLETE (smoke tier)`. Push to `origin main` after the checks
   pass.
3. Report to the user: a table of gate → result → evidence, and any decision
   still open (D1).

## 7. Release batch (later, unattended, before M6 ships)

Not part of this plan's execution. Listed so nothing is forgotten. Run all of
it in one unattended batch (overnight) once code is frozen:
GX 12 × 4 depths; null 72; full settings; depth curve both policies (post-fix
refresh); cost ratios at width 8; quant speed on 12 prompts.

## Total budget

About **1h45 wall-clock**: grammar 15 min, quant 20 min, memory 45 min,
guard 5 min, close-out 15 min. About 30 minutes of that is GPU time; the
rest is code and builds.
