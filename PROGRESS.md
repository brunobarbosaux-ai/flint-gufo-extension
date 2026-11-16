# M1c Progress — acceptance against the oracle

Plan: `docs/PLAN-M1C-ACCEPTANCE.md`

## Phase status

| Phase | Title | Status | Commit | Notes |
|---|---|---|---|---|
| A | Selector batch regression test | COMPLETE | 5bfc314 | guard-tail shrink experiment confirmed the bug is caught |
| B | Facts + shared workload + oracle baseline | COMPLETE | fc9eeb3 | B0: only suspect is ConfigJson layer_types (harness-only). Oracle mean len ~3.35 code, not 4.23 |
| C | Extension on same workload, verdict | COMPLETE | 3d78abb | **Verdict A** (workload, not a bug). Forced-stream mean len within ±0.04 of oracle in every category |
| D | Input diagnostics (only if verdict B) | SKIPPED (verdict A) | | |
| E | 20-position parity sweep | NOT STARTED | | |
| F | One-command gate script | NOT STARTED | | |
| G | Docs + final report | NOT STARTED | | |

## Verdict

(pending Phase C)

## Log

- Session start: plan written, uncommitted. No tracking files existed.
