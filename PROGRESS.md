# M1c Progress — acceptance against the oracle

Plan: `docs/PLAN-M1C-ACCEPTANCE.md`

## Phase status

| Phase | Title | Status | Commit | Notes |
|---|---|---|---|---|
| A | Selector batch regression test | COMPLETE | 5bfc314 | guard-tail shrink experiment confirmed the bug is caught |
| B | Facts + shared workload + oracle baseline | COMPLETE | fc9eeb3 | B0: only suspect is ConfigJson layer_types (harness-only). Oracle mean len ~3.35 code, not 4.23 |
| C | Extension on same workload, verdict | NOT STARTED | | |
| D | Input diagnostics (only if verdict B) | BLOCKED (on C) | | |
| E | 20-position parity sweep | NOT STARTED | | |
| F | One-command gate script | NOT STARTED | | |
| G | Docs + final report | NOT STARTED | | |

## Verdict

(pending Phase C)

## Log

- Session start: plan written, uncommitted. No tracking files existed.
