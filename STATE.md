# M1c State

## Current phase
A — Selector batch regression test

## Decisions
- Follow `docs/PLAN-M1C-ACCEPTANCE.md` phases A→G in order.
- D runs only if C verdict is B.
- Never loosen gates (1e-4 stage, 5e-6 selector).
- One GPU model process at a time.

## Environment
- Extension repo: /home/bruno/Projects/flint-gufo-extension
- Build dir: /home/bruno/AI/Gufo/build/host-test
- Tool binaries: /home/bruno/AI/Gufo/build/host-test/extensions/build/flint/build/
- Target: /home/bruno/AI/GGUF/flint-UD-Q4_K_XL.gguf
- Drafters: /home/bruno/AI/GGUF/mtp/Flint-DFlash2-{BF16,Q6_K}.gguf
- Oracle: /home/bruno/AI/llama/llama-b11804-rocm/llama-server
- Upstream: /tmp/opencode/dflash-upstream (pinned 07ebd93)

## Open questions
- B0: layer_types full vs sliding attention (suspect).
