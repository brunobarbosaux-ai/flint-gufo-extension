# State — DFlash2 integration

## Current phase
M3 — Cycle + runner integration (M2 done)

## Decisions
- Follow `docs/PLAN-DFLASH2.md` M2 -> M6 in order.
- Never loosen gates (1e-4 stage, 5e-6 selector, 95% level-2).
- One GPU model process at a time.
- Cycle hook: `SelectNext`/`Advance` (plan §3.2), not `DecodeStep`; keeps the sampler's history updated between rows.
- Speculation is off by default; `--speculative dflash2 --dflash-model <gguf>` opts in (no env vars).

## Environment
- Extension repo: /home/bruno/Projects/flint-gufo-extension
- Build dir: /home/bruno/AI/Gufo/build/host-test
- Tool binaries: /home/bruno/AI/Gufo/build/host-test/extensions/build/flint/build/
- Target: /home/bruno/AI/GGUF/flint-UD-Q4_K_XL.gguf
- Drafters: /home/bruno/AI/GGUF/mtp/Flint-DFlash2-{BF16,Q6_K}.gguf
- Oracle: /home/bruno/AI/llama/llama-b11804-rocm/llama-server

## Open questions
- (none)
