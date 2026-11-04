# Flint 40B parity and quality gates

Speed and correctness gates for strycast's Flint 40B MoE port
(`src/models/flint/`, from [mikealanni/gufo](https://github.com/mikealanni/gufo)
`fork-enhancements`). Most tools are HTTP-only: they talk to an OpenAI-style
`/v1/chat/completions` endpoint and need no engine code. They were originally
written against the fork's own tree and sandbox gateway; the paths, ports and
model ids below are strycast's.

## Runner

```sh
tools/flint/run_gates.sh quick    # G8 + depth 2048/32768, 1 run
tools/flint/run_gates.sh stage    # G8 + depth 512/2048/8192/32768, 2 runs
tools/flint/run_gates.sh full     # G8, G4, GX, G5, six-depth Q4 + fork + Q6
```

Logs and JSON land in `$RESULTS_DIR` (default `/tmp/opencode/flint-gates`).
The runner starts and stops the model servers itself, one model on the GPU at a
time. Overrides: `GUFO_BIN`, `FORK_BIN`, `MODEL_Q4`, `MODEL_Q6`, `PORT`,
`ORACLE_PORT`, `RESULTS_DIR`, `PYTHON`.

## Tools

| Tool | Purpose |
| --- | --- |
| `flint_depth_curve.py` | Prefill/decode tokens-per-second over HTTP at each prompt depth. Streams, derives prefill from TTFT and decode from `(completion-1)/(t_last-t_first)`, records thermals. `--no-cache-prompt` forces cold prefill. Prompt text comes from `fixtures/fj4/`. |
| `serve_gates.py` | `g4` (greedy divergence vs the oracle's own top-20, rank ≤ 3 / delta ≤ 0.5), `g4-compare`, `g5` (T1.0/p0.95/k20 distribution arm), `g5-verdict` (support + ordering + TV ≤ 0.25), `g8` (concurrent identical requests must be byte-identical), `fi4` (peer routing isolation). All take `--base-url`/`--model`. |
| `gx_capture.py` / `gx_compare.py` | GX greedy-exactness: five fixed prompts × 256 tokens at T0, captured once and diffed against a later build. A change detector, not a quality test. |
| `drafter_batt.py` | Mixed-content decode-speed battery (draft acceptance). Decode check, not GX. |
| `corpus.py` | Deterministic 2,700-case corpus (nine classes) used by G4 to build prompts. |
| `golden.py` | Captures tokenizer (`tokenize`) and template (`template`) goldens from the pinned llama.cpp oracle; `compare` diffs two tokenize goldens. |
| `g1_cases.py` / `g2_cases.py` | Join the corpus/fixtures with their goldens into the packed probe inputs. |
| `tokenizer_probe.cpp` / `template_probe.cpp` | G1/G2 C++ probes: encode the corpus / render the fixtures through strycast's tokenizer and `models::flint::RenderPrompt`, compare against the golden. Built with `-DGUFO_BUILD_TOOLS=ON` as `flint_tokenizer_probe` / `flint_template_probe`. |
| `oracle_server.sh` | Pinned llama.cpp b11804 oracle on `:8098`. Modes `cpu` (no GPU — enough for tokenize/template goldens and G4 prompt building), `gpu`, `cpu-q4`, `gpu-q4`. Set `MODEL`/`MODEL_Q4` for the artifact. |

## G1 / G2

```sh
OUT=/home/bruno/AI/Gufo-testing/flint-oracle   # goldens live outside git
python3 tools/flint/corpus.py "$OUT" --probe
python3 tools/flint/golden.py tokenize "$OUT"      # spawns the cpu oracle
python3 tools/flint/golden.py template "$OUT"
python3 tools/flint/g1_cases.py "$OUT/tokenize_golden.json" /tmp/g1-cases.bin
python3 tools/flint/g2_cases.py "$OUT/template_golden.json" /tmp/g2-cases.bin
./build/release/flint_tokenizer_probe MODEL.gguf /tmp/g1-cases.bin
./build/release/flint_template_probe  MODEL.gguf /tmp/g2-cases.bin
```

The template probe skips fixtures strycast's `RenderPrompt` cannot express
(tool preamble, `<tool_call>` argument rendering, trailing-assistant continue
mode) and reports them as `SKIPPED` with the reason rather than counting them
as passes.

## Oracle

The oracle is pinned to llama.cpp **b11804** (`0.7.0-dev, commit c7fc50941`),
which carries `flint` in `libllama.so`. `LLAMA_DIR` defaults to
`/home/bruno/AI/llama/llama-b11804-rocm`. Never run the oracle during a timed
benchmark: the CPU oracle slowed decode ~40% in an earlier session. Stop every
server with `fuser -k <port>/tcp` between phases.

Attribution: `serve_gates.py`, `corpus.py`, `golden.py`, `gx_*.py`,
`drafter_batt.py` and the probe layouts come from mikealanni's Flint work
(the `Gufo-flint` oracle harness); the probes were rewritten against
strycast's headers and the runner is new.
