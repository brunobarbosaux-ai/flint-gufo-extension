# flint — Gufo extension

Flint 40B MoE (text generation) as a standalone Gufo source extension. The
extension is self-contained: it vendors the operators, tokenizer and chat
framing it needs, so it builds against an unmodified Gufo checkout.

## Status

Working on gfx1151 (AMD Strix Halo) with HIP. See `docs/QUALITY.md` for what
is validated and what is still open.

The Flint model implementation is derived from [`mikealanni/gufo`](https://github.com/mikealanni/gufo),
an MIT-licensed Gufo fork, with its NPU-specific sections removed and the code
adapted to the external extension contract. See `THIRD_PARTY_NOTICES.md`.

- Maintainer: Bruno Barbosa (brunobarbosaux-ai)
- Gufo pin: `4a7d8c6d178fc6979e5ae38c07b40e1fe89dd1ce` (v0.36.0, API 0.3.42).
  That is the tree 0.5.0 is validated on. The source builds on any 0.3-series
  host (API revisions only add); the earlier `c40862ff` (v0.32.0, API 0.2)
  pin cannot build 0.3-series code.
- Gufo API floor: `>= 0.3.0`. Derived from the upstream change records this
  extension relies on: the series-0.2 component-cache-only continuation baseline
  (`0.2.0`), reasoning delimiters (`0.2.5`), provider-defined text sampling
  defaults (`0.2.7`), provider-declared speculative defaults (`0.2.15`) and the
  required text scheduling profile (`0.3.0`, `TextRunnerDescriptor::scheduling`).
- Platforms: Linux

## Build

From a Gufo checkout with its documented dependencies:

```sh
cmake --preset gpu-test -DGUFO_EXTENSION_DIRS=/absolute/path/to/flint-gufo-extension
cmake --build --preset gpu-test --target gufo flint_config_test flint_probe_test flint_tool_template_test flint_tool_scan_test flint_ops_test --parallel 16
ctest --preset gpu-test -R '^flint\.' --output-on-failure
build/gpu-test/gufo diagnose --section extensions --json
```

A CPU build still discovers the artifact and reports the provider as
unavailable (probe-only registration).

## Weights

The extension consumes a GGUF artifact keyed `gguf:flint` (architecture
`flint`). Weights are distributed separately; see `docs/QUALITY.md` for the
artifact used in validation runs. Weight licensing is the model publisher's;
this repository only reads the format.

## Speculative decoding (DFlash2)

**Context ceiling.** Verified serve startup on Gufo 0.36.0 (API 0.3.42):
contexts up to **262144 (the model's native maximum) load**. At 131072 one
session loads in 8.0 s and two in 8.0-8.5 s (36,479 MiB of device, about 25 GiB
of that the weights). The ring cuts the per-session request state at 262144 from
26.0 GiB to about 8.7 GiB, so on this 128 GB box four sessions load in 9.5 s
using 60.0 GiB and eight fit (94.9 GiB of the 114,688 MiB device, 22.6 GiB host
left); 0.4.0 refused a fourth session outright. Load time still depends on the
page cache: releasing the GGUF mapping after upload (0.4.0) makes those pages
evictable, so a cold or pressured load pays a disk read, and the staged weight
copy (#11) halves that case (133 s -> 64 s in our A/B) without removing the
spread. Quote load times only with a warmed page cache. The 262144 hang in
session setup reported on 2026-11-17 has not reproduced in 24 loads across two
nodes.

Note that this is a **load** check. Long-context *requests* were exercised to
130k tokens (dense, block-sparse, speculative, and disk-restart restores); a
262144-token prefill has not been run through serve, so no throughput claim is
made at that depth. The continuation cache stores the K/V of each layer
as append rows, so a capture copies only the valid tokens (123 MB instead of
822 MB for a 1.2k-token prompt at any context) and a follow-up turn copies only
the new tail (25 MB at 128,981 tokens); a warm restore is 2-25 ms from RAM and
about 175 ms-1.4 s from disk, growing with depth. Capture is still synchronous
on the request thread: the upstream linear-time fix in gufo-org/gufo#1319,
which Gufo 0.36.0 ships, is what keeps a 130k capture near 5 s instead of
about 14 s. The retained cache stays capped at **8 GiB** (~85,600 full rows),
and in 0.5.0 the ring makes that cap non-binding: a window layer keeps only the
rows a pass can still read, so the 130k frontier retains 3,995,190,656 B and
its follow-up 3,776,601,216 B, both inside the cap. Measured on 0.36.0 with the
0.5.0 build: 24k reuse is exact in RAM and on disk (snapshot 959 MB, restore
12.5 ms); at 60k the snapshot drops to 1.92 GB, an identical repeat restores in
26 ms and, after a **disk restart**, the restored frontier plus the follow-up
both fit, so the follow-up capture that 0.4.0 refused now succeeds; at 130k an
identical repeat restores all 128,981 tokens from RAM in 51 ms, and after a
**disk restart** 128,980 tokens restore from disk in 1.4 s. Raising the cap is
no longer needed and is not part of 0.5.0 (see `DEFERRED.md`).

Flint can run DFlash2 speculative decoding: a small drafter proposes several
tokens, and the target verifies them in one batched pass. Accepted tokens are
emitted without extra target steps, so single-stream decode is faster. It is
**opt-in and off by default**.

Enable it with CLI options on `gufo serve`. The drafter must be passed
explicitly; there is no default path:

```sh
gufo serve llm -m flint-UD-Q4_K_XL.gguf \
  --speculative dflash2 \
  --dflash-model /path/to/Flint-DFlash2-Q6_K.gguf
```

Speculation is on only when `--speculative dflash2` is set, and it then requires
`--dflash-model`. The runner fails at startup with a clear error if the drafter
is missing. No environment variable turns speculation on or off.

Drafter files (the names are the published DFlash2 drafter GGUFs; pass the path
of the quant you want with `--dflash-model`):

| File | Size | Notes |
| --- | --- | --- |
| `Flint-DFlash2-Q6_K.gguf` | 336 MB | GX 3/3 |
| `Flint-DFlash2-Q4_K_M.gguf` | 240 MB | smallest; GX 3/3 |
| `Flint-DFlash2-BF16.gguf` | 811 MB | reference precision; GX 3/3 |

Measured decode speedup at depth 0, temp 0, adaptive: 1.24 to 1.35 across the
three quants. Adaptive is at least as fast as fixed at every depth measured.
Measured decode-only speedup falls to about 1.07 to 1.20 at depths up to 131k;
whole-request speedup is below 1.0 at depth >= 8k because prefill dominates.
See `docs/QUALITY.md` for the full table.

**VRAM.** Each speculative session adds about 0.44 GB to its claim (hidden-state
ring about 320 MiB, plus context scratch). The drafter weights (240 to 811 MB,
depending on quant) are resident once per runner. A single speculative session
therefore costs about 0.7 to 1.2 GB more than spec-off. The ring and drafter are
counted in `ResourceClaim`; the pre-existing base-session undercount is in
`DEFERRED.md`.

**Knobs** (per request or as provider defaults; the host applies them):

- `--draft-tokens N` / `max_draft_tokens` (default 7)
- `--min-draft-tokens N` / `min_draft_tokens` (default 1; must not exceed max)
- `--draft-policy fixed|adaptive` / `dflash_policy` (default `adaptive`)
- `--speculative dflash2` selects the backend

The extension declares these defaults with `speculative_defaults` in
`provider.cpp`; a value given on the command line overrides them.

**When off, output is unchanged.** Without `--speculative dflash2`, no drafter
is loaded and no speculative cycle runs. The null drafter gate (72/72) confirms
the cycle itself does not change output.

**Sampled mode caveat.** Greedy (`temp 0`) output is byte-identical with
speculation on or off (12/12 at depths 0, 8k, 32k, 131k). At `temp > 0` the
accept/reject step consumes RNG draws differently, so a fixed seed does **not**
guarantee identical output with speculation on. The sampling distribution is
preserved by design, but field validation of sampled output is still open.

## Tool calls

Flint writes tool calls in the GLM markup:

```
<tool_call>NAME<arg_key>KEY</arg_key><arg_value>VALUE</arg_value>...</tool_call>
```

The extension serves OpenAI-style `tools`, assistant `tool_calls` and `tool` messages, and `tool_choice`
(`auto`, `required`, `none`, or a named function), buffered and streamed. `tool_choice: none` leaves the
tools out of the prompt.

- **Prompt.** `RenderPrompt` in `src/chat_template.cpp` renders the `# Tools` system block, the tool calls of
  earlier assistant turns and the `<|observation|>` blocks exactly as the official Jinja template does, byte
  for byte. `tests/golden/chat_template.jinja` is that template (revision pinned in `tests/golden/README.md`);
  `tools/gen_template_goldens.py` renders 21 conversations with it (jinja2, with the whitespace settings and
  the `tojson` filter of Hugging Face transformers) into `tests/golden/template_cases.json`, and
  `flint_tool_template_test` compares them with `RenderPrompt`.
- **Dialect.** `src/tool_dialect_register.cpp` registers a `flint-glm` tool dialect, which the runner selects
  with `ToolDialectId()`. Its `ToolSyntax` (tagged arguments) gives Gufo the grammar that constrains calls for
  `tool_choice` `required`/named functions, and `ScanFlintToolCalls` (`src/tool_dialect.cpp`) reads the calls
  back, including a call that is still being written while streaming. `flint_tool_scan_test` covers complete,
  typed, malformed and unfinished calls, and checks that streamed arguments only grow.
- **Value types.** The markup carries no types (`5` may be an integer or a string), and the host does not pass
  the tool schemas to a provider scanner. The runner therefore remembers, by tool name, which parameters of the
  tools it rendered into prompts are strings (`RememberToolSchemas`); the scanner reads those as strings and
  every other declared parameter as JSON, which the host then checks against the schema. A tool or parameter
  that was not declared is read as a string. Two concurrent requests that declare the same tool name with
  different schemas can therefore mislead the scanner.
- **Behaviour.** Flint always reasons first and may open a call inside its reasoning. With `tool_choice:
  required` and a prompt that needs no tool, it can reason up to `max_tokens` without calling.

```sh
curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "flint",
  "messages": [{"role": "user", "content": "What is the weather in Rennes?"}],
  "tools": [{"type": "function", "function": {"name": "get_weather", "description": "Weather of a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
}'
```

## Layout

- `extension.json` / `extension.cmake` — manifest and targets
- `extension.cpp`, `provider.cpp`, `probe.cpp` — registration and artifact probe
- `runner.cpp`, `drivers.cpp` — text runner contract and CLI drivers
- `src/` — engine, config, chat template, tool dialect, tokenizer support
- `hip/vendor/` — vendored kernel/tokenizer tiers (see `THIRD_PARTY_NOTICES.md`
  for provenance and the namespace mapping)
- `tests/` — config, probe, tool template/scanner and GPU operator tests; `tests/golden/` holds the official
  chat template and the prompts it renders

## Maintenance

Pin the candidate Gufo and extension revisions, rebuild, and rerun
`flint.config`, `flint.probe`, `flint.tool_template`, `flint.tool_scan`, `flint.ops` plus the parity gates in
`docs/QUALITY.md` before publishing an update. Source compatibility applies
after rebuilding; there is no binary ABI compatibility promise.
