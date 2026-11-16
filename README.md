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
- Gufo pin: `6cc8a42ca84438a0566ee498a315f35c6190dc8e` (v0.31.0)
- Gufo API floor: `>= 0.2.7`. Derived from the upstream change records this
  extension relies on: the series-0.2 component-cache-only continuation baseline
  (`0.2.0`), reasoning delimiters (`0.2.5`) and provider-defined text sampling
  defaults (`0.2.7`), which the descriptor and provider both set.
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
