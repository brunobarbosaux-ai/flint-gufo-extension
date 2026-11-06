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
cmake --build --preset gpu-test --target gufo flint_config_test flint_probe_test flint_ops_test --parallel 16
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

## Layout

- `extension.json` / `extension.cmake` — manifest and targets
- `extension.cpp`, `provider.cpp`, `probe.cpp` — registration and artifact probe
- `runner.cpp`, `drivers.cpp` — text runner contract and CLI drivers
- `src/` — engine, config, chat template, tokenizer support
- `hip/vendor/` — vendored kernel/tokenizer tiers (see `THIRD_PARTY_NOTICES.md`
  for provenance and the namespace mapping)
- `tests/` — config, probe and GPU operator tests

## Maintenance

Pin the candidate Gufo and extension revisions, rebuild, and rerun
`flint.config`, `flint.probe`, `flint.ops` plus the parity gates in
`docs/QUALITY.md` before publishing an update. Source compatibility applies
after rebuilding; there is no binary ABI compatibility promise.
