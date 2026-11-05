# flint — Gufo extension

Flint 40B MoE (text generation) as a standalone Gufo source extension. The
extension is self-contained: it vendors the operators, tokenizer and chat
framing it needs, so it builds against an unmodified Gufo checkout.

## Status

Working on gfx1151 (AMD Strix Halo) with HIP. See `docs/QUALITY.md` for what
is validated and what is still open.

- Maintainer: Bruno Barbosa (brunobarbosaux-ai)
- Gufo pin: `186a507da77bcbb3e8f0cfbd49f688f88535e9e2`
- Gufo API floor: `>= 0.1.19` (matches the LFM3 reference extension; revalidate
  when moving the pin — this repository was established against a shallow
  Gufo checkout, so the floor was not re-derived from upstream history)
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
