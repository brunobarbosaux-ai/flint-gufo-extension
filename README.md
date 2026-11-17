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
- Gufo pin: `c40862ffd740832f47187b23f615be5c5b750415` (v0.32.0). The earlier
  `6cc8a42` (v0.31.0) pin predates `TextProvider::speculative_defaults` and
  cannot build 0.3.0.
- Gufo API floor: `>= 0.2.15`. Derived from the upstream change records this
  extension relies on: the series-0.2 component-cache-only continuation baseline
  (`0.2.0`), reasoning delimiters (`0.2.5`), provider-defined text sampling
  defaults (`0.2.7`) and provider-declared speculative defaults (`0.2.15`).
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

## Speculative decoding (DFlash2)

**Context ceiling.** Verified serve startup: contexts up to **196608 (192k)
load**; **262144 (256k) currently hangs in session setup** (pre-existing,
unrelated to speculation — see `DEFERRED.md`). Advertise 192k until the
higher ceiling is fixed.

Flint can run DFlash2 speculative decoding: a small drafter proposes several
tokens, and the target verifies them in one batched pass. Accepted tokens are
emitted without extra target steps, so single-stream decode is faster. It is
**opt-in and off by default**.

Enable it on the process that loads the model (Gufo or llama-swap env):

```sh
FLINT_SPEC=dflash                                   # dflash | off | null (null = test mode)
FLINT_DRAFT=/path/to/Flint-DFlash2-Q6_K.gguf        # drafter; default name Flint-DFlash2-Q6_K.gguf
```

Drafter files (the default is Q6_K; the names are the published DFlash2 drafter
GGUFs, and the path is whatever you point `FLINT_DRAFT` at):

| File | Size | Notes |
| --- | --- | --- |
| `Flint-DFlash2-Q6_K.gguf` | 336 MB | default; GX 3/3 |
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

**When off, output is unchanged.** With `FLINT_SPEC` unset or `off`, no drafter
is loaded and no speculative cycle runs. The null drafter gate (72/72) confirms
the cycle itself does not change output.

**Sampled mode caveat.** Greedy (`temp 0`) output is byte-identical with
speculation on or off (12/12 at depths 0, 8k, 32k, 131k). At `temp > 0` the
accept/reject step consumes RNG draws differently, so a fixed seed does **not**
guarantee identical output with speculation on. The sampling distribution is
preserved by design, but field validation of sampled output is still open.

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
