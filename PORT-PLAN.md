# Flint external Gufo extension — port plan

## Target

Standalone repo `flint-gufo-extension` that builds against **unmodified Gufo
main** (`6cc8a42ca84438a0566ee498a315f35c6190dc8e`, v0.31.0, extension API
series 0.2) via `-DGUFO_EXTENSION_DIRS=`. Follows `docs/extensions.md` and
mirrors `gufo-org/lfm3-gufo-extension`.

## Public API boundary (from include/gufo/ext/SURFACE)

Allowed and used:
- `gufo/ext/text.hpp` (`TextModelRunner`, `HipTextModelRunner`, `TextProvider`,
  `TextRunnerOptions`, `TextPromptOptions`)
- `gufo/ext/registry.hpp` (`Registry::AddProvider`, `AddTextProvider`)
- `gufo/ext/gguf_reader.hpp` -> forwards to `src/core/gguf_reader.hpp`
- `gufo/ext/experimental/ggml_dequant.hpp` -> `src/core/quant/ggml_dequant.hpp`
- `gufo/ext/experimental/ggml_gemm.hpp` -> `src/core/quant/ggml_gemm.hpp`
  (dequantize only; NOT the routed GEMM)
- `gufo/ext/cache.hpp`, `continuation.hpp`, `sha256.hpp`, `host_budget.hpp`

Not available (must vendor):
- `qwen38_flash_next::rocm` GEMM/MoE kernels (14 symbols).

## Vendored kernel closure

Flint uses 14 symbols, all declared in 3 headers, implemented in 3 cpp files:

| Header | Impl | Lines |
|---|---|---|
| `kernels/rocm/kernels.hpp` | `kernels.hip.cpp` | 674 / 8482 |
| `kernels/rocm/blaslt.hpp` | `blaslt.cpp` | 60 / 201 |
| `kernels/rocm/mmq/qfn_mmq.h` | `qfn_mmq.hip.cpp` (+ mmq/*) | 120 / 788 |

Symbols: `WeightType`, `DenseF16Gemm`, `DenseHalfWeightGemm`, `DequantizeHalf`,
`ExpertCounts`, `Q8TiledBytes`, `QuantizeQ8Tiled`, `RoutedCompact`,
`RoutedCompactRows`, `RoutedF16Gemm`, `RoutedGatedF16Gemm`,
`RoutedGatedF16Supported`, `W8A8Gemm`, `BlasLt`.

`kernels.hip.cpp` bundles unrelated kernels (HcMix, penalty sampling, Yarn
rope). Two options:
1. Vendor whole files verbatim, re-namespaced `flint_vendor::rocm` (fast,
   ~9.5k lines but self-contained).
2. Prune to the used closure (smaller, riskier).

Decision: **vendor whole files** first to get a green build, prune later.

## Repo layout

```
flint-gufo-extension/
  extension.json         # name flint, key gguf:flint, provider flint
  extension.cmake        # gufo_flint, gufo_flint_hip, gufo_flint_text + tests
  extension.cpp          # flint_register
  provider.cpp           # flint_provider_register -> AddProvider + AddTextProvider
  probe.cpp/.hpp         # ProbeFlint, ProbeFlintGguf (no GPU)
  runner.cpp/.hpp        # CreateFlintTextRunner (lifted from strycast)
  drivers.cpp            # RunFlintBench/Prompt/Chat
  src/{chat_template,reference,runtime/config}.{hpp,cpp}
  src/gpu/{engine,device_model}.{hpp,cpp}
  src/gpu/kernels/kernels.{hpp,hip.cpp}   # Flint's own 26 launchers
  hip/vendor/rocm/{kernels,blaslt,mmq/*}  # vendored qwen38 GEMM
  tests/{config,probe,snapshot}_test.cpp
  tools/flint/           # existing HTTP harness + probes
  docs/{README,QUALITY,BENCHMARKS}.md
  flake.nix, flake.lock, LICENSE
```

## Include retargets

| From | To |
|---|---|
| `src/core/gguf_reader.hpp` | `gufo/ext/gguf_reader.hpp` |
| `src/core/quant/ggml_dequant.hpp` | `gufo/ext/experimental/ggml_dequant.hpp` |
| `src/models/qwen38_flash_next/kernels/rocm/...` | `hip/vendor/rocm/...` |
| `src/models/flint/...` | repo-root-relative `...` |

## Status

- [x] Gufo main cloned at 6cc8a42, extension API series 0.2 (migrated from the 186a507d series-0.1 pin)
- [x] Public API boundary mapped
- [x] Vendor closure identified
- [x] Vendor kernel files (rocm GEMM/MoE, mmq, blaslt)
- [x] Vendor tokenizer tier (qwen tokenizer + chat template + reasoning + control tokens)
- [x] Retarget includes (zero private `"src/` includes in vendored tier and in src/)
- [x] Write extension glue (json/cmake/cpp/provider/probe)
- [x] Lift runner + drivers
- [x] Host tier compiles against Gufo main; flint.config + flint.probe pass
- [x] HIP tier compiles (`-fsyntax-only`) against Gufo main + ROCm headers

## Verification (this session)

Independently reproduced, not just self-reported:

- `provider.cpp`, `runner.cpp`, `drivers.cpp`, `src/tokenizer_support.cpp` all
  pass `g++ -std=c++20 -fsyntax-only` with `-I<gufo-root> -I/opt/rocm/include`
  and `ENGINE_ENABLE_HIP=1`.
- `tests/config_test.cpp` and `tests/probe_test.cpp` build against Gufo's
  `gguf_reader.cpp` + `platform/mapped_file.cpp` + `quant/ggml_dequant.cpp` and
  both pass.

Not yet verified: a full `cmake --preset gpu-test -DGUFO_EXTENSION_DIRS=...`
build of the `gufo` host with the extension linked, and any real-model run.

## Open items

- `extension.json` declares `flint_ops_test`; no matching CMake target exists.
  Either add the target or drop the manifest entry.
- `QwenChatTemplate::IsOrnithTemplate` is defined by `src/tokenizer_support.cpp`
  (SHA-256 identity check) rather than vendored into `hip/vendor/qwen/`.
  Flint never reads the flag; consider vendoring the original definition.
- `extension.json` declares the provider, so `ProviderInfo` is generated;
  `provider.cpp` only calls `AddTextProvider`. Confirm against the host's
  generated-registry expectation (LFM3's provider.cpp differs here).
