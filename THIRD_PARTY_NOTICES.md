# Third-party notices

The Flint extension vendors code from the Gufo repository so that it can build
against an unmodified Gufo source tree. Everything below is MIT-licensed Gufo
code, vendored at Gufo main
`186a507da77bcbb3e8f0cfbd49f688f88535e9e2`, except the quantized kernel tier,
which Gufo itself derives from `ggml-org/llama.cpp` (also MIT).

## Flint model implementation

The Flint model implementation itself (the runner, engine, chat template and
tokenizer support) is derived from [`mikealanni/gufo`](https://github.com/mikealanni/gufo),
an MIT-licensed Gufo fork that adds Flint 40B support. This extension adapts
that implementation to the external extension contract and to a Gufo main
tree: the model, kernel and tokenizer code is carried over largely as-is, with
the fork's NPU-specific sections removed (the extension targets the HIP/GPU
path on gfx1151). Later sections of this file cover the Gufo upstream tiers the
implementation depends on.

## Vendored tiers

| Path | Origin | Notes |
| --- | --- | --- |
| `hip/vendor/rocm/` | Gufo `src/models/qwen38_flash_next/kernels/rocm/` | GEMM/MoE kernels and the hipBLASLt wrapper. |
| `hip/vendor/rocm/mmq/` | Gufo `.../rocm/mmq/` | Quantized matmul kernels adapted from `ggml-org/llama.cpp`. See `hip/vendor/rocm/mmq/VENDOR.md`. |
| `hip/vendor/qwen/` | Gufo `src/families/qwen/` and `src/core/` | Qwen tokenizer, chat template, control tokens and reasoning helpers. |
| `hip/vendor/qwen_kernels/` | Gufo `src/families/qwen/kernels/`, `src/families/qwen/hip_common/`, `src/core/hip/` | Wave64 W8A8 GEMM launcher and its private header closure. |
| `hip/vendor/qwen38/` | Gufo `src/models/qwen38_flash_next/` | MTP sampling and RoPE scaling headers used by the vendored kernels. |
| `hip/vendor/vision/` | Gufo `src/models/qwen/vision/` | RoPE helper required by the vendored kernels' include closure. |
| `hip/vendor/quant/` | Gufo `src/core/quant/` | Quantized block layouts, dequant/dot tables and IQ grids. |
| `tests/gguf_test_shim.hpp` | `gufo-org/lfm3-gufo-extension` `tests/gguf_test_shim.hpp` | Header-only `GgufReader` implementation used by tests, following the LFM3 extension pattern. |

## Namespace adaptation

Gufo compiles extensions into the same executable as the in-tree models, so
identical symbol names would collide at link time. Vendored code therefore
keeps its implementation but lives under extension-owned namespaces:

| Vendored namespace | Original namespace |
| --- | --- |
| `gufo::vendor::qwen38_flash_next::rocm` | `gufo::models::qwen38_flash_next::rocm` |
| `gufo::vendor::qwen38_flash_next` | `gufo::models::qwen38_flash_next` |
| `gufo::vendor::qwen::tokenization` | `gufo::tokenization` |
| `gufo::vendor::qwen::vision` | `gufo::models::qwen::vision` |
| `gufo::vendor::quant` | `gufo::quant` |
| `flint_mmq`, `flint_ggml_stubs` (files `flint_mmq*.h`, `flint_ggml_stubs.*`) | `qfn_mmq`, `qfn_ggml_stubs` |

When rebasing onto a newer Gufo revision, refresh these directories from the
upstream paths, re-apply the include retargeting to `<gufo/ext/...>` public
headers and to the vendor-local paths, re-apply the namespace mapping above,
and re-run `flint.ops`.

## Upstream license

Gufo and the vendored code are distributed under the MIT License reproduced in
`LICENSE`. The `mmq/` tier carries llama.cpp's MIT notice in its own
`VENDOR.md`.
