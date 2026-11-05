# Third-party notices

The Flint extension vendors code from the Gufo repository so that it can build
against an unmodified Gufo source tree. Everything below is MIT-licensed Gufo
code, vendored at Gufo main
`186a507da77bcbb3e8f0cfbd49f688f88535e9e2`, except the quantized kernel tier,
which Gufo itself derives from `ggml-org/llama.cpp` (also MIT).

## Vendored tiers

| Path | Origin | Notes |
| --- | --- | --- |
| `hip/vendor/rocm/` | Gufo `src/models/qwen38_flash_next/kernels/rocm/` | GEMM/MoE kernels and the hipBLASLt wrapper. Namespace retained as `gufo::models::qwen38_flash_next::rocm`. |
| `hip/vendor/rocm/mmq/` | Gufo `.../rocm/mmq/` | Quantized matmul kernels adapted from `ggml-org/llama.cpp`. See `hip/vendor/rocm/mmq/VENDOR.md`. |
| `hip/vendor/qwen/` | Gufo `src/models/qwen/` and `src/core/` | Qwen tokenizer, chat template, control tokens and reasoning helpers. Namespace retained as `gufo::tokenization`. |
| `hip/vendor/qwen38/` | Gufo `src/models/qwen38_flash_next/` | MTP sampling and RoPE scaling headers used by the vendored kernels. |
| `hip/vendor/vision/` | Gufo `src/models/qwen/vision/` | RoPE helper required by the vendored kernels' include closure. |

When rebasing onto a newer Gufo revision, refresh these directories from the
upstream paths, re-apply the include retargeting to `<gufo/ext/...>` public
headers and to the vendor-local paths, and re-run `flint.ops`.

## Upstream license

Gufo and the vendored code are distributed under the MIT License reproduced in
`LICENSE`. The `mmq/` tier carries llama.cpp's MIT notice in its own
`VENDOR.md`.
