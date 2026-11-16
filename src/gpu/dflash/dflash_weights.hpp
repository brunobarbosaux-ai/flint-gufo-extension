// DFlash2 drafter weights, vendored from the in-tree loader
// (src/models/qwen/dflash_weights.{hpp,cpp}, MIT, see THIRD_PARTY_NOTICES.md).
//
// Two deliberate differences from the in-tree file, both forced by the
// artifacts: the export carries no `dflash.rope.*` metadata, so the rotary
// width falls back to `head_dim` (llama.cpp's rule) instead of failing the
// load; and Q5_K is an accepted matrix type, because the upload path
// host-dequantizes every type the draft GEMM tier does not read natively.
//
// Everything else -- the metadata keys, the topology rules, the tap
// convention, the inventory and the storage checks -- is the in-tree logic, so
// `flint_drafter_probe --loader` can diff this loader against the probe's
// independent parse of the same file.
#ifndef GUFO_MODELS_FLINT_GPU_DFLASH_WEIGHTS_HPP_
#define GUFO_MODELS_FLINT_GPU_DFLASH_WEIGHTS_HPP_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "gufo/ext/gguf_reader.hpp"

namespace gufo::models::flint::gpu::dflash {

/// One draft block-diffusion model's geometry and diffusion topology.
struct DflashConfig {
  // Transformer geometry.
  std::uint32_t hidden_size{0};
  std::uint32_t intermediate_size{0};
  std::uint32_t num_attention_heads{0};
  std::uint32_t num_key_value_heads{0};
  std::uint32_t head_dim{0};
  std::uint32_t rotary_dim{0};
  float rope_theta{0.0F};
  /// Epsilon of every RMSNorm in the draft block. The artifact carries it as
  /// `dflash.attention.layer_norm_rms_epsilon`; unlike the rotary width it is
  /// optional, and llama.cpp's default applies when it is absent.
  float rms_norm_eps{1e-6F};
  std::uint32_t context_length{0};
  std::uint32_t num_layers{0};
  /// Inferred from the selector codebooks: the drafter has no head of its own.
  std::uint32_t vocab_size{0};

  /// Zero-based target layer *outputs*. GGUF stores layer *inputs* (llama.cpp's
  /// `dflash.target_layers = [6, 16, 27, 37, 47]`), so the loader subtracts
  /// one: these are the ids `Session::SetHiddenCapture` takes.
  std::vector<std::uint32_t> target_layer_ids;

  // Diffusion topology.
  /// Total diffusion block size, including the committed anchor token.
  std::uint32_t block_size{0};
  std::uint32_t mask_token_id{0};
  std::uint32_t conv_kernel_size{0};
  std::uint32_t conv_group_size{0};
  std::uint32_t selector_rank{0};
  std::uint32_t selector_top_k{0};
  /// Attention is non-causal inside the block and bounded by this window.
  std::uint32_t sliding_window{0};

  /// True when the GGUF named no rotary width and `head_dim` was used. Not an
  /// error; recorded so a diagnostic can say which rule was applied.
  bool rotary_from_head_dim{false};

  [[nodiscard]] std::uint32_t AttentionSize() const noexcept {
    return num_attention_heads * head_dim;
  }
  [[nodiscard]] std::uint32_t KvSize() const noexcept {
    return num_key_value_heads * head_dim;
  }
  [[nodiscard]] std::uint32_t ConvGroups() const noexcept {
    return conv_group_size == 0 ? 0 : hidden_size / conv_group_size;
  }
  /// Dynamic-convolution coefficients per token, both sides.
  [[nodiscard]] std::uint32_t ConvCoefficients() const noexcept {
    return 2U * conv_kernel_size * ConvGroups();
  }
};

/// Non-owning reference to a tensor in the mapped GGUF.
struct DflashTensor {
  const void* data{nullptr};
  core::GgmlType type{core::GgmlType::kF32};
  std::size_t num_elements{0};
  /// Bytes from `data` to the end of its mapped region.
  std::size_t available_bytes{0};

  [[nodiscard]] bool empty() const noexcept {
    return data == nullptr || num_elements == 0;
  }
  [[nodiscard]] std::size_t EncodedSizeBytes() const noexcept;
  [[nodiscard]] bool FitsAvailableStorage() const noexcept;
  /// Element `index` as a float. Only the F32 and BF16 vectors the loader gates
  /// on are supported: the scalars it reads are always one of those.
  [[nodiscard]] float Get(std::size_t index) const noexcept;
};

/// The draft transformer block's projections plus its two layer-local dynamic
/// convolutions.
struct DflashLayerWeights {
  DflashTensor attn_norm;
  DflashTensor attn_q, attn_k, attn_v, attn_output;
  DflashTensor attn_q_norm, attn_k_norm;
  DflashTensor ffn_norm, ffn_gate, ffn_up, ffn_down;
  /// [hidden, conv_kernel, 2] static kernel table.
  DflashTensor attn_conv_base, ffn_conv_base;
  /// [hidden, 2 * conv_kernel * (hidden / conv_group)] dynamic coefficients.
  DflashTensor attn_conv_projection, ffn_conv_projection;
};

/// The whole drafter. The token embedding and the LM head are the target's:
/// the drafter borrows both, so they are not part of this artifact.
struct DflashWeights {
  DflashConfig config;

  /// Feature fusion encoder: compresses the target's tapped residuals.
  DflashTensor fc_projection;  // [layers * hidden, hidden]
  DflashTensor fc_norm;        // [hidden]
  DflashTensor output_norm;    // [hidden]

  /// Low-rank candidate path selector.
  DflashTensor selector_predecessor;  // [vocab, rank]
  DflashTensor selector_successor;    // [vocab, rank]
  DflashTensor selector_hidden;       // [hidden, rank]

  std::vector<DflashLayerWeights> layers;

  [[nodiscard]] static std::optional<DflashWeights> LoadFromGguf(
      const core::GgufReader& reader, std::string* error_msg = nullptr);
};

}  // namespace gufo::models::flint::gpu::dflash

#endif  // GUFO_MODELS_FLINT_GPU_DFLASH_WEIGHTS_HPP_
