// DFlash2 drafter weights loader. Vendored from
// src/models/qwen/dflash_weights.cpp (MIT); see the header for the two
// deliberate differences and for why the rest is the in-tree logic verbatim.

#include "src/gpu/dflash/dflash_weights.hpp"

#include <cstring>
#include <limits>
#include <string_view>
#include <utility>

#include "gufo/ext/experimental/ggml_dequant.hpp"

namespace gufo::models::flint::gpu::dflash {
namespace {

namespace quant = ::gufo::quant;

/// Binds a mapped tensor and records how much storage follows it, so a bogus
/// element count cannot read past the mapping.
[[nodiscard]] DflashTensor TensorRef(const core::GgufReader& reader,
                                    std::string_view name) {
  const auto* tensor = reader.FindTensor(name);
  if (tensor == nullptr || tensor->data == nullptr)
    return {};

  const auto tensor_address = reinterpret_cast<std::uintptr_t>(tensor->data);
  std::size_t available_bytes = 0;
  for (const auto& region : reader.GetMappedRegions()) {
    const auto region_address = reinterpret_cast<std::uintptr_t>(region.data);
    if (tensor_address < region_address)
      continue;
    const auto offset = tensor_address - region_address;
    if (offset < region.size) {
      available_bytes = region.size - offset;
      break;
    }
  }
  return {.data = tensor->data,
          .type = tensor->type,
          .num_elements = tensor->ElementCount(),
          .available_bytes = available_bytes};
}

void SetError(std::string* error_msg, std::string message) {
  if (error_msg != nullptr)
    *error_msg = std::move(message);
}

/// Types the draft GEMM tier reads as stored. Anything else (Q5_K, Q5_1, F16,
/// the IQ family) is accepted by the loader and host-dequantized to BF16 at
/// upload: drafting is bandwidth bound, so a non-native type costs memory, not
/// correctness.
[[nodiscard]] bool IsNativeMatrixType(core::GgmlType type) noexcept {
  return type == core::GgmlType::kF32 || type == core::GgmlType::kBF16 ||
         type == core::GgmlType::kQ8_0 || type == core::GgmlType::kQ4_K ||
         type == core::GgmlType::kQ6_K;
}

[[nodiscard]] bool IsLoadableMatrixType(core::GgmlType type) noexcept {
  return IsNativeMatrixType(type) || type == core::GgmlType::kQ5_K;
}

[[nodiscard]] bool IsSupportedVectorType(core::GgmlType type) noexcept {
  return type == core::GgmlType::kF32 || type == core::GgmlType::kBF16;
}

[[nodiscard]] bool BindRequiredTensor(const core::GgufReader& reader,
                                      std::string_view name,
                                      std::size_t expected_elements,
                                      bool matrix, DflashTensor& destination,
                                      std::string* error_msg) {
  destination = TensorRef(reader, name);
  if (destination.empty()) {
    SetError(error_msg, "DFlash-2 GGUF is missing required tensor '" +
                            std::string(name) + "'");
    return false;
  }
  if (destination.num_elements != expected_elements) {
    SetError(error_msg, "DFlash-2 tensor '" + std::string(name) + "' has " +
                            std::to_string(destination.num_elements) +
                            " elements; expected " +
                            std::to_string(expected_elements));
    return false;
  }
  const bool supported = matrix ? IsLoadableMatrixType(destination.type)
                                : IsSupportedVectorType(destination.type);
  if (!supported) {
    SetError(error_msg, "DFlash-2 tensor '" + std::string(name) +
                            "' has unsupported type " +
                            std::string(core::ToString(destination.type)));
    return false;
  }
  if (!destination.FitsAvailableStorage()) {
    SetError(error_msg, "DFlash-2 tensor '" + std::string(name) +
                            "' exceeds its mapped GGUF storage");
    return false;
  }
  return true;
}

[[nodiscard]] std::optional<std::uint32_t> RequiredUint32(
    const core::GgufReader& reader, std::string_view key,
    std::string* error_msg) {
  auto value = reader.GetMetadataUint32(key);
  if (!value.has_value()) {
    SetError(error_msg, "DFlash-2 GGUF is missing required metadata '" +
                            std::string(key) + "'");
  }
  return value;
}

[[nodiscard]] std::optional<std::vector<std::uint64_t>>
GetNonnegativeIntegerArray(const core::GgufReader& reader,
                           std::string_view key) {
  const auto* metadata = reader.FindMetadata(key);
  if (metadata == nullptr)
    return std::nullopt;
  if (const auto* values =
          std::get_if<std::vector<std::uint64_t>>(&metadata->value)) {
    return *values;
  }
  if (const auto* values =
          std::get_if<std::vector<std::int64_t>>(&metadata->value)) {
    std::vector<std::uint64_t> converted;
    converted.reserve(values->size());
    for (const std::int64_t value : *values) {
      if (value < 0)
        return std::nullopt;
      converted.push_back(static_cast<std::uint64_t>(value));
    }
    return converted;
  }
  return std::nullopt;
}

struct RotaryDim {
  std::uint32_t value{0};
  /// True when the file named no rotary width and `head_dim` was used.
  bool from_head_dim{false};
};

/// Explicit rotary width, the degenerate four-section M-RoPE layout, or the
/// `head_dim` fallback this export needs.
[[nodiscard]] RotaryDim ResolveRotaryDim(const core::GgufReader& reader,
                                        std::string_view prefix,
                                        std::uint32_t head_dim) noexcept {
  if (const auto value =
          reader.GetMetadataUint32(std::string(prefix) + "rope.dimension_count"))
    return {*value, false};
  const auto sections = GetNonnegativeIntegerArray(
      reader, std::string(prefix) + "rope.dimension_sections");
  if (sections.has_value()) {
    const auto& values = *sections;
    if (values.size() == 4 && values[0] > 0 && values[1] == 0 &&
        values[2] == 0 && values[3] == 0 &&
        values[0] <= std::numeric_limits<std::uint32_t>::max() / 2U)
      return {static_cast<std::uint32_t>(values[0] * 2U), false};
  }
  // The reason this file differs from the in-tree loader: the export carries no
  // dflash.rope.* dimensions, and llama.cpp falls back to the head width, so
  // requiring the metadata would reject every Flint drafter.
  return {head_dim, true};
}

}  // namespace

std::size_t DflashTensor::EncodedSizeBytes() const noexcept {
  return quant::EncodedSizeBytes(type, num_elements);
}

bool DflashTensor::FitsAvailableStorage() const noexcept {
  const std::size_t encoded_bytes = EncodedSizeBytes();
  return encoded_bytes != 0 && encoded_bytes <= available_bytes;
}

float DflashTensor::Get(std::size_t index) const noexcept {
  if (data == nullptr || index >= num_elements)
    return std::numeric_limits<float>::quiet_NaN();
  if (type == core::GgmlType::kF32)
    return static_cast<const float*>(data)[index];
  if (type == core::GgmlType::kBF16) {
    const auto bits = static_cast<std::uint32_t>(
                          static_cast<const std::uint16_t*>(data)[index])
                      << 16U;
    float value = 0.0F;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
  }
  return std::numeric_limits<float>::quiet_NaN();
}

std::optional<DflashWeights> DflashWeights::LoadFromGguf(
    const core::GgufReader& reader, std::string* error_msg) {
  const auto architecture = reader.GetMetadataString("general.architecture");
  if (!architecture.has_value() || *architecture != "dflash") {
    SetError(error_msg, "DFlash-2 GGUF requires general.architecture='dflash'");
    return std::nullopt;
  }

  DflashWeights weights;
  DflashConfig& config = weights.config;
  const std::string prefix = "dflash.";

  const auto hidden_size =
      RequiredUint32(reader, prefix + "embedding_length", error_msg);
  const auto intermediate_size =
      RequiredUint32(reader, prefix + "feed_forward_length", error_msg);
  const auto num_attention_heads =
      RequiredUint32(reader, prefix + "attention.head_count", error_msg);
  const auto num_key_value_heads =
      RequiredUint32(reader, prefix + "attention.head_count_kv", error_msg);
  const auto head_dim =
      RequiredUint32(reader, prefix + "attention.key_length", error_msg);
  const auto context_length =
      RequiredUint32(reader, prefix + "context_length", error_msg);
  const auto num_layers =
      RequiredUint32(reader, prefix + "block_count", error_msg);
  const auto rope_theta = reader.GetMetadataFloat32(prefix + "rope.freq_base");
  if (!hidden_size || !intermediate_size || !num_attention_heads ||
      !num_key_value_heads || !head_dim || !context_length || !num_layers ||
      !rope_theta) {
    if (!rope_theta) {
      SetError(
          error_msg,
          "DFlash-2 GGUF is missing required metadata 'dflash.rope.freq_base'");
    }
    return std::nullopt;
  }

  const auto rotary = ResolveRotaryDim(reader, prefix, *head_dim);
  config.hidden_size = *hidden_size;
  config.intermediate_size = *intermediate_size;
  config.num_attention_heads = *num_attention_heads;
  config.num_key_value_heads = *num_key_value_heads;
  config.head_dim = *head_dim;
  config.rotary_dim = rotary.value;
  config.rope_theta = *rope_theta;
  config.context_length = *context_length;
  config.num_layers = *num_layers;
  config.rotary_from_head_dim = rotary.from_head_dim;
  if (const auto eps =
          reader.GetMetadataFloat32(prefix + "attention.layer_norm_rms_epsilon");
      eps.has_value() && *eps > 0.0F) {
    config.rms_norm_eps = *eps;
  }

  if (config.hidden_size == 0 || config.intermediate_size == 0 ||
      config.num_attention_heads == 0 || config.num_key_value_heads == 0 ||
      config.head_dim == 0 || config.num_layers == 0 ||
      config.num_attention_heads % config.num_key_value_heads != 0 ||
      config.rotary_dim == 0 || config.rotary_dim > config.head_dim ||
      (config.rotary_dim % 2U) != 0) {
    SetError(error_msg, "DFlash-2 GGUF has an invalid transformer topology");
    return std::nullopt;
  }

  const auto block_size =
      RequiredUint32(reader, prefix + "block_size", error_msg);
  const auto conv_kernel_size =
      RequiredUint32(reader, prefix + "conv_kernel_size", error_msg);
  const auto conv_group_size =
      RequiredUint32(reader, prefix + "conv_group_size", error_msg);
  const auto selector_rank =
      RequiredUint32(reader, prefix + "selector_rank", error_msg);
  const auto selector_top_k =
      RequiredUint32(reader, prefix + "selector_top_k", error_msg);
  const auto sliding_window =
      RequiredUint32(reader, prefix + "attention.sliding_window", error_msg);
  const auto mask_token_id =
      RequiredUint32(reader, "tokenizer.ggml.mask_token_id", error_msg);
  const auto is_causal = reader.GetMetadataBool(prefix + "attention.causal");
  if (!block_size || !conv_kernel_size || !conv_group_size || !selector_rank ||
      !selector_top_k || !sliding_window || !mask_token_id ||
      !is_causal.has_value()) {
    if (!is_causal.has_value()) {
      SetError(error_msg,
               "DFlash-2 GGUF is missing required metadata "
               "'dflash.attention.causal'");
    }
    return std::nullopt;
  }

  config.block_size = *block_size;
  config.conv_kernel_size = *conv_kernel_size;
  config.conv_group_size = *conv_group_size;
  config.selector_rank = *selector_rank;
  config.selector_top_k = *selector_top_k;
  config.sliding_window = *sliding_window;
  config.mask_token_id = *mask_token_id;

  if (config.block_size < 2 || config.conv_kernel_size == 0 ||
      config.conv_group_size == 0 ||
      config.hidden_size % config.conv_group_size != 0 ||
      config.selector_rank == 0 || config.selector_top_k == 0 ||
      config.selector_top_k > 16U ||
      config.sliding_window < config.block_size || *is_causal) {
    SetError(error_msg, "DFlash-2 GGUF has an invalid diffusion topology");
    return std::nullopt;
  }

  // GGUF target_layers are layer-input indices (the converter's convention);
  // the residual this drafter reads is the layer *output*, one index lower.
  const auto encoded_target_layers =
      GetNonnegativeIntegerArray(reader, prefix + "target_layers");
  if (!encoded_target_layers.has_value()) {
    SetError(error_msg,
             "DFlash-2 GGUF is missing required integer array "
             "'dflash.target_layers'");
    return std::nullopt;
  }
  if (encoded_target_layers->empty()) {
    SetError(error_msg, "DFlash-2 target layer list must not be empty");
    return std::nullopt;
  }
  config.target_layer_ids.reserve(encoded_target_layers->size());
  for (const std::uint64_t encoded_layer : *encoded_target_layers) {
    if (encoded_layer == 0 ||
        !std::in_range<std::uint32_t>(encoded_layer - 1U)) {
      SetError(error_msg,
               "DFlash-2 target layer indices must use positive GGUF "
               "layer-input numbering");
      return std::nullopt;
    }
    const auto normalized = static_cast<std::uint32_t>(encoded_layer - 1U);
    if (!config.target_layer_ids.empty() &&
        normalized <= config.target_layer_ids.back()) {
      SetError(error_msg,
               "DFlash-2 target layer indices must be strictly increasing");
      return std::nullopt;
    }
    config.target_layer_ids.push_back(normalized);
  }

  const std::size_t hidden = config.hidden_size;
  const std::size_t intermediate = config.intermediate_size;
  const std::size_t attention = config.AttentionSize();
  const std::size_t kv = config.KvSize();
  const std::size_t dynamic_coefficients = config.ConvCoefficients();

  // DFlash-2 uses the target model's embedding and output head. Private copies
  // are accepted but not required.
  if (reader.FindTensor("d2t") != nullptr) {
    SetError(error_msg,
             "DFlash-2 reduced-vocabulary d2t exports are not supported yet");
    return std::nullopt;
  }

  const auto* predecessor_tensor =
      reader.FindTensor("selector_predecessor.weight");
  if (predecessor_tensor == nullptr || predecessor_tensor->data == nullptr ||
      predecessor_tensor->ElementCount() % config.selector_rank != 0) {
    SetError(error_msg,
             "DFlash-2 selector_predecessor.weight has invalid dimensions");
    return std::nullopt;
  }
  weights.selector_predecessor = TensorRef(reader, "selector_predecessor.weight");
  if (!IsLoadableMatrixType(weights.selector_predecessor.type) ||
      !weights.selector_predecessor.FitsAvailableStorage()) {
    SetError(error_msg,
             "DFlash-2 selector_predecessor.weight has unsupported storage");
    return std::nullopt;
  }
  const std::size_t inferred_vocab =
      weights.selector_predecessor.num_elements / config.selector_rank;
  if (inferred_vocab == 0 || !std::in_range<std::uint32_t>(inferred_vocab)) {
    SetError(error_msg, "DFlash-2 selector vocabulary size is invalid");
    return std::nullopt;
  }
  config.vocab_size = static_cast<std::uint32_t>(inferred_vocab);

  if (!BindRequiredTensor(reader, "selector_successor.weight",
                          inferred_vocab * config.selector_rank, true,
                          weights.selector_successor, error_msg) ||
      !BindRequiredTensor(reader, "selector_hidden.weight",
                          hidden * config.selector_rank, true,
                          weights.selector_hidden, error_msg)) {
    return std::nullopt;
  }

  const std::size_t encoder_input = config.target_layer_ids.size() * hidden;
  if (!BindRequiredTensor(reader, "fc.weight", hidden * encoder_input, true,
                          weights.fc_projection, error_msg) ||
      !BindRequiredTensor(reader, "enc.output_norm.weight", hidden, false,
                          weights.fc_norm, error_msg) ||
      !BindRequiredTensor(reader, "output_norm.weight", hidden, false,
                          weights.output_norm, error_msg)) {
    return std::nullopt;
  }
  const auto fc_scale = TensorRef(reader, "fc.scale");
  if (!fc_scale.empty() &&
      (fc_scale.num_elements != 1 || !IsSupportedVectorType(fc_scale.type) ||
       !fc_scale.FitsAvailableStorage() || fc_scale.Get(0) != 1.0F)) {
    SetError(error_msg,
             "DFlash-2 fc.scale must be absent or a unit F32/BF16 scalar");
    return std::nullopt;
  }

  weights.layers.resize(config.num_layers);
  for (std::uint32_t i = 0; i < config.num_layers; ++i) {
    const std::string layer_prefix = "blk." + std::to_string(i) + ".";
    auto& layer = weights.layers[i];
    if (!BindRequiredTensor(reader, layer_prefix + "attn_norm.weight", hidden,
                            false, layer.attn_norm, error_msg) ||
        !BindRequiredTensor(reader, layer_prefix + "attn_q.weight",
                            attention * hidden, true, layer.attn_q, error_msg) ||
        !BindRequiredTensor(reader, layer_prefix + "attn_k.weight", kv * hidden,
                            true, layer.attn_k, error_msg) ||
        !BindRequiredTensor(reader, layer_prefix + "attn_v.weight", kv * hidden,
                            true, layer.attn_v, error_msg) ||
        !BindRequiredTensor(reader, layer_prefix + "attn_output.weight",
                            hidden * attention, true, layer.attn_output,
                            error_msg) ||
        !BindRequiredTensor(reader, layer_prefix + "attn_q_norm.weight",
                            config.head_dim, false, layer.attn_q_norm,
                            error_msg) ||
        !BindRequiredTensor(reader, layer_prefix + "attn_k_norm.weight",
                            config.head_dim, false, layer.attn_k_norm,
                            error_msg) ||
        !BindRequiredTensor(reader, layer_prefix + "ffn_norm.weight", hidden,
                            false, layer.ffn_norm, error_msg) ||
        !BindRequiredTensor(reader, layer_prefix + "ffn_gate.weight",
                            intermediate * hidden, true, layer.ffn_gate,
                            error_msg) ||
        !BindRequiredTensor(reader, layer_prefix + "ffn_up.weight",
                            intermediate * hidden, true, layer.ffn_up,
                            error_msg) ||
        !BindRequiredTensor(reader, layer_prefix + "ffn_down.weight",
                            hidden * intermediate, true, layer.ffn_down,
                            error_msg) ||
        !BindRequiredTensor(reader, layer_prefix + "attn_conv_base",
                            hidden * config.conv_kernel_size * 2U, false,
                            layer.attn_conv_base, error_msg) ||
        !BindRequiredTensor(reader, layer_prefix + "attn_conv_proj.weight",
                            dynamic_coefficients * hidden, true,
                            layer.attn_conv_projection, error_msg) ||
        !BindRequiredTensor(reader, layer_prefix + "ffn_conv_base",
                            hidden * config.conv_kernel_size * 2U, false,
                            layer.ffn_conv_base, error_msg) ||
        !BindRequiredTensor(reader, layer_prefix + "ffn_conv_proj.weight",
                            dynamic_coefficients * hidden, true,
                            layer.ffn_conv_projection, error_msg)) {
      return std::nullopt;
    }
  }

  return weights;
}

}  // namespace gufo::models::flint::gpu::dflash
