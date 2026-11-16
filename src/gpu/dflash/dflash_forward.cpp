// DFlash2 draft forward: one block of proposals for one session. See the
// header for the op mapping and for the one stage whose code differs from the
// oracle's.

#include "src/gpu/dflash/dflash_forward.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

#include "hip/vendor/quant/ggml_dequant.hpp"
#include "hip/vendor/rocm/kernels.hpp"
#include "hip/vendor/rocm/mmq/flint_mmq.h"
#include "src/gpu/dflash/dflash_kernels.hpp"
#include "src/gpu/kernels/kernels.hpp"

namespace gufo::models::flint::gpu::dflash {
namespace {

/// The engine's private kernels (norm, residual add) and the vendored Qwen tier
/// (embedding, NEOX rotary, projections).
namespace fk = ::gufo::models::flint::gpu::rocm;
namespace qk = ::gufo::vendor::qwen38_flash_next::rocm;
namespace vq = ::gufo::vendor::quant;

/// Rows the routed projection accepts per launch (the engine's kVecBatch).
constexpr std::uint32_t kVecRows = 8;

/// Rows one block ever holds: the anchor plus `block_size - 1` proposals. The
/// artifact's block size is 8, and the projections below run in groups of eight.
constexpr std::uint32_t kMaxRows = 8;

void SetError(std::string* error_msg, std::string message) {
  if (error_msg != nullptr)
    *error_msg = std::move(message);
}

void Check(hipError_t error, std::string* error_msg, const char* what) {
  if (error != hipSuccess)
    SetError(error_msg, std::string("DFlash2 ") + what + ": " +
                            hipGetErrorString(error));
}

/// Round-to-nearest-even float -> BF16, for the host-side repack.
std::uint16_t ToBf16(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const std::uint32_t rounding = 0x7FFFU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>((bits + rounding) >> 16U);
}

/// One drafter weight in device memory.
struct DeviceWeight {
  void* data{nullptr};
  core::GgmlType type{core::GgmlType::kF32};

  [[nodiscard]] bool empty() const noexcept { return data == nullptr; }
};

/// True for the types the draft GEMM tier reads as stored.
bool IsNativeMatrix(core::GgmlType type) noexcept {
  return type == core::GgmlType::kF32 || type == core::GgmlType::kBF16 ||
         type == core::GgmlType::kQ8_0 || type == core::GgmlType::kQ4_K ||
         type == core::GgmlType::kQ6_K;
}

}  // namespace

struct DflashForward::Impl {
  DflashConfig config;
  hipStream_t stream{nullptr};
  DflashBorrowed borrowed;

  std::vector<DeviceWeight> vectors;  // norms, always F32
  DeviceWeight fc_projection, fc_norm, output_norm;
  /// The target's LM head, read as stored.
  DeviceWeight output_weight;
  DeviceWeight selector_predecessor, selector_successor, selector_hidden;

  struct Layer {
    DeviceWeight attn_norm, attn_q, attn_k, attn_v, attn_output;
    DeviceWeight attn_q_norm, attn_k_norm, ffn_norm, ffn_gate, ffn_up, ffn_down;
    DeviceWeight attn_conv_base, attn_conv_projection;
    DeviceWeight ffn_conv_base, ffn_conv_projection;
  };
  std::vector<Layer> layers;

  // Block workspace, sized for kMaxRows.
  float *hidden{nullptr}, *normed{nullptr}, *conv{nullptr}, *dynamic{nullptr};
  float *query{nullptr}, *key{nullptr}, *value{nullptr}, *attention{nullptr};
  float *projected{nullptr}, *gate{nullptr}, *up{nullptr}, *down{nullptr};
  float *logits{nullptr}, *selector{nullptr};
  /// Context-injection scratch: the block workspace is block-sized (8 rows),
  /// while injection runs in 256-row chunks.
  float *context_features{nullptr}, *ctx_hidden{nullptr}, *ctx_normed{nullptr};
  float *ctx_key{nullptr}, *ctx_value{nullptr};
  /// Traced context stages, sized for every row at once. Null unless a trace
  /// was requested; only the parity harness asks for them.
  float* trace_normed{nullptr};
  std::vector<float*> trace_key, trace_value;
  float *confidences{nullptr}, *probabilities{nullptr}, *uniforms{nullptr};
  float *partial_scores{nullptr};
  std::uint32_t *tokens{nullptr}, *candidates{nullptr}, *partial_ids{nullptr};
  std::uint32_t* position_device{nullptr};
  /// Device zeros: the "no expert selected" id set the routed projection needs
  /// to behave as a dense one.
  std::int32_t* zero_ids{nullptr};
  std::string project_error;

  /// Host staging for traced stages. The trace callback is handed host memory,
  /// as in the in-tree executor: the stage lives in device memory, and reading
  /// it from the host is not what a device pointer means.
  ///
  /// The copy is ordered on this block's own stream. A blocking hipMemcpy on
  /// the null stream would not be: `stream` is created non-blocking, so it is
  /// not implicitly synchronized with the null stream, and a dump could then
  /// read a buffer the launch had not written yet.
  std::vector<float> trace_host;
  void Emit(const DflashTrace& trace, std::string_view name, const float* device,
            std::size_t count) {
    if (!trace)
      return;
    trace_host.resize(count);
    Check(hipMemcpyAsync(trace_host.data(), device, count * sizeof(float),
                         hipMemcpyDeviceToHost, stream),
          nullptr, "trace copy failed");
    Check(hipStreamSynchronize(stream), nullptr, "trace synchronize failed");
    trace(name, trace_host.data(), count);
  }
  std::vector<void*> allocations;

  // Injected target context: one K/V ring per draft layer.
  std::vector<float*> injected_k, injected_v;
  std::uint32_t history_capacity{0};
  std::uint32_t injected_context_len{0};

  ~Impl() {
    if (stream != nullptr)
      (void)hipStreamSynchronize(stream);
    for (void* allocation : allocations)
      (void)hipFree(allocation);
    if (stream != nullptr)
      (void)hipStreamDestroy(stream);
  }

  [[nodiscard]] void* Allocate(std::size_t bytes, std::string* error_msg) {
    void* pointer = nullptr;
    if (Check(hipMalloc(&pointer, bytes), error_msg, "allocation failed");
        pointer == nullptr)
      return nullptr;
    allocations.push_back(pointer);
    return pointer;
  }

  [[nodiscard]] float* AllocateFloats(std::size_t count, std::string* e) {
    return static_cast<float*>(Allocate(count * sizeof(float), e));
  }

  /// Uploads one weight row-major with `cols` elements per row. Types the draft
  /// GEMM tier reads stay packed; everything else (Q5_K) is host-dequantized to
  /// BF16, which costs device memory but keeps drafting correct.
  ///
  /// `force_bf16` widens every artifact type, including the native ones. The
  /// selector codebooks need it: the selector kernel reads them as BF16
  /// unconditionally, so an artifact that stores them Q6_K (or Q8_0) has to be
  /// widened here or the kernel reads quant blocks as half floats.
  [[nodiscard]] bool Upload(const DflashTensor& source, std::uint32_t cols,
                            DeviceWeight* destination, std::string* error_msg,
                            bool force_bf16 = false) {
    if (source.empty() || cols == 0) {
      SetError(error_msg, "DFlash2 drafter tensor is missing");
      return false;
    }
    const std::size_t rows = source.num_elements / cols;
    if (rows * cols != source.num_elements || !source.FitsAvailableStorage()) {
      SetError(error_msg, "DFlash2 drafter tensor has an invalid shape");
      return false;
    }
    if (!force_bf16 && IsNativeMatrix(source.type)) {
      const std::size_t bytes = source.EncodedSizeBytes();
      void* device = Allocate(bytes, error_msg);
      if (device == nullptr)
        return false;
      Check(hipMemcpy(device, source.data, bytes, hipMemcpyHostToDevice),
            error_msg, "weight upload failed");
      *destination = {device, source.type};
      return true;
    }
    // Host repack: dequantize a row at a time to keep peak host memory to one
    // row's worth of floats, then round to BF16.
    std::vector<float> row(cols);
    std::vector<std::uint16_t> packed(rows * cols);
    const auto* bytes = static_cast<const unsigned char*>(source.data);
    const std::size_t row_bytes = vq::EncodedSizeBytes(source.type, cols);
    if (row_bytes == 0) {
      SetError(error_msg, "DFlash2 drafter tensor has an unsupported type");
      return false;
    }
    for (std::size_t r = 0; r < rows; ++r) {
      if (!vq::DequantizeRow(source.type, bytes + r * row_bytes, row.data(),
                             cols)) {
        SetError(error_msg, "DFlash2 drafter tensor dequantization failed");
        return false;
      }
      for (std::uint32_t i = 0; i < cols; ++i)
        packed[r * cols + i] = ToBf16(row[i]);
    }
    void* device = Allocate(packed.size() * sizeof(std::uint16_t), error_msg);
    if (device == nullptr)
      return false;
    Check(hipMemcpy(device, packed.data(), packed.size() * sizeof(std::uint16_t),
                    hipMemcpyHostToDevice),
          error_msg, "weight upload failed");
    *destination = {device, core::GgmlType::kBF16};
    return true;
  }

  /// Vectors are small, so every norm gamma is widened to F32 once: the forward
  /// then never branches on a scalar's storage type.
  [[nodiscard]] bool UploadVector(const DflashTensor& source,
                                  DeviceWeight* destination,
                                  std::string* error_msg) {
    if (source.empty()) {
      SetError(error_msg, "DFlash2 drafter norm is missing");
      return false;
    }
    std::vector<float> values(source.num_elements);
    for (std::size_t i = 0; i < values.size(); ++i)
      values[i] = source.Get(i);
    float* device = AllocateFloats(values.size(), error_msg);
    if (device == nullptr)
      return false;
    Check(hipMemcpy(device, values.data(), values.size() * sizeof(float),
                    hipMemcpyHostToDevice),
          error_msg, "norm upload failed");
    *destination = {device, core::GgmlType::kF32};
    return true;
  }

  [[nodiscard]] bool UploadLayer(const DflashLayerWeights& source, Layer* layer,
                                 std::string* error_msg) {
    const std::uint32_t hidden = config.hidden_size;
    const std::uint32_t intermediate = config.intermediate_size;
    const std::uint32_t query = config.AttentionSize();
    // The injected context projects through K and V over whole 256-row
    // chunks, the one place the in-tree executor widens these two matrices to
    // BF16 regardless of the artifact's quant. Mirroring it keeps the context
    // keys on the same arithmetic as the oracle and costs ~40 MB.
    return UploadVector(source.attn_norm, &layer->attn_norm, error_msg) &&
           Upload(source.attn_q, hidden, &layer->attn_q, error_msg) &&
           Upload(source.attn_k, hidden, &layer->attn_k, error_msg,
                  /*force_bf16=*/true) &&
           Upload(source.attn_v, hidden, &layer->attn_v, error_msg,
                  /*force_bf16=*/true) &&
           Upload(source.attn_output, query, &layer->attn_output, error_msg) &&
           UploadVector(source.attn_q_norm, &layer->attn_q_norm, error_msg) &&
           UploadVector(source.attn_k_norm, &layer->attn_k_norm, error_msg) &&
           UploadVector(source.ffn_norm, &layer->ffn_norm, error_msg) &&
           Upload(source.ffn_gate, hidden, &layer->ffn_gate, error_msg) &&
           Upload(source.ffn_up, hidden, &layer->ffn_up, error_msg) &&
           Upload(source.ffn_down, intermediate, &layer->ffn_down, error_msg) &&
           UploadVector(source.attn_conv_base, &layer->attn_conv_base,
                        error_msg) &&
           Upload(source.attn_conv_projection, hidden,
                  &layer->attn_conv_projection, error_msg) &&
           UploadVector(source.ffn_conv_base, &layer->ffn_conv_base,
                        error_msg) &&
           Upload(source.ffn_conv_projection, hidden,
                  &layer->ffn_conv_projection, error_msg);
  }

  /// Y[rows][m] = X[rows][k] * W[m][k]^T. The routes are the engine's own
  /// (`Session::Impl::Dense`): the MMQ dense kernel for Q8_0, the routed vector
  /// projection with a single unselected expert for every other K-quant (it
  /// reads the weights as stored, so a Q6_K drafter stays Q6_K), and the
  /// family's small-batch GEMV for unquantized weights.
  ///
  /// A failure lands in `project_error`, which Run/AppendContext report: these
  /// shapes are validated at Create, so a failure here is a device fault, not a
  /// caller error.
  void Project(const DeviceWeight& weight, const float* input, float* output,
               std::uint32_t rows, std::uint32_t m, std::uint32_t k) {
    if (!project_error.empty() || rows == 0 || m == 0 || k == 0)
      return;
    if (weight.type == core::GgmlType::kF32 ||
        weight.type == core::GgmlType::kBF16) {
      qk::SmallGemm(weight.data,
                    weight.type == core::GgmlType::kF32 ? qk::WeightType::kF32
                                                        : qk::WeightType::kBF16,
                    input, output, rows, m, k, stream);
      return;
    }
    if (weight.type == core::GgmlType::kQ8_0) {
      if (flint_mmq_q8_0_dense(weight.data, input, output,
                               static_cast<int>(m), static_cast<int>(rows),
                               static_cast<int>(k), stream) != 0) {
        project_error = "Q8_0 dense projection failed";
      }
      return;
    }
    for (std::uint32_t r0 = 0; r0 < rows; r0 += kVecRows) {
      const std::uint32_t count = std::min(kVecRows, rows - r0);
      const int rc = flint_mmq_moe_vec(
          static_cast<int>(weight.type), weight.data, input + std::size_t{r0} * k,
          zero_ids, output + std::size_t{r0} * m, static_cast<int>(m),
          static_cast<int>(k), static_cast<int>(count), 1, 1, stream, nullptr,
          nullptr);
      if (rc != 0)
        project_error = "quantized dense projection failed";
    }
  }

  void Norm(const float* input, const DeviceWeight& gamma, float* output,
            std::uint32_t rows) {
    fk::RmsNormRows(input, static_cast<const float*>(gamma.data), output, rows,
                    config.hidden_size, 1, config.rms_norm_eps, stream);
  }

  /// Per-head RMSNorm on Q and K, then NEOX rotary at `position`.
  void ApplyQkNormRope(float* query, float* key, const Layer& layer,
                       std::uint32_t rows, std::uint32_t position) {
    const std::uint32_t head_dim = config.head_dim;
    fk::RmsNormRows(query, static_cast<const float*>(layer.attn_q_norm.data),
                    query, rows * config.num_attention_heads, head_dim, 1,
                    config.rms_norm_eps, stream);
    fk::RmsNormRows(key, static_cast<const float*>(layer.attn_k_norm.data), key,
                    rows * config.num_key_value_heads, head_dim, 1,
                    config.rms_norm_eps, stream);
    Check(hipMemcpyAsync(position_device, &position, sizeof(position),
                         hipMemcpyHostToDevice, stream),
          nullptr, "position upload failed");
    qk::Rope(query, rows, config.num_attention_heads, head_dim,
             config.rotary_dim, position_device, config.rope_theta, stream);
    qk::Rope(key, rows, config.num_key_value_heads, head_dim, config.rotary_dim,
             position_device, config.rope_theta, stream);
  }

  /// Per-head RMSNorm on K only, then NEOX rotary: the injected context has no
  /// queries.
  void ApplyKNormRope(float* key, const Layer& layer, std::uint32_t rows,
                      std::uint32_t position) {
    const std::uint32_t head_dim = config.head_dim;
    fk::RmsNormRows(key, static_cast<const float*>(layer.attn_k_norm.data), key,
                    rows * config.num_key_value_heads, head_dim, 1,
                    config.rms_norm_eps, stream);
    Check(hipMemcpyAsync(position_device, &position, sizeof(position),
                         hipMemcpyHostToDevice, stream),
          nullptr, "position upload failed");
    qk::Rope(key, rows, config.num_key_value_heads, head_dim, config.rotary_dim,
             position_device, config.rope_theta, stream);
  }

  /// The target's embedding table, in whatever encoding it is stored.
  [[nodiscard]] bool EmbedTokens(const std::uint32_t* tokens, std::uint32_t rows,
                                 float* out) {
    const auto* weights = static_cast<const void*>(borrowed.token_embedding);
    const auto type = borrowed.token_embedding_type;
    qk::WeightType vendor{};
    switch (type) {
      case core::GgmlType::kF32:
        vendor = qk::WeightType::kF32;
        break;
      case core::GgmlType::kF16:
        vendor = qk::WeightType::kF16;
        break;
      case core::GgmlType::kBF16:
        vendor = qk::WeightType::kBF16;
        break;
      case core::GgmlType::kQ8_0:
        vendor = qk::WeightType::kQ8_0;
        break;
      case core::GgmlType::kQ4_K:
        vendor = qk::WeightType::kQ4_K;
        break;
      case core::GgmlType::kQ6_K:
        vendor = qk::WeightType::kQ6_K;
        break;
      default:
        return false;
    }
    qk::EmbedTokens(weights, vendor,
                    reinterpret_cast<const std::int32_t*>(tokens), out, rows,
                    config.hidden_size, 1, stream);
    return true;
  }
};

DflashForward::DflashForward() : impl_(std::make_unique<Impl>()) {}
DflashForward::~DflashForward() = default;

std::unique_ptr<DflashForward> DflashForward::Create(
    const DflashWeights& weights, const DflashBorrowed& borrowed,
    std::uint32_t max_context, std::string* error_msg,
    DflashHeadRoute head_route) {
  auto forward = std::unique_ptr<DflashForward>(new DflashForward());
  Impl& s = *forward->impl_;
  s.config = weights.config;
  s.borrowed = borrowed;
  s.output_weight = {const_cast<void*>(borrowed.output),
                     borrowed.output_type};
  const DflashConfig& c = s.config;

  if (borrowed.token_embedding == nullptr || borrowed.output == nullptr) {
    SetError(error_msg, "DFlash2 drafter needs the target's embedding and head");
    return nullptr;
  }
  if (Check(hipStreamCreateWithFlags(&s.stream, hipStreamNonBlocking), error_msg,
            "stream creation failed");
      s.stream == nullptr)
    return nullptr;
  // The MMQ dense projection needs its per-device context, exactly as the
  // engine's sessions initialize it.
  if (flint_mmq_init(0) != 0) {
    SetError(error_msg, "DFlash2 MMQ initialization failed");
    return nullptr;
  }

  const std::uint32_t hidden = c.hidden_size;
  const std::uint32_t query = c.AttentionSize();
  const std::uint32_t kv = c.KvSize();
  const std::uint32_t intermediate = c.intermediate_size;
  const std::uint32_t dynamic = c.ConvCoefficients();
  const std::uint32_t rows = std::max(kMaxRows, c.block_size + 1U);

  s.hidden = s.AllocateFloats(std::size_t{rows} * hidden, error_msg);
  s.normed = s.AllocateFloats(std::size_t{rows} * hidden, error_msg);
  s.conv = s.AllocateFloats(std::size_t{rows} * hidden, error_msg);
  s.dynamic = s.AllocateFloats(std::size_t{rows} * dynamic, error_msg);
  s.query = s.AllocateFloats(std::size_t{rows} * query, error_msg);
  s.key = s.AllocateFloats(std::size_t{rows} * kv, error_msg);
  s.value = s.AllocateFloats(std::size_t{rows} * kv, error_msg);
  s.attention = s.AllocateFloats(std::size_t{rows} * query, error_msg);
  s.projected = s.AllocateFloats(std::size_t{rows} * hidden, error_msg);
  s.gate = s.AllocateFloats(std::size_t{rows} * intermediate, error_msg);
  s.up = s.AllocateFloats(std::size_t{rows} * intermediate, error_msg);
  s.down = s.AllocateFloats(std::size_t{rows} * hidden, error_msg);
  s.logits = s.AllocateFloats(std::size_t{rows} * c.vocab_size, error_msg);
  s.selector = s.AllocateFloats(std::size_t{rows} * c.selector_rank, error_msg);
  s.confidences = s.AllocateFloats(rows, error_msg);
  s.uniforms = s.AllocateFloats(rows, error_msg);
  s.probabilities =
      s.AllocateFloats(std::size_t{rows} * c.selector_top_k, error_msg);
  // The selector's partial top-k scratch is per row: the batch launcher runs a
  // grid of (partial blocks, rows) and offsets each row by a full block set.
  const std::size_t selector_scratch =
      DFlashSelectorScratchElements(c.vocab_size);
  s.partial_scores =
      s.AllocateFloats(std::size_t{rows} * selector_scratch, error_msg);
  s.tokens = static_cast<std::uint32_t*>(
      s.Allocate(std::size_t{rows} * sizeof(std::uint32_t), error_msg));
  s.candidates = static_cast<std::uint32_t*>(
      s.Allocate(std::size_t{rows} * c.selector_top_k * sizeof(std::uint32_t),
                 error_msg));
  s.partial_ids = static_cast<std::uint32_t*>(s.Allocate(
      std::size_t{rows} * selector_scratch * sizeof(std::uint32_t),
      error_msg));
  s.position_device = static_cast<std::uint32_t*>(
      s.Allocate(sizeof(std::uint32_t), error_msg));
  s.zero_ids = static_cast<std::int32_t*>(
      s.Allocate(std::size_t{rows} * sizeof(std::int32_t), error_msg));
  Check(hipMemset(s.zero_ids, 0, std::size_t{rows} * sizeof(std::int32_t)),
        error_msg, "id set initialization failed");

  // Context ring: the attention window is all the drafter ever reads.
  s.history_capacity = std::max(std::uint32_t{1},
                                std::min(max_context == 0 ? c.sliding_window
                                                          : max_context,
                                         c.sliding_window));
  s.injected_k.resize(c.num_layers);
  s.injected_v.resize(c.num_layers);
  for (std::uint32_t i = 0; i < c.num_layers; ++i) {
    s.injected_k[i] = s.AllocateFloats(std::size_t{s.history_capacity} * kv,
                                       error_msg);
    s.injected_v[i] = s.AllocateFloats(std::size_t{s.history_capacity} * kv,
                                       error_msg);
  }

  std::string error;
  bool ok =
      s.UploadVector(weights.fc_norm, &s.fc_norm, &error) &&
      s.UploadVector(weights.output_norm, &s.output_norm, &error) &&
      // The encoder's row length is the fused feature width, not the hidden
      // size: `fc.weight` is [hidden, taps * hidden].
      s.Upload(weights.fc_projection,
               static_cast<std::uint32_t>(weights.config.target_layer_ids.size()) *
                   hidden,
               &s.fc_projection, &error) &&
      // The codebooks are [vocab][rank], read element-wise as BF16 by the
      // selector kernel, so rank is their row length and BF16 their storage.
      s.Upload(weights.selector_predecessor, c.selector_rank,
               &s.selector_predecessor, &error, /*force_bf16=*/true) &&
      s.Upload(weights.selector_successor, c.selector_rank,
               &s.selector_successor, &error, /*force_bf16=*/true) &&
      s.Upload(weights.selector_hidden, hidden, &s.selector_hidden, &error);
  s.layers.resize(c.num_layers);
  for (std::uint32_t i = 0; ok && i < c.num_layers; ++i)
    ok = s.UploadLayer(weights.layers[i], &s.layers[i], &error);
  if (!ok) {
    SetError(error_msg, error);
    return nullptr;
  }

  // The parity route: one F32 copy of the borrowed head, so the head's
  // arithmetic stops being the comparison's floor. The shipped route keeps the
  // target's own storage and the engine's projection for it.
  if (head_route == DflashHeadRoute::kFp32) {
    std::vector<float> head(std::size_t{c.vocab_size} * hidden);
    const std::size_t row_bytes =
        vq::EncodedSizeBytes(borrowed.output_type, hidden);
    if (row_bytes == 0) {
      SetError(error_msg, "DFlash2 cannot read the target head's storage type");
      return nullptr;
    }
    const auto* bytes = static_cast<const unsigned char*>(borrowed.output);
    for (std::uint32_t row = 0; row < c.vocab_size; ++row) {
      if (!vq::DequantizeRow(borrowed.output_type, bytes + row * row_bytes,
                             head.data() + std::size_t{row} * hidden, hidden)) {
        SetError(error_msg, "DFlash2 target head dequantization failed");
        return nullptr;
      }
    }
    float* device = s.AllocateFloats(head.size(), error_msg);
    if (device == nullptr)
      return nullptr;
    if (hipMemcpy(device, head.data(), head.size() * sizeof(float),
                  hipMemcpyHostToDevice) != hipSuccess) {
      SetError(error_msg, "DFlash2 target head upload failed");
      return nullptr;
    }
    s.output_weight = {device, core::GgmlType::kF32};
  }
  return forward;
}

const DflashConfig& DflashForward::config() const noexcept {
  return impl_->config;
}

std::uint32_t DflashForward::ContextLength() const noexcept {
  return impl_->injected_context_len;
}

void DflashForward::ResetContext() noexcept {
  impl_->injected_context_len = 0;
}

std::uint32_t DflashForward::HistoryCapacity() const noexcept {
  return impl_->history_capacity;
}

const float* DflashForward::InjectedKey(std::uint32_t layer) const noexcept {
  return layer < impl_->injected_k.size() ? impl_->injected_k[layer] : nullptr;
}

const float* DflashForward::InjectedValue(std::uint32_t layer) const noexcept {
  return layer < impl_->injected_v.size() ? impl_->injected_v[layer] : nullptr;
}

bool DflashForward::AppendContext(std::span<const float> features,
                                  std::uint32_t position, std::uint32_t count,
                                  std::string* error_msg,
                                  const DflashTrace& trace) {
  Impl& s = *impl_;
  const DflashConfig& c = s.config;
  const std::uint32_t hidden = c.hidden_size;
  const std::uint32_t kv = c.KvSize();
  const std::uint32_t row = static_cast<std::uint32_t>(
      std::size_t{c.target_layer_ids.size()} * hidden);
  if (position != s.injected_context_len || count == 0) {
    SetError(error_msg, "DFlash2 context rows are not contiguous");
    return false;
  }
  if (features.size() != std::size_t{count} * row) {
    SetError(error_msg, "DFlash2 context feature size mismatch");
    return false;
  }
  // The oracle chunks the injection at 256 rows; the batch boundaries decide
  // which GEMM route each chunk takes, so keep them.
  constexpr std::uint32_t kContextChunkRows = 256;
  const std::uint32_t chunk_rows = std::min(kContextChunkRows, count);
  if (s.context_features == nullptr) {
    // Scratch is sized for the cap, not this call's chunk: a later call with
    // a larger count uses up to kContextChunkRows rows per batch, and sizing
    // from the first call's count would overflow these buffers.
    s.context_features =
        s.AllocateFloats(std::size_t{kContextChunkRows} * row, error_msg);
    s.ctx_hidden =
        s.AllocateFloats(std::size_t{kContextChunkRows} * hidden, error_msg);
    s.ctx_normed =
        s.AllocateFloats(std::size_t{kContextChunkRows} * hidden, error_msg);
    s.ctx_key = s.AllocateFloats(std::size_t{kContextChunkRows} * kv, error_msg);
    s.ctx_value =
        s.AllocateFloats(std::size_t{kContextChunkRows} * kv, error_msg);
    if (s.context_features == nullptr || s.ctx_hidden == nullptr ||
        s.ctx_normed == nullptr || s.ctx_key == nullptr ||
        s.ctx_value == nullptr)
      return false;
  }
  // A traced run dumps the whole context as one tensor per stage, the way the
  // oracle records it, so the encoder output and each layer's injected K and V
  // get a contiguous buffer sized for every row. Untraced runs allocate none of
  // this: the parity harness is the only consumer.
  if (trace) {
    s.trace_normed = s.AllocateFloats(std::size_t{count} * hidden, error_msg);
    s.trace_key.resize(c.num_layers);
    s.trace_value.resize(c.num_layers);
    for (std::uint32_t i = 0; i < c.num_layers; ++i) {
      s.trace_key[i] = s.AllocateFloats(std::size_t{count} * kv, error_msg);
      s.trace_value[i] = s.AllocateFloats(std::size_t{count} * kv, error_msg);
    }
    if (s.trace_normed == nullptr)
      return false;
    for (std::uint32_t i = 0; i < c.num_layers; ++i) {
      if (s.trace_key[i] == nullptr || s.trace_value[i] == nullptr)
        return false;
    }
  }
  // A chunk the ring would overwrite before anything reads it is skipped; the
  // surviving rows keep their absolute positions and batch boundaries.
  const auto expired = count > s.history_capacity ? count - s.history_capacity
                                                  : std::uint32_t{0};
  // A traced run must fill the whole context tensor the harness compares, so it
  // cannot skip the rows the ring would overwrite; an untraced run can.
  const auto first_chunk = trace ? std::uint32_t{0} : (expired / chunk_rows) * chunk_rows;
  for (std::uint32_t offset = first_chunk; offset < count;) {
    const auto batch = std::min(chunk_rows, count - offset);
    const auto* source = features.data() + std::size_t{offset} * row;
    Check(hipMemcpyAsync(s.context_features, source,
                         std::size_t{batch} * row * sizeof(float),
                         hipMemcpyHostToDevice, s.stream),
          error_msg, "context upload failed");
    s.Project(s.fc_projection, s.context_features, s.ctx_hidden, batch, hidden,
              row);
    s.Norm(s.ctx_hidden, s.fc_norm, s.ctx_normed, batch);
    if (trace) {
      Check(hipMemcpyAsync(s.trace_normed + std::size_t{offset} * hidden,
                           s.ctx_normed, std::size_t{batch} * hidden * sizeof(float),
                           hipMemcpyDeviceToDevice, s.stream),
            error_msg, "context trace copy failed");
    }
    for (std::uint32_t i = 0; i < c.num_layers; ++i) {
      const auto& layer = s.layers[i];
      s.Project(layer.attn_k, s.ctx_normed, s.ctx_key, batch, kv, hidden);
      s.Project(layer.attn_v, s.ctx_normed, s.ctx_value, batch, kv, hidden);
      s.ApplyKNormRope(s.ctx_key, layer, batch, position + offset);
      if (trace) {
        Check(hipMemcpyAsync(s.trace_key[i] + std::size_t{offset} * kv,
                             s.ctx_key,
                             std::size_t{batch} * kv * sizeof(float),
                             hipMemcpyDeviceToDevice, s.stream),
              error_msg, "context trace copy failed");
        Check(hipMemcpyAsync(s.trace_value[i] + std::size_t{offset} * kv,
                             s.ctx_value,
                             std::size_t{batch} * kv * sizeof(float),
                             hipMemcpyDeviceToDevice, s.stream),
              error_msg, "context trace copy failed");
      }
      // Split a write that crosses the physical end of the ring.
      for (std::uint32_t written = 0; written < batch;) {
        const auto slot = (position + offset + written) % s.history_capacity;
        const auto run =
            std::min(batch - written, s.history_capacity - slot);
        const std::size_t bytes = std::size_t{run} * kv * sizeof(float);
        Check(hipMemcpyAsync(s.injected_k[i] + std::size_t{slot} * kv,
                             s.ctx_key + std::size_t{written} * kv, bytes,
                             hipMemcpyDeviceToDevice, s.stream),
              error_msg, "context key write failed");
        Check(hipMemcpyAsync(s.injected_v[i] + std::size_t{slot} * kv,
                             s.ctx_value + std::size_t{written} * kv, bytes,
                             hipMemcpyDeviceToDevice, s.stream),
              error_msg, "context value write failed");
        written += run;
      }
    }
    offset += batch;
  }
  if (!s.project_error.empty()) {
    SetError(error_msg, s.project_error);
    s.project_error.clear();
    return false;
  }
  if (trace) {
    // Stage names match the oracle's: one encoder output and, per layer, the
    // injected keys after the per-head norm and rotary, and the raw values.
    s.Emit(trace, "context.0.normalized", s.trace_normed,
           std::size_t{count} * hidden);
    for (std::uint32_t i = 0; i < c.num_layers; ++i) {
      const std::string prefix = "context.0." + std::to_string(i) + ".";
      s.Emit(trace, prefix + "k", s.trace_key[i], std::size_t{count} * kv);
      s.Emit(trace, prefix + "v", s.trace_value[i], std::size_t{count} * kv);
    }
  }
  s.injected_context_len = std::max(s.injected_context_len, position + count);
  return true;
}

bool DflashForward::Run(std::uint32_t anchor, std::uint32_t position,
                        std::uint32_t draft_count, float temperature,
                        std::span<const float> uniforms,
                        const DflashTrace& trace, DflashDraft* draft,
                        std::string* error_msg) {
  Impl& s = *impl_;
  const DflashConfig& c = s.config;
  const std::uint32_t hidden = c.hidden_size;
  const std::uint32_t intermediate = c.intermediate_size;
  const std::uint32_t query = c.AttentionSize();
  const std::uint32_t kv = c.KvSize();
  const std::uint32_t dynamic = c.ConvCoefficients();
  const std::uint32_t head_dim = c.head_dim;
  const float scale = 1.0F / std::sqrt(static_cast<float>(head_dim));

  draft_count = std::min(draft_count, c.block_size > 0 ? c.block_size - 1U : 0U);
  if (draft == nullptr || draft_count == 0) {
    SetError(error_msg, "DFlash2 block needs at least one proposal");
    return false;
  }
  if (anchor >= c.vocab_size || c.mask_token_id >= c.vocab_size) {
    SetError(error_msg,
             "DFlash2 anchor or mask token is outside the vocabulary");
    return false;
  }
  if (position != s.injected_context_len) {
    SetError(error_msg,
             "DFlash2 block position is ahead of the injected context");
    return false;
  }
  if (temperature > 0.0F && uniforms.size() < draft_count) {
    SetError(error_msg, "DFlash2 sampled drafting needs one draw per proposal");
    return false;
  }
  const std::uint32_t rows = draft_count + 1U;
  const auto emit = [&](std::string_view name, const float* data,
                        std::size_t width, std::uint32_t row_count) {
    s.Emit(trace, name, data, std::size_t{row_count} * width);
  };
  const auto emit_block = [&](std::string_view name, const float* data,
                              std::size_t width) {
    emit(name, data, width, rows);
  };
  const auto emit_proposals = [&](std::string_view name, const float* data,
                                  std::size_t width) {
    emit(name, data, width, draft_count);
  };

  // Slot zero is the selector's predecessor; proposals start as masks.
  std::vector<std::uint32_t> block_tokens(rows, c.mask_token_id);
  block_tokens.front() = anchor;
  Check(hipMemcpyAsync(s.tokens, block_tokens.data(),
                       std::size_t{rows} * sizeof(std::uint32_t),
                       hipMemcpyHostToDevice, s.stream),
        error_msg, "token upload failed");
  if (!s.EmbedTokens(s.tokens, rows, s.hidden)) {
    SetError(error_msg, "DFlash2 cannot read the target's embedding type");
    return false;
  }
  emit_block("embedding", s.hidden, hidden);

  // The draft decoder blocks.
  for (std::uint32_t i = 0; i < c.num_layers; ++i) {
    const auto& layer = s.layers[i];
    const auto stage = [&](std::string_view name, const float* data,
                           std::size_t width) {
      s.Emit(trace, "layer." + std::to_string(i) + "." + std::string(name),
             data, std::size_t{rows} * width);
    };

    s.Norm(s.hidden, layer.attn_norm, s.normed, rows);
    stage("attn_norm", s.normed, hidden);
    s.Project(layer.attn_conv_projection, s.normed, s.dynamic, rows, dynamic,
              hidden);
    LaunchDFlashGroupedDynamicConv(
        s.normed, s.dynamic,
        static_cast<const float*>(layer.attn_conv_base.data), s.conv, rows,
        hidden, c.conv_kernel_size, c.conv_group_size, 0, s.stream);
    stage("attn_conv_in", s.conv, hidden);

    s.Project(layer.attn_q, s.conv, s.query, rows, query, hidden);
    s.Project(layer.attn_k, s.conv, s.key, rows, kv, hidden);
    s.Project(layer.attn_v, s.conv, s.value, rows, kv, hidden);
    s.ApplyQkNormRope(s.query, s.key, layer, rows, position);
    stage("q", s.query, query);
    stage("k", s.key, kv);
    stage("v", s.value, kv);

    LaunchDFlashNonCausalAttention(
        s.query, s.injected_k[i], s.injected_v[i], s.key, s.value, s.attention,
        position, std::min(position, s.injected_context_len), rows,
        c.sliding_window, c.num_attention_heads, c.num_key_value_heads, head_dim,
        scale, s.stream, s.history_capacity);
    stage("attention", s.attention, query);

    s.Project(layer.attn_output, s.attention, s.projected, rows, hidden, query);
    stage("attn_output", s.projected, hidden);
    LaunchDFlashGroupedDynamicConv(
        s.projected, s.dynamic,
        static_cast<const float*>(layer.attn_conv_base.data), s.conv, rows,
        hidden, c.conv_kernel_size, c.conv_group_size, 1, s.stream);
    stage("attn_conv_out", s.conv, hidden);
    fk::AddInPlace(s.hidden, s.conv, std::size_t{rows} * hidden, s.stream);
    stage("attn_residual", s.hidden, hidden);

    s.Norm(s.hidden, layer.ffn_norm, s.normed, rows);
    stage("ffn_norm", s.normed, hidden);
    s.Project(layer.ffn_conv_projection, s.normed, s.dynamic, rows, dynamic,
              hidden);
    LaunchDFlashGroupedDynamicConv(
        s.normed, s.dynamic,
        static_cast<const float*>(layer.ffn_conv_base.data), s.conv, rows,
        hidden, c.conv_kernel_size, c.conv_group_size, 0, s.stream);
    stage("ffn_conv_in", s.conv, hidden);

    s.Project(layer.ffn_gate, s.conv, s.gate, rows, intermediate, hidden);
    s.Project(layer.ffn_up, s.conv, s.up, rows, intermediate, hidden);
    LaunchDFlashSiLUMul(s.gate, s.up, rows * intermediate, s.stream);
    s.Project(layer.ffn_down, s.gate, s.down, rows, hidden, intermediate);
    stage("ffn_down", s.down, hidden);
    LaunchDFlashGroupedDynamicConv(
        s.down, s.dynamic, static_cast<const float*>(layer.ffn_conv_base.data),
        s.conv, rows, hidden, c.conv_kernel_size, c.conv_group_size, 1,
        s.stream);
    stage("ffn_conv_out", s.conv, hidden);
    fk::AddInPlace(s.hidden, s.conv, std::size_t{rows} * hidden, s.stream);
    stage("output", s.hidden, hidden);
  }

  s.Norm(s.hidden, s.output_norm, s.normed, rows);
  emit_block("normalized", s.normed, hidden);

  // Both the lattice and the vocabulary projection read proposals only: the
  // anchor's row is finished.
  const float* proposal_hidden = s.normed + hidden;
  s.Project(s.selector_hidden, proposal_hidden, s.selector, draft_count,
            c.selector_rank, hidden);
  emit_proposals("selector_hidden", s.selector, c.selector_rank);
  s.Project(s.output_weight, proposal_hidden, s.logits, draft_count,
            c.vocab_size, hidden);
  emit_proposals("logits", s.logits, c.vocab_size);

  if (temperature > 0.0F) {
    Check(hipMemcpyAsync(s.uniforms, uniforms.data(),
                         std::size_t{draft_count} * sizeof(float),
                         hipMemcpyHostToDevice, s.stream),
          error_msg, "sampling draw upload failed");
  }
  const DFlashSelectorSequence sequence{draft_count, temperature,
                                        DFlashSampling{},
                                        DFlashSelectorVocab{}};
  LaunchDFlashSelectorBatch(
      s.logits, s.selector, s.selector_predecessor.data,
      s.selector_successor.data, s.tokens, s.confidences, s.partial_scores,
      s.partial_ids, temperature > 0.0F ? s.uniforms : nullptr, s.candidates,
      s.probabilities, std::span(&sequence, 1), c.vocab_size, c.selector_rank,
      c.selector_top_k, s.stream);

  draft->tokens.resize(draft_count);
  draft->confidences.resize(draft_count);
  Check(hipMemcpyAsync(draft->tokens.data(), s.tokens + 1,
                       std::size_t{draft_count} * sizeof(std::uint32_t),
                       hipMemcpyDeviceToHost, s.stream),
        error_msg, "draft download failed");
  Check(hipMemcpyAsync(draft->confidences.data(), s.confidences,
                       std::size_t{draft_count} * sizeof(float),
                       hipMemcpyDeviceToHost, s.stream),
        error_msg, "confidence download failed");
  const std::size_t candidate_count =
      std::size_t{draft_count} * c.selector_top_k;
  draft->candidate_ids.clear();
  draft->candidate_probabilities.clear();
  if (temperature > 0.0F) {
    draft->candidate_ids.resize(candidate_count);
    draft->candidate_probabilities.resize(candidate_count);
    Check(hipMemcpyAsync(draft->candidate_ids.data(), s.candidates,
                         candidate_count * sizeof(std::uint32_t),
                         hipMemcpyDeviceToHost, s.stream),
          error_msg, "candidate download failed");
    Check(hipMemcpyAsync(draft->candidate_probabilities.data(), s.probabilities,
                         candidate_count * sizeof(float),
                         hipMemcpyDeviceToHost, s.stream),
          error_msg, "candidate probability download failed");
  }
  Check(hipStreamSynchronize(s.stream), error_msg, "block synchronize failed");
  if (!s.project_error.empty()) {
    SetError(error_msg, s.project_error);
    s.project_error.clear();
    return false;
  }
  return error_msg == nullptr || error_msg->empty();
}

}  // namespace gufo::models::flint::gpu::dflash
