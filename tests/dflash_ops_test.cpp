// DFlash2 draft-kernel tests: every vendored kernel against a host reference on
// a shape small enough to reason about. No weights and no target model, so this
// runs before the loader exists and catches a wrong namespace, a wrong include
// or a re-derived index while the only thing that could be wrong is the
// vendoring itself.
//
// The three non-causal attention entry points are also compared against each
// other: scalar, wave and the dispatcher must agree wherever the dispatcher
// picks a route, otherwise a parity failure later lands on the wrong kernel.

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/gpu/dflash/dflash_kernels.hpp"

namespace dflash = gufo::models::flint::gpu::dflash;

namespace {

void Check(hipError_t error, const char* operation) {
  if (error != hipSuccess)
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(error));
}

/// A device buffer of `count` T, freed on scope exit.
template <typename T>
class Dev {
public:
  explicit Dev(std::size_t count)
      : count_(count), owned_(true) {
    Check(hipMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)),
          "hipMalloc");
  }
  ~Dev() {
    if (owned_)
      (void)hipFree(data_);
  }
  Dev(const Dev&) = delete;
  Dev& operator=(const Dev&) = delete;

  [[nodiscard]] T* get() const noexcept { return data_; }
  [[nodiscard]] std::size_t count() const noexcept { return count_; }

  void Upload(const std::vector<T>& values) {
    if (values.size() != count_)
      throw std::runtime_error("Dev::Upload size mismatch");
    Check(hipMemcpy(data_, values.data(), count_ * sizeof(T),
                    hipMemcpyHostToDevice),
          "hipMemcpy host to device");
  }
  [[nodiscard]] std::vector<T> Download() const {
    std::vector<T> values(count_);
    Check(hipMemcpy(values.data(), data_, count_ * sizeof(T),
                    hipMemcpyDeviceToHost),
          "hipMemcpy device to host");
    return values;
  }

private:
  T* data_{nullptr};
  std::size_t count_{0};
  bool owned_{false};
};

/// Values in [-1, 1) with no exact ties: a mismatch is always a real one.
std::vector<float> Data(std::size_t count, float scale = 1.0F) {
  std::vector<float> values(count);
  for (std::size_t i = 0; i < count; ++i) {
    const auto mixed = (i * 2654435761U) % 1000U;
    values[i] = (static_cast<float>(mixed) / 500.0F - 1.0F) * scale;
  }
  return values;
}

void Near(const std::vector<float>& got, const std::vector<float>& want,
          float limit, const char* label) {
  if (got.size() != want.size())
    throw std::runtime_error(std::string(label) + ": size mismatch");
  double worst = 0.0;
  std::size_t at = 0;
  for (std::size_t i = 0; i < got.size(); ++i) {
    if (std::isinf(want[i]) || std::isinf(got[i])) {
      if (std::isinf(want[i]) != std::isinf(got[i]) ||
          (std::isinf(want[i]) && (want[i] > 0) != (got[i] > 0)))
        throw std::runtime_error(std::string(label) + ": infinity mismatch at " +
                                 std::to_string(i));
      continue;
    }
    const auto diff = std::abs(static_cast<double>(got[i]) -
                               static_cast<double>(want[i]));
    if (diff > worst) {
      worst = diff;
      at = i;
    }
  }
  if (!std::isfinite(worst) || worst > limit)
    throw std::runtime_error(std::string(label) + ": max |diff| " +
                             std::to_string(worst) + " at " +
                             std::to_string(at) + " exceeds " +
                             std::to_string(limit));
}

// ---------------------------------------------------------------------------
// BF16 bit patterns. The host compiler sees hip_bfloat16 as a bare 16-bit field
// (the header only gives it conversions under __HIPCC__), so the test owns the
// conversion instead of depending on which header branch is active. Every value
// used here is a multiple of 1/64 well inside BF16's range, so the pattern is
// exact and the round trip is asserted.
// ---------------------------------------------------------------------------
std::uint16_t Bf16Bits(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return static_cast<std::uint16_t>(bits >> 16);
}

float Bf16Value(std::uint16_t bits) {
  const std::uint32_t wide = static_cast<std::uint32_t>(bits) << 16;
  float value = 0.0F;
  std::memcpy(&value, &wide, sizeof(value));
  return value;
}

std::uint16_t ExactBf16(float value) {
  const auto bits = Bf16Bits(value);
  if (Bf16Value(bits) != value)
    throw std::runtime_error("fixture value " + std::to_string(value) +
                             " is not exact in BF16");
  return bits;
}

// ---------------------------------------------------------------------------
// Grouped dynamic convolution: the draft block's only non-standard operator.
// ---------------------------------------------------------------------------
void ConvReference(const std::vector<float>& input,
                   const std::vector<float>& dynamic,
                   const std::vector<float>& base, std::uint32_t tokens,
                   std::uint32_t hidden, std::uint32_t kernel,
                   std::uint32_t group_size, std::uint32_t side,
                   std::vector<float>* out) {
  const std::uint32_t groups = hidden / group_size;
  const std::uint32_t per_token = 2U * kernel * groups;
  out->assign(std::size_t{tokens} * hidden, 0.0F);
  for (std::uint32_t t = 0; t < tokens; ++t) {
    for (std::uint32_t c = 0; c < hidden; ++c) {
      const std::uint32_t group = c / group_size;
      double sum = 0.0;
      for (std::uint32_t tap = 0; tap < kernel; ++tap) {
        if (t < tap)
          continue;
        const auto coefficient =
            static_cast<double>(base[(side * kernel + tap) * hidden + c]) +
            static_cast<double>(
                dynamic[std::size_t{t} * per_token +
                        (side * kernel + tap) * groups + group]);
        sum += coefficient * static_cast<double>(input[std::size_t{t - tap} *
                                                             hidden +
                                                         c]);
      }
      (*out)[std::size_t{t} * hidden + c] = static_cast<float>(sum);
    }
  }
}

void TestGroupedDynamicConv() {
  constexpr std::uint32_t kTokens = 6, kHidden = 32, kKernel = 2, kGroup = 4;
  const auto input = Data(std::size_t{kTokens} * kHidden);
  const auto dynamic =
      Data(std::size_t{kTokens} * 2 * kKernel * (kHidden / kGroup));
  const auto base = Data(std::size_t{2} * kKernel * kHidden);

  Dev<float> in(input.size()), dyn(dynamic.size()), kernel(base.size()),
      out(std::size_t{kTokens} * kHidden);
  in.Upload(input);
  dyn.Upload(dynamic);
  kernel.Upload(base);

  for (std::uint32_t side = 0; side < 2; ++side) {
    dflash::LaunchDFlashGroupedDynamicConv(
        in.get(), dyn.get(), kernel.get(), out.get(), kTokens, kHidden, kKernel,
        kGroup, side, nullptr);
    Check(hipStreamSynchronize(nullptr), "sync conv");
    std::vector<float> want;
    ConvReference(input, dynamic, base, kTokens, kHidden, kKernel, kGroup, side,
                  &want);
    Near(out.Download(), want, 1e-5F,
         side == 0 ? "GroupedDynamicConv.attn" : "GroupedDynamicConv.ffn");
  }
}

// ---------------------------------------------------------------------------
// SiLU-mul and the BF16 -> Q8_0 draft-head copy.
// ---------------------------------------------------------------------------
void TestSiluMul() {
  constexpr std::size_t kCount = 257;  // one partial block
  const auto gate = Data(kCount, 4.0F);
  const auto up = Data(kCount, 2.0F);
  Dev<float> gate_dev(kCount), up_dev(kCount);
  gate_dev.Upload(gate);
  up_dev.Upload(up);
  dflash::LaunchDFlashSiLUMul(gate_dev.get(), up_dev.get(),
                              static_cast<std::uint32_t>(kCount), nullptr);
  Check(hipStreamSynchronize(nullptr), "sync silu-mul");
  std::vector<float> want(kCount);
  for (std::size_t i = 0; i < kCount; ++i) {
    const double g = gate[i];
    want[i] = static_cast<float>(g / (1.0 + std::exp(-g)) * up[i]);
  }
  Near(gate_dev.Download(), want, 1e-5F, "SiLUMul");
}

void TestBf16ToQ8_0() {
  constexpr std::size_t kCount = 64;  // two Q8_0 blocks
  std::vector<std::uint16_t> source(kCount);
  std::vector<float> source_f(kCount);
  for (std::size_t i = 0; i < kCount; ++i) {
    const auto value = static_cast<float>(static_cast<int>(i % 17) - 8) / 64.0F;
    source[i] = ExactBf16(value);
    source_f[i] = Bf16Value(source[i]);
  }
  Dev<std::uint16_t> bf16(kCount);
  Dev<std::int8_t> q8(kCount * 34 / 32);
  bf16.Upload(source);
  dflash::LaunchDFlashQuantizeBf16ToQ8_0(bf16.get(), q8.get(), kCount, nullptr);
  Check(hipStreamSynchronize(nullptr), "sync quantize");
  const auto packed = q8.Download();

  // block_q8_0: {__half d; int8_t qs[32]}, 34 bytes per 32 values.
  std::vector<float> want(kCount);
  for (std::size_t block = 0; block < kCount / 32; ++block) {
    const auto* bytes = reinterpret_cast<const unsigned char*>(
        packed.data() + block * 34);
    // block_q8_0 starts with the __half scale, then 32 signed bytes.
    const float scale =
        __half2float(*reinterpret_cast<const __half*>(bytes));
    float max_abs = 0.0F;
    for (std::size_t i = 0; i < 32; ++i)
      max_abs = std::fmax(max_abs, std::fabs(source_f[block * 32 + i]));
    const float want_scale = max_abs / 127.0F;
    if (std::fabs(scale - want_scale) > 1e-3F)
      throw std::runtime_error("Bf16ToQ8_0: block " + std::to_string(block) +
                               " scale " + std::to_string(scale) +
                               " vs " + std::to_string(want_scale));
    const float inverse = want_scale > 0.0F ? 1.0F / want_scale : 0.0F;
    for (std::size_t i = 0; i < 32; ++i) {
      const auto q = static_cast<signed char>(bytes[2 + i]);
      const auto quantized =
          std::round(source_f[block * 32 + i] * inverse);
      const auto clamped = std::fmin(std::fmax(quantized, -127.0F), 127.0F);
      if (std::fabs(static_cast<float>(q) - clamped) > 0.0F)
        throw std::runtime_error("Bf16ToQ8_0: block " + std::to_string(block) +
                                 " value " + std::to_string(i) + " is " +
                                 std::to_string(q) + " vs " +
                                 std::to_string(clamped));
      want[block * 32 + i] = static_cast<float>(q) * want_scale;
    }
  }
  // Round trip: the reconstruction is within one step of the input.
  std::vector<float> got(kCount);
  for (std::size_t block = 0; block < kCount / 32; ++block) {
    const auto* bytes = reinterpret_cast<const unsigned char*>(
        packed.data() + block * 34);
    const float scale =
        __half2float(*reinterpret_cast<const __half*>(bytes));
    for (std::size_t i = 0; i < 32; ++i)
      got[block * 32 + i] =
          static_cast<float>(static_cast<signed char>(bytes[2 + i])) * scale;
  }
  // The block scale is stored as __half, so the reconstruction carries the
  // half's rounding on top of the quantized value.
  Near(got, want, 1e-5F, "Bf16ToQ8_0.roundtrip");
}

// ---------------------------------------------------------------------------
// Non-causal attention: three routes, one host reference.
// ---------------------------------------------------------------------------
struct AttentionShape {
  std::uint32_t current_pos;
  std::uint32_t history_length;
  std::uint32_t history_capacity;
  std::uint32_t block_count;
  std::uint32_t sliding_window;
  std::uint32_t heads;
  std::uint32_t kv_heads;
  std::uint32_t head_dim;
};

void AttentionReference(const AttentionShape& shape,
                        const std::vector<float>& q,
                        const std::vector<float>& injected_k,
                        const std::vector<float>& injected_v,
                        const std::vector<float>& block_k,
                        const std::vector<float>& block_v, float scale,
                        std::vector<float>* out) {
  const auto q_dim = shape.heads * shape.head_dim;
  const auto kv_dim = shape.kv_heads * shape.head_dim;
  const auto group = shape.heads / shape.kv_heads;
  const auto history_start = shape.current_pos + 1U > shape.sliding_window
                                 ? shape.current_pos + 1U - shape.sliding_window
                                 : 0U;
  const auto history_count = shape.history_length > history_start
                                 ? shape.history_length - history_start
                                 : 0U;
  out->assign(std::size_t{shape.block_count} * q_dim, 0.0F);
  for (std::uint32_t token = 0; token < shape.block_count; ++token) {
    const auto query_position = shape.current_pos + token;
    for (std::uint32_t h = 0; h < shape.heads; ++h) {
      const auto kv_h = h / group;
      const float* q_row = q.data() + std::size_t{token} * q_dim +
                           std::size_t{h} * shape.head_dim;
      std::vector<double> scores;
      scores.reserve(history_count + shape.block_count);
      for (std::uint32_t local = 0; local < history_count; ++local) {
        const auto position = history_start + local;
        const float* k_row =
            injected_k.data() +
            std::size_t{position % shape.history_capacity} * kv_dim +
            std::size_t{kv_h} * shape.head_dim;
        double dot = 0.0;
        for (std::uint32_t d = 0; d < shape.head_dim; ++d)
          dot += static_cast<double>(q_row[d]) * k_row[d];
        scores.push_back(query_position - position < shape.sliding_window
                             ? dot * scale
                             : -1e20);
      }
      for (std::uint32_t p = 0; p < shape.block_count; ++p) {
        const float* k_row = block_k.data() + std::size_t{p} * kv_dim +
                             std::size_t{kv_h} * shape.head_dim;
        double dot = 0.0;
        for (std::uint32_t d = 0; d < shape.head_dim; ++d)
          dot += static_cast<double>(q_row[d]) * k_row[d];
        const auto key_position = shape.current_pos + p;
        const auto distance = query_position > key_position
                                  ? query_position - key_position
                                  : key_position - query_position;
        scores.push_back(distance < shape.sliding_window ? dot * scale
                                                         : -1e20);
      }
      const auto max_score =
          *std::max_element(scores.begin(), scores.end());
      double denominator = 0.0;
      for (auto& score : scores) {
        score = std::exp(score - max_score);
        denominator += score;
      }
      float* out_row = out->data() + std::size_t{token} * q_dim +
                       std::size_t{h} * shape.head_dim;
      for (std::uint32_t d = 0; d < shape.head_dim; ++d) {
        double acc = 0.0;
        for (std::uint32_t local = 0; local < history_count; ++local) {
          const auto position = history_start + local;
          const float* v_row =
              injected_v.data() +
              std::size_t{position % shape.history_capacity} * kv_dim +
              std::size_t{kv_h} * shape.head_dim;
          acc += scores[local] * v_row[d];
        }
        for (std::uint32_t p = 0; p < shape.block_count; ++p) {
          const float* v_row = block_v.data() + std::size_t{p} * kv_dim +
                               std::size_t{kv_h} * shape.head_dim;
          acc += scores[history_count + p] * v_row[d];
        }
        out_row[d] = static_cast<float>(acc / denominator);
      }
    }
  }
}

void TestAttention(const AttentionShape& shape, const char* label) {
  const auto q_dim = shape.heads * shape.head_dim;
  const auto kv_dim = shape.kv_heads * shape.head_dim;
  const auto q = Data(std::size_t{shape.block_count} * q_dim);
  // The injected cache wraps, so the small capacity is exercised too.
  const auto injected_k = Data(std::size_t{shape.history_capacity} * kv_dim);
  const auto injected_v = Data(std::size_t{shape.history_capacity} * kv_dim);
  const auto block_k = Data(std::size_t{shape.block_count} * kv_dim);
  const auto block_v = Data(std::size_t{shape.block_count} * kv_dim);
  const float scale = 1.0F / std::sqrt(static_cast<float>(shape.head_dim));

  Dev<float> q_dev(q.size()), ik_dev(injected_k.size()),
      iv_dev(injected_v.size()), bk_dev(block_k.size()),
      bv_dev(block_v.size()), out_dev(std::size_t{shape.block_count} * q_dim);
  q_dev.Upload(q);
  ik_dev.Upload(injected_k);
  iv_dev.Upload(injected_v);
  bk_dev.Upload(block_k);
  bv_dev.Upload(block_v);

  std::vector<float> want;
  AttentionReference(shape, q, injected_k, injected_v, block_k, block_v, scale,
                     &want);

  const auto launch = [&](auto&& kernel) {
    kernel(q_dev.get(), ik_dev.get(), iv_dev.get(), bk_dev.get(), bv_dev.get(),
           out_dev.get(), shape.current_pos, shape.history_length,
           shape.block_count, shape.sliding_window, shape.heads,
           shape.kv_heads, shape.head_dim, scale, nullptr,
           shape.history_capacity);
    Check(hipStreamSynchronize(nullptr), "sync attention");
    return out_dev.Download();
  };

  const auto scalar = launch([&](auto... args) {
    return dflash::LaunchDFlashNonCausalAttentionScalar(args...);
  });
  const auto dispatched = launch([&](auto... args) {
    return dflash::LaunchDFlashNonCausalAttention(args...);
  });

  const std::string prefix = std::string("NonCausalAttention.") + label;
  Near(scalar, want, 1e-5F, (prefix + ".scalar").c_str());
  Near(dispatched, want, 1e-5F, (prefix + ".dispatch").c_str());
  Near(dispatched, scalar, 1e-6F, (prefix + ".routes").c_str());

  // The wave route reads each key row as float4, so it only covers a head dim
  // that is a multiple of four -- which is exactly the dispatcher's condition.
  if (shape.head_dim % 4U != 0U)
    return;
  const auto wave = launch([&](auto... args) {
    return dflash::LaunchDFlashNonCausalAttentionWave(args...);
  });
  Near(wave, want, 1e-5F, (prefix + ".wave").c_str());
  Near(wave, scalar, 1e-6F, (prefix + ".wave_routes").c_str());
}

// ---------------------------------------------------------------------------
// Row gather and draft-table scoring.
// ---------------------------------------------------------------------------
void TestGatherRows() {
  constexpr std::uint32_t kRows = 16, kRowFloats = 4, kGathered = 5;
  std::vector<float> source(std::size_t{kRows} * kRowFloats);
  for (std::uint32_t row = 0; row < kRows; ++row)
    for (std::uint32_t i = 0; i < kRowFloats; ++i)
      source[row * kRowFloats + i] = static_cast<float>(row * 10 + i);
  const std::vector<std::int32_t> ids{3, 15, 0, 9, 7};

  Dev<float> source_dev(source.size()), destination_dev(kGathered * kRowFloats);
  Dev<std::int32_t> ids_dev(ids.size());
  source_dev.Upload(source);
  ids_dev.Upload(ids);
  dflash::LaunchDFlashGatherRows(source_dev.get(), kRowFloats * sizeof(float),
                                 ids_dev.get(), destination_dev.get(),
                                 kGathered, nullptr);
  Check(hipStreamSynchronize(nullptr), "sync gather");

  std::vector<float> want;
  for (const auto id : ids)
    want.insert(want.end(), source.begin() + std::size_t{id} * kRowFloats,
                source.begin() + std::size_t{id + 1} * kRowFloats);
  Near(destination_dev.Download(), want, 0.0F, "GatherRows");
}

void TestDraftTableLogits() {
  constexpr std::uint32_t kRows = 2, kStride = 12, kTable = 6;
  const std::vector<std::int32_t> ids{11, -1, 0, 5, -1, 3};
  std::vector<float> source(std::size_t{kRows} * kStride);
  for (std::uint32_t row = 0; row < kRows; ++row)
    for (std::uint32_t i = 0; i < kStride; ++i)
      source[row * kStride + i] = static_cast<float>(row * 100 + i);
  const std::vector<float> sentinel(std::size_t{kRows} * kTable, -7.0F);

  Dev<float> source_dev(source.size());
  Dev<std::int32_t> ids_dev(ids.size());
  source_dev.Upload(source);
  ids_dev.Upload(ids);

  // With a source row: gather the table columns, empty slots become -inf.
  {
    Dev<float> table(std::size_t{kRows} * kTable);
    dflash::LaunchDFlashDraftTableLogits(source_dev.get(), kStride,
                                         ids_dev.get(), table.get(), kRows,
                                         kTable, nullptr);
    Check(hipStreamSynchronize(nullptr), "sync table logits");
    std::vector<float> want(std::size_t{kRows} * kTable);
    for (std::uint32_t row = 0; row < kRows; ++row)
      for (std::uint32_t slot = 0; slot < kTable; ++slot)
        want[row * kTable + slot] =
            ids[slot] < 0 ? -std::numeric_limits<float>::infinity()
                          : source[row * kStride + std::uint32_t{ids[slot]}];
    Near(table.Download(), want, 0.0F, "DraftTableLogits.source");
  }

  // Without one: the table's own projection survives, empty slots still -inf.
  {
    Dev<float> table(std::size_t{kRows} * kTable);
    table.Upload(sentinel);
    dflash::LaunchDFlashDraftTableLogits(nullptr, kStride, ids_dev.get(),
                                         table.get(), kRows, kTable, nullptr);
    Check(hipStreamSynchronize(nullptr), "sync table logits");
    std::vector<float> want(sentinel);
    for (std::uint32_t row = 0; row < kRows; ++row)
      for (std::uint32_t slot = 0; slot < kTable; ++slot)
        if (ids[slot] < 0)
          want[row * kTable + slot] = -std::numeric_limits<float>::infinity();
    Near(table.Download(), want, 0.0F, "DraftTableLogits.table");
  }
}

// ---------------------------------------------------------------------------
// Candidate selector: the greedy path, scored against a host top-k.
// ---------------------------------------------------------------------------
void TestSelectorGreedy() {
  constexpr std::uint32_t kVocab = 5000;  // three 2048-token partial blocks
  constexpr std::uint32_t kRank = 4, kTopK = 4;

  // Distinct logits with the global top four spread across the partial blocks,
  // so a merge that drops a block is caught.
  std::vector<float> logits(kVocab);
  for (std::uint32_t token = 0; token < kVocab; ++token)
    logits[token] = std::sin(static_cast<float>(token) * 0.37F) * 4.0F;
  const std::vector<std::uint32_t> planted{17, 2100, 4500, 900};
  for (std::size_t i = 0; i < planted.size(); ++i)
    logits[planted[i]] = 8.0F + static_cast<float>(i);

  // The transition codebooks are indexed by *token id*, so they are vocabulary
  // wide: row t is candidate t's low-rank successor vector. Values are exact in
  // BF16, so the host scores the same numbers the kernel reads.
  constexpr std::uint32_t kPredToken = 0;  // the anchor the path starts from
  std::vector<std::uint16_t> predecessor(std::size_t{kVocab} * kRank, 0);
  std::vector<std::uint16_t> successor(std::size_t{kVocab} * kRank, 0);
  std::vector<float> projected(kRank);
  std::vector<float> predecessor_f(kRank);
  std::vector<float> successor_f(successor.size());
  for (std::uint32_t r = 0; r < kRank; ++r) {
    predecessor[std::size_t{kPredToken} * kRank + r] =
        ExactBf16(0.25F * static_cast<float>(r + 1));
    predecessor_f[r] = Bf16Value(predecessor[std::size_t{kPredToken} * kRank + r]);
    projected[r] = 0.5F * static_cast<float>(r + 1);
  }
  for (std::uint32_t token = 0; token < kVocab; ++token) {
    for (std::uint32_t r = 0; r < kRank; ++r) {
      // Cycled by token id so a candidate's transition depends on the token the
      // kernel names, not on its rank in the top-k list.
      const auto value = 0.25F * static_cast<float>((token % kTopK) + 1) *
                         static_cast<float>(r + 1);
      successor[std::size_t{token} * kRank + r] = ExactBf16(value);
      successor_f[std::size_t{token} * kRank + r] =
          Bf16Value(successor[std::size_t{token} * kRank + r]);
    }
  }

  // Host top-k: the same rule the kernel's insertion list uses, but on the full
  // vocabulary, so the partial merge and the finalize merge are both checked.
  std::vector<std::uint32_t> order(kVocab);
  for (std::uint32_t token = 0; token < kVocab; ++token)
    order[token] = token;
  std::partial_sort(order.begin(), order.begin() + kTopK, order.end(),
                    [&](std::uint32_t a, std::uint32_t b) {
                      return logits[a] > logits[b];
                    });
  double best_score = -std::numeric_limits<double>::infinity();
  std::uint32_t best = 0;
  for (std::uint32_t c = 0; c < kTopK; ++c) {
    const auto candidate = order[c];
    double transition = 0.0;
    for (std::uint32_t r = 0; r < kRank; ++r)
      transition += static_cast<double>(predecessor_f[r]) * projected[r] *
                    successor_f[std::size_t{candidate} * kRank + r];
    const double score = logits[candidate] + transition;
    if (score > best_score) {
      best_score = score;
      best = candidate;
    }
  }
  // The transition has to *change* the answer, or the check would also pass on
  // a kernel that ignored the codebooks and returned the logits argmax.
  if (best == order.front())
    throw std::runtime_error("Selector.greedy: the fixture does not move the "
                             "winner off the logits argmax");

  const auto scratch = dflash::DFlashSelectorScratchElements(kVocab);
  if (scratch != 3 * 16)
    throw std::runtime_error("Selector: unexpected scratch size " +
                             std::to_string(scratch));

  Dev<float> logits_dev(kVocab), projected_dev(kRank),
      partial_scores_dev(scratch), confidence_dev(1);
  Dev<std::uint16_t> predecessor_dev(predecessor.size()),
      successor_dev(successor.size());
  Dev<std::uint32_t> partial_ids_dev(scratch), predecessor_token_dev(1),
      output_token_dev(1), candidate_ids_dev(kTopK);
  Dev<float> candidate_probabilities_dev(kTopK);
  logits_dev.Upload(logits);
  projected_dev.Upload(projected);
  predecessor_dev.Upload(predecessor);
  successor_dev.Upload(successor);
  predecessor_token_dev.Upload({kPredToken});

  dflash::LaunchDFlashSelectorStep(
      logits_dev.get(), projected_dev.get(), predecessor_dev.get(),
      successor_dev.get(), predecessor_token_dev.get(), output_token_dev.get(),
      confidence_dev.get(), partial_scores_dev.get(), partial_ids_dev.get(),
      0.0F, /*sample_uniform=*/nullptr, candidate_ids_dev.get(),
      candidate_probabilities_dev.get(), kVocab, kRank, kTopK, nullptr);
  Check(hipStreamSynchronize(nullptr), "sync selector");

  const auto chosen = output_token_dev.Download()[0];
  if (chosen != best)
    throw std::runtime_error("Selector.greedy: chose " +
                             std::to_string(chosen) + ", want " +
                             std::to_string(best));
  const auto confidence = confidence_dev.Download()[0];
  if (confidence != 1.0F)
    throw std::runtime_error("Selector.greedy: confidence " +
                             std::to_string(confidence) + " is not 1");

  auto candidates = candidate_ids_dev.Download();
  std::vector<std::uint32_t> wanted(order.begin(), order.begin() + kTopK);
  std::sort(candidates.begin(), candidates.end());
  std::sort(wanted.begin(), wanted.end());
  if (candidates != wanted)
    throw std::runtime_error("Selector.greedy: candidate set differs from the "
                             "host top-k");
  const auto probabilities = candidate_probabilities_dev.Download();
  std::size_t ones = 0;
  for (const auto probability : probabilities)
    ones += probability == 1.0F;
  if (ones != 1)
    throw std::runtime_error("Selector.greedy: candidate probabilities are not "
                             "one-hot");
}

}  // namespace

int main() {
  int devices = 0;
  if (hipGetDeviceCount(&devices) != hipSuccess || devices == 0) {
    std::cout << "flint dflash kernel tests skipped: no HIP device\n";
    return 0;
  }
  try {
    TestGroupedDynamicConv();
    TestSiluMul();
    TestBf16ToQ8_0();
    // heads 4 / kv 2 with a wide block: the dispatcher's two-heads-per-CTA
    // route. Sliding window 16 is shorter than the block, so the mask bites.
    TestAttention(
        {.current_pos = 40,
         .history_length = 48,
         .history_capacity = 64,
         .block_count = 8,
         .sliding_window = 16,
         .heads = 4,
         .kv_heads = 2,
         .head_dim = 8},
        "wide");
    // heads 3 / kv 1 and a narrow block: the single-head wave route.
    TestAttention(
        {.current_pos = 6,
         .history_length = 33,
         .history_capacity = 33,
         .block_count = 2,
         .sliding_window = 2048,
         .heads = 3,
         .kv_heads = 1,
         .head_dim = 12},
        "narrow");
    // A head dim that is not a multiple of four: the dispatcher must fall back
    // to the scalar route rather than read partial float4 rows.
    TestAttention(
        {.current_pos = 20,
         .history_length = 40,
         .history_capacity = 64,
         .block_count = 3,
         .sliding_window = 8,
         .heads = 2,
         .kv_heads = 1,
         .head_dim = 6},
        "scalar_only");
    // A short context that fits the window entirely, with a full cache wrap.
    TestAttention(
        {.current_pos = 3,
         .history_length = 9,
         .history_capacity = 9,
         .block_count = 8,
         .sliding_window = 2048,
         .heads = 2,
         .kv_heads = 2,
         .head_dim = 32},
        "unmasked");
    TestGatherRows();
    TestDraftTableLogits();
    TestSelectorGreedy();
  } catch (const std::exception& error) {
    std::cerr << "flint dflash kernel tests failed: " << error.what() << "\n";
    return 1;
  }
  std::cout << "flint dflash kernel tests passed\n";
  return 0;
}
