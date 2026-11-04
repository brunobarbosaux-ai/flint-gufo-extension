#include "src/gpu/device_model.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <unordered_map>

#include "gufo/ext/experimental/ggml_dequant.hpp"
#include "src/gpu/kernels/kernels.hpp"
#include "hip/vendor/rocm/kernels.hpp"

namespace gufo::models::flint::gpu {
namespace {

constexpr std::size_t kCopyChunk = std::size_t{256} << 20;
constexpr std::uint32_t kQ8Block = 32;
constexpr std::size_t kQ8Bytes = 34;
constexpr std::uint32_t kQ6KBlock = 256;
constexpr std::size_t kQ6KBytes = 210;

void SetError(std::string* error_msg, const std::string& message) {
  if (error_msg != nullptr)
    *error_msg = message;
}

std::string P(std::uint32_t layer, const char* name) {
  return "blk." + std::to_string(layer) + "." + name;
}

std::uint16_t FloatToHalf(float f) {
  const _Float16 h = static_cast<_Float16>(f);
  std::uint16_t bits = 0;
  std::memcpy(&bits, &h, sizeof(bits));
  return bits;
}

class Uploader {
public:
  Uploader(const core::GgufReader& reader, std::vector<void*>* allocations,
           std::size_t* bytes, std::string* error)
      : allocations_(allocations), bytes_(bytes), error_(error) {
    for (const auto& t : reader.GetTensors()) {
      tensors_.emplace(std::string(t.name), &t);
    }
  }

  /// Binds q, k and v as one stacked Q8_0 matrix when all three are Q8_0 and
  /// reports whether it did; otherwise nothing is uploaded.
  bool BindStacked(DeviceTensor* out, const std::string& q,
                   const std::string& k, const std::string& v, bool* stacked) {
    *stacked = false;
    const auto iq = tensors_.find(q);
    const auto ik = tensors_.find(k);
    const auto iv = tensors_.find(v);
    if (iq == tensors_.end() || ik == tensors_.end() || iv == tensors_.end()) {
      return true;
    }
    const core::GgufTensorInfo* parts[3] = {iq->second, ik->second, iv->second};
    std::size_t bytes = 0;
    for (const auto* t : parts) {
      if (t->type != core::GgmlType::kQ8_0 ||
          t->dimensions[0] != parts[0]->dimensions[0]) {
        return true;
      }
      bytes += t->size_bytes;
    }
    out->type = core::GgmlType::kQ8_0;
    out->cols = static_cast<std::uint32_t>(parts[0]->dimensions[0]);
    out->rows = 0;
    out->experts = 1;
    if (!Alloc(out, bytes))
      return false;
    std::size_t offset = 0;
    for (const auto* t : parts) {
      if (hipMemcpy(static_cast<char*>(out->data) + offset, t->data,
                    t->size_bytes, hipMemcpyHostToDevice) != hipSuccess) {
        SetError(error_, "device copy failed");
        return false;
      }
      offset += t->size_bytes;
      out->rows += static_cast<std::uint32_t>(t->dimensions[1]);
    }
    *stacked = true;
    return true;
  }

  /// F16 copy of a BF16 matrix; the router weights are well inside F16 range.
  bool BindHalfFromBf16(DeviceTensor* t, const std::string& name) {
    const auto it = tensors_.find(name);
    if (it == tensors_.end() || it->second->type != core::GgmlType::kBF16) {
      return true;
    }
    const std::size_t count = it->second->size_bytes / sizeof(std::uint16_t);
    const auto* src = static_cast<const std::uint16_t*>(it->second->data);
    std::vector<__half> host(count);
    for (std::size_t i = 0; i < count; ++i) {
      const std::uint32_t bits = static_cast<std::uint32_t>(src[i]) << 16;
      float f;
      std::memcpy(&f, &bits, sizeof(f));
      host[i] = __float2half(f);
    }
    DeviceTensor half;
    if (!Alloc(&half, count * sizeof(__half)))
      return false;
    if (hipMemcpy(half.data, host.data(), count * sizeof(__half),
                  hipMemcpyHostToDevice) != hipSuccess) {
      SetError(error_, "device copy failed");
      return false;
    }
    t->half_data = half.data;
    return true;
  }

  /// Adds an F16 copy of a dense matrix for wide-batch GEMMs.
  bool ExpandHalf(DeviceTensor* t) {
    if (t->data == nullptr || t->experts != 1 ||
        t->type == core::GgmlType::kQ8_0) {
      return true;
    }
    const std::size_t count = static_cast<std::size_t>(t->rows) * t->cols;
    DeviceTensor half;
    if (!Alloc(&half, count * sizeof(std::uint16_t)))
      return false;
    t->half_data = half.data;
    if (t->type == core::GgmlType::kQ8_0) {
      rocm::ExpandQ8ToHalf(t->data, static_cast<__half*>(t->half_data), count,
                           nullptr);
    } else if (!qwen38_flash_next::rocm::DequantizeHalf(
                   t->data, t->type, static_cast<__half*>(t->half_data), count,
                   nullptr)) {
      t->half_data = nullptr;
    }
    return hipDeviceSynchronize() == hipSuccess;
  }

  bool Bind(DeviceTensor* out, const std::string& name, bool required = true,
            bool q6k_to_q8 = false) {
    const auto it = tensors_.find(name);
    if (it == tensors_.end()) {
      if (required)
        SetError(error_, "missing tensor: " + name);
      return !required;
    }
    const auto& t = *it->second;
    out->cols = static_cast<std::uint32_t>(t.dimensions[0]);
    out->rows = t.dimensions.size() > 1
                    ? static_cast<std::uint32_t>(t.dimensions[1])
                    : 1;
    out->experts = t.dimensions.size() > 2
                       ? static_cast<std::uint32_t>(t.dimensions[2])
                       : 1;
    out->type = t.type;
    if (q6k_to_q8 && t.type == core::GgmlType::kQ6_K) {
      out->type = core::GgmlType::kQ8_0;
      return ReencodeQ6KToQ8(out, t);
    }
    return Copy(out, t.data, t.size_bytes);
  }

private:
  bool Alloc(DeviceTensor* out, std::size_t bytes) {
    if (hipMalloc(&out->data, bytes) != hipSuccess) {
      SetError(error_, "device allocation failed (" + std::to_string(bytes) +
                           " bytes)");
      return false;
    }
    allocations_->push_back(out->data);
    *bytes_ += bytes;
    return true;
  }

  bool Copy(DeviceTensor* out, const void* src, std::size_t bytes) {
    if (!Alloc(out, bytes))
      return false;
    for (std::size_t off = 0; off < bytes; off += kCopyChunk) {
      const std::size_t n = std::min(kCopyChunk, bytes - off);
      if (hipMemcpy(static_cast<char*>(out->data) + off,
                    static_cast<const char*>(src) + off, n,
                    hipMemcpyHostToDevice) != hipSuccess) {
        SetError(error_, "device copy failed");
        return false;
      }
    }
    return true;
  }

  bool ReencodeQ6KToQ8(DeviceTensor* out, const core::GgufTensorInfo& t) {
    const std::uint64_t n_rows =
        static_cast<std::uint64_t>(out->rows) * out->experts;
    const std::size_t src_row = out->cols / kQ6KBlock * kQ6KBytes;
    const std::size_t dst_row = out->cols / kQ8Block * kQ8Bytes;
    if (!Alloc(out, dst_row * n_rows))
      return false;
    const std::uint64_t rows_per_chunk =
        std::max<std::uint64_t>(1, kCopyChunk / dst_row);
    std::vector<char> host(rows_per_chunk * dst_row);
    std::vector<float> values(out->cols);
    for (std::uint64_t r0 = 0; r0 < n_rows; r0 += rows_per_chunk) {
      const std::uint64_t n = std::min(rows_per_chunk, n_rows - r0);
      for (std::uint64_t r = 0; r < n; ++r) {
        const char* src = static_cast<const char*>(t.data) + (r0 + r) * src_row;
        quant::DequantizeQ6_K(src, values.data(), out->cols);
        char* dst = host.data() + r * dst_row;
        for (std::uint32_t b = 0; b < out->cols / kQ8Block; ++b) {
          const float* v = values.data() + b * kQ8Block;
          float amax = 0.0F;
          for (std::uint32_t i = 0; i < kQ8Block; ++i) {
            amax = std::max(amax, std::fabs(v[i]));
          }
          const float scale = amax / 127.0F;
          const float inv = scale > 0.0F ? 1.0F / scale : 0.0F;
          const std::uint16_t d = FloatToHalf(scale);
          char* block = dst + b * kQ8Bytes;
          std::memcpy(block, &d, sizeof(d));
          for (std::uint32_t i = 0; i < kQ8Block; ++i) {
            block[2 + i] = static_cast<char>(
                static_cast<signed char>(std::lround(v[i] * inv)));
          }
        }
      }
      if (hipMemcpy(static_cast<char*>(out->data) + r0 * dst_row, host.data(),
                    n * dst_row, hipMemcpyHostToDevice) != hipSuccess) {
        SetError(error_, "device copy failed");
        return false;
      }
    }
    return true;
  }

  std::unordered_map<std::string, const core::GgufTensorInfo*> tensors_;
  std::vector<void*>* allocations_;
  std::size_t* bytes_;
  std::string* error_;
};

}  // namespace

DeviceModel::~DeviceModel() {
  for (void* p : allocations_) {
    (void)hipFree(p);
  }
}

std::unique_ptr<DeviceModel> DeviceModel::Upload(const core::GgufReader& reader,
                                                 const Config& config,
                                                 std::string* error_msg) {
  std::unique_ptr<DeviceModel> model(new DeviceModel());
  model->config_ = config;
  Uploader up(reader, &model->allocations_, &model->bytes_, error_msg);

  if (!up.Bind(&model->token_embd_, "token_embd.weight") ||
      !up.Bind(&model->output_norm_, "output_norm.weight") ||
      !up.Bind(&model->output_, "output.weight")) {
    return nullptr;
  }
  if (model->token_embd_.type != core::GgmlType::kQ8_0) {
    SetError(error_msg, "Flint: token_embd must be Q8_0");
    return nullptr;
  }

  model->layers_.resize(config.num_layers);
  for (std::uint32_t il = 0; il < config.num_layers; ++il) {
    auto& L = model->layers_[il];
    L.is_moe = il >= config.leading_dense;
    bool stacked = false;
    bool ok = up.Bind(&L.attn_norm, P(il, "attn_norm.weight")) &&
              up.BindStacked(&L.attn_qkv, P(il, "attn_q.weight"),
                             P(il, "attn_k.weight"), P(il, "attn_v.weight"),
                             &stacked) &&
              (stacked || (up.Bind(&L.attn_q, P(il, "attn_q.weight")) &&
                           up.Bind(&L.attn_k, P(il, "attn_k.weight")) &&
                           up.Bind(&L.attn_v, P(il, "attn_v.weight")))) &&
              up.Bind(&L.attn_output, P(il, "attn_output.weight")) &&
              up.Bind(&L.attn_q_norm, P(il, "attn_q_norm.weight")) &&
              up.Bind(&L.attn_k_norm, P(il, "attn_k_norm.weight")) &&
              up.Bind(&L.attn_sinks, P(il, "attn_sinks.weight")) &&
              up.Bind(&L.ffn_norm, P(il, "ffn_norm.weight"));
    if (ok && !L.is_moe) {
      ok = up.Bind(&L.ffn_gate, P(il, "ffn_gate.weight")) &&
           up.Bind(&L.ffn_up, P(il, "ffn_up.weight")) &&
           up.Bind(&L.ffn_down, P(il, "ffn_down.weight"));
    }
    if (ok && L.is_moe) {
      ok = up.Bind(&L.router, P(il, "ffn_gate_inp.weight")) &&
           up.BindHalfFromBf16(&L.router, P(il, "ffn_gate_inp.weight")) &&
           up.Bind(&L.gate_exps, P(il, "ffn_gate_exps.weight")) &&
           up.Bind(&L.up_exps, P(il, "ffn_up_exps.weight")) &&
           up.Bind(&L.down_exps, P(il, "ffn_down_exps.weight"), true, true) &&
           up.Bind(&L.gate_shexp, P(il, "ffn_gate_shexp.weight")) &&
           up.Bind(&L.up_shexp, P(il, "ffn_up_shexp.weight")) &&
           up.Bind(&L.down_shexp, P(il, "ffn_down_shexp.weight"));
    }
    if (!ok)
      return nullptr;
    {
      // F16 copies where the matrix-core GEMM beats the quantized wide path.
      for (DeviceTensor* t : {&L.attn_qkv, &L.attn_q, &L.attn_k, &L.attn_v,
                              &L.gate_shexp, &L.up_shexp, &L.down_shexp}) {
        if (!up.ExpandHalf(t)) {
          SetError(error_msg, "F16 weight expansion failed");
          return nullptr;
        }
      }
      if (L.attn_output.type != core::GgmlType::kQ8_0 &&
          !up.ExpandHalf(&L.attn_output)) {
        SetError(error_msg, "F16 weight expansion failed");
        return nullptr;
      }
    }
  }
  return model;
}

}  // namespace gufo::models::flint::gpu
