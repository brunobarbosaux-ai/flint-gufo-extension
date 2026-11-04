#ifndef GUFO_MODELS_FLINT_GPU_DEVICE_MODEL_HPP_
#define GUFO_MODELS_FLINT_GPU_DEVICE_MODEL_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "gufo/ext/gguf_reader.hpp"
#include "src/runtime/config.hpp"

namespace gufo::models::flint::gpu {

/// One weight resident in device memory, still in its GGUF encoding.
/// `rows` is ne[1] and `experts` ne[2], so M and the expert count stay apart.
struct DeviceTensor {
  void* data{nullptr};
  core::GgmlType type{core::GgmlType::kF32};
  std::uint32_t cols{0};
  std::uint32_t rows{1};
  std::uint32_t experts{1};
  /// F16 copy for wide batches (dense and shared projections only).
  void* half_data{nullptr};
};

struct DeviceLayer {
  bool is_moe{false};
  DeviceTensor attn_norm, attn_q, attn_k, attn_v, attn_output;
  /// Q, K and V rows stacked in one Q8_0 matrix; then attn_q/k/v are empty.
  DeviceTensor attn_qkv;
  DeviceTensor attn_q_norm, attn_k_norm, attn_sinks;
  DeviceTensor ffn_norm;
  DeviceTensor ffn_gate, ffn_up, ffn_down;
  DeviceTensor router;
  DeviceTensor gate_exps, up_exps, down_exps;
  DeviceTensor gate_shexp, up_shexp, down_shexp;
};

/// The whole model uploaded to the GPU. Routed down projections stored as
/// Q6_K are re-encoded as Q8_0, the nearest type the routed kernels read.
class DeviceModel {
public:
  ~DeviceModel();
  DeviceModel(const DeviceModel&) = delete;
  DeviceModel& operator=(const DeviceModel&) = delete;

  [[nodiscard]] static std::unique_ptr<DeviceModel> Upload(
      const core::GgufReader& reader, const Config& config,
      std::string* error_msg);

  const Config& config() const noexcept { return config_; }
  const DeviceTensor& token_embd() const noexcept { return token_embd_; }
  const DeviceTensor& output() const noexcept { return output_; }
  const DeviceTensor& output_norm() const noexcept { return output_norm_; }
  const std::vector<DeviceLayer>& layers() const noexcept { return layers_; }
  std::size_t resident_bytes() const noexcept { return bytes_; }

private:
  DeviceModel() = default;

  Config config_;
  DeviceTensor token_embd_, output_, output_norm_;
  std::vector<DeviceLayer> layers_;
  std::vector<void*> allocations_;
  std::size_t bytes_{0};
};

}  // namespace gufo::models::flint::gpu

#endif  // GUFO_MODELS_FLINT_GPU_DEVICE_MODEL_HPP_
