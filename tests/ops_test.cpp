// Flint kernel ops test: exercises the vendored ROCm kernels on the device and
// checks them against a host reference. Model-independent, so it needs no
// weights artifact.
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "hip/vendor/rocm/kernels.hpp"
#include "src/gpu/kernels/kernels.hpp"

namespace rocm = gufo::vendor::qwen38_flash_next::rocm;
namespace flint_rocm = gufo::models::flint::gpu::rocm;
namespace {

void Check(hipError_t error, const char* operation) {
  if (error != hipSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(error));
  }
}

void Near(const std::vector<float>& got, const std::vector<float>& want,
          float limit, const char* label) {
  if (got.size() != want.size())
    throw std::runtime_error(std::string(label) + ": size mismatch");
  for (std::size_t i = 0; i < got.size(); ++i) {
    if (!std::isfinite(got[i]) || std::abs(got[i] - want[i]) > limit)
      throw std::runtime_error(std::string(label) + ": mismatch at " +
                               std::to_string(i) + ": " +
                               std::to_string(got[i]) + " vs " +
                               std::to_string(want[i]));
  }
}

std::vector<float> Data(std::size_t n, float scale = 0.1F) {
  std::vector<float> x(n);
  for (std::size_t i = 0; i < n; ++i)
    x[i] = scale * std::sin(static_cast<float>(i) * 0.73F);
  return x;
}

/// out[t][d] = x[t][d] / sqrt(mean(x[t]^2) + eps) * gamma[d]
void RmsNormReference(const std::vector<float>& x,
                      const std::vector<float>& gamma, std::uint32_t rows,
                      std::uint32_t dim, std::uint32_t groups, float eps,
                      std::vector<float>* out) {
  out->assign(x.size(), 0.0F);
  const std::uint32_t span = dim / groups;
  for (std::uint32_t r = 0; r < rows; ++r) {
    for (std::uint32_t g = 0; g < groups; ++g) {
      double sum = 0.0;
      const std::size_t base = static_cast<std::size_t>(r) * dim +
                               static_cast<std::size_t>(g) * span;
      for (std::uint32_t i = 0; i < span; ++i) {
        const double v = x[base + i];
        sum += v * v;
      }
      const double inv = 1.0 / std::sqrt(sum / span + eps);
      for (std::uint32_t i = 0; i < span; ++i)
        (*out)[base + i] =
            static_cast<float>(x[base + i] * inv) * gamma[g * span + i];
    }
  }
}

void TestRmsNorm(float* x_dev, float* gamma_dev, float* out_dev,
                 hipStream_t stream) {
  constexpr std::uint32_t kRows = 7, kDim = 256, kGroups = 2;
  constexpr float kEps = 1e-5F;
  auto x = Data(kRows * kDim, 0.3F);
  auto gamma = Data(kDim, 0.5F);
  for (auto& v : gamma)
    v += 1.0F;
  Check(hipMemcpy(x_dev, x.data(), x.size() * 4, hipMemcpyHostToDevice),
        "copy x");
  Check(hipMemcpy(gamma_dev, gamma.data(), gamma.size() * 4,
                  hipMemcpyHostToDevice),
        "copy gamma");
  rocm::RmsNormRows(x_dev, gamma_dev, out_dev, kRows, kDim, kGroups, kEps,
                    stream);
  Check(hipStreamSynchronize(stream), "sync rmsnorm");
  std::vector<float> got(kRows * kDim);
  Check(hipMemcpy(got.data(), out_dev, got.size() * 4,
                  hipMemcpyDeviceToHost),
        "copy out");
  std::vector<float> want;
  RmsNormReference(x, gamma, kRows, kDim, kGroups, kEps, &want);
  Near(got, want, 1e-3F, "RmsNormRows");
}

void TestSiluScale(float* x_dev, hipStream_t stream) {
  constexpr std::size_t kCount = 512;
  constexpr float kScale = 0.5F;
  auto x = Data(kCount, 3.0F);
  Check(hipMemcpy(x_dev, x.data(), x.size() * 4, hipMemcpyHostToDevice),
        "copy silu x");
  rocm::SiluScale(x_dev, kScale, kCount, stream);
  Check(hipStreamSynchronize(stream), "sync silu");
  std::vector<float> got(kCount);
  Check(hipMemcpy(got.data(), x_dev, got.size() * 4, hipMemcpyDeviceToHost),
        "copy silu out");
  std::vector<float> want(kCount);
  for (std::size_t i = 0; i < kCount; ++i) {
    const double v = x[i] * kScale;
    want[i] = static_cast<float>(v / (1.0 + std::exp(-v)));
  }
  Near(got, want, 1e-4F, "SiluScale");
}

/// Decode attention of a global layer: each row of a 7-row pass must equal,
/// bit for bit, the same row computed alone. Every K/V row of the buffer holds
/// random data, so the keys past a row's position are the stale cache a real
/// session has there: a kernel that lets a masked key reach a row fails here.
/// The 32 start offsets move the rows across every position in a key tile.
void TestAttentionRowInvariance(std::uint32_t depth, std::uint32_t rows,
                                bool matrix) {
  constexpr std::uint32_t kHeads = 32, kKvHeads = 4, kDim = 128;
  constexpr std::uint32_t kOffsets = 32;
  const std::uint32_t splits = matrix ? 64 : 32;
  const std::size_t kv_elems =
      std::size_t{depth + kOffsets + rows + 64} * kKvHeads * kDim;
  const std::size_t row_floats = std::size_t{kHeads} * kDim;

  const auto random_half = [](std::size_t n, std::uint32_t x) {
    std::vector<__half> v(n);
    for (auto& h : v) {
      x = x * 1664525U + 1013904223U;
      h = __float2half((static_cast<float>(x >> 9) / 4194304.0F - 1.0F) * 0.5F);
    }
    return v;
  };
  const auto k = random_half(kv_elems, 1);
  const auto v = random_half(kv_elems, 2);
  const auto q = Data(rows * row_floats, 1.0F);
  const std::vector<float> sinks(kHeads, 0.0F);

  __half *k_dev = nullptr, *v_dev = nullptr;
  float *q_dev = nullptr, *sink_dev = nullptr, *out_dev = nullptr,
        *part_dev = nullptr;
  std::uint32_t* pos_dev = nullptr;
  Check(hipMalloc(&k_dev, kv_elems * sizeof(__half)), "malloc k");
  Check(hipMalloc(&v_dev, kv_elems * sizeof(__half)), "malloc v");
  Check(hipMalloc(&q_dev, q.size() * 4), "malloc q");
  Check(hipMalloc(&sink_dev, sinks.size() * 4), "malloc sinks");
  Check(hipMalloc(&out_dev, q.size() * 4), "malloc attention out");
  Check(hipMalloc(&part_dev,
                  std::size_t{rows} * kHeads * splits * (kDim + 2) * 4),
        "malloc partial");
  Check(hipMalloc(&pos_dev, 4), "malloc position");
  Check(hipMemcpy(k_dev, k.data(), kv_elems * sizeof(__half),
                  hipMemcpyHostToDevice),
        "copy k");
  Check(hipMemcpy(v_dev, v.data(), kv_elems * sizeof(__half),
                  hipMemcpyHostToDevice),
        "copy v");
  Check(hipMemcpy(q_dev, q.data(), q.size() * 4, hipMemcpyHostToDevice),
        "copy q");
  Check(hipMemcpy(sink_dev, sinks.data(), sinks.size() * 4,
                  hipMemcpyHostToDevice),
        "copy sinks");

  flint_rocm::SetMatrixAttention(matrix);
  const auto run = [&](std::uint32_t start, std::uint32_t first,
                       std::uint32_t count) {
    const std::uint32_t position = start + first;
    Check(hipMemcpy(pos_dev, &position, 4, hipMemcpyHostToDevice),
          "copy position");
    flint_rocm::AttentionSplit(
        q_dev + first * row_floats, k_dev, v_dev, sink_dev, out_dev, nullptr,
        part_dev, count, pos_dev, kHeads, kKvHeads, kDim, 0, splits, nullptr);
    Check(hipDeviceSynchronize(), "sync attention");
    std::vector<float> got(count * row_floats);
    Check(hipMemcpy(got.data(), out_dev, got.size() * 4,
                    hipMemcpyDeviceToHost),
          "copy attention out");
    return got;
  };
  for (std::uint32_t offset = 0; offset < kOffsets; ++offset) {
    const std::uint32_t start = depth + offset;
    const auto block = run(start, 0, rows);
    for (std::uint32_t row = 0; row < rows; ++row) {
      const auto alone = run(start, row, 1);
      if (std::memcmp(block.data() + row * row_floats, alone.data(),
                      row_floats * 4) != 0) {
        throw std::runtime_error(
            std::string("attention row invariance (") +
            (matrix ? "matrix" : "scalar") + "): depth " +
            std::to_string(depth) + " offset " + std::to_string(offset) +
            " row " + std::to_string(row) + " differs from the row alone");
      }
    }
  }
  flint_rocm::SetMatrixAttention(flint_rocm::MatrixAttentionBuild());

  for (void* p : {static_cast<void*>(k_dev), static_cast<void*>(v_dev),
                  static_cast<void*>(q_dev), static_cast<void*>(sink_dev),
                  static_cast<void*>(out_dev), static_cast<void*>(part_dev),
                  static_cast<void*>(pos_dev)})
    (void)hipFree(p);
  std::cout << "  attention rows invariant, rows " << rows << ", depth " << depth
            << ", " << (matrix ? "matrix" : "scalar") << " kernel\n";
}

}  // namespace

int main() {
  int devices = 0;
  if (hipGetDeviceCount(&devices) != hipSuccess || devices == 0) {
    std::cout << "flint ops tests skipped: no HIP device\n";
    return 0;
  }
  try {
    float *x_dev = nullptr, *gamma_dev = nullptr, *out_dev = nullptr;
    Check(hipMalloc(&x_dev, 4096 * 4), "malloc x");
    Check(hipMalloc(&gamma_dev, 1024 * 4), "malloc gamma");
    Check(hipMalloc(&out_dev, 4096 * 4), "malloc out");
    TestRmsNorm(x_dev, gamma_dev, out_dev, nullptr);
    TestSiluScale(x_dev, nullptr);
    (void)hipFree(x_dev);
    (void)hipFree(gamma_dev);
    (void)hipFree(out_dev);
    for (const std::uint32_t depth : {8192U, 32768U})
      for (const std::uint32_t rows : {7U, 8U})
        for (const bool matrix : {false, true})
          TestAttentionRowInvariance(depth, rows, matrix);
  } catch (const std::exception& error) {
    std::cerr << "flint ops tests failed: " << error.what() << "\n";
    return 1;
  }
  std::cout << "flint ops tests passed\n";
  return 0;
}
