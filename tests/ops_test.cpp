// Flint kernel ops test: exercises the vendored ROCm kernels on the device and
// checks them against a host reference. Model-independent, so it needs no
// weights artifact.
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "hip/vendor/rocm/kernels.hpp"

namespace rocm = gufo::vendor::qwen38_flash_next::rocm;
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
  } catch (const std::exception& error) {
    std::cerr << "flint ops tests failed: " << error.what() << "\n";
    return 1;
  }
  std::cout << "flint ops tests passed\n";
  return 0;
}
