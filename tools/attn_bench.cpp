// Decode attention of a global layer on synthetic KV at 8k / 32k / 131k keys,
// scalar kernels against the matrix-core slice kernel: timings for 1, 2, 4 and
// 8 rows, the largest difference to the scalar result relative to the largest
// output value, row invariance (block against pair and lone row) and a probe of
// whether a masked key reaches a row. One `event=attn_bench` line per result.
// Needs a HIP device, no model.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "src/gpu/kernels/kernels.hpp"

namespace {

void Log(const std::string& line) {
  std::printf("%s\n", line.c_str());
  std::fflush(stdout);
}

}  // namespace

int main() {
  int devices = 0;
  if (hipGetDeviceCount(&devices) != hipSuccess || devices == 0) {
    std::fprintf(stderr, "flint attention bench: no HIP device\n");
    return 1;
  }
  namespace rk = gufo::models::flint::gpu::rocm;
  constexpr std::uint32_t heads = 32, kv_heads = 4, d = 128;
  const auto fill = [](std::vector<__half>& v, std::uint32_t seed) {
    std::uint32_t x = seed;
    for (auto& h : v) {
      x = x * 1664525U + 1013904223U;
      h = __float2half((static_cast<float>(x >> 9) / 4194304.0F - 1.0F) *
                       0.5F);
    }
  };
  for (const std::uint32_t depth : {8192U, 32768U, 131072U}) {
    const std::size_t rows = depth + 16;
    const std::size_t elems = rows * kv_heads * d;
    std::vector<__half> hk(elems), hv(elems);
    fill(hk, 1);
    fill(hv, 2);
    std::vector<float> hq(std::size_t{8} * heads * d);
    {
      std::uint32_t x = 7;
      for (auto& f : hq) {
        x = x * 1664525U + 1013904223U;
        f = static_cast<float>(x >> 9) / 4194304.0F - 1.0F;
      }
    }
    __half *dk = nullptr, *dv = nullptr;
    float *dq = nullptr, *dsink = nullptr, *dout = nullptr, *dpart = nullptr;
    std::uint32_t* dpos = nullptr;
    constexpr std::uint32_t kMaxSplits = 256;
    const bool ok =
        hipMalloc(&dk, elems * 2) == hipSuccess &&
        hipMalloc(&dv, elems * 2) == hipSuccess &&
        hipMalloc(&dq, hq.size() * 4) == hipSuccess &&
        hipMalloc(&dsink, heads * 4) == hipSuccess &&
        hipMalloc(&dout, hq.size() * 4) == hipSuccess &&
        hipMalloc(&dpart, std::size_t{8} * heads * kMaxSplits * (d + 2) * 4) ==
            hipSuccess &&
        hipMalloc(&dpos, 4) == hipSuccess;
    if (!ok) {
      Log("event=attn_bench result=failed reason=allocation");
      return 1;
    }
    hipMemcpy(dk, hk.data(), elems * 2, hipMemcpyHostToDevice);
    hipMemcpy(dv, hv.data(), elems * 2, hipMemcpyHostToDevice);
    hipMemcpy(dq, hq.data(), hq.size() * 4, hipMemcpyHostToDevice);
    hipMemset(dsink, 0, heads * 4);
    hipEvent_t t0, t1;
    hipEventCreate(&t0);
    hipEventCreate(&t1);
    const double bytes = static_cast<double>(depth) * kv_heads * d * 2 * 2;
    for (const std::uint32_t n : {1U, 2U, 4U, 8U}) {
      const std::uint32_t start = depth - n;
      hipMemcpy(dpos, &start, 4, hipMemcpyHostToDevice);
      const auto run = [&](int mode, std::uint32_t splits) {
        rk::SetAttentionMode(mode);
        rk::AttentionSplit(dq, dk, dv, dsink, dout, nullptr, dpart, n, dpos,
                           heads, kv_heads, d, 0, splits, nullptr);
      };
      std::vector<float> ref(std::size_t{n} * heads * d), got(ref.size());
      run(1, 32);
      hipDeviceSynchronize();
      hipMemcpy(ref.data(), dout, ref.size() * 4, hipMemcpyDeviceToHost);
      float scale = 0.0F;
      for (const float f : ref)
        scale = std::max(scale, std::fabs(f));
      for (const int mode : {1, 0}) {
        for (const std::uint32_t splits : {32U, 64U}) {
          if (mode == 1 && splits != 32)
            continue;
          for (int i = 0; i < 3; ++i)
            run(mode, splits);
          hipEventRecord(t0, nullptr);
          constexpr int kIters = 20;
          for (int i = 0; i < kIters; ++i)
            run(mode, splits);
          hipEventRecord(t1, nullptr);
          hipEventSynchronize(t1);
          float ms = 0.0F;
          hipEventElapsedTime(&ms, t0, t1);
          ms /= kIters;
          hipMemcpy(got.data(), dout, got.size() * 4, hipMemcpyDeviceToHost);
          float diff = 0.0F;
          for (std::size_t i = 0; i < got.size(); ++i)
            diff = std::max(diff, std::fabs(got[i] - ref[i]));
          char line[256];
          std::snprintf(line, sizeof(line),
                        "event=attn_bench depth=%u rows=%u mode=%s splits=%u "
                        "ms=%.3f gbps=%.0f max_rel_diff=%.2e",
                        depth, n, mode == 1 ? "scalar" : "wmma", splits, ms,
                        bytes / (ms * 1e6), scale > 0 ? diff / scale : 0.0F);
          Log(line);
        }
      }
    }
    // Row invariance over 8 start offsets: every row of an 8-row block, of a
    // pair and alone, bit for bit (64 rows per depth, and per pair size).
    for (const int mode : {1, 0}) {
      const std::uint32_t splits = mode == 1 ? 32 : 64;
      rk::SetAttentionMode(mode);
      std::uint32_t bad_pair = 0, bad_single = 0, total = 0;
      for (std::uint32_t offs = 0; offs < 8; ++offs) {
        const std::uint32_t start8 = depth - 8 + offs;
        const auto call = [&](std::uint32_t first_row, std::uint32_t count,
                              std::vector<float>* out) {
          const std::uint32_t start = start8 + first_row;
          hipMemcpy(dpos, &start, 4, hipMemcpyHostToDevice);
          rk::AttentionSplit(dq + std::size_t{first_row} * heads * d, dk, dv,
                             dsink, dout, nullptr, dpart, count, dpos, heads,
                             kv_heads, d, 0, splits, nullptr);
          hipDeviceSynchronize();
          out->resize(std::size_t{count} * heads * d);
          hipMemcpy(out->data(), dout, out->size() * 4,
                    hipMemcpyDeviceToHost);
        };
        std::vector<float> b8, p2, s1;
        call(0, 8, &b8);
        const std::size_t n = std::size_t{heads} * d * 4;
        for (std::uint32_t row = 0; row < 8; ++row) {
          call(row & ~1U, 2, &p2);
          call(row, 1, &s1);
          const float* in8 = b8.data() + std::size_t{row} * heads * d;
          ++total;
          if (std::memcmp(in8, p2.data() + std::size_t{row & 1U} * heads * d,
                          n) != 0)
            ++bad_pair;
          if (std::memcmp(in8, s1.data(), n) != 0)
            ++bad_single;
        }
      }
      Log(std::string("event=attn_bench depth=") + std::to_string(depth) +
          " row_invariance mode=" + (mode == 1 ? "scalar" : "wmma") +
          " rows=" + std::to_string(total) + " block_vs_pair_bad=" +
          std::to_string(bad_pair) + " block_vs_single_bad=" +
          std::to_string(bad_single));
    }
    // Where does a masked key leak into a row? Perturb the key just past row
    // 0 (it is masked for row 0 and live for row 1) and see what moves.
    {
      const std::uint32_t start8 = depth - 8;
      rk::SetAttentionMode(0);
      hipMemcpy(dpos, &start8, 4, hipMemcpyHostToDevice);
      const auto pair_run = [&](std::vector<float>* out) {
        rk::AttentionSplit(dq, dk, dv, dsink, dout, nullptr, dpart, 2, dpos,
                           heads, kv_heads, d, 0, 64, nullptr);
        hipDeviceSynchronize();
        out->resize(std::size_t{2} * heads * d);
        hipMemcpy(out->data(), dout, out->size() * 4, hipMemcpyDeviceToHost);
      };
      std::vector<float> base, moved;
      pair_run(&base);
      const std::size_t off = std::size_t{start8 + 1} * kv_heads * d;
      const std::size_t bytes = std::size_t{kv_heads} * d * 2;
      std::vector<__half> keep_k(bytes / 2), keep_v(bytes / 2),
          zero(bytes / 2, __float2half(0.0F));
      hipMemcpy(keep_k.data(), dk + off, bytes, hipMemcpyDeviceToHost);
      hipMemcpy(keep_v.data(), dv + off, bytes, hipMemcpyDeviceToHost);
      const std::size_t half_row = std::size_t{heads} * d * 4;
      std::string out_line;
      hipMemcpy(dv + off, zero.data(), bytes, hipMemcpyHostToDevice);
      pair_run(&moved);
      out_line += std::memcmp(base.data(), moved.data(), half_row) == 0
                      ? "V0 row0 same; "
                      : "V0 row0 MOVES; ";
      out_line += std::memcmp(base.data() + heads * d, moved.data() + heads * d,
                              half_row) == 0
                      ? "row1 same; "
                      : "row1 moves; ";
      hipMemcpy(dv + off, keep_v.data(), bytes, hipMemcpyHostToDevice);
      hipMemcpy(dk + off, zero.data(), bytes, hipMemcpyHostToDevice);
      pair_run(&moved);
      out_line += std::memcmp(base.data(), moved.data(), half_row) == 0
                      ? "K0 row0 same; "
                      : "K0 row0 MOVES; ";
      hipMemcpy(dk + off, keep_k.data(), bytes, hipMemcpyHostToDevice);
      Log(std::string("event=attn_bench depth=") + std::to_string(depth) +
          " leak_probe " + out_line);
    }
    rk::SetAttentionMode(0);
    hipEventDestroy(t0);
    hipEventDestroy(t1);
    hipFree(dk);
    hipFree(dv);
    hipFree(dq);
    hipFree(dsink);
    hipFree(dout);
    hipFree(dpart);
    hipFree(dpos);
  }
  return 0;
}
