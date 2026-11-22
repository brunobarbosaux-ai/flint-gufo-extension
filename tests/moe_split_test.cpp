// Row independence of the routed MMQ expert projections against random
// weights (iq4_xs, iq3_s, q5_K), in the gate/up call shape (6 experts per
// token) and the down shape (one expert per slot). A projection of tokens
// [s, n) or [0, s) must give every row the bits of the projection of [0, n),
// under the default tile width and the engine's per-call knobs. Prefill split
// invariance rests on it. Exit 1 on any difference.

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "hip/vendor/rocm/mmq/flint_mmq.h"

namespace {

constexpr int kN = 600;     // tokens
constexpr int kE = 256;     // experts, as Flint
constexpr int kU = 6;       // experts per token

struct Type {
  const char* name;
  int block_bytes;
  int (*fn)(const void*, const float*, const std::int32_t*, float*, int, int,
            int, int, int, hipStream_t);
};

void Check(hipError_t e, const char* what) {
  if (e != hipSuccess) {
    std::cerr << what << ": " << hipGetErrorString(e) << "\n";
    std::exit(1);
  }
}

// Mirrors engine.cpp's RoutedTileCols.
int TileCols(int n_tokens, int used, int experts) {
  const int mean = std::max(1, n_tokens * used / experts);
  for (int cols = 16; cols < 80; cols += 16)
    if (cols >= 2 * mean)
      return cols;
  return 80;
}

struct Case {
  const Type* type;
  int m, k, used;     // used = 1: the down projection's call shape
  const void* w;
  const float* x;     // [rows][k]
  const std::int32_t* ids;
  int engine_tokens;  // tokens of the whole batch, for the engine knobs
};

std::vector<float> Run(const Case& c, int first, int count, bool engine_knobs) {
  // count rows of the call; for used = 1 a "row" is one (token, expert) slot.
  const int tokens = c.used == 1 ? count / kU : count;
  if (engine_knobs) {
    flint_mmq_set_routed_max_expert_rows(tokens);
    flint_mmq_set_routed_tile_cols(TileCols(tokens, kU, kE));
  } else {
    flint_mmq_set_routed_max_expert_rows(0);
    flint_mmq_set_routed_tile_cols(0);
  }
  float* out = nullptr;
  const std::size_t out_elems = std::size_t(count) * c.used * c.m;
  Check(hipMalloc(&out, out_elems * sizeof(float)), "malloc out");
  Check(hipMemset(out, 0, out_elems * sizeof(float)), "memset out");
  if (c.type->fn(c.w, c.x + std::size_t(first) * c.k, c.ids + first * c.used,
                 out, c.m, c.k, count, kE, c.used, nullptr) != 0) {
    std::cerr << c.type->name << " moe_raw failed\n";
    std::exit(1);
  }
  Check(hipDeviceSynchronize(), "sync");
  std::vector<float> host(out_elems);
  Check(hipMemcpy(host.data(), out, out_elems * sizeof(float),
                  hipMemcpyDeviceToHost), "copy out");
  Check(hipFree(out), "free out");
  return host;
}

int DiffRows(const Case& c, const std::vector<float>& full,
             const std::vector<float>& part, int first, int count) {
  int rows = 0;
  const std::size_t row = std::size_t(c.used) * c.m;
  for (int t = 0; t < count; ++t)
    if (std::memcmp(full.data() + (first + t) * row, part.data() + t * row,
                    row * sizeof(float)) != 0)
      ++rows;
  return rows;
}

void* RandomWeights(std::mt19937& rng, const Type& t, int m, int k) {
  const std::size_t blocks = std::size_t(kE) * m * (k / 256);
  std::vector<std::uint8_t> host(blocks * t.block_bytes);
  for (auto& b : host)
    b = static_cast<std::uint8_t>(rng());
  for (std::size_t i = 0; i < blocks; ++i) {
    // Every type here starts with a half scale; Q5_K has a half dmin next.
    const std::uint16_t d = 0x2000 + (rng() & 0x3ff);
    std::memcpy(host.data() + i * t.block_bytes, &d, 2);
    if (std::string(t.name) == "q5_K")
      std::memcpy(host.data() + i * t.block_bytes + 2, &d, 2);
  }
  void* w = nullptr;
  Check(hipMalloc(&w, host.size()), "malloc w");
  Check(hipMemcpy(w, host.data(), host.size(), hipMemcpyHostToDevice), "w");
  return w;
}

}  // namespace

int main() {
  int devices = 0;
  if (hipGetDeviceCount(&devices) != hipSuccess || devices == 0) {
    std::cout << "moe split test skipped: no HIP device\n";
    return 0;
  }
  const Type types[] = {{"iq4_xs", 136, flint_mmq_iq4_xs_moe_raw},
                        {"iq3_s", 110, flint_mmq_iq3_s_moe_raw},
                        {"q5_K", 176, flint_mmq_q5_K_moe_raw}};
  std::mt19937 rng(1234);
  std::normal_distribution<float> normal(0.0F, 1.0F);
  // Routing: kU distinct experts per token, skewed so some groups are large.
  std::vector<std::int32_t> ids_host(std::size_t(kN) * kU);
  for (int t = 0; t < kN; ++t) {
    std::vector<int> perm(kE);
    for (int e = 0; e < kE; ++e)
      perm[e] = e;
    std::shuffle(perm.begin(), perm.begin() + 24, rng);
    std::shuffle(perm.begin() + 24, perm.end(), rng);
    for (int u = 0; u < kU; ++u)
      perm[u] = u < 3 ? perm[u] : perm[24 + u];
    for (int u = 0; u < kU; ++u)
      ids_host[t * kU + u] = perm[u];
  }
  int differing = 0;
  std::int32_t* ids = nullptr;
  Check(hipMalloc(&ids, ids_host.size() * sizeof(std::int32_t)), "malloc ids");
  Check(hipMemcpy(ids, ids_host.data(), ids_host.size() * sizeof(std::int32_t),
                  hipMemcpyHostToDevice), "ids");
  for (const Type& type : types) {
    for (const int used : {kU, 1}) {
      const int m = used == kU ? 512 : 2048;   // gate/up: ff x hidden; down
      const int k = used == kU ? 2048 : 512;
      const int rows = used == kU ? kN : kN * kU;
      std::vector<float> x_host(std::size_t(rows) * k);
      for (auto& v : x_host)
        v = normal(rng);
      float* x = nullptr;
      Check(hipMalloc(&x, x_host.size() * sizeof(float)), "malloc x");
      Check(hipMemcpy(x, x_host.data(), x_host.size() * sizeof(float),
                      hipMemcpyHostToDevice), "x");
      void* w = RandomWeights(rng, type, m, k);
      const Case c{&type, m, k, used, w, x, ids, kN};
      for (const bool knobs : {false, true}) {
        const auto full = Run(c, 0, rows, knobs);
        std::cout << type.name << (used == 1 ? " down-shape" : " gate-shape")
                  << (knobs ? " engine-knobs" : " defaults") << ":";
        for (const int s_tok : {1, 9, 64, 100, 300}) {
          const int s = used == 1 ? s_tok * kU : s_tok;
          const auto tail = Run(c, s, rows - s, knobs);
          const auto head = Run(c, 0, s, knobs);
          const int dt = DiffRows(c, full, tail, s, rows - s);
          const int dh = DiffRows(c, full, head, 0, s);
          differing += dt + dh;
          std::cout << " s=" << s_tok << " tail " << dt << "/" << rows - s
                    << " head " << dh << "/" << s << ";";
        }
        std::cout << "\n";
      }
      Check(hipFree(w), "free w");
      Check(hipFree(x), "free x");
    }
  }
  std::cout << (differing == 0 ? "flint moe split test passed\n"
                               : "flint moe split test failed\n");
  return differing == 0 ? 0 : 1;
}
