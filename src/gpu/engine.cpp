#include "src/gpu/engine.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#include "gufo/ext/gguf_reader.hpp"
#include "src/gpu/device_model.hpp"
#include "src/gpu/kernels/kernels.hpp"
#include "hip/vendor/rocm/blaslt.hpp"
#include "hip/vendor/rocm/kernels.hpp"
#include "hip/vendor/rocm/mmq/flint_mmq.h"

namespace gufo::models::flint::gpu {
namespace {

constexpr std::uint32_t kVecBatch = 8;
// Routed-expert activations are stored divided by this (F16 range); the
// combine multiplies it back.
constexpr float kRoutedActScale = 64.0F;
constexpr std::uint32_t kSparseSinkBlocks = 2;
constexpr std::uint32_t kSparseLocalBlocks = 32;
// Rows the gather-style MoE path accepts: kVecBatch (8) tokens times the
// model's 6 routed experts. Widths 6..8 must stay on this path — it is the one
// whose logits are bit-identical to single-token decode, and it avoids the
// per-layer host round-trip of the routed path. At 32 the verify widths DFlash2
// needs silently took the routed path, which broke both properties.
constexpr std::uint32_t kVecExpertRows = 48;
constexpr std::uint32_t kMaxTopK = 16;
constexpr std::uint32_t kMaxExperts = 256;
constexpr std::uint32_t kSwaSplits = 8;
constexpr std::uint32_t kGlobalSplits = 32;
constexpr std::uint32_t kQ8MaxK = 6144;
// The W8A8 kernel mis-computes the stacked Q/K/V matrix; it takes mmq.
constexpr std::uint32_t kStackedQkvRows = 5120;
constexpr std::uint32_t kRoutedTileMin = 16;
constexpr std::uint32_t kRoutedTileMax = 80;
constexpr std::uint32_t kRoutedTileStep = 16;

using core::GgmlType;

struct HipError {
  std::string message;
};

#define FLINT_HIP(call)                                                    \
  do {                                                                     \
    const hipError_t status_ = (call);                                     \
    if (status_ != hipSuccess) {                                           \
      return Fail(error_msg,                                               \
                  std::string(#call) + ": " + hipGetErrorString(status_)); \
    }                                                                      \
  } while (false)

bool Fail(std::string* error_msg, const std::string& message) {
  if (error_msg != nullptr)
    *error_msg = message;
  return false;
}

template<typename T>
class DeviceBuffer {
public:
  DeviceBuffer() = default;
  ~DeviceBuffer() { Reset(); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& other) noexcept
      : ptr_(other.ptr_), count_(other.count_) {
    other.ptr_ = nullptr;
    other.count_ = 0;
  }

  bool Allocate(std::size_t count) {
    Reset();
    if (hipMalloc(reinterpret_cast<void**>(&ptr_), count * sizeof(T)) !=
        hipSuccess) {
      ptr_ = nullptr;
      return false;
    }
    count_ = count;
    return true;
  }
  void Reset() {
    if (ptr_ != nullptr)
      (void)hipFree(ptr_);
    ptr_ = nullptr;
    count_ = 0;
  }
  T* get() const noexcept { return ptr_; }
  std::size_t bytes() const noexcept { return count_ * sizeof(T); }

private:
  T* ptr_{nullptr};
  std::size_t count_{0};
};

int RoutedTileCols(std::uint32_t n_tokens, std::uint32_t n_used,
                   std::uint32_t n_experts) {
  const std::uint32_t mean =
      std::max<std::uint32_t>(1, n_tokens * n_used / std::max(n_experts, 1U));
  for (std::uint32_t cols = kRoutedTileMin; cols < kRoutedTileMax;
       cols += kRoutedTileStep) {
    if (cols >= 2 * mean)
      return static_cast<int>(cols);
  }
  return static_cast<int>(kRoutedTileMax);
}

enum Stage : int {
  kStageEmbed,
  kStageNorm,
  kStageQkv,
  kStageRopeKv,
  kStageAttention,
  kStageOutProj,
  kStageDenseFfn,
  kStageRouter,
  kStageExpertsGateUp,
  kStageExpertsDown,
  kStageShared,
  kStageCombine,
  kStageHead,
  kStageSwiglu,
  kStageCount,
};

constexpr const char* kStageNames[kStageCount] = {
    "embed",    "norm",        "qkv",     "rope+kv",     "attention",
    "out_proj", "dense_ffn",   "router",  "exp_gate_up", "exp_down",
    "shared",   "moe_combine", "lm_head", "swiglu"};

/// Attributes the time between consecutive marks to a stage. Off unless
/// FLINT_PROFILE is set; syncs once per forward pass.
class StageTimer {
public:
  StageTimer() : on_(std::getenv("FLINT_PROFILE") != nullptr) {}
  ~StageTimer() {
    for (hipEvent_t e : events_)
      (void)hipEventDestroy(e);
    if (on_ && passes_ > 0)
      Report();
  }
  bool on() const noexcept { return on_; }

  void Mark(int stage, hipStream_t stream) {
    if (!on_)
      return;
    if (used_ == events_.size()) {
      hipEvent_t e = nullptr;
      (void)hipEventCreate(&e);
      events_.push_back(e);
    }
    (void)hipEventRecord(events_[used_], stream);
    stages_.push_back(stage);
    ++used_;
  }

  void Finish(std::uint32_t tokens) {
    if (!on_ || used_ < 2) {
      used_ = 0;
      stages_.clear();
      return;
    }
    (void)hipEventSynchronize(events_[used_ - 1]);
    for (std::size_t i = 1; i < used_; ++i) {
      float ms = 0.0F;
      (void)hipEventElapsedTime(&ms, events_[i - 1], events_[i]);
      auto& bucket = tokens > 1 ? prefill_ : decode_;
      bucket[stages_[i]] += ms;
    }
    (tokens > 1 ? prefill_tokens_ : decode_tokens_) += tokens;
    ++passes_;
    used_ = 0;
    stages_.clear();
  }

private:
  void Report() const {
    const auto print = [](const char* title, const double* ms,
                          std::uint64_t tokens) {
      double total = 0.0;
      for (int i = 0; i < kStageCount; ++i)
        total += ms[i];
      if (tokens == 0 || total <= 0.0)
        return;
      std::fprintf(stderr,
                   "[flint profile] %s: %llu tokens, %.1f ms (%.1f tok/s)\n",
                   title, static_cast<unsigned long long>(tokens), total,
                   tokens * 1000.0 / total);
      for (int i = 0; i < kStageCount; ++i) {
        std::fprintf(stderr, "  %-12s %9.2f ms %5.1f%%  %8.3f ms/tok\n",
                     kStageNames[i], ms[i], 100.0 * ms[i] / total,
                     ms[i] / static_cast<double>(tokens));
      }
    };
    print("prefill", prefill_, prefill_tokens_);
    print("decode", decode_, decode_tokens_);
  }

  bool on_;
  std::vector<hipEvent_t> events_;
  std::vector<int> stages_;
  std::size_t used_{0};
  std::uint32_t passes_{0};
  std::uint64_t prefill_tokens_{0};
  std::uint64_t decode_tokens_{0};
  double prefill_[kStageCount]{};
  double decode_[kStageCount]{};
};

}  // namespace

struct ModelState {
  EngineOptions options;
  std::shared_ptr<core::GgufReader> reader;
  Config config;
  std::unique_ptr<DeviceModel> device;
};

struct Session::Impl {
  std::shared_ptr<ModelState> model;
  hipStream_t stream{nullptr};
  std::uint32_t capacity{0};
  std::uint32_t tile{0};
  /// Set while Session::Prefill runs: every pass takes the wide (prefill)
  /// kernels, also passes of up to kVecBatch rows, so a prompt row's numerics
  /// do not depend on how many rows its pass has. Decode and verification
  /// keep the vector path.
  bool wide_prefill{false};
  /// Rows that take the vector path: decode/verify widths outside a prefill.
  [[nodiscard]] bool Narrow(std::uint32_t n) const noexcept {
    return n <= kVecBatch && !wide_prefill;
  }
  /// Routed-expert slots that take the gather-style vector kernels.
  [[nodiscard]] bool VecExperts(std::uint32_t slots) const noexcept {
    return slots <= kVecExpertRows && !wide_prefill;
  }
  std::uint32_t position{0};
  float* h_logits{nullptr};
  std::int32_t* h_tokens{nullptr};
  std::uint32_t* h_pos{nullptr};
  void* q8_a{nullptr};
  void* q8_b{nullptr};
  void* q8_tiled{nullptr};
  __half* x_half{nullptr};
  /// Half-precision copy of the normed FFN input, valid until xn is rewritten.
  __half* xh{nullptr};
  bool xh_valid{false};

  std::unique_ptr<gufo::vendor::qwen38_flash_next::rocm::BlasLt> blaslt;
  struct StepGraph {
    hipGraphExec_t exec{nullptr};
    bool failed{false};
    std::uint32_t steps{0};
  };
  StepGraph graphs[Session::kMaxVerifyRows + 1];
  std::uint32_t logit_rows{1};
  StageTimer timer;
  /// Notified with the rows an Append/AppendAll is about to write.
  std::function<void(std::uint32_t, std::uint32_t)> row_write_hook;

  std::vector<DeviceBuffer<__half>> k_cache, v_cache;
  /// Opt-in block-sparse attention on the global layers (FLINT_SPARSE_ATTN):
  /// pooled key bounds per 64-key block, and the number of older blocks kept.
  std::vector<DeviceBuffer<__half>> kmm;
  bool sparse{false};
  std::uint32_t sparse_top{256};
  std::uint32_t sparse_nblk{0};
  DeviceBuffer<std::int32_t> tokens, zero_ids;
  /// Routing of one bank: the expert choice stays alive until its experts run.
  struct RouteBank {
    DeviceBuffer<std::int32_t> ids;
    DeviceBuffer<float> topk;
    DeviceBuffer<std::uint32_t> counts;
    std::uint32_t* h_counts{nullptr};
    hipEvent_t ready{nullptr};
    RouteBank() = default;
    RouteBank(const RouteBank&) = delete;
    RouteBank& operator=(const RouteBank&) = delete;
    ~RouteBank() {
      if (ready != nullptr)
        (void)hipEventDestroy(ready);
      if (h_counts != nullptr)
        (void)hipHostFree(h_counts);
    }
    bool Allocate(std::size_t slots, std::uint32_t experts) {
      return ids.Allocate(slots) && topk.Allocate(slots) &&
             counts.Allocate(experts) &&
             hipHostMalloc(reinterpret_cast<void**>(&h_counts),
                           experts * sizeof(std::uint32_t)) == hipSuccess &&
             hipEventCreateWithFlags(&ready, hipEventDisableTiming) ==
                 hipSuccess;
    }
  };
  RouteBank rb;
  RouteBank* rt{nullptr};
  DeviceBuffer<std::int32_t> route_bounds, route_cursors, rows_token, rows_slot;
  DeviceBuffer<std::int32_t> tiles_pair, tiles_down;
  DeviceBuffer<__half> act_half, q_half;
  /// Decode: the shared expert is independent of routing, so its GEMVs run on a
  /// second stream (a parallel branch of the captured graph).
  hipStream_t side_stream{nullptr};
  hipEvent_t fork_event{nullptr};
  hipEvent_t join_event{nullptr};
  bool shared_forked{false};
  float* cx{nullptr};
  float* cxn{nullptr};
  const std::uint32_t* cpos{nullptr};
  std::int32_t* h_tiles{nullptr};
  /// Layer state between AttnCore and AttnOut: row count of the last chunk,
  /// its Q8_1 rows and whether the attention output is quantized.
  std::uint32_t ln{0};
  void* lq8{nullptr};
  bool ao_q8{false};
  std::uint32_t tile_capacity{0};
  DeviceBuffer<std::uint32_t> pos;
  DeviceBuffer<float> x, xn, q, k, v, qkv, attn, o, router;
  DeviceBuffer<float> gate_e, up_e, down_e, sh_gate, sh_up, sh_out;
  DeviceBuffer<float> expert_scale, shared_scale, expert_const;
  DeviceBuffer<float> dense_gate, dense_up, last, d_logits, attn_part;
  /// Layer-residual capture for the DFlash2 drafter: `tap_layers` names the
  /// layers and the ring keeps the last `hidden_ring_rows` positions as
  /// [row][tap][hidden] F32.
  std::vector<std::uint32_t> tap_layers;
  DeviceBuffer<float> hidden_ring;
  std::uint32_t hidden_ring_rows{0};
  std::vector<float> h_hidden;

  ~Impl() {
    if (stream != nullptr) {
      (void)hipStreamSynchronize(stream);
      for (auto& g : graphs) {
        if (g.exec != nullptr)
          (void)hipGraphExecDestroy(g.exec);
      }
      (void)hipStreamDestroy(stream);
    }
    if (q8_a != nullptr)
      (void)hipFree(q8_a);
    if (q8_b != nullptr)
      (void)hipFree(q8_b);
    if (q8_tiled != nullptr)
      (void)hipFree(q8_tiled);
    if (side_stream != nullptr)
      (void)hipStreamDestroy(side_stream);
    if (fork_event != nullptr)
      (void)hipEventDestroy(fork_event);
    if (join_event != nullptr)
      (void)hipEventDestroy(join_event);
    if (x_half != nullptr)
      (void)hipFree(x_half);
    if (xh != nullptr)
      (void)hipFree(xh);
    if (h_logits != nullptr)
      (void)hipHostFree(h_logits);
    if (h_tokens != nullptr)
      (void)hipHostFree(h_tokens);
    if (h_pos != nullptr)
      (void)hipHostFree(h_pos);
    if (h_tiles != nullptr)
      (void)hipHostFree(h_tiles);
  }

  struct Q8Rows {
    const float* x{nullptr};
    const void* q{nullptr};
  };

  bool Init(std::uint32_t max_context, std::string* error_msg);
  void SetBank() {
    cx = x.get();
    cxn = xn.get();
    cpos = pos.get();
    rt = &rb;
  }
  const __half* XnHalf(std::uint32_t n);
  void XnChanged() { xh_valid = false; }
  bool MoePre(const DeviceLayer& layer, std::uint32_t n,
              std::string* error_msg);
  bool ForkShared(const DeviceLayer& layer, std::uint32_t n,
                  std::string* error_msg);
  bool AttnCore(std::uint32_t il, std::uint32_t n, bool prune,
                std::string* error_msg);
  bool AttnOut(std::uint32_t il, std::string* error_msg);
  bool Forward(std::span<const std::int32_t> tokens, bool all_rows,
               std::string* error_msg);
  bool Enqueue(std::uint32_t n, bool all_rows, std::string* error_msg);
  bool Quantize(const float* x, std::uint32_t n, std::uint32_t k, void* slot,
                Q8Rows* out, std::string* error_msg);
  bool Project(const DeviceTensor& w, const Q8Rows& in, float* out,
               std::uint32_t n, std::string* error_msg);
  bool Gated(const DeviceTensor& up, const DeviceTensor& gate, const Q8Rows& in,
             float* out, float* scratch, std::uint32_t n,
             std::string* error_msg, float* scale_out = nullptr);
  bool Dense(const DeviceTensor& w, const float* in, float* out,
             std::uint32_t n, std::string* error_msg);
  bool Experts(const DeviceTensor& w, const float* in, float* out,
               std::uint32_t rows, std::uint32_t used, std::uint32_t n_tokens,
               std::string* error_msg);
  bool RoutedBegin(std::uint32_t n, std::string* error_msg);
  bool RoutedExperts(const DeviceLayer& layer, std::uint32_t n,
                     std::string* error_msg);
  bool Ffn(const DeviceLayer& layer, std::uint32_t n, const float* next_gamma,
           std::string* error_msg);
  bool Moe(const DeviceLayer& layer, std::uint32_t n, const float* next_gamma,
           std::string* error_msg);
  /// Copies the residual of layer `il` (rows [0, rows) of cx) into the ring.
  bool CaptureHidden(std::uint32_t il, std::uint32_t n, std::uint32_t rows,
                     std::string* error_msg);
  /// Drops every captured step graph: the capture copies are part of a step,
  /// so a change to the tap set changes the graph.
  void InvalidateGraphs();
};

constexpr std::uint32_t kW8A8MinK = 4096;
constexpr std::uint32_t kWarmTokens = 64;
constexpr std::uint32_t kRoutePadRows = 16;
constexpr std::uint32_t kPairTileRows = 128;
constexpr std::uint32_t kPlainTileRows = 48;
constexpr std::uint32_t kQ8TileRows = 64;

std::uint32_t tile_capacity_for(std::size_t slots, std::uint32_t experts) {
  return static_cast<std::uint32_t>(
      (slots + std::size_t{experts} * (kRoutePadRows - 1)) / kRoutePadRows +
      experts + 1);
}

/// Device bytes of every tile-scratch buffer that Session::Impl::Init allocates
/// (all of it except the KV caches, block-sparse key bounds, hidden ring and
/// pinned host staging). Model::SessionBytes uses this same function so the
/// reported footprint follows the allocations; keep the two in step.
std::size_t SessionScratchBytes(const Config& c, std::uint32_t tile) {
  const std::size_t T = tile;
  const std::size_t H = c.hidden_size;
  const std::size_t Q = std::size_t{c.num_heads} * c.head_dim;
  const std::size_t KV = std::size_t{c.num_kv} * c.head_dim;
  const std::size_t E = c.expert_ff;
  const std::size_t slots = T * c.expert_used;
  const std::size_t rows = gufo::vendor::qwen38_flash_next::rocm::RoutedCompactRows(
      slots, c.expert_count);
  const std::size_t tiles = tile_capacity_for(slots, c.expert_count);
  const std::size_t q8_bytes = flint_mmq_q8_1_bytes(kVecBatch, kQ8MaxK);
  const std::size_t tiled_bytes =
      gufo::vendor::qwen38_flash_next::rocm::Q8TiledBytes(tile, kQ8MaxK);
  const std::size_t f4 = sizeof(float);
  const std::size_t h2 = sizeof(__half);
  const std::size_t i4 = sizeof(std::int32_t);
  const std::size_t u4 = sizeof(std::uint32_t);
  return
      // tokens, zero_ids, pos
      2 * T * i4 + std::size_t{kVecBatch} * i4 + 2 * u4 +
      // hidden states, attention, QKV, router
      2 * T * H * f4 + T * Q * f4 + 2 * T * KV * f4 + T * (Q + 2 * KV) * f4 +
      T * Q * f4 + T * H * f4 + T * std::size_t{c.expert_count} * f4 +
      // routed experts: gate/up/down, routing bank, compaction, tiles
      2 * slots * E * f4 + slots * H * f4 + slots * (i4 + f4) +
      std::size_t{kMaxExperts} * u4 + (std::size_t{kMaxExperts} + 1) * i4 +
      std::size_t{kMaxExperts} * i4 + 2 * rows * i4 + 2 * tiles * i4 +
      slots * E * h2 + T * Q * h2 + slots * f4 + slots * f4 + T * f4 +
      // shared and dense FFN, final norm, logits, global attention partials
      2 * T * std::size_t{c.shared_ff} * f4 + T * H * f4 +
      2 * T * std::size_t{c.dense_ff} * f4 + H * f4 +
      std::size_t{Session::kMaxVerifyRows} * c.vocab_size * f4 +
      std::size_t{kVecBatch} * c.num_heads * kGlobalSplits * (c.head_dim + 2) *
          f4 +
      // quantized activations: tiled W8A8, F16 copies, Q8_1 scratch
      tiled_bytes + T * H * h2 + T * kQ8MaxK * h2 + 2 * q8_bytes;
}

bool Session::Impl::Init(std::uint32_t max_context, std::string* error_msg) {
  const Config& c = model->config;
  if (c.expert_used > kMaxTopK || c.expert_count > kMaxExperts) {
    return Fail(error_msg, "Flint: expert counts exceed the router limit");
  }
  capacity = max_context;
  tile = std::max<std::uint32_t>(1, model->options.prefill_tile);
  FLINT_HIP(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
  FLINT_HIP(hipStreamCreateWithFlags(&side_stream, hipStreamNonBlocking));
  FLINT_HIP(hipEventCreateWithFlags(&fork_event, hipEventDisableTiming));
  FLINT_HIP(hipEventCreateWithFlags(&join_event, hipEventDisableTiming));
  if (flint_mmq_init(0) != 0)
    return Fail(error_msg, "mmq initialization failed");
  std::string blaslt_error;
  blaslt = gufo::vendor::qwen38_flash_next::rocm::BlasLt::Create(stream, &blaslt_error);
  if (!blaslt)
    return Fail(error_msg, "hipBLASLt: " + blaslt_error);

  const std::size_t kv_row = static_cast<std::size_t>(c.num_kv) * c.head_dim;
  const std::size_t q_row = static_cast<std::size_t>(c.num_heads) * c.head_dim;
  const std::size_t slots = static_cast<std::size_t>(tile) * c.expert_used;
  k_cache.resize(c.num_layers);
  v_cache.resize(c.num_layers);
  bool ok = true;
  for (std::uint32_t il = 0; il < c.num_layers; ++il) {
    ok = ok && k_cache[il].Allocate(kv_row * capacity) &&
         v_cache[il].Allocate(kv_row * capacity);
  }
  if (const char* e = std::getenv("FLINT_SPARSE_ATTN");
      e != nullptr && e[0] != '0') {
    sparse_nblk = capacity / 64 + 2;
    if (const char* t = std::getenv("FLINT_SPARSE_TOPK"))
      sparse_top = static_cast<std::uint32_t>(std::atoi(t));
    sparse = sparse_nblk <= 4096 && sparse_top >= 1 && sparse_top <= 400;
    if (sparse) {
      kmm.resize(c.num_layers);
      for (std::uint32_t il = 0; il < c.num_layers; ++il) {
        if (c.IsSwa(il))
          continue;
        ok = ok && kmm[il].Allocate(std::size_t{c.num_kv} * sparse_nblk * 256);
        FLINT_HIP(hipMemsetAsync(
            kmm[il].get(), 0,
            std::size_t{c.num_kv} * sparse_nblk * 256 * sizeof(__half),
            stream));
      }
      std::fprintf(stderr, "[flint] block-sparse attention: %u blocks of 64\n",
                   sparse_top);
    }
  }
  ok = ok && tokens.Allocate(2 * std::size_t{tile}) && pos.Allocate(2) &&
       zero_ids.Allocate(kVecBatch) &&
       x.Allocate(std::size_t{tile} * c.hidden_size) &&
       xn.Allocate(std::size_t{tile} * c.hidden_size) &&
       q.Allocate(std::size_t{tile} * q_row) &&
       k.Allocate(std::size_t{tile} * kv_row) &&
       v.Allocate(std::size_t{tile} * kv_row) &&
       qkv.Allocate(std::size_t{tile} * (q_row + 2 * kv_row)) &&
       attn.Allocate(std::size_t{tile} * q_row) &&
       o.Allocate(std::size_t{tile} * c.hidden_size) &&
       router.Allocate(std::size_t{tile} * c.expert_count) &&
       gate_e.Allocate(slots * c.expert_ff) &&
       up_e.Allocate(slots * c.expert_ff) &&
       down_e.Allocate(slots * c.hidden_size) &&
       rb.Allocate(slots, kMaxExperts) &&
       route_bounds.Allocate(kMaxExperts + 1) &&
       route_cursors.Allocate(kMaxExperts) &&
       rows_token.Allocate(
           gufo::vendor::qwen38_flash_next::rocm::RoutedCompactRows(slots, c.expert_count)) &&
       rows_slot.Allocate(
           gufo::vendor::qwen38_flash_next::rocm::RoutedCompactRows(slots, c.expert_count)) &&
       tiles_pair.Allocate(tile_capacity_for(slots, c.expert_count)) &&
       tiles_down.Allocate(tile_capacity_for(slots, c.expert_count)) &&
       act_half.Allocate(slots * c.expert_ff) &&
       q_half.Allocate(std::size_t{tile} * q_row) &&
       hipHostMalloc(reinterpret_cast<void**>(&h_tiles),
                     2 * std::size_t{tile_capacity_for(slots, c.expert_count)} *
                         sizeof(std::int32_t)) == hipSuccess &&
       expert_scale.Allocate(slots) && shared_scale.Allocate(tile) &&
       expert_const.Allocate(slots) &&
       sh_gate.Allocate(std::size_t{tile} * c.shared_ff) &&
       sh_up.Allocate(std::size_t{tile} * c.shared_ff) &&
       sh_out.Allocate(std::size_t{tile} * c.hidden_size) &&
       dense_gate.Allocate(std::size_t{tile} * c.dense_ff) &&
       dense_up.Allocate(std::size_t{tile} * c.dense_ff) &&
       last.Allocate(c.hidden_size) &&
       d_logits.Allocate(std::size_t{Session::kMaxVerifyRows} * c.vocab_size) &&
       attn_part.Allocate(std::size_t{kVecBatch} * c.num_heads * kGlobalSplits *
                          (c.head_dim + 2));
  const std::size_t q8_bytes = flint_mmq_q8_1_bytes(kVecBatch, kQ8MaxK);
  const std::size_t tiled_bytes =
      gufo::vendor::qwen38_flash_next::rocm::Q8TiledBytes(tile, kQ8MaxK);
  ok = ok && hipMalloc(&q8_tiled, tiled_bytes) == hipSuccess &&
       hipMalloc(reinterpret_cast<void**>(&xh),
                 std::size_t{tile} * c.hidden_size * sizeof(__half)) ==
           hipSuccess &&
       hipMalloc(reinterpret_cast<void**>(&x_half),
                 std::size_t{tile} * kQ8MaxK * sizeof(__half)) == hipSuccess &&
       hipMalloc(&q8_a, q8_bytes) == hipSuccess &&
       hipMalloc(&q8_b, q8_bytes) == hipSuccess &&
       hipHostMalloc(reinterpret_cast<void**>(&h_logits),
                     std::size_t{Session::kMaxVerifyRows} * c.vocab_size *
                         sizeof(float)) == hipSuccess &&
       hipHostMalloc(reinterpret_cast<void**>(&h_tokens),
                     2 * std::size_t{tile} * sizeof(std::int32_t)) ==
           hipSuccess &&
       hipHostMalloc(reinterpret_cast<void**>(&h_pos),
                     2 * sizeof(std::uint32_t)) == hipSuccess;
  if (!ok)
    return Fail(error_msg, "Flint: session allocation failed");
  {
    // Drift guard: the buffers above must total what SessionScratchBytes
    // reports, which is what Model::SessionBytes uses for admission.
    const std::size_t allocated =
        tokens.bytes() + zero_ids.bytes() + pos.bytes() + x.bytes() +
        xn.bytes() + q.bytes() + k.bytes() + v.bytes() + qkv.bytes() +
        attn.bytes() + o.bytes() + router.bytes() + gate_e.bytes() +
        up_e.bytes() + down_e.bytes() + rb.ids.bytes() + rb.topk.bytes() +
        rb.counts.bytes() + route_bounds.bytes() + route_cursors.bytes() +
        rows_token.bytes() + rows_slot.bytes() + tiles_pair.bytes() +
        tiles_down.bytes() + act_half.bytes() + q_half.bytes() +
        expert_scale.bytes() + shared_scale.bytes() + expert_const.bytes() +
        sh_gate.bytes() + sh_up.bytes() + sh_out.bytes() + dense_gate.bytes() +
        dense_up.bytes() + last.bytes() + d_logits.bytes() +
        attn_part.bytes() + tiled_bytes +
        std::size_t{tile} * c.hidden_size * sizeof(__half) +
        std::size_t{tile} * kQ8MaxK * sizeof(__half) + 2 * q8_bytes;
    if (allocated != SessionScratchBytes(c, tile))
      return Fail(error_msg, "Flint: session scratch size mismatch");
  }
  SetBank();
  {
    // A constant fill done on the device, on the session's stream. A pageable
    // host->device hipMemcpy here goes through the runtime's staged copy and
    // never completes once enough sessions are resident (the host thread
    // spins in hsa_signal_wait with the GPU idle).
    static_assert(sizeof(float) == sizeof(std::uint32_t));
    std::uint32_t bits = 0;
    const float scale = kRoutedActScale;
    std::memcpy(&bits, &scale, sizeof(bits));
    FLINT_HIP(hipMemsetD32Async(
        reinterpret_cast<hipDeviceptr_t>(expert_const.get()), bits, slots,
        stream));
  }
  FLINT_HIP(
      hipMemsetAsync(zero_ids.get(), 0, kVecBatch * sizeof(int32_t), stream));
  std::fill(h_logits,
            h_logits + std::size_t{Session::kMaxVerifyRows} * c.vocab_size,
            0.0F);
  return true;
}

bool Session::Impl::Dense(const DeviceTensor& w, const float* in, float* out,
                          std::uint32_t n, std::string* error_msg) {
  if (w.type == GgmlType::kQ8_0 && !Narrow(n) && w.cols < kW8A8MinK) {
    rocm::NarrowRows(in, x_half, n, w.cols, w.cols, stream);
    if (!gufo::vendor::qwen38_flash_next::rocm::DenseF16Gemm(w.data, x_half, out, n, w.rows,
                                               w.cols, stream)) {
      return Fail(error_msg, "Q8 F16 GEMM failed");
    }
    return true;
  }
  if (w.half_data != nullptr && !Narrow(n)) {
    rocm::NarrowRows(in, x_half, n, w.cols, w.cols, stream);
    if (!gufo::vendor::qwen38_flash_next::rocm::DenseHalfWeightGemm(
            static_cast<const __half*>(w.half_data), x_half, out, n, w.rows,
            w.cols, stream)) {
      return Fail(error_msg, "F16 GEMM failed");
    }
    return true;
  }
  if (w.type == GgmlType::kQ8_0) {
    if (!Narrow(n) && w.rows != kStackedQkvRows) {
      gufo::vendor::qwen38_flash_next::rocm::QuantizeQ8Tiled(in, q8_tiled, n, w.cols, stream);
      if (gufo::vendor::qwen38_flash_next::rocm::W8A8Gemm(w.data, q8_tiled, out, n, w.rows,
                                            w.cols, stream)) {
        return true;
      }
    }
    if (flint_mmq_q8_0_dense(w.data, in, out, static_cast<int>(w.rows),
                           static_cast<int>(n), static_cast<int>(w.cols),
                           stream) != 0) {
      return Fail(error_msg, "Q8_0 dense projection failed");
    }
    return true;
  }
  for (std::uint32_t r0 = 0; r0 < n; r0 += kVecBatch) {
    const std::uint32_t rows = std::min(kVecBatch, n - r0);
    const float* src = in + static_cast<std::size_t>(r0) * w.cols;
    float* dst = out + static_cast<std::size_t>(r0) * w.rows;
    // An unselected expert id set: one ungated row per token.
    const int rc =
        flint_mmq_moe_vec(static_cast<int>(w.type), w.data, src, zero_ids.get(),
                        dst, static_cast<int>(w.rows), static_cast<int>(w.cols),
                        static_cast<int>(rows), 1, 1, stream, nullptr, nullptr);
    if (rc != 0)
      return Fail(error_msg, "quantized dense projection failed");
  }
  return true;
}

bool Session::Impl::Experts(const DeviceTensor& w, const float* in, float* out,
                            std::uint32_t rows, std::uint32_t used,
                            std::uint32_t n_tokens, std::string* error_msg) {
  const int M = static_cast<int>(w.rows);
  const int K = static_cast<int>(w.cols);
  const int E = static_cast<int>(w.experts);
  const int T = static_cast<int>(rows);
  const int U = static_cast<int>(used);
  const std::int32_t* routed_ids = rt->ids.get();
  int rc = -1;
  if (VecExperts(rows * used)) {
    rc = flint_mmq_moe_vec(static_cast<int>(w.type), w.data, in, routed_ids, out,
                         M, K, T, E, U, stream, nullptr, nullptr);
  } else {
    flint_mmq_set_routed_max_expert_rows(static_cast<int>(n_tokens));
    flint_mmq_set_routed_tile_cols(
        RoutedTileCols(n_tokens, model->config.expert_used, w.experts));
    switch (w.type) {
      case GgmlType::kQ4_K:
        rc = flint_mmq_q4_K_moe_raw(w.data, in, routed_ids, out, M, K, T, E, U,
                                  stream);
        break;
      case GgmlType::kQ5_K:
        rc = flint_mmq_q5_K_moe_raw(w.data, in, routed_ids, out, M, K, T, E, U,
                                  stream);
        break;
      case GgmlType::kQ5_1:
        rc = flint_mmq_q5_1_moe_raw(w.data, in, routed_ids, out, M, K, T, E, U,
                                  stream);
        break;
      case GgmlType::kQ8_0:
        rc = flint_mmq_q8_0_moe_raw(w.data, in, routed_ids, out, M, K, T, E, U,
                                  stream);
        break;
      case GgmlType::kIQ3_S:
        rc = flint_mmq_iq3_s_moe_raw(w.data, in, routed_ids, out, M, K, T, E, U,
                                   stream);
        break;
      case GgmlType::kIQ4_XS:
        rc = flint_mmq_iq4_xs_moe_raw(w.data, in, routed_ids, out, M, K, T, E, U,
                                    stream);
        break;
      default:
        break;
    }
  }
  if (!VecExperts(rows * used)) {
    flint_mmq_set_routed_max_expert_rows(0);
    flint_mmq_set_routed_tile_cols(0);
  }
  if (rc != 0)
    return Fail(error_msg, "routed expert projection failed");
  return true;
}

bool Session::Impl::Quantize(const float* x, std::uint32_t n, std::uint32_t k,
                             void* slot, Q8Rows* out, std::string* error_msg) {
  out->x = x;
  out->q = nullptr;
  if (!Narrow(n))
    return true;
  if (flint_mmq_quantize_q8_1(x, slot, static_cast<int>(n), static_cast<int>(k),
                            stream) != 0) {
    return Fail(error_msg, "activation quantization failed");
  }
  out->q = slot;
  return true;
}

bool Session::Impl::Project(const DeviceTensor& w, const Q8Rows& in, float* out,
                            std::uint32_t n, std::string* error_msg) {
  if (in.q != nullptr && w.type == GgmlType::kQ8_0) {
    if (flint_mmq_q8_0_dense_vec_preq(
            w.data, nullptr, in.q, out, static_cast<int>(w.rows),
            static_cast<int>(n), static_cast<int>(w.cols), stream) != 0) {
      return Fail(error_msg, "Q8_0 GEMV failed");
    }
    return true;
  }
  return Dense(w, in.x, out, n, error_msg);
}

bool Session::Impl::Gated(const DeviceTensor& up, const DeviceTensor& gate,
                          const Q8Rows& in, float* out, float* scratch,
                          std::uint32_t n, std::string* error_msg,
                          float* scale_out) {
  if (in.q != nullptr && up.type == GgmlType::kQ8_0 &&
      gate.type == GgmlType::kQ8_0) {
    if (flint_mmq_q8_0_dense_vec_preq(
            up.data, gate.data, in.q, out, static_cast<int>(up.rows),
            static_cast<int>(n), static_cast<int>(up.cols), stream) != 0) {
      return Fail(error_msg, "gated Q8_0 GEMV failed");
    }
    return true;
  }
  if (!Dense(gate, in.x, out, n, error_msg) ||
      !Dense(up, in.x, scratch, n, error_msg)) {
    return false;
  }
  if (scale_out != nullptr) {
    rocm::SwigluScaled(out, scratch, scale_out, n, up.rows, stream);
  } else {
    rocm::Swiglu(out, scratch, static_cast<std::size_t>(n) * up.rows, stream);
  }
  return true;
}

bool Session::Impl::Ffn(const DeviceLayer& layer, std::uint32_t n,
                        const float* next_gamma, std::string* error_msg) {
  const Config& c = model->config;
  const Q8Rows in{cxn, Narrow(n) ? q8_a : nullptr};
  if (!Gated(layer.ffn_up, layer.ffn_gate, in, dense_gate.get(), dense_up.get(),
             n, error_msg)) {
    return false;
  }
  Q8Rows act;
  if (!Quantize(dense_gate.get(), n, c.dense_ff, q8_b, &act, error_msg) ||
      !Project(layer.ffn_down, act, o.get(), n, error_msg)) {
    return false;
  }
  XnChanged();
  rocm::ResidualNormQuant(cx, o.get(), next_gamma, cxn,
                          Narrow(n) ? q8_a : nullptr, n, c.hidden_size,
                          c.rms_eps, stream);
  timer.Mark(kStageDenseFfn, stream);
  return true;
}

bool Session::Impl::RoutedBegin(std::uint32_t n, std::string* error_msg) {
  const Config& c = model->config;
  gufo::vendor::qwen38_flash_next::rocm::ExpertCounts(rt->ids.get(), rt->counts.get(), n,
                                        c.expert_count, c.expert_used, stream);
  FLINT_HIP(hipMemcpyAsync(rt->h_counts, rt->counts.get(),
                           c.expert_count * sizeof(std::uint32_t),
                           hipMemcpyDeviceToHost, stream));
  FLINT_HIP(hipEventRecord(rt->ready, stream));
  return true;
}

bool Session::Impl::RoutedExperts(const DeviceLayer& layer, std::uint32_t n,
                                  std::string* error_msg) {
  using gufo::vendor::qwen38_flash_next::rocm::WeightType;
  const Config& c = model->config;
  const std::uint32_t used = c.expert_used;
  const std::uint32_t experts = c.expert_count;
  const DeviceTensor& ge = layer.gate_exps;
  const DeviceTensor& ue = layer.up_exps;
  const DeviceTensor& de = layer.down_exps;
  const auto gate_type = static_cast<WeightType>(ge.type);
  const auto down_type = static_cast<WeightType>(de.type);
  const bool gated =
      ge.type == ue.type &&
      (ge.type == GgmlType::kQ4_K || ge.type == GgmlType::kQ5_K ||
       ge.type == GgmlType::kIQ3_S || ge.type == GgmlType::kIQ4_XS);
  const std::uint32_t pair_rows = gated ? kPairTileRows : kPlainTileRows;
  const std::uint32_t down_rows =
      de.type == GgmlType::kQ8_0 ? kQ8TileRows : kPlainTileRows;

  FLINT_HIP(hipEventSynchronize(rt->ready));
  std::uint32_t pair_tiles = 0;
  std::uint32_t down_tiles = 0;
  std::int32_t* host_pair = h_tiles;
  std::int32_t* host_down =
      h_tiles +
      tile_capacity_for(static_cast<std::size_t>(tile) * used, experts);
  for (std::uint32_t e = 0; e < experts; ++e) {
    const std::uint32_t padded =
        (rt->h_counts[e] + kRoutePadRows - 1) / kRoutePadRows * kRoutePadRows;
    for (std::uint32_t j = 0; j * pair_rows < padded; ++j) {
      host_pair[pair_tiles++] = static_cast<std::int32_t>(e | (j << 16));
    }
    for (std::uint32_t j = 0; j * down_rows < padded; ++j) {
      host_down[down_tiles++] = static_cast<std::int32_t>(e | (j << 16));
    }
  }
  if (pair_tiles == 0)
    return Fail(error_msg, "routed experts: no tiles");
  FLINT_HIP(hipMemcpyAsync(tiles_pair.get(), host_pair,
                           pair_tiles * sizeof(std::int32_t),
                           hipMemcpyHostToDevice, stream));
  FLINT_HIP(hipMemcpyAsync(tiles_down.get(), host_down,
                           down_tiles * sizeof(std::int32_t),
                           hipMemcpyHostToDevice, stream));
  gufo::vendor::qwen38_flash_next::rocm::RoutedCompact(
      rt->ids.get(), rt->counts.get(), route_bounds.get(), route_cursors.get(),
      rows_token.get(), rows_slot.get(), n, used, experts, stream);
  const __half* xh_in = XnHalf(n);

  using gufo::vendor::qwen38_flash_next::rocm::RoutedF16Gemm;
  using gufo::vendor::qwen38_flash_next::rocm::RoutedGatedF16Gemm;
  bool ok;
  if (gated) {
    ok = RoutedGatedF16Gemm(
        ge.data, ue.data, gate_type, xh_in, tiles_pair.get(), pair_tiles,
        pair_rows, route_bounds.get(), rows_token.get(), rows_slot.get(),
        act_half.get(), ge.rows, ge.cols, stream, 1.0F / kRoutedActScale);
  } else {
    ok = RoutedF16Gemm(ge.data, gate_type, xh_in, tiles_pair.get(), pair_tiles,
                       pair_rows, route_bounds.get(), rows_token.get(),
                       rows_slot.get(), nullptr, gate_e.get(), nullptr, ge.rows,
                       ge.cols, stream) &&
         RoutedF16Gemm(ue.data, static_cast<WeightType>(ue.type), xh_in,
                       tiles_pair.get(), pair_tiles, pair_rows,
                       route_bounds.get(), rows_token.get(), rows_slot.get(),
                       gate_e.get(), nullptr, act_half.get(), ue.rows, ue.cols,
                       stream);
  }
  if (!ok)
    return Fail(error_msg, "routed gate/up projection failed");
  timer.Mark(kStageExpertsGateUp, stream);
  if (!RoutedF16Gemm(de.data, down_type, act_half.get(), tiles_down.get(),
                     down_tiles, down_rows, route_bounds.get(), rows_slot.get(),
                     rows_slot.get(), nullptr, down_e.get(), nullptr, de.rows,
                     de.cols, stream)) {
    return Fail(error_msg, "routed down projection failed");
  }
  timer.Mark(kStageExpertsDown, stream);
  return true;
}

const __half* Session::Impl::XnHalf(std::uint32_t n) {
  const std::uint32_t hidden = model->config.hidden_size;
  if (!xh_valid) {
    rocm::NarrowRows(cxn, xh, n, hidden, hidden, stream);
    xh_valid = true;
  }
  return xh;
}

bool Session::Impl::MoePre(const DeviceLayer& layer, std::uint32_t n,
                           std::string* error_msg) {
  const Config& c = model->config;
  const std::uint32_t used = c.expert_used;
  if (!Narrow(n) && layer.router.half_data != nullptr) {
    const __half* xh_in = XnHalf(n);
    if (!gufo::vendor::qwen38_flash_next::rocm::DenseHalfWeightGemm(
            static_cast<const __half*>(layer.router.half_data), xh_in,
            router.get(), n, c.expert_count, c.hidden_size, stream)) {
      return Fail(error_msg, "router GEMM failed");
    }
  } else {
    rocm::RouterBf16(layer.router.data, cxn, router.get(), n, c.expert_count,
                     c.hidden_size, stream);
  }
  rocm::SigmoidTopK(router.get(), rt->ids.get(), rt->topk.get(), n,
                    c.expert_count, used, c.expert_weights_norm,
                    c.expert_weights_scale, stream);
  timer.Mark(kStageRouter, stream);
  if (!VecExperts(n * used) && !RoutedBegin(n, error_msg))
    return false;
  return true;
}

bool Session::Impl::ForkShared(const DeviceLayer& layer, std::uint32_t n,
                               std::string* error_msg) {
  shared_forked = false;
  const Config& c = model->config;
  // Only the pre-quantized Q8_0 GEMVs, which keep no shared scratch pool.
  if (!Narrow(n) || timer.on() || layer.up_shexp.type != GgmlType::kQ8_0 ||
      layer.gate_shexp.type != GgmlType::kQ8_0 ||
      layer.down_shexp.type != GgmlType::kQ8_0) {
    return true;
  }
  FLINT_HIP(hipEventRecord(fork_event, stream));
  FLINT_HIP(hipStreamWaitEvent(side_stream, fork_event, 0));
  std::swap(stream, side_stream);
  const Q8Rows in{cxn, q8_a};
  Q8Rows act;
  const bool ok =
      Gated(layer.up_shexp, layer.gate_shexp, in, sh_gate.get(), sh_up.get(), n,
            error_msg, nullptr) &&
      Quantize(sh_gate.get(), n, c.shared_ff, q8_b, &act, error_msg) &&
      Project(layer.down_shexp, act, sh_out.get(), n, error_msg);
  const hipError_t rec = hipEventRecord(join_event, stream);
  std::swap(stream, side_stream);
  if (!ok)
    return false;
  if (rec != hipSuccess)
    return Fail(error_msg, "shared-expert join event failed");
  shared_forked = true;
  return true;
}

namespace {
/// Whether the routed F16 tile kernels read this weight format. Expert
/// tensors outside this set (IQ3_S/IQ4_XS in the released GGUF) run through
/// the quantized routed kernels at every width instead.
bool RoutedF16Weights(GgmlType type) {
  return type == GgmlType::kQ4_K || type == GgmlType::kQ5_K ||
         type == GgmlType::kQ5_1 || type == GgmlType::kQ6_K ||
         type == GgmlType::kQ8_0;
}
}  // namespace

bool Session::Impl::Moe(const DeviceLayer& layer, std::uint32_t n,
                        const float* next_gamma, std::string* error_msg) {
  const Config& c = model->config;
  const std::uint32_t used = c.expert_used;
  using gufo::vendor::qwen38_flash_next::rocm::WeightType;
  const DeviceTensor& ge_check = layer.gate_exps;
  const DeviceTensor& ue_check = layer.up_exps;
  const bool gated_check =
      ge_check.type == ue_check.type &&
      (ge_check.type == GgmlType::kQ4_K || ge_check.type == GgmlType::kQ5_K ||
       ge_check.type == GgmlType::kIQ3_S || ge_check.type == GgmlType::kIQ4_XS);
  const bool routed_f16 =
      (gated_check ? gufo::vendor::qwen38_flash_next::rocm::RoutedGatedF16Supported(
                         static_cast<WeightType>(ge_check.type))
                   : RoutedF16Weights(ge_check.type) &&
                         RoutedF16Weights(ue_check.type)) &&
      RoutedF16Weights(layer.down_exps.type);
  const bool wide = !VecExperts(n * used) && routed_f16;
  const DeviceTensor& ge = layer.gate_exps;
  const DeviceTensor& ue = layer.up_exps;
  // Wide batches feed the shared down projection through F16, and non-Q8_0
  // types through Q8_1 half sums: both need rows kept inside the half range.
  bool scale_shared = layer.down_shexp.type != GgmlType::kQ8_0 || wide;
  const float* shared_scale_ptr = shared_scale.get();
  const bool forked = shared_forked;
  shared_forked = false;
  if (!forked) {
    const Q8Rows in{cxn, Narrow(n) ? q8_a : nullptr};
    Q8Rows act;
    if (!Gated(layer.up_shexp, layer.gate_shexp, in, sh_gate.get(), sh_up.get(),
               n, error_msg, scale_shared ? shared_scale.get() : nullptr) ||
        !Quantize(sh_gate.get(), n, c.shared_ff, q8_b, &act, error_msg) ||
        !Project(layer.down_shexp, act, sh_out.get(), n, error_msg)) {
      return false;
    }
    timer.Mark(kStageShared, stream);
  }
  bool scale_experts = false;
  const float* expert_scale_ptr = expert_scale.get();
  if (wide) {
    scale_experts = true;
    expert_scale_ptr = expert_const.get();
    if (!RoutedExperts(layer, n, error_msg))
      return false;
  }
  if (!wide) {
    const bool paired = VecExperts(n * used) && ge.type == ue.type &&
                        ge.rows == ue.rows && ge.cols == ue.cols;
    if (paired) {
      if (flint_mmq_moe_vec(static_cast<int>(ge.type), ge.data, cxn,
                          rt->ids.get(), gate_e.get(),
                          static_cast<int>(ge.rows), static_cast<int>(ge.cols),
                          static_cast<int>(n), static_cast<int>(ge.experts),
                          static_cast<int>(used), stream, ue.data,
                          up_e.get()) != 0) {
        return Fail(error_msg, "paired expert projection failed");
      }
    } else if (!Experts(ge, cxn, gate_e.get(), n, used, n, error_msg) ||
               !Experts(ue, cxn, up_e.get(), n, used, n, error_msg)) {
      return false;
    }
    // Q4_K/Q5_K/IQ4_XS down projections read the half-precision sum of each
    // Q8_1 block, which overflows for the very large rows of sink-like tokens.
    timer.Mark(kStageExpertsGateUp, stream);
    scale_experts = layer.down_exps.type != GgmlType::kQ8_0;
    if (scale_experts) {
      rocm::SwigluScaled(gate_e.get(), up_e.get(), expert_scale.get(), n * used,
                         c.expert_ff, stream);
    } else {
      rocm::Swiglu(gate_e.get(), up_e.get(),
                   static_cast<std::size_t>(n) * used * c.expert_ff, stream);
    }
    timer.Mark(kStageSwiglu, stream);
    if (!Experts(layer.down_exps, gate_e.get(), down_e.get(), n * used, 1, n,
                 error_msg)) {
      return false;
    }
    timer.Mark(kStageExpertsDown, stream);
  }
  if (forked)
    FLINT_HIP(hipStreamWaitEvent(stream, join_event, 0));
  rocm::MoeCombineNormQuant(
      cx, down_e.get(), rt->topk.get(),
      scale_experts ? expert_scale_ptr : nullptr, sh_out.get(),
      scale_shared ? shared_scale_ptr : nullptr, used, next_gamma, cxn,
      Narrow(n) ? q8_a : nullptr, n, c.hidden_size, c.rms_eps, stream);
  XnChanged();
  timer.Mark(kStageCombine, stream);
  return true;
}

bool Session::Impl::CaptureHidden(std::uint32_t il, std::uint32_t n,
                                  std::uint32_t rows, std::string*) {
  std::uint32_t tap = 0;
  while (tap < tap_layers.size() && tap_layers[tap] != il)
    ++tap;
  if (tap == tap_layers.size())
    return true;
  // AttnCore's pruned final-layer path keeps only the last row and moves it to
  // cx row 0, so it is the residual of position n - 1, not of the chunk start.
  const bool pruned = rows < n;
  std::uint32_t count = pruned ? 1 : n;
  const std::uint32_t pos_offset = pruned ? n - 1 : 0;
  // A prefill chunk can be wider than the ring; only the newest rows survive.
  std::uint32_t first = 0;
  if (count > hidden_ring_rows) {
    first = count - hidden_ring_rows;
    count = hidden_ring_rows;
  }
  const std::size_t hidden = model->config.hidden_size;
  rocm::HiddenCapture(cx + std::size_t{first} * hidden, hidden_ring.get(),
                      pos.get(), pos_offset + first, count, hidden_ring_rows,
                      static_cast<std::uint32_t>(tap_layers.size()), tap,
                      static_cast<std::uint32_t>(hidden), stream);
  return true;
}

void Session::Impl::InvalidateGraphs() {
  for (auto& g : graphs) {
    if (g.exec != nullptr) {
      (void)hipGraphExecDestroy(g.exec);
      g.exec = nullptr;
    }
    g.failed = false;
    g.steps = 0;
  }
}

bool Session::Impl::AttnCore(std::uint32_t il, std::uint32_t n, bool prune,
                             std::string* error_msg) {
  const Config& c = model->config;
  const DeviceLayer& L = model->device->layers()[il];
  const std::uint32_t q_row = c.num_heads * c.head_dim;
  const std::uint32_t kv_row = c.num_kv * c.head_dim;
  const bool vec = Narrow(n);
  void* const q8_in = vec ? q8_a : nullptr;
  ln = n;
  lq8 = q8_in;
  ao_q8 = false;
  const Q8Rows in{cxn, q8_in};
  rocm::QkvSources src;
  if (L.attn_qkv.data != nullptr) {
    if (!Project(L.attn_qkv, in, qkv.get(), n, error_msg)) {
      return false;
    }
    const std::uint32_t stride = q_row + 2 * kv_row;
    src = {qkv.get(),
           stride,
           qkv.get() + q_row,
           stride,
           qkv.get() + q_row + kv_row,
           stride};
  } else {
    if (!Project(L.attn_q, in, q.get(), n, error_msg) ||
        !Project(L.attn_k, in, k.get(), n, error_msg) ||
        !Project(L.attn_v, in, v.get(), n, error_msg)) {
      return false;
    }
    src = {q.get(), q_row, k.get(), kv_row, v.get(), kv_row};
  }
  timer.Mark(kStageQkv, stream);
  rocm::QkvPrepare(src, static_cast<const float*>(L.attn_q_norm.data),
                   static_cast<const float*>(L.attn_k_norm.data), q.get(),
                   k_cache[il].get(), v_cache[il].get(), n, c.num_heads,
                   c.num_kv, c.head_dim, cpos, c.rope_base, c.rms_eps,
                   c.HasRope(il), stream, vec ? nullptr : q_half.get());
  const auto chunk_start = static_cast<std::uint32_t>(position);
  if (sparse && !vec && !c.IsSwa(il)) {
    rocm::PoolKeyBlocks(k_cache[il].get(), kmm[il].get(), cpos, chunk_start, n,
                        c.num_kv, sparse_nblk, stream);
  }
  timer.Mark(kStageRopeKv, stream);
  std::uint32_t ln = n;
  void* lq8 = q8_in;
  const std::uint32_t window = c.IsSwa(il) ? c.sliding_window : 0;
  Q8Rows ao{attn.get(), nullptr};
  if (vec) {
    // Only Q8_0 projections read the Q8_1 rows; other types need floats.
    const bool q8_out = L.attn_output.type == GgmlType::kQ8_0;
    rocm::AttentionSplit(q.get(), k_cache[il].get(), v_cache[il].get(),
                         static_cast<const float*>(L.attn_sinks.data),
                         attn.get(), q8_out ? q8_b : nullptr, attn_part.get(),
                         n, cpos, c.num_heads, c.num_kv, c.head_dim, window,
                         window > 0 ? kSwaSplits : kGlobalSplits, stream);
    ao.q = q8_out ? q8_b : nullptr;
    ao_q8 = q8_out;
  } else {
    // Past the last layer only the final token is read: its attention,
    // projection and FFN run as a single row.
    const __half* aq = q_half.get();
    const std::uint32_t* apos = cpos;
    std::uint32_t an = n;
    if (prune) {
      aq += static_cast<std::size_t>(n - 1) * q_row;
      apos += 1;
      an = 1;
      FLINT_HIP(hipMemcpyAsync(
          cx, cx + static_cast<std::size_t>(n - 1) * c.hidden_size,
          c.hidden_size * sizeof(float), hipMemcpyDeviceToDevice, stream));
      ln = 1;
      lq8 = q8_a;
    }
    const bool use_sparse =
        sparse && window == 0 &&
        chunk_start + n >
            (kSparseSinkBlocks + kSparseLocalBlocks + sparse_top + 1) * 64;
    if (use_sparse
            ? !rocm::AttentionWmmaSparse(
                  aq, k_cache[il].get(), v_cache[il].get(), kmm[il].get(),
                  sparse_nblk, sparse_top,
                  static_cast<const float*>(L.attn_sinks.data), attn.get(), an,
                  apos, c.num_heads, c.num_kv, c.head_dim, stream)
            : !rocm::AttentionWmma(aq, k_cache[il].get(), v_cache[il].get(),
                                   static_cast<const float*>(L.attn_sinks.data),
                                   attn.get(), an, apos, c.num_heads, c.num_kv,
                                   c.head_dim, window, stream)) {
      return Fail(error_msg, "Flint: unsupported attention geometry");
    }
  }
  timer.Mark(kStageAttention, stream);
  this->ln = ln;
  this->lq8 = lq8;
  return true;
}

bool Session::Impl::AttnOut(std::uint32_t il, std::string* error_msg) {
  const Config& c = model->config;
  const DeviceLayer& L = model->device->layers()[il];
  Q8Rows ao{attn.get(), nullptr};
  if (ao_q8)
    ao.q = q8_b;
  if (!Project(L.attn_output, ao, o.get(), ln, error_msg)) {
    return false;
  }
  timer.Mark(kStageNorm, stream);
  rocm::ResidualNormQuant(cx, o.get(),
                          static_cast<const float*>(L.ffn_norm.data), cxn, lq8,
                          ln, c.hidden_size, c.rms_eps, stream);
  XnChanged();
  timer.Mark(kStageOutProj, stream);
  return true;
}

bool Session::Impl::Enqueue(std::uint32_t n, bool all_rows,
                            std::string* error_msg) {
  const Config& c = model->config;
  const DeviceModel& dm = *model->device;
  const bool vec = Narrow(n);
  void* const q8_in = vec ? q8_a : nullptr;
  const bool prefill_call = wide_prefill;

  FLINT_HIP(hipMemcpyAsync(tokens.get(), h_tokens, n * sizeof(int32_t),
                           hipMemcpyHostToDevice, stream));
  FLINT_HIP(hipMemcpyAsync(pos.get(), h_pos, 2 * sizeof(std::uint32_t),
                           hipMemcpyHostToDevice, stream));
  timer.Mark(kStageEmbed, stream);
  rocm::EmbedQ8(dm.token_embd().data, tokens.get(), cx, n, c.hidden_size,
                stream);
  rocm::ResidualNormQuant(
      cx, nullptr, static_cast<const float*>(dm.layers()[0].attn_norm.data),
      cxn, q8_in, n, c.hidden_size, c.rms_eps, stream);
  XnChanged();
  timer.Mark(kStageEmbed, stream);

  for (std::uint32_t il = 0; il < c.num_layers; ++il) {
    const DeviceLayer& L = dm.layers()[il];
    const bool last = il + 1 == c.num_layers;
    const float* next_gamma = static_cast<const float*>(
        !last ? dm.layers()[il + 1].attn_norm.data : dm.output_norm().data);
    const bool prune = last && !all_rows && !vec;
    if (!AttnCore(il, n, prune, error_msg))
      return false;
    // The pruned final layer and the head work on one row in the vector
    // layout (AttnCore leaves it in cx row 0 with lq8 = q8_a), as in decode.
    // Its input row is final already, so this does not depend on the split.
    if (prune)
      wide_prefill = false;
    if (!AttnOut(il, error_msg))
      return false;
    if (L.is_moe && !ForkShared(L, ln, error_msg))
      return false;
    if (L.is_moe && !MoePre(L, ln, error_msg))
      return false;
    if (!(L.is_moe ? Moe(L, ln, next_gamma, error_msg)
                   : Ffn(L, ln, next_gamma, error_msg))) {
      return false;
    }
    // The residual of this layer is now final: cx holds the layer output.
    if (!tap_layers.empty() && !CaptureHidden(il, n, ln, error_msg))
      return false;
  }

  std::uint32_t head_rows = 1;
  const bool pruned = !all_rows && !vec;
  if (all_rows) {
    head_rows = n;
    const Q8Rows head{cxn, q8_a};
    if (!Project(dm.output(), head, d_logits.get(), n, error_msg))
      return false;
  } else {
    const std::size_t last_row = pruned ? 0 : static_cast<std::size_t>(n - 1);
    const Q8Rows head{cxn + last_row * c.hidden_size,
                      vec || pruned ? static_cast<const char*>(q8_a) +
                                          last_row * (c.hidden_size / 32) * 36
                                    : nullptr};
    Q8Rows head_rows_in = head;
    if (!vec && !pruned &&
        !Quantize(head.x, 1, c.hidden_size, q8_a, &head_rows_in, error_msg)) {
      return false;
    }
    if (!Project(dm.output(), head_rows_in, d_logits.get(), 1, error_msg)) {
      return false;
    }
  }
  timer.Mark(kStageHead, stream);
  wide_prefill = prefill_call;
  FLINT_HIP(
      hipMemcpyAsync(h_logits, d_logits.get(),
                     std::size_t{head_rows} * c.vocab_size * sizeof(float),
                     hipMemcpyDeviceToHost, stream));
  return true;
}

bool Session::Impl::Forward(std::span<const std::int32_t> input, bool all_rows,
                            std::string* error_msg) {
  const auto n = static_cast<std::uint32_t>(input.size());
  std::copy(input.begin(), input.end(), h_tokens);
  h_pos[0] = position;
  h_pos[1] = position + n - 1;
  logit_rows = all_rows ? n : 1;

  StepGraph& g = graphs[std::min<std::uint32_t>(n, Session::kMaxVerifyRows)];
  // FLINT_NO_GRAPH=1 keeps every step on the eager path: the row check uses it
  // to separate a kernel that depends on the row count from a captured graph
  // that replays with stale state.
  static const bool graphs_off = [] {
    const char* env = std::getenv("FLINT_NO_GRAPH");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
  }();
  const bool graphable = all_rows && n <= Session::kMaxVerifyRows && !graphs_off;
  if (graphable)
    ++g.steps;
  // The first step of each width runs eagerly so the quantized kernels'
  // scratch pools reach their final size before the step is captured.
  if (graphable && g.steps > 1 && g.exec == nullptr && !g.failed &&
      !timer.on()) {
    FLINT_HIP(hipStreamSynchronize(stream));
    FLINT_HIP(hipStreamBeginCapture(stream, hipStreamCaptureModeRelaxed));
    std::string capture_error;
    const bool enqueued = Enqueue(n, true, &capture_error);
    hipGraph_t graph = nullptr;
    const hipError_t ended = hipStreamEndCapture(stream, &graph);
    if (!enqueued || ended != hipSuccess ||
        hipGraphInstantiate(&g.exec, graph, nullptr, nullptr, 0) !=
            hipSuccess) {
      g.failed = true;
      g.exec = nullptr;
      (void)hipGetLastError();
      std::fprintf(stderr, "[flint] decode graph unavailable: %s\n",
                   capture_error.c_str());
    }
    if (graph != nullptr)
      (void)hipGraphDestroy(graph);
  }
  if (graphable && g.exec != nullptr) {
    FLINT_HIP(hipGraphLaunch(g.exec, stream));
  } else if (!Enqueue(n, all_rows, error_msg)) {
    return false;
  }
  FLINT_HIP(hipStreamSynchronize(stream));
  timer.Finish(n);
  position += n;
  return true;
}

Session::Session() : impl_(std::make_unique<Impl>()) {}
Session::~Session() = default;

bool Session::Append(std::span<const std::int32_t> tokens,
                     std::string* error_msg) {
  if (tokens.empty())
    return Fail(error_msg, "Flint: no tokens to append");
  if (impl_->position + tokens.size() > impl_->capacity) {
    return Fail(error_msg, "Flint: context capacity exceeded");
  }
  if (impl_->row_write_hook) {
    impl_->row_write_hook(impl_->position,
                          impl_->position +
                              static_cast<std::uint32_t>(tokens.size()));
  }
  const std::size_t chunk_max = impl_->tile;
  for (std::size_t off = 0; off < tokens.size();) {
    const std::size_t remaining = tokens.size() - off;
    const std::size_t n = std::min(remaining, chunk_max);
    const bool single = tokens.size() == 1 && !impl_->wide_prefill;
    if (!impl_->Forward(tokens.subspan(off, n), single, error_msg))
      return false;
    off += n;
  }
  return true;
}

bool Session::Prefill(std::span<const std::int32_t> tokens,
                      std::string* error_msg) {
  impl_->wide_prefill = true;
  const bool ok = Append(tokens, error_msg);
  impl_->wide_prefill = false;
  return ok;
}

bool Session::AppendAll(std::span<const std::int32_t> tokens,
                        std::string* error_msg) {
  if (tokens.empty() || tokens.size() > kMaxVerifyRows) {
    return Fail(error_msg, "Flint: unsupported verification width");
  }
  if (impl_->position + tokens.size() > impl_->capacity) {
    return Fail(error_msg, "Flint: context capacity exceeded");
  }
  if (impl_->row_write_hook) {
    impl_->row_write_hook(impl_->position,
                          impl_->position +
                              static_cast<std::uint32_t>(tokens.size()));
  }
  return impl_->Forward(tokens, true, error_msg);
}

void Session::SetRowWriteHook(
    std::function<void(std::uint32_t first, std::uint32_t end)> hook) {
  impl_->row_write_hook = std::move(hook);
}

void Session::Rewind(std::uint32_t position) noexcept {
  if (position < impl_->position)
    impl_->position = position;
}

bool Session::SetHiddenCapture(std::span<const std::uint32_t> layer_output_ids,
                               std::string* error_msg) {
  const Config& c = impl_->model->config;
  std::vector<std::uint32_t> layers(layer_output_ids.begin(),
                                    layer_output_ids.end());
  {
    std::vector<std::uint32_t> sorted(layers);
    std::sort(sorted.begin(), sorted.end());
    if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end())
      return Fail(error_msg, "Flint: duplicate hidden capture layer");
  }
  if (layers.size() > kMaxHiddenTaps)
    return Fail(error_msg, "Flint: too many hidden capture taps");
  for (const auto il : layers) {
    if (il >= c.num_layers)
      return Fail(error_msg, "Flint: hidden capture layer out of range");
  }
  // Identical tap sets (the per-request case) keep the captured step graphs
  // and the hidden ring valid; only a real change forces re-capture.
  if (layers == impl_->tap_layers)
    return true;
  // The capture copies live inside the step, so the step graphs change.
  impl_->InvalidateGraphs();
  impl_->tap_layers.clear();
  impl_->hidden_ring_rows = 0;
  impl_->hidden_ring.Reset();
  if (layers.empty()) {
    impl_->h_hidden.clear();
    return true;
  }
  const std::uint32_t rows = std::min(kHiddenRingRows, impl_->capacity);
  if (rows == 0)
    return Fail(error_msg, "Flint: session capacity is zero");
  if (!impl_->hidden_ring.Allocate(std::size_t{rows} * layers.size() *
                                   c.hidden_size)) {
    return Fail(error_msg, "Flint: hidden capture ring allocation failed");
  }
  impl_->hidden_ring_rows = rows;
  impl_->tap_layers = std::move(layers);
  impl_->h_hidden.assign(c.hidden_size, 0.0F);
  return true;
}

std::span<const std::uint32_t> Session::HiddenCaptureLayers() const noexcept {
  return impl_->tap_layers;
}

std::uint32_t Session::HiddenRingRows() const noexcept {
  return impl_->hidden_ring_rows;
}

const float* Session::CapturedHiddenDevice(std::uint32_t row,
                                           std::uint32_t tap) const noexcept {
  const auto& s = *impl_;
  const std::size_t taps = s.tap_layers.size();
  if (taps == 0 || tap >= taps || s.hidden_ring_rows == 0)
    return nullptr;
  // Only positions at or after the ring start, and before the next token, are
  // still in the ring.
  if (row >= s.position || row + s.hidden_ring_rows < s.position)
    return nullptr;
  const std::size_t hidden = s.model->config.hidden_size;
  const std::size_t slot = row % s.hidden_ring_rows;
  return s.hidden_ring.get() + (slot * taps + tap) * hidden;
}

bool Session::HasHidden(std::uint32_t row) const noexcept {
  return CapturedHiddenDevice(row, 0) != nullptr;
}

std::span<const float> Session::CapturedHidden(std::uint32_t row,
                                               std::uint32_t tap) {
  const float* src = CapturedHiddenDevice(row, tap);
  const std::uint32_t hidden = impl_->model->config.hidden_size;
  if (src == nullptr || hidden == 0)
    return {};
  impl_->h_hidden.resize(hidden);
  if (hipMemcpy(impl_->h_hidden.data(), src,
                std::size_t{hidden} * sizeof(float),
                hipMemcpyDeviceToHost) != hipSuccess) {
    return {};
  }
  return std::span<const float>(impl_->h_hidden.data(), hidden);
}

std::span<const float> Session::Logits() const noexcept {
  const std::uint32_t vocab = impl_->model->config.vocab_size;
  return {impl_->h_logits + std::size_t{impl_->logit_rows - 1} * vocab, vocab};
}

std::span<const float> Session::Logits(std::uint32_t row) const noexcept {
  const std::uint32_t vocab = impl_->model->config.vocab_size;
  return {impl_->h_logits + std::size_t{row} * vocab, vocab};
}
std::uint32_t Session::Position() const noexcept {
  return impl_->position;
}
std::uint32_t Session::Capacity() const noexcept {
  return impl_->capacity;
}

namespace {
constexpr std::uint32_t kSnapshotMagic = 0x544e4c46;  // "FLNT"
constexpr std::size_t kSnapshotHeaderBytes = 32;
}  // namespace

TiedWeights Model::Tied() const noexcept {
  const DeviceModel& device = *state_->device;
  return {.token_embedding = device.token_embd().data,
          .token_embedding_type = device.token_embd().type,
          .output = device.output().data,
          .output_type = device.output().type};
}

std::size_t Model::SnapshotBytes(std::uint32_t tokens) const noexcept {
  const Config& c = state_->config;
  const std::size_t kv_row = std::size_t{c.num_kv} * c.head_dim;
  return kSnapshotHeaderBytes +
         std::size_t{c.num_layers} * 2 * tokens * kv_row * sizeof(__half) +
         std::size_t{c.vocab_size} * sizeof(float);
}

std::size_t Model::SessionBytes(std::uint32_t max_context) const noexcept {
  const Config& c = state_->config;
  const std::size_t kv_row = std::size_t{c.num_kv} * c.head_dim;
  const std::uint32_t tile = std::max<std::uint32_t>(1, state_->options.prefill_tile);
  // Each session runs its warmup forward in CreateSession, which materializes
  // a private MMQ scratch arena: qfn_stack_pool::reserve starts each
  // (device, stream) arena at 256 MiB, and every session owns its own stream.
  // Nothing releases it (flint_mmq_release_stream is never called).
  constexpr std::size_t kMmqScratchArenaBytes = std::size_t{256} << 20;
  // hipGraphInstantiate for the decode step graph (fixed, independent of
  // context).
  constexpr std::size_t kSessionGraphBytes = std::size_t{64} << 20;
  // Each KV buffer is one hipMalloc. The HIP allocator hands out 2 MiB-granule
  // device blocks (measured with hipMemGetInfo: a 1 MiB request costs 2 MiB, a
  // 2 MiB request costs 2 MiB), so each buffer is charged its rounded-up size.
  constexpr std::size_t kDeviceGranule = std::size_t{2} << 20;
  const std::size_t kv_buffer = std::size_t{max_context} * kv_row * sizeof(__half);
  const std::size_t kv_rounded =
      (kv_buffer + kDeviceGranule - 1) / kDeviceGranule * kDeviceGranule;
  return std::size_t{c.num_layers} * 2 * kv_rounded +
         SessionScratchBytes(c, tile) + kMmqScratchArenaBytes +
         kSessionGraphBytes;
}

void Session::Reset() noexcept {
  impl_->position = 0;
}

std::size_t Session::AllocatedBytes() const noexcept {
  std::size_t total = 0;
  for (const auto& b : impl_->k_cache)
    total += b.bytes();
  for (const auto& b : impl_->v_cache)
    total += b.bytes();
  for (const auto& b : impl_->kmm)
    total += b.bytes();
  total += impl_->hidden_ring.bytes();
  return total;
}

std::size_t Session::SnapshotBytes() const noexcept {
  const Config& c = impl_->model->config;
  const std::size_t kv_row = std::size_t{c.num_kv} * c.head_dim;
  return kSnapshotHeaderBytes +
         std::size_t{c.num_layers} * 2 * impl_->position * kv_row *
             sizeof(__half) +
         std::size_t{c.vocab_size} * sizeof(float);
}

bool Session::SaveSnapshot(std::span<std::uint8_t> out,
                           std::string* error_msg) const {
  const Config& c = impl_->model->config;
  const std::size_t kv_row = std::size_t{c.num_kv} * c.head_dim;
  const std::size_t layer_bytes =
      std::size_t{impl_->position} * kv_row * sizeof(__half);
  if (out.size() != SnapshotBytes())
    return Fail(error_msg, "Flint: snapshot buffer has the wrong size");
  if (hipStreamSynchronize(impl_->stream) != hipSuccess)
    return Fail(error_msg, "Flint: device synchronization failed");
  const std::uint32_t header[8] = {kSnapshotMagic,
                                   Session::kSnapshotPayloadVersion,
                                   impl_->position,
                                   c.num_layers,
                                   static_cast<std::uint32_t>(kv_row),
                                   c.vocab_size,
                                   0,
                                   0};
  std::memcpy(out.data(), header, sizeof(header));
  std::uint8_t* at = out.data() + kSnapshotHeaderBytes;
  for (std::uint32_t il = 0; il < c.num_layers; ++il) {
    for (const auto* cache : {&impl_->k_cache[il], &impl_->v_cache[il]}) {
      if (layer_bytes != 0 && hipMemcpy(at, cache->get(), layer_bytes,
                                        hipMemcpyDeviceToHost) != hipSuccess) {
        return Fail(error_msg, "Flint: snapshot copy failed");
      }
      at += layer_bytes;
    }
  }
  std::memcpy(at, impl_->h_logits, std::size_t{c.vocab_size} * sizeof(float));
  return true;
}

bool Session::RestoreSnapshot(std::span<const std::uint8_t> payload,
                              std::string* error_msg) {
  const Config& c = impl_->model->config;
  const std::size_t kv_row = std::size_t{c.num_kv} * c.head_dim;
  std::uint32_t header[8];
  if (payload.size() < kSnapshotHeaderBytes)
    return Fail(error_msg, "Flint: snapshot payload is too short");
  std::memcpy(header, payload.data(), sizeof(header));
  const std::uint32_t position = header[2];
  if (header[0] != kSnapshotMagic ||
      header[1] != Session::kSnapshotPayloadVersion ||
      header[3] != c.num_layers || header[4] != kv_row ||
      header[5] != c.vocab_size || position == 0 ||
      position > impl_->capacity) {
    return Fail(error_msg, "Flint: snapshot does not match this session");
  }
  const std::size_t layer_bytes =
      std::size_t{position} * kv_row * sizeof(__half);
  const std::size_t expected = kSnapshotHeaderBytes +
                               std::size_t{c.num_layers} * 2 * layer_bytes +
                               std::size_t{c.vocab_size} * sizeof(float);
  if (payload.size() != expected)
    return Fail(error_msg, "Flint: snapshot payload has the wrong size");
  if (hipStreamSynchronize(impl_->stream) != hipSuccess)
    return Fail(error_msg, "Flint: device synchronization failed");
  const std::uint8_t* at = payload.data() + kSnapshotHeaderBytes;
  for (std::uint32_t il = 0; il < c.num_layers; ++il) {
    for (auto* cache : {&impl_->k_cache[il], &impl_->v_cache[il]}) {
      if (hipMemcpy(cache->get(), at, layer_bytes, hipMemcpyHostToDevice) !=
          hipSuccess) {
        return Fail(error_msg, "Flint: snapshot restore copy failed");
      }
      at += layer_bytes;
    }
  }
  // The payload offset is not guaranteed to be float-aligned: copy the bytes.
  std::memcpy(impl_->h_logits, at, std::size_t{c.vocab_size} * sizeof(float));
  impl_->logit_rows = 1;
  return RestoreRows(position, error_msg);
}

bool Session::Synchronize(std::string* error_msg) const {
  if (hipStreamSynchronize(impl_->stream) != hipSuccess)
    return Fail(error_msg, "Flint: device synchronization failed");
  return true;
}

std::uint32_t Session::NumLayers() const noexcept {
  return impl_->model->config.num_layers;
}

std::size_t Session::KvRowBytes() const noexcept {
  const Config& c = impl_->model->config;
  return std::size_t{c.num_kv} * c.head_dim * sizeof(__half);
}

const void* Session::KvRows(std::uint32_t layer, bool value) const noexcept {
  if (layer >= NumLayers())
    return nullptr;
  return (value ? impl_->v_cache : impl_->k_cache)[layer].get();
}

void* Session::KvRows(std::uint32_t layer, bool value) noexcept {
  if (layer >= NumLayers())
    return nullptr;
  return (value ? impl_->v_cache : impl_->k_cache)[layer].get();
}

bool Session::RestoreRows(std::uint32_t position, std::string* error_msg) {
  const Config& c = impl_->model->config;
  if (position > impl_->capacity)
    return Fail(error_msg, "Flint: restore position exceeds the capacity");
  if (!Synchronize(error_msg))
    return false;
  // State is assigned only after the device work has completed, so a failed
  // restore leaves position and logits untouched.
  if (position == 0) {
    if (impl_->sparse) {
      for (std::uint32_t il = 0; il < c.num_layers; ++il) {
        if (c.IsSwa(il))
          continue;
        if (hipMemsetAsync(impl_->kmm[il].get(), 0,
                           impl_->kmm[il].bytes(),
                           impl_->stream) != hipSuccess) {
          return Fail(error_msg, "Flint: key bound reset failed");
        }
      }
      if (!Synchronize(error_msg))
        return false;
    }
    // Logits are zeroed rather than left stale. logit_rows stays 1: Logits()
    // indexes logit_rows - 1, so 0 would underflow.
    std::memset(impl_->h_logits, 0, std::size_t{c.vocab_size} * sizeof(float));
    impl_->logit_rows = 1;
    impl_->position = 0;
    return true;
  }
  // The block-sparse key bounds are derived from the keys: rebuild them.
  if (impl_->sparse) {
    impl_->h_pos[0] = 0;
    if (hipMemcpy(impl_->pos.get(), impl_->h_pos, sizeof(std::uint32_t),
                  hipMemcpyHostToDevice) != hipSuccess) {
      return Fail(error_msg, "Flint: position upload failed");
    }
    for (std::uint32_t il = 0; il < c.num_layers; ++il) {
      if (c.IsSwa(il))
        continue;
      rocm::PoolKeyBlocks(impl_->k_cache[il].get(), impl_->kmm[il].get(),
                          impl_->pos.get(), 0, position, c.num_kv,
                          impl_->sparse_nblk, impl_->stream);
    }
    if (hipStreamSynchronize(impl_->stream) != hipSuccess)
      return Fail(error_msg, "Flint: key bound rebuild failed");
  }
  impl_->logit_rows = 1;
  impl_->position = position;
  return true;
}

bool Session::SetLogits(std::span<const float> row, std::string* error_msg) {
  const std::uint32_t vocab = impl_->model->config.vocab_size;
  if (row.size() != vocab)
    return Fail(error_msg, "Flint: logits row has the wrong size");
  std::memcpy(impl_->h_logits, row.data(), std::size_t{vocab} * sizeof(float));
  impl_->logit_rows = 1;
  return true;
}

Model::~Model() = default;

std::shared_ptr<Model> Model::Load(const std::string& model_path,
                                   const EngineOptions& options,
                                   std::string* error_msg) {
  std::shared_ptr<Model> model(new Model());
  auto impl = std::make_shared<ModelState>();
  impl->options = options;
  std::string err;
  impl->reader = core::GgufReader::OpenFile(model_path, &err);
  if (!impl->reader) {
    Fail(error_msg, "failed to open GGUF: " + err);
    return nullptr;
  }
  const auto config = ParseConfig(*impl->reader, &err);
  if (!config) {
    Fail(error_msg, "failed to parse Flint config: " + err);
    return nullptr;
  }
  impl->config = *config;
  impl->device = DeviceModel::Upload(*impl->reader, impl->config, &err);
  if (!impl->device) {
    Fail(error_msg, "failed to upload Flint weights: " + err);
    return nullptr;
  }
  // The weights are on the device: drop the GGUF mapping. Kept alive it pins
  // ~16 GB of mapped file pages that the kernel cannot reclaim cheaply, which
  // starves the GPU allocations of the next sessions on a unified-memory node.
  impl->reader.reset();
  model->state_ = std::move(impl);
  return model;
}

std::unique_ptr<Session> Model::CreateSession(std::uint32_t max_context,
                                              std::string* error_msg) {
  std::unique_ptr<Session> session(new Session());
  session->impl_->model = state_;
  if (!session->impl_->Init(max_context, error_msg))
    return nullptr;
  // First use of each path sizes the quantized kernels' scratch pools; do it
  // before the first request instead of inside its prefill. This warmup forward
  // is also what materializes this session's private MMQ scratch arena (256 MiB,
  // keyed by the session's own stream), which Model::SessionBytes accounts for.
  const std::int32_t warm[kWarmTokens] = {};
  for (const std::uint32_t width : {1U, kWarmTokens}) {
    if (width > max_context)
      break;
    if (!session->impl_->Forward({warm, width}, false, error_msg)) {
      return nullptr;
    }
    session->impl_->position = 0;
  }
  return session;
}

std::uint64_t Model::FreeDeviceBytes() noexcept {
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  if (hipMemGetInfo(&free_bytes, &total_bytes) != hipSuccess)
    return 0;
  return free_bytes;
}

const Config& Model::config() const noexcept {
  return state_->config;
}
std::size_t Model::ResidentBytes() const noexcept {
  return state_->device->resident_bytes();
}

}  // namespace gufo::models::flint::gpu
