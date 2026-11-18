// DFlash2 draft forward: one block of proposals for one session.
//
// This is the single-request path of the in-tree executor
// (src/models/qwen/hip/dflash/executor.cpp, ForwardBlock + InjectTargetContext),
// extracted so it does not depend on the Qwen target model: the drafter borrows
// only the target's token embedding and LM head, and its context comes from the
// engine's layer-residual capture ring.
//
// Op mapping. Where the in-tree code calls a Qwen-family helper, this file
// calls the extension's vendor of the same tier, so the arithmetic is the same
// code: quantized and BF16 projections use LaunchBatchedQuantGEMMFp32 and
// LaunchExactBf16GEMMFp32SmallBatch (both documented as matching decode
// accumulation order), RMSNorm uses the engine's RmsNormRows, and RoPE uses the
// vendored NEOX kernel. RoPE is the one place where the oracle's fused
// QK-norm+RoPE kernel is not the same code, so it is isolated behind
// ApplyQkNormRope() and is the first suspect if stage parity drifts.
#ifndef GUFO_MODELS_FLINT_GPU_DFLASH_FORWARD_HPP_
#define GUFO_MODELS_FLINT_GPU_DFLASH_FORWARD_HPP_

#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/gpu/dflash/dflash_weights.hpp"

namespace gufo::models::flint::gpu::dflash {

/// The two weights the drafter borrows from the target model. Both are
/// **device** pointers in their GGUF encoding, exactly as the engine holds
/// them; the drafter never copies or re-encodes them.
struct DflashBorrowed {
  const void* token_embedding{nullptr};
  core::GgmlType token_embedding_type{core::GgmlType::kF32};
  const void* output{nullptr};
  core::GgmlType output_type{core::GgmlType::kF32};
};

/// One draft block's proposals.
struct DflashDraft {
  /// `count` proposals; the anchor is not repeated.
  std::vector<std::uint32_t> tokens;
  /// When sampling: `count * selector_top_k` candidate ids and their
  /// probabilities, row-major per proposal.
  std::vector<std::uint32_t> candidate_ids;
  std::vector<float> candidate_probabilities;
  std::vector<float> confidences;
};

/// Optional stage dump: `name`, `count` floats. Used by the parity harness and
/// by FLINT_CHECK_DRAFTER; a null trace costs one branch per stage.
using DflashTrace =
    std::function<void(std::string_view name, const float* data,
                       std::size_t count)>;

/// How the drafter reads the head it borrows from the target.
enum class DflashHeadRoute {
  /// As stored, on the route the engine uses for that tensor. Flint's head is
  /// Q8_0, so this quantizes the activation the way every decode does -- which
  /// is what the drafter should approximate, because the verifier reads the
  /// same head the same way.
  kNative,
  /// Dequantized to F32 once and projected in F32. Only the parity harness
  /// asks for this: it compares the block against an FP32 reference, and a
  /// quantized head would put the logits stage at the head's own quantisation
  /// floor instead of the block's error.
  kFp32,
};

/// The per-session drafter state: the target-context K/V ring the block
/// attends to, and the position bookkeeping that goes with it. Each session
/// owns one; the weights and block workspace of the DflashForward that made it
/// are shared. Only the forward that created a context may use it.
class DflashContext final {
 public:
  ~DflashContext();
  DflashContext(const DflashContext&) = delete;
  DflashContext& operator=(const DflashContext&) = delete;

  /// Positions the drafter can attend to: rows appended by AppendContext.
  [[nodiscard]] std::uint32_t ContextLength() const noexcept;
  /// Rows the injected ring holds.
  [[nodiscard]] std::uint32_t HistoryCapacity() const noexcept;
  /// Device pointer to layer `layer`'s injected key/value ring, for tests.
  [[nodiscard]] const float* InjectedKey(std::uint32_t layer) const noexcept;
  [[nodiscard]] const float* InjectedValue(std::uint32_t layer) const noexcept;

  /// Restarts the injected context at `frontier`: the next AppendContext must
  /// start there. Rows below `frontier` were never injected (the target's
  /// history is not re-fed), so Run attends to no history until rows at or
  /// after the frontier fill the sliding window. ResetContext(0) is a fresh
  /// prompt. Ring rows are not cleared; Run never reads rows below the
  /// frontier.
  void ResetContext(std::uint32_t frontier = 0) noexcept;

 private:
  friend class DflashForward;
  DflashContext();
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

/// A drafter resident in device memory: read-only weights and the block
/// workspace, shared by every session. The per-session K/V ring lives in a
/// DflashContext from CreateContext(). Calls are not thread-safe; the host
/// serializes device work (requires_device_runtime_lock).
class DflashForward final {
 public:
  ~DflashForward();
  DflashForward(const DflashForward&) = delete;
  DflashForward& operator=(const DflashForward&) = delete;

  /// Uploads `weights`. `max_context` bounds each context ring: a ring holds
  /// at most `min(max_context, sliding_window)` rows (the drafter only ever
  /// reads the last `sliding_window`). `head_route` selects `kFp32` only for
  /// parity runs; it costs a full F32 copy of the target's head.
  [[nodiscard]] static std::unique_ptr<DflashForward> Create(
      const DflashWeights& weights, const DflashBorrowed& borrowed,
      std::uint32_t max_context, std::string* error_msg = nullptr,
      DflashHeadRoute head_route = DflashHeadRoute::kNative);

  const DflashConfig& config() const noexcept;

  /// A fresh, empty context for one session. Allocates its K/V ring now, so
  /// the cost is known when a session is bound. Returns nullptr on failure.
  [[nodiscard]] std::unique_ptr<DflashContext> CreateContext(
      std::string* error_msg = nullptr) const;

  /// Appends `count` target-context rows starting at `position` to `context`.
  /// `features` is `count * taps * hidden` floats, row-major by position and
  /// then by tap in the order of `config().target_layer_ids` -- exactly the
  /// layout the drafter's `fc.weight` expects.
  /// When `trace` is set it also emits `context.0.normalized` and, per layer,
  /// `context.0.<layer>.k` / `.v` for every appended row — the stages the
  /// parity harness compares. Tracing allocates buffers for the whole context,
  /// so it is a diagnostic path, not the shipped one.
  [[nodiscard]] bool AppendContext(DflashContext& context,
                                   std::span<const float> features,
                                   std::uint32_t position,
                                   std::uint32_t count,
                                   std::string* error_msg = nullptr,
                                   const DflashTrace& trace = {});

  /// Runs one block against `context`: the committed `anchor` at `position`
  /// plus `draft_count` proposals. `uniforms` supplies one draw per proposal
  /// when `temperature` is positive.
  [[nodiscard]] bool Run(const DflashContext& context, std::uint32_t anchor,
                         std::uint32_t position, std::uint32_t draft_count,
                         float temperature, std::span<const float> uniforms,
                         const DflashTrace& trace, DflashDraft* draft,
                         std::string* error_msg = nullptr);

  /// Rows the context-injection scratch is sized for. AppendContext allocates
  /// that scratch on first use, in chunks of at most this many rows.
  static constexpr std::uint32_t kContextChunkRows = 256;
  /// Device bytes of the context-injection scratch not yet allocated: the full
  /// scratch before the first AppendContext, zero after. The scratch is shared
  /// by all contexts, so this is resident cost, counted once.
  [[nodiscard]] std::size_t ContextScratchPendingBytes() const noexcept;
  /// Device bytes of one context's K/V ring (all layers, keys and values), as
  /// CreateContext allocates it.
  [[nodiscard]] std::size_t ContextRingBytes() const noexcept;
  /// Device bytes this drafter has allocated so far: shared weights and block
  /// workspace (contexts made by CreateContext are not included).
  [[nodiscard]] std::size_t AllocatedBytes() const noexcept;

 private:
  DflashForward();
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::models::flint::gpu::dflash

#endif  // GUFO_MODELS_FLINT_GPU_DFLASH_FORWARD_HPP_
