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

/// A drafter resident in device memory, with its own block workspace and the
/// ring that holds the target context it attends to.
class DflashForward final {
 public:
  ~DflashForward();
  DflashForward(const DflashForward&) = delete;
  DflashForward& operator=(const DflashForward&) = delete;

  /// Uploads `weights` and allocates the context ring for `max_context`
  /// positions (the drafter only ever reads the last `sliding_window`).
  [[nodiscard]] static std::unique_ptr<DflashForward> Create(
      const DflashWeights& weights, const DflashBorrowed& borrowed,
      std::uint32_t max_context, std::string* error_msg = nullptr);

  const DflashConfig& config() const noexcept;

  /// Appends `count` target-context rows starting at `position`. `features` is
  /// `count * taps * hidden` floats, row-major by position and then by tap in
  /// the order of `config().target_layer_ids` -- exactly the layout the
  /// drafter's `fc.weight` expects.
  [[nodiscard]] bool AppendContext(std::span<const float> features,
                                   std::uint32_t position,
                                   std::uint32_t count,
                                   std::string* error_msg = nullptr);

  /// Positions the drafter can attend to: rows appended by AppendContext.
  [[nodiscard]] std::uint32_t ContextLength() const noexcept;

  /// Runs one block: the committed `anchor` at `position` plus `draft_count`
  /// proposals. `uniforms` supplies one draw per proposal when `temperature`
  /// is positive.
  [[nodiscard]] bool Run(std::uint32_t anchor, std::uint32_t position,
                         std::uint32_t draft_count, float temperature,
                         std::span<const float> uniforms,
                         const DflashTrace& trace, DflashDraft* draft,
                         std::string* error_msg = nullptr);

  /// Device pointer to layer `layer`'s injected key/value ring, for tests.
  [[nodiscard]] const float* InjectedKey(std::uint32_t layer) const noexcept;
  [[nodiscard]] const float* InjectedValue(std::uint32_t layer) const noexcept;
  /// Rows the injected ring holds.
  [[nodiscard]] std::uint32_t HistoryCapacity() const noexcept;

 private:
  DflashForward();
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::models::flint::gpu::dflash

#endif  // GUFO_MODELS_FLINT_GPU_DFLASH_FORWARD_HPP_
