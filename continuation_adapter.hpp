#ifndef GUFO_MODELS_FLINT_CONTINUATION_ADAPTER_HPP_
#define GUFO_MODELS_FLINT_CONTINUATION_ADAPTER_HPP_

#include <cstdint>
#include <functional>
#include <gufo/ext/adapter.hpp>
#include <memory>
#include <mutex>
#include <span>
#include <vector>

#include "continuation_layout.hpp"
#include "src/gpu/dflash/dflash_forward.hpp"
#include "src/gpu/dflash/dflash_weights.hpp"
#include "src/gpu/engine.hpp"

namespace gufo::models::flint {

using FlintModel = gpu::Model;
using FlintSession = gpu::Session;

/// A DFlash-2 drafter artifact: the weights reference the mapped GGUF, so the
/// reader travels with them.
struct DraftArtifact {
  std::shared_ptr<const core::GgufReader> reader;
  gpu::dflash::DflashWeights weights;
};

// Prefix-continuation adapter for the Flint ROCm session (docs/CACHE-ROWS.md).
//
// The continuation is the K and V rows of every layer (append rows for a global
// layer, window rows for a sliding-window layer, 256 rows per chunk) plus the
// last logits as private state. The cache therefore holds checkpoints sized by
// what they keep, shares chunks between checkpoints, and can restore from any
// earlier checkpoint boundary. Row transfers copy between the session's device
// caches (a ring for window layers) and cache-owned host pages.
class FlintContinuationAdapter final : public cache::Adapter {
public:
  FlintContinuationAdapter(std::shared_ptr<FlintModel> model,
                           std::uint32_t context, cache::Identity identity,
                           std::shared_ptr<const DraftArtifact> draft = {});
  ~FlintContinuationAdapter() override;

  [[nodiscard]] FlintSession& GetSession(cache::Slot&);
  /// The slot's drafter, or null when speculation is off.
  [[nodiscard]] gpu::dflash::DflashForward* GetDraft(cache::Slot&);

  [[nodiscard]] std::vector<cache::BackingClass> BackingPlan(
      std::size_t ram_bytes) const override;
  [[nodiscard]] cache::Capabilities GetCapabilities() const override;
  [[nodiscard]] std::span<const cache::ComponentDescriptor> Components()
      const override;
  [[nodiscard]] cache::Identity CompatibilityIdentity() const override;
  [[nodiscard]] std::unique_ptr<cache::Slot> CreateSlot(
      cache::MutationGuard&) override;
  [[nodiscard]] std::vector<cache::ComponentPosition> Positions(
      const cache::Slot&) const override;
  [[nodiscard]] std::vector<cache::Rows> PlanPrefill(
      const cache::Slot&, cache::Rows first, cache::Rows end) const override;
  void BeginRestore(cache::Slot&,
                    std::span<const cache::ComponentPosition>) override;
  [[nodiscard]] cache::Completion CapturePrivate(const cache::Slot&,
                                                 cache::ComponentId,
                                                 std::span<std::byte>,
                                                 cache::Stream&) override;
  [[nodiscard]] cache::Completion CapturePrivatePiece(const cache::Slot&,
                                                      cache::ComponentId,
                                                      std::size_t,
                                                      std::span<std::byte>,
                                                      cache::Stream&) override;
  [[nodiscard]] cache::Completion LoadPrivate(cache::Slot&, cache::ComponentId,
                                              std::span<const std::byte>,
                                              cache::Stream&) override;
  [[nodiscard]] cache::Completion LoadPrivatePiece(cache::Slot&,
                                                   cache::ComponentId,
                                                   std::size_t,
                                                   std::span<const std::byte>,
                                                   cache::Stream&) override;
  [[nodiscard]] cache::Completion CopyRowsOut(
      const cache::Slot&, cache::ComponentId, cache::Rows first,
      cache::Rows end, std::span<std::byte>, cache::Stream&) override;
  [[nodiscard]] cache::Completion CopyRowsIn(cache::Slot&, cache::ComponentId,
                                             cache::Rows first, cache::Rows end,
                                             std::span<const std::byte>,
                                             cache::Stream&) override;
  [[nodiscard]] bool Validate(
      cache::Slot&, std::span<const cache::ComponentPosition>) override;
  [[nodiscard]] bool Invalidate(cache::Slot&) noexcept override;

private:
  class State;
  State& As(cache::Slot&) const;

  std::shared_ptr<FlintModel> model_;
  std::shared_ptr<const DraftArtifact> draft_;
  std::uint32_t context_;
  cache::Identity identity_;
  ContinuationLayout layout_;
};

}  // namespace gufo::models::flint

#endif  // GUFO_MODELS_FLINT_CONTINUATION_ADAPTER_HPP_
