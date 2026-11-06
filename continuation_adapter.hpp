#ifndef GUFO_MODELS_FLINT_CONTINUATION_ADAPTER_HPP_
#define GUFO_MODELS_FLINT_CONTINUATION_ADAPTER_HPP_

#include <cstdint>
#include <functional>
#include <gufo/ext/adapter.hpp>
#include <memory>
#include <mutex>
#include <span>
#include <vector>

#include "src/gpu/engine.hpp"

namespace gufo::models::flint {

using FlintModel = gpu::Model;
using FlintSession = gpu::Session;

// Prefix-continuation adapter for the Flint ROCm session.
//
// Flint's continuation state is a single opaque device snapshot: the KV rows of
// every layer plus the last logits, saved and restored as one blob
// (Session::SaveSnapshot / RestoreSnapshot). Unlike per-layer KV models there
// are no independently addressable append rows here, so the whole continuation
// is modelled as one private-state component. A Flint snapshot's byte count
// tracks the live KV position, so the component is sized to the model's maximum
// possible snapshot and each payload records its own live length in a header.
// Prefix caching then works by copying that blob between the slot's session and
// cache-owned host pages.
class FlintContinuationAdapter final : public cache::Adapter {
public:
  FlintContinuationAdapter(std::shared_ptr<FlintModel> model,
                           std::uint32_t context, cache::Identity identity);
  ~FlintContinuationAdapter() override;

  [[nodiscard]] FlintSession& GetSession(cache::Slot&);

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
  void RequireComponent(cache::ComponentId) const;

  std::shared_ptr<FlintModel> model_;
  std::uint32_t context_;
  cache::Identity identity_;
  std::size_t snapshot_bytes_{0};
  std::vector<cache::ComponentDescriptor> descriptors_;
};

}  // namespace gufo::models::flint

#endif  // GUFO_MODELS_FLINT_CONTINUATION_ADAPTER_HPP_
