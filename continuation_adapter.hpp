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
// The continuation is split into cache components (layout table in the .cpp):
// two private components (metadata, last logits row) and one kAppendRows K and
// one V component per layer, each token-major with row_bytes =
// num_kv * head_dim * 2. Capture copies only valid rows; restore copies rows in
// place through the session's KvRows() buffers and then commits the frontier in
// Validate.
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
  [[nodiscard]] std::span<const std::byte> BorrowRowsView(
      const cache::Slot&, cache::ComponentId, cache::Rows first,
      cache::Rows end) const override;
  [[nodiscard]] cache::Completion CopyRowsIn(cache::Slot&, cache::ComponentId,
                                             cache::Rows first, cache::Rows end,
                                             std::span<const std::byte>,
                                             cache::Stream&) override;
  [[nodiscard]] bool Validate(
      cache::Slot&, std::span<const cache::ComponentPosition>) override;
  [[nodiscard]] bool Invalidate(cache::Slot&) noexcept override;

private:
  enum class PrivatePart : std::uint8_t { kMetadata, kLogits };
  struct Component {
    cache::ComponentDescriptor descriptor;
    PrivatePart private_part{PrivatePart::kMetadata};  // kPrivateState only.
    std::uint32_t layer{0};                            // kAppendRows only.
    bool is_value{false};                              // kAppendRows only.
  };

  class State;
  State& As(cache::Slot&) const;
  const State& As(const cache::Slot&) const;
  const Component& RequireComponent(cache::ComponentId) const;

  std::shared_ptr<FlintModel> model_;
  std::uint32_t context_;
  cache::Identity identity_;
  std::size_t row_bytes_{0};
  std::size_t logits_bytes_{0};
  std::vector<Component> components_;
  std::vector<cache::ComponentDescriptor> descriptors_;
};

}  // namespace gufo::models::flint

#endif  // GUFO_MODELS_FLINT_CONTINUATION_ADAPTER_HPP_
