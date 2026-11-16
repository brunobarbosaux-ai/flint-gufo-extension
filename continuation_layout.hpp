#ifndef GUFO_MODELS_FLINT_CONTINUATION_LAYOUT_HPP_
#define GUFO_MODELS_FLINT_CONTINUATION_LAYOUT_HPP_

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "src/cache/types.hpp"
#include "src/runtime/config.hpp"

namespace gufo::models::flint {

// Geometry of the Flint continuation cache (see docs/CACHE-ROWS.md): the K and
// V rows of every layer as row components (append rows for a global layer,
// window rows for a sliding-window layer) plus the last logits as private
// state. Pure host arithmetic: no device, no session.
class ContinuationLayout {
public:
  enum class Part : std::uint8_t { kKey, kValue, kLogits };

  struct Component {
    cache::ComponentDescriptor descriptor;
    Part part{Part::kKey};
    std::uint32_t layer{0};
  };

  // Chunk size shared by every row component. gufo needs a window ring of at
  // least window_rows + kRowsPerChunk - 1 rows; Flint's ring holds
  // window + prefill_tile rows, far above.
  static constexpr cache::Rows kRowsPerChunk = 256;
  // Layout of the descriptors; part of the cache identity.
  static constexpr std::uint32_t kLayoutVersion = 2;

  ContinuationLayout(const Config& config, std::uint32_t context)
      : context_(context) {
    if (!context || !config.num_layers || !config.num_kv || !config.head_dim ||
        !config.vocab_size)
      throw std::invalid_argument("invalid Flint continuation geometry");
    row_bytes_ = std::size_t{config.num_kv} * config.head_dim * 2;  // f16
    const auto add = [&](Part part, std::uint32_t layer,
                         cache::ComponentKind kind, std::size_t row,
                         std::size_t state, cache::Rows window) {
      const auto id = static_cast<std::uint32_t>(components_.size() + 1);
      components_.push_back(
          {{{id}, kLayoutVersion, kind, row,
            kind == cache::ComponentKind::kPrivateState ? 0 : kRowsPerChunk,
            state, window},
           part,
           layer});
    };
    for (std::uint32_t layer = 0; layer < config.num_layers; ++layer) {
      const bool window = config.IsSwa(layer) && config.sliding_window != 0;
      const auto kind = window ? cache::ComponentKind::kWindowRows
                               : cache::ComponentKind::kAppendRows;
      const cache::Rows rows = window ? config.sliding_window : 0;
      add(Part::kKey, layer, kind, row_bytes_, 0, rows);
      add(Part::kValue, layer, kind, row_bytes_, 0, rows);
    }
    logits_id_ = static_cast<std::uint32_t>(components_.size() + 1);
    add(Part::kLogits, 0, cache::ComponentKind::kPrivateState, 0,
        std::size_t{config.vocab_size} * sizeof(float), 0);
    for (const auto& c : components_)
      descriptors_.push_back(c.descriptor);
  }

  [[nodiscard]] std::uint32_t context() const noexcept { return context_; }
  [[nodiscard]] std::size_t row_bytes() const noexcept { return row_bytes_; }
  [[nodiscard]] const std::vector<Component>& components() const noexcept {
    return components_;
  }
  [[nodiscard]] const std::vector<cache::ComponentDescriptor>& descriptors()
      const noexcept {
    return descriptors_;
  }
  [[nodiscard]] const Component& at(cache::ComponentId id) const {
    if (id.value == 0 || id.value > components_.size())
      throw std::invalid_argument("invalid Flint component id");
    return components_[id.value - 1];
  }
  [[nodiscard]] cache::ComponentId logits_id() const noexcept {
    return {logits_id_};
  }

  // Every component, row or private, sits at the same frontier: the session
  // position. (A window component keeps fewer rows, but its position is still
  // the absolute row number.)
  [[nodiscard]] std::vector<cache::ComponentPosition> Positions(
      cache::Rows position) const {
    std::vector<cache::ComponentPosition> positions;
    positions.reserve(components_.size());
    for (const auto& c : components_)
      positions.push_back({c.descriptor.id, position});
    return positions;
  }

  // First row a restore to `position` must supply for `component`: row 0 for an
  // append component, the first chunk that holds `position - window_rows` for
  // a window component.
  [[nodiscard]] cache::Rows FirstRetainedRow(const Component& component,
                                             cache::Rows position) const {
    return cache::FirstChunkRow(component.descriptor, position);
  }

private:
  std::uint32_t context_;
  std::size_t row_bytes_{0};
  std::uint32_t logits_id_{0};
  std::vector<Component> components_;
  std::vector<cache::ComponentDescriptor> descriptors_;
};

}  // namespace gufo::models::flint

#endif  // GUFO_MODELS_FLINT_CONTINUATION_LAYOUT_HPP_
