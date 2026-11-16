#include "continuation_adapter.hpp"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace gufo::models::flint {

class FlintContinuationAdapter::State final : public cache::Slot {
public:
  State(FlintContinuationAdapter& adapter, cache::MutationGuard& guard,
        std::unique_ptr<FlintSession> session)
      : adapter(adapter), guard(guard), session(std::move(session)) {}

  struct Job {
    bool load{false};
    mutable std::mutex mutex;
    std::optional<cache::TransferResult> result;
    std::optional<cache::Completion> completion;
  };

  class Signal final : public cache::CompletionSignal {
  public:
    explicit Signal(std::shared_ptr<Job> job) : job_(std::move(job)) {}
    [[nodiscard]] bool Ready() const noexcept override {
      const std::lock_guard lock(job_->mutex);
      return job_->result.has_value() ||
             (job_->completion && job_->completion->Ready());
    }
    [[nodiscard]] cache::TransferResult Wait() noexcept override {
      const std::lock_guard lock(job_->mutex);
      if (!job_->result) {
        job_->result = job_->completion->Wait();
        job_->completion.reset();
      }
      return *job_->result;
    }

  private:
    std::shared_ptr<Job> job_;
  };

  // A completed transfer that already moved its bytes.
  static cache::Completion Settled(bool success) {
    auto job = std::make_shared<Job>();
    job->result = success ? cache::TransferResult::kSucceeded
                          : cache::TransferResult::kFailed;
    return cache::Completion(std::make_unique<Signal>(std::move(job)));
  }

  [[nodiscard]] bool IsValid() const noexcept override {
    return !restoring && !failed;
  }

  // Row ranges received during a restore, per component (index = id - 1).
  void Cover(std::size_t component, cache::Rows first, cache::Rows end) {
    const std::lock_guard lock(mutex);
    coverage[component].emplace_back(first, end);
  }

  // True when the ranges received for `component` tile [first, end) exactly.
  [[nodiscard]] bool Covers(std::size_t component, cache::Rows first,
                            cache::Rows end) {
    const std::lock_guard lock(mutex);
    auto ranges = coverage[component];
    std::sort(ranges.begin(), ranges.end());
    cache::Rows next = first;
    for (const auto& [from, to] : ranges) {
      if (from != next)
        return false;
      next = to;
    }
    return next == end;
  }

  void ResetRestore() {
    const std::lock_guard lock(mutex);
    coverage.clear();
    logits.clear();
    logits_loaded.clear();
    expected_position = 0;
  }

  FlintContinuationAdapter& adapter;
  cache::MutationGuard& guard;
  std::unique_ptr<FlintSession> session;
  bool restoring{false};
  bool failed{false};
  std::mutex mutex;
  std::vector<std::vector<std::pair<cache::Rows, cache::Rows>>> coverage;
  // The restored logits, assembled from private pieces, and the byte ranges
  // received so far.
  std::vector<float> logits;
  std::vector<std::pair<std::size_t, std::size_t>> logits_loaded;
  cache::Rows expected_position{0};
};

FlintContinuationAdapter::FlintContinuationAdapter(
    std::shared_ptr<FlintModel> model, std::uint32_t context,
    cache::Identity identity)
    : model_(std::move(model)),
      context_(context),
      identity_(std::move(identity)),
      layout_(model_ ? model_->config() : Config{}, context) {
  if (!model_ || context_ == 0)
    throw std::invalid_argument("invalid Flint continuation configuration");
  if (identity_.empty())
    throw std::invalid_argument("Flint adapter requires an identity");
  const auto put = [&](std::uint32_t value) {
    for (unsigned i = 0; i < 4; ++i)
      identity_.push_back(static_cast<std::uint8_t>(value >> (8 * i)));
  };
  put(ContinuationLayout::kLayoutVersion);
  put(context_);
}

FlintContinuationAdapter::~FlintContinuationAdapter() = default;

FlintContinuationAdapter::State& FlintContinuationAdapter::As(
    cache::Slot& slot) const {
  auto& state = dynamic_cast<State&>(slot);
  if (state.adapter.model_.get() != model_.get())
    throw std::invalid_argument("slot belongs to another Flint adapter");
  return state;
}

FlintSession& FlintContinuationAdapter::GetSession(cache::Slot& slot) {
  auto& state = As(slot);
  if (!state.IsValid())
    throw std::logic_error("Flint slot is not valid");
  return *state.session;
}

std::vector<cache::BackingClass> FlintContinuationAdapter::BackingPlan(
    std::size_t ram_bytes) const {
  // Mirror the LFM3/hip backing shape: contiguous 64 KiB page groups with a
  // per-block metadata header, sized to nine tenths of the RAM budget after a
  // 4 MiB reserve. Rows and private state share the same committed pages.
  constexpr std::size_t page = 65536, per_block = 1024, header = 4096;
  const auto budget =
      ram_bytes > (4ULL << 20) ? (ram_bytes - (4ULL << 20)) / 10 * 9 : 0;
  if (budget <= header)
    return {};
  const auto blocks = (budget - header) / (page + per_block);
  if (!blocks)
    return {};
  return {{page, blocks, header + blocks * per_block}};
}

cache::Capabilities FlintContinuationAdapter::GetCapabilities() const {
  return {.continuation = true, .persistent_encoding = true};
}

std::span<const cache::ComponentDescriptor>
FlintContinuationAdapter::Components() const {
  return layout_.descriptors();
}

cache::Identity FlintContinuationAdapter::CompatibilityIdentity() const {
  return identity_;
}

std::unique_ptr<cache::Slot> FlintContinuationAdapter::CreateSlot(
    cache::MutationGuard& guard) {
  std::string error;
  auto session = model_->CreateSession(context_, &error);
  if (!session)
    throw std::runtime_error(error);
  return std::make_unique<State>(*this, guard, std::move(session));
}

std::vector<cache::ComponentPosition> FlintContinuationAdapter::Positions(
    const cache::Slot& slot) const {
  const auto& state = As(const_cast<cache::Slot&>(slot));
  if (!state.IsValid())
    throw std::logic_error("cannot query an invalid Flint slot");
  return layout_.Positions(state.session->Position());
}

std::vector<cache::Rows> FlintContinuationAdapter::PlanPrefill(
    const cache::Slot& slot, cache::Rows first, cache::Rows end) const {
  const auto& state = As(const_cast<cache::Slot&>(slot));
  if (!state.IsValid() || first > end || end > context_)
    throw std::invalid_argument("invalid Flint prefill plan");
  std::vector<cache::Rows> result;
  // EngineOptions::prefill_tile; a multiple of the 256-row chunk, so a pass
  // boundary is also a chunk boundary.
  constexpr cache::Rows capacity = 4096;
  while (first < end) {
    first += std::min(end - first, capacity);
    result.push_back(first);
  }
  return result;
}

void FlintContinuationAdapter::BeginRestore(
    cache::Slot& slot, std::span<const cache::ComponentPosition> positions) {
  auto& state = As(slot);
  if (!state.IsValid())
    throw std::logic_error("Flint restore requires a valid slot");
  const auto& components = layout_.components();
  if (positions.size() != components.size())
    throw std::invalid_argument("incomplete Flint restore positions");
  const cache::Rows position = positions.front().valid_rows;
  for (std::size_t i = 0; i < positions.size(); ++i) {
    // Every Flint component sits at the same frontier, in id order.
    if (positions[i].id.value != i + 1 || positions[i].valid_rows != position)
      throw std::invalid_argument("inconsistent Flint restore positions");
  }
  if (position == 0 || position > context_)
    throw std::invalid_argument("Flint frontier is out of range");
  // The rows about to be replaced must not be borrowed by a checkpoint (none
  // is lent today: BorrowRowsView is empty), and the guard may refuse.
  for (const auto& component : components) {
    if (!cache::HasRows(component.descriptor.kind))
      continue;
    state.guard.BeforeOverwrite(component.descriptor.id,
                                layout_.FirstRetainedRow(component, position),
                                position);
  }
  state.ResetRestore();
  state.coverage.resize(components.size());
  state.expected_position = position;
  state.failed = false;
  state.restoring = true;
}

cache::Completion FlintContinuationAdapter::CapturePrivate(
    const cache::Slot& slot, cache::ComponentId id, std::span<std::byte> bytes,
    cache::Stream& stream) {
  return CapturePrivatePiece(slot, id, 0, bytes, stream);
}

cache::Completion FlintContinuationAdapter::CapturePrivatePiece(
    const cache::Slot& slot, cache::ComponentId id, std::size_t offset,
    std::span<std::byte> bytes, cache::Stream& stream) {
  (void)stream;
  if (id.value != layout_.logits_id().value)
    throw std::invalid_argument("Flint private state is the logits only");
  auto& state = As(const_cast<cache::Slot&>(slot));
  if (!state.IsValid() || state.session->Position() == 0)
    throw std::logic_error("cannot capture an empty or invalid Flint slot");
  const auto total = layout_.at(id).descriptor.state_bytes;
  if (offset > total || bytes.size() > total - offset)
    throw std::invalid_argument("invalid Flint private piece");
  const auto logits = state.session->Logits();
  if (logits.size() * sizeof(float) != total)
    throw std::logic_error("Flint logits have an unexpected size");
  std::memcpy(bytes.data(),
              reinterpret_cast<const std::byte*>(logits.data()) + offset,
              bytes.size());
  return State::Settled(true);
}

cache::Completion FlintContinuationAdapter::LoadPrivate(
    cache::Slot& slot, cache::ComponentId id, std::span<const std::byte> bytes,
    cache::Stream& stream) {
  return LoadPrivatePiece(slot, id, 0, bytes, stream);
}

cache::Completion FlintContinuationAdapter::LoadPrivatePiece(
    cache::Slot& slot, cache::ComponentId id, std::size_t offset,
    std::span<const std::byte> bytes, cache::Stream& stream) {
  (void)stream;
  if (id.value != layout_.logits_id().value)
    throw std::invalid_argument("Flint private state is the logits only");
  auto& state = As(slot);
  if (!state.restoring || state.failed)
    throw std::logic_error("Flint load requires an unfailed restore");
  const auto total = layout_.at(id).descriptor.state_bytes;
  if (offset > total || bytes.size() > total - offset)
    throw std::invalid_argument("invalid Flint private piece");
  const std::lock_guard lock(state.mutex);
  if (state.logits.empty())
    state.logits.resize(total / sizeof(float));
  std::memcpy(reinterpret_cast<std::byte*>(state.logits.data()) + offset,
              bytes.data(), bytes.size());
  state.logits_loaded.emplace_back(offset, offset + bytes.size());
  return State::Settled(true);
}

cache::Completion FlintContinuationAdapter::CopyRowsOut(
    const cache::Slot& slot, cache::ComponentId id, cache::Rows first,
    cache::Rows end, std::span<std::byte> bytes, cache::Stream& stream) {
  (void)stream;
  auto& state = As(const_cast<cache::Slot&>(slot));
  const auto& component = layout_.at(id);
  if (!cache::HasRows(component.descriptor.kind))
    throw std::invalid_argument("Flint component has no rows");
  if (!state.IsValid())
    throw std::logic_error("cannot read an invalid Flint slot");
  const auto row = layout_.row_bytes();
  if (first > end || end > state.session->Position() ||
      bytes.size() != (end - first) * row)
    throw std::invalid_argument("invalid Flint row capture");
  std::string error;
  if (!state.session->TransferRows(
          component.layer, component.part == ContinuationLayout::Part::kValue,
          static_cast<std::uint32_t>(first), static_cast<std::uint32_t>(end),
          {reinterpret_cast<std::uint8_t*>(bytes.data()), bytes.size()}, true,
          &error))
    throw std::runtime_error("Flint row capture failed: " + error);
  return State::Settled(true);
}

cache::Completion FlintContinuationAdapter::CopyRowsIn(
    cache::Slot& slot, cache::ComponentId id, cache::Rows first,
    cache::Rows end, std::span<const std::byte> bytes, cache::Stream& stream) {
  (void)stream;
  auto& state = As(slot);
  if (!state.restoring || state.failed)
    throw std::logic_error("Flint load requires an unfailed restore");
  try {
    const auto& component = layout_.at(id);
    if (!cache::HasRows(component.descriptor.kind))
      throw std::invalid_argument("Flint component has no rows");
    const auto row = layout_.row_bytes();
    const auto lowest =
        layout_.FirstRetainedRow(component, state.expected_position);
    if (first > end || first < lowest || end > state.expected_position ||
        bytes.size() != (end - first) * row)
      throw std::invalid_argument("invalid Flint row restore");
    std::string error;
    if (!state.session->TransferRows(
            component.layer, component.part == ContinuationLayout::Part::kValue,
            static_cast<std::uint32_t>(first), static_cast<std::uint32_t>(end),
            {const_cast<std::uint8_t*>(
                 reinterpret_cast<const std::uint8_t*>(bytes.data())),
             bytes.size()},
            false, &error)) {
      state.failed = true;
      return State::Settled(false);
    }
    state.Cover(id.value - 1, first, end);
    return State::Settled(true);
  } catch (...) {
    state.failed = true;
    throw;
  }
}

bool FlintContinuationAdapter::Validate(
    cache::Slot& slot, std::span<const cache::ComponentPosition> positions) {
  auto& state = As(slot);
  if (!state.restoring)
    throw std::logic_error("Flint validation requires an active restore");
  const auto refuse = [&] {
    state.failed = true;
    return false;
  };
  const auto& components = layout_.components();
  if (state.failed || positions.size() != components.size())
    return refuse();
  for (std::size_t i = 0; i < positions.size(); ++i) {
    if (positions[i].id.value != i + 1 ||
        positions[i].valid_rows != state.expected_position)
      return refuse();
  }
  const auto position = state.expected_position;
  for (std::size_t i = 0; i < components.size(); ++i) {
    if (!cache::HasRows(components[i].descriptor.kind))
      continue;
    if (!state.Covers(i, layout_.FirstRetainedRow(components[i], position),
                      position))
      return refuse();
  }
  // The logits must have arrived whole.
  std::vector<float> logits;
  {
    const std::lock_guard lock(state.mutex);
    auto ranges = state.logits_loaded;
    std::sort(ranges.begin(), ranges.end());
    std::size_t next = 0;
    for (const auto& [from, to] : ranges) {
      if (from != next)
        return refuse();
      next = to;
    }
    const auto total = layout_.at(layout_.logits_id()).descriptor.state_bytes;
    if (next != total || state.logits.size() * sizeof(float) != total)
      return refuse();
    logits = state.logits;
  }
  std::string error;
  if (!state.session->SetState(static_cast<std::uint32_t>(position), logits,
                               &error))
    return refuse();
  if (state.session->Position() != position)
    return refuse();
  state.restoring = false;
  state.ResetRestore();
  return true;
}

bool FlintContinuationAdapter::Invalidate(cache::Slot& slot) noexcept {
  try {
    auto& state = As(slot);
    state.session->Reset();
    state.failed = false;
    state.restoring = false;
    state.ResetRestore();
    state.guard.AfterReset();
    return true;
  } catch (...) {
    return false;
  }
}

}  // namespace gufo::models::flint
