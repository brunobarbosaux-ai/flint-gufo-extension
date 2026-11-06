#include "continuation_adapter.hpp"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>

namespace gufo::models::flint {
namespace {

// The private payload is one metadata header followed by the raw session
// snapshot, padded to a fixed capacity. A Flint snapshot's byte count tracks
// the live KV position, so the cache's fixed-size component holds the largest
// possible snapshot and the header records how many bytes are live.
constexpr std::uint32_t kPayloadVersion = 1;
struct PayloadHeader {
  std::uint32_t version{kPayloadVersion};
  std::uint32_t reserved{0};
  std::uint64_t position{0};
  std::uint64_t snapshot_len{0};
};
static_assert(sizeof(PayloadHeader) == 24);

}  // namespace

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

  FlintContinuationAdapter& adapter;
  cache::MutationGuard& guard;
  std::unique_ptr<FlintSession> session;
  bool restoring{false};
  bool failed{false};
  std::vector<std::byte> staging;
  std::uint64_t expected_position{0};
};

FlintContinuationAdapter::FlintContinuationAdapter(
    std::shared_ptr<FlintModel> model, std::uint32_t context,
    cache::Identity identity)
    : model_(std::move(model)),
      context_(context),
      identity_(std::move(identity)) {
  if (!model_ || context_ == 0)
    throw std::invalid_argument("invalid Flint continuation configuration");
  // Flint snapshot bytes track the live KV position; the model-sized value is
  // the fixed capacity the cache allocates, and each payload records its own
  // live length.
  snapshot_bytes_ = model_->SnapshotBytes(context_);
  if (snapshot_bytes_ == 0)
    throw std::invalid_argument("Flint snapshot has no bytes");
  cache::ComponentDescriptor descriptor;
  descriptor.id = {1};
  descriptor.layout_version = 1;
  descriptor.kind = cache::ComponentKind::kPrivateState;
  descriptor.state_bytes = sizeof(PayloadHeader) + snapshot_bytes_;
  descriptors_.push_back(descriptor);

  if (identity_.empty())
    throw std::invalid_argument("Flint adapter requires an identity");
  const auto put = [&](std::uint32_t value) {
    for (unsigned i = 0; i < 4; ++i)
      identity_.push_back(static_cast<std::uint8_t>(value >> (8 * i)));
  };
  put(1);  // Component layout version.
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

void FlintContinuationAdapter::RequireComponent(cache::ComponentId id) const {
  if (id.value != 1)
    throw std::invalid_argument("invalid Flint component id");
}

std::vector<cache::BackingClass> FlintContinuationAdapter::BackingPlan(
    std::size_t ram_bytes) const {
  // Mirror the LFM3/hip backing shape: contiguous 64 KiB page groups with a
  // per-block metadata header, sized to nine tenths of the RAM budget after a
  // 4 MiB reserve. A private-only adapter still needs a plan or the HIP
  // resource provider rejects it.
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
  return descriptors_;
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
  return {{descriptors_.front().id, state.session->Position()}};
}

std::vector<cache::Rows> FlintContinuationAdapter::PlanPrefill(
    const cache::Slot& slot, cache::Rows first, cache::Rows end) const {
  const auto& state = As(const_cast<cache::Slot&>(slot));
  if (!state.IsValid() || first > end || end > context_)
    throw std::invalid_argument("invalid Flint prefill plan");
  std::vector<cache::Rows> result;
  constexpr cache::Rows capacity = 4096;  // EngineOptions::prefill_tile.
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
  if (positions.size() != 1 || positions.front().id.value != 1)
    throw std::invalid_argument("incomplete Flint restore positions");
  if (positions.front().valid_rows > context_)
    throw std::invalid_argument("Flint frontier exceeds context");
  state.expected_position = positions.front().valid_rows;
  state.failed = false;
  state.restoring = true;
  state.staging.clear();
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
  RequireComponent(id);
  auto& state = As(const_cast<cache::Slot&>(slot));
  if (!state.IsValid())
    throw std::logic_error("cannot capture an invalid Flint slot");
  const auto total = descriptors_.front().state_bytes;
  if (offset > total || bytes.size() > total - offset)
    throw std::invalid_argument("invalid Flint private piece");
  // Save into a temporary sized for the live snapshot, then place header and
  // snapshot directly into the destination span. Bytes of the payload beyond
  // the live snapshot are zeroed so a payload is deterministic without
  // materialising a second full-capacity buffer.
  const auto live_bytes = state.session->SnapshotBytes();
  if (live_bytes > snapshot_bytes_)
    throw std::logic_error("Flint live snapshot exceeds its capacity");
  std::vector<std::uint8_t> live(live_bytes);
  std::string error;
  if (!state.session->SaveSnapshot(live, &error))
    throw std::runtime_error("Flint snapshot capture failed: " + error);
  const PayloadHeader header{.version = kPayloadVersion,
                             .reserved = 0,
                             .position = state.session->Position(),
                             .snapshot_len = live_bytes};
  const auto copy = [&](std::size_t from, const void* source,
                        std::size_t count) {
    const auto to = from + count;
    if (to <= offset || from >= offset + bytes.size())
      return;
    const auto begin = std::max(from, offset);
    const auto end = std::min(to, offset + bytes.size());
    std::memcpy(bytes.data() + (begin - offset),
                static_cast<const std::uint8_t*>(source) + (begin - from),
                end - begin);
  };
  // Zero the region the live snapshot does not cover.
  const auto covered = sizeof(PayloadHeader) + live_bytes;
  const auto zero_begin = std::max(offset, covered);
  const auto zero_end = std::min(offset + bytes.size(), total);
  if (zero_begin < zero_end)
    std::memset(bytes.data() + (zero_begin - offset), 0, zero_end - zero_begin);
  copy(0, &header, sizeof(header));
  copy(sizeof(PayloadHeader), live.data(), live_bytes);
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
  RequireComponent(id);
  auto& state = As(slot);
  if (!state.restoring || state.failed)
    throw std::logic_error("Flint load requires an unfailed restore");
  const auto total = descriptors_.front().state_bytes;
  if (offset > total || bytes.size() > total - offset)
    throw std::invalid_argument("invalid Flint private piece");
  if (state.staging.size() != total)
    state.staging.resize(total);
  std::memcpy(state.staging.data() + offset, bytes.data(), bytes.size());
  return State::Settled(true);
}

cache::Completion FlintContinuationAdapter::CopyRowsOut(
    const cache::Slot&, cache::ComponentId, cache::Rows, cache::Rows,
    std::span<std::byte>, cache::Stream&) {
  throw std::logic_error("Flint has no append-row components");
}

cache::Completion FlintContinuationAdapter::CopyRowsIn(
    cache::Slot&, cache::ComponentId, cache::Rows, cache::Rows,
    std::span<const std::byte>, cache::Stream&) {
  throw std::logic_error("Flint has no append-row components");
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
  if (state.failed || positions.size() != 1 ||
      positions.front().id.value != 1 ||
      positions.front().valid_rows != state.expected_position)
    return refuse();
  const auto total = descriptors_.front().state_bytes;
  if (state.staging.size() != total)
    return refuse();
  PayloadHeader header{};
  std::memcpy(&header, state.staging.data(), sizeof(header));
  if (header.version != kPayloadVersion ||
      header.position != state.expected_position || header.snapshot_len == 0 ||
      header.snapshot_len > snapshot_bytes_)
    return refuse();
  const auto snapshot = std::span<const std::uint8_t>(
      reinterpret_cast<const std::uint8_t*>(state.staging.data() +
                                            sizeof(PayloadHeader)),
      header.snapshot_len);
  std::string error;
  if (!state.session->RestoreSnapshot(snapshot, &error))
    return refuse();
  if (state.session->Position() != state.expected_position)
    return refuse();
  state.restoring = false;
  state.staging.clear();
  return true;
}

bool FlintContinuationAdapter::Invalidate(cache::Slot& slot) noexcept {
  try {
    auto& state = As(slot);
    state.session->Reset();
    state.failed = false;
    state.restoring = false;
    state.staging.clear();
    state.guard.AfterReset();
    return true;
  } catch (...) {
    return false;
  }
}

}  // namespace gufo::models::flint
