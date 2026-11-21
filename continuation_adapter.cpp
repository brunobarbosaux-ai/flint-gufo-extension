#include "continuation_adapter.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

// Component layout (ids ascend; layout_version = 1 for every component):
//
// | id      | component | kind          | row_bytes        | rows_per_chunk | state_bytes  |
// |---------|-----------|---------------|------------------|----------------|--------------|
// | 1       | metadata  | kPrivateState | 0                | 0              | 64           |
// | 2       | logits    | kPrivateState | 0                | 0              | vocab_size*4 |
// | 3 + 2*l | K(l)      | kAppendRows   | num_kv*head_dim*2| 256            | 0            |
// | 4 + 2*l | V(l)      | kAppendRows   | num_kv*head_dim*2| 256            | 0            |
//
// Rows are token-major, [position][num_kv * head_dim] of __half, exactly the
// engine's per-layer K/V buffer layout. The component table is parallel to
// descriptors_: index i holds id i + 1.

namespace gufo::models::flint {
namespace {

using cache::Rows;
using TransferResult = cache::TransferResult;

constexpr std::size_t kMetadataBytes = 64;
constexpr Rows kRowsPerChunk = 256;

void Check(hipError_t result) {
  if (result != hipSuccess)
    throw std::runtime_error(hipGetErrorString(result));
}

void SyncSession(const FlintSession& session) {
  std::string error;
  if (!session.Synchronize(&error))
    throw std::runtime_error("Flint synchronize failed: " + error);
}

using Coverage = std::vector<std::pair<std::size_t, std::size_t>>;
bool Covered(const Coverage& ranges, std::size_t end) {
  Coverage sorted = ranges;
  std::ranges::sort(sorted);
  std::size_t frontier = 0;
  for (const auto& [first, last] : sorted) {
    if (first > frontier)
      return false;
    frontier = std::max(frontier, last);
  }
  return frontier == end;
}

// Host view of the 64-byte metadata component. Fields past `reserved` are zero.
struct Metadata {
  std::uint32_t version{1};
  std::uint32_t position{0};
  std::uint32_t logits_valid{0};
  std::uint32_t reserved{0};
  std::array<std::uint8_t, kMetadataBytes - 16> tail{};
};
static_assert(sizeof(Metadata) == kMetadataBytes);

}  // namespace

class FlintContinuationAdapter::State final : public cache::Slot {
public:
  State(FlintContinuationAdapter& adapter, cache::MutationGuard& guard,
        std::unique_ptr<FlintSession> session)
      : adapter(adapter), guard(guard), session(std::move(session)) {}

  struct Job {
    mutable std::mutex mutex;
    std::optional<cache::TransferResult> result;
  };

  class Signal final : public cache::CompletionSignal {
  public:
    explicit Signal(std::shared_ptr<Job> job) : job_(std::move(job)) {}
    [[nodiscard]] bool Ready() const noexcept override {
      const std::lock_guard lock(job_->mutex);
      return job_->result.has_value();
    }
    [[nodiscard]] cache::TransferResult Wait() noexcept override {
      const std::lock_guard lock(job_->mutex);
      return *job_->result;
    }

  private:
    std::shared_ptr<Job> job_;
  };

  // Every transfer here completes synchronously on the calling thread.
  static cache::Completion Settled(bool success) {
    auto job = std::make_shared<Job>();
    job->result = success ? TransferResult::kSucceeded : TransferResult::kFailed;
    return cache::Completion(std::make_unique<Signal>(std::move(job)));
  }

  ~State() override {
    // Destruction must not terminate: release what can be released and drop
    // the row hook so the session cannot call back into a dead guard.
    try {
      session->SetRowWriteHook({});
      for (const auto& component : adapter.components_)
        if (component.descriptor.kind == cache::ComponentKind::kAppendRows)
          guard.BeforeRelease(component.descriptor.id, 0, adapter.context_);
    } catch (...) {
    }
  }

  [[nodiscard]] bool IsValid() const noexcept override {
    return !restoring && !failed.load();
  }

  void Cover(std::size_t index, std::size_t first, std::size_t last) {
    const std::lock_guard lock(coverage_mutex);
    coverage[index].emplace_back(first, last);
  }

  FlintContinuationAdapter& adapter;
  cache::MutationGuard& guard;
  std::unique_ptr<FlintSession> session;
  bool restoring{false};
  // Written by concurrent load pieces, read by IsValid() on the request thread.
  std::atomic<bool> failed{false};
  cache::Rows expected_position{0};
  std::vector<cache::ComponentPosition> expected;
  std::mutex coverage_mutex;
  std::vector<Coverage> coverage;
  std::array<std::byte, kMetadataBytes> metadata{};
  std::vector<std::byte> logits;
};

FlintContinuationAdapter::FlintContinuationAdapter(
    std::shared_ptr<FlintModel> model, std::uint32_t context,
    cache::Identity identity)
    : model_(std::move(model)),
      context_(context),
      identity_(std::move(identity)) {
  if (!model_ || context_ == 0)
    throw std::invalid_argument("invalid Flint continuation configuration");
  if (identity_.empty())
    throw std::invalid_argument("Flint adapter requires an identity");

  const auto& config = model_->config();
  if (config.num_layers == 0 || config.num_kv == 0 || config.head_dim == 0 ||
      config.vocab_size == 0)
    throw std::invalid_argument("Flint model has no continuation geometry");
  row_bytes_ = std::size_t{config.num_kv} * config.head_dim * 2;
  logits_bytes_ = std::size_t{config.vocab_size} * sizeof(float);

  Component metadata;
  metadata.descriptor = {.id = {1},
                         .layout_version = 1,
                         .kind = cache::ComponentKind::kPrivateState,
                         .row_bytes = 0,
                         .rows_per_chunk = 0,
                         .state_bytes = kMetadataBytes};
  metadata.private_part = PrivatePart::kMetadata;
  components_.push_back(metadata);

  Component logits;
  logits.descriptor = {.id = {2},
                       .layout_version = 1,
                       .kind = cache::ComponentKind::kPrivateState,
                       .row_bytes = 0,
                       .rows_per_chunk = 0,
                       .state_bytes = logits_bytes_};
  logits.private_part = PrivatePart::kLogits;
  components_.push_back(logits);

  for (std::uint32_t layer = 0; layer < config.num_layers; ++layer) {
    for (const bool is_value : {false, true}) {
      Component rows;
      rows.descriptor = {
          .id = {3 + 2 * layer + (is_value ? 1U : 0U)},
          .layout_version = 1,
          .kind = cache::ComponentKind::kAppendRows,
          .row_bytes = row_bytes_,
          .rows_per_chunk = kRowsPerChunk,
          .state_bytes = 0};
      rows.layer = layer;
      rows.is_value = is_value;
      components_.push_back(rows);
    }
  }
  for (const auto& component : components_)
    descriptors_.push_back(component.descriptor);

  const auto put = [&](std::uint32_t value) {
    for (unsigned i = 0; i < 4; ++i)
      identity_.push_back(static_cast<std::uint8_t>(value >> (8 * i)));
  };
  // Identity version 2 is the compatibility epoch for this layout. It is a
  // separate versioning from the descriptor layout_version = 1 (per-component
  // format), and bumps whenever the component table changes.
  put(2);
  put(context_);
}

FlintContinuationAdapter::~FlintContinuationAdapter() = default;

FlintContinuationAdapter::State& FlintContinuationAdapter::As(
    cache::Slot& slot) const {
  auto& state = dynamic_cast<State&>(slot);
  if (&state.adapter != this)
    throw std::invalid_argument("slot belongs to another Flint adapter");
  return state;
}

const FlintContinuationAdapter::State& FlintContinuationAdapter::As(
    const cache::Slot& slot) const {
  return As(const_cast<cache::Slot&>(slot));
}

FlintSession& FlintContinuationAdapter::GetSession(cache::Slot& slot) {
  auto& state = As(slot);
  if (!state.IsValid())
    throw std::logic_error("Flint slot is not valid");
  return *state.session;
}

const FlintContinuationAdapter::Component&
FlintContinuationAdapter::RequireComponent(cache::ComponentId id) const {
  if (id.value == 0 || id.value > components_.size())
    throw std::invalid_argument("invalid Flint component id");
  return components_[id.value - 1];
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
  if (session->KvRowBytes() != row_bytes_)
    throw std::runtime_error("Flint session row size differs from the layout");
  std::size_t layers = 0;
  for (const auto& component : components_)
    if (component.descriptor.kind == cache::ComponentKind::kAppendRows)
      ++layers;
  if (session->NumLayers() * std::size_t{2} != layers)
    throw std::runtime_error("Flint session layer count differs from the layout");
  if (session->Capacity() != context_)
    throw std::runtime_error("Flint session capacity differs from the context");

  auto state = std::make_unique<State>(*this, guard, std::move(session));
  State* owner = state.get();
  owner->session->SetRowWriteHook(
      [owner](std::uint32_t first, std::uint32_t end) {
        if (first == end)
          return;
        for (const auto& component : owner->adapter.components_)
          if (component.descriptor.kind == cache::ComponentKind::kAppendRows)
            owner->guard.BeforeOverwrite(component.descriptor.id, first, end);
      });
  return state;
}

std::vector<cache::ComponentPosition> FlintContinuationAdapter::Positions(
    const cache::Slot& slot) const {
  const auto& state = As(slot);
  if (!state.IsValid())
    throw std::logic_error("cannot query an invalid Flint slot");
  std::vector<cache::ComponentPosition> positions;
  positions.reserve(descriptors_.size());
  for (const auto& descriptor : descriptors_)
    positions.push_back({descriptor.id, state.session->Position()});
  return positions;
}

std::vector<cache::Rows> FlintContinuationAdapter::PlanPrefill(
    const cache::Slot& slot, cache::Rows first, cache::Rows end) const {
  const auto& state = As(slot);
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
  if (positions.size() != descriptors_.size())
    throw std::invalid_argument("incomplete Flint restore positions");

  std::vector<cache::ComponentPosition> ordered;
  ordered.reserve(descriptors_.size());
  for (const auto& descriptor : descriptors_) {
    if (std::ranges::count(positions, descriptor.id,
                           &cache::ComponentPosition::id) != 1)
      throw std::invalid_argument("duplicate or missing Flint frontier");
    ordered.push_back(*std::ranges::find(positions, descriptor.id,
                                         &cache::ComponentPosition::id));
  }
  const auto frontier = ordered.front().valid_rows;
  for (const auto& position : ordered)
    if (position.valid_rows != frontier)
      throw std::invalid_argument("inconsistent Flint frontiers");
  if (frontier > state.session->Capacity())
    throw std::invalid_argument("Flint frontier exceeds session capacity");

  // Rows are written in place: guard every append range the restore can
  // replace while the old slot is still readable, then drain the device.
  for (const auto& component : components_)
    if (component.descriptor.kind == cache::ComponentKind::kAppendRows)
      state.guard.BeforeOverwrite(component.descriptor.id, 0, context_);
  SyncSession(*state.session);

  state.expected = std::move(ordered);
  state.expected_position = frontier;
  state.coverage.assign(descriptors_.size(), {});
  state.metadata.fill(std::byte{});
  state.logits.assign(logits_bytes_, std::byte{});
  state.failed.store(false);
  state.restoring = true;
}

cache::Completion FlintContinuationAdapter::CapturePrivate(
    const cache::Slot& slot, cache::ComponentId id, std::span<std::byte> bytes,
    cache::Stream& stream) {
  if (bytes.size() != RequireComponent(id).descriptor.state_bytes)
    throw std::invalid_argument("incorrect Flint private buffer");
  return CapturePrivatePiece(slot, id, 0, bytes, stream);
}

cache::Completion FlintContinuationAdapter::CapturePrivatePiece(
    const cache::Slot& slot, cache::ComponentId id, std::size_t offset,
    std::span<std::byte> bytes, cache::Stream& stream) {
  (void)stream;
  const auto& component = RequireComponent(id);
  auto& state = As(slot);
  if (!state.IsValid())
    throw std::logic_error("cannot capture an invalid Flint slot");
  if (component.descriptor.kind != cache::ComponentKind::kPrivateState)
    throw std::invalid_argument("Flint component is not private state");
  const auto total = component.descriptor.state_bytes;
  if (offset > total || bytes.size() > total - offset)
    throw std::invalid_argument("invalid Flint private piece");

  // Build the complete component so a piece never exposes a partial region.
  SyncSession(*state.session);
  const auto position = state.session->Position();
  std::vector<std::byte> whole(total, std::byte{});
  if (component.private_part == PrivatePart::kMetadata) {
    const Metadata metadata{.version = 1,
                            .position = position,
                            .logits_valid = position > 0 ? 1U : 0U};
    std::memcpy(whole.data(), &metadata, sizeof(metadata));
  } else if (position > 0) {
    const auto logits = state.session->Logits();
    if (logits.size_bytes() != total)
      throw std::logic_error("Flint logits size differs from the layout");
    std::memcpy(whole.data(), logits.data(), total);
  }
  if (!bytes.empty())
    std::memcpy(bytes.data(), whole.data() + offset, bytes.size());
  return State::Settled(true);
}

cache::Completion FlintContinuationAdapter::LoadPrivate(
    cache::Slot& slot, cache::ComponentId id, std::span<const std::byte> bytes,
    cache::Stream& stream) {
  if (bytes.size() != RequireComponent(id).descriptor.state_bytes)
    throw std::invalid_argument("incorrect Flint private buffer");
  return LoadPrivatePiece(slot, id, 0, bytes, stream);
}

cache::Completion FlintContinuationAdapter::LoadPrivatePiece(
    cache::Slot& slot, cache::ComponentId id, std::size_t offset,
    std::span<const std::byte> bytes, cache::Stream& stream) {
  (void)stream;
  auto& state = As(slot);
  if (!state.restoring || state.failed.load())
    throw std::logic_error("Flint load requires an unfailed restore");
  try {
    const auto& component = RequireComponent(id);
    if (component.descriptor.kind != cache::ComponentKind::kPrivateState)
      throw std::invalid_argument("Flint component is not private state");
    const auto total = component.descriptor.state_bytes;
    if (offset > total || bytes.size() > total - offset)
      throw std::invalid_argument("invalid Flint private piece");
    auto* target = component.private_part == PrivatePart::kMetadata
                       ? state.metadata.data()
                       : state.logits.data();
    if (!bytes.empty())
      std::memcpy(target + offset, bytes.data(), bytes.size());
    state.Cover(id.value - 1, offset, offset + bytes.size());
    return State::Settled(true);
  } catch (...) {
    state.failed.store(true);
    throw;
  }
}

cache::Completion FlintContinuationAdapter::CopyRowsOut(
    const cache::Slot& slot, cache::ComponentId id, cache::Rows first,
    cache::Rows end, std::span<std::byte> bytes, cache::Stream& stream) {
  (void)stream;
  const auto& state = As(slot);
  if (!state.IsValid())
    throw std::logic_error("cannot read an invalid Flint slot");
  const auto& component = RequireComponent(id);
  if (component.descriptor.kind != cache::ComponentKind::kAppendRows)
    throw std::invalid_argument("Flint component has no append rows");
  const auto row = row_bytes_;
  // Bound by Capacity, not Position: a rewound slot may still hold rows that
  // a borrowed frontier needs preserved.
  if (first > end || end > state.session->Capacity() ||
      bytes.size() != (end - first) * row)
    throw std::invalid_argument("invalid Flint row capture");
  if (first == end)
    return State::Settled(true);
  SyncSession(*state.session);
  const auto* base = static_cast<const std::byte*>(state.session->KvRows(
      component.layer, component.is_value));
  if (base == nullptr)
    throw std::logic_error("Flint KV rows are not allocated");
  Check(hipMemcpy(bytes.data(), base + first * row, bytes.size(),
                  hipMemcpyDeviceToHost));
  return State::Settled(true);
}

cache::Completion FlintContinuationAdapter::CopyRowsIn(
    cache::Slot& slot, cache::ComponentId id, cache::Rows first,
    cache::Rows end, std::span<const std::byte> bytes, cache::Stream& stream) {
  (void)stream;
  auto& state = As(slot);
  if (!state.restoring || state.failed.load())
    throw std::logic_error("Flint load requires an unfailed restore");
  try {
    const auto& component = RequireComponent(id);
    if (component.descriptor.kind != cache::ComponentKind::kAppendRows)
      throw std::invalid_argument("Flint component has no append rows");
    const auto row = row_bytes_;
    if (first > end || end > state.expected_position ||
        bytes.size() != (end - first) * row)
      throw std::invalid_argument("invalid Flint row restore");
    if (first == end)
      return State::Settled(true);
    auto* base = static_cast<std::byte*>(state.session->KvRows(
        component.layer, component.is_value));
    if (base == nullptr)
      throw std::logic_error("Flint KV rows are not allocated");
    auto* target = base + first * row;
    // The source is either host bytes (RAM cache copy) or a device span from
    // BorrowRowsView on a live slot. hipMemcpyDefault infers the direction
    // from unified addressing; a fixed H2D kind would misread device sources.
    Check(hipMemcpy(target, bytes.data(), bytes.size(), hipMemcpyDefault));
    state.Cover(id.value - 1, first * row, end * row);
    return State::Settled(true);
  } catch (...) {
    state.failed.store(true);
    throw;
  }
}

std::span<const std::byte> FlintContinuationAdapter::BorrowRowsView(
    const cache::Slot& slot, cache::ComponentId id, cache::Rows first,
    cache::Rows end) const {
  const auto& state = As(slot);
  if (!state.IsValid())
    throw std::logic_error("cannot borrow invalid Flint rows");
  const auto& component = RequireComponent(id);
  if (component.descriptor.kind != cache::ComponentKind::kAppendRows)
    throw std::logic_error("Flint component has no append rows");
  if (first >= end || end > state.session->Position())
    throw std::invalid_argument("invalid Flint borrowed range");
  SyncSession(*state.session);
  const auto* base = static_cast<const std::byte*>(state.session->KvRows(
      component.layer, component.is_value));
  if (base == nullptr)
    throw std::logic_error("Flint KV rows are not allocated");
  return {base + first * row_bytes_, (end - first) * row_bytes_};
}

bool FlintContinuationAdapter::Validate(
    cache::Slot& slot, std::span<const cache::ComponentPosition> positions) {
  auto& state = As(slot);
  if (!state.restoring)
    throw std::logic_error("Flint validation requires an active restore");
  const auto refuse = [&] {
    state.failed.store(true);
    return false;
  };
  if (state.failed.load() || positions.size() != descriptors_.size())
    return refuse();

  // Every component must report exactly the expected frontier and be fully
  // covered by loads: rows up to valid_rows, private state whole. Reading
  // state.coverage without coverage_mutex is safe here: the caller settles
  // all loads before calling Validate, so no Cover() call is in flight.
  for (std::size_t i = 0; i < state.expected.size(); ++i) {
    const auto& expected = state.expected[i];
    const auto& descriptor = descriptors_[i];
    if (std::ranges::count(positions, expected) != 1 ||
        !Covered(state.coverage[i],
                 descriptor.row_bytes
                     ? expected.valid_rows * descriptor.row_bytes
                     : descriptor.state_bytes))
      return refuse();
  }

  Metadata metadata{};
  std::memcpy(&metadata, state.metadata.data(), sizeof(metadata));
  if (metadata.version != 1 ||
      metadata.position != state.expected_position ||
      metadata.logits_valid != (metadata.position > 0 ? 1U : 0U))
    return refuse();

  try {
    std::string error;
    if (!state.session->RestoreRows(
            static_cast<std::uint32_t>(state.expected_position), &error))
      return refuse();
    if (metadata.logits_valid == 1) {
      std::vector<float> logits(logits_bytes_ / sizeof(float));
      std::memcpy(logits.data(), state.logits.data(), logits_bytes_);
      if (!state.session->SetLogits(logits, &error))
        return refuse();
    }
    if (state.session->Position() != state.expected_position)
      return refuse();
  } catch (...) {
    return refuse();
  }
  state.restoring = false;
  state.expected.clear();
  state.coverage.clear();
  return true;
}

bool FlintContinuationAdapter::Invalidate(cache::Slot& slot) noexcept {
  try {
    auto& state = As(slot);
    for (const auto& component : components_)
      if (component.descriptor.kind == cache::ComponentKind::kAppendRows)
        state.guard.BeforeRelease(component.descriptor.id, 0, context_);
    state.session->Reset();
    state.restoring = false;
    state.failed.store(false);
    state.expected.clear();
    state.coverage.clear();
    state.guard.AfterReset();
    return true;
  } catch (...) {
    return false;
  }
}

}  // namespace gufo::models::flint
