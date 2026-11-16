#ifndef GUFO_MODELS_FLINT_GPU_ENGINE_HPP_
#define GUFO_MODELS_FLINT_GPU_ENGINE_HPP_

#include <cstdint>
#include <memory>
#include <span>
#include <string>

#include "src/runtime/config.hpp"

namespace gufo::models::flint::gpu {

struct EngineOptions {
  /// Tokens one prefill pass processes.
  std::uint32_t prefill_tile{4096};
};

class Model;
struct ModelState;

/// One conversation: KV cache, position and the logits of the last token.
class Session final {
public:
  ~Session();
  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;

  /// Appends `tokens` and leaves the logits of the last one in Logits().
  [[nodiscard]] bool Append(std::span<const std::int32_t> tokens,
                            std::string* error_msg);

  /// Appends 1..kMaxVerifyRows tokens and keeps the logits of every row:
  /// row i is the distribution after tokens[i]. Used to verify drafts.
  [[nodiscard]] bool AppendAll(std::span<const std::int32_t> tokens,
                               std::string* error_msg);
  /// Widths 1..8 use the engine's vectorised path (kVecBatch = 8); DFlash2
  /// drafts 7 tokens, so the verify width is at most 8 rows.
  static constexpr std::uint32_t kMaxVerifyRows = 8;

  /// Drops cache rows past `position`; later appends overwrite them.
  void Rewind(std::uint32_t position) noexcept;

  // --- Layer-residual capture (DFlash2 drafter input) ---------------------
  //
  // `SetHiddenCapture` names zero-based layers whose residual output is what
  // CapturedHidden() later exposes: the stream that layer's attention and
  // FFN/MoE wrote, before the next layer's norm. The copies are part of the
  // step, so a captured decode graph replays them; changing the set therefore
  // invalidates those graphs.
  //
  // Tap order is the caller's order, verbatim (HiddenCaptureLayers() returns
  // it). Positions older than the ring, or at/after Position(), are not
  // available. The ring is not part of a snapshot: a restored session must
  // re-prime it from fresh appends before a drafter can read past positions.

  /// Positions the ring covers; covers the drafter's 2048-token window.
  static constexpr std::uint32_t kHiddenRingRows = 2048;
  /// Taps one ring can hold; the drafter uses five.
  static constexpr std::uint32_t kMaxHiddenTaps = 16;

  /// Enables capture of `layer_output_ids` (empty disables and frees the ring).
  [[nodiscard]] bool SetHiddenCapture(
      std::span<const std::uint32_t> layer_output_ids,
      std::string* error_msg = nullptr);
  [[nodiscard]] std::span<const std::uint32_t> HiddenCaptureLayers()
      const noexcept;
  /// Rows the ring holds for this session (0 when capture is off).
  [[nodiscard]] std::uint32_t HiddenRingRows() const noexcept;
  /// True when `row` is still in the ring at the current position.
  [[nodiscard]] bool HasHidden(std::uint32_t row) const noexcept;
  /// Copies row `row`, tap `tap` to host scratch; empty when unavailable.
  /// The span aliases per-session scratch: copy it before the next call.
  [[nodiscard]] std::span<const float> CapturedHidden(std::uint32_t row,
                                                      std::uint32_t tap);
  /// Device pointer to the same row, for a consumer that reads it in place.
  /// nullptr when unavailable; valid until the row leaves the ring.
  [[nodiscard]] const float* CapturedHiddenDevice(std::uint32_t row,
                                                  std::uint32_t tap)
      const noexcept;

  [[nodiscard]] std::span<const float> Logits() const noexcept;
  /// Row `row` of the last AppendAll (or Append of up to kMaxVerifyRows).
  [[nodiscard]] std::span<const float> Logits(std::uint32_t row) const noexcept;
  [[nodiscard]] std::uint32_t Position() const noexcept;
  [[nodiscard]] std::uint32_t Capacity() const noexcept;

  /// Forgets everything: position 0, empty cache.
  void Reset() noexcept;

  /// Device bytes of the KV caches and key summaries (scratch excluded).
  [[nodiscard]] std::size_t AllocatedBytes() const noexcept;

  /// Exact continuation at the current position: the K and V rows of every
  /// layer plus the last logits, so a restored state can sample at once. A
  /// sliding-window layer keeps only its last `sliding_window` positions.
  /// Bumped when the layout changes.
  static constexpr std::uint32_t kSnapshotPayloadVersion = 2;
  [[nodiscard]] std::size_t SnapshotBytes() const noexcept;
  [[nodiscard]] bool SaveSnapshot(std::span<std::uint8_t> out,
                                  std::string* error_msg) const;
  /// Replaces the session state with a payload from SaveSnapshot (of this
  /// model). The position becomes the payload's.
  [[nodiscard]] bool RestoreSnapshot(std::span<const std::uint8_t> payload,
                                     std::string* error_msg);

  /// Row-wise access to the KV caches, for a continuation cache that holds
  /// rows instead of one blob. Row `r` of a layer is position `r` (absolute);
  /// a window layer's ring maps it. `host` holds (end - first) rows of
  /// RowBytes() each. Copying out needs the rows to still be in the ring.
  [[nodiscard]] std::size_t RowBytes() const noexcept;
  [[nodiscard]] bool TransferRows(std::uint32_t layer, bool value,
                                  std::uint32_t first, std::uint32_t end,
                                  std::span<std::uint8_t> host, bool to_host,
                                  std::string* error_msg);
  /// After the rows of a restore are in place: makes `position` the live
  /// position and `logits` (vocab_size floats) the last logits.
  [[nodiscard]] bool SetState(std::uint32_t position,
                              std::span<const float> logits,
                              std::string* error_msg);

private:
  friend class Model;
  Session();
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

/// The target's tied embedding table and LM head, as device pointers in their
/// GGUF encoding. A DFlash2 drafter borrows both instead of carrying its own
/// million-row copy, so it reads them from here.
struct TiedWeights {
  const void* token_embedding{nullptr};
  core::GgmlType token_embedding_type{core::GgmlType::kF32};
  const void* output{nullptr};
  core::GgmlType output_type{core::GgmlType::kF32};
};

/// The Flint weights resident on the GPU, shared by sessions.
class Model final {
public:
  ~Model();
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  [[nodiscard]] static std::shared_ptr<Model> Load(
      const std::string& model_path, const EngineOptions& options,
      std::string* error_msg = nullptr);

  [[nodiscard]] std::unique_ptr<Session> CreateSession(
      std::uint32_t max_context, std::string* error_msg = nullptr);

  [[nodiscard]] const Config& config() const noexcept;
  [[nodiscard]] std::size_t ResidentBytes() const noexcept;
  /// Free device memory right now, in bytes; 0 when unknown (CPU build).
  /// Session sizing uses this so long contexts degrade instead of failing.
  [[nodiscard]] static std::uint64_t FreeDeviceBytes() noexcept;
  /// Device bytes of one session with `max_context` positions.
  [[nodiscard]] std::size_t SessionBytes(
      std::uint32_t max_context) const noexcept;
  /// Bytes of a snapshot at `tokens` positions.
  [[nodiscard]] std::size_t SnapshotBytes(std::uint32_t tokens) const noexcept;
  /// The embedding table and LM head a drafter borrows. Valid for the life of
  /// the model; the pointers are the ones sessions read.
  [[nodiscard]] TiedWeights Tied() const noexcept;

private:
  Model() = default;
  std::shared_ptr<ModelState> state_;
};

}  // namespace gufo::models::flint::gpu

#endif  // GUFO_MODELS_FLINT_GPU_ENGINE_HPP_
