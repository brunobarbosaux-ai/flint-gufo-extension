#ifndef GUFO_MODELS_FLINT_GPU_ENGINE_HPP_
#define GUFO_MODELS_FLINT_GPU_ENGINE_HPP_

#include <cstdint>
#include <functional>
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

  /// Positions the ring covers; sized to a whole prefill chunk (8192) so that
  /// speculation prefills in the same chunks as spec-off. The drafter's own
  /// context ring keeps its 2048-token sliding window separately.
  static constexpr std::uint32_t kHiddenRingRows = 8192;
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

  // --- Row-level access (per-layer KV components) -------------------------
  //
  // Each layer's K and V cache is a device buffer laid out as
  // [position][num_kv * head_dim] of __half, token-major. The layout and the
  // buffers are stable for the session lifetime. A row is KvRowBytes() bytes.

  /// Number of layers; the valid range for KvRows().
  [[nodiscard]] std::uint32_t NumLayers() const noexcept;
  /// Bytes of one token's K (or V) row in one layer: num_kv*head_dim*2.
  [[nodiscard]] std::size_t KvRowBytes() const noexcept;
  /// Device pointer to layer `layer`'s K cache (`value == false`) or V cache
  /// (`value == true`). nullptr when `layer >= NumLayers()`.
  [[nodiscard]] const void* KvRows(std::uint32_t layer,
                                   bool value) const noexcept;
  [[nodiscard]] void* KvRows(std::uint32_t layer, bool value) noexcept;
  /// Waits for all work this session has queued.
  [[nodiscard]] bool Synchronize(std::string* error_msg) const;
  /// Declares the first `position` rows of every layer to be in place (written
  /// through KvRows() or copied from a snapshot) and makes them the session
  /// state: position, one logits row, and sparse key bounds rebuilt from the
  /// keys. Does not write logits; call SetLogits() for those. position 0 empties
  /// the session (logits zeroed, sparse bounds cleared).
  [[nodiscard]] bool RestoreRows(std::uint32_t position,
                                 std::string* error_msg);
  /// Sets the logits Logits() reports to `row` (vocab_size floats).
  [[nodiscard]] bool SetLogits(std::span<const float> row,
                               std::string* error_msg);
  /// Called with [first, end) once per Append/AppendAll, before any row in
  /// that range is written. Not called by Rewind. An empty function clears it.
  void SetRowWriteHook(
      std::function<void(std::uint32_t first, std::uint32_t end)> hook);

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
