// Target adapter for the DFlash2 speculative cycle.
//
// A thin wrapper over one gpu::Session that exposes exactly what the cycle
// needs and nothing more: batched prompt append, one-pass verification with
// every row's logits kept, the packed tapped residuals the drafter consumes,
// rewind and position. It knows no drafter; it only knows the session and the
// tap ids.
//
// Hidden-ring lifetime. The engine's capture ring holds the last
// gpu::Session::kHiddenRingRows positions. Features() reads from the ring, so a
// caller must take a row's features before a later append pushes that row out.
// AppendPrompt() therefore appends in chunks, and the caller calls Features()
// for each chunk before the next one; the same holds for any verify pass whose
// rows the drafter has not yet consumed.
#ifndef GUFO_MODELS_FLINT_SPEC_TARGET_ADAPTER_HPP_
#define GUFO_MODELS_FLINT_SPEC_TARGET_ADAPTER_HPP_

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "src/gpu/engine.hpp"

namespace gufo::models::flint::spec {

class TargetAdapter {
 public:
  /// Tokens per prefill call. Matches the drafter_parity harness, so the
  /// hidden ring keeps the previous chunk's rows while the caller copies them.
  static constexpr std::uint32_t kPrefillChunk = 1024;

  /// Enables residual capture of `taps` (zero-based layer outputs) on
  /// `session` and returns an adapter over it. `hidden_width` is the model's
  /// hidden size; each captured row must have exactly that many floats.
  /// Returns nullptr and sets *error_msg on failure. Call before the first
  /// append so the ring covers the whole conversation.
  [[nodiscard]] static std::unique_ptr<TargetAdapter> Create(
      gpu::Session& session, std::span<const std::uint32_t> taps,
      std::uint32_t hidden_width, std::string* error_msg);

  /// Appends `tokens` in chunks of at most kPrefillChunk; the logits of the
  /// last token are in Logits(). Fails when the session would overflow.
  [[nodiscard]] bool AppendPrompt(std::span<const std::int32_t> tokens,
                                  std::string* error_msg);

  /// Appends 1..gpu::Session::kMaxVerifyRows tokens in one pass and keeps the
  /// logits of every row: row i is the distribution after tokens[i].
  [[nodiscard]] bool VerifyRows(std::span<const std::int32_t> tokens,
                                std::string* error_msg);

  /// Row `row` of the last VerifyRows (or the last token of AppendPrompt).
  [[nodiscard]] std::span<const float> RowLogits(std::uint32_t row) const;
  /// The last token's logits.
  [[nodiscard]] std::span<const float> Logits() const;

  /// Packed residuals for rows [first, first+count): count * taps * hidden
  /// floats, row-major by position and then by tap in Taps() order (the layout
  /// fc.weight expects). Fails if a row is not in the ring.
  [[nodiscard]] bool Features(std::uint32_t first, std::uint32_t count,
                              std::vector<float>* out, std::string* error_msg);

  /// Drops cache rows past `position`; later appends overwrite them.
  void Rewind(std::uint32_t position) noexcept;
  [[nodiscard]] std::uint32_t Position() const noexcept;
  [[nodiscard]] std::uint32_t Capacity() const noexcept;
  [[nodiscard]] std::uint32_t HiddenWidth() const noexcept;
  [[nodiscard]] std::span<const std::uint32_t> Taps() const noexcept;

  /// The FLINT_CHECK_ROWS property, asserted from inside the adapter: appends
  /// `tokens` one at a time, then rewinds and runs them as one VerifyRows, and
  /// compares every row's logits bit for bit (no tolerance). Returns false and
  /// sets *error_msg on any mismatch. On return the session is in the state
  /// VerifyRows(tokens) leaves it in.
  [[nodiscard]] bool CheckRowsBitIdentical(
      std::span<const std::int32_t> tokens, std::string* error_msg);

 private:
  TargetAdapter(gpu::Session& session, std::vector<std::uint32_t> taps,
                std::uint32_t hidden_width);

  gpu::Session& session_;
  std::vector<std::uint32_t> taps_;
  std::uint32_t hidden_width_;
};

}  // namespace gufo::models::flint::spec

#endif  // GUFO_MODELS_FLINT_SPEC_TARGET_ADAPTER_HPP_
