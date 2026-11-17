// DFlash2 draft-length policy (M4). Pure host code: no device, no sessions.
//
// `fixed` always proposes the maximum. `adaptive` picks the length that
// maximises expected tokens per cycle divided by the cycle's cost in decode
// units, using:
//   - a per-position acceptance estimate over a rolling window of cycles (the
//     Strycast window idea, applied to the accepted-prefix chain rather than to
//     the pooled accepted/drafted ratio, which the chain makes misleading);
//   - the measured verify-width ratio at the cycle's target depth.
//
// Cycle cost in decode units, for a block of k proposals (width k+1):
//   drafter forward (kDrafterRatio) + verify ratio V(k+1, depth). The emitted
//   token's row is written by the next verify, so no append term. Expected
//   tokens with per-position acceptance a: 1 + sum_{i=1..k} a^i.
#ifndef GUFO_MODELS_FLINT_SPEC_DRAFT_POLICY_HPP_
#define GUFO_MODELS_FLINT_SPEC_DRAFT_POLICY_HPP_

#include <array>
#include <cstddef>
#include <cstdint>

#include <gufo/ext/text.hpp>

namespace gufo::models::flint::spec {

/// Drafter block forward, in single-token decode units (measured M1, Q6_K).
inline constexpr double kDrafterRatio = 0.48;

/// Verify cost at `width` rows (1..8) relative to a single-token decode, at
/// `depth` cached positions. Interpolated linearly in depth between the
/// measured depths 0 / 8k / 32k / 131k and clamped outside them.
[[nodiscard]] double VerifyRatio(std::uint32_t width, std::uint32_t depth);

class DraftPolicy {
 public:
  /// Rolling window of finished cycles that feeds the acceptance estimate.
  static constexpr std::size_t kWindow = 16;

  /// `max_draft` is the largest proposal count the drafter will be asked for;
  /// `min_draft` is the floor used by the adaptive policy. Both are clamped to
  /// 1..kMaxDraft and min <= max.
  DraftPolicy(ext::TextDFlashDraftPolicy policy, std::uint32_t min_draft,
              std::uint32_t max_draft) noexcept;

  /// Proposal count for a cycle whose anchor sits at target position `depth`.
  [[nodiscard]] std::uint32_t Length(std::uint32_t depth) const noexcept;

  /// Records a finished cycle: `drafted` proposals, `accepted` of them matched
  /// the sampled tokens as a prefix. Only natural cycle ends are recorded;
  /// aborted cycles say nothing about acceptance.
  void Observe(std::uint32_t drafted, std::uint32_t accepted) noexcept;

  /// Per-position acceptance estimate over the window (1.0 when empty).
  [[nodiscard]] double Acceptance() const noexcept;

  [[nodiscard]] std::uint32_t min_draft() const noexcept { return min_draft_; }
  [[nodiscard]] std::uint32_t max_draft() const noexcept { return max_draft_; }
  [[nodiscard]] ext::TextDFlashDraftPolicy policy() const noexcept {
    return policy_;
  }

  /// Largest proposal count a cycle may request.
  static constexpr std::uint32_t kMaxDraft = 7;

 private:
  struct Outcome {
    std::uint32_t drafted;
    std::uint32_t accepted;
  };

  ext::TextDFlashDraftPolicy policy_;
  std::uint32_t min_draft_;
  std::uint32_t max_draft_;
  std::array<Outcome, kWindow> window_{};
  std::size_t count_{0};   // filled slots, <= kWindow
  std::size_t next_{0};    // ring write index
};

}  // namespace gufo::models::flint::spec

#endif  // GUFO_MODELS_FLINT_SPEC_DRAFT_POLICY_HPP_
