#include "src/spec/draft_policy.hpp"

#include <algorithm>

namespace gufo::models::flint::spec {

namespace {

// Measured verify ratios (`gufo bench` with FLINT_BENCH_VERIFY=1, Q4_K_XL,
// one repetition), indexed [width - 1][depth]. Depths: 0 / 8k / 32k / 131k.
// Width 8 is the M2 measurement.
constexpr std::array<std::array<double, 4>, 8> kVerifyRatio{{
    {{1.00, 1.00, 1.00, 1.00}},  // width 1
    {{1.182, 1.192, 1.275, 1.468}},
    {{1.378, 1.465, 1.772, 2.282}},
    {{1.542, 1.639, 1.907, 2.358}},
    {{1.817, 1.983, 2.420, 3.147}},
    {{1.949, 2.105, 2.517, 3.206}},
    {{2.086, 2.358, 3.044, 4.216}},
    {{2.215, 2.48, 3.16, 4.25}},  // width 8
}};
constexpr std::array<double, 4> kDepths{0.0, 8192.0, 32768.0, 131072.0};

}  // namespace

double VerifyRatio(std::uint32_t width, std::uint32_t depth) {
  const std::uint32_t w = std::clamp<std::uint32_t>(width, 1U, 8U);
  const auto& row = kVerifyRatio[w - 1U];
  const double d = static_cast<double>(depth);
  if (d <= kDepths.front())
    return row.front();
  if (d >= kDepths.back())
    return row.back();
  std::size_t hi = 1;
  while (d > kDepths[hi])
    ++hi;
  const double t = (d - kDepths[hi - 1]) / (kDepths[hi] - kDepths[hi - 1]);
  return row[hi - 1] + t * (row[hi] - row[hi - 1]);
}

DraftPolicy::DraftPolicy(ext::TextDFlashDraftPolicy policy,
                         std::uint32_t min_draft,
                         std::uint32_t max_draft) noexcept
    : policy_(policy) {
  max_draft_ = std::clamp<std::uint32_t>(max_draft, 1U, kMaxDraft);
  min_draft_ = std::clamp<std::uint32_t>(min_draft, 1U, max_draft_);
}

std::uint32_t DraftPolicy::Length(std::uint32_t depth) const noexcept {
  if (policy_ == ext::TextDFlashDraftPolicy::kFixed || count_ == 0)
    return max_draft_;

  // Expected tokens per cycle: the anchor-row sample plus the accepted prefix,
  // 1 + sum_{i<=k} a^i. Cost: drafter + verify of k+1 rows. The emitted
  // token's row is written by the next verify, so there is no append term.
  const double a = Acceptance();
  std::uint32_t best = min_draft_;
  double best_score = -1.0;
  double expected = 1.0;
  double power = 1.0;
  for (std::uint32_t k = 1; k <= max_draft_; ++k) {
    power *= a;
    expected += power;
    if (k < min_draft_)
      continue;
    const double cost = kDrafterRatio + VerifyRatio(k + 1U, depth);
    const double score = expected / cost;
    // Strict '>' keeps the shorter block on ties: it costs less at equal yield.
    if (score > best_score) {
      best_score = score;
      best = k;
    }
  }
  return best;
}

void DraftPolicy::Observe(std::uint32_t drafted,
                          std::uint32_t accepted) noexcept {
  if (drafted == 0)
    return;
  window_[next_] = {drafted, std::min(accepted, drafted)};
  next_ = (next_ + 1) % kWindow;
  count_ = std::min(count_ + 1, kWindow);
}

double DraftPolicy::Acceptance() const noexcept {
  // Maximum likelihood for a geometric prefix: each position accepts with
  // probability a. A cycle with j < k accepted contributes j successes and one
  // failure; a full block contributes k successes and no failure.
  if (count_ == 0)
    return 1.0;
  std::size_t successes = 0;
  std::size_t failures = 0;
  for (std::size_t i = 0; i < count_; ++i) {
    successes += window_[i].accepted;
    if (window_[i].accepted < window_[i].drafted)
      ++failures;
  }
  if (successes + failures == 0)
    return 1.0;
  return static_cast<double>(successes) /
         static_cast<double>(successes + failures);
}

}  // namespace gufo::models::flint::spec
