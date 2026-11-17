// Draft policy (M4) unit test. Host only: no device, no weights.
//
//   1. Fixed always proposes max_draft, whatever the history and depth.
//   2. Bounds: every adaptive length lies in [min, max] for all depths and
//      histories; max is clamped to 1..7 and min to 1..max.
//   3. Adaptive shortens when acceptance is low and lengthens when high.
//   4. Acceptance estimate is the geometric MLE over the window and is
//      recorded only through Observe (zero-draft cycles are ignored).
//   5. Verify ratios are monotone in width and depth and match the table.

#include <cstdint>
#include <cstdio>
#include <string>

#include <gufo/ext/text.hpp>

#include "src/spec/draft_policy.hpp"

namespace spec = gufo::models::flint::spec;
namespace ext = gufo::ext;

namespace {

int failures = 0;

void Check(bool ok, const std::string& what) {
  if (!ok) {
    ++failures;
    std::printf("FAIL %s\n", what.c_str());
  }
}

bool Near(double a, double b) { return a > b - 1e-9 && a < b + 1e-9; }

void TestFixedIgnoresHistory() {
  spec::DraftPolicy policy(ext::TextDFlashDraftPolicy::kFixed, 1, 5);
  Check(policy.Length(0) == 5, "fixed starts at max");
  for (int i = 0; i < 20; ++i)
    policy.Observe(5, 0);
  Check(policy.Length(0) == 5, "fixed ignores zero acceptance");
  Check(policy.Length(131072) == 5, "fixed ignores depth");
}

void TestBoundsAndClamping() {
  // max above the drafter's block (7) clamps to 7; min above max clamps down.
  spec::DraftPolicy high(ext::TextDFlashDraftPolicy::kFixed, 1, 99);
  Check(high.max_draft() == 7, "max clamps to 7");
  spec::DraftPolicy inverted(ext::TextDFlashDraftPolicy::kAdaptive, 6, 3);
  Check(inverted.max_draft() == 3 && inverted.min_draft() == 3,
        "min clamps to max");
  spec::DraftPolicy zero(ext::TextDFlashDraftPolicy::kAdaptive, 0, 0);
  Check(zero.max_draft() == 1 && zero.min_draft() == 1,
        "zero bounds clamp to 1");

  // Adaptive stays inside [min, max] over every history and depth.
  const std::uint32_t depths[] = {0, 8192, 32768, 131072, 400000};
  for (std::uint32_t lo = 1; lo <= 7; ++lo) {
    for (std::uint32_t hi = lo; hi <= 7; ++hi) {
      spec::DraftPolicy policy(ext::TextDFlashDraftPolicy::kAdaptive, lo, hi);
      for (int step = 0; step < 40; ++step) {
        for (const auto depth : depths) {
          const auto length = policy.Length(depth);
          Check(length >= lo && length <= hi,
                "adaptive length within [min, max]");
        }
        // Alternate outcomes so the window sweeps through a range of rates.
        policy.Observe(7, static_cast<std::uint32_t>(step % 8));
      }
    }
  }
}

void TestAdaptiveDirection() {
  // Full acceptance: longest block allowed.
  spec::DraftPolicy hot(ext::TextDFlashDraftPolicy::kAdaptive, 1, 7);
  for (int i = 0; i < 16; ++i)
    hot.Observe(7, 7);
  Check(Near(hot.Acceptance(), 1.0), "full acceptance estimates 1.0");
  Check(hot.Length(0) == 7, "full acceptance drafts max at depth 0");

  // Near-zero acceptance: the floor, not the max.
  spec::DraftPolicy cold(ext::TextDFlashDraftPolicy::kAdaptive, 1, 7);
  for (int i = 0; i < 16; ++i)
    cold.Observe(7, 0);
  Check(cold.Length(0) == 1, "zero acceptance drafts the floor");
  Check(cold.Length(131072) == 1, "zero acceptance drafts the floor deep");

  // Min floor is honoured even when pricing says shorter is better.
  spec::DraftPolicy floor(ext::TextDFlashDraftPolicy::kAdaptive, 3, 7);
  for (int i = 0; i < 16; ++i)
    floor.Observe(7, 0);
  Check(floor.Length(0) == 3, "min floor holds at zero acceptance");

  // Deeper context makes verify rows dearer, so the length cannot grow with
  // depth at fixed acceptance.
  spec::DraftPolicy mid(ext::TextDFlashDraftPolicy::kAdaptive, 1, 7);
  for (int i = 0; i < 16; ++i)
    mid.Observe(7, 2);
  Check(mid.Length(131072) <= mid.Length(0),
        "length does not grow with depth at fixed acceptance");
}

void TestAcceptanceEstimate() {
  // Geometric MLE: successes / (successes + failures). One cycle of 4 with 1
  // accepted gives 1 success and 1 failure; one full cycle of 4 gives 4
  // successes and no failure. Total 5 / 6.
  spec::DraftPolicy policy(ext::TextDFlashDraftPolicy::kAdaptive, 1, 7);
  Check(Near(policy.Acceptance(), 1.0), "empty window estimates 1.0");
  policy.Observe(4, 1);
  policy.Observe(4, 4);
  Check(Near(policy.Acceptance(), 5.0 / 6.0), "geometric MLE over window");
  policy.Observe(0, 0);
  Check(Near(policy.Acceptance(), 5.0 / 6.0), "zero-draft cycle is ignored");
  policy.Observe(2, 9);  // accepted clamps to drafted: a full block of 2
  Check(Near(policy.Acceptance(), 7.0 / 8.0), "accepted clamps to drafted");
}

void TestWindowRolls() {
  // The window keeps the last kWindow cycles: old outcomes age out.
  spec::DraftPolicy policy(ext::TextDFlashDraftPolicy::kAdaptive, 1, 7);
  for (std::size_t i = 0; i < spec::DraftPolicy::kWindow; ++i)
    policy.Observe(4, 0);
  Check(Near(policy.Acceptance(), 0.0), "all-rejected window");
  for (std::size_t i = 0; i < spec::DraftPolicy::kWindow; ++i)
    policy.Observe(4, 4);
  Check(Near(policy.Acceptance(), 1.0), "old cycles age out of the window");
}

void TestVerifyRatio() {
  Check(Near(spec::VerifyRatio(1, 0), 1.0), "width 1 is one decode");
  Check(Near(spec::VerifyRatio(8, 0), 2.215), "width 8 at depth 0");
  Check(Near(spec::VerifyRatio(8, 131072), 4.25), "width 8 at 131k");
  // Linear in depth between 8k (2.48) and 32k (3.16): the midpoint at 20480.
  Check(Near(spec::VerifyRatio(8, 20480), (2.48 + 3.16) / 2.0),
        "verify ratio interpolates linearly in depth");
  for (std::uint32_t depth : {0U, 8192U, 32768U, 131072U, 500000U}) {
    for (std::uint32_t w = 2; w <= 8; ++w) {
      Check(spec::VerifyRatio(w, depth) >= spec::VerifyRatio(w - 1, depth),
            "verify ratio grows with width");
    }
  }
}

}  // namespace

int main() {
  TestFixedIgnoresHistory();
  TestBoundsAndClamping();
  TestAdaptiveDirection();
  TestAcceptanceEstimate();
  TestWindowRolls();
  TestVerifyRatio();
  if (failures == 0) {
    std::printf("flint.draft_policy: all checks passed\n");
    return 0;
  }
  std::printf("flint.draft_policy: %d failure(s)\n", failures);
  return 1;
}
