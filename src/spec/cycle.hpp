// DFlash2 speculative cycle: draft -> verify -> lazy row acceptance -> rewind.
//
// Mirrors RunAcceptance in tools/drafter_parity.cpp. The cycle owns no
// sampler: every emitted token is one Sample() call on the caller's live
// sampling::SamplerState, in order, so penalties, grammar and RNG order are
// the ones spec-off decoding would use.
//
// Position convention. The anchor is the last committed token. Two starts:
//  - After a cycle's natural end the emitted token is the anchor, uncached:
//    the cache holds [0, P) with P == target.Position() and the anchor at P.
//    The verify pass appends [anchor, drafts] at P..P+k (row 0 caches the
//    anchor's row).
//  - The first cycle after prefill has the anchor cached at Position()-1; the
//    start rewinds onto it so the verify re-appends it as row 0.
// Each cycle ends with Rewind(P + accepted + 1), keeping the anchor and the
// accepted drafts cached. The emitted token then becomes pending (uncached).
#ifndef GUFO_MODELS_FLINT_SPEC_CYCLE_HPP_
#define GUFO_MODELS_FLINT_SPEC_CYCLE_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <gufo/ext/sampling.hpp>

#include "src/gpu/dflash/dflash_forward.hpp"
#include "src/spec/draft_policy.hpp"
#include "src/spec/target_adapter.hpp"

namespace gufo::models::flint::spec {

/// The host's TextDecodeStep draft fields, accumulated by the cycle.
struct CycleStats {
  std::size_t draft_rounds{0};
  std::size_t draft_tokens{0};
  std::size_t draft_accepted_tokens{0};
};

class SpecCycle {
 public:
  /// `policy` picks each block's length (see DraftPolicy). Borrows `target`
  /// and `drafter`; both must outlive it. `context` is this cycle's own
  /// drafter context (one per session), so sessions never share a ring.
  SpecCycle(TargetAdapter& target, gpu::dflash::DflashForward& drafter,
            std::unique_ptr<gpu::dflash::DflashContext> context,
            DraftPolicy policy);

  /// Returns the next token drawn from `sampler`. When no cycle is active,
  /// starts one from `anchor` (the last committed token, not yet cached),
  /// which drafts and verifies in one pass. Otherwise samples the next row.
  /// Sets *token and returns false with *error_msg on failure.
  [[nodiscard]] bool Next(std::uint32_t anchor, sampling::SamplerState& sampler,
                          std::uint32_t* token, std::string* error_msg);

  /// Feeds the drafter the target residuals for rows [injected, position).
  /// Call after each prefill chunk so the ring never loses a row.
  [[nodiscard]] bool SyncContext(std::uint32_t position,
                                 std::string* error_msg);

  /// Ends an active cycle early (EOS or cancellation): rewinds the target to
  /// the rows the emitted tokens justify. Safe to call when idle.
  void Abort() noexcept;

  /// Forgets the drafter context and any active cycle (new prompt). The draft
  /// policy's acceptance history is kept: it tunes speed, not output.
  void Reset() noexcept { Reset(0); }

  /// Restarts the drafter context at `frontier` (the target's cached position
  /// after an external reposition, e.g. prefix reuse or snapshot restore). The
  /// rows below it are not re-fed: the drafter's history restarts there.
  void Reset(std::uint32_t frontier) noexcept;

  /// Test hook (null drafter, flint_spec_gate only): rewrites each proposal before verification.
  /// The rewritten block must still be verified like any other, so the cycle's
  /// accept/reject and sampling path is exercised with nothing accepted.
  using DraftTransform = std::uint32_t (*)(std::uint32_t token,
                                           std::uint32_t vocab_size);
  void set_draft_transform(DraftTransform transform,
                           std::uint32_t vocab_size) noexcept {
    draft_transform_ = transform;
    vocab_size_ = vocab_size;
  }

  [[nodiscard]] bool active() const noexcept { return active_; }
  /// True after a block ended with an emitted token whose row is not cached.
  /// The next Next() without an active cycle starts at the target's Position(),
  /// and its verify pass caches that token's row as row 0.
  [[nodiscard]] bool pending() const noexcept { return pending_; }
  [[nodiscard]] const CycleStats& stats() const noexcept { return stats_; }
  [[nodiscard]] const DraftPolicy& policy() const noexcept { return policy_; }

 private:
  [[nodiscard]] bool Start(std::uint32_t anchor, std::string* error_msg);
  // `natural`: the block ended by a verify decision (mismatch or full block),
  // so its acceptance is recorded. False for Abort and Reset.
  void End(bool natural) noexcept;

  TargetAdapter& target_;
  gpu::dflash::DflashForward& drafter_;
  std::unique_ptr<gpu::dflash::DflashContext> context_;
  DraftPolicy policy_;
  std::uint32_t injected_{0};
  std::vector<float> features_;

  bool active_{false};
  // The last emitted token (the anchor) is not cached; its row is written by
  // the next verify. Set only by a natural End.
  bool pending_{false};
  std::uint32_t base_{0};      // P: target position at cycle start
  std::uint32_t draft_k_{0};   // proposals in the active block
  std::uint32_t row_{0};       // next verify row to sample
  std::uint32_t accepted_{0};  // drafts accepted so far in this cycle
  std::vector<std::uint32_t> drafts_;
  CycleStats stats_;
  DraftTransform draft_transform_{nullptr};
  std::uint32_t vocab_size_{0};
};

}  // namespace gufo::models::flint::spec

#endif  // GUFO_MODELS_FLINT_SPEC_CYCLE_HPP_
