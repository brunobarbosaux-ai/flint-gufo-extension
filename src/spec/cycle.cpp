#include "src/spec/cycle.hpp"

#include <memory>
#include <utility>

namespace gufo::models::flint::spec {

namespace {

bool Fail(std::string* error_msg, std::string message) {
  if (error_msg != nullptr)
    *error_msg = std::move(message);
  return false;
}

}  // namespace

SpecCycle::SpecCycle(TargetAdapter& target,
                     gpu::dflash::DflashForward& drafter,
                     std::unique_ptr<gpu::dflash::DflashContext> context,
                     DraftPolicy policy)
    : target_(target),
      drafter_(drafter),
      context_(std::move(context)),
      policy_(policy) {}

bool SpecCycle::SyncContext(std::uint32_t position, std::string* error_msg) {
  if (position == injected_)
    return true;
  if (position < injected_)
    return Fail(error_msg, "drafter context cannot move backwards");
  const std::uint32_t count = position - injected_;
  if (!target_.Features(injected_, count, &features_, error_msg))
    return false;
  if (!drafter_.AppendContext(*context_, features_, injected_, count,
                              error_msg))
    return false;
  injected_ = position;
  return true;
}

void SpecCycle::Reset(std::uint32_t frontier) noexcept {
  End(false);
  pending_ = false;
  context_->ResetContext(frontier);
  injected_ = frontier;
  stats_ = {};
}

void SpecCycle::Abort() noexcept {
  if (active_)
    End(false);
}

void SpecCycle::End(bool natural) noexcept {
  if (!active_)
    return;
  // Keep the anchor (row 0's input) and every accepted draft. A natural end
  // emits one more token whose row is not cached: it becomes the pending
  // anchor, and the next verify writes its row. Abort leaves nothing pending.
  target_.Rewind(base_ + accepted_ + 1);
  if (natural)
    policy_.Observe(draft_k_, accepted_);
  pending_ = natural;
  active_ = false;
  row_ = 0;
  accepted_ = 0;
  draft_k_ = 0;
}

bool SpecCycle::Start(std::uint32_t anchor, std::string* error_msg) {
  // The anchor is the last committed token. After a natural end it is the
  // pending emitted token, uncached at Position(), so base = Position(). The
  // first cycle after prefill or a reused prefix has the anchor cached at
  // Position()-1: rewind onto it so the verify pass re-appends it as row 0.
  // The drafter needs residuals for [0, base) only, so sync before the rewind
  // to keep its context at or ahead of base.
  if (target_.Position() == 0)
    return Fail(error_msg, "speculative cycle needs a committed token");
  const bool pending = pending_;
  const std::uint32_t base = pending ? target_.Position()
                                     : target_.Position() - 1;
  // The policy picks this block's length from the cycle's target depth.
  const std::uint32_t draft_count = policy_.Length(base);
  if (base + draft_count + 1 > target_.Capacity())
    return Fail(error_msg,
                "target session is too small for the next block; raise "
                "its context");
  if (!SyncContext(base, error_msg))
    return false;
  if (!pending)
    target_.Rewind(base);

  gpu::dflash::DflashDraft draft;
  if (!drafter_.Run(*context_, anchor, base, draft_count, 0.0F, {}, {}, &draft,
                    error_msg))
    return false;
  if (draft.tokens.size() != draft_count)
    return Fail(error_msg, "draft block returned the wrong proposal count");
  if (draft_transform_ != nullptr)
    for (auto& token : draft.tokens)
      token = draft_transform_(token, vocab_size_);

  // [anchor, draft_0 .. draft_{k-1}]: row 0 is the distribution after the
  // anchor, row i the distribution after draft_{i-1}.
  std::vector<std::int32_t> batch(draft_count + 1);
  batch[0] = static_cast<std::int32_t>(anchor);
  for (std::uint32_t i = 0; i < draft_count; ++i)
    batch[i + 1] = static_cast<std::int32_t>(draft.tokens[i]);
  if (!target_.VerifyRows(batch, error_msg))
    return false;

  pending_ = false;
  drafts_ = draft.tokens;
  base_ = base;
  draft_k_ = draft_count;
  row_ = 0;
  accepted_ = 0;
  active_ = true;
  stats_.draft_rounds += 1;
  stats_.draft_tokens += draft_count;
  return true;
}

bool SpecCycle::Next(std::uint32_t anchor, sampling::SamplerState& sampler,
                     std::uint32_t* token, std::string* error_msg) {
  if (token == nullptr)
    return Fail(error_msg, "token output is null");
  if (!active_ && !Start(anchor, error_msg))
    return false;

  // Row `row_` is conditioned on the anchor plus the accepted drafts, i.e. the
  // committed stream so far. Sampling it is the one draw for this token.
  const auto logits = target_.RowLogits(row_);
  if (logits.empty())
    return Fail(error_msg, "verify row has no logits");
  const auto sampled = static_cast<std::uint32_t>(sampler.Sample(logits));
  *token = sampled;

  // Accepted draft: the sampled token equals the proposal, and its row is
  // already in the cache. Stay active; after row k (the bonus) the next call
  // samples it.
  if (row_ < draft_k_ && sampled == drafts_[row_]) {
    ++row_;
    ++accepted_;
    stats_.draft_accepted_tokens += 1;
    return true;
  }

  // First mismatch, or the bonus row after all k drafts: emit and end. The
  // cache keeps the anchor and the accepted drafts; the emitted token is
  // pending until the next verify.
  End(true);
  return true;
}

}  // namespace gufo::models::flint::spec
