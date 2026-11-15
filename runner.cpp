#include "runner.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <gufo/ext/component_cache_resources.hpp>
#include <gufo/ext/gguf_identity.hpp>
#include <gufo/ext/gguf_reader.hpp>
#include <gufo/ext/host_budget.hpp>
#include <gufo/ext/sampling.hpp>
#include <gufo/ext/text.hpp>
#include <limits>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "continuation_adapter.hpp"
#include "hip/vendor/qwen/tokenizer.hpp"
#include "src/chat_template.hpp"
#include "src/gpu/dflash/dflash_forward.hpp"
#include "src/gpu/dflash/dflash_weights.hpp"
#include "src/gpu/engine.hpp"
#include "src/runtime/config.hpp"
#include "src/spec/cycle.hpp"
#include "src/spec/draft_policy.hpp"
#include "src/spec/target_adapter.hpp"
#include "src/tool_dialect.hpp"

namespace gufo::models::flint {
namespace {

using FlintModel = gpu::Model;
using FlintSession = gpu::Session;
using ext::TextRunnerToken;

constexpr std::string_view kFlintStateAbi = "flint-rocm-session-v2";
// Per-layer K/V append rows plus a metadata/logits private state. A v1
// single-blob disk cache must not load into this layout.
constexpr std::string_view kFlintPayloadLayout = "flint-kv-rows-v1";

// A retained snapshot is one full private state (~15.7 GiB at 130k context
// on this model). The host captures it synchronously on the request thread,
// so an unbounded budget turns the first request at long context into a
// multi-second stall (measured 14.2 s at -c 130000). Cap the budget: larger
// snapshots are refused instead of captured.
constexpr std::size_t kFlintMaxRetainedSnapshotBytes = std::size_t{8} << 30;

void Log(const std::string& line) {
  std::fprintf(stderr, "[flint] %s\n", line.c_str());
}

std::vector<std::uint8_t> FlintCompatibilityIdentity(
    const std::string& artifact_fingerprint, std::uint32_t max_context) {
  if (artifact_fingerprint.size() != 64) {
    throw std::invalid_argument(
        "Flint disk cache requires an artifact fingerprint");
  }
  std::ostringstream identity;
  identity << "schema=gufo-text-continuation-v1\n"
           << "artifact_sha256=" << artifact_fingerprint << '\n'
           << "model_kind=flint\n"
           << "context_tokens=" << max_context << '\n'
           << "chat_template=flint-glm-compiled-v1\n"
           << "state_abi=" << kFlintStateAbi << '\n'
           << "payload_layout=" << kFlintPayloadLayout << '\n';
  const auto canonical = identity.str();
  return {canonical.begin(), canonical.end()};
}

std::vector<std::uint8_t> ComponentIdentity(const FlintModel* model,
                                            std::uint32_t max_context) {
  std::ostringstream identity;
  identity << "schema=gufo-process-text-continuation-v1\n"
           << "model_kind=flint\n"
           // Without a validated artifact fingerprint the loaded model instance
           // is the identity; durable caches require the fingerprint path.
           << "model_instance=" << model << '\n'
           << "context_tokens=" << max_context << '\n'
           << "state_abi=" << kFlintStateAbi << '\n'
           << "payload_layout=" << kFlintPayloadLayout << '\n';
  const auto canonical = identity.str();
  return {canonical.begin(), canonical.end()};
}

std::vector<std::int32_t> FlintEngineTokens(
    std::span<const TextRunnerToken> tokens) {
  std::vector<std::int32_t> converted;
  converted.reserve(tokens.size());
  for (const auto token : tokens) {
    if (token > static_cast<TextRunnerToken>(
                    std::numeric_limits<std::int32_t>::max())) {
      throw std::invalid_argument("Flint token ID exceeds engine range");
    }
    converted.push_back(static_cast<std::int32_t>(token));
  }
  return converted;
}
// Provider sampling defaults for the prompt/chat drivers and the HTTP path.
// Flint has no published preset; this mirrors a moderate general-purpose
// setting and is overridden by any explicit CLI or per-request value. Held as
// one function so the descriptor and the provider probe stay in agreement.
sampling::TextSamplingDefaults FlintSamplingDefaults() {
  sampling::TextSamplingValues unspecified;
  unspecified.temperature = 0.6F;
  unspecified.top_p = 0.95F;
  sampling::TextSamplingValues thinking_on = unspecified;
  thinking_on.top_k = 20;
  sampling::TextSamplingValues thinking_off = unspecified;
  thinking_off.top_k = 20;
  return {unspecified, thinking_on, thinking_off};
}

// Speculation mode, from the options only. `null` is a test-only mode (see
// CreateFlintNullDrafterRunnerForTest): the drafter is a stub whose proposals
// are deliberately wrong, so the cycle runs and nothing is ever accepted.
enum class FlintSpecMode { kOff, kDFlash, kNull };

class FlintTextRunnerState final : public ext::TextRunnerState {
public:
  // Component-cache slot: the session is owned by the adapter's slot, which
  // outlives this state through the retained adapter/slot handles. Invalidate
  // routes through the adapter so the slot resets its session and guard.
  FlintTextRunnerState(FlintSession& session, cache::Adapter& adapter,
                       cache::Slot& slot)
      : session_(&session), adapter_(&adapter), slot_(&slot) {}

  void Invalidate() noexcept override {
    position_ = 0;
    last_token_.reset();
    emitted_first_ = false;
    uncached_ = false;
    if (cycle_)
      cycle_->Reset();
    (void)adapter_->Invalidate(*slot_);
  }

  [[nodiscard]] FlintSession& session() const { return *session_; }
  [[nodiscard]] std::size_t position() const noexcept { return position_; }
  void set_position(std::size_t position) noexcept { position_ = position; }

  // The last token committed to the stream (prompt tail or emitted token). It
  // is cached at position()-1 once Advance has run, and is the speculative
  // cycle's anchor.
  [[nodiscard]] const std::optional<std::int32_t>& last_token() const noexcept {
    return last_token_;
  }
  void set_last_token(std::optional<std::int32_t> token) noexcept {
    last_token_ = token;
  }
  // True once the first token after prefill has been emitted. Until then the
  // plain decode path samples the prompt's logits; the cycle starts after.
  [[nodiscard]] bool emitted_first() const noexcept { return emitted_first_; }
  void set_emitted_first(bool value) noexcept { emitted_first_ = value; }
  // True when the last advanced token is committed but its row is not cached
  // (the speculative cycle's pending emitted token). The session holds
  // position()-1 rows until the next verify caches it.
  [[nodiscard]] bool uncached() const noexcept { return uncached_; }
  void set_uncached(bool value) noexcept { uncached_ = value; }

  // Speculative cycle bound to this slot's session; null when speculation is
  // off. Owned here so its target adapter lives as long as the session.
  [[nodiscard]] spec::SpecCycle* cycle() const noexcept { return cycle_.get(); }
  [[nodiscard]] spec::TargetAdapter* target() const noexcept {
    return target_.get();
  }
  void AttachCycle(std::unique_ptr<spec::TargetAdapter> target,
                   std::unique_ptr<spec::SpecCycle> cycle) {
    target_ = std::move(target);
    cycle_ = std::move(cycle);
  }

private:
  FlintSession* session_{nullptr};
  cache::Adapter* adapter_{nullptr};
  cache::Slot* slot_{nullptr};
  std::size_t position_{0};
  std::optional<std::int32_t> last_token_;
  bool emitted_first_{false};
  bool uncached_{false};
  std::unique_ptr<spec::TargetAdapter> target_;
  std::unique_ptr<spec::SpecCycle> cycle_;
};

FlintTextRunnerState& RequireFlintState(ext::TextRunnerState& state) {
  auto* flint = dynamic_cast<FlintTextRunnerState*>(&state);
  if (flint == nullptr)
    throw std::logic_error("text runner state is not Flint");
  return *flint;
}

const FlintTextRunnerState& RequireFlintState(
    const ext::TextRunnerState& state) {
  const auto* flint = dynamic_cast<const FlintTextRunnerState*>(&state);
  if (flint == nullptr)
    throw std::logic_error("text runner state is not Flint");
  return *flint;
}

// Maps the public chat role onto the role names Flint's template expects.
std::string FlintRoleName(ext::ChatRole role) {
  switch (role) {
    case ext::ChatRole::kSystem:
      return "system";
    case ext::ChatRole::kDeveloper:
      return "developer";
    case ext::ChatRole::kUser:
      return "user";
    case ext::ChatRole::kAssistant:
      return "assistant";
    case ext::ChatRole::kTool:
      return "tool";
  }
  throw std::invalid_argument("unknown chat role");
}

class FlintTextRunner final : public ext::HipTextModelRunner {
public:
  static constexpr std::size_t kPrefillChunk = 8192;
  static constexpr std::size_t kWarmupTokens = 9000;

  FlintTextRunner(
      std::shared_ptr<FlintModel> model,
      std::unique_ptr<gufo::vendor::qwen::tokenization::QwenTokenizer>
          tokenizer,
      std::vector<std::int32_t> stop_tokens, std::uint32_t max_context,
      std::string artifact_fingerprint, FlintSpecMode spec_mode,
      std::unique_ptr<gpu::dflash::DflashForward> drafter,
      std::vector<std::uint32_t> taps, std::uint32_t hidden_width,
      spec::DraftPolicy draft_policy, std::uint32_t vocab_size)
      : model_(std::move(model)),
        tokenizer_(std::move(tokenizer)),
        stop_tokens_(std::move(stop_tokens)),
        max_context_(max_context),
        spec_mode_(spec_mode),
        drafter_(std::move(drafter)),
        taps_(std::move(taps)),
        hidden_width_(hidden_width),
        draft_policy_(draft_policy),
        vocab_size_(vocab_size) {
    // Durable caches require the validated artifact fingerprint; without it the
    // loaded model instance is the identity, as for the LFM3 continuation.
    if (!artifact_fingerprint.empty()) {
      if (artifact_fingerprint.size() != 64 ||
          artifact_fingerprint.find_first_not_of("0123456789abcdef") !=
              std::string::npos)
        throw std::invalid_argument("invalid Flint artifact fingerprint");
      component_identity_ =
          FlintCompatibilityIdentity(artifact_fingerprint, max_context_);
    } else {
      component_identity_ = ComponentIdentity(model_.get(), max_context_);
    }
    // Field diagnostics: which cache layout this build speaks. Disk manifests
    // store only a digest of the identity, so this line is the only readable
    // record of it.
    Log("event=cache_identity state_abi=" + std::string(kFlintStateAbi) +
        " payload_layout=" + std::string(kFlintPayloadLayout) +
        " context_tokens=" + std::to_string(max_context_) +
        " identity_bytes=" + std::to_string(component_identity_.size()));
    Warmup();
    SnapshotSelfTest();
  }

  // FLINT_SNAPSHOT_SELFTEST=1: with throw-away sessions, checks that a snapshot
  // restores the state exactly. A 9000-token prefix is longer than the ring of
  // a sliding-window layer, so the ring has wrapped when the snapshot is taken.
  // The snapshot is restored into the session that has since moved on and into
  // a fresh one; each must give the logits of the original state, then the
  // logits the original run gave for the same continuation, bit for bit.
  void SnapshotSelfTest() const {
    const char* enabled = std::getenv("FLINT_SNAPSHOT_SELFTEST");
    if (enabled == nullptr || enabled[0] == '\0' || enabled[0] == '0')
      return;
    try {
      constexpr std::uint32_t kPrefix = 9000;
      constexpr std::uint32_t kBlock = 1200;
      constexpr std::uint32_t kSteps = 24;
      constexpr std::uint32_t kCapacity = kPrefix + kBlock + kSteps + 16;
      std::string error;
      auto session = model_->CreateSession(kCapacity, &error);
      auto fresh = model_->CreateSession(kCapacity, &error);
      if (session == nullptr || fresh == nullptr) {
        Log("event=snapshot_selftest result=skipped reason=" + error);
        return;
      }
      const auto token = [](std::uint32_t i) {
        return static_cast<std::int32_t>((i * 2654435761U) % 100000U + 1000U);
      };
      std::vector<std::int32_t> prefix(kPrefix);
      std::vector<std::int32_t> block(kBlock);
      for (std::uint32_t i = 0; i < kPrefix; ++i)
        prefix[i] = token(i);
      for (std::uint32_t i = 0; i < kBlock; ++i)
        block[i] = token(kPrefix + i);
      struct Tail {
        std::vector<float> after_block;
        std::vector<float> after_steps;
      };
      const auto copy = [](std::span<const float> logits) {
        return std::vector<float>(logits.begin(), logits.end());
      };
      const auto run_tail = [&](FlintSession& target, Tail* out) {
        if (!target.Append(block, &error))
          return false;
        out->after_block = copy(target.Logits());
        for (std::uint32_t k = 0; k < kSteps; ++k) {
          const std::int32_t next = token(kPrefix + kBlock + k);
          if (!target.Append({&next, 1}, &error))
            return false;
        }
        out->after_steps = copy(target.Logits());
        return true;
      };
      const auto same = [](const std::vector<float>& a,
                           const std::vector<float>& b) {
        return a.size() == b.size() &&
               std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
      };
      if (!session->Append(prefix, &error)) {
        Log("event=snapshot_selftest result=FAIL step=prefix reason=" + error);
        return;
      }
      const std::vector<float> at_snapshot = copy(session->Logits());
      std::vector<std::uint8_t> snapshot(session->SnapshotBytes());
      if (!session->SaveSnapshot(snapshot, &error)) {
        Log("event=snapshot_selftest result=FAIL step=save reason=" + error);
        return;
      }
      Tail original;
      if (!run_tail(*session, &original)) {
        Log("event=snapshot_selftest result=FAIL step=continue reason=" + error);
        return;
      }
      bool all_ok = true;
      for (const bool use_fresh : {false, true}) {
        FlintSession& target = use_fresh ? *fresh : *session;
        Tail restored;
        const bool ok =
            target.RestoreSnapshot(snapshot, &error) &&
            target.Position() == kPrefix &&
            same(copy(target.Logits()), at_snapshot) &&
            run_tail(target, &restored) &&
            same(restored.after_block, original.after_block) &&
            same(restored.after_steps, original.after_steps);
        all_ok = all_ok && ok;
        Log(std::string("event=snapshot_selftest result=") +
            (ok ? "pass" : "FAIL") +
            " target=" + (use_fresh ? "fresh_session" : "same_session") +
            " position=" + std::to_string(target.Position()) +
            (ok ? "" : " reason=" + error));
      }
      Log("event=snapshot_selftest_summary result=" +
          std::string(all_ok ? "pass" : "FAIL") +
          " prefix_tokens=" + std::to_string(kPrefix) +
          " snapshot_bytes=" + std::to_string(snapshot.size()) +
          " state_bytes=" + std::to_string(session->AllocatedBytes()));
    } catch (const std::exception& e) {
      Log(std::string("event=snapshot_selftest result=FAIL reason=") + e.what());
    }
  }

  void Warmup() const {
    const char* disabled = std::getenv("GUFO_WARMUP");
    if (disabled != nullptr && disabled[0] == '0')
      return;
    try {
      const std::size_t room = max_context_ > 128 ? max_context_ - 128 : 0;
      const std::size_t length = std::min<std::size_t>(kWarmupTokens, room);
      if (length < 1024)
        return;
      const auto started = std::chrono::steady_clock::now();
      std::string error;
      auto session = model_->CreateSession(
          static_cast<std::uint32_t>(length + 16), &error);
      if (session == nullptr) {
        Log("event=warmup_skipped reason=" + error);
        return;
      }
      const auto unit = tokenizer_->Encode(
          "The quick brown fox jumps over the lazy dog while the engine warms "
          "its kernels. ");
      if (unit.empty())
        return;
      std::vector<std::int32_t> prompt;
      prompt.reserve(length);
      for (std::size_t i = 0; i < length; ++i)
        prompt.push_back(static_cast<std::int32_t>(unit[i % unit.size()]));
      if (!session->Append(prompt, &error)) {
        Log("event=warmup_failed reason=" + error);
        return;
      }
      for (int step = 0; step < 4; ++step) {
        const std::int32_t next = 1;
        if (!session->Append({&next, 1}, &error))
          break;
      }
      const auto elapsed = std::chrono::duration<double>(
                               std::chrono::steady_clock::now() - started)
                               .count();
      Log("event=warmup_completed tokens=" + std::to_string(length) +
          " elapsed_s=" + std::to_string(elapsed));
    } catch (const std::exception& failure) {
      Log(std::string("event=warmup_failed reason=") + failure.what());
    }
  }

  [[nodiscard]] ext::TextRunnerDescriptor Descriptor() const override {
    return {
        .model_id = "Flint",
        .state_abi = std::string(kFlintStateAbi),
        .max_context = max_context_,
        .capabilities =
            ext::TextRunnerCapabilities{
                .incremental_prefill = true,
                .final_token_advance_required = false,
                .incremental_text_is_exact = true,
                .prefix_reuse = true,
            },
        .sampling_defaults = FlintSamplingDefaults(),
        // API 0.3: required. Flint prefills in unsplit kPrefillChunk passes
        // and ignores SetPrefillYield (allowed).
        .scheduling =
            ext::TextSchedulingProfile{
                .pass_tokens = kPrefillChunk,
                .max_layer_groups = 1,
                .max_yield_decode_width = 1,
            },
    };
  }

  [[nodiscard]] ext::TextRunnerResourceClaim ResourceClaim() const override {
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    std::optional<std::size_t> capacity;
    if (hipMemGetInfo(&free_bytes, &total_bytes) == hipSuccess)
      capacity = free_bytes;
    // Sessions are created at max_context_ (continuation_adapter.cpp); the
    // verify padding is only in the KV estimate below, as before.
    std::size_t resident = model_->ResidentBytes();
    std::size_t per_request =
        model_->SessionBytes(max_context_ + FlintSession::kMaxVerifyRows);
    if (drafter_) {
      // Speculation. Per session: the hidden ring and the drafter's K/V ring
      // (each slot's DflashContext). Resident, shared by all sessions: the
      // drafter's weights and block workspace, and the injection scratch.
      const std::size_t hidden_ring =
          std::size_t{std::min(FlintSession::kHiddenRingRows, max_context_)} *
          taps_.size() * hidden_width_ * sizeof(float);
      resident += drafter_->AllocatedBytes() + drafter_->ContextScratchPendingBytes();
      per_request += hidden_ring + drafter_->ContextRingBytes();
    }
    return {
        .resident_weights_bytes = resident,
        .state_capacity_bytes = capacity,
        .per_request_state_bytes = per_request,
        .temporary_scratch_bytes = 0,
        .retained_snapshot_capacity_bytes =
            std::min(ext::HostSnapshotBudgetBytes(),
                     kFlintMaxRetainedSnapshotBytes),
        .retained_snapshot_ceiling_bytes =
            std::min(ext::HostSnapshotCeilingBytes(),
                     kFlintMaxRetainedSnapshotBytes),
        .requires_device_runtime_lock = true,
    };
  }

  [[nodiscard]] std::vector<ext::TextExecutionPlan> SupportedPlans()
      const override {
    return {{.kind = ext::TextExecutionPlanKind::kSerial, .physical_width = 1}};
  }

  [[nodiscard]] std::unique_ptr<ext::ComponentCacheResources>
  CreateComponentCacheResources(cache::ResourceLedger& ledger,
                                std::size_t ram) const override {
    (void)ram;
    return hip::CreateHipComponentCacheResources(
        std::make_unique<FlintContinuationAdapter>(model_, max_context_,
                                                   component_identity_),
        ledger,
        hip::ComponentCacheHooks{
            [] {
              return std::min(ext::HostSnapshotCeilingBytes(),
                              kFlintMaxRetainedSnapshotBytes);
            },
            [](std::string_view) {},
            [](std::string_view) {},
        });
  }

  [[nodiscard]] std::unique_ptr<ext::TextRunnerState> BindComponentState(
      cache::Adapter& adapter, cache::Slot& slot) const override {
    auto& flint = dynamic_cast<FlintContinuationAdapter&>(adapter);
    auto state = std::make_unique<FlintTextRunnerState>(
        flint.GetSession(slot), adapter, slot);
    if (spec_mode_ != FlintSpecMode::kOff) {
      std::string error;
      auto target = spec::TargetAdapter::Create(state->session(), taps_,
                                                hidden_width_, &error);
      if (!target)
        throw std::runtime_error("Flint target adapter failed: " + error);
      // Each slot owns its drafter context (K/V ring and frontier) and its own
      // acceptance history, so sessions never see each other's target context.
      // The drafter's weights and block workspace stay shared.
      std::string context_error;
      auto context = drafter_->CreateContext(&context_error);
      if (!context)
        throw std::runtime_error("Flint drafter context failed: " +
                                 context_error);
      auto cycle = std::make_unique<spec::SpecCycle>(
          *target, *drafter_, std::move(context), draft_policy_);
      if (spec_mode_ == FlintSpecMode::kNull)
        cycle->set_draft_transform(&NullDraftTransform, vocab_size_);
      state->AttachCycle(std::move(target), std::move(cycle));
    }
    return state;
  }

  // The `null` drafter: every proposal is shifted to a different id inside the
  // vocabulary. The verifier then rejects it on nearly every row, which is the
  // point of the test; the wrap keeps the id in range.
  static std::uint32_t NullDraftTransform(std::uint32_t token,
                                          std::uint32_t vocab_size) {
    return vocab_size == 0 ? token : (token + 1U) % vocab_size;
  }

  void ReconcileComponentState(ext::TextRunnerState& state,
                               std::size_t position) const override {
    auto& flint = RequireFlintState(state);
    flint.set_position(position);
    // The adapter's Validate owns the session position; a mismatch means cache and runner disagree on the frontier.
    if (static_cast<std::size_t>(flint.session().Position()) != position)
      throw std::logic_error(
          "Flint restored session position disagrees with the checkpoint");
    flint.set_uncached(false);
    // The reused prefix is cached through `position`. The drafter's context
    // cannot be re-fed from 0 (the hidden ring holds only recent rows), so it
    // restarts at `position`; this also drops any pending row or active cycle.
    if (flint.cycle() != nullptr)
      flint.cycle()->Reset(static_cast<std::uint32_t>(position));
  }

  [[nodiscard]] std::shared_ptr<const sampling::ConstraintVocabulary>
  BuildConstraintVocabulary() const override {
    return std::make_shared<const sampling::ConstraintVocabulary>(
        tokenizer_->GetVocabSize(), [this](std::uint32_t id) {
          return sampling::ConstraintVocabulary::Piece{
              tokenizer_->DecodeTokenCopy(id), IsStop(id)};
        });
  }

  [[nodiscard]] std::vector<TextRunnerToken> Tokenize(
      std::string_view text) const override {
    return tokenizer_->Encode(text);
  }

  // The reasoning budget tokenizes its delimiters in isolation and requires
  // each to be exactly one control token. Flint's framing is the special-token
  // pair the chat template names, so special-token parsing is on here even
  // though ordinary prompt text does not need it.
  [[nodiscard]] std::vector<TextRunnerToken> TokenizeControl(
      std::string_view text) const override {
    gufo::vendor::qwen::tokenization::TokenizerOptions options;
    options.parse_special_tokens = true;
    return tokenizer_->Encode(text, options);
  }

  [[nodiscard]] std::optional<std::vector<TextRunnerToken>> RenderAndTokenize(
      const ext::ChatRequest& request) const override {
    // tool_choice none leaves the declared tools out of the prompt, so the
    // model does not try to call them.
    const bool with_tools =
        !request.tools.empty() &&
        request.tool_choice != ext::ChatRequest::ToolChoice::kNone;
    std::vector<ChatMessage> messages;
    messages.reserve(request.messages.size());
    for (const auto& message : request.messages) {
      if (!message.images.empty())
        throw std::invalid_argument("Flint does not support image input");
      ChatMessage rendered{FlintRoleName(message.role),
                           message.content + message.framing_suffix,
                           message.thought};
      for (const auto& call : message.tool_calls) {
        ::gufo::models::flint::ToolCall rendered_call;
        rendered_call.name = call.name;
        for (const auto& argument : call.arguments)
          rendered_call.arguments.push_back(
              {argument.name, argument.value, argument.is_string});
        rendered.tool_calls.push_back(std::move(rendered_call));
      }
      messages.push_back(std::move(rendered));
    }
    std::vector<::gufo::models::flint::ChatTool> tools;
    std::vector<std::pair<std::string, std::string>> schemas;
    if (with_tools) {
      for (const auto& tool : request.tools) {
        tools.push_back({tool.definition_json, tool.name, tool.description,
                         tool.parameters_json});
        schemas.emplace_back(tool.name, tool.parameters_json);
      }
    }
    // The parser reads the model's calls without the request: it learns which
    // parameters are strings from the tools of the prompts rendered here.
    RememberToolSchemas(schemas);
    gufo::vendor::qwen::tokenization::TokenizerOptions options;
    options.parse_special_tokens = true;
    // Qualified: the inherited TextModelRunner::RenderPrompt member would hide
    // the namespace-scope overload under unqualified lookup.
    return tokenizer_->Encode(
        ::gufo::models::flint::RenderPrompt(messages, tools), options);
  }

  [[nodiscard]] std::string Decode(
      std::span<const TextRunnerToken> tokens) const override {
    return tokenizer_->Decode(tokens);
  }

  void PreparePrefixReuse(
      ext::TextRunnerState& state,
      std::span<const TextRunnerToken> prefix) const override {
    if (RequireFlintState(state).position() != prefix.size())
      throw std::logic_error("Flint reused prefix does not match checkpoint");
  }

  [[nodiscard]] ext::TextPrefillStep Prefill(
      ext::TextRunnerState& state, std::span<const TextRunnerToken> prompt,
      std::size_t offset, std::size_t max_input_tokens) const override {
    auto& flint = RequireFlintState(state);
    if (offset != flint.position() || flint.uncached())
      throw std::logic_error("Flint prefill offset does not match state");
    if (offset >= prompt.size())
      throw std::logic_error("Flint prefill has no remaining input");
    // Same chunking with and without speculation: the multi-token prefill
    // result depends on chunk size, so spec-on must not chunk differently.
    static_assert(
        FlintSession::kHiddenRingRows >= kPrefillChunk,
        "hidden ring must hold a whole prefill chunk when speculation is on");
    const std::size_t limit = kPrefillChunk;
    const std::size_t consumed = std::min<std::size_t>(
        {max_input_tokens, prompt.size() - offset, limit});
    const auto chunk = FlintEngineTokens(prompt.subspan(offset, consumed));
    std::string error;
    if (!flint.session().Append(chunk, &error)) {
      flint.Invalidate();
      throw std::runtime_error("Flint prefill failed: " + error);
    }
    flint.set_position(offset + consumed);
    if (offset == 0) {
      flint.set_emitted_first(false);
      if (flint.cycle() != nullptr)
        flint.cycle()->Reset();
    }
    if (flint.cycle() != nullptr &&
        !flint.cycle()->SyncContext(flint.session().Position(), &error)) {
      flint.Invalidate();
      throw std::runtime_error("Flint drafter context failed: " + error);
    }
    flint.set_last_token(
        static_cast<std::int32_t>(prompt[offset + consumed - 1]));
    return {.consumed_tokens = consumed,
            .decode_ready = offset + consumed == prompt.size()};
  }

  [[nodiscard]] ext::TextDecodeSelection SelectNext(
      ext::TextRunnerState& state,
      sampling::SamplerState& sampler) const override {
    auto& flint = RequireFlintState(state);
    if (flint.position() >= max_context_)
      return {.stop = true, .piece = {}, .logprobs = std::nullopt};
    std::int32_t token = 0;
    // A cycle Next below verifies the pending row, so the session covers the
    // committed stream again.
    flint.set_uncached(false);
    if (flint.cycle() != nullptr && flint.emitted_first()) {
      // Speculative path: the cycle draws the token from the verify rows (or
      // starts a cycle from the last committed token) with the same sampler.
      std::string error;
      std::uint32_t sampled = 0;
      if (!flint.cycle()->Next(flint.last_token().value_or(0), sampler,
                               &sampled, &error)) {
        flint.Invalidate();
        throw std::runtime_error("Flint speculative decode failed: " + error);
      }
      token = static_cast<std::int32_t>(sampled);
    } else {
      // The first token after prefill is a plain decode: its logits are the
      // prompt's last row, which is what RunAcceptance samples.
      const auto logits = flint.session().Logits();
      if (logits.empty())
        throw std::runtime_error("Flint token selection has no logits");
      token = static_cast<std::int32_t>(sampler.Sample(logits));
    }
    if (flint.stop_at_eos() && IsStop(token)) {
      // EOS mid-block: drop the rows past the committed stream before the
      // checkpoint reads the session.
      if (flint.cycle() != nullptr)
        flint.cycle()->Abort();
      return {.stop = true, .token = 0, .piece = {}, .logprobs = std::nullopt};
    }
    return {.stop = false,
            .token = static_cast<TextRunnerToken>(token),
            .piece = std::string(tokenizer_->DecodeToken(token)),
            .logprobs = std::nullopt};
  }

  void Advance(ext::TextRunnerState& state,
               TextRunnerToken token) const override {
    auto& flint = RequireFlintState(state);
    const std::int32_t input = FlintEngineTokens({&token, 1}).front();
    // An accepted draft is already cached at its row: the verify pass wrote it.
    // After a cycle's natural end the emitted token's row is left pending: the
    // next cycle's verify writes it as row 0. Only the first token after
    // prefill (no cycle) is appended here.
    const spec::SpecCycle* cycle = flint.cycle();
    const bool cycle_cached = cycle != nullptr && cycle->active();
    const bool deferred = cycle != nullptr && !cycle_cached && cycle->pending();
    if (!cycle_cached && !deferred) {
      std::string error;
      if (!flint.session().Append({&input, 1}, &error)) {
        flint.Invalidate();
        throw std::runtime_error("Flint decode failed: " + error);
      }
    }
    flint.set_uncached(deferred);
    flint.set_position(flint.position() + 1);
    flint.set_last_token(input);
    flint.set_emitted_first(true);
  }

  [[nodiscard]] std::optional<ext::TextDecodeSelection> PreviewFirstToken(
      ext::TextRunnerState& state,
      sampling::SamplerState& sampler) const override {
    // Preview must not start a cycle or consume RNG beyond the plain sample the
    // real first decode repeats: it samples the prompt's logits on a copy of
    // the sampler and never touches the cycle.
    auto& flint = RequireFlintState(state);
    if (flint.emitted_first())
      return std::nullopt;
    const auto logits = flint.session().Logits();
    if (logits.empty())
      return std::nullopt;
    auto preview = sampler;
    const auto token = static_cast<std::int32_t>(preview.Sample(logits));
    if (flint.stop_at_eos() && IsStop(token))
      return ext::TextDecodeSelection{.stop = true,
                                      .token = 0,
                                      .piece = {},
                                      .logprobs = std::nullopt};
    return ext::TextDecodeSelection{
        .stop = false,
        .token = static_cast<TextRunnerToken>(token),
        .piece = std::string(tokenizer_->DecodeToken(token)),
        .logprobs = std::nullopt};
  }

  // Reports the cycle's draft counters for this step: the difference over the
  // call, so the host's per-step stats sum to the cycle's totals.
  [[nodiscard]] ext::TextDecodeStep DecodeStep(
      ext::TextRunnerState& state, std::size_t max_tokens,
      sampling::SamplerState& sampler) const override {
    auto& flint = RequireFlintState(state);
    const spec::CycleStats before =
        flint.cycle() != nullptr ? flint.cycle()->stats() : spec::CycleStats{};
    ext::TextDecodeStep step =
        TextModelRunner::DecodeStep(state, max_tokens, sampler);
    if (flint.cycle() != nullptr) {
      const auto& after = flint.cycle()->stats();
      step.draft_rounds = after.draft_rounds - before.draft_rounds;
      step.draft_tokens = after.draft_tokens - before.draft_tokens;
      step.draft_accepted_tokens =
          after.draft_accepted_tokens - before.draft_accepted_tokens;
    }
    return step;
  }

  [[nodiscard]] std::size_t CheckpointPosition(
      const ext::TextRunnerState& state) const override {
    // The checkpoint is the committed tokens whose rows the session holds. A
    // deferred emitted token (uncached) is committed but not yet cached, so it
    // is excluded; the snapshot payload records the same session position.
    const auto& flint = RequireFlintState(state);
    return flint.position() - (flint.uncached() ? 1 : 0);
  }

  void PrepareCancellation(ext::TextRunnerState& state) const override {
    auto& flint = RequireFlintState(state);
    if (flint.cycle() != nullptr)
      flint.cycle()->Abort();
  }

  // Flint writes its calls in the GLM markup, read by the dialect that
  // provider registration adds (src/tool_dialect.cpp).
  [[nodiscard]] std::string ToolDialectId() const override {
    return std::string(kToolDialectId);
  }

  [[nodiscard]] ext::ReasoningTags ReasoningBudgetTags() const override {
    // Flint emits the plain XML reasoning framing, the same literals its
    // chat template uses (kThinkOpen/kThinkClose).
    return {"<think>", "</think>", {"</think>"}};
  }

  [[nodiscard]] ext::ReasoningMarkup ReasoningDelimiters() const override {
    // Literal delimiters for the response parsers. open must not begin with
    // whitespace; the leading '<' satisfies that and matches the model bytes.
    return {"<think>", "</think>"};
  }

private:
  [[nodiscard]] bool IsStop(std::int32_t token) const {
    return std::find(stop_tokens_.begin(), stop_tokens_.end(), token) !=
           stop_tokens_.end();
  }

  std::shared_ptr<FlintModel> model_;
  std::unique_ptr<gufo::vendor::qwen::tokenization::QwenTokenizer> tokenizer_;
  std::vector<std::int32_t> stop_tokens_;
  std::uint32_t max_context_;
  cache::Identity component_identity_;
  FlintSpecMode spec_mode_{FlintSpecMode::kOff};
  // One drafter per runner: weights and block workspace, shared. Each bound
  // session owns its DflashContext (see BindComponentState).
  std::unique_ptr<gpu::dflash::DflashForward> drafter_;
  std::vector<std::uint32_t> taps_;
  std::uint32_t hidden_width_{0};
  spec::DraftPolicy draft_policy_{ext::TextDFlashDraftPolicy::kAdaptive, 1, 1};
  std::uint32_t vocab_size_{0};
};

}  // namespace

std::optional<sampling::TextSamplingDefaults> FlintTextSamplingDefaults(
    const core::GgufReader&) {
  return FlintSamplingDefaults();
}

namespace {

// Shared factory body. `null_drafter` is set only by the test entry point.
std::shared_ptr<ext::TextModelRunner> CreateFlintRunner(
    const ext::TextRunnerOptions& options, bool null_drafter,
    std::string* error) {
  try {
    if (options.model_path.empty()) {
      if (error)
        *error = "Flint requires a model path";
      return nullptr;
    }
    // Speculation comes only from the options: the host's speculative backend.
    FlintSpecMode spec_mode = FlintSpecMode::kOff;
    if (null_drafter) {
      spec_mode = FlintSpecMode::kNull;
    } else if (options.speculative.backend ==
               ext::TextSpeculativeBackend::kDFlash) {
      spec_mode = FlintSpecMode::kDFlash;
    } else if (options.speculative.backend !=
               ext::TextSpeculativeBackend::kDisabled) {
      if (error)
        *error = "Flint supports only the DFlash2 speculative backend";
      return nullptr;
    }
    if (spec_mode != FlintSpecMode::kOff &&
        options.speculative.draft_model_path.empty()) {
      if (error)
        *error =
            "speculation enabled but no drafter given: pass --dflash-model "
            "<gguf>";
      return nullptr;
    }
    if (!options.vision_model_path.empty()) {
      if (error)
        *error = "Flint does not support --mmproj";
      return nullptr;
    }

    std::string open_error;
    std::shared_ptr<const core::GgufReader> reader = options.reader;
    if (!reader) {
      auto opened = core::GgufReader::OpenFile(options.model_path, &open_error);
      if (!opened) {
        if (error)
          *error = "Flint GGUF open failed: " + open_error;
        return nullptr;
      }
      reader = std::move(opened);
    }

    std::string config_error;
    const auto config = ParseConfig(*reader, &config_error);
    if (!config.has_value()) {
      if (error)
        *error = "Flint config failed: " + config_error;
      return nullptr;
    }

    std::string tokenizer_error;
    auto tokenizer =
        gufo::vendor::qwen::tokenization::QwenTokenizer::CreateFromGguf(
            *reader, &tokenizer_error);
    if (tokenizer == nullptr) {
      if (error)
        *error = "Flint tokenizer failed: " + tokenizer_error;
      return nullptr;
    }

    // Stop ids: every EOS-class key the artifact names, plus any EOG list.
    std::vector<std::int32_t> stop_tokens;
    for (const char* key :
         {"tokenizer.ggml.eos_token_id", "tokenizer.ggml.eot_token_id",
          "tokenizer.ggml.eom_token_id"}) {
      if (const auto id = reader->GetMetadataUint32(key))
        stop_tokens.push_back(static_cast<std::int32_t>(*id));
    }
    if (const auto* meta =
            reader->FindMetadata("tokenizer.ggml.eog_token_ids")) {
      if (const auto* ids =
              std::get_if<std::vector<std::int64_t>>(&meta->value)) {
        for (const auto id : *ids)
          stop_tokens.push_back(static_cast<std::int32_t>(id));
      } else if (const auto* ids =
                     std::get_if<std::vector<std::uint64_t>>(&meta->value)) {
        for (const auto id : *ids)
          stop_tokens.push_back(static_cast<std::int32_t>(id));
      }
    }

    std::string load_error;
    auto model = FlintModel::Load(options.model_path, {}, &load_error);
    if (model == nullptr) {
      if (error)
        *error = "Failed to create Flint model: " + load_error;
      return nullptr;
    }

    std::uint32_t max_context = options.max_context;
    if (max_context == 0)
      max_context = config->context_length;
    if (max_context < 2 || max_context > config->context_length) {
      if (error)
        *error = "Flint context exceeds the loaded model context";
      return nullptr;
    }

    std::string fingerprint = options.disk_cache.model_artifact_fingerprint;
    if (!options.disk_cache.directory.empty() && fingerprint.empty())
      fingerprint = core::GgufIdentityHex(*reader);

    std::unique_ptr<gpu::dflash::DflashForward> drafter;
    std::vector<std::uint32_t> taps;
    std::optional<spec::DraftPolicy> draft_policy;
    if (spec_mode != FlintSpecMode::kOff) {
      const std::string& draft_path = options.speculative.draft_model_path;
      auto draft_reader = core::GgufReader::OpenFile(draft_path, &open_error);
      if (!draft_reader) {
        if (error)
          *error = "Flint drafter GGUF open failed: " + open_error;
        return nullptr;
      }
      auto weights = gpu::dflash::DflashWeights::LoadFromGguf(*draft_reader,
                                                              &open_error);
      if (!weights.has_value()) {
        if (error)
          *error = "Flint drafter load failed: " + open_error;
        return nullptr;
      }
      const auto& draft_config = weights->config;
      taps = draft_config.target_layer_ids;
      // max 0 means the drafter's full block. The policy clamps both bounds to
      // 1..block_size-1 and orders min <= max.
      const std::uint32_t block_max = draft_config.block_size - 1U;
      const std::uint32_t max_draft =
          options.speculative.max_draft_tokens == 0
              ? block_max
              : std::min(options.speculative.max_draft_tokens, block_max);
      draft_policy.emplace(options.speculative.dflash_policy,
                           options.speculative.min_draft_tokens, max_draft);
      const gpu::TiedWeights tied = model->Tied();
      if (tied.token_embedding == nullptr || tied.output == nullptr) {
        if (error)
          *error = "Flint target exposes no tied weights for the drafter";
        return nullptr;
      }
      const gpu::dflash::DflashBorrowed borrowed{
          .token_embedding = tied.token_embedding,
          .token_embedding_type = tied.token_embedding_type,
          .output = tied.output,
          .output_type = tied.output_type};
      drafter = gpu::dflash::DflashForward::Create(*weights, borrowed,
                                                   max_context, &open_error);
      if (!drafter) {
        if (error)
          *error = "Flint drafter forward failed: " + open_error;
        return nullptr;
      }
    }

    return std::make_shared<FlintTextRunner>(
        std::move(model), std::move(tokenizer), std::move(stop_tokens),
        max_context, std::move(fingerprint), spec_mode, std::move(drafter),
        std::move(taps), config->hidden_size,
        draft_policy.value_or(spec::DraftPolicy(
            ext::TextDFlashDraftPolicy::kFixed, 1, 1)),
        config->vocab_size);
  } catch (const std::exception& e) {
    if (error)
      *error = std::string("Flint runner creation failed: ") + e.what();
    return nullptr;
  }
}

}  // namespace

std::shared_ptr<ext::TextModelRunner> CreateFlintTextRunner(
    const ext::TextRunnerOptions& options, std::string* error) {
  return CreateFlintRunner(options, /*null_drafter=*/false, error);
}

std::shared_ptr<ext::TextModelRunner> CreateFlintNullDrafterRunnerForTest(
    const ext::TextRunnerOptions& options, std::string* error) {
  return CreateFlintRunner(options, /*null_drafter=*/true, error);
}

}  // namespace gufo::models::flint
