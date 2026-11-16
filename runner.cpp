#include "runner.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <atomic>
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
#include "src/gpu/dflash/dflash_forward.hpp"
#include "src/gpu/dflash/dflash_weights.hpp"
#include "hip/vendor/qwen/tokenizer.hpp"
#include "src/chat_template.hpp"
#include "src/gpu/engine.hpp"
#include "src/runtime/config.hpp"
#include "src/tool_dialect.hpp"

namespace gufo::models::flint {
namespace {

using FlintModel = gpu::Model;
using FlintSession = gpu::Session;
using ext::TextRunnerToken;

constexpr std::string_view kFlintStateAbi = "flint-rocm-session-v1";

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
           << "payload_layout=flint-rocm-row-components-v"
           << ContinuationLayout::kLayoutVersion << '\n';
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
           << "payload_layout=flint-rocm-row-components-v"
           << ContinuationLayout::kLayoutVersion << '\n';
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

class FlintTextRunnerState final : public ext::TextRunnerState {
public:
  // Component-cache slot: the session is owned by the adapter's slot, which
  // outlives this state through the retained adapter/slot handles. Invalidate
  // routes through the adapter so the slot resets its session and guard.
  // `draft` (may be null) is the slot's drafter, owned by the same slot.
  FlintTextRunnerState(FlintSession& session, cache::Adapter& adapter,
                       cache::Slot& slot, gpu::dflash::DflashForward* draft = nullptr)
      : session_(&session),
        adapter_(&adapter),
        slot_(&slot),
        draft_(draft),
        speculating_(draft != nullptr) {}

  void Invalidate() noexcept override {
    position_ = 0;
    pending_.reset();
    (void)adapter_->Invalidate(*slot_);
  }

  [[nodiscard]] FlintSession& session() const { return *session_; }
  [[nodiscard]] std::size_t position() const noexcept { return position_; }
  void set_position(std::size_t position) noexcept { position_ = position; }

  [[nodiscard]] gpu::dflash::DflashForward* draft() const noexcept { return draft_; }
  [[nodiscard]] bool speculating() const noexcept { return speculating_; }
  void stop_speculating() noexcept { speculating_ = false; }
  // A token already sampled and handed to the host but not yet consumed by the
  // session: it is the anchor of the next draft block, so the verification
  // pass consumes it with the drafted tokens instead of paying a forward pass
  // of its own.
  [[nodiscard]] std::optional<std::int32_t>& pending() noexcept {
    return pending_;
  }
  [[nodiscard]] std::vector<float>& feature_scratch() noexcept {
    return feature_scratch_;
  }

private:
  FlintSession* session_{nullptr};
  cache::Adapter* adapter_{nullptr};
  cache::Slot* slot_{nullptr};
  gpu::dflash::DflashForward* draft_{nullptr};
  bool speculating_{false};
  std::optional<std::int32_t> pending_;
  std::vector<float> feature_scratch_;
  std::size_t position_{0};
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
  // One engine pass: the drafter ingests the target features of the last pass
  // only, so a prefill chunk must not span two.
  static constexpr std::size_t kPrefillChunk = 4096;
  static constexpr std::size_t kWarmupTokens = 9000;

  FlintTextRunner(
      std::shared_ptr<FlintModel> model,
      std::unique_ptr<gufo::vendor::qwen::tokenization::QwenTokenizer>
          tokenizer,
      std::vector<std::int32_t> stop_tokens, std::uint32_t max_context,
      std::string artifact_fingerprint = {},
      std::shared_ptr<const DraftArtifact> draft = {},
      std::uint32_t max_draft_tokens = 0)
      : model_(std::move(model)),
        draft_model_(std::move(draft)),
        tokenizer_(std::move(tokenizer)),
        stop_tokens_(std::move(stop_tokens)),
        max_context_(max_context),
        max_draft_(max_draft_tokens) {
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
    if (const char* env = std::getenv("FLINT_DRAFT_MAX_CONTEXT");
        env != nullptr && env[0] != '\0')
      draft_max_context_ = static_cast<std::uint32_t>(std::atoi(env));
    if (const char* env = std::getenv("FLINT_DRAFT_TOKENS");
        env != nullptr && env[0] != '\0') {
      max_draft_ = static_cast<std::uint32_t>(std::atoi(env));
      draft_fixed_ = true;
    }
    if (draft_model_)
      std::fprintf(stderr,
                   "[flint] event=speculation backend=dflash2 max_draft=%u "
                   "drafter_layers=%u taps=%zu\n",
                   max_draft_, draft_model_->weights.config.num_layers,
                   draft_model_->weights.config.target_layer_ids.size());
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
      // Row path (docs/CACHE-ROWS.md): the same state captured as K and V rows
      // (only the rows a restore retains) plus the logits, before the session
      // moves on.
      const ContinuationLayout layout(model_->config(), kCapacity);
      struct RowCapture {
        const ContinuationLayout::Component* component;
        std::uint32_t first;
        std::vector<std::uint8_t> bytes;
      };
      std::vector<RowCapture> rows;
      bool rows_captured = true;
      for (const auto& component : layout.components()) {
        if (!cache::HasRows(component.descriptor.kind))
          continue;
        const auto first = static_cast<std::uint32_t>(
            layout.FirstRetainedRow(component, kPrefix));
        RowCapture row{&component, first,
                       std::vector<std::uint8_t>((kPrefix - first) *
                                                 layout.row_bytes())};
        rows_captured =
            rows_captured &&
            session->TransferRows(
                component.layer,
                component.part == ContinuationLayout::Part::kValue, first,
                kPrefix, row.bytes, true, &error);
        rows.push_back(std::move(row));
      }
      if (!rows_captured)
        Log("event=snapshot_selftest result=FAIL step=row_capture reason=" +
            error);
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
      // The same continuation restored from rows instead of the blob.
      for (const bool use_fresh : {false, true}) {
        FlintSession& target = use_fresh ? *fresh : *session;
        Tail restored;
        bool ok = rows_captured;
        for (const auto& row : rows) {
          ok = ok && target.TransferRows(
                         row.component->layer,
                         row.component->part ==
                             ContinuationLayout::Part::kValue,
                         row.first, kPrefix,
                         {const_cast<std::uint8_t*>(row.bytes.data()),
                          row.bytes.size()},
                         false, &error);
        }
        ok = ok && target.SetState(kPrefix, at_snapshot, &error) &&
             target.Position() == kPrefix &&
             same(copy(target.Logits()), at_snapshot) &&
             run_tail(target, &restored) &&
             same(restored.after_block, original.after_block) &&
             same(restored.after_steps, original.after_steps);
        all_ok = all_ok && ok;
        Log(std::string("event=snapshot_selftest result=") +
            (ok ? "pass" : "FAIL") + " path=rows target=" +
            (use_fresh ? "fresh_session" : "same_session") +
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
                // The scheduler calls DecodeStep with the whole remaining
                // budget (instead of Advance, one token at a time) only for a
                // runner that can emit several tokens per step.
                .multi_token_decode = draft_model_ != nullptr,
                .prefix_reuse = true,
            },
        .sampling_defaults = FlintSamplingDefaults(),
    };
  }

  [[nodiscard]] ext::TextRunnerResourceClaim ResourceClaim() const override {
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    std::optional<std::size_t> capacity;
    if (hipMemGetInfo(&free_bytes, &total_bytes) == hipSuccess)
      capacity = free_bytes;
    return {
        .resident_weights_bytes = model_->ResidentBytes(),
        .state_capacity_bytes = capacity,
        .per_request_state_bytes =
            model_->SessionBytes(max_context_ + FlintSession::kMaxVerifyRows),
        .temporary_scratch_bytes = 0,
        .retained_snapshot_capacity_bytes = ext::HostSnapshotBudgetBytes(),
        .retained_snapshot_ceiling_bytes = ext::HostSnapshotCeilingBytes(),
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
        std::make_unique<FlintContinuationAdapter>(
            model_, max_context_, component_identity_, draft_model_),
        ledger,
        hip::ComponentCacheHooks{
            [] { return ext::HostSnapshotCeilingBytes(); },
            [](std::string_view) {},
            [](std::string_view) {},
        });
  }

  [[nodiscard]] std::unique_ptr<ext::TextRunnerState> BindComponentState(
      cache::Adapter& adapter, cache::Slot& slot) const override {
    auto& flint = dynamic_cast<FlintContinuationAdapter&>(adapter);
    return std::make_unique<FlintTextRunnerState>(
        flint.GetSession(slot), adapter, slot, flint.GetDraft(slot));
  }

  void ReconcileComponentState(ext::TextRunnerState& state,
                               std::size_t position) const override {
    auto& flint = RequireFlintState(state);
    flint.set_position(position);
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
    if (offset != flint.position())
      throw std::logic_error("Flint prefill offset does not match state");
    if (offset >= prompt.size())
      throw std::logic_error("Flint prefill has no remaining input");
    const std::size_t consumed = std::min<std::size_t>(
        {max_input_tokens, prompt.size() - offset, kPrefillChunk});
    if (flint.draft() != nullptr && prompt.size() > draft_max_context_)
      flint.stop_speculating();
    const auto chunk = FlintEngineTokens(prompt.subspan(offset, consumed));
    std::string error;
    if (!flint.session().Append(chunk, &error)) {
      flint.Invalidate();
      throw std::runtime_error("Flint prefill failed: " + error);
    }
    InjectFeatures(flint, consumed);
    flint.set_position(offset + consumed);
    return {.consumed_tokens = consumed,
            .decode_ready = offset + consumed == prompt.size()};
  }

  [[nodiscard]] ext::TextDecodeSelection SelectNext(
      ext::TextRunnerState& state,
      sampling::SamplerState& sampler) const override {
    auto& flint = RequireFlintState(state);
    if (flint.position() >= max_context_)
      return {.stop = true, .piece = {}, .logprobs = std::nullopt};
    const auto logits = flint.session().Logits();
    if (logits.empty())
      throw std::runtime_error("Flint token selection has no logits");
    const auto token = static_cast<std::int32_t>(sampler.Sample(logits));
    if (flint.stop_at_eos() && IsStop(token))
      return {.stop = true, .token = 0, .piece = {}, .logprobs = std::nullopt};
    return {.stop = false,
            .token = static_cast<TextRunnerToken>(token),
            .piece = std::string(tokenizer_->DecodeToken(token)),
            .logprobs = std::nullopt};
  }

  void Advance(ext::TextRunnerState& state,
               TextRunnerToken token) const override {
    auto& flint = RequireFlintState(state);
    const std::int32_t input = FlintEngineTokens({&token, 1}).front();
    std::string error;
    if (!flint.session().Append({&input, 1}, &error)) {
      flint.Invalidate();
      throw std::runtime_error("Flint decode failed: " + error);
    }
    InjectFeatures(flint, 1);
    flint.set_position(flint.position() + 1);
  }

  // Speculative decoding (docs/DFLASH2.md). Each call is one draft block: the
  // drafter proposes k tokens after the anchor, one verification pass runs the
  // anchor and the proposals through the target, and the tokens are accepted
  // by the target's own sampler: a proposal is kept when the sample drawn from
  // the target distribution at its row equals it, otherwise that sample is the
  // next token. Each emitted token is therefore one draw from the target
  // distribution given its prefix, so the output distribution is unchanged
  // (greedy output is identical). The token that ends a block is sampled but
  // left unconsumed (pending): it anchors the next block.
  [[nodiscard]] ext::TextDecodeStep DecodeStep(
      ext::TextRunnerState& state, std::size_t max_tokens,
      sampling::SamplerState& sampler) const override {
    auto& flint = RequireFlintState(state);
    if (flint.draft() != nullptr && flint.speculating() &&
        flint.position() > draft_max_context_)
      flint.stop_speculating();
    if (flint.draft() == nullptr || !flint.speculating() || max_draft_ == 0) {
      FlushPending(flint);
      return ext::TextModelRunner::DecodeStep(state, max_tokens, sampler);
    }
    ext::TextDecodeStep step;
    if (max_tokens == 0)
      return step;
    if (flint.position() >= max_context_) {
      step.stop = true;
      return step;
    }
    // The anchor: the pending token, or a first token sampled now.
    std::int32_t anchor = 0;
    if (flint.pending().has_value()) {
      anchor = *flint.pending();
    } else {
      auto selection = SelectNext(state, sampler);
      if (selection.stop) {
        step.stop = true;
        step.terminal_logprobs = std::move(selection.logprobs);
        return step;
      }
      anchor = static_cast<std::int32_t>(selection.token);
      step.selections.push_back(std::move(selection));
      --max_tokens;
      flint.pending() = anchor;
    }
    if (max_tokens == 0)
      return step;
    // The block is sampled on a tentative copy of the request's sampler: each
    // token is accepted into the copy as it is chosen (repetition penalties and
    // grammar state follow), and only the draw state is published at the end.
    // The host accepts the emitted tokens into the real sampler itself.
    sampling::SamplerState tentative = sampler;
    const auto& sampling_config = tentative.config();
    // Rejection sampling against the drafter's own distribution (opt-in with
    // FLINT_DRAFT_REJECTION=1: measured no better than the exact match rule,
    // which is rejection sampling for a deterministic draft). Grammars and
    // reasoning budgets always use the match rule.
    static const bool rejection_enabled = [] {
      const char* e = std::getenv("FLINT_DRAFT_REJECTION");
      return e != nullptr && e[0] != '\0' && e[0] != '0';
    }();
    const bool rejection = rejection_enabled &&
                           sampling_config.temperature > 0.0F &&
                           sampling_config.uses_random_sampling() &&
                           !sampling_config.constraint &&
                           !sampling_config.reasoning_budget &&
                           !tentative.ForcedToken().has_value();

    // Each drafted row costs about 2.5 ms of verification and is accepted less
    // often than the one before: measured on Flint, the best block is 3 tokens
    // greedy and 2 when sampling. FLINT_DRAFT_TOKENS fixes it.
    const std::uint32_t tuned = sampling_config.temperature > 0.0F ? 2U : 3U;
    const std::uint32_t wanted = draft_fixed_ ? max_draft_ : std::min(max_draft_, tuned);
    const std::uint32_t room = max_context_ - static_cast<std::uint32_t>(
                                                  flint.position());
    std::uint32_t k = std::min<std::uint32_t>(
        {wanted, flint.draft()->config().block_size - 1U,
         static_cast<std::uint32_t>(max_tokens - 1),
         room > 2 ? room - 2 : 0});

    const auto t_start = std::chrono::steady_clock::now();
    gpu::dflash::DflashDraft proposal;
    std::vector<std::int32_t> drafted;
    const std::uint32_t top_k = flint.draft()->config().selector_top_k;
    if (k > 0 && flint.draft()->ContextLength() == flint.position()) {
      std::vector<float> uniforms;
      if (rejection) {
        uniforms.resize(k);
        for (auto& u : uniforms)
          u = static_cast<float>(tentative.Uniform());
      }
      std::string error;
      if (!flint.draft()->Run(static_cast<std::uint32_t>(anchor),
                              static_cast<std::uint32_t>(flint.position()), k,
                              rejection ? sampling_config.temperature : 0.0F,
                              uniforms, {}, &proposal, &error)) {
        std::fprintf(stderr, "[flint] drafter failed, speculation off: %s\n",
                     error.c_str());
        flint.stop_speculating();
        proposal = {};
      } else {
        drafted.assign(proposal.tokens.begin(), proposal.tokens.end());
      }
    }
    k = static_cast<std::uint32_t>(drafted.size());
    const auto t_drafted = std::chrono::steady_clock::now();

    std::vector<std::int32_t> run;
    run.reserve(k + 1);
    run.push_back(anchor);
    run.insert(run.end(), drafted.begin(), drafted.end());
    const auto base = static_cast<std::uint32_t>(flint.position());
    std::string error;
    const bool ok = run.size() == 1
                        ? flint.session().Append(run, &error)
                        : flint.session().AppendAll(run, &error);
    if (!ok) {
      flint.Invalidate();
      throw std::runtime_error("Flint decode failed: " + error);
    }
    const auto t_verified = std::chrono::steady_clock::now();

    // Row i is the distribution after run[0..i].
    std::uint32_t consumed = 0;  // index of the last consumed row
    bool stop = false;
    const auto emit = [&](std::int32_t token) {
      step.selections.push_back(ext::TextDecodeSelection{
          .stop = false,
          .token = static_cast<TextRunnerToken>(token),
          .piece = std::string(tokenizer_->DecodeToken(token)),
          .logprobs = std::nullopt});
    };
    for (std::uint32_t i = 0; i <= k; ++i) {
      const auto logits = run.size() == 1 ? flint.session().Logits()
                                          : flint.session().Logits(i);
      consumed = i;
      std::int32_t token = 0;
      bool accepted = false;
      if (rejection && i < k) {
        // Keep the proposal with probability min(1, p / q); on rejection draw
        // from the residual of p over q, which keeps the output distribution.
        const auto distribution = tentative.Distribution(logits);
        const auto proposed = static_cast<sampling::TokenId>(drafted[i]);
        const double p = distribution.probability(proposed);
        const double q = proposal.confidences[i];
        if (q > 0.0 && tentative.Uniform() * q < p) {
          token = drafted[i];
          accepted = true;
        } else {
          const auto ids = std::span<const sampling::TokenId>(
              proposal.candidate_ids.data() + std::size_t{i} * top_k, top_k);
          const auto probs = std::span<const float>(
              proposal.candidate_probabilities.data() +
                  std::size_t{i} * top_k, top_k);
          token = static_cast<std::int32_t>(distribution.SampleResidual(
              ids, probs, tentative.mutable_rng_state()));
        }
      } else {
        token = static_cast<std::int32_t>(tentative.Sample(logits));
        accepted = i < k && token == drafted[i];
      }
      if (flint.stop_at_eos() && IsStop(token)) {
        stop = true;
        flint.pending().reset();
        break;
      }
      tentative.Accept(static_cast<sampling::TokenId>(token));
      emit(token);
      if (accepted) {
        ++step.draft_accepted_tokens;
        continue;
      }
      flint.pending() = token;
      break;
    }
    sampler.CopyDrawStateFrom(tentative);
    const std::uint32_t kept = consumed + 1;  // the anchor and what it accepted
    if (run.size() > 1) {
      if (kept < run.size())
        flint.session().Rewind(base + kept);
      flint.session().KeepLogitRow(consumed);
    }
    InjectFeatures(flint, kept);
    flint.set_position(base + kept);
    ReportRound(t_start, t_drafted, t_verified, std::chrono::steady_clock::now(),
                k, kept);
    step.draft_rounds = k > 0 ? 1 : 0;
    step.draft_tokens = k;
    step.stop = stop;
    return step;
  }

  [[nodiscard]] std::optional<ext::TextDecodeSelection> PreviewFirstToken(
      ext::TextRunnerState& state,
      sampling::SamplerState& sampler) const override {
    return SelectNext(state, sampler);
  }

  [[nodiscard]] std::size_t CheckpointPosition(
      const ext::TextRunnerState& state) const override {
    return RequireFlintState(state).position();
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
  // FLINT_DRAFT_TIMING=1: mean milliseconds per speculation round (draft,
  // verification pass, acceptance and feature injection), every 64 rounds.
  static void ReportRound(std::chrono::steady_clock::time_point start,
                          std::chrono::steady_clock::time_point drafted,
                          std::chrono::steady_clock::time_point verified,
                          std::chrono::steady_clock::time_point done,
                          std::uint32_t proposed, std::uint32_t kept) {
    static const bool on = [] {
      const char* e = std::getenv("FLINT_DRAFT_TIMING");
      return e != nullptr && e[0] != '\0' && e[0] != '0';
    }();
    if (!on)
      return;
    static std::atomic<std::uint64_t> rounds{0}, proposed_sum{0}, kept_sum{0};
    static std::atomic<std::uint64_t> draft_us{0}, verify_us{0}, tail_us{0};
    const auto us = [](auto a, auto b) {
      return static_cast<std::uint64_t>(
          std::chrono::duration_cast<std::chrono::microseconds>(b - a).count());
    };
    draft_us += us(start, drafted);
    verify_us += us(drafted, verified);
    tail_us += us(verified, done);
    proposed_sum += proposed;
    kept_sum += kept;
    const auto n = ++rounds;
    if (n % 64 == 0)
      std::fprintf(stderr,
                   "[flint] event=round_timing rounds=%llu draft_ms=%.1f "
                   "verify_ms=%.1f tail_ms=%.1f proposed=%.1f kept=%.2f\n",
                   static_cast<unsigned long long>(n),
                   draft_us.exchange(0) / 64000.0,
                   verify_us.exchange(0) / 64000.0,
                   tail_us.exchange(0) / 64000.0,
                   proposed_sum.exchange(0) / 64.0, kept_sum.exchange(0) / 64.0);
  }

  // Lets the drafter read the features of the forward pass that just ran for
  // `rows` tokens. A failure ends speculation for this state, not the request.
  void InjectFeatures(FlintTextRunnerState& flint, std::size_t rows) const {
    auto* draft = flint.draft();
    if (draft == nullptr || !flint.speculating() || rows == 0)
      return;
    auto& session = flint.session();
    // The `rows` tokens that just ran sit at [first, first + rows). The target
    // keeps the last HiddenRingRows of them; the drafter attends to its window
    // only, so earlier rows are skipped rather than injected.
    const std::uint32_t first = static_cast<std::uint32_t>(flint.position());
    // A row stays available while row + ring > position, so a ring of N rows
    // serves N - 1 of the rows a pass just wrote.
    const std::uint32_t usable = session.HiddenRingRows() - 1;
    std::uint32_t skip = 0;
    if (rows > usable)
      skip = static_cast<std::uint32_t>(rows) - usable;
    std::string error;
    bool ok = draft->ContextLength() == first;
    if (!ok) {
      error = "drafter context is not contiguous with the target";
    } else {
      if (skip > 0)
        draft->SkipContextTo(first + skip);
      const auto count = static_cast<std::uint32_t>(rows) - skip;
      auto& scratch = flint.feature_scratch();
      scratch.resize(std::size_t{count} * draft->config().target_layer_ids.size() *
                     draft->config().hidden_size);
      ok = session.CopyHiddenRows(first + skip, count, scratch, &error) &&
           draft->AppendContext(scratch, first + skip, count, &error);
    }
    if (!ok) {
      std::fprintf(stderr, "[flint] drafter context lost, speculation off: %s\n",
                   error.c_str());
      flint.stop_speculating();
    }
  }

  // Consumes a sampled-but-unconsumed token so the state can be driven one
  // token at a time again.
  void FlushPending(FlintTextRunnerState& flint) const {
    if (!flint.pending().has_value())
      return;
    const std::int32_t token = *flint.pending();
    flint.pending().reset();
    std::string error;
    if (!flint.session().Append({&token, 1}, &error)) {
      flint.Invalidate();
      throw std::runtime_error("Flint decode failed: " + error);
    }
    InjectFeatures(flint, 1);
    flint.set_position(flint.position() + 1);
  }

  [[nodiscard]] bool IsStop(std::int32_t token) const {
    return std::find(stop_tokens_.begin(), stop_tokens_.end(), token) !=
           stop_tokens_.end();
  }

  std::shared_ptr<FlintModel> model_;
  std::shared_ptr<const DraftArtifact> draft_model_;
  std::unique_ptr<gufo::vendor::qwen::tokenization::QwenTokenizer> tokenizer_;
  std::vector<std::int32_t> stop_tokens_;
  std::uint32_t max_context_;
  std::uint32_t max_draft_{0};
  bool draft_fixed_{false};
  // Speculation pays up to about this many tokens of context: beyond it the
  // verification pass costs more in attention than the accepted tokens save
  // (measured: parity at 9k, -10 % at 38k, -15 % at 125k).
  std::uint32_t draft_max_context_{8192};
  cache::Identity component_identity_;
};

}  // namespace

std::optional<sampling::TextSamplingDefaults> FlintTextSamplingDefaults(
    const core::GgufReader&) {
  return FlintSamplingDefaults();
}

std::shared_ptr<ext::TextModelRunner> CreateFlintTextRunner(
    const ext::TextRunnerOptions& options, std::string* error) {
  try {
    if (options.model_path.empty()) {
      if (error)
        *error = "Flint requires a model path";
      return nullptr;
    }
    const bool speculative =
        options.speculative.backend == ext::TextSpeculativeBackend::kDFlash;
    if (options.speculative.backend != ext::TextSpeculativeBackend::kDisabled &&
        !speculative) {
      if (error)
        *error = "Flint supports dflash2 speculation only";
      return nullptr;
    }
    if (speculative && options.speculative.draft_model_path.empty()) {
      if (error)
        *error = "Flint dflash2 speculation requires --draft-model";
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
    // The drafter's weights reference its mapped GGUF, which they keep alive.
    std::shared_ptr<const DraftArtifact> draft;
    if (speculative) {
      std::shared_ptr<const core::GgufReader> draft_reader =
          core::GgufReader::OpenFile(options.speculative.draft_model_path,
                                     &load_error);
      std::optional<gpu::dflash::DflashWeights> weights;
      if (draft_reader)
        weights = gpu::dflash::DflashWeights::LoadFromGguf(*draft_reader,
                                                           &load_error);
      if (!weights) {
        if (error)
          *error = "Failed to load the Flint drafter: " + load_error;
        return nullptr;
      }
      if (weights->config.hidden_size != config->hidden_size ||
          weights->config.vocab_size != config->vocab_size) {
        if (error)
          *error = "the drafter does not match this Flint model";
        return nullptr;
      }
      draft = std::make_shared<const DraftArtifact>(
          DraftArtifact{std::move(draft_reader), std::move(*weights)});
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

    return std::make_shared<FlintTextRunner>(
        std::move(model), std::move(tokenizer), std::move(stop_tokens),
        max_context, std::move(fingerprint), std::move(draft),
        speculative ? options.speculative.max_draft_tokens : 0U);
  } catch (const std::exception& e) {
    if (error)
      *error = std::string("Flint runner creation failed: ") + e.what();
    return nullptr;
  }
}

}  // namespace gufo::models::flint
