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

#include "hip/vendor/qwen/tokenizer.hpp"
#include "continuation_adapter.hpp"
#include "src/chat_template.hpp"
#include "src/gpu/engine.hpp"
#include "src/runtime/config.hpp"

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
           << "payload_layout=flint-rocm-session-snapshot-v"
           << FlintSession::kSnapshotPayloadVersion << '\n';
  const auto canonical = identity.str();
  return {canonical.begin(), canonical.end()};
}

std::vector<std::uint8_t> ComponentIdentity(
    const FlintModel* model, std::uint32_t max_context) {
  std::ostringstream identity;
  identity << "schema=gufo-process-text-continuation-v1\n"
           << "model_kind=flint\n"
           // Without a validated artifact fingerprint the loaded model instance
           // is the identity; durable caches require the fingerprint path.
           << "model_instance=" << model << '\n'
           << "context_tokens=" << max_context << '\n'
           << "state_abi=" << kFlintStateAbi << '\n'
           << "payload_layout=flint-rocm-session-snapshot-v"
           << FlintSession::kSnapshotPayloadVersion << '\n';
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

class FlintTextRunnerState final : public ext::TextRunnerState {
public:
  FlintTextRunnerState(const std::shared_ptr<FlintModel>& model,
                       std::uint32_t max_context) {
    std::string error;
    session_ = model->CreateSession(max_context + FlintSession::kMaxVerifyRows,
                                    &error);
    if (session_ == nullptr) {
      throw std::runtime_error("Failed to create Flint session: " + error);
    }
  }

  // Component-cache slot: the session is owned by the adapter's slot, which
  // outlives this state through the retained adapter/slot handles.
  FlintTextRunnerState(FlintSession& session, cache::Adapter& adapter,
                       cache::Slot& slot)
      : bound_session_(&session), adapter_(&adapter), slot_(&slot) {}

  void Invalidate() noexcept override {
    position_ = 0;
    session().Reset();
  }

  [[nodiscard]] FlintSession& session() const {
    return bound_session_ != nullptr ? *bound_session_ : *session_;
  }
  [[nodiscard]] std::size_t position() const noexcept { return position_; }
  void set_position(std::size_t position) noexcept { position_ = position; }

private:
  std::unique_ptr<FlintSession> session_;
  FlintSession* bound_session_{nullptr};
  cache::Adapter* adapter_{nullptr};
  cache::Slot* slot_{nullptr};
  std::size_t position_{0};
};

class FlintTextRunnerSnapshot final : public ext::TextRunnerSnapshot {
public:
  FlintTextRunnerSnapshot(std::shared_ptr<FlintModel> model,
                          std::vector<std::uint8_t> bytes, std::size_t position)
      : model(std::move(model)), bytes(std::move(bytes)), position(position) {}

  [[nodiscard]] std::size_t PayloadBytes() const noexcept override {
    return bytes.size();
  }

  std::shared_ptr<FlintModel> model;
  std::vector<std::uint8_t> bytes;
  std::size_t position;
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

  FlintTextRunner(std::shared_ptr<FlintModel> model,
                  std::unique_ptr<gufo::vendor::qwen::tokenization::QwenTokenizer> tokenizer,
                  std::vector<std::int32_t> stop_tokens,
                  std::uint32_t max_context,
                  std::string artifact_fingerprint = {})
      : model_(std::move(model)),
        tokenizer_(std::move(tokenizer)),
        stop_tokens_(std::move(stop_tokens)),
        max_context_(max_context) {
    if (!artifact_fingerprint.empty()) {
      persistence_ = ext::TextRunnerPersistenceDescriptor{
          .compatibility_identity =
              FlintCompatibilityIdentity(artifact_fingerprint, max_context_),
          .payload_version = FlintSession::kSnapshotPayloadVersion,
      };
    }
    Warmup();
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
                .snapshot = true,
                .fork = true,
                .final_token_advance_required = false,
                .incremental_text_is_exact = true,
                .prefix_reuse = true,
            },
        .persistence = persistence_,
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

  [[nodiscard]] bool UsesComponentCache() const override { return true; }

  [[nodiscard]] std::unique_ptr<ext::ComponentCacheResources>
  CreateComponentCacheResources(cache::ResourceLedger& ledger,
                                std::size_t ram) const override {
    (void)ram;
    cache::Identity identity;
    if (persistence_.has_value())
      identity = persistence_->compatibility_identity;
    else
      identity = ComponentIdentity(model_.get(), max_context_);
    return hip::CreateHipComponentCacheResources(
        std::make_unique<FlintContinuationAdapter>(model_, max_context_,
                                                   std::move(identity)),
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
    return std::make_unique<FlintTextRunnerState>(flint.GetSession(slot),
                                                  adapter, slot);
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

  [[nodiscard]] std::optional<std::vector<TextRunnerToken>> RenderAndTokenize(
      const ext::ChatRequest& request) const override {
    if (!request.tools.empty() &&
        request.tool_choice != ext::ChatRequest::ToolChoice::kNone) {
      throw std::invalid_argument("Flint does not support tools yet");
    }
    std::vector<ChatMessage> messages;
    messages.reserve(request.messages.size());
    for (const auto& message : request.messages) {
      if (!message.images.empty())
        throw std::invalid_argument("Flint does not support image input");
      messages.push_back({FlintRoleName(message.role),
                          message.content + message.framing_suffix,
                          message.thought});
    }
    gufo::vendor::qwen::tokenization::TokenizerOptions options;
    options.parse_special_tokens = true;
    // Qualified: the inherited TextModelRunner::RenderPrompt member would hide
    // the namespace-scope overload under unqualified lookup.
    return tokenizer_->Encode(::gufo::models::flint::RenderPrompt(messages),
                              options);
  }

  [[nodiscard]] std::string Decode(
      std::span<const TextRunnerToken> tokens) const override {
    return tokenizer_->Decode(tokens);
  }

  [[nodiscard]] std::unique_ptr<ext::TextRunnerState> CreateState()
      const override {
    return std::make_unique<FlintTextRunnerState>(model_, max_context_);
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
    const auto chunk = FlintEngineTokens(prompt.subspan(offset, consumed));
    std::string error;
    if (!flint.session().Append(chunk, &error)) {
      flint.Invalidate();
      throw std::runtime_error("Flint prefill failed: " + error);
    }
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
    flint.set_position(flint.position() + 1);
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

  [[nodiscard]] std::size_t SnapshotPayloadBytes(
      const ext::TextRunnerState& state) const override {
    return RequireFlintState(state).session().SnapshotBytes();
  }

  [[nodiscard]] std::unique_ptr<ext::TextRunnerSnapshot> Snapshot(
      const ext::TextRunnerState& state) const override {
    const auto& flint = RequireFlintState(state);
    std::vector<std::uint8_t> bytes(flint.session().SnapshotBytes());
    std::string error;
    if (!flint.session().SaveSnapshot(bytes, &error))
      throw std::runtime_error("Flint snapshot failed: " + error);
    return std::make_unique<FlintTextRunnerSnapshot>(model_, std::move(bytes),
                                                     flint.position());
  }

  void RestoreOrFork(ext::TextRunnerState& state,
                     const ext::TextRunnerSnapshot& snapshot) const override {
    const auto& flint_snapshot = RequireSnapshot(snapshot);
    auto& restored = RequireFlintState(state);
    std::string error;
    if (!restored.session().RestoreSnapshot(flint_snapshot.bytes, &error)) {
      restored.Invalidate();
      throw std::runtime_error("Flint snapshot restore failed: " + error);
    }
    restored.set_position(flint_snapshot.position);
  }

  [[nodiscard]] std::size_t PersistentSnapshotPayloadBytes(
      const ext::TextRunnerSnapshot& snapshot) const override {
    return RequireSnapshot(snapshot).bytes.size();
  }

  [[nodiscard]] std::size_t SerializePersistentSnapshot(
      const ext::TextRunnerSnapshot& snapshot,
      std::span<std::uint8_t> destination) const override {
    const auto& bytes = RequireSnapshot(snapshot).bytes;
    if (destination.size() != bytes.size()) {
      throw std::invalid_argument(
          "Flint persistent snapshot serialization failed");
    }
    std::memcpy(destination.data(), bytes.data(), bytes.size());
    return destination.size();
  }

  void StreamPersistentSnapshot(const ext::TextRunnerSnapshot& snapshot,
                                const SnapshotSink& sink) const override {
    sink(RequireSnapshot(snapshot).bytes);
  }

  void RestorePersistentSnapshot(
      ext::TextRunnerState& state,
      std::span<const std::uint8_t> payload) const override {
    auto& restored = RequireFlintState(state);
    std::string error;
    if (!restored.session().RestoreSnapshot(payload, &error)) {
      restored.Invalidate();
      throw std::runtime_error("Flint persistent snapshot restore failed: " +
                               error);
    }
    const std::uint32_t position = restored.session().Position();
    if (position == 0 || position > max_context_) {
      restored.Invalidate();
      throw std::runtime_error(
          "Flint persistent snapshot restored an invalid position");
    }
    restored.set_position(position);
  }

private:
  [[nodiscard]] bool IsStop(std::int32_t token) const {
    return std::find(stop_tokens_.begin(), stop_tokens_.end(), token) !=
           stop_tokens_.end();
  }

  [[nodiscard]] const FlintTextRunnerSnapshot& RequireSnapshot(
      const ext::TextRunnerSnapshot& snapshot) const {
    const auto* flint = dynamic_cast<const FlintTextRunnerSnapshot*>(&snapshot);
    if (flint == nullptr || flint->model.get() != model_.get())
      throw std::invalid_argument(
          "Flint snapshot does not belong to this model");
    return *flint;
  }

  std::shared_ptr<FlintModel> model_;
  std::unique_ptr<gufo::vendor::qwen::tokenization::QwenTokenizer> tokenizer_;
  std::vector<std::int32_t> stop_tokens_;
  std::uint32_t max_context_;
  std::optional<ext::TextRunnerPersistenceDescriptor> persistence_;
};

}  // namespace

std::shared_ptr<ext::TextModelRunner> CreateFlintTextRunner(
    const ext::TextRunnerOptions& options, std::string* error) {
  try {
    if (options.model_path.empty()) {
      if (error)
        *error = "Flint requires a model path";
      return nullptr;
    }
    if (options.speculative.backend != ext::TextSpeculativeBackend::kDisabled) {
      if (error)
        *error = "Flint does not support draft-model speculation";
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
        gufo::vendor::qwen::tokenization::QwenTokenizer::CreateFromGguf(*reader, &tokenizer_error);
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

    return std::make_shared<FlintTextRunner>(
        std::move(model), std::move(tokenizer), std::move(stop_tokens),
        max_context, std::move(fingerprint));
  } catch (const std::exception& e) {
    if (error)
      *error = std::string("Flint runner creation failed: ") + e.what();
    return nullptr;
  }
}

}  // namespace gufo::models::flint
