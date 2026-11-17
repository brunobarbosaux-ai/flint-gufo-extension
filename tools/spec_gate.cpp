// M3 gates for the DFlash2 speculative cycle, driven through the runner's own
// text-runner contract (the calls the host makes): CreateFlintTextRunner ->
// CreateComponentCacheResources -> CreateSlot -> BindComponentState -> Prefill
// -> DecodeStep. Speculation is selected by FLINT_SPEC, read when the runner
// is created, so each configuration gets its own runner.
//
//   --mode gx       greedy spec-on (dflash) byte-identical to spec-off
//   --mode null     FLINT_SPEC=null byte-identical to spec-off at each
//                   temperature and seed
//   --mode speed    tokens per cycle and speedup of dflash over spec-off,
//                   once per --policies entry (fixed, adaptive)
//   --mode settings each sampler setting (top-k, top-p, min-p, repeat and
//                   frequency penalty, JSON grammar): spec-on must equal
//                   spec-off, and grammar output must parse and validate
//                   --settings <substr> runs only the matching settings
//   --mode memory   runner claim (resident + per-request) vs the device bytes
//                   taken by runner + one slot + one bound state
//
// --depth N pads every chat prompt with deterministic filler so the prompt is
// exactly N tokens (0 = unpadded). The filler is the same for spec-off and
// spec-on. Decode throughput excludes prefill.
//
// Usage: flint_spec_gate --target <target.gguf> --prompts <prompts.jsonl>
//          --mode gx|null|speed|settings [--depth N] [--max-tokens N]
//          [--temps 0,0.6,1.0] [--seeds 1,2,3] [--prompts-limit N]
//          [--policies fixed,adaptive] [--max-draft N] [--min-draft N]
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>

#include <gufo/ext/gguf_reader.hpp>
#include <gufo/ext/json.hpp>
#include <gufo/ext/sampling.hpp>
#include <gufo/ext/text.hpp>

#include "continuation_adapter.hpp"
#include "hip/vendor/qwen/tokenizer.hpp"
#include "runner.hpp"
#include "src/chat_template.hpp"
#include "src/cache/ledger.hpp"

namespace core = gufo::core;
namespace cache = gufo::cache;
namespace ext = gufo::ext;
namespace sampling = gufo::sampling;
namespace model = gufo::models::flint;
namespace qwen = gufo::vendor::qwen::tokenization;

namespace {

[[noreturn]] void Fail(const std::string& message) {
  throw std::runtime_error(message);
}

struct Options {
  std::string target;
  std::string prompts;
  std::string mode;
  std::size_t max_tokens{192};
  std::vector<float> temps{0.0F};
  std::vector<std::uint64_t> seeds{1};
  std::size_t prompts_limit{12};
  // Settings mode: run only settings whose name contains this (empty = all).
  std::string settings_filter;
  // Speed mode: which draft policies to measure, and the draft bounds.
  std::vector<ext::TextDFlashDraftPolicy> policies{
      ext::TextDFlashDraftPolicy::kAdaptive};
  std::uint32_t max_draft{7};
  std::uint32_t min_draft{1};
  // Prompt depth in tokens (0 = unpadded). Sets the runner's max_context.
  std::size_t depth{0};
};

// The policy the gate's dflash runs use; the spec-off runs ignore it.
struct DraftSettings {
  ext::TextDFlashDraftPolicy policy{ext::TextDFlashDraftPolicy::kAdaptive};
  std::uint32_t max_draft{7};
  std::uint32_t min_draft{1};
};

std::vector<std::string> Split(const std::string& text) {
  std::vector<std::string> parts;
  std::stringstream stream(text);
  std::string part;
  while (std::getline(stream, part, ','))
    if (!part.empty())
      parts.push_back(part);
  return parts;
}

Options ParseArgs(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    const auto next = [&](const char* flag) -> std::string {
      if (i + 1 >= argc)
        Fail(std::string(flag) + " needs a value");
      return argv[++i];
    };
    if (arg == "--target") {
      options.target = next("--target");
    } else if (arg == "--prompts") {
      options.prompts = next("--prompts");
    } else if (arg == "--mode") {
      options.mode = next("--mode");
    } else if (arg == "--max-tokens") {
      options.max_tokens = std::stoul(next("--max-tokens"));
    } else if (arg == "--prompts-limit") {
      options.prompts_limit = std::stoul(next("--prompts-limit"));
    } else if (arg == "--temps") {
      options.temps.clear();
      for (const auto& part : Split(next("--temps")))
        options.temps.push_back(std::stof(part));
    } else if (arg == "--policies") {
      options.policies.clear();
      for (const auto& part : Split(next("--policies"))) {
        if (part == "fixed")
          options.policies.push_back(ext::TextDFlashDraftPolicy::kFixed);
        else if (part == "adaptive")
          options.policies.push_back(ext::TextDFlashDraftPolicy::kAdaptive);
        else
          Fail("--policies takes fixed and/or adaptive");
      }
    } else if (arg == "--max-draft") {
      options.max_draft = std::stoul(next("--max-draft"));
    } else if (arg == "--min-draft") {
      options.min_draft = std::stoul(next("--min-draft"));
    } else if (arg == "--seeds") {
      options.seeds.clear();
      for (const auto& part : Split(next("--seeds")))
        options.seeds.push_back(std::stoull(part));
    } else if (arg == "--depth") {
      options.depth = std::stoul(next("--depth"));
    } else if (arg == "--settings") {
      options.settings_filter = next("--settings");
    } else {
      Fail("unknown argument " + arg);
    }
  }
  // Memory mode builds no prompts, so --prompts is optional there.
  if (options.target.empty() || options.mode.empty() ||
      (options.prompts.empty() && options.mode != "memory"))
    Fail("--target, --prompts and --mode are required");
  return options;
}

struct Prompt {
  std::string id;
  std::vector<std::uint32_t> tokens;
};

// Deterministic filler: a fixed-seed word stream, encoded block by block, cut
// to exactly `count` tokens. Identical for every run at a given count.
std::vector<std::uint32_t> FillerTokens(const qwen::QwenTokenizer& tokenizer,
                                        std::size_t count) {
  static constexpr const char* kWords[] = {
      "river", "lantern", "copper", "harbor", "signal", "meadow", "garden",
      "engine", "thread", "compass", "orchard", "ledger", "beacon", "fabric",
      "window", "bridge", "marble", "quarry", "saddle", "tunnel", "violet",
      "pepper", "cinder", "anchor"};
  constexpr std::size_t kWordCount = sizeof(kWords) / sizeof(kWords[0]);
  std::vector<std::uint32_t> tokens;
  std::uint64_t state = 0x9E3779B97F4A7C15ULL;
  while (tokens.size() < count) {
    std::string text;
    for (int word = 0; word < 256; ++word) {
      state = state * 6364136223846793005ULL + 1442695040888963407ULL;
      text += kWords[(state >> 33) % kWordCount];
      text += ' ';
    }
    const auto ids = tokenizer.Encode(text);
    tokens.insert(tokens.end(), ids.begin(), ids.end());
  }
  tokens.resize(count);
  return tokens;
}

// Chat prompts through the chat template with thinking disabled, the same
// encoding the acceptance workload was measured with. With depth > 0 the
// filler is prepended so every prompt is exactly `depth` tokens.
std::vector<Prompt> ReadPrompts(const std::string& path,
                                const qwen::QwenTokenizer& tokenizer,
                                std::size_t limit, std::size_t depth) {
  std::ifstream input(path, std::ios::binary);
  if (!input)
    Fail("cannot read " + path);
  std::vector<Prompt> prompts;
  std::string line;
  while (std::getline(input, line) && prompts.size() < limit) {
    if (line.find_first_not_of(" \t\r") == std::string::npos)
      continue;
    const auto spec = gufo::json::parse(line);
    if (spec.member_str("kind") != "chat")
      continue;
    model::ChatTemplateOptions template_options;
    template_options.enable_thinking = false;
    qwen::TokenizerOptions encode_options;
    encode_options.parse_special_tokens = true;
    const auto ids = tokenizer.Encode(
        model::RenderPrompt(
            std::vector<model::ChatMessage>{
                {"user", spec.member_str("text"), ""}},
            template_options),
        encode_options);
    if (ids.size() > depth && depth > 0)
      Fail("prompt " + spec.member_str("id") + " is longer than --depth");
    Prompt prompt;
    prompt.id = spec.member_str("id");
    if (depth > 0) {
      prompt.tokens = FillerTokens(tokenizer, depth - ids.size());
      prompt.tokens.insert(prompt.tokens.end(), ids.begin(), ids.end());
    } else {
      prompt.tokens.assign(ids.begin(), ids.end());
    }
    prompts.push_back(std::move(prompt));
  }
  if (prompts.empty())
    Fail("no chat prompts in " + path);
  return prompts;
}

struct RunResult {
  std::vector<std::uint32_t> tokens;
  std::size_t draft_rounds{0};
  std::size_t draft_tokens{0};
  std::size_t draft_accepted_tokens{0};
  // True when the runner reported a stop (EOS or constraint complete); false
  // when the loop ended at the --max-tokens budget.
  bool stopped{false};
  double seconds{0.0};
  double decode_seconds{0.0};
};

// The gate runs one request at a time and never restores a snapshot, so the
// guard's hooks have nothing to protect.
class NoGuard final : public cache::MutationGuard {
public:
  void BeforeOverwrite(cache::ComponentId, cache::Rows, cache::Rows) override {}
  void BeforeRelease(cache::ComponentId, cache::Rows, cache::Rows) noexcept override {}
};

// One runner per configuration. FLINT_SPEC is read when the runner is created.
class Harness {
public:
  explicit Harness(const std::string& spec_value, const std::string& target,
                   const DraftSettings& draft, std::uint32_t max_context) {
    if (spec_value.empty())
      unsetenv("FLINT_SPEC");
    else
      setenv("FLINT_SPEC", spec_value.c_str(), 1);
    ext::TextRunnerOptions options;
    options.model_path = target;
    options.max_context = max_context;
    options.speculative.backend = spec_value == "dflash"
                                      ? ext::TextSpeculativeBackend::kDFlash
                                      : ext::TextSpeculativeBackend::kDisabled;
    options.speculative.dflash_policy = draft.policy;
    options.speculative.max_draft_tokens = draft.max_draft;
    options.speculative.min_draft_tokens = draft.min_draft;
    std::string error;
    runner_ = model::CreateFlintTextRunner(options, &error);
    if (!runner_)
      Fail("runner creation failed: " + error);
    // The ledger is sized from the runner's own claim, as the serve layer
    // sizes it from the same numbers.
    const auto claim = runner_->ResourceClaim();
    const std::size_t host = claim.retained_snapshot_capacity_bytes.value_or(
        std::size_t{1} << 34);
    ledger_ = std::make_unique<cache::ResourceLedger>(cache::ResourceLimits{
        .total_bytes = host,
        .ram_bytes = host,
        .staging_bytes = std::size_t{1} << 30,
        .persistence_pinned_bytes = 0});
    resources_ = runner_->CreateComponentCacheResources(*ledger_, 0);
    if (!resources_ || !resources_->adapter)
      Fail("component cache resources unavailable");
    guard_ = std::make_unique<NoGuard>();
    slot_ = resources_->adapter->CreateSlot(*guard_);
  }

  // Runs one prompt from an empty slot. The cycle state persists across runs
  // on this slot; Invalidate resets it, as a new request would.
  RunResult Run(const Prompt& prompt, const sampling::SamplingConfig& config,
                std::size_t max_tokens) {
    auto state = runner_->BindComponentState(*resources_->adapter, *slot_);
    state->Invalidate();
    state->SetStopAtEos(true);
    sampling::SamplerState sampler(config, prompt.tokens);
    const std::span<const ext::TextRunnerToken> tokens(prompt.tokens);

    RunResult result;
    const auto started = std::chrono::steady_clock::now();
    std::size_t offset = 0;
    for (;;) {
      // No gate-side cap: the runner's own limits pick the chunk size.
      const auto step = runner_->Prefill(
          *state, tokens, offset, std::numeric_limits<std::size_t>::max());
      offset += step.consumed_tokens;
      if (step.decode_ready)
        break;
    }
    const auto decode_started = std::chrono::steady_clock::now();

    // The host previews the first token on a copy of the sampler and requires
    // the real first decode to repeat it. The gate checks the same thing.
    const auto preview = runner_->PreviewFirstToken(*state, sampler);

    while (result.tokens.size() < max_tokens) {
      const auto decode = runner_->DecodeStep(
          *state, max_tokens - result.tokens.size(), sampler);
      for (const auto& selection : decode.selections) {
        if (result.tokens.empty() && preview && preview->token != selection.token)
          Fail("first-token preview disagrees with decoding");
        result.tokens.push_back(selection.token);
        sampler.Accept(selection.token);
      }
      result.draft_rounds += decode.draft_rounds;
      result.draft_tokens += decode.draft_tokens;
      result.draft_accepted_tokens += decode.draft_accepted_tokens;
      if (decode.stop) {
        result.stopped = true;
        break;
      }
      if (decode.selections.empty())
        break;
    }
    const auto finished = std::chrono::steady_clock::now();
    result.seconds = std::chrono::duration<double>(finished - started).count();
    result.decode_seconds =
        std::chrono::duration<double>(finished - decode_started).count();
    return result;
  }

  // Binds a grammar the way the host does: through the runner's vocabulary.
  std::shared_ptr<const sampling::TokenConstraint> Bind(
      std::shared_ptr<const sampling::JsonConstraint> grammar) const {
    return runner_->BindConstraint(std::move(grammar));
  }

  // Binds the slot's request state (allocates the hidden ring when spec is
  // on). The caller keeps it alive while measuring.
  std::unique_ptr<ext::TextRunnerState> BindState() const {
    return runner_->BindComponentState(*resources_->adapter, *slot_);
  }

  // A further slot on the same adapter, with its own bound state.
  std::unique_ptr<cache::Slot> NewSlot() {
    return resources_->adapter->CreateSlot(*guard_);
  }
  std::unique_ptr<ext::TextRunnerState> BindStateOn(cache::Slot& slot) const {
    return runner_->BindComponentState(*resources_->adapter, slot);
  }

  [[nodiscard]] ext::TextRunnerResourceClaim Claim() const {
    return runner_->ResourceClaim();
  }

private:
  std::shared_ptr<ext::TextModelRunner> runner_;
  std::unique_ptr<cache::ResourceLedger> ledger_;
  std::unique_ptr<ext::ComponentCacheResources> resources_;
  std::unique_ptr<NoGuard> guard_;
  std::unique_ptr<cache::Slot> slot_;
};

sampling::SamplingConfig Greedy() { return {}; }

sampling::SamplingConfig Sampled(float temperature, std::uint64_t seed) {
  sampling::SamplingConfig config;
  config.temperature = temperature;
  config.seed = static_cast<std::int64_t>(seed);
  return config;
}

bool SameTokens(const RunResult& a, const RunResult& b) {
  return a.tokens == b.tokens;
}

void Report(const std::string& label, const std::string& id,
            const RunResult& a, const RunResult& b, bool same) {
  std::cout << (same ? "PASS " : "FAIL ") << label << " " << id
            << " tokens=" << a.tokens.size() << "/" << b.tokens.size()
            << "\n";
}

struct Job {
  const Prompt* prompt;
  sampling::SamplingConfig config;
  // Optional structured-output grammar, bound per runner by RunConfig.
  std::shared_ptr<const sampling::JsonConstraint> grammar;
};

// Context for a prompt of `depth` tokens: the prompt, the generated tokens and
// a margin. Depth 0 keeps the historical 8192 so existing numbers hold.
std::uint32_t ContextFor(std::size_t depth, std::size_t max_tokens) {
  if (depth == 0)
    return 8192;
  return static_cast<std::uint32_t>(depth + max_tokens + 256);
}

// Runs every job on one harness, then destroys it. OFF and ON are never
// resident together: two 25 GB target models stall HIP allocation and upload.
std::vector<RunResult> RunConfig(const char* spec_value, const Options& options,
                                 const std::vector<Job>& jobs,
                                 std::size_t max_tokens,
                                 const DraftSettings& draft = {}) {
  std::vector<RunResult> results;
  results.reserve(jobs.size());
  Harness harness(spec_value, options.target, draft,
                  ContextFor(options.depth, max_tokens));
  for (const auto& job : jobs) {
    auto config = job.config;
    if (job.grammar)
      config.constraint = harness.Bind(job.grammar);
    results.push_back(harness.Run(*job.prompt, config, max_tokens));
  }
  return results;
}

// GX: greedy spec-on must equal spec-off, prompt by prompt.
int RunGx(const Options& options, const std::vector<Prompt>& prompts) {
  std::vector<Job> jobs;
  for (const auto& prompt : prompts)
    jobs.push_back({&prompt, Greedy(), nullptr});
  const auto offs = RunConfig("off", options, jobs, options.max_tokens);
  const auto ons = RunConfig("dflash", options, jobs, options.max_tokens);
  std::size_t failures = 0;
  for (std::size_t i = 0; i < prompts.size(); ++i) {
    const auto& reference = offs[i];
    const auto& spec = ons[i];
    const bool same = SameTokens(reference, spec);
    if (!same)
      ++failures;
    Report("gx", prompts[i].id, reference, spec, same);
  }
  std::cout << "gx: " << (prompts.size() - failures) << "/" << prompts.size()
            << " byte-identical\n";
  return failures == 0 ? 0 : 1;
}

// Null drafter: byte-identical to spec-off at every temperature and seed.
int RunNull(const Options& options, const std::vector<Prompt>& prompts) {
  std::vector<Job> jobs;
  for (const float temperature : options.temps) {
    for (const std::uint64_t seed : options.seeds) {
      for (const auto& prompt : prompts) {
        jobs.push_back({&prompt, temperature == 0.0F
                                     ? Greedy()
                                     : Sampled(temperature, seed),
                        nullptr});
      }
    }
  }
  const auto offs = RunConfig("off", options, jobs, options.max_tokens);
  const auto nulls = RunConfig("null", options, jobs, options.max_tokens);
  std::size_t runs = 0;
  std::size_t failures = 0;
  for (std::size_t i = 0; i < jobs.size(); ++i) {
    const auto& reference = offs[i];
    const auto& spec = nulls[i];
    const bool same = SameTokens(reference, spec);
    ++runs;
    if (!same)
      ++failures;
    std::cout << (same ? "PASS " : "FAIL ") << "null temp="
              << jobs[i].config.temperature << " seed=" << jobs[i].config.seed
              << " " << jobs[i].prompt->id
              << " tokens=" << reference.tokens.size() << "/"
              << spec.tokens.size() << "\n";
  }
  std::cout << "null: " << (runs - failures) << "/" << runs
            << " byte-identical\n";
  return failures == 0 ? 0 : 1;
}

void ReportSpeed(const Options& options, const std::vector<RunResult>& offs,
                 const std::vector<RunResult>& ons);

// One sampler setting: the job config it applies, and the check it must pass.
// `equals_greedy` settings must also match plain greedy output; `penalty`
// settings report how often they change it, as evidence the penalty applies.
struct Setting {
  std::string name;
  sampling::SamplingConfig config;
  std::shared_ptr<const sampling::JsonConstraint> grammar;
  bool equals_greedy{false};
  bool penalty{false};
};

// Grammar: a JSON object with one required string member "answer".
std::shared_ptr<const sampling::JsonConstraint> AnswerGrammar() {
  const auto schema = gufo::json::parse(
      R"({"type":"object","properties":{"answer":{"type":"string"}},)"
      R"("required":["answer"],"additionalProperties":false})");
  return sampling::JsonConstraint::Compile(schema, true);
}

// True when the text parses as a JSON object whose only member is "answer".
bool ValidAnswer(const std::string& text) {
  try {
    const auto value = gufo::json::parse(text);
    return value.is_object() && value.contains("answer") &&
           value.members().size() == 1;
  } catch (const std::exception&) {
    return false;
  }
}

// Settings honour. Each setting runs spec-on and spec-off with the same
// sampler config. Identity is required for every setting. top-k 1 and a tiny
// top-p at temperature 0.7 must reproduce plain greedy, which shows the filter
// is applied and not merely equal on both sides. The grammar output must
// parse and validate.
int RunSettings(const Options& options, const std::vector<Prompt>& prompts,
                const qwen::QwenTokenizer& tokenizer) {
  std::vector<Setting> settings;
  {
    Setting s{"top_k=40 temp=0.7", Sampled(0.7F, 1), nullptr};
    s.config.top_k = 40;
    settings.push_back(s);
  }
  {
    Setting s{"top_k=1 temp=0.7", Sampled(0.7F, 1), nullptr, true};
    s.config.top_k = 1;
    settings.push_back(s);
  }
  {
    Setting s{"top_p=0.9 temp=0.7", Sampled(0.7F, 1), nullptr};
    s.config.top_p = 0.9F;
    settings.push_back(s);
  }
  {
    Setting s{"top_p=1e-6 temp=0.7", Sampled(0.7F, 1), nullptr, true};
    s.config.top_p = 1e-6F;
    settings.push_back(s);
  }
  {
    Setting s{"min_p=0.05 temp=0.7", Sampled(0.7F, 1), nullptr};
    s.config.min_p = 0.05F;
    settings.push_back(s);
  }
  {
    Setting s{"repeat_penalty=1.15 temp=0", Greedy(), nullptr, false, true};
    s.config.repeat_penalty = 1.15F;
    settings.push_back(s);
  }
  {
    Setting s{"frequency_penalty=0.5 temp=0", Greedy(), nullptr, false, true};
    s.config.frequency_penalty = 0.5F;
    settings.push_back(s);
  }
  settings.push_back({"grammar answer-object temp=0", Greedy(), AnswerGrammar()});

  // Settings filter: keep only the settings whose name contains the substring.
  if (!options.settings_filter.empty()) {
    settings.erase(std::remove_if(settings.begin(), settings.end(),
                                  [&](const Setting& setting) {
                                    return setting.name.find(
                                               options.settings_filter) ==
                                           std::string::npos;
                                  }),
                   settings.end());
    if (settings.empty())
      Fail("--settings matches no setting: " + options.settings_filter);
  }

  // Plain greedy (no settings) is the reference for equals_greedy and penalty.
  std::vector<Job> jobs;
  for (const auto& setting : settings)
    for (const auto& prompt : prompts)
      jobs.push_back({&prompt, setting.config, setting.grammar});
  for (const auto& prompt : prompts)
    jobs.push_back({&prompt, Greedy(), nullptr});

  const auto offs = RunConfig("off", options, jobs, options.max_tokens);
  const auto ons = RunConfig("dflash", options, jobs, options.max_tokens);

  const std::size_t n = prompts.size();
  const std::size_t plain = settings.size() * n;
  std::size_t failures = 0;
  for (std::size_t s = 0; s < settings.size(); ++s) {
    const auto& setting = settings[s];
    std::size_t identical = 0;
    std::size_t violations = 0;
    std::size_t not_evaluated = 0;
    std::size_t greedy_mismatch = 0;
    std::size_t differs_from_plain = 0;
    for (std::size_t p = 0; p < n; ++p) {
      const std::size_t i = s * n + p;
      const auto& reference = offs[i];
      const auto& spec = ons[i];
      if (SameTokens(reference, spec))
        ++identical;
      if (setting.grammar && !ValidAnswer(tokenizer.Decode(reference.tokens))) {
        // End reason: no runner stop means the output was cut by --max-tokens.
        const bool budget = !reference.stopped;
        const std::string text = tokenizer.Decode(reference.tokens);
        const std::string tail =
            text.size() > 80 ? text.substr(text.size() - 80) : text;
        // D1: a budget-cut output is not a grammar violation; it is not evaluated.
        std::cout << "  " << (budget ? "not_evaluated " : "violation ")
                  << prompts[p].id
                  << " ended_by=" << (budget ? "budget" : "stop")
                  << " tokens=" << reference.tokens.size()
                  << " tail=" << std::quoted(tail) << "\n";
        if (budget)
          ++not_evaluated;
        else
          ++violations;
      }
      if (!SameTokens(offs[plain + p], reference)) {
        if (setting.equals_greedy)
          ++greedy_mismatch;
        ++differs_from_plain;
      }
    }
    const bool pass = identical == n && violations == 0 &&
                      greedy_mismatch == 0;
    if (!pass)
      ++failures;
    std::cout << (pass ? "PASS " : "FAIL ") << "settings " << setting.name
              << " identical=" << identical << "/" << n;
    if (setting.grammar)
      std::cout << " violations=" << violations << "/" << n
                << " not_evaluated=" << not_evaluated << "/" << n;
    if (setting.equals_greedy)
      std::cout << " differs_from_greedy=" << greedy_mismatch << "/" << n;
    if (setting.penalty)
      std::cout << " differs_from_plain=" << differs_from_plain << "/" << n;
    std::cout << "\n";
  }
  std::cout << "settings: " << (settings.size() - failures) << "/"
            << settings.size() << " pass\n";
  return failures == 0 ? 0 : 1;
}

// Memory accounting: for each mode, the runner's claim (resident + per-request)
// against the device memory actually taken by runner creation, one slot, and
// one bound request state (hipMemGetInfo before and after each step).
int RunMemory(const Options& options) {
  struct Mode {
    const char* label;
    const char* spec;
  };
  const Mode modes[] = {{"spec-off", "off"}, {"spec-on", "dflash"}};
  const auto context = static_cast<std::uint32_t>(
      ContextFor(options.depth, options.max_tokens));
  const auto free_now = [] {
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    if (hipMemGetInfo(&free_bytes, &total_bytes) != hipSuccess)
      Fail("hipMemGetInfo failed");
    return free_bytes;
  };
  // One-time device costs (HIP context, kernel module load, MMQ scratch) would
  // otherwise land in whichever mode runs first. Prime them with a throwaway
  // runner so both measured deltas cover only that mode's allocations.
  { Harness prime("off", options.target, DraftSettings{}, context); }
  // The host sizes concurrency from per_request_state_bytes, so the gate
  // checks the incremental cost of a session. The first-slot total includes
  // one-time process overhead and is reported for information only.
  std::size_t failures = 0;
  for (const auto& mode : modes) {
    const std::size_t before = free_now();
    Harness harness(mode.spec, options.target, DraftSettings{}, context);
    const std::size_t after_harness = free_now();
    const auto state1 = harness.BindState();
    const std::size_t slot1_free = free_now();
    const auto claim = harness.Claim();
    const std::size_t resident_claim = claim.resident_weights_bytes.value_or(0);
    const std::size_t per_request_claim =
        claim.per_request_state_bytes.value_or(0);
    const std::size_t first_slot_total = before - slot1_free;

    // Declared after the harness so they are destroyed before it.
    auto slot2 = harness.NewSlot();
    const auto state2 = harness.BindStateOn(*slot2);
    const std::size_t slot2_free = free_now();
    const std::size_t per_request_measured = slot1_free - slot2_free;

    const bool pass = per_request_claim >= per_request_measured;
    if (!pass)
      ++failures;
    std::cout << (pass ? "PASS " : "FAIL ") << "memory " << mode.label
              << " per_request_claimed=" << per_request_claim
              << " per_request_measured=" << per_request_measured
              << " (second slot)\n"
              << "  info " << mode.label
              << " resident_weights_claimed=" << resident_claim
              << " first_slot_total_delta=" << first_slot_total
              << " (includes one-time overhead; harness="
              << before - after_harness
              << " bind=" << after_harness - slot1_free << ")\n";
  }
  std::cout << "memory: " << (2 - failures) << "/2 pass\n";
  return failures == 0 ? 0 : 1;
}

// Speedup and tokens per cycle at each temperature, over the prompt set.
int RunSpeed(const Options& options, const std::vector<Prompt>& prompts) {
  std::vector<Job> jobs;
  for (const float temperature : options.temps) {
    for (const auto& prompt : prompts) {
      jobs.push_back({&prompt, temperature == 0.0F ? Greedy()
                                                   : Sampled(temperature, 1),
                      nullptr});
    }
  }
  const auto offs = RunConfig("off", options, jobs, options.max_tokens);
  for (const auto policy : options.policies) {
    const DraftSettings draft{policy, options.max_draft, options.min_draft};
    const auto ons = RunConfig("dflash", options, jobs, options.max_tokens,
                               draft);
    std::cout << "policy=" << ext::TextDFlashDraftPolicyName(policy)
              << " max_draft=" << draft.max_draft
              << " min_draft=" << draft.min_draft << "\n";
    ReportSpeed(options, offs, ons);
  }
  return 0;
}

// Per-temperature summary of one policy's dflash run against spec-off.
void ReportSpeed(const Options& options, const std::vector<RunResult>& offs,
                 const std::vector<RunResult>& ons) {
  const std::size_t prompts = offs.size() / options.temps.size();
  std::size_t index = 0;
  for (const float temperature : options.temps) {
    double off_seconds = 0.0;
    double on_seconds = 0.0;
    double off_decode = 0.0;
    double on_decode = 0.0;
    std::size_t off_tokens = 0;
    std::size_t on_tokens = 0;
    std::size_t rounds = 0;
    std::size_t drafted = 0;
    std::size_t accepted = 0;
    for (std::size_t p = 0; p < prompts; ++p, ++index) {
      const auto& reference = offs[index];
      const auto& spec = ons[index];
      off_seconds += reference.seconds;
      on_seconds += spec.seconds;
      off_decode += reference.decode_seconds;
      on_decode += spec.decode_seconds;
      off_tokens += reference.tokens.size();
      on_tokens += spec.tokens.size();
      rounds += spec.draft_rounds;
      drafted += spec.draft_tokens;
      accepted += spec.draft_accepted_tokens;
    }
    const double off_rate = off_tokens / off_seconds;
    const double on_rate = on_tokens / on_seconds;
    std::cout << std::fixed << std::setprecision(3)
              << "speed temp=" << temperature << " depth=" << options.depth
              << " tokens_off=" << off_tokens
              << " tokens_on=" << on_tokens << " tps_off=" << off_rate
              << " tps_on=" << on_rate << " speedup=" << on_rate / off_rate
              << " decode_tps_off=" << off_tokens / off_decode
              << " decode_tps_on=" << on_tokens / on_decode
              << " decode_speedup=" << (on_tokens / on_decode) /
                                           (off_tokens / off_decode)
              << " cycles=" << rounds << " tokens_per_cycle="
              << (rounds == 0 ? 0.0
                              : static_cast<double>(on_tokens) / rounds)
              << " accept_rate="
              << (drafted == 0 ? 0.0
                               : static_cast<double>(accepted) / drafted)
              << " mean_draft="
              << (rounds == 0 ? 0.0 : static_cast<double>(drafted) / rounds)
              << "\n";
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = ParseArgs(argc, argv);
    std::string error;
    auto reader = core::GgufReader::OpenFile(options.target, &error);
    if (!reader)
      Fail("target GGUF: " + error);
    auto tokenizer = qwen::QwenTokenizer::CreateFromGguf(*reader, &error);
    if (!tokenizer)
      Fail("tokenizer: " + error);
    if (options.mode == "memory")
      return RunMemory(options);
    const auto prompts = ReadPrompts(options.prompts, *tokenizer,
                                     options.prompts_limit, options.depth);
    if (options.mode == "gx")
      return RunGx(options, prompts);
    if (options.mode == "null")
      return RunNull(options, prompts);
    if (options.mode == "speed")
      return RunSpeed(options, prompts);
    if (options.mode == "settings")
      return RunSettings(options, prompts, *tokenizer);
    Fail("--mode must be gx, null, speed, settings or memory");
  } catch (const std::exception& failure) {
    std::cerr << "flint_spec_gate: " << failure.what() << "\n";
    return 2;
  }
}
