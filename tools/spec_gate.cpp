// M3 gates for the DFlash2 speculative cycle, driven through the runner's own
// text-runner contract (the calls the host makes): CreateFlintTextRunner ->
// CreateComponentCacheResources -> CreateSlot -> BindComponentState -> Prefill
// -> DecodeStep. Speculation is selected by FLINT_SPEC, read when the runner
// is created, so each configuration gets its own runner.
//
//   --mode gx       greedy spec-on (dflash) byte-identical to spec-off
//   --mode null     FLINT_SPEC=null byte-identical to spec-off at each
//                   temperature and seed
//   --mode speed    tokens per cycle and speedup of dflash over spec-off
//
// Usage: flint_spec_gate --target <target.gguf> --prompts <prompts.jsonl>
//          --mode gx|null|speed [--max-tokens N] [--temps 0,0.6,1.0]
//          [--seeds 1,2,3] [--prompts-limit N]
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

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
    } else if (arg == "--seeds") {
      options.seeds.clear();
      for (const auto& part : Split(next("--seeds")))
        options.seeds.push_back(std::stoull(part));
    } else {
      Fail("unknown argument " + arg);
    }
  }
  if (options.target.empty() || options.prompts.empty() ||
      options.mode.empty())
    Fail("--target, --prompts and --mode are required");
  return options;
}

struct Prompt {
  std::string id;
  std::vector<std::uint32_t> tokens;
};

// Chat prompts through the chat template with thinking disabled, the same
// encoding the acceptance workload was measured with.
std::vector<Prompt> ReadPrompts(const std::string& path,
                                const qwen::QwenTokenizer& tokenizer,
                                std::size_t limit) {
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
    Prompt prompt;
    prompt.id = spec.member_str("id");
    prompt.tokens.assign(ids.begin(), ids.end());
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
  double seconds{0.0};
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
  explicit Harness(const std::string& spec_value, const std::string& target) {
    if (spec_value.empty())
      unsetenv("FLINT_SPEC");
    else
      setenv("FLINT_SPEC", spec_value.c_str(), 1);
    ext::TextRunnerOptions options;
    options.model_path = target;
    options.max_context = 8192;
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
      const auto step = runner_->Prefill(*state, tokens, offset, tokens.size());
      offset += step.consumed_tokens;
      if (step.decode_ready)
        break;
    }

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
      if (decode.stop || decode.selections.empty())
        break;
    }
    result.seconds = std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - started)
                         .count();
    return result;
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
};

// Runs every job on one harness, then destroys it. OFF and ON are never
// resident together: two 25 GB target models stall HIP allocation and upload.
std::vector<RunResult> RunConfig(const char* spec_value, const Options& options,
                                 const std::vector<Job>& jobs,
                                 std::size_t max_tokens) {
  std::vector<RunResult> results;
  results.reserve(jobs.size());
  Harness harness(spec_value, options.target);
  for (const auto& job : jobs)
    results.push_back(harness.Run(*job.prompt, job.config, max_tokens));
  return results;
}

// GX: greedy spec-on must equal spec-off, prompt by prompt.
int RunGx(const Options& options, const std::vector<Prompt>& prompts) {
  std::vector<Job> jobs;
  for (const auto& prompt : prompts)
    jobs.push_back({&prompt, Greedy()});
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
                                     : Sampled(temperature, seed)});
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

// Speedup and tokens per cycle at each temperature, over the prompt set.
int RunSpeed(const Options& options, const std::vector<Prompt>& prompts) {
  std::vector<Job> jobs;
  for (const float temperature : options.temps) {
    for (const auto& prompt : prompts) {
      jobs.push_back({&prompt, temperature == 0.0F ? Greedy()
                                                   : Sampled(temperature, 1)});
    }
  }
  const auto offs = RunConfig("off", options, jobs, options.max_tokens);
  const auto ons = RunConfig("dflash", options, jobs, options.max_tokens);
  std::size_t index = 0;
  for (const float temperature : options.temps) {
    double off_seconds = 0.0;
    double on_seconds = 0.0;
    std::size_t off_tokens = 0;
    std::size_t on_tokens = 0;
    std::size_t rounds = 0;
    std::size_t drafted = 0;
    std::size_t accepted = 0;
    for (std::size_t p = 0; p < prompts.size(); ++p, ++index) {
      const auto& reference = offs[index];
      const auto& spec = ons[index];
      off_seconds += reference.seconds;
      on_seconds += spec.seconds;
      off_tokens += reference.tokens.size();
      on_tokens += spec.tokens.size();
      rounds += spec.draft_rounds;
      drafted += spec.draft_tokens;
      accepted += spec.draft_accepted_tokens;
    }
    const double off_rate = off_tokens / off_seconds;
    const double on_rate = on_tokens / on_seconds;
    std::cout << std::fixed << std::setprecision(3)
              << "speed temp=" << temperature << " tokens_off=" << off_tokens
              << " tokens_on=" << on_tokens << " tps_off=" << off_rate
              << " tps_on=" << on_rate << " speedup=" << on_rate / off_rate
              << " cycles=" << rounds << " tokens_per_cycle="
              << (rounds == 0 ? 0.0
                              : static_cast<double>(on_tokens) / rounds)
              << " accept_rate="
              << (drafted == 0 ? 0.0
                               : static_cast<double>(accepted) / drafted)
              << "\n";
  }
  return 0;
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
    const auto prompts =
        ReadPrompts(options.prompts, *tokenizer, options.prompts_limit);
    if (options.mode == "gx")
      return RunGx(options, prompts);
    if (options.mode == "null")
      return RunNull(options, prompts);
    if (options.mode == "speed")
      return RunSpeed(options, prompts);
    Fail("--mode must be gx, null or speed");
  } catch (const std::exception& failure) {
    std::cerr << "flint_spec_gate: " << failure.what() << "\n";
    return 2;
  }
}
