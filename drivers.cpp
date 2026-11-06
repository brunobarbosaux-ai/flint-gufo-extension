#include "drivers.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <gufo/ext/gguf_reader.hpp>
#include <gufo/ext/sampling.hpp>
#include <gufo/ext/text.hpp>
#include <iostream>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "hip/vendor/qwen/tokenizer.hpp"
#include "src/chat_template.hpp"
#include "src/gpu/engine.hpp"
#include "src/runtime/config.hpp"

// CLI drivers. Since extension API series 0.2 removed
// TextModelRunner::CreateState, a driver owns its execution slots directly: it
// loads the model, creates a session and drives the engine's Append/Logits,
// mirroring the LFM3 extension. The runner contract is exercised by the host's
// HTTP/serve path instead.
namespace gufo::models::flint {
namespace {

using Clock = std::chrono::steady_clock;
using FlintModel = gpu::Model;
using FlintSession = gpu::Session;

constexpr std::uint32_t kDriverContext = 8192;

[[noreturn]] void Fail(const std::string& message) {
  throw std::runtime_error(message);
}

double Seconds(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

[[nodiscard]] std::vector<std::int32_t> EngineTokens(
    std::span<const std::uint32_t> tokens) {
  std::vector<std::int32_t> converted;
  converted.reserve(tokens.size());
  for (const auto token : tokens)
    converted.push_back(static_cast<std::int32_t>(token));
  return converted;
}

// The engine, tokenizer and config a driver needs, loaded once.
struct Loaded {
  std::shared_ptr<FlintModel> model;
  std::unique_ptr<gufo::vendor::qwen::tokenization::QwenTokenizer> tokenizer;
};

[[nodiscard]] Loaded Load(const std::string& model_path,
                          Clock::time_point load_start) {
  std::string error;
  auto model = FlintModel::Load(model_path, {}, &error);
  if (model == nullptr)
    Fail("Failed to create Flint model: " + error);
  auto metadata = core::GgufReader::OpenFile(model_path, &error);
  if (!metadata)
    Fail("Flint GGUF open failed: " + error);
  auto tokenizer =
      gufo::vendor::qwen::tokenization::QwenTokenizer::CreateFromGguf(*metadata,
                                                                      &error);
  if (tokenizer == nullptr)
    Fail("Flint tokenizer failed: " + error);
  const auto load =
      load_start == Clock::time_point{} ? Clock::now() : load_start;
  std::cerr << "[ModelLoad]: " << Seconds(load) << " s\n";
  return {std::move(model), std::move(tokenizer)};
}

// Appends the whole prompt and samples up to `max_tokens` tokens, streaming
// pieces to stdout. Returns the generated ids.
[[nodiscard]] std::vector<std::uint32_t> Generate(
    FlintSession& session, const sampling::SamplingConfig& sampling,
    const std::vector<std::uint32_t>& prompt, std::size_t max_tokens,
    std::size_t context,
    const gufo::vendor::qwen::tokenization::QwenTokenizer& tokenizer) {
  std::string error;
  const auto chunk = EngineTokens(prompt);
  std::size_t cursor = 0;
  while (cursor < chunk.size()) {
    const auto take = std::min<std::size_t>(4096, chunk.size() - cursor);
    if (!session.Append(
            std::span<const std::int32_t>(chunk).subspan(cursor, take), &error))
      Fail("Flint prefill failed: " + error);
    cursor += take;
  }
  sampling::SamplerState sampler(sampling, prompt);
  std::vector<std::uint32_t> generated;
  for (std::size_t i = 0; i < max_tokens; ++i) {
    if (session.Position() >= context)
      break;
    const auto logits = session.Logits();
    if (logits.empty())
      Fail("Flint token selection has no logits");
    const auto token = static_cast<std::uint32_t>(sampler.Sample(logits));
    generated.push_back(token);
    sampler.Accept(token);
    std::cout << tokenizer.DecodeToken(token) << std::flush;
    const std::int32_t input = static_cast<std::int32_t>(token);
    if (!session.Append({&input, 1}, &error))
      Fail("Flint decode failed: " + error);
  }
  std::cout << '\n';
  return generated;
}

int RunPromptImpl(const ext::TextPromptOptions& opt, bool interactive,
                  Clock::time_point load_start) {
  try {
    if (opt.force_cpu || !opt.speculative_backend.empty())
      Fail("Flint supports autoregressive HIP inference");
    if (opt.reasoning_mode != "auto" && opt.reasoning_mode != "on" &&
        opt.reasoning_mode != "off")
      Fail("Flint reasoning must be auto, on, or off");
    if (opt.reasoning_effort != "auto" || opt.preserve_thinking != "auto")
      Fail(
          "Flint does not support reasoning effort or preserve-thinking "
          "overrides");
    if (!opt.image_paths.empty())
      Fail("Flint does not support image input");

    auto loaded = Load(opt.model_path, load_start);
    std::string error;
    auto session = loaded.model->CreateSession(kDriverContext, &error);
    if (session == nullptr)
      Fail("Failed to create Flint session: " + error);

    std::vector<ChatMessage> history;
    if (!opt.system_prompt.empty())
      history.push_back({"system", opt.system_prompt, ""});
    std::string input = opt.prompt_text;
    do {
      if (interactive) {
        std::cout << ">>> User: " << std::flush;
        if (!std::getline(std::cin, input) || input == "exit")
          break;
      }
      history.push_back({"user", input, ""});
      ChatTemplateOptions template_options;
      template_options.enable_thinking = opt.reasoning_mode != "off";
      std::vector<std::uint32_t> tokens;
      if (opt.use_chat_template) {
        gufo::vendor::qwen::tokenization::TokenizerOptions encode_options;
        encode_options.parse_special_tokens = true;
        tokens = loaded.tokenizer->Encode(
            RenderPrompt(history, template_options), encode_options);
      } else {
        tokens = loaded.tokenizer->Encode(input);
      }
      if (tokens.empty())
        Fail("Flint prompt produced no tokens");

      session->Reset();
      const auto prefill = Clock::now();
      const auto generated =
          Generate(*session, opt.sampling, tokens, opt.max_tokens,
                   kDriverContext, *loaded.tokenizer);
      const auto total = Seconds(prefill);
      if (opt.verbose)
        std::cerr << "Flint " << tokens.size() << " prompt tokens, "
                  << generated.size() << " generated, " << total << " s\n";
      history.push_back({"assistant", loaded.tokenizer->Decode(generated), ""});
    } while (interactive);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Flint: " << e.what() << '\n';
    return 1;
  }
}

}  // namespace

int RunFlintPrompt(const ext::TextPromptOptions& opt, const core::GgufReader&,
                   std::chrono::steady_clock::time_point load_start) {
  return RunPromptImpl(opt, false, load_start);
}

int RunFlintChat(const ext::TextPromptOptions& opt, const core::GgufReader&,
                 std::chrono::steady_clock::time_point load_start) {
  return RunPromptImpl(opt, true, load_start);
}

int RunFlintBench(const ext::TextBenchOptions& opt,
                  const std::shared_ptr<const core::GgufReader>&,
                  std::chrono::steady_clock::time_point load_start) {
  try {
    if (opt.concurrency != std::vector<std::size_t>{1} ||
        !opt.speculative_backend.empty() || opt.enable_yarn ||
        opt.validate_prefill_tokens)
      Fail("Flint bench supports C1 autoregressive timing only");

    auto loaded = Load(opt.model_path, load_start);
    const auto max = [](const auto& v) {
      return v.empty() ? std::size_t(0) : *std::max_element(v.begin(), v.end());
    };
    const std::size_t context =
        max(opt.n_depths) + std::max(max(opt.n_prompts), max(opt.n_gens)) + 1;
    if (context > kDriverContext)
      Fail("Flint benchmark exceeds the driver context");
    std::string error;
    auto session = loaded.model->CreateSession(
        static_cast<std::uint32_t>(context), &error);
    if (session == nullptr)
      Fail("Failed to create Flint session: " + error);

    const auto seed = loaded.tokenizer->Encode(
        "The quick brown fox jumps over the lazy dog. ");
    if (seed.empty())
      Fail("Flint bench seed text produced no tokens");
    const auto fill = [&](std::size_t n) {
      std::vector<std::uint32_t> ids(n);
      for (std::size_t i = 0; i < n; ++i)
        ids[i] = seed[i % seed.size()];
      return ids;
    };
    const auto append_all = [&](const std::vector<std::uint32_t>& ids) {
      const auto chunk = EngineTokens(ids);
      if (!session->Append(chunk, &error))
        Fail("Flint bench forward failed: " + error);
    };

    std::cout << "| model | test | depth | repetition | tokens/s | seconds "
                 "|\n| :-- | :-- | --: | --: | --: | --: |\n";
    for (const auto depth : opt.n_depths) {
      const auto prepare = [&] {
        session->Reset();
        if (depth)
          append_all(fill(depth));
      };
      // Warm the prompt shapes before timing.
      for (const auto n : opt.n_prompts) {
        if (!n)
          continue;
        prepare();
        append_all(fill(n));
      }
      for (std::size_t r = 0; r < opt.repetitions; ++r) {
        for (const auto n : opt.n_prompts) {
          if (!n)
            continue;
          prepare();
          const auto prompt = fill(n);
          const auto start = Clock::now();
          append_all(prompt);
          const auto dt = Seconds(start);
          std::cout << "| Flint | pp" << n << " | " << depth << " | " << r
                    << " | " << n / dt << " | " << dt << " |\n";
        }
        for (const auto n : opt.n_gens) {
          if (!n)
            continue;
          prepare();
          const auto first = fill(1);
          append_all(first);
          sampling::SamplerState sampler(opt.sampling, first);
          const auto start = Clock::now();
          std::size_t generated = 0;
          for (std::size_t i = 0; i < n; ++i) {
            const auto logits = session->Logits();
            if (logits.empty())
              break;
            const auto token =
                static_cast<std::uint32_t>(sampler.Sample(logits));
            sampler.Accept(token);
            const std::int32_t input = static_cast<std::int32_t>(token);
            if (!session->Append({&input, 1}, &error))
              Fail("Flint bench decode failed: " + error);
            ++generated;
          }
          const auto dt = Seconds(start);
          std::cout << "| Flint | tg" << n << " | " << depth << " | " << r
                    << " | " << generated / dt << " | " << dt << " |\n";
        }
      }
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Flint: " << e.what() << '\n';
    return 1;
  }
}

}  // namespace gufo::models::flint
