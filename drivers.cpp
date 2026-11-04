#include "drivers.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <gufo/ext/text.hpp>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "runner.hpp"
#include "src/chat_template.hpp"

namespace gufo::models::flint {
namespace {

using Clock = std::chrono::steady_clock;

double Seconds(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

[[noreturn]] void Fail(const std::string& message) {
  throw std::runtime_error(message);
}

std::shared_ptr<ext::TextModelRunner> LoadRunner(
    const ext::TextRunnerOptions& options, Clock::time_point load_start) {
  std::string error;
  const auto load =
      load_start == Clock::time_point{} ? Clock::now() : load_start;
  auto runner = CreateFlintTextRunner(options, &error);
  if (!runner)
    Fail(error);
  std::cerr << "[ModelLoad]: " << Seconds(load) << " s\n";
  return runner;
}

ext::TextRunnerOptions RunnerOptions(const std::string& model_path,
                                     std::uint32_t max_context) {
  ext::TextRunnerOptions options;
  options.model_path = model_path;
  options.max_context = max_context;
  return options;
}

// Greedy-decodes `prompt` on `runner`, streaming pieces to stdout.
int RunPrompt(const ext::TextPromptOptions& opt, Clock::time_point load_start,
              bool interactive) {
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

    auto runner = LoadRunner(RunnerOptions(opt.model_path, 8192), load_start);
    auto state = runner->CreateState();
    std::string input = opt.prompt_text;
    do {
      if (interactive) {
        std::cout << ">>> User: " << std::flush;
        if (!std::getline(std::cin, input) || input == "exit")
          break;
      }
      ext::ChatRequest request;
      if (!opt.system_prompt.empty())
        request.messages.emplace_back(ext::ChatRole::kSystem,
                                      opt.system_prompt);
      request.messages.emplace_back(ext::ChatRole::kUser, input);
      std::optional<std::vector<ext::TextRunnerToken>> tokens =
          opt.use_chat_template
              ? runner->RenderAndTokenize(request)
              : std::optional<std::vector<ext::TextRunnerToken>>(
                    runner->Tokenize(input));
      if (!tokens || tokens->empty())
        Fail("Flint prompt produced no tokens");

      const auto prefill = Clock::now();
      state->Invalidate();
      std::size_t cursor = 0;
      while (cursor < tokens->size()) {
        const auto step = runner->Prefill(*state, *tokens, cursor, 4096);
        if (step.consumed_tokens == 0)
          Fail("Flint prefill made no progress");
        cursor += step.consumed_tokens;
      }
      const auto pp = Seconds(prefill);

      sampling::SamplerState sampler(opt.sampling, *tokens);
      std::vector<ext::TextRunnerToken> generated;
      const auto decode = Clock::now();
      for (std::size_t i = 0; i < opt.max_tokens; ++i) {
        const auto selection = runner->SelectNext(*state, sampler);
        if (selection.stop)
          break;
        generated.push_back(selection.token);
        sampler.Accept(selection.token);
        std::cout << selection.piece << std::flush;
        runner->Advance(*state, selection.token);
      }
      const auto tg = Seconds(decode);
      std::cout << '\n';
      if (opt.verbose)
        std::cerr << "Flint pp " << tokens->size() << " tokens " << pp
                  << " s; tg " << generated.size() << " tokens " << tg
                  << " s\n";
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
  return RunPrompt(opt, load_start, false);
}

int RunFlintChat(const ext::TextPromptOptions& opt, const core::GgufReader&,
                 std::chrono::steady_clock::time_point load_start) {
  return RunPrompt(opt, load_start, true);
}

int RunFlintBench(const ext::TextBenchOptions& opt,
                  const std::shared_ptr<const core::GgufReader>&,
                  std::chrono::steady_clock::time_point load_start) {
  try {
    if (opt.concurrency != std::vector<std::size_t>{1} ||
        !opt.speculative_backend.empty() || opt.enable_yarn ||
        opt.validate_prefill_tokens)
      Fail("Flint bench supports C1 autoregressive timing only");

    const auto max = [](const auto& v) {
      return v.empty() ? std::size_t(0) : *std::max_element(v.begin(), v.end());
    };
    const std::size_t context =
        max(opt.n_depths) + std::max(max(opt.n_prompts), max(opt.n_gens)) + 1;
    auto runner = LoadRunner(
        RunnerOptions(opt.model_path, static_cast<std::uint32_t>(context)),
        load_start);
    auto state = runner->CreateState();

    const auto seed =
        runner->Tokenize("The quick brown fox jumps over the lazy dog. ");
    if (seed.empty())
      Fail("Flint bench seed text produced no tokens");
    const auto fill = [&](std::size_t n) {
      std::vector<ext::TextRunnerToken> ids(n);
      for (std::size_t i = 0; i < n; ++i)
        ids[i] = seed[i % seed.size()];
      return ids;
    };
    // Prefill `ids` into `state` from its current position.
    const auto prefill_all = [&](const std::vector<ext::TextRunnerToken>& ids) {
      std::size_t cursor = 0;
      while (cursor < ids.size()) {
        const auto step = runner->Prefill(*state, ids, cursor, 4096);
        if (step.consumed_tokens == 0)
          Fail("Flint prefill made no progress");
        cursor += step.consumed_tokens;
      }
    };

    std::cout << "| model | test | depth | repetition | tokens/s | seconds "
                 "|\n| :-- | :-- | --: | --: | --: | --: |\n";
    for (const auto depth : opt.n_depths) {
      auto prepare = [&] {
        state->Invalidate();
        if (depth)
          prefill_all(fill(depth));
      };
      // Warm the prompt shapes before timing, as LFM3 does.
      for (const auto n : opt.n_prompts) {
        if (!n)
          continue;
        prepare();
        prefill_all(fill(n));
      }
      for (std::size_t r = 0; r < opt.repetitions; ++r) {
        for (const auto n : opt.n_prompts) {
          if (!n)
            continue;
          prepare();
          const auto prompt = fill(n);
          const auto start = Clock::now();
          prefill_all(prompt);
          const auto dt = Seconds(start);
          std::cout << "| Flint | pp" << n << " | " << depth << " | " << r
                    << " | " << n / dt << " | " << dt << " |\n";
        }
        for (const auto n : opt.n_gens) {
          if (!n)
            continue;
          prepare();
          const auto first = fill(1);
          prefill_all(first);
          sampling::SamplerState sampler(opt.sampling, first);
          const auto start = Clock::now();
          for (std::size_t i = 0; i < n; ++i) {
            const auto selection = runner->SelectNext(*state, sampler);
            if (selection.stop)
              break;
            sampler.Accept(selection.token);
            runner->Advance(*state, selection.token);
          }
          const auto dt = Seconds(start);
          std::cout << "| Flint | tg" << n << " | " << depth << " | " << r
                    << " | " << n / dt << " | " << dt << " |\n";
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
