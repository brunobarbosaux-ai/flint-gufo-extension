#include "drivers.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

// Session contexts are sized to what the request needs, clamped by the
// model's native context and the device memory actually free. KV caches
// grow linearly with the session length, so a fixed cap would either waste
// memory on short requests or refuse long ones the hardware can serve.
[[nodiscard]] std::uint32_t PickContext(const FlintModel& model,
                                        std::uint64_t wanted) {
  constexpr std::uint64_t kMinContext = 2048;
  constexpr std::uint64_t kFallbackContext = 262144;
  // Keep a margin for allocator overhead, workspace and transient buffers
  // beyond what SessionBytes accounts for.
  constexpr std::uint64_t kFreeMarginNum = 3;
  constexpr std::uint64_t kFreeMarginDen = 4;

  const std::uint64_t native = model.config().context_length != 0
                                   ? model.config().context_length
                                   : kFallbackContext;
  std::uint64_t budget = std::min(wanted, native);
  const std::uint64_t free_bytes = FlintModel::FreeDeviceBytes();
  if (free_bytes > 0) {
    const std::uint64_t usable = free_bytes * kFreeMarginNum / kFreeMarginDen;
    std::uint64_t feasible = budget;
    while (feasible > kMinContext &&
           model.SessionBytes(static_cast<std::uint32_t>(feasible)) > usable)
      feasible /= 2;
    budget = std::min(budget, feasible);
  }
  return static_cast<std::uint32_t>(std::max<std::uint64_t>(budget, 1));
}

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

    std::vector<ChatMessage> history;
    if (!opt.system_prompt.empty())
      history.push_back({"system", opt.system_prompt, ""});
    std::string input = opt.prompt_text;
    std::unique_ptr<FlintSession> session;
    std::uint32_t context = 0;
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

      // Size the session to this turn: prompt plus generation. A later chat
      // turn that needs more room recreates the session with a larger one.
      const std::uint32_t wanted = PickContext(
          *loaded.model,
          static_cast<std::uint64_t>(tokens.size()) + opt.max_tokens + 1);
      if (session == nullptr || wanted > context) {
        context = wanted;
        session = loaded.model->CreateSession(context, &error);
        if (session == nullptr)
          Fail("Failed to create Flint session: " + error);
      } else {
        session->Reset();
      }

      const auto prefill = Clock::now();
      const auto generated =
          Generate(*session, opt.sampling, tokens, opt.max_tokens,
                   context, *loaded.tokenizer);
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
    std::string error;
    auto session = loaded.model->CreateSession(
        PickContext(*loaded.model, static_cast<std::uint64_t>(context)),
        &error);
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

    // Verify-width bench (FLINT_BENCH_VERIFY=1): times AppendAll at every
    // width the engine supports, rewinding the cache between steps so each
    // step verifies the same positions. ms/step at width w over width 1 is the
    // speculative verify cost: a draft of w-1 tokens pays that ratio to yield
    // the accepted tokens of a single verify step. Ratios above the accepted
    // length per step mean speculation loses.
    if (const char* env = std::getenv("FLINT_BENCH_VERIFY");
        env != nullptr && env[0] != '\0' && env[0] != '0') {
      constexpr std::size_t kSteps = 64;
      // Varied rows, so the router spreads over experts the way real drafts
      // do: the bench seed cycle repeats one short pattern at every position.
      const auto pool = loaded.tokenizer->Encode(
          "Flint is a mixture-of-experts decoder: 49 layers, hidden 2048, 32 "
          "query heads and 4 KV heads of 128, 256 experts with 6 routed. "
          "Sliding-window attention of 513 covers 35 layers and the leading "
          "two layers are dense. The drafter reads five layer residuals and "
          "drafts an eight-token block in one forward pass. ");
      if (pool.empty())
        Fail("Flint verify bench text produced no tokens");

      std::cout << "\n| width | depth | ms/step | vs w1 | rows/s |\n"
                   "| --: | --: | --: | --: | --: |\n";
      for (const auto depth : opt.n_depths) {
        session->Reset();
        if (depth)
          append_all(fill(depth));
        const auto pos0 = session->Position();

        const std::uint32_t widths = FlintSession::kMaxVerifyRows;
        std::vector<double> ms(widths + 1, 0.0);
        for (std::uint32_t w = 1; w <= widths; ++w) {
          const auto rows = [&](std::size_t step) {
            std::vector<std::int32_t> ids(w);
            for (std::uint32_t i = 0; i < w; ++i)
              ids[i] = static_cast<std::int32_t>(
                  pool[(step * 13 + i * 7 + depth) % pool.size()]);
            return ids;
          };
          // The first call of a width runs eagerly and the second captures
          // its step graph, so both stay out of the timed loop.
          for (std::size_t k = 0; k < 2; ++k) {
            if (!session->AppendAll(rows(k), &error))
              Fail("Flint verify bench warmup failed: " + error);
            session->Rewind(pos0);
          }
          double best = 0.0;
          const std::size_t reps = std::max<std::size_t>(1, opt.repetitions);
          for (std::size_t r = 0; r < reps; ++r) {
            const auto start = Clock::now();
            for (std::size_t k = 0; k < kSteps; ++k) {
              if (!session->AppendAll(rows(k), &error))
                Fail("Flint verify bench failed: " + error);
              session->Rewind(pos0);
            }
            const double per_step = Seconds(start) / kSteps;
            best = best == 0.0 ? per_step : std::min(best, per_step);
          }
          ms[w] = best * 1000.0;
        }
        for (std::uint32_t w = 1; w <= widths; ++w)
          std::cout << "| " << w << " | " << depth << " | " << ms[w] << " | "
                    << ms[w] / ms[1] << " | " << w / (ms[w] / 1000.0) << " |\n";
      }
    }

    // Row-invariance check (FLINT_CHECK_ROWS=1): greedy speculation is only
    // lossless if the target's argmax does not depend on the row count of the
    // pass that produced it. At one cache position, the production single-token
    // Append must agree with row 0 of AppendAll(w) for every width w, because
    // the verifier compares draft tokens against exactly those rows. llama.cpp
    // fails this: its greedy output changes with ubatch size. Ours decides
    // whether a lossless gate is reachable.
    if (const char* env = std::getenv("FLINT_CHECK_ROWS");
        env != nullptr && env[0] != '\0' && env[0] != '0') {
      constexpr std::size_t kChecks = 256;
      const auto pool = loaded.tokenizer->Encode(
          "Row invariance decides whether verification can be lossless: the "
          "anchor row of a batched verify pass must return the same argmax as "
          "a single-token decode at the same position, or greedy drafts flip "
          "on near ties and the output diverges from the reference stream. ");
      if (pool.empty())
        Fail("Flint row check text produced no tokens");

      const std::uint32_t widths = FlintSession::kMaxVerifyRows;
      // FLINT_CHECK_ROWS=2 walks the widths in reverse: if the mismatches move
      // from the last calls to the first ones, the cause is state leaked across
      // calls rather than the row count itself.
      std::vector<std::uint32_t> order(widths);
      for (std::uint32_t w = 1; w <= widths; ++w)
        order[w - 1] = w;
      if (env[0] == '2')
        std::reverse(order.begin(), order.end());
      // FLINT_CHECK_ROWS=3 changes only the trailing rows of the same verify
      // width: row 0 must not depend on them at all. A change means the batch
      // lets later rows leak into earlier ones, which is a causal-mask fault
      // and not a numerics one.
      const bool tail_test = env[0] == '3';
      const auto argmax = [](std::span<const float> v) {
        std::size_t best = 0;
        for (std::size_t i = 1; i < v.size(); ++i)
          if (v[i] > v[best])
            best = i;
        return best;
      };

      if (tail_test) {
        session->Reset();
        const std::size_t ctx_len =
            session->Capacity() > 32 ? session->Capacity() - 24 : 8;
        std::vector<std::int32_t> ctx;
        for (std::size_t i = 0; i < ctx_len; ++i)
          ctx.push_back(static_cast<std::int32_t>(pool[i % pool.size()]));
        if (!session->Append(ctx, &error))
          Fail("Flint tail check prefill failed: " + error);
        const auto pos0 = session->Position();
        const std::uint32_t vocab = session->Logits().size();
        std::vector<float> ref(vocab);

        std::cout << "\n| width | same row0 | max |dlogit| | argmax flips |\n"
                     "| --: | --: | --: | --: |\n";
        for (const std::uint32_t w : order) {
          if (w < 2)
            continue;
          std::size_t same = 0;
          std::size_t flips = 0;
          double worst = 0.0;
          for (std::size_t i = 0; i < kChecks; ++i) {
            std::vector<std::int32_t> a(w);
            std::vector<std::int32_t> b(w);
            for (std::uint32_t j = 0; j < w; ++j) {
              a[j] = static_cast<std::int32_t>(
                  pool[(i * 13 + j * 7) % pool.size()]);
              b[j] = static_cast<std::int32_t>(
                  pool[(i * 29 + j * 11 + 5) % pool.size()]);
            }
            b[0] = a[0];
            if (!session->AppendAll({a.data(), w}, &error))
              Fail("Flint tail check failed: " + error);
            const auto row_a = session->Logits(0);
            std::copy(row_a.begin(), row_a.end(), ref.begin());
            session->Rewind(pos0);
            if (!session->AppendAll({b.data(), w}, &error))
              Fail("Flint tail check failed: " + error);
            const auto row_b = session->Logits(0);
            double dmax = 0.0;
            for (std::size_t t = 0; t < row_b.size(); ++t)
              dmax = std::max(
                  dmax, static_cast<double>(std::abs(row_b[t] - ref[t])));
            same += std::memcmp(row_b.data(), ref.data(),
                                vocab * sizeof(float)) == 0;
            flips += argmax(row_b) != argmax(ref);
            worst = std::max(worst, dmax);
            session->Rewind(pos0);
          }
          std::cout << "| " << w << " | " << same << "/" << kChecks << " | "
                    << worst << " | " << flips << " |\n";
        }
        return 0;
      }

      std::cout << "\n| depth | width | flips | bit-identical | max |dlogit| |"
                   " median |dlogit| | top1-top2 min | flip checks |\n"
                   "| --: | --: | --: | --: | --: | --: | --: | --: |\n";
      for (const auto depth : opt.n_depths) {
        session->Reset();
        if (depth)
          append_all(fill(depth));
        const auto pos0 = session->Position();
        const std::uint32_t vocab = session->Logits().size();
        std::vector<float> ref(vocab);

        std::vector<std::size_t> flips(widths + 1, 0);
        std::vector<std::size_t> exact(widths + 1, 0);
        std::vector<double> worst(widths + 1, 0.0);
        std::vector<std::vector<std::size_t>> flip_at(widths + 1);
        std::vector<std::vector<double>> deltas(widths + 1);
        double tightest = 1e9;  // smallest top1-top2 gap seen in the reference

        for (std::size_t i = 0; i < kChecks; ++i) {
          std::vector<std::int32_t> ids(widths);
          for (std::uint32_t j = 0; j < widths; ++j)
            ids[j] = static_cast<std::int32_t>(
                pool[(i * 13 + j * 7 + depth) % pool.size()]);

          const std::int32_t first = ids[0];
          if (!session->Append({&first, 1}, &error))
            Fail("Flint row check failed: " + error);
          const auto ref_logits = session->Logits();
          std::copy(ref_logits.begin(), ref_logits.end(), ref.begin());
          session->Rewind(pos0);

          // Top-two gap on the reference row: a flip inside this gap is a
          // near tie, not a bug.
          std::size_t top1 = 0, top2 = 1;
          for (std::size_t t = 1; t < ref.size(); ++t) {
            if (ref[t] > ref[top1]) {
              top2 = top1;
              top1 = t;
            } else if (t != top1 && ref[t] > ref[top2]) {
              top2 = t;
            }
          }
          tightest = std::min(tightest,
                              static_cast<double>(ref[top1] - ref[top2]));

          for (const std::uint32_t w : order) {
            const std::span<const std::int32_t> rows{ids.data(), w};
            if (!session->AppendAll(rows, &error))
              Fail("Flint row check failed: " + error);
            const auto row = session->Logits(0);
            double dmax = 0.0;
            for (std::size_t t = 0; t < row.size(); ++t)
              dmax = std::max(dmax, static_cast<double>(
                                        std::abs(row[t] - ref[t])));
            if (argmax(row) != argmax(ref)) {
              ++flips[w];
              if (flip_at[w].size() < 8)
                flip_at[w].push_back(i);
            }
            deltas[w].push_back(dmax);
            if (std::memcmp(row.data(), ref.data(), vocab * sizeof(float)) == 0)
              ++exact[w];
            worst[w] = std::max(worst[w], dmax);
            session->Rewind(pos0);
          }
        }
        for (std::uint32_t w = 1; w <= widths; ++w) {
          auto d = deltas[w];
          std::sort(d.begin(), d.end());
          const double med = d.empty() ? 0.0 : d[d.size() / 2];
          std::cout << "| " << depth << " | " << w << " | " << flips[w] << "/"
                    << kChecks << " | " << exact[w] << "/" << kChecks << " | "
                    << worst[w] << " | " << med << " | " << tightest << " | ";
          for (const auto i : flip_at[w])
            std::cout << i << " ";
          std::cout << "|\n";
        }
      }
    }

    // Layer-residual capture check (FLINT_CHECK_HIDDEN=1): the DFlash2 drafter
    // reads the residual of five layers, so each captured row must equal what a
    // single-token decode produced at that position, at every verify width.
    // Anything else silently changes the drafter's context. Also measures the
    // capture's cost on the eager single-token path.
    if (const char* env = std::getenv("FLINT_CHECK_HIDDEN");
        env != nullptr && env[0] != '\0' && env[0] != '0') {
      const std::vector<std::uint32_t> taps{5, 15, 26, 36, 46};
      const Config& config = loaded.model->config();
      for (const auto tap : taps) {
        if (tap >= config.num_layers)
          Fail("Flint hidden check: tap layer out of range");
      }
      const std::size_t hidden = config.hidden_size;
      const auto pool = loaded.tokenizer->Encode(
          "Layer residuals are the drafter's input: every captured row must "
          "equal the single-token decode's residual at the same position, or a "
          "draft block starts from the wrong context and acceptance drops. ");
      if (pool.empty())
        Fail("Flint hidden check text produced no tokens");

      // The bench session is sized to the requested workload; the check needs
      // its own context regardless of what was asked for on the command line.
      auto probe =
          loaded.model->CreateSession(PickContext(*loaded.model, 8192), &error);
      if (probe == nullptr)
        Fail("Flint hidden check session failed: " + error);
      const auto prefill = [&](std::size_t n) {
        const auto chunk = EngineTokens(fill(n));
        if (!probe->Append(chunk, &error))
          Fail("Flint hidden check prefill failed: " + error);
      };
      if (!probe->SetHiddenCapture(taps, &error))
        Fail("Flint hidden check: capture setup failed: " + error);
      std::cout << "\n| capture | taps | ring rows | ring rows expected |\n"
                   "| :-- | --: | --: | --: |\n"
                << "| on | " << probe->HiddenCaptureLayers().size() << " | "
                << probe->HiddenRingRows() << " | "
                << std::min(FlintSession::kHiddenRingRows, probe->Capacity())
                << " |\n";

      std::cout << "\n| width | eager | capture | replay | max |dhidden| "
                   "| tap spread |\n| --: | --: | --: | --: | --: | --: |\n";
      const std::uint32_t widths = FlintSession::kMaxVerifyRows;
      for (std::uint32_t w = 1; w <= widths; ++w) {
        probe->Reset();
        prefill(64);
        const auto pos0 = probe->Position();
        std::vector<std::int32_t> ids(w);
        for (std::uint32_t j = 0; j < w; ++j)
          ids[j] =
              static_cast<std::int32_t>(pool[(j * 7 + w) % pool.size()]);

        // Reference: one Append per row, captured where the row was produced.
        std::vector<std::vector<std::vector<float>>> ref(w);
        for (std::uint32_t j = 0; j < w; ++j) {
          if (!probe->Append({&ids[j], 1}, &error))
            Fail("Flint hidden check failed: " + error);
          ref[j].resize(taps.size());
          for (std::size_t t = 0; t < taps.size(); ++t) {
            const auto row = probe->CapturedHidden(pos0 + j, t);
            if (row.size() != hidden)
              Fail("Flint hidden check: reference row missing");
            ref[j][t].assign(row.begin(), row.end());
          }
        }
        probe->Rewind(pos0);

        // Pass 0 runs eagerly, pass 1 captures the width's step graph and
        // replays it, pass 2 is a pure replay: the capture copies live inside
        // the step, so all three must agree with the reference.
        std::vector<std::size_t> exact(3, 0);
        std::size_t total = 0;
        double worst = 0.0;
        double spread = 0.0;
        for (int pass = 0; pass < 3; ++pass) {
          probe->Rewind(pos0);
          if (!probe->AppendAll(ids, &error))
            Fail("Flint hidden check failed: " + error);
          for (std::uint32_t j = 0; j < w; ++j) {
            for (std::size_t t = 0; t < taps.size(); ++t) {
              const auto row = probe->CapturedHidden(pos0 + j, t);
              if (row.size() != hidden)
                Fail("Flint hidden check: captured row missing");
              exact[pass] += std::memcmp(row.data(), ref[j][t].data(),
                                         hidden * sizeof(float)) == 0;
              if (pass == 0)
                ++total;
              for (std::size_t i = 0; i < hidden; ++i)
                worst = std::max(worst, static_cast<double>(std::abs(
                                            row[i] - ref[j][t][i])));
            }
            // Distinct layers, distinct residuals: a capture that copied one
            // buffer into every tap would pass the comparison above. The host
            // accessor reuses one scratch row, so copy before the second read.
            const auto first = probe->CapturedHidden(pos0 + j, 0);
            std::vector<float> first_row(first.begin(), first.end());
            const auto last = probe->CapturedHidden(pos0 + j, taps.size() - 1);
            for (std::size_t i = 0; i < first_row.size(); ++i)
              spread = std::max(
                  spread,
                  static_cast<double>(std::abs(first_row[i] - last[i])));
          }
        }
        for (int pass = 0; pass < 3; ++pass) {
          if (exact[pass] != total)
            Fail("Flint hidden check: width " + std::to_string(w) +
                 " capture differs from single-token decode");
        }
        // Ring bounds: everything before Position() and inside the ring.
        if (!probe->HasHidden(pos0) ||
            probe->HasHidden(probe->Position()) ||
            !probe->HasHidden(probe->Position() - 1)) {
          Fail("Flint hidden check: ring bounds are wrong");
        }
        probe->Rewind(pos0);
        std::cout << "| " << w << " | " << exact[0] << "/" << total << " | "
                  << exact[1] << "/" << total << " | " << exact[2] << "/"
                  << total << " | " << worst << " | " << spread << " |\n";
      }

      // Ring wraparound: slots are position % ring_rows, so a context longer
      // than the ring is the only path that exercises the wrap. Prefill past
      // it, then repeat the width-8 comparison and the oldest-row bounds.
      {
        probe->Reset();
        prefill(FlintSession::kHiddenRingRows + 52);
        const auto pos0 = probe->Position();
        if (pos0 <= FlintSession::kHiddenRingRows)
          Fail("Flint hidden check: prefill did not pass the ring");
        std::vector<std::int32_t> ids(FlintSession::kMaxVerifyRows);
        for (std::uint32_t j = 0; j < ids.size(); ++j)
          ids[j] = static_cast<std::int32_t>(pool[(j * 11) % pool.size()]);

        std::vector<std::vector<std::vector<float>>> ref(ids.size());
        for (std::uint32_t j = 0; j < ids.size(); ++j) {
          if (!probe->Append({&ids[j], 1}, &error))
            Fail("Flint hidden check failed: " + error);
          ref[j].resize(taps.size());
          for (std::size_t t = 0; t < taps.size(); ++t) {
            const auto row = probe->CapturedHidden(pos0 + j, t);
            if (row.size() != hidden)
              Fail("Flint hidden check: wrapped reference row missing");
            ref[j][t].assign(row.begin(), row.end());
          }
        }
        std::size_t exact = 0;
        std::size_t total = 0;
        for (int pass = 0; pass < 3; ++pass) {
          probe->Rewind(pos0);
          if (!probe->AppendAll(ids, &error))
            Fail("Flint hidden check failed: " + error);
          for (std::uint32_t j = 0; j < ids.size(); ++j) {
            for (std::size_t t = 0; t < taps.size(); ++t) {
              const auto row = probe->CapturedHidden(pos0 + j, t);
              if (row.size() != hidden)
                Fail("Flint hidden check: wrapped row missing");
              exact += std::memcmp(row.data(), ref[j][t].data(),
                                   hidden * sizeof(float)) == 0;
              if (pass == 0)
                ++total;
            }
          }
        }
        if (exact != 3 * total)
          Fail("Flint hidden check: wrapped capture differs from "
               "single-token decode");
        const auto end = probe->Position();
        const auto ring = probe->HiddenRingRows();
        if (probe->HasHidden(end - ring) ||
            !probe->HasHidden(end - ring + 1)) {
          Fail("Flint hidden check: wrapped ring bounds are wrong");
        }
        probe->Rewind(pos0);
        std::cout << "\n| wrap | exact rows over 3 passes | ring rows |\n"
                     "| :-- | --: | --: |\n"
                  << "| position " << pos0 << " | " << exact << "/"
                  << 3 * total << " | " << ring << " |\n";
      }

      // Cost on the eager single-token path: the capture never touches the
      // model's arithmetic, so the only cost is the D2D copies.
      const auto step_ms = [&](bool capture) {
        const std::span<const std::uint32_t> layers =
            capture ? std::span<const std::uint32_t>(taps)
                    : std::span<const std::uint32_t>();
        if (!probe->SetHiddenCapture(layers, &error))
          Fail("Flint hidden check: capture setup failed: " + error);
        probe->Reset();
        prefill(256);
        constexpr std::size_t kSteps = 64;
        const auto start = Clock::now();
        for (std::size_t i = 0; i < kSteps; ++i) {
          const std::int32_t id =
              static_cast<std::int32_t>(pool[i % pool.size()]);
          if (!probe->Append({&id, 1}, &error))
            Fail("Flint hidden check failed: " + error);
        }
        return Seconds(start) * 1000.0 / kSteps;
      };
      const double off = step_ms(false);
      const double on = step_ms(true);
      std::cout << "\n| capture | ms/token | vs off |\n| :-- | --: | --: |\n"
                << "| off | " << off << " | 1.00 |\n"
                << "| on | " << on << " | " << (on / off) << " |\n";
    }

    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Flint: " << e.what() << '\n';
    return 1;
  }
}

}  // namespace gufo::models::flint
