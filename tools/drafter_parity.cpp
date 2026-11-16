// DFlash2 drafter parity driver (M1b).
//
// Produces the trace directory the upstream reference harness consumes
// (tools/qwen27b/dflash_reference.py in the Gufo tree): one `<stage>.f32` per
// stage plus `trace.json` and the `config.json` the harness reconstructs the
// draft model from, so a single comparator covers both implementations.
//
// The extension's block forward traces the oracle's stage names already, so
// this tool is the glue: run a fixed prompt through the target, keep the five
// tapped residuals, feed them to the drafter as context, draft one block with
// tracing on, and write whatever the harness needs to rebuild the same block
// in PyTorch.
//
// Which artifact can be held to the harness's 1e-4:
//   * **BF16 (or F32)**: every drafter-owned stage agrees to ~5e-6, the
//     candidate sets match row for row and the proposal distributions agree to
//     ~7e-6 total variation. This is the gate, and it passes.
//   * **Q6_K**: the same block run through the artifact's quantized route sits
//     at 1-2e-2, because a quantized draft GEMM rounds both its weights (to the
//     block format) and, on the decode path, its activation (to int8). That is
//     the artifact's arithmetic, not the block's, so the harness's fp32 gate
//     reports the quantisation delta instead of a parity failure; compare Q6_K
//     against BF16 to read it.
//
// `logits` is the *target's* LM head, which Flint stores as Q8_0. The shipped
// route reads it the way every decode does (quantized activation, int8 dot),
// which puts that one stage at the head's own quantisation floor (~3e-3) and,
// through the top-k boundary, moves a candidate or two. `--head reference`
// reads the head in F32 instead, which is what a stage-by-stage comparison
// against an fp32 reference needs; the shipped route is the one that matches
// what the verifier will do.
//
// Usage:
//   flint_drafter_parity TARGET.gguf DRAFT.gguf --out DIR [options]
//     --position N          context rows fed to the drafter (default 64)
//     --draft N             proposals in the block (default block_size - 1)
//     --temperature T       selector temperature (default 0.8). A greedy run
//                           (0) writes one-hot probabilities, which the
//                           harness's sampling checks cannot verify.
//     --text TEXT           prompt text (default: a fixed paragraph)
//     --text-file PATH      prompt text from a file; a repeated prompt makes
//                           acceptance meaningless, so measure on real content
//     --target-layer TAP    repeatable; overrides the artifact's taps
//     --head native|reference  see above (default native)

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <gufo/ext/gguf_reader.hpp>

#include "hip/vendor/qwen/tokenizer.hpp"
#include "src/gpu/dflash/dflash_forward.hpp"
#include "src/gpu/dflash/dflash_weights.hpp"
#include "src/gpu/engine.hpp"

namespace flint = gufo::models::flint::gpu;
namespace model = gufo::models::flint;
namespace dflash = gufo::models::flint::gpu::dflash;
namespace core = gufo::core;

namespace {

constexpr const char* kDefaultText =
    "Speculative decoding pays for itself when the verifier accepts more tokens "
    "than the block costs to draft. A block diffusion drafter proposes a whole "
    "block at once and the target checks it in one pass, so acceptance is the "
    "only thing that decides whether the trade works. Everything before that "
    "measurement is bookkeeping.";

[[noreturn]] void Fail(const std::string& message) {
  throw std::runtime_error(message);
}

// ---------------------------------------------------------------------------
// Arguments.
// ---------------------------------------------------------------------------
struct Options {
  std::string target;
  std::string draft;
  std::string out;
  std::string text{kDefaultText};
  std::string head{"native"};
  std::uint32_t position{64};
  std::uint32_t draft_count{0};  // 0 means block_size - 1
  float temperature{0.8F};
  std::vector<std::uint32_t> taps;
  /// Non-zero switches from "dump one block" to "measure acceptance".
  std::uint32_t accept_tokens{0};
};

[[nodiscard]] std::uint32_t ParseUint(const char* value, const char* flag) {
  std::size_t consumed = 0;
  const std::string text(value);
  std::uint32_t parsed = 0;
  try {
    parsed = static_cast<std::uint32_t>(std::stoul(text, &consumed));
  } catch (const std::exception&) {
    Fail(std::string(flag) + " needs a non-negative integer, got '" + text +
         "'");
  }
  if (consumed != text.size())
    Fail(std::string(flag) + " needs a non-negative integer, got '" + text +
         "'");
  return parsed;
}

[[nodiscard]] Options ParseArgs(int argc, char** argv) {
  Options options;
  std::vector<std::string> positional;
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg = argv[i];
    const auto next = [&](const char* flag) -> const char* {
      if (i + 1 >= argc)
        Fail(std::string(flag) + " needs a value");
      return argv[++i];
    };
    if (arg == "--out") {
      options.out = next("--out");
    } else if (arg == "--text") {
      options.text = next("--text");
    } else if (arg == "--text-file") {
      std::ifstream file(next("--text-file"), std::ios::binary);
      if (!file)
        Fail(std::string("cannot read ") + argv[i]);
      std::ostringstream buffer;
      buffer << file.rdbuf();
      options.text = buffer.str();
    } else if (arg == "--head") {
      options.head = next("--head");
    } else if (arg == "--position") {
      options.position = ParseUint(next("--position"), "--position");
    } else if (arg == "--draft") {
      options.draft_count = ParseUint(next("--draft"), "--draft");
    } else if (arg == "--temperature") {
      options.temperature =
          std::stof(std::string(next("--temperature")));
    } else if (arg == "--accept") {
      options.accept_tokens = ParseUint(next("--accept"), "--accept");
    } else if (arg == "--target-layer") {
      options.taps.push_back(
          ParseUint(next("--target-layer"), "--target-layer"));
    } else if (arg.size() > 1 && arg.front() == '-') {
      Fail("unknown option '" + std::string(arg) +
           "'; usage: flint_drafter_parity TARGET.gguf DRAFT.gguf --out DIR");
    } else {
      positional.emplace_back(arg);
    }
  }
  if (positional.size() != 2)
    Fail("usage: flint_drafter_parity TARGET.gguf DRAFT.gguf --out DIR");
  options.target = positional[0];
  options.draft = positional[1];
  if (options.out.empty() && options.accept_tokens == 0)
    Fail("--out is required unless --accept is given");
  if (options.head != "native" && options.head != "reference")
    Fail("--head must be 'native' or 'reference'");
  if (options.position == 0)
    Fail("--position must be at least one row");
  if (options.temperature < 0.0F)
    Fail("--temperature must not be negative");
  if (options.accept_tokens != 0 && options.temperature != 0.0F)
    Fail("--accept measures greedy acceptance; use --temperature 0");
  return options;
}

// ---------------------------------------------------------------------------
// Trace directory.
// ---------------------------------------------------------------------------
/// Writes one stage per file as raw little-endian float32, which is what the
/// harness reads back with `np.fromfile(dtype="<f4")`.
class TraceDir {
 public:
  explicit TraceDir(std::string path) : path_(std::move(path)) {
    std::error_code error;
    std::filesystem::create_directories(path_, error);
    if (error)
      Fail("cannot create trace directory " + path_ + ": " + error.message());
  }

  [[nodiscard]] const std::string& path() const noexcept { return path_; }

  void Write(const std::string_view name, const float* data,
             std::size_t count) const {
    const std::string file = path_ + "/" + std::string(name) + ".f32";
    std::ofstream out(file, std::ios::binary | std::ios::trunc);
    if (!out)
      Fail("cannot open " + file);
    out.write(reinterpret_cast<const char*>(data),
              static_cast<std::streamsize>(count * sizeof(float)));
    if (!out)
      Fail("cannot write " + file);
    ++stages_;
    bytes_ += count * sizeof(float);
  }

  void WriteJson(const std::string& name, const std::string& json) const {
    const std::string file = path_ + "/" + name;
    std::ofstream out(file, std::ios::trunc);
    if (!out)
      Fail("cannot open " + file);
    out << json;
  }

  [[nodiscard]] std::size_t stages() const noexcept { return stages_; }
  [[nodiscard]] std::size_t bytes() const noexcept { return bytes_; }

 private:
  std::string path_;
  mutable std::size_t stages_{0};
  mutable std::size_t bytes_{0};
};

// ---------------------------------------------------------------------------
// Target model.
// ---------------------------------------------------------------------------
struct Target {
  std::shared_ptr<flint::Model> model;
  std::unique_ptr<gufo::vendor::qwen::tokenization::QwenTokenizer> tokenizer;
};

[[nodiscard]] Target LoadTarget(const std::string& path) {
  std::string error;
  auto model = flint::Model::Load(path, {}, &error);
  if (model == nullptr)
    Fail("target model: " + error);
  auto metadata = core::GgufReader::OpenFile(path, &error);
  if (!metadata)
    Fail("target GGUF: " + error);
  auto tokenizer =
      gufo::vendor::qwen::tokenization::QwenTokenizer::CreateFromGguf(*metadata,
                                                                     &error);
  if (tokenizer == nullptr)
    Fail("target tokenizer: " + error);
  return {std::move(model), std::move(tokenizer)};
}

/// The prompt as engine token ids, exactly `count` of them.
[[nodiscard]] std::vector<std::int32_t> PromptTokens(
    const gufo::vendor::qwen::tokenization::QwenTokenizer& tokenizer,
    const std::string& text, std::uint32_t count) {
  const auto pool = tokenizer.Encode(text);
  if (pool.empty())
    Fail("prompt produced no tokens");
  if (pool.size() < count)
    std::cerr << "warning: prompt text is only " << pool.size()
              << " tokens; the " << count
              << " asked for repeat it, and a repeated prompt inflates\n"
                 "         acceptance because the continuation is predictable\n";
  std::vector<std::int32_t> ids(count);
  for (std::uint32_t i = 0; i < count; ++i)
    ids[i] = static_cast<std::int32_t>(pool[i % pool.size()]);
  return ids;
}

/// `count * taps * hidden` floats, row-major by position and then by tap in the
/// order `taps` is given -- the layout `fc.weight` expects.
[[nodiscard]] std::vector<float> CaptureFeatures(
    flint::Session& session, const std::vector<std::uint32_t>& taps,
    std::uint32_t count, std::uint32_t hidden) {
  std::vector<float> features(std::size_t{count} * taps.size() * hidden);
  for (std::uint32_t row = 0; row < count; ++row) {
    for (std::size_t tap = 0; tap < taps.size(); ++tap) {
      if (!session.HasHidden(row))
        Fail("captured residual for row " + std::to_string(row) +
             " left the ring; lower --position");
      const auto captured = session.CapturedHidden(row, tap);
      if (captured.size() != hidden)
        Fail("captured residual has the wrong width");
      std::memcpy(features.data() + (std::size_t{row} * taps.size() + tap) *
                                        hidden,
                  captured.data(), hidden * sizeof(float));
    }
  }
  return features;
}

// ---------------------------------------------------------------------------
// JSON.
// ---------------------------------------------------------------------------
[[nodiscard]] std::string Join(const std::vector<std::uint32_t>& values) {
  std::ostringstream out;
  for (std::size_t i = 0; i < values.size(); ++i)
    out << (i == 0 ? "" : ", ") << values[i];
  return out.str();
}

[[nodiscard]] std::string Float(float value) {
  std::ostringstream out;
  out.precision(9);
  out << value;
  return out.str();
}

/// The `Qwen3Config` the harness rebuilds the draft model from. `dflash_config`
/// carries the artifact's diffusion topology verbatim, so a geometry change in
/// the artifact shows up as a config the harness rejects rather than a silent
/// mismatch.
[[nodiscard]] std::string ConfigJson(const dflash::DflashConfig& config,
                                     std::uint32_t target_layers) {
  std::ostringstream out;
  out << "{\n"
      << "  \"architectures\": [\"Qwen3ForCausalLM\"],\n"
      << "  \"hidden_size\": " << config.hidden_size << ",\n"
      << "  \"intermediate_size\": " << config.intermediate_size << ",\n"
      << "  \"num_attention_heads\": " << config.num_attention_heads << ",\n"
      << "  \"num_key_value_heads\": " << config.num_key_value_heads << ",\n"
      << "  \"head_dim\": " << config.head_dim << ",\n"
      << "  \"num_hidden_layers\": " << config.num_layers << ",\n"
      // Read even though `target_layer_ids` is always present: the upstream
      // constructor passes a `build_target_layer_ids` fallback as a default
      // argument, and Python evaluates it before the lookup.
      << "  \"num_target_layers\": " << target_layers << ",\n"
      << "  \"vocab_size\": " << config.vocab_size << ",\n"
      << "  \"max_position_embeddings\": " << config.context_length << ",\n"
      << "  \"rms_norm_eps\": " << Float(config.rms_norm_eps) << ",\n"
      << "  \"rope_theta\": " << Float(config.rope_theta) << ",\n"
      << "  \"attention_bias\": false,\n"
      << "  \"attention_dropout\": 0.0,\n"
      << "  \"is_causal\": false,\n"
      // Every draft block is non-causal over its own block and unmasked over
      // the injected context, so no layer may be a sliding-attention layer:
      // the harness derives the SDPA mask from this list.
      << "  \"layer_types\": [";
  for (std::uint32_t i = 0; i < config.num_layers; ++i)
    out << (i == 0 ? "" : ", ") << "\"full_attention\"";
  out << "],\n"
      << "  \"dflash_config\": {\n"
      << "    \"target_layer_ids\": [" << Join(config.target_layer_ids) << "],\n"
      << "    \"num_target_layers\": " << target_layers << ",\n"
      << "    \"block_size\": " << config.block_size << ",\n"
      << "    \"mask_token_id\": " << config.mask_token_id << ",\n"
      << "    \"conv_kernel_size\": " << config.conv_kernel_size << ",\n"
      << "    \"conv_group_size\": " << config.conv_group_size << ",\n"
      << "    \"selector_rank\": " << config.selector_rank << ",\n"
      << "    \"selector_top_k\": " << config.selector_top_k << ",\n"
      << "    \"sliding_window\": " << config.sliding_window << "\n"
      << "  }\n"
      << "}\n";
  return out.str();
}

/// The per-run facts the harness cannot derive from the config: the block it
/// must rebuild, the draw that produced each proposal, and the head's storage
/// type so it can refuse a repacked head.
[[nodiscard]] std::string TraceJson(const Options& options,
                                    const dflash::DflashConfig& config,
                                    const std::vector<std::uint32_t>& pools,
                                    std::uint32_t anchor,
                                    std::uint32_t draft_count,
                                    const dflash::DflashDraft& draft,
                                    core::GgmlType head_type) {
  std::ostringstream out;
  out << "{\n"
      << "  \"target_layer_ids\": [" << Join(config.target_layer_ids) << "],\n"
      << "  \"anchor\": " << anchor << ",\n"
      << "  \"position\": " << options.position << ",\n"
      << "  \"draft_count\": " << draft_count << ",\n"
      << "  \"temperature\": " << Float(options.temperature) << ",\n"
      << "  \"block_size\": " << config.block_size << ",\n"
      << "  \"mask_token_id\": " << config.mask_token_id << ",\n"
      << "  \"sliding_window\": " << config.sliding_window << ",\n"
      << "  \"head_type\": " << static_cast<int>(head_type) << ",\n"
      // The extension keeps activations in FP32 on every small-batch route, so
      // the harness must not round Linear inputs to BF16.
      << "  \"block_activation_dtype\": \"f32\",\n"
      << "  \"prompt_tokens\": [";
  for (std::size_t i = 0; i < pools.size(); ++i)
    out << (i == 0 ? "" : ", ") << pools[i];
  out << "],\n"
      << "  \"candidates\": [";
  for (std::size_t i = 0; i < draft.candidate_ids.size(); ++i)
    out << (i == 0 ? "" : ", ") << draft.candidate_ids[i];
  out << "],\n"
      << "  \"proposed\": [";
  for (std::size_t i = 0; i < draft.tokens.size(); ++i)
    out << (i == 0 ? "" : ", ") << draft.tokens[i];
  out << "]\n"
      << "}\n";
  return out.str();
}

// ---------------------------------------------------------------------------
// The parity run.
// ---------------------------------------------------------------------------
/// Uniform per proposal that lands inside the chosen candidate's interval of
/// the dumped distribution. The harness re-derives the choice from the
/// probabilities and the uniform, so the draw has to agree with the proposal
/// and not merely be some number in [0, 1).
[[nodiscard]] std::vector<float> DrawsFromProbabilities(
    const dflash::DflashDraft& draft, std::uint32_t top_k) {
  const std::size_t rows = draft.tokens.size();
  if (draft.candidate_probabilities.size() != rows * top_k)
    Fail("draft is missing candidate probabilities");
  std::vector<float> draws(rows, 0.5F);
  for (std::size_t row = 0; row < rows; ++row) {
    const float* probabilities =
        draft.candidate_probabilities.data() + row * top_k;
    std::size_t chosen = top_k;
    for (std::uint32_t index = 0; index < top_k; ++index) {
      if (draft.candidate_ids[row * top_k + index] == draft.tokens[row]) {
        chosen = index;
        break;
      }
    }
    if (chosen == top_k)
      Fail("proposal " + std::to_string(row) + " is not one of its candidates");
    double below = 0.0;
    for (std::size_t index = 0; index < chosen; ++index)
      below += probabilities[index];
    const double above = below + probabilities[chosen];
    draws[row] = static_cast<float>(0.5 * (below + above));
    // The interval can be empty only if the probability is denormal-small; a
    // draw at its lower edge still lands on the same candidate.
    if (!(draws[row] >= 0.0F) || !(draws[row] < 1.0F))
      draws[row] = static_cast<float>(below);
  }
  return draws;
}

/// One row of the acceptance table.
struct AcceptanceRow {
  std::size_t drafted{0};
  std::size_t accepted{0};
};

/// Greedy speculative acceptance: draft a block, verify it in one target pass,
/// keep the leading proposals the target's own argmax agrees with, and continue
/// from the first disagreement. This is the same commit rule the runner will
/// use, so the number it reports is the ratio the cycle would get -- measured
/// without the runner.
[[nodiscard]] int MeasureAcceptance(
    flint::Session& session, dflash::DflashForward& forward,
    const dflash::DflashConfig& config, const std::vector<std::uint32_t>& taps,
    std::uint32_t hidden, std::uint32_t draft_count, std::uint32_t prefill,
    std::uint32_t budget, const std::string& head_route) {
  std::string error;
  const std::size_t feature_row = std::size_t{taps.size()} * hidden;
  std::vector<float> features;
  std::uint32_t injected = 0;

  // The drafter reads the target's residuals up to the position it drafts at.
  const auto append_context_to = [&](std::uint32_t position) {
    if (position == injected)
      return;
    features.resize(std::size_t{position - injected} * feature_row);
    for (std::uint32_t r = injected; r < position; ++r) {
      for (std::size_t t = 0; t < taps.size(); ++t) {
        const auto captured = session.CapturedHidden(r, t);
        if (captured.size() != hidden)
          Fail("residual for row " + std::to_string(r) +
               " is not in the ring; lower --position or --accept");
        std::memcpy(features.data() +
                        (std::size_t{r - injected} * taps.size() + t) * hidden,
                    captured.data(), hidden * sizeof(float));
      }
    }
    if (!forward.AppendContext(features, injected, position - injected, &error))
      Fail("context injection: " + error);
    injected = position;
  };

  const auto argmax = [](std::span<const float> logits) {
    if (logits.empty())
      Fail("target produced no logits");
    return static_cast<std::uint32_t>(
        std::distance(logits.begin(),
                      std::max_element(logits.begin(), logits.end())));
  };

  std::uint32_t position = prefill;
  std::uint32_t anchor = argmax(session.Logits());
  std::vector<AcceptanceRow> rows(draft_count);
  std::size_t cycles = 0;
  std::size_t emitted = 0;
  std::size_t accepted_total = 0;

  while (emitted < budget) {
    if (position + draft_count + 1 > session.Capacity())
      break;
    append_context_to(position);
    dflash::DflashDraft draft;
    if (!forward.Run(anchor, position, draft_count, 0.0F, {}, {}, &draft,
                     &error))
      Fail("draft block: " + error);
    std::vector<std::int32_t> batch(draft_count + 1);
    batch[0] = static_cast<std::int32_t>(anchor);
    for (std::uint32_t i = 0; i < draft_count; ++i)
      batch[i + 1] = static_cast<std::int32_t>(draft.tokens[i]);
    if (!session.AppendAll(batch, &error))
      Fail("verify pass: " + error);

    std::uint32_t accepted = 0;
    while (accepted < draft_count &&
           argmax(session.Logits(accepted)) == draft.tokens[accepted]) {
      ++accepted;
    }
    for (std::uint32_t i = 0; i < draft_count; ++i) {
      ++rows[i].drafted;
      if (i < accepted)
        ++rows[i].accepted;
    }
    // Row `accepted` is the token that replaces the first disagreement (or the
    // free token after a fully accepted block); it becomes the next anchor.
    const std::uint32_t bonus = argmax(session.Logits(accepted));
    accepted_total += accepted;
    ++cycles;
    emitted += accepted + 1;
    position += accepted + 1;
    anchor = bonus;
    session.Rewind(position);
  }

  if (cycles == 0)
    Fail("no cycle ran; the context is too small for one block");

  std::cout << "| quantity | value |\n| :-- | --: |\n"
            << "| prefill rows | " << prefill << " |\n"
            << "| proposals per block | " << draft_count << " |\n"
            << "| cycles | " << cycles << " |\n"
            << "| tokens emitted | " << emitted << " |\n"
            << "| accepted drafts | " << accepted_total << " |\n"
            << "| acceptance | "
            << 100.0 * double{accepted_total} / (double{cycles} * draft_count)
            << "% |\n"
            << "| tokens per cycle | " << (double{emitted} / double{cycles})
            << " |\n"
            << "| head route | " << head_route << " |\n\n"
            << "| proposal | drafted | accepted | acceptance |\n"
            << "| --: | --: | --: | --: |\n";
  for (std::uint32_t i = 0; i < draft_count; ++i) {
    const auto& row = rows[i];
    std::cout << "| " << i << " | " << row.drafted << " | " << row.accepted
              << " | "
              << (row.drafted == 0 ? 0.0
                                   : 100.0 * double{row.accepted} /
                                         double{row.drafted})
              << "% |\n";
  }
  std::cout << "\nacceptance is `P(the target's greedy token equals the draft)`"
               " -- distribution overlap, not the min(1, p/q) rate.\n";
  return 0;
}

[[nodiscard]] int Run(const Options& options) {
  const Target target = LoadTarget(options.target);
  const model::Config& config = target.model->config();

  std::string error;
  auto metadata = core::GgufReader::OpenFile(options.draft, &error);
  if (!metadata)
    Fail("drafter GGUF: " + error);
  auto weights = dflash::DflashWeights::LoadFromGguf(*metadata, &error);
  if (!weights.has_value())
    Fail("drafter loader: " + error);
  const dflash::DflashConfig& draft_config = weights->config;

  if (!options.taps.empty() && options.taps != draft_config.target_layer_ids)
    Fail("--target-layer disagrees with the artifact's taps");

  std::cout << "# DFlash2 drafter parity\n\n| quantity | value |\n| :-- | --: |\n"
            << "| target | " << options.target << " |\n"
            << "| drafter | " << options.draft << " |\n"
            << "| context rows | " << options.position << " |\n"
            << "| block / proposals | " << draft_config.block_size << " / "
            << (options.draft_count == 0 ? draft_config.block_size - 1U
                                         : options.draft_count)
            << " |\n"
            << "| temperature | " << options.temperature << " |\n"
            << "| head route | " << options.head << " |\n\n";

  // A session that only ever prefills the prompt: the drafter needs residuals
  // for `position` rows, so the ring must hold at least that many.
  const auto taps = options.taps.empty() ? draft_config.target_layer_ids
                                         : options.taps;
  const std::uint32_t context = std::max<std::uint32_t>(
      options.position + options.accept_tokens + draft_config.block_size + 16U,
      flint::Session::kHiddenRingRows);
  auto session = target.model->CreateSession(context, &error);
  if (session == nullptr)
    Fail("target session: " + error);
  if (!session->SetHiddenCapture(taps, &error))
    Fail("hidden capture: " + error);

  const auto prompt = PromptTokens(*target.tokenizer, options.text,
                                   options.position);
  if (!session->Append(prompt, &error))
    Fail("prompt prefill: " + error);
  if (session->Position() != options.position)
    Fail("prompt prefill stopped short of the requested position");

  const auto features =
      CaptureFeatures(*session, taps, options.position, config.hidden_size);

  // Anchor: what the target itself would emit next. The block is then a
  // proposal *from* that token, which is the shape of a real cycle.
  std::uint32_t anchor = 0;
  {
    const auto logits = session->Logits();
    if (logits.empty())
      Fail("target produced no logits for the anchor");
    anchor = static_cast<std::uint32_t>(
        std::distance(logits.begin(),
                      std::max_element(logits.begin(), logits.end())));
  }

  const flint::TiedWeights tied = target.model->Tied();
  if (tied.token_embedding == nullptr || tied.output == nullptr)
    Fail("target exposes no tied weights");
  const dflash::DflashBorrowed borrowed{
      .token_embedding = tied.token_embedding,
      .token_embedding_type = tied.token_embedding_type,
      .output = tied.output,
      .output_type = tied.output_type};

  const std::uint32_t draft_count =
      std::min(options.draft_count == 0 ? draft_config.block_size - 1U
                                        : options.draft_count,
               draft_config.block_size - 1U);

  auto forward = dflash::DflashForward::Create(
      *weights, borrowed,
      options.position + options.accept_tokens + draft_config.block_size,
      &error,
      options.head == "reference" ? dflash::DflashHeadRoute::kFp32
                                  : dflash::DflashHeadRoute::kNative);
  if (forward == nullptr)
    Fail("drafter forward: " + error);

  if (options.accept_tokens != 0) {
    return MeasureAcceptance(*session, *forward, draft_config, taps,
                             config.hidden_size, draft_count, options.position,
                             options.accept_tokens, options.head);
  }

  TraceDir trace(options.out);
  const auto hook = [&](std::string_view name, const float* data,
                        std::size_t count) { trace.Write(name, data, count); };

  if (!forward->AppendContext(features, 0, options.position, &error, hook))
    Fail("context injection: " + error);

  // Two passes: the first finds the block, the second redraws it with uniforms
  // that agree with the returned distribution. A block is a pure function of
  // the injected context, so the second pass returns the same proposals.
  dflash::DflashDraft draft;
  std::vector<float> uniforms(draft_count, 0.5F);
  if (!forward->Run(anchor, options.position, draft_count, options.temperature,
                    uniforms, {}, &draft, &error)) {
    Fail("draft block: " + error);
  }
  if (options.temperature > 0.0F) {
    uniforms = DrawsFromProbabilities(draft, draft_config.selector_top_k);
    draft = {};
    if (!forward->Run(anchor, options.position, draft_count,
                      options.temperature, uniforms, hook, &draft, &error)) {
      Fail("draft block: " + error);
    }
  } else {
    draft = {};
    if (!forward->Run(anchor, options.position, draft_count, 0.0F, {},
                      hook, &draft, &error)) {
      Fail("draft block: " + error);
    }
  }
  if (draft.tokens.size() != draft_count)
    Fail("draft block returned the wrong proposal count");

  trace.Write("target_features", features.data(),
              static_cast<std::size_t>(options.position) * taps.size() *
                  config.hidden_size);

  if (options.temperature > 0.0F) {
    trace.Write("probabilities", draft.candidate_probabilities.data(),
                draft.candidate_probabilities.size());
    trace.Write("uniforms", uniforms.data(), uniforms.size());
    std::vector<float> selected(draft_count, 0.0F);
    for (std::uint32_t row = 0; row < draft_count; ++row) {
      for (std::uint32_t index = 0; index < draft_config.selector_top_k;
           ++index) {
        if (draft.candidate_ids[row * draft_config.selector_top_k + index] ==
            draft.tokens[row]) {
          selected[row] =
              draft.candidate_probabilities[row * draft_config.selector_top_k +
                                            index];
        }
      }
    }
    trace.Write("selected_probabilities", selected.data(), selected.size());
  }

  std::vector<std::uint32_t> prompt_ids(prompt.size());
  for (std::size_t i = 0; i < prompt.size(); ++i)
    prompt_ids[i] = static_cast<std::uint32_t>(prompt[i]);
  trace.WriteJson(
      "trace.json",
      TraceJson(options, draft_config, prompt_ids, anchor, draft_count, draft,
                tied.output_type));
  trace.WriteJson("config.json",
                  ConfigJson(draft_config, config.num_layers));

  std::cout << "| stage | file |\n| :-- | :-- |\n"
            << "| trace stages | " << trace.stages() << " |\n"
            << "| trace bytes | " << trace.bytes() << " |\n"
            << "| directory | " << trace.path() << " |\n\n"
            << "| proposal | token | confidence |\n| --: | --: | --: |\n";
  for (std::size_t i = 0; i < draft.tokens.size(); ++i) {
    std::cout << "| " << i << " | " << draft.tokens[i] << " | "
              << (i < draft.confidences.size() ? draft.confidences[i] : 0.0F)
              << " |\n";
  }
  std::cout << "\nanchor " << anchor << " at position " << options.position
            << "; run the harness with --config " << trace.path()
            << "/config.json --trace " << trace.path() << "\n";
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    int devices = 0;
    if (hipGetDeviceCount(&devices) != hipSuccess || devices == 0)
      Fail("no HIP device");
    return Run(ParseArgs(argc, argv));
  } catch (const std::exception& error) {
    std::cerr << "flint_drafter_parity: " << error.what() << '\n';
    return 1;
  }
}
