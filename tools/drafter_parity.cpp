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
//   flint_drafter_parity TARGET.gguf DRAFT.gguf --prompts P.jsonl --dump-prompts OUT.jsonl
//     Writes the shared acceptance workload as prompt token ids (M1c). Needs
//     only the tokenizer, so it does not load the target weights.
//   flint_drafter_parity TARGET.gguf DRAFT.gguf --temperature 0
//       --prompts-ids DUMP.jsonl --report ours.jsonl [--stream ORACLE.jsonl]
//     Greedy acceptance on the shared workload (M1c C2). One line per prompt:
//     acceptance, mean_len and per_row_accepted. With --stream the continuation
//     is the oracle's ids, not the target's argmax, and stream_divergence
//     reports how often the target's argmax disagrees with it.

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <gufo/ext/gguf_reader.hpp>
#include <gufo/ext/json.hpp>

#include "hip/vendor/qwen/tokenizer.hpp"
#include "src/chat_template.hpp"
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
  /// Shared acceptance workload (JSONL), read by --dump-prompts.
  std::string prompts_path;
  /// Non-empty switches to "dump prompt token ids" (M1c B2).
  std::string dump_path;
  /// Non-empty switches to the shared-workload acceptance run (M1c C2).
  std::string prompts_ids;
  /// Per-prompt JSONL report (--prompts-ids).
  std::string report_path;
  /// Forced continuation: the oracle's generated ids (--prompts-ids only).
  std::string stream_path;
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
    } else if (arg == "--prompts") {
      options.prompts_path = next("--prompts");
    } else if (arg == "--dump-prompts") {
      options.dump_path = next("--dump-prompts");
    } else if (arg == "--prompts-ids") {
      options.prompts_ids = next("--prompts-ids");
    } else if (arg == "--report") {
      options.report_path = next("--report");
    } else if (arg == "--stream") {
      options.stream_path = next("--stream");
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
  if (options.out.empty() && options.accept_tokens == 0 &&
      options.dump_path.empty() && options.prompts_ids.empty())
    Fail("--out is required unless --accept, --prompts-ids or --dump-prompts "
         "is given");
  if (options.prompts_ids.empty() != options.report_path.empty())
    Fail("--prompts-ids and --report must be given together");
  if (options.prompts_ids.empty() && !options.stream_path.empty())
    Fail("--stream needs --prompts-ids");
  if (!options.prompts_ids.empty() && options.temperature != 0.0F)
    Fail("--prompts-ids measures greedy acceptance; use --temperature 0");
  if (!options.prompts_ids.empty() && options.accept_tokens != 0)
    Fail("--prompts-ids and --accept are different runs; give one");
  if (options.dump_path.empty() != options.prompts_path.empty())
    Fail("--prompts and --dump-prompts must be given together");
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

[[nodiscard]] std::unique_ptr<gufo::vendor::qwen::tokenization::QwenTokenizer>
LoadTokenizer(const std::string& path) {
  std::string error;
  auto metadata = core::GgufReader::OpenFile(path, &error);
  if (!metadata)
    Fail("target GGUF: " + error);
  auto tokenizer =
      gufo::vendor::qwen::tokenization::QwenTokenizer::CreateFromGguf(*metadata,
                                                                     &error);
  if (tokenizer == nullptr)
    Fail("target tokenizer: " + error);
  return tokenizer;
}

[[nodiscard]] Target LoadTarget(const std::string& path) {
  std::string error;
  auto model = flint::Model::Load(path, {}, &error);
  if (model == nullptr)
    Fail("target model: " + error);
  return {std::move(model), LoadTokenizer(path)};
}

// ---------------------------------------------------------------------------
// Prompt dump (M1c B2). The oracle consumes these ids byte for byte, so the
// chat template is applied here and never by llama.cpp.
// ---------------------------------------------------------------------------
/// The git work tree holding the prompts file: `raw` prompts name repo paths
/// and must resolve the same way whatever directory the tool runs from.
[[nodiscard]] std::filesystem::path RepoRoot(const std::string& prompts) {
  std::error_code error;
  auto dir = std::filesystem::absolute(prompts, error).parent_path();
  while (!dir.empty()) {
    if (std::filesystem::exists(dir / ".git", error))
      return dir;
    if (dir == dir.parent_path())
      break;
    dir = dir.parent_path();
  }
  Fail("no git repository above " + prompts);
}

/// `git show REV:PATH` as bytes. Pinning the revision keeps the raw prompts
/// fixed while the files change.
[[nodiscard]] std::string GitShow(const std::filesystem::path& root,
                                  const std::string& rev,
                                  const std::string& path) {
  const std::string root_text = root.string();
  if (root_text.find('\'') != std::string::npos ||
      rev.find('\'') != std::string::npos ||
      path.find('\'') != std::string::npos)
    Fail("prompt paths and revisions may not contain single quotes");
  const std::string command = "git -C '" + root_text + "' show '" + rev + ":" +
                              path + "'";
  FILE* pipe = popen(command.c_str(), "r");
  if (pipe == nullptr)
    Fail("cannot run: " + command);
  std::string content;
  std::array<char, 4096> buffer{};
  std::size_t read = 0;
  while ((read = std::fread(buffer.data(), 1, buffer.size(), pipe)) > 0)
    content.append(buffer.data(), read);
  if (pclose(pipe) != 0)
    Fail("git show failed: " + rev + ":" + path);
  return content;
}

[[nodiscard]] std::vector<std::uint32_t> EncodeChat(
    const gufo::vendor::qwen::tokenization::QwenTokenizer& tokenizer,
    const std::string& text) {
  model::ChatTemplateOptions template_options;
  template_options.enable_thinking = false;
  gufo::vendor::qwen::tokenization::TokenizerOptions encode_options;
  encode_options.parse_special_tokens = true;
  return tokenizer.Encode(
      model::RenderPrompt(std::vector<model::ChatMessage>{{"user", text, ""}},
                          template_options),
      encode_options);
}

[[nodiscard]] int DumpPrompts(const Options& options) {
  const auto tokenizer = LoadTokenizer(options.target);
  const auto root = RepoRoot(options.prompts_path);

  std::ifstream input(options.prompts_path, std::ios::binary);
  if (!input)
    Fail("cannot read " + options.prompts_path);
  std::error_code error;
  const auto dump_parent =
      std::filesystem::absolute(options.dump_path, error).parent_path();
  std::filesystem::create_directories(dump_parent, error);
  if (error)
    Fail("cannot create " + dump_parent.string() + ": " + error.message());
  std::ofstream output(options.dump_path, std::ios::binary | std::ios::trunc);
  if (!output)
    Fail("cannot open " + options.dump_path);

  std::string line;
  std::size_t line_number = 0;
  std::size_t written = 0;
  while (std::getline(input, line)) {
    ++line_number;
    if (line.find_first_not_of(" \t\r") == std::string::npos)
      continue;
    const std::string where =
        options.prompts_path + ":" + std::to_string(line_number) + ": ";
    gufo::json::Value spec;
    try {
      spec = gufo::json::parse(line);
    } catch (const std::exception& error) {
      Fail(where + error.what());
    }

    const std::string id = spec.member_str("id");
    const std::string category = spec.member_str("category");
    const std::string kind = spec.member_str("kind");
    const std::size_t max_tokens = spec.member_size("max_tokens");
    if (id.empty() || max_tokens == 0)
      Fail(where + "needs an id and a positive max_tokens");

    std::vector<std::uint32_t> ids;
    bool stop_on_eos = false;
    if (kind == "chat") {
      const std::string text = spec.member_str("text");
      if (text.empty())
        Fail(where + "chat prompt needs text");
      ids = EncodeChat(*tokenizer, text);
      stop_on_eos = true;
    } else if (kind == "raw") {
      // Raw text must not parse special tokens: it is document content, not
      // a chat turn, and the prefill length is part of the contract.
      const std::string file = spec.member_str("file");
      const std::string rev = spec.member_str("git_rev");
      const std::size_t prefill = spec.member_size("prefill");
      if (file.empty() || rev.empty() || prefill == 0)
        Fail(where + "raw prompt needs file, git_rev and prefill");
      const std::string content = GitShow(root, rev, file);
      gufo::vendor::qwen::tokenization::TokenizerOptions raw_options;
      raw_options.parse_special_tokens = false;
      ids = tokenizer->Encode(content, raw_options);
      if (ids.size() < prefill)
        Fail(where + file + " has only " + std::to_string(ids.size()) +
             " tokens at " + rev);
      ids.resize(prefill);
      stop_on_eos = false;
    } else {
      Fail(where + "unknown kind '" + kind + "'");
    }

    gufo::json::Value prompt_ids = gufo::json::Value::array();
    for (const std::uint32_t token : ids)
      prompt_ids.push_back(
          gufo::json::Value(static_cast<unsigned long long>(token)));
    gufo::json::Value record = gufo::json::Value::object();
    record["id"] = id;
    record["category"] = category;
    record["kind"] = kind;
    record["prompt_ids"] = prompt_ids;
    record["max_tokens"] = static_cast<unsigned long long>(max_tokens);
    record["stop_on_eos"] = stop_on_eos;
    output << record.dump() << '\n';
    ++written;
    std::cout << id << "\t" << kind << "\t" << ids.size() << " prompt tokens\n";
  }
  if (!output)
    Fail("cannot write " + options.dump_path);
  std::cout << "wrote " << written << " prompts to " << options.dump_path
            << '\n';
  return 0;
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

// ---------------------------------------------------------------------------
// Acceptance: M1b --accept and M1c --prompts-ids run the same cycle.
// ---------------------------------------------------------------------------
using Tokenizer = gufo::vendor::qwen::tokenization::QwenTokenizer;

/// One row of the acceptance table.
struct AcceptanceRow {
  std::size_t drafted{0};
  std::size_t accepted{0};
};

/// One prompt for the speculative cycle.
struct AcceptanceJob {
  std::vector<std::uint32_t> prompt;
  std::uint32_t draft_count{0};
  /// Tokens to generate, counting the first one the prompt's logits give.
  std::uint32_t max_tokens{1};
  /// Stop after the first stop token (EOS). The stop token is counted.
  bool stop_on_eos{false};
  /// Empty: free-running, the target's argmax is the continuation. Non-empty:
  /// forced stream, the continuation is these ids (the oracle's output), so
  /// the acceptance measures the drafter alone on a shared text.
  std::vector<std::uint32_t> stream;
};

struct AcceptanceResult {
  std::size_t prompt_len{0};
  std::size_t cycles{0};
  /// Tokens after the first one: sum of (accepted + 1) over the cycles, clipped
  /// at max_tokens and at the stop token. Unclipped, emitted / cycles is the
  /// oracle's mean length, 1 + accepted / cycles.
  std::size_t emitted{0};
  std::size_t drafted{0};
  std::size_t accepted{0};
  std::vector<AcceptanceRow> rows;
  /// Forced streams only: verify rows whose target argmax was compared with
  /// the stream token, and how many of them disagreed.
  std::size_t divergence_rows{0};
  std::size_t divergence_mismatches{0};
};

/// numerator / denominator, or 0 for an empty denominator so reports never
/// print NaN.
[[nodiscard]] double Ratio(std::size_t numerator, std::size_t denominator) {
  return denominator == 0 ? 0.0
                          : static_cast<double>(numerator) /
                                static_cast<double>(denominator);
}

[[nodiscard]] std::uint32_t Argmax(std::span<const float> logits) {
  if (logits.empty())
    Fail("target produced no logits");
  return static_cast<std::uint32_t>(std::distance(
      logits.begin(), std::max_element(logits.begin(), logits.end())));
}

[[nodiscard]] bool IsStop(const Tokenizer* tokenizer, std::uint32_t token) {
  return tokenizer != nullptr && tokenizer->IsStopToken(token);
}

/// One prompt through the speculative cycle: prefill, then draft, verify and
/// commit until the stop rule fires. Greedy acceptance: a draft is kept when
/// the target's own argmax agrees with it (free) or when it is the stream's
/// token (forced). This is the commit rule the runner will use, so the ratio
/// it reports is the one the cycle would get, without the runner.
[[nodiscard]] AcceptanceResult RunAcceptance(
    flint::Session& session, dflash::DflashForward& forward,
    const std::vector<std::uint32_t>& taps, std::uint32_t hidden,
    const Tokenizer* tokenizer, const AcceptanceJob& job) {
  if (job.prompt.empty() || job.max_tokens == 0)
    Fail("acceptance needs a prompt and at least one token to generate");
  const bool forced = !job.stream.empty();
  const std::uint32_t draft_count = job.draft_count;
  const std::size_t feature_row = std::size_t{taps.size()} * hidden;
  std::string error;
  std::vector<float> features;
  std::uint32_t injected = 0;

  // The drafter reads the target's residuals up to the position it drafts at.
  const auto inject_to = [&](std::uint32_t position) {
    if (position == injected)
      return;
    features.resize(std::size_t{position - injected} * feature_row);
    for (std::uint32_t r = injected; r < position; ++r) {
      for (std::size_t t = 0; t < taps.size(); ++t) {
        const auto captured = session.CapturedHidden(r, t);
        if (captured.size() != hidden)
          Fail("residual for row " + std::to_string(r) +
               " is not in the ring; lower the prompt length or the budget");
        std::memcpy(features.data() +
                        (std::size_t{r - injected} * taps.size() + t) * hidden,
                    captured.data(), hidden * sizeof(float));
      }
    }
    if (!forward.AppendContext(features, injected, position - injected, &error))
      Fail("context injection: " + error);
    injected = position;
  };

  // Each prompt starts from an empty target cache and an empty drafter
  // context.
  session.Rewind(0);
  forward.ResetContext();
  // The hidden ring keeps only kHiddenRingRows positions. Prefill in chunks and
  // let the drafter take each chunk's rows before the next chunk can push them
  // out of the ring.
  constexpr std::size_t kPrefillChunk = 1024;
  for (std::size_t offset = 0; offset < job.prompt.size();
       offset += kPrefillChunk) {
    const std::size_t count =
        std::min(kPrefillChunk, job.prompt.size() - offset);
    std::vector<std::int32_t> chunk(count);
    for (std::size_t i = 0; i < count; ++i)
      chunk[i] = static_cast<std::int32_t>(job.prompt[offset + i]);
    if (!session.Append(chunk, &error))
      Fail("prompt prefill: " + error);
    if (session.Position() != offset + count)
      Fail("prompt prefill stopped short of the requested position");
    inject_to(session.Position());
  }

  AcceptanceResult result;
  result.prompt_len = job.prompt.size();
  result.rows.resize(draft_count);

  std::uint32_t anchor = Argmax(session.Logits());
  if (forced) {
    // The stream's first token replaces the target's. The comparison still
    // counts toward the divergence: the prompt's logits are a real prediction.
    const std::uint32_t target_first = anchor;
    anchor = job.stream.front();
    ++result.divergence_rows;
    if (target_first != anchor)
      ++result.divergence_mismatches;
  }
  bool done = IsStop(tokenizer, anchor) && job.stop_on_eos;
  // max_tokens counts the first token, which is not part of `emitted`.
  const std::size_t budget = job.max_tokens - 1U;

  while (!done && result.emitted < budget) {
    const std::uint32_t position = session.Position();
    if (position + draft_count + 1 > session.Capacity())
      Fail("target session is too small for the next block; raise its context");
    inject_to(position);
    dflash::DflashDraft draft;
    if (!forward.Run(anchor, position, draft_count, 0.0F, {}, {}, &draft,
                     &error))
      Fail("draft block: " + error);
    if (draft.tokens.size() != draft_count)
      Fail("draft block returned the wrong proposal count");
    std::vector<std::int32_t> batch(draft_count + 1);
    batch[0] = static_cast<std::int32_t>(anchor);
    for (std::uint32_t i = 0; i < draft_count; ++i)
      batch[i + 1] = static_cast<std::int32_t>(draft.tokens[i]);
    if (!session.AppendAll(batch, &error))
      Fail("verify pass: " + error);

    // Stream index of the token that proposal i (and verify row i) predicts.
    const std::size_t base = result.emitted + 1;
    const auto stream_at = [&](std::size_t index) -> const std::uint32_t* {
      return index < job.stream.size() ? &job.stream[index] : nullptr;
    };
    const auto matches = [&](std::uint32_t i) {
      if (!forced)
        return Argmax(session.Logits(i)) == draft.tokens[i];
      const std::uint32_t* reference = stream_at(base + i);
      return reference != nullptr && *reference == draft.tokens[i];
    };
    std::uint32_t accepted = 0;
    while (accepted < draft_count && matches(accepted))
      ++accepted;

    // Divergence: rows 0..accepted are conditioned on the stream prefix (the
    // accepted drafts equal it), so their argmax is the target's own greedy
    // choice on the shared text.
    if (forced) {
      for (std::uint32_t row = 0; row <= accepted; ++row) {
        const std::uint32_t* reference = stream_at(base + row);
        if (reference == nullptr)
          break;
        ++result.divergence_rows;
        if (Argmax(session.Logits(row)) != *reference)
          ++result.divergence_mismatches;
      }
    }

    // The token after the accepted drafts: the target's argmax when free, the
    // stream's when forced. Empty means the forced stream has run out.
    std::optional<std::uint32_t> bonus;
    if (forced) {
      if (const std::uint32_t* reference = stream_at(base + accepted))
        bonus = *reference;
    } else {
      bonus = Argmax(session.Logits(accepted));
    }

    for (std::uint32_t i = 0; i < draft_count; ++i) {
      ++result.rows[i].drafted;
      if (i < accepted)
        ++result.rows[i].accepted;
    }
    result.drafted += draft_count;
    result.accepted += accepted;
    ++result.cycles;

    // Commit the accepted drafts and the bonus. Clip at the budget, and at the
    // stop token, which is counted and ends the run.
    for (std::uint32_t j = 0; j <= accepted; ++j) {
      std::uint32_t token = 0;
      if (j < accepted) {
        token = draft.tokens[j];
      } else if (bonus) {
        token = *bonus;
      } else {
        done = true;
        break;
      }
      if (result.emitted >= budget) {
        done = true;
        break;
      }
      ++result.emitted;
      anchor = token;
      if (job.stop_on_eos && IsStop(tokenizer, token)) {
        done = true;
        break;
      }
    }
    // Keep the anchor and the accepted drafts; the next block re-appends the
    // bonus as its anchor.
    session.Rewind(position + accepted + 1);
  }
  return result;
}

/// M1b's --accept: greedy acceptance on one repeated-text prompt, free-running.
/// Prints the markdown table.
[[nodiscard]] int MeasureAcceptance(
    flint::Session& session, dflash::DflashForward& forward,
    const std::vector<std::uint32_t>& taps, std::uint32_t hidden,
    std::uint32_t draft_count, const std::vector<std::uint32_t>& prompt,
    std::uint32_t budget, const std::string& head_route) {
  AcceptanceJob job;
  job.prompt = prompt;
  job.draft_count = draft_count;
  job.max_tokens = budget + 1U;
  const AcceptanceResult result =
      RunAcceptance(session, forward, taps, hidden, nullptr, job);
  if (result.cycles == 0)
    Fail("no cycle ran; the context is too small for one block");

  std::cout << "| quantity | value |\n| :-- | --: |\n"
            << "| prefill rows | " << prompt.size() << " |\n"
            << "| proposals per block | " << draft_count << " |\n"
            << "| cycles | " << result.cycles << " |\n"
            << "| tokens emitted | " << result.emitted << " |\n"
            << "| accepted drafts | " << result.accepted << " |\n"
            << "| acceptance | "
            << 100.0 * static_cast<double>(result.accepted) /
                   (static_cast<double>(result.cycles) * draft_count)
            << "% |\n"
            << "| tokens per cycle | "
            << (Ratio(result.emitted, result.cycles)) << " |\n"
            << "| head route | " << head_route << " |\n\n"
            << "| proposal | drafted | accepted | acceptance |\n"
            << "| --: | --: | --: | --: |\n";
  for (std::uint32_t i = 0; i < draft_count; ++i) {
    const auto& row = result.rows[i];
    std::cout << "| " << i << " | " << row.drafted << " | " << row.accepted
              << " | "
              << (row.drafted == 0 ? 0.0
                                   : 100.0 * Ratio(row.accepted, row.drafted))
              << "% |\n";
  }
  std::cout << "\nacceptance is `P(the target's greedy token equals the draft)`"
               " -- distribution overlap, not the min(1, p/q) rate.\n";
  return 0;
}

/// The drafter, borrowing the target's tied embedding and LM head.
[[nodiscard]] std::unique_ptr<dflash::DflashForward> CreateDrafter(
    const Target& target, const dflash::DflashWeights& weights,
    const std::string& head, std::uint32_t max_context) {
  const flint::TiedWeights tied = target.model->Tied();
  if (tied.token_embedding == nullptr || tied.output == nullptr)
    Fail("target exposes no tied weights");
  const dflash::DflashBorrowed borrowed{
      .token_embedding = tied.token_embedding,
      .token_embedding_type = tied.token_embedding_type,
      .output = tied.output,
      .output_type = tied.output_type};
  std::string error;
  auto forward = dflash::DflashForward::Create(
      weights, borrowed, max_context, &error,
      head == "reference" ? dflash::DflashHeadRoute::kFp32
                          : dflash::DflashHeadRoute::kNative);
  if (forward == nullptr)
    Fail("drafter forward: " + error);
  return forward;
}

/// One prompt from the shared workload dump (M1c B2).
struct PromptRecord {
  std::string id;
  std::string category;
  std::vector<std::uint32_t> prompt;
  std::uint32_t max_tokens{0};
  bool stop_on_eos{false};
};

[[nodiscard]] std::vector<PromptRecord> ReadPromptIds(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input)
    Fail("cannot read " + path);
  std::vector<PromptRecord> records;
  std::string line;
  std::size_t line_number = 0;
  while (std::getline(input, line)) {
    ++line_number;
    if (line.find_first_not_of(" \t\r") == std::string::npos)
      continue;
    const std::string where = path + ":" + std::to_string(line_number) + ": ";
    gufo::json::Value spec;
    try {
      spec = gufo::json::parse(line);
    } catch (const std::exception& error) {
      Fail(where + error.what());
    }
    PromptRecord record;
    record.id = spec.member_str("id");
    record.category = spec.member_str("category");
    record.max_tokens =
        static_cast<std::uint32_t>(spec.member_size("max_tokens"));
    const gufo::json::Value* ids = spec.find("prompt_ids");
    if (record.id.empty() || record.max_tokens == 0 || ids == nullptr ||
        !ids->is_array())
      Fail(where + "needs id, max_tokens and prompt_ids");
    for (const auto& token : ids->items())
      record.prompt.push_back(static_cast<std::uint32_t>(token.as_size()));
    const gufo::json::Value* stop = spec.find("stop_on_eos");
    record.stop_on_eos = stop != nullptr && stop->as_bool();
    records.push_back(std::move(record));
  }
  return records;
}

/// Forced streams: the oracle's generated ids keyed by prompt id. Lines without
/// a token array (the oracle's trailing summary) are skipped.
[[nodiscard]] std::map<std::string, std::vector<std::uint32_t>> ReadStreams(
    const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input)
    Fail("cannot read " + path);
  std::map<std::string, std::vector<std::uint32_t>> streams;
  std::string line;
  std::size_t line_number = 0;
  while (std::getline(input, line)) {
    ++line_number;
    if (line.find_first_not_of(" \t\r") == std::string::npos)
      continue;
    const std::string where = path + ":" + std::to_string(line_number) + ": ";
    gufo::json::Value spec;
    try {
      spec = gufo::json::parse(line);
    } catch (const std::exception& error) {
      Fail(where + error.what());
    }
    const gufo::json::Value* tokens = spec.find("tokens");
    if (tokens == nullptr || !tokens->is_array())
      continue;
    const std::string id = spec.member_str("id");
    if (id.empty())
      Fail(where + "stream line needs an id");
    std::vector<std::uint32_t> ids;
    for (const auto& token : tokens->items())
      ids.push_back(static_cast<std::uint32_t>(token.as_size()));
    streams[id] = std::move(ids);
  }
  return streams;
}

/// One report line: the fields the compare step joins on `id`.
[[nodiscard]] std::string ReportLine(const PromptRecord& record,
                                     const AcceptanceResult& result,
                                     bool forced) {
  const auto round4 = [](double value) {
    return std::round(value * 1e4) / 1e4;
  };
  const double acceptance =
      result.drafted == 0 ? 0.0
                          : Ratio(result.accepted, result.drafted);
  const double mean_len =
      result.cycles == 0 ? 0.0 : Ratio(result.emitted, result.cycles);
  using U = unsigned long long;
  gufo::json::Value line = gufo::json::Value::object();
  line["id"] = record.id;
  line["category"] = record.category;
  line["prompt_len"] = static_cast<U>(result.prompt_len);
  line["emitted"] = static_cast<U>(result.emitted);
  line["cycles"] = static_cast<U>(result.cycles);
  line["drafted"] = static_cast<U>(result.drafted);
  line["accepted"] = static_cast<U>(result.accepted);
  line["acceptance"] = round4(acceptance);
  line["mean_len"] = round4(mean_len);
  gufo::json::Value per_row = gufo::json::Value::array();
  for (const auto& row : result.rows)
    per_row.push_back(gufo::json::Value(static_cast<U>(row.accepted)));
  line["per_row_accepted"] = per_row;
  if (forced) {
    const double divergence =
        result.divergence_rows == 0
            ? 0.0
            : Ratio(result.divergence_mismatches, result.divergence_rows);
    line["stream_divergence"] = round4(divergence);
  }
  return line.dump();
}

/// M1c: the shared workload. One target and one drafter serve every prompt;
/// each prompt starts from an empty session and an empty drafter context.
[[nodiscard]] int MeasurePromptIds(const Options& options,
                                   const Target& target,
                                   const dflash::DflashWeights& weights,
                                   const std::vector<std::uint32_t>& taps,
                                   std::uint32_t draft_count) {
  const std::vector<PromptRecord> records = ReadPromptIds(options.prompts_ids);
  if (records.empty())
    Fail("no prompts in " + options.prompts_ids);
  std::map<std::string, std::vector<std::uint32_t>> streams;
  if (!options.stream_path.empty())
    streams = ReadStreams(options.stream_path);

  // Each session needs its prompt, the whole budget and one block of headroom.
  std::size_t longest = 0;
  for (const auto& record : records)
    longest = std::max(longest, record.prompt.size() + record.max_tokens);
  const auto context = static_cast<std::uint32_t>(std::max<std::size_t>(
      longest + weights.config.block_size + 16U,
      flint::Session::kHiddenRingRows));

  std::string error;
  auto session = target.model->CreateSession(context, &error);
  if (session == nullptr)
    Fail("target session: " + error);
  if (!session->SetHiddenCapture(taps, &error))
    Fail("hidden capture: " + error);
  auto forward = CreateDrafter(target, weights, options.head, context);

  std::ofstream report(options.report_path, std::ios::binary | std::ios::trunc);
  if (!report)
    Fail("cannot open " + options.report_path);
  std::cout << "# DFlash2 acceptance on the shared workload\n\n"
            << "| quantity | value |\n| :-- | --: |\n"
            << "| prompts | " << records.size() << " |\n"
            << "| proposals per block | " << draft_count << " |\n"
            << "| head route | " << options.head << " |\n"
            << "| forced stream | "
            << (streams.empty() ? std::string("none") : options.stream_path)
            << " |\n\n";

  const bool forced = !streams.empty();
  for (const auto& record : records) {
    AcceptanceJob job;
    job.prompt = record.prompt;
    job.draft_count = draft_count;
    job.max_tokens = record.max_tokens;
    job.stop_on_eos = record.stop_on_eos;
    if (forced) {
      const auto found = streams.find(record.id);
      if (found == streams.end() || found->second.empty())
        Fail("stream " + options.stream_path + " has no tokens for prompt " +
             record.id);
      job.stream = found->second;
    }
    const AcceptanceResult result = RunAcceptance(
        *session, *forward, taps, target.model->config().hidden_size,
        target.tokenizer.get(), job);
    report << ReportLine(record, result, forced) << '\n';
    std::cout << record.id << "\tcycles " << result.cycles << "\temitted "
              << result.emitted << "\tacceptance "
              << (result.drafted == 0
                      ? 0.0
                      : Ratio(result.accepted, result.drafted))
              << "\tmean_len "
              << (result.cycles == 0
                      ? 0.0
                      : Ratio(result.emitted, result.cycles))
              << '\n';
  }
  report.flush();
  if (!report)
    Fail("cannot write " + options.report_path);
  std::cout << "wrote " << records.size() << " reports to "
            << options.report_path << '\n';
  return 0;
}

[[nodiscard]] int Run(const Options& options) {
  if (!options.dump_path.empty())
    return DumpPrompts(options);
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

  const auto taps = options.taps.empty() ? draft_config.target_layer_ids
                                         : options.taps;
  const std::uint32_t draft_count =
      std::min(options.draft_count == 0 ? draft_config.block_size - 1U
                                        : options.draft_count,
               draft_config.block_size - 1U);
  if (!options.prompts_ids.empty())
    return MeasurePromptIds(options, target, *weights, taps, draft_count);

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
  const std::uint32_t context = std::max<std::uint32_t>(
      options.position + options.accept_tokens + draft_config.block_size + 16U,
      flint::Session::kHiddenRingRows);
  auto session = target.model->CreateSession(context, &error);
  if (session == nullptr)
    Fail("target session: " + error);
  if (!session->SetHiddenCapture(taps, &error))
    Fail("hidden capture: " + error);

  auto forward = CreateDrafter(
      target, *weights, options.head,
      options.position + options.accept_tokens + draft_config.block_size);

  // M1b --accept: the acceptance loop does its own prefill, so it gets the
  // prompt ids and the session as it is.
  if (options.accept_tokens != 0) {
    const auto pool = PromptTokens(*target.tokenizer, options.text,
                                   options.position);
    std::vector<std::uint32_t> prompt(pool.begin(), pool.end());
    return MeasureAcceptance(*session, *forward, taps, config.hidden_size,
                             draft_count, prompt, options.accept_tokens,
                             options.head);
  }

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
  const std::uint32_t anchor = Argmax(session->Logits());

  const flint::TiedWeights tied = target.model->Tied();
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
