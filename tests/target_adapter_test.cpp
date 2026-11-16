// Target adapter acceptance (M2) against the real target model.
//
//   1. VerifyRows rows are bit-identical to single-token appends for every
//      width 1..kMaxVerifyRows (CheckRowsBitIdentical).
//   2. Features() returns the same bits as a fresh session that captured the
//      same prefix, and the packed layout is row-major by position then tap.
//   3. Rewind() leaves the session in the state of a fresh session that
//      appended exactly that prefix: position, logits, and the next append.
//
// The target is FLINT_TARGET, or the Q4_K_XL artifact the other tools default
// to. If the file or a HIP device is absent the test skips with exit 0.

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <gufo/ext/gguf_reader.hpp>

#include "hip/vendor/qwen/tokenizer.hpp"
#include "src/gpu/engine.hpp"
#include "src/spec/target_adapter.hpp"

namespace flint = gufo::models::flint::gpu;
namespace spec = gufo::models::flint::spec;
namespace core = gufo::core;

namespace {

constexpr const char* kDefaultTarget = "/home/bruno/AI/GGUF/flint-UD-Q4_K_XL.gguf";
// The drafter's taps: residual outputs of layers 5,15,26,36,46 (the layer
// inputs 6,16,27,37,47 that llama.cpp lists).
constexpr std::uint32_t kTaps[] = {5, 15, 26, 36, 46};
constexpr std::uint32_t kContext = 512;

void Require(bool condition, const std::string& message) {
  if (!condition)
    throw std::runtime_error(message);
}

void RequireOk(bool ok, const std::string& error, const char* what) {
  if (!ok)
    throw std::runtime_error(std::string(what) + ": " + error);
}

/// Exact float-bit equality of two spans.
bool SameBits(std::span<const float> a, std::span<const float> b) {
  return a.size() == b.size() &&
         (a.empty() || std::memcmp(a.data(), b.data(),
                                   a.size() * sizeof(float)) == 0);
}

std::vector<std::int32_t> EncodePrompt(const std::string& path,
                                       std::size_t tokens) {
  std::string error;
  auto metadata = core::GgufReader::OpenFile(path, &error);
  if (!metadata)
    throw std::runtime_error("target GGUF: " + error);
  auto tokenizer =
      gufo::vendor::qwen::tokenization::QwenTokenizer::CreateFromGguf(*metadata,
                                                                     &error);
  if (tokenizer == nullptr)
    throw std::runtime_error("target tokenizer: " + error);
  const std::string text =
      "A block diffusion drafter proposes a whole block at once, and the target "
      "checks every proposal in one batched pass. Acceptance decides whether the "
      "trade pays, and bit-identical rows decide whether acceptance is honest. ";
  std::vector<std::int32_t> ids;
  while (ids.size() < tokens) {
    for (const std::uint32_t id : tokenizer->Encode(text))
      ids.push_back(static_cast<std::int32_t>(id));
  }
  ids.resize(tokens);
  return ids;
}

struct Fixture {
  std::shared_ptr<flint::Model> model;
  std::vector<std::int32_t> prompt;
  std::uint32_t hidden{0};
};

void TestRowsBitIdentical(const Fixture& fx) {
  std::string error;
  auto session = fx.model->CreateSession(kContext, &error);
  RequireOk(session != nullptr, error, "session");
  auto adapter = spec::TargetAdapter::Create(*session, kTaps, fx.hidden, &error);
  RequireOk(adapter != nullptr, error, "adapter");

  RequireOk(adapter->AppendPrompt(fx.prompt, &error), error, "prompt");
  Require(adapter->Position() == fx.prompt.size(), "prompt position");

  // Anchor plus drafts: any tokens work for the property; use prompt-derived
  // ids so the widths see realistic rows.
  for (std::uint32_t width = 1; width <= flint::Session::kMaxVerifyRows;
       ++width) {
    std::vector<std::int32_t> tokens(width);
    for (std::uint32_t i = 0; i < width; ++i)
      tokens[i] = fx.prompt[(i * 37 + width) % fx.prompt.size()];
    const std::uint32_t before = adapter->Position();
    RequireOk(adapter->CheckRowsBitIdentical(tokens, &error), error,
              "rows bit-identical");
    Require(adapter->Position() == before + width,
            "row check leaves the verify state at position + width");
    adapter->Rewind(before);
  }
  std::cout << "  rows bit-identical for widths 1.."
            << flint::Session::kMaxVerifyRows << "\n";
}

void TestFeatures(const Fixture& fx) {
  std::string error;
  const std::size_t n = 96;
  const std::span<const std::int32_t> prefix(fx.prompt.data(), n);

  auto session = fx.model->CreateSession(kContext, &error);
  RequireOk(session != nullptr, error, "session");
  auto adapter = spec::TargetAdapter::Create(*session, kTaps, fx.hidden, &error);
  RequireOk(adapter != nullptr, error, "adapter");
  RequireOk(adapter->AppendPrompt(prefix, &error), error, "prefix");

  // A second, independent session with the same prefix: the reference bits.
  auto fresh = fx.model->CreateSession(kContext, &error);
  RequireOk(fresh != nullptr, error, "fresh session");
  auto fresh_adapter =
      spec::TargetAdapter::Create(*fresh, kTaps, fx.hidden, &error);
  RequireOk(fresh_adapter != nullptr, error, "fresh adapter");
  RequireOk(fresh_adapter->AppendPrompt(prefix, &error), error, "fresh prefix");

  std::vector<float> features;
  RequireOk(adapter->Features(0, static_cast<std::uint32_t>(n), &features,
                              &error),
            error, "features");
  std::vector<float> reference;
  RequireOk(fresh_adapter->Features(0, static_cast<std::uint32_t>(n),
                                    &reference, &error),
            error, "fresh features");
  const std::size_t row_floats = std::size_t{std::size(kTaps)} * fx.hidden;
  Require(features.size() == n * row_floats, "feature count");
  Require(SameBits(features, reference), "features bit-identical across sessions");

  // Layout: row-major by position, then tap. Check a sub-range against the
  // per-(row,tap) captured spans directly.
  std::vector<float> window;
  RequireOk(adapter->Features(40, 3, &window, &error), error, "window");
  for (std::uint32_t r = 0; r < 3; ++r) {
    for (std::size_t t = 0; t < std::size(kTaps); ++t) {
      const auto captured = session->CapturedHidden(40 + r, static_cast<std::uint32_t>(t));
      Require(captured.size() == fx.hidden, "captured width");
      Require(std::memcmp(window.data() + (r * std::size(kTaps) + t) * fx.hidden,
                          captured.data(), fx.hidden * sizeof(float)) == 0,
              "packed layout matches row-major position-then-tap order");
    }
  }

  // Features past the current position are refused, not read.
  std::vector<float> beyond;
  Require(!adapter->Features(static_cast<std::uint32_t>(n), 1, &beyond, &error),
          "features past the position must fail");
  std::cout << "  features bit-identical to a fresh capture over " << n
            << " rows, " << std::size(kTaps) << " taps\n";
}

void TestRewind(const Fixture& fx) {
  std::string error;
  const std::size_t base = 64;
  const std::span<const std::int32_t> prefix(fx.prompt.data(), base);

  auto session = fx.model->CreateSession(kContext, &error);
  RequireOk(session != nullptr, error, "session");
  auto adapter = spec::TargetAdapter::Create(*session, kTaps, fx.hidden, &error);
  RequireOk(adapter != nullptr, error, "adapter");
  RequireOk(adapter->AppendPrompt(prefix, &error), error, "prefix");

  // Speculate a 7-draft block past the prefix, then rewind to prefix + 2.
  std::vector<std::int32_t> block(8);
  for (std::size_t i = 0; i < block.size(); ++i)
    block[i] = fx.prompt[base + i];
  RequireOk(adapter->VerifyRows(block, &error), error, "verify");
  const std::uint32_t keep = static_cast<std::uint32_t>(base + 2);
  adapter->Rewind(keep);
  Require(adapter->Position() == keep, "position after rewind");

  // Fresh session that appended exactly those `keep` tokens.
  auto fresh = fx.model->CreateSession(kContext, &error);
  RequireOk(fresh != nullptr, error, "fresh session");
  auto fresh_adapter =
      spec::TargetAdapter::Create(*fresh, kTaps, fx.hidden, &error);
  RequireOk(fresh_adapter != nullptr, error, "fresh adapter");
  // Prefill the base and decode the rest one token at a time, the same way the
  // rewound session wrote its rows. Multi-token prefill is not bit-identical
  // to single-token decode (different kernels), so the reference must use the
  // decode path for rows past the prefix.
  RequireOk(fresh_adapter->AppendPrompt(
                std::span<const std::int32_t>(fx.prompt.data(), base), &error),
            error, "fresh prefix");
  for (std::size_t k = base; k < keep; ++k)
    RequireOk(fresh_adapter->VerifyRows({&fx.prompt[k], 1}, &error), error,
              "fresh decode");
  Require(fresh_adapter->Position() == keep, "fresh position");

  // Logits() is NOT restored by Rewind (it still shows the verify pass's last
  // row until the next append), so the check starts at the next append.
  for (std::size_t i = keep; i < keep + 4; ++i) {
    const std::int32_t token = fx.prompt[i];
    RequireOk(adapter->VerifyRows({&token, 1}, &error), error, "next append");
    RequireOk(fresh_adapter->VerifyRows({&token, 1}, &error), error,
              "fresh next append");
    Require(SameBits(adapter->Logits(), fresh_adapter->Logits()),
            "next append after rewind matches a fresh session");
  }
  std::cout << "  rewind to " << keep
            << " matches a fresh session (position and next 4 appends)\n";
}

}  // namespace

int main() {
  const char* env = std::getenv("FLINT_TARGET");
  const std::string target = (env != nullptr && env[0] != '\0') ? env : kDefaultTarget;
  if (!std::filesystem::exists(target)) {
    std::cout << "flint target adapter tests skipped: no target model at "
              << target << "\n";
    return 0;
  }
  int devices = 0;
  if (hipGetDeviceCount(&devices) != hipSuccess || devices == 0) {
    std::cout << "flint target adapter tests skipped: no HIP device\n";
    return 0;
  }
  try {
    Fixture fx;
    std::string error;
    fx.model = flint::Model::Load(target, {}, &error);
    RequireOk(fx.model != nullptr, error, "target model");
    fx.hidden = fx.model->config().hidden_size;
    fx.prompt = EncodePrompt(target, 256);
    TestRowsBitIdentical(fx);
    TestFeatures(fx);
    TestRewind(fx);
  } catch (const std::exception& error) {
    std::cerr << "flint target adapter tests failed: " << error.what() << "\n";
    return 1;
  }
  std::cout << "flint target adapter tests passed\n";
  return 0;
}
