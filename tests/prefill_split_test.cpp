// Prefill split invariance against the real target model.
//
// Session::Prefill must leave the same K/V rows and logits whether a prompt
// arrives in one call or in two calls split at any even row: that is what lets
// a cached prefix continue exactly (the cache restores at a checkpoint and the
// rest of the prompt is a second call). For each split row `a` the test
// compares one call over the prompt with calls [0, a) and [a, N): logits
// bits, and the K and V rows of every layer, rows below the split and rows at
// or past it counted separately. It also checks that Prefill's logits equal
// Append's on the unsplit prompt.
//
// When rows differ, the first differing layer and the side of the split point
// at the kernel:
//   rows < a differ     a kernel's result depends on how many rows a pass has
//   only rows >= a      the second pass reads the first pass's rows differently
//   layer 0 K/V differ  the QKV projection itself
//
// FLINT_SPLITS overrides the split rows (an odd row is outside the contract
// and reported only), FLINT_SPLIT_TOKENS the prompt length, and
// FLINT_SPLIT_APPEND=1 splits with Append instead. Fails (exit 1) when an even
// split differs. The target is FLINT_TARGET, or the Q4_K_XL artifact the
// other tools default to. If the file or a HIP device is absent the test
// skips with exit 0.

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
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

namespace flint = gufo::models::flint::gpu;
namespace core = gufo::core;

namespace {

constexpr const char* kDefaultTarget = "/home/bruno/AI/GGUF/flint-UD-Q4_K_XL.gguf";
constexpr std::uint32_t kContext = 8192;
const std::uint32_t kPromptTokens = [] {
  const char* e = std::getenv("FLINT_SPLIT_TOKENS");
  return e != nullptr ? static_cast<std::uint32_t>(std::atoi(e)) : 6000U;
}();

void RequireOk(bool ok, const std::string& error, const char* what) {
  if (!ok)
    throw std::runtime_error(std::string(what) + ": " + error);
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
  // Varied text, so neighbouring rows differ and a row mix-up shows.
  std::vector<std::int32_t> ids;
  for (std::uint32_t i = 0; ids.size() < tokens; ++i) {
    const std::string text =
        "Record " + std::to_string(i * 7919 % 100003) +
        ": the river passes mill " + std::to_string(i % 37) +
        ", herons wait by reed bank " + std::to_string(i * 31 % 211) + ". ";
    for (const std::uint32_t id : tokenizer->Encode(text))
      ids.push_back(static_cast<std::int32_t>(id));
  }
  ids.resize(tokens);
  return ids;
}

std::vector<std::uint8_t> CopyRows(const flint::Session& s, std::uint32_t layer,
                                   bool value, std::uint32_t rows) {
  std::vector<std::uint8_t> out(std::size_t{rows} * s.KvRowBytes());
  if (hipMemcpy(out.data(), s.KvRows(layer, value), out.size(),
                hipMemcpyDeviceToHost) != hipSuccess)
    throw std::runtime_error("hipMemcpy of KV rows failed");
  return out;
}

struct RowDiff {
  std::uint32_t below{0};   // differing rows < split
  std::uint32_t above{0};   // differing rows >= split
  std::uint32_t first{~0U};
};

RowDiff Compare(const std::vector<std::uint8_t>& a,
                const std::vector<std::uint8_t>& b, std::size_t row_bytes,
                std::uint32_t rows, std::uint32_t split) {
  RowDiff d;
  for (std::uint32_t r = 0; r < rows; ++r) {
    if (std::memcmp(a.data() + r * row_bytes, b.data() + r * row_bytes,
                    row_bytes) == 0)
      continue;
    (r < split ? d.below : d.above)++;
    d.first = std::min(d.first, r);
  }
  return d;
}

/// Prefills `prompt` in calls ending at each row of `cuts` (then the end).
/// FLINT_SPLIT_APPEND=1 uses Append instead of Prefill.
const bool use_append = std::getenv("FLINT_SPLIT_APPEND") != nullptr;

std::unique_ptr<flint::Session> Prefill(flint::Model& model,
                                        std::span<const std::int32_t> prompt,
                                        std::span<const std::uint32_t> cuts) {
  std::string error;
  auto s = model.CreateSession(kContext, &error);
  RequireOk(s != nullptr, error, "session");
  std::uint32_t at = 0;
  auto append_to = [&](std::uint32_t end) {
    const auto part = prompt.subspan(at, end - at);
    RequireOk(use_append ? s->Append(part, &error) : s->Prefill(part, &error),
              error, "prefill");
    at = end;
  };
  for (const std::uint32_t cut : cuts)
    append_to(cut);
  append_to(static_cast<std::uint32_t>(prompt.size()));
  return s;
}

}  // namespace

int main() {
  const char* env = std::getenv("FLINT_TARGET");
  const std::string target =
      (env != nullptr && env[0] != '\0') ? env : kDefaultTarget;
  if (!std::filesystem::exists(target)) {
    std::cout << "flint prefill split test skipped: no target model at "
              << target << "\n";
    return 0;
  }
  int devices = 0;
  if (hipGetDeviceCount(&devices) != hipSuccess || devices == 0) {
    std::cout << "flint prefill split test skipped: no HIP device\n";
    return 0;
  }
  std::vector<std::uint32_t> splits = {2,    4,    8,    10,   16,   100,
                                       1000, 2048, 4094, 4096, 4098, 5000,
                                       5990, 5998};
  if (const char* list = std::getenv("FLINT_SPLITS")) {
    splits.clear();
    for (std::string s(list); !s.empty();) {
      const auto comma = s.find(',');
      splits.push_back(static_cast<std::uint32_t>(std::stoul(s.substr(0, comma))));
      s = comma == std::string::npos ? "" : s.substr(comma + 1);
    }
  }
  int differing = 0;
  try {
    const auto prompt = EncodePrompt(target, kPromptTokens);
    std::string error;
    auto model = flint::Model::Load(target, {}, &error);
    RequireOk(model != nullptr, error, "target model");

    auto ref = Prefill(*model, prompt, {});
    const std::uint32_t layers = ref->NumLayers();
    const std::size_t row_bytes = ref->KvRowBytes();
    std::vector<std::vector<std::uint8_t>> ref_rows;
    for (std::uint32_t il = 0; il < layers; ++il)
      for (const bool value : {false, true})
        ref_rows.push_back(CopyRows(*ref, il, value, kPromptTokens));
    const auto ref_logits = ref->Logits();
    const std::vector<float> ref_logit(ref_logits.begin(), ref_logits.end());
    ref.reset();

    {
      // Prefill vs Append on the unsplit prompt: the wide path must agree
      // with the established one up to rounding (same argmax, small diff).
      std::string e2;
      auto app = model->CreateSession(kContext, &e2);
      RequireOk(app != nullptr && app->Append(prompt, &e2), e2, "append ref");
      const auto la = app->Logits();
      float md = 0.0F;
      std::size_t am = 0, bm = 0;
      for (std::size_t i = 0; i < la.size(); ++i) {
        md = std::max(md, std::fabs(la[i] - ref_logit[i]));
        if (la[i] > la[am]) am = i;
        if (ref_logit[i] > ref_logit[bm]) bm = i;
      }
      std::cout << "prefill vs append (unsplit): max |diff| " << md
                << ", argmax " << bm << " vs " << am << ", top logit "
                << ref_logit[bm] << " vs " << la[am] << "\n";
    }
    std::cout << "prompt " << kPromptTokens << " tokens, " << layers
              << " layers; reference = one Append\n";
    for (const std::uint32_t split : splits) {
      const std::uint32_t cut[1] = {split};
      auto s = Prefill(*model, prompt, cut);
      const auto logits = s->Logits();
      float max_diff = 0.0F;
      for (std::size_t i = 0; i < logits.size(); ++i)
        max_diff = std::max(max_diff, std::fabs(logits[i] - ref_logit[i]));
      const bool same_logits =
          std::memcmp(logits.data(), ref_logit.data(),
                      ref_logit.size() * sizeof(float)) == 0;
      int first_layer = -1, diff_layers = 0;
      std::string detail;
      for (std::uint32_t il = 0; il < layers; ++il) {
        for (const bool value : {false, true}) {
          const auto rows = CopyRows(*s, il, value, kPromptTokens);
          const RowDiff d = Compare(ref_rows[il * 2 + (value ? 1 : 0)], rows,
                                    row_bytes, kPromptTokens, split);
          if (d.below + d.above == 0)
            continue;
          if (first_layer < 0) {
            first_layer = static_cast<int>(il);
            detail = std::string(value ? "V" : "K") + " rows below split " +
                     std::to_string(d.below) + ", at/after " +
                     std::to_string(d.above) + ", first row " +
                     std::to_string(d.first);
          }
          if (!value || first_layer != static_cast<int>(il))
            ++diff_layers;
        }
      }
      if ((!same_logits || first_layer >= 0) && split % 2 == 0)
        ++differing;
      std::cout << "split " << split << ": logits "
                << (same_logits ? "identical" : "differ")
                << " (max |diff| " << max_diff << ")";
      if (first_layer < 0)
        std::cout << ", all K/V rows identical\n";
      else
        std::cout << ", first differing layer " << first_layer << " (" << detail
                  << "), layers with differences ~" << diff_layers << "\n";
    }
  } catch (const std::exception& error) {
    std::cerr << "flint prefill split test failed: " << error.what() << "\n";
    return 1;
  }
  std::cout << differing << " of the even splits differ from one call\n";
  return differing > 0 ? 1 : 0;
}
