// Session row access against the real target model (Stage 1 of the KV-rows
// plan). Checks, against a fresh reference session with the same prefix:
//
//   1. SaveSnapshot / RestoreSnapshot round trip: position, logits bits and
//      the next append are bit-identical to the reference.
//   2. Split path: copy only the K/V rows out per layer (KvRows + hipMemcpy),
//      then RestoreRows + SetLogits. Same three assertions.
//   3. SetRowWriteHook: one call per Append/AppendAll covering exactly the
//      rows written, none from Rewind.
//
// Sparse attention is read from FLINT_SPARSE_ATTN when a session is created,
// so the test runs its body twice in-process: once with the variable unset,
// once set to a valid top-k. The sparse leg is only meaningful when the
// context is long enough for sparse attention to engage; at kContext it
// exercises key-bound pooling and rebuild only.
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

namespace flint = gufo::models::flint::gpu;
namespace core = gufo::core;

namespace {

constexpr const char* kDefaultTarget = "/home/bruno/AI/GGUF/flint-UD-Q4_K_XL.gguf";
constexpr std::uint32_t kContext = 512;
constexpr std::uint32_t kPrefix = 64;
constexpr std::uint32_t kExtra = 16;

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
      "Snapshots and split row copies must reproduce the continuation exactly, "
      "so a restored session samples the same token as one that never stopped. ";
  std::vector<std::int32_t> ids;
  while (ids.size() < tokens) {
    for (const std::uint32_t id : tokenizer->Encode(text))
      ids.push_back(static_cast<std::int32_t>(id));
  }
  ids.resize(tokens);
  return ids;
}

void CopyOrThrow(void* dst, const void* src, std::size_t bytes, hipMemcpyKind kind,
                 const char* what) {
  if (hipMemcpy(dst, src, bytes, kind) != hipSuccess)
    throw std::runtime_error(std::string(what) + ": hipMemcpy failed");
}

std::unique_ptr<flint::Session> Make(flint::Model& model) {
  std::string error;
  auto session = model.CreateSession(kContext, &error);
  RequireOk(session != nullptr, error, "session");
  return session;
}

/// Asserts the restored `session` matches `reference` after the append of
/// `next` on both. `reference` is at the prefix position.
void CheckContinuation(flint::Session& session, flint::Session& reference,
                       std::int32_t next, const char* what) {
  std::string error;
  Require(session.Position() == kPrefix, std::string(what) + ": position");
  Require(SameBits(session.Logits(), reference.Logits()),
          std::string(what) + ": logits bit-identical");
  const std::int32_t token[1] = {next};
  RequireOk(session.Append(token, &error), error, what);
  RequireOk(reference.Append(token, &error), error, what);
  Require(session.Position() == reference.Position(),
          std::string(what) + ": position after append");
  Require(SameBits(session.Logits(), reference.Logits()),
          std::string(what) + ": next append bit-identical");
}

void TestSnapshotRoundTrip(flint::Model& model,
                           const std::vector<std::int32_t>& prompt) {
  std::string error;
  auto session = Make(model);
  auto reference = Make(model);
  const std::span<const std::int32_t> prefix(prompt.data(), kPrefix);
  const std::span<const std::int32_t> extra(prompt.data() + kPrefix, kExtra);
  RequireOk(session->Append(prefix, &error), error, "prefix");
  RequireOk(reference->Append(prefix, &error), error, "reference prefix");

  std::vector<std::uint8_t> payload(session->SnapshotBytes());
  RequireOk(session->SaveSnapshot(payload, &error), error, "save");
  RequireOk(session->Append(extra, &error), error, "extra");
  Require(session->Position() == kPrefix + kExtra, "extra position");

  RequireOk(session->RestoreSnapshot(payload, &error), error, "restore");
  CheckContinuation(*session, *reference, prompt[kPrefix + kExtra], "snapshot");
  std::cout << "  snapshot round trip ok\n";
}

void TestSplitPath(flint::Model& model,
                   const std::vector<std::int32_t>& prompt) {
  std::string error;
  auto session = Make(model);
  auto reference = Make(model);
  const std::span<const std::int32_t> prefix(prompt.data(), kPrefix);
  RequireOk(session->Append(prefix, &error), error, "prefix");
  RequireOk(reference->Append(prefix, &error), error, "reference prefix");

  const std::uint32_t layers = session->NumLayers();
  const std::size_t row = session->KvRowBytes();
  const std::size_t bytes = std::size_t{kPrefix} * row;
  Require(layers > 0 && row > 0, "layer count and row size");
  Require(session->KvRows(layers, false) == nullptr,
          "out-of-range layer returns nullptr");
  Require(session->KvRows(layers, true) == nullptr,
          "out-of-range layer returns nullptr (value)");

  // Capture only the valid rows of each layer, K then V.
  std::vector<std::vector<std::uint8_t>> captured(std::size_t{layers} * 2,
                                                  std::vector<std::uint8_t>(bytes));
  for (std::uint32_t il = 0; il < layers; ++il) {
    for (const bool value : {false, true}) {
      const void* src = session->KvRows(il, value);
      Require(src != nullptr, "KvRows returned nullptr for a valid layer");
      CopyOrThrow(captured[std::size_t{il} * 2 + (value ? 1 : 0)].data(), src,
                  bytes, hipMemcpyDeviceToHost, "capture");
    }
  }
  RequireOk(session->Synchronize(&error), error, "synchronize");
  const auto logits = session->Logits();
  const std::vector<float> logit_row(logits.begin(), logits.end());

  // Restore into a session that already holds other rows, so the copies must
  // overwrite them: rows first, then logits, then the position.
  auto restored = Make(model);
  RequireOk(restored->Append(std::span<const std::int32_t>(prompt.data(), 8),
                             &error),
            error, "dirty restore target");
  for (std::uint32_t il = 0; il < layers; ++il) {
    for (const bool value : {false, true}) {
      void* dst = restored->KvRows(il, value);
      Require(dst != nullptr, "KvRows returned nullptr for a valid layer");
      CopyOrThrow(dst, captured[std::size_t{il} * 2 + (value ? 1 : 0)].data(),
                  bytes, hipMemcpyHostToDevice, "restore");
    }
  }
  Require(!restored->SetLogits(std::span<const float>(logit_row.data(), 1),
                               &error),
          "SetLogits rejects a short row");
  RequireOk(restored->SetLogits(logit_row, &error), error, "set logits");
  RequireOk(restored->RestoreRows(kPrefix, &error), error, "restore rows");
  Require(!restored->RestoreRows(kContext + 1, &error),
          "RestoreRows rejects position > capacity");
  RequireOk(restored->RestoreRows(kPrefix, &error), error, "restore rows again");
  CheckContinuation(*restored, *reference, prompt[kPrefix], "split");
  std::cout << "  split row path ok\n";
}

void TestRowHook(flint::Model& model, const std::vector<std::int32_t>& prompt) {
  std::string error;
  auto session = Make(model);
  std::vector<std::pair<std::uint32_t, std::uint32_t>> calls;
  session->SetRowWriteHook([&calls](std::uint32_t first, std::uint32_t end) {
    calls.emplace_back(first, end);
  });

  const std::span<const std::int32_t> prefix(prompt.data(), kPrefix);
  RequireOk(session->Append(prefix, &error), error, "prefix");
  Require(calls.size() == 1 && calls[0].first == 0 &&
              calls[0].second == kPrefix,
          "Append reports one call covering the prefix");
  calls.clear();

  const std::int32_t abc[3] = {prompt[0], prompt[1], prompt[2]};
  RequireOk(session->Append(abc, &error), error, "append 3");
  Require(calls.size() == 1 && calls[0].first == kPrefix &&
              calls[0].second == kPrefix + 3,
          "Append(3) reports one call [position, position+3)");
  calls.clear();

  session->Rewind(kPrefix);
  Require(calls.empty(), "Rewind reports nothing");

  RequireOk(session->AppendAll(abc, &error), error, "appendall 3");
  Require(calls.size() == 1 && calls[0].first == kPrefix &&
              calls[0].second == kPrefix + 3,
          "AppendAll(3) reports one call of width 3");
  calls.clear();

  session->SetRowWriteHook({});
  RequireOk(session->Append(abc, &error), error, "append after clear");
  Require(calls.empty(), "cleared hook is not called");
  std::cout << "  row write hook ok\n";
}

void RunAll(const std::string& target, const std::vector<std::int32_t>& prompt) {
  std::string error;
  auto model = flint::Model::Load(target, {}, &error);
  RequireOk(model != nullptr, error, "target model");
  TestSnapshotRoundTrip(*model, prompt);
  TestSplitPath(*model, prompt);
  TestRowHook(*model, prompt);
}

}  // namespace

int main() {
  const char* env = std::getenv("FLINT_TARGET");
  const std::string target =
      (env != nullptr && env[0] != '\0') ? env : kDefaultTarget;
  if (!std::filesystem::exists(target)) {
    std::cout << "flint engine snapshot tests skipped: no target model at "
              << target << "\n";
    return 0;
  }
  int devices = 0;
  if (hipGetDeviceCount(&devices) != hipSuccess || devices == 0) {
    std::cout << "flint engine snapshot tests skipped: no HIP device\n";
    return 0;
  }
  try {
    const auto prompt = EncodePrompt(target, kPrefix + kExtra + 8);

    // Sparse is read at session creation, so each leg sets it before loading.
    ::unsetenv("FLINT_SPARSE_ATTN");
    ::unsetenv("FLINT_SPARSE_TOPK");
    std::cout << "leg: sparse off\n";
    RunAll(target, prompt);

    ::setenv("FLINT_SPARSE_ATTN", "1", 1);
    ::setenv("FLINT_SPARSE_TOPK", "256", 1);
    std::cout << "leg: sparse on (top-k 256)\n";
    RunAll(target, prompt);
  } catch (const std::exception& error) {
    std::cerr << "flint engine snapshot tests failed: " << error.what() << "\n";
    return 1;
  }
  std::cout << "flint engine snapshot tests passed\n";
  return 0;
}
