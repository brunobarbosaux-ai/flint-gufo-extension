// Session row access against the real target model (Stage 1 of the KV-rows
// plan). Checks, against a fresh reference session with the same prefix:
//
//   1. SaveSnapshot / RestoreSnapshot round trip: position, logits bits and
//      the next append are bit-identical to the reference.
//   2. Split path: copy only the K/V rows out per layer (KvRows + hipMemcpy),
//      then RestoreRows + SetLogits. Same three assertions.
//   3. RestoreRows(0): position 0, all-zero logits, and the next append is
//      bit-identical to a fresh session.
//   4. SetRowWriteHook: one call per Append/AppendAll covering exactly the
//      rows written, none from Rewind; one call for a multi-chunk append; and
//      the row is still unwritten when the hook runs.
//
// Two legs share one loaded model:
//   - sparse off: context 512, prefix 64 (fast).
//   - sparse on:  FLINT_SPARSE_ATTN=1, FLINT_SPARSE_TOPK=1, context 4096,
//     prefix 2560. Sparse is read per session at creation, so the env is set
//     before each leg's sessions are made. Sparse attention engages on a
//     chunk only when chunk_start + n > (sink + local + top + 1) * 64 =
//     2304 tokens. The sparse leg asserts its prefix exceeds that threshold,
//     so every append after the prefix engages it too, and a wrong key-bound
//     rebuild changes the logits.
//
// The target is FLINT_TARGET, or the Q4_K_XL artifact the other tools default
// to. If the file or a HIP device is absent the test skips with exit 0.

#include <hip/hip_runtime.h>

#include <algorithm>
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
constexpr std::uint32_t kExtra = 16;
// Tokens compared after a restore: one wide append (more than the 8-row vector
// batch, so the sparse kernel runs) then single-token decode steps. The
// snapshot leg needs prefix + kExtra + kContinue prompt tokens.
constexpr std::uint32_t kWideAppend = 16;
constexpr std::uint32_t kContinue = kWideAppend + 4;
// Sparse attention threshold for FLINT_SPARSE_TOPK=1: (sink 2 + local 32 +
// top 1 + 1) * 64. Mirrors kSparseSinkBlocks/kSparseLocalBlocks in engine.cpp.
constexpr std::uint32_t kSparseTop = 1;
constexpr std::uint32_t kSparseEngageTokens = (2 + 32 + kSparseTop + 1) * 64;
// Multi-chunk append: 5000 tokens cross the 4096-token prefill tile.
constexpr std::uint32_t kLongContext = 8192;
constexpr std::uint32_t kLongAppend = 5000;
constexpr std::uint32_t kPromptTokens = kLongAppend + 8;

struct Params {
  std::uint32_t context;
  std::uint32_t prefix;
};

constexpr Params kDenseParams{512, 64};
constexpr Params kSparseParams{4096, 2560};
// The prompt must cover the longest per-leg use: prefix + extra + decode steps.
static_assert(kPromptTokens >= kSparseParams.prefix + kExtra + kContinue);
static_assert(kPromptTokens >= kDenseParams.prefix + kExtra + kContinue);

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

std::unique_ptr<flint::Session> Make(flint::Model& model, std::uint32_t context) {
  std::string error;
  auto session = model.CreateSession(context, &error);
  RequireOk(session != nullptr, error, "session");
  return session;
}

/// Asserts the restored `session` matches `reference` (both at `prefix`):
/// logits are bit-identical at the start, after one kWideAppend-token append
/// (n > 8, so the sparse kernel reads the restored key bounds), and after
/// each of `tokens` appended one at a time. Single-token decode takes the
/// vector path and never reads key bounds, so the wide append is the check
/// that matters for sparse.
void CheckContinuation(flint::Session& session, flint::Session& reference,
                       std::uint32_t prefix, std::span<const std::int32_t> tokens,
                       const char* what) {
  std::string error;
  Require(session.Position() == prefix, std::string(what) + ": position");
  Require(SameBits(session.Logits(), reference.Logits()),
          std::string(what) + ": logits bit-identical");
  const auto wide = tokens.subspan(0, kWideAppend);
  RequireOk(session.Append(wide, &error), error, what);
  RequireOk(reference.Append(wide, &error), error, what);
  Require(SameBits(session.Logits(), reference.Logits()),
          std::string(what) + ": wide append bit-identical");
  for (std::size_t step = kWideAppend; step < tokens.size(); ++step) {
    const std::span<const std::int32_t> token = tokens.subspan(step, 1);
    RequireOk(session.Append(token, &error), error, what);
    RequireOk(reference.Append(token, &error), error, what);
    Require(session.Position() == reference.Position(),
            std::string(what) + ": position after append");
    Require(SameBits(session.Logits(), reference.Logits()),
            std::string(what) + ": decode step " + std::to_string(step) +
                " bit-identical");
  }
}

void TestSnapshotRoundTrip(flint::Model& model, const Params& p,
                           const std::vector<std::int32_t>& prompt) {
  std::string error;
  auto session = Make(model, p.context);
  auto reference = Make(model, p.context);
  const std::span<const std::int32_t> prefix(prompt.data(), p.prefix);
  const std::span<const std::int32_t> extra(prompt.data() + p.prefix, kExtra);
  RequireOk(session->Append(prefix, &error), error, "prefix");
  RequireOk(reference->Append(prefix, &error), error, "reference prefix");

  std::vector<std::uint8_t> payload(session->SnapshotBytes());
  RequireOk(session->SaveSnapshot(payload, &error), error, "save");
  RequireOk(session->Append(extra, &error), error, "extra");
  Require(session->Position() == p.prefix + kExtra, "extra position");

  // Restore into a session holding other rows, so a stale key-bound rebuild
  // shows up (restoring over the saving session would leave matching bounds).
  auto restored = Make(model, p.context);
  RequireOk(restored->Append(std::span<const std::int32_t>(prompt.data(), 8),
                             &error),
            error, "dirty snapshot target");
  RequireOk(restored->RestoreSnapshot(payload, &error), error, "restore");
  CheckContinuation(*restored, *reference, p.prefix,
                    std::span<const std::int32_t>(
                        prompt.data() + p.prefix + kExtra, kContinue),
                    "snapshot");
  std::cout << "  snapshot round trip ok\n";
}

void TestSplitPath(flint::Model& model, const Params& p,
                   const std::vector<std::int32_t>& prompt) {
  std::string error;
  auto session = Make(model, p.context);
  auto reference = Make(model, p.context);
  const std::span<const std::int32_t> prefix(prompt.data(), p.prefix);
  RequireOk(session->Append(prefix, &error), error, "prefix");
  RequireOk(reference->Append(prefix, &error), error, "reference prefix");

  const std::uint32_t layers = session->NumLayers();
  const std::size_t row = session->KvRowBytes();
  const std::size_t bytes = std::size_t{p.prefix} * row;
  Require(layers > 0 && row > 0, "layer count and row size");
  Require(session->KvRows(layers, false) == nullptr,
          "out-of-range layer returns nullptr");
  Require(session->KvRows(layers, true) == nullptr,
          "out-of-range layer returns nullptr (value)");

  // Capture only the valid rows of each layer, K then V. The D2H copies are
  // blocking and Append has already synchronized the stream.
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
  const auto logits = session->Logits();
  const std::vector<float> logit_row(logits.begin(), logits.end());

  // Restore into a session that already holds other rows, so the copies must
  // overwrite them: rows first, then logits, then the position.
  auto restored = Make(model, p.context);
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
  std::vector<float> long_row(logit_row);
  long_row.push_back(0.0f);
  Require(!restored->SetLogits(long_row, &error),
          "SetLogits rejects a long row (vocab + 1)");
  RequireOk(restored->SetLogits(logit_row, &error), error, "set logits");
  RequireOk(restored->RestoreRows(p.prefix, &error), error, "restore rows");
  Require(!restored->RestoreRows(p.context + 1, &error),
          "RestoreRows rejects position > capacity");
  RequireOk(restored->RestoreRows(p.prefix, &error), error, "restore rows again");
  CheckContinuation(*restored, *reference, p.prefix,
                    std::span<const std::int32_t>(prompt.data() + p.prefix,
                                                  kContinue),
                    "split");
  std::cout << "  split row path ok\n";
}

void TestRestoreZero(flint::Model& model, const Params& p,
                     const std::vector<std::int32_t>& prompt) {
  std::string error;
  auto session = Make(model, p.context);
  const std::span<const std::int32_t> prefix(prompt.data(), p.prefix);
  RequireOk(session->Append(prefix, &error), error, "prefix");

  RequireOk(session->RestoreRows(0, &error), error, "restore rows to zero");
  Require(session->Position() == 0, "RestoreRows(0) position is 0");
  const auto logits = session->Logits();
  Require(std::all_of(logits.begin(), logits.end(),
                      [](float v) { return v == 0.0f; }),
          "RestoreRows(0) logits are all zero");

  // The next append must match a fresh session appending the same token at 0.
  auto fresh = Make(model, p.context);
  const std::int32_t token[1] = {prompt[0]};
  RequireOk(session->Append(token, &error), error, "append after restore to 0");
  RequireOk(fresh->Append(token, &error), error, "fresh append");
  Require(session->Position() == fresh->Position(),
          "RestoreRows(0) position after append");
  Require(SameBits(session->Logits(), fresh->Logits()),
          "RestoreRows(0) then append is bit-identical to a fresh session");
  std::cout << "  restore to zero ok\n";
}

void TestRowHook(flint::Model& model, const Params& p,
                 const std::vector<std::int32_t>& prompt) {
  std::string error;
  auto session = Make(model, p.context);
  std::vector<std::pair<std::uint32_t, std::uint32_t>> calls;
  session->SetRowWriteHook([&calls](std::uint32_t first, std::uint32_t end) {
    calls.emplace_back(first, end);
  });

  const std::span<const std::int32_t> prefix(prompt.data(), p.prefix);
  RequireOk(session->Append(prefix, &error), error, "prefix");
  Require(calls.size() == 1 && calls[0].first == 0 &&
              calls[0].second == p.prefix,
          "Append reports one call covering the prefix");
  calls.clear();

  const std::int32_t abc[3] = {prompt[0], prompt[1], prompt[2]};
  RequireOk(session->Append(abc, &error), error, "append 3");
  Require(calls.size() == 1 && calls[0].first == p.prefix &&
              calls[0].second == p.prefix + 3,
          "Append(3) reports one call [position, position+3)");
  calls.clear();

  session->Rewind(p.prefix);
  Require(calls.empty(), "Rewind reports nothing");

  RequireOk(session->AppendAll(abc, &error), error, "appendall 3");
  Require(calls.size() == 1 && calls[0].first == p.prefix &&
              calls[0].second == p.prefix + 3,
          "AppendAll(3) reports one call of width 3");
  calls.clear();

  session->SetRowWriteHook({});
  RequireOk(session->Append(abc, &error), error, "append after clear");
  Require(calls.empty(), "cleared hook is not called");
  std::cout << "  row write hook ok\n";
}

void TestRowHookMultiChunk(flint::Model& model,
                           const std::vector<std::int32_t>& prompt) {
  std::string error;
  auto session = Make(model, kLongContext);
  std::vector<std::pair<std::uint32_t, std::uint32_t>> calls;
  session->SetRowWriteHook([&calls](std::uint32_t first, std::uint32_t end) {
    calls.emplace_back(first, end);
  });
  RequireOk(session->Append(
                std::span<const std::int32_t>(prompt.data(), kLongAppend), &error),
            error, "long append");
  Require(calls.size() == 1 && calls[0].first == 0 &&
              calls[0].second == kLongAppend,
          "multi-chunk Append reports exactly one call [0, 5000)");
  Require(session->Position() == kLongAppend, "multi-chunk position");
  std::cout << "  multi-chunk hook ok\n";
}

void TestRowHookOrdering(flint::Model& model, const Params& p,
                         const std::vector<std::int32_t>& prompt) {
  constexpr std::uint32_t kAt = 64;
  std::string error;
  auto session = Make(model, p.context);
  RequireOk(session->Append(std::span<const std::int32_t>(prompt.data(), kAt),
                            &error),
            error, "prefix at 64");
  Require(session->Position() == kAt, "ordering session at position 64");

  // Row kAt of layer 0 K, before any write by the next append.
  const std::size_t row = session->KvRowBytes();
  const auto* row_ptr = static_cast<const std::uint8_t*>(
                            session->KvRows(0, false)) + kAt * row;
  std::vector<std::uint8_t> captured(row);
  CopyOrThrow(captured.data(), row_ptr, row, hipMemcpyDeviceToHost, "capture row");

  bool fired = false;
  bool unwritten = false;
  std::vector<std::uint8_t> seen(row);
  session->SetRowWriteHook([&](std::uint32_t first, std::uint32_t end) {
    if (first != kAt || end != kAt + 1)
      return;
    fired = true;
    CopyOrThrow(seen.data(), row_ptr, row, hipMemcpyDeviceToHost, "hook read");
    unwritten = std::memcmp(seen.data(), captured.data(), row) == 0;
  });
  const std::int32_t token[1] = {prompt[kAt]};
  RequireOk(session->Append(token, &error), error, "append at 64");
  Require(fired, "row hook fired for the append at 64");
  Require(unwritten, "hook fired after the row was written");
  std::cout << "  row hook ordering ok\n";
}

void RunLeg(flint::Model& model, const Params& p,
            const std::vector<std::int32_t>& prompt) {
  TestSnapshotRoundTrip(model, p, prompt);
  TestSplitPath(model, p, prompt);
  TestRestoreZero(model, p, prompt);
  TestRowHook(model, p, prompt);
  TestRowHookOrdering(model, p, prompt);
  TestRowHookMultiChunk(model, prompt);
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
    Require(kSparseParams.prefix > kSparseEngageTokens,
            "sparse leg prefix must exceed the sparse engage threshold");
    Require(kSparseParams.context > kSparseParams.prefix + kExtra,
            "sparse leg context must hold the prefix and extra tokens");
    const auto prompt = EncodePrompt(target, kPromptTokens);

    std::string error;
    auto model = flint::Model::Load(target, {}, &error);
    RequireOk(model != nullptr, error, "target model");

    // Sparse is read at session creation, so set the env before each leg.
    ::unsetenv("FLINT_SPARSE_ATTN");
    ::unsetenv("FLINT_SPARSE_TOPK");
    std::cout << "leg: sparse off (context " << kDenseParams.context
              << ", prefix " << kDenseParams.prefix << ")\n";
    RunLeg(*model, kDenseParams, prompt);

    ::setenv("FLINT_SPARSE_ATTN", "1", 1);
    ::setenv("FLINT_SPARSE_TOPK", "1", 1);
    std::cout << "leg: sparse on (top-k " << kSparseTop << ", engages past "
              << kSparseEngageTokens << " tokens; context "
              << kSparseParams.context << ", prefix " << kSparseParams.prefix
              << ")\n";
    RunLeg(*model, kSparseParams, prompt);
  } catch (const std::exception& error) {
    std::cerr << "flint engine snapshot tests failed: " << error.what() << "\n";
    return 1;
  }
  std::cout << "flint engine snapshot tests passed\n";
  return 0;
}
