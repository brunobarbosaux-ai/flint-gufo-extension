// Bit-exact snapshot restore across a wrapped sliding-window ring, against the
// real target model. A 9000-token prefix is longer than the ring of a
// sliding-window layer, so the ring has wrapped twice when the snapshot is
// taken. The snapshot is restored into the session that has since moved on and
// into a fresh one; each must give the logits of the original state, then the
// logits the original run gave for the same continuation, bit for bit.
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

#include "src/gpu/engine.hpp"

namespace flint = gufo::models::flint::gpu;

namespace {

constexpr const char* kDefaultTarget = "/home/bruno/AI/GGUF/flint-UD-Q4_K_XL.gguf";
constexpr std::uint32_t kPrefix = 9000;
constexpr std::uint32_t kBlock = 1200;
constexpr std::uint32_t kSteps = 24;
constexpr std::uint32_t kCapacity = kPrefix + kBlock + kSteps + 16;

void Require(bool condition, const std::string& message) {
  if (!condition)
    throw std::runtime_error(message);
}

std::int32_t Token(std::uint32_t i) {
  return static_cast<std::int32_t>((i * 2654435761U) % 100000U + 1000U);
}

std::vector<float> Copy(std::span<const float> logits) {
  return {logits.begin(), logits.end()};
}

bool Same(const std::vector<float>& a, const std::vector<float>& b) {
  return a.size() == b.size() &&
         std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

struct Tail {
  std::vector<float> after_block;
  std::vector<float> after_steps;
};

/// Appends the block, then one token at a time, recording the logits.
void RunTail(flint::Session& target, const std::vector<std::int32_t>& block,
             Tail* out) {
  std::string error;
  Require(target.Append(block, &error), "tail block: " + error);
  out->after_block = Copy(target.Logits());
  for (std::uint32_t k = 0; k < kSteps; ++k) {
    const std::int32_t next = Token(kPrefix + kBlock + k);
    Require(target.Append({&next, 1}, &error), "tail step: " + error);
  }
  out->after_steps = Copy(target.Logits());
}

}  // namespace

int main() {
  const char* env = std::getenv("FLINT_TARGET");
  const std::string target =
      (env != nullptr && env[0] != '\0') ? env : kDefaultTarget;
  if (!std::filesystem::exists(target)) {
    std::cout << "flint ring snapshot test skipped: no target model at "
              << target << "\n";
    return 0;
  }
  int devices = 0;
  if (hipGetDeviceCount(&devices) != hipSuccess || devices == 0) {
    std::cout << "flint ring snapshot test skipped: no HIP device\n";
    return 0;
  }
  try {
    std::string error;
    auto model = flint::Model::Load(target, {}, &error);
    Require(model != nullptr, "target model: " + error);
    auto session = model->CreateSession(kCapacity, &error);
    Require(session != nullptr, "session: " + error);
    auto fresh = model->CreateSession(kCapacity, &error);
    Require(fresh != nullptr, "fresh session: " + error);

    std::vector<std::int32_t> prefix(kPrefix);
    std::vector<std::int32_t> block(kBlock);
    for (std::uint32_t i = 0; i < kPrefix; ++i)
      prefix[i] = Token(i);
    for (std::uint32_t i = 0; i < kBlock; ++i)
      block[i] = Token(kPrefix + i);

    Require(session->Append(prefix, &error), "prefix: " + error);
    const std::vector<float> at_snapshot = Copy(session->Logits());
    std::vector<std::uint8_t> snapshot(session->SnapshotBytes());
    Require(session->SaveSnapshot(snapshot, &error), "save: " + error);

    Tail original;
    RunTail(*session, block, &original);

    for (const bool use_fresh : {false, true}) {
      const char* name = use_fresh ? "fresh session" : "same session";
      flint::Session& restored_into = use_fresh ? *fresh : *session;
      Require(restored_into.RestoreSnapshot(snapshot, &error),
              std::string(name) + " restore: " + error);
      Require(restored_into.Position() == kPrefix,
              std::string(name) + ": position");
      Require(Same(Copy(restored_into.Logits()), at_snapshot),
              std::string(name) + ": logits at the snapshot");
      Tail restored;
      RunTail(restored_into, block, &restored);
      Require(Same(restored.after_block, original.after_block),
              std::string(name) + ": logits after the block");
      Require(Same(restored.after_steps, original.after_steps),
              std::string(name) + ": logits after the decode steps");
      std::cout << "  " << name << " ok\n";
    }

    // Rewind: a speculative cycle rewinds a few rows, which the ring must
    // serve; a rewind of a whole prefix has lost keys and must fail loudly
    // instead of reading overwritten rows.
    Require(fresh->RestoreSnapshot(snapshot, &error),
            "rewind legs restore: " + error);
    fresh->Rewind(kPrefix - 7);
    Require(fresh->Append({prefix.data() + kPrefix - 7, 7}, &error),
            "append after a 7-row rewind: " + error);
    Require(fresh->Position() == kPrefix, "position after the short rewind");
    fresh->Rewind(0);
    Require(!fresh->Append({prefix.data(), 1}, &error),
            "append after rewinding past the ring must fail");
    Require(fresh->RestoreSnapshot(snapshot, &error),
            "restore clears the broken ring: " + error);
    std::cout << "  rewind ok\n";
    std::cout << "flint ring snapshot tests passed\n";
  } catch (const std::exception& e) {
    std::cerr << "flint ring snapshot test FAILED: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
