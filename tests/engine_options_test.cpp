#include "src/gpu/engine.hpp"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

namespace f = gufo::models::flint::gpu;
namespace {

void Require(bool ok, const char* message) {
  if (!ok)
    throw std::runtime_error(message);
}

bool Accepts(std::uint32_t prefill_tile, std::string* error) {
  return f::ValidateEngineOptions(f::EngineOptions{prefill_tile}, error);
}

}  // namespace

int main() {
  try {
    // Defaults and even tiles of at least 2 are accepted.
    for (std::uint32_t tile : {4096u, 4158u, 4160u, 2u}) {
      std::string error;
      Require(Accepts(tile, &error), "even prefill_tile rejected");
      Require(error.empty(), "accepted prefill_tile set an error");
    }

    // Zero, one and odd tiles are rejected with a diagnostic.
    for (std::uint32_t tile : {0u, 1u, 4157u, 4159u}) {
      std::string error;
      Require(!Accepts(tile, &error), "invalid prefill_tile accepted");
      Require(!error.empty(), "invalid prefill_tile gave no diagnostic");
    }

    std::cout << "flint engine options tests passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "flint engine options test failed: " << e.what() << '\n';
    return 1;
  }
}
