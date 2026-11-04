#include "src/runtime/config.hpp"

#include <cstring>
#include <gufo/ext/gguf_reader.hpp>
#include <iostream>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include "src/chat_template.hpp"

namespace f = gufo::models::flint;
namespace {

using Value =
    std::variant<std::uint32_t, float, bool, std::string,
                 std::vector<std::uint64_t>, std::vector<std::string>>;
using Fields = std::map<std::string, Value>;

void Require(bool ok, const char* message) {
  if (!ok)
    throw std::runtime_error(message);
}

// Serializes a minimal GGUF v3 container with the given metadata. Mirrors the
// layout gufo's reader accepts; tensors are omitted.
std::vector<std::uint8_t> BuildGguf(const Fields& fields) {
  std::vector<std::uint8_t> bytes;
  auto pod = [&]<class T>(T value) {
    const auto at = bytes.size();
    bytes.resize(at + sizeof(T));
    std::memcpy(bytes.data() + at, &value, sizeof(T));
  };
  auto text = [&](const std::string& value) {
    pod(std::uint64_t(value.size()));
    bytes.insert(bytes.end(), value.begin(), value.end());
  };
  pod(std::uint32_t(0x46554747));  // "GGUF"
  pod(std::uint32_t(3));
  pod(std::uint64_t(0));
  pod(std::uint64_t(fields.size()));
  for (const auto& [key, value] : fields) {
    text(key);
    std::visit(
        [&](const auto& v) {
          using T = std::decay_t<decltype(v)>;
          if constexpr (std::is_same_v<T, std::uint32_t>) {
            pod(std::uint32_t(4));
            pod(v);
          } else if constexpr (std::is_same_v<T, float>) {
            pod(std::uint32_t(6));
            pod(v);
          } else if constexpr (std::is_same_v<T, bool>) {
            pod(std::uint32_t(7));
            pod(std::uint8_t(v));
          } else if constexpr (std::is_same_v<T, std::string>) {
            pod(std::uint32_t(8));
            text(v);
          } else {
            pod(std::uint32_t(9));
            pod(std::uint32_t(
                std::is_same_v<T, std::vector<std::string>> ? 8 : 10));
            pod(std::uint64_t(v.size()));
            for (const auto& x : v) {
              if constexpr (std::is_same_v<T, std::vector<std::string>>)
                text(x);
              else
                pod(x);
            }
          }
        },
        value);
  }
  while (bytes.size() % 32)
    bytes.push_back(0);
  return bytes;
}

std::optional<f::Config> Parse(const Fields& fields, std::string* error) {
  const auto bytes = BuildGguf(fields);
  auto reader =
      gufo::core::GgufReader::OpenMemory(bytes.data(), bytes.size(), error);
  Require(bool(reader), "synthetic GGUF did not open");
  return f::ParseConfig(*reader, error);
}

// A valid-architecture artifact that is missing the required geometry keys.
Fields Incomplete() {
  return {{"general.architecture", std::string("flint")}};
}

}  // namespace

int main() {
  try {
    // Wrong architecture is rejected with a diagnostic, never parsed.
    {
      std::string error;
      auto bad = Parse({{"general.architecture", std::string("lfm3")}}, &error);
      Require(!bad, "non-flint architecture accepted");
      Require(error.find("not flint") != std::string::npos,
              "wrong-architecture error lacks the reason");
    }

    // Missing geometry keys fail closed.
    {
      std::string error;
      Require(!Parse(Incomplete(), &error), "incomplete metadata accepted");
      Require(!error.empty(), "incomplete metadata gave no diagnostic");
    }

    // The chat template renders the system/user framing and the generation
    // prompt. This pins the public prompt shape Flint sends.
    {
      const auto prompt = f::RenderPrompt("Be concise.", "Hello");
      Require(prompt == "<|system|>Be concise.<|user|>Hello<|assistant|>",
              "rendered prompt framing differs from the Flint template");
    }

    std::cout << "flint config tests passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "flint config test failed: " << e.what() << '\n';
    return 1;
  }
}
