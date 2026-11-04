#include "probe.hpp"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <gufo/ext/gguf_reader.hpp>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace f = gufo::models::flint;
namespace fs = std::filesystem;
namespace {

void Require(bool ok, const char* message) {
  if (!ok)
    throw std::runtime_error(message);
}

// A GGUF v3 container with string metadata only. Enough for the architecture
// probe, which reads general.architecture.
std::vector<std::uint8_t> ArchitectureGguf(const std::string& architecture) {
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
  pod(std::uint32_t(0x46554747));
  pod(std::uint32_t(3));
  pod(std::uint64_t(0));
  pod(std::uint64_t(1));
  text("general.architecture");
  pod(std::uint32_t(8));
  text(architecture);
  while (bytes.size() % 32)
    bytes.push_back(0);
  return bytes;
}

std::unique_ptr<gufo::core::GgufReader> OpenArchitecture(
    const std::string& architecture) {
  const auto bytes = ArchitectureGguf(architecture);
  std::string error;
  auto reader =
      gufo::core::GgufReader::OpenMemory(bytes.data(), bytes.size(), &error);
  Require(bool(reader), "synthetic GGUF did not open");
  return reader;
}

}  // namespace

int main() {
  try {
    // ProbeFlintGguf: positive and negative architecture matches.
    Require(f::ProbeFlintGguf(*OpenArchitecture("flint")),
            "flint architecture not recognized");
    Require(!f::ProbeFlintGguf(*OpenArchitecture("lfm3")),
            "non-flint architecture claimed");
    Require(!f::ProbeFlintGguf(*OpenArchitecture("flint-ish")),
            "near-miss architecture claimed");

    // ProbeFlint on a nonexistent path reports no artifact.
    Require(!f::ProbeFlint("/nonexistent/flint/artifact.gguf"),
            "missing path claimed as flint");

    // ProbeFlint on a directory without a .gguf reports no artifact.
    const fs::path empty =
        fs::temp_directory_path() / "flint_probe_test_empty_dir";
    fs::create_directories(empty);
    {
      std::ofstream(empty / "README.txt") << "no artifact here\n";
    }
    Require(!f::ProbeFlint(empty.string()),
            "directory with no .gguf claimed as flint");
    fs::remove_all(empty);

    std::cout << "flint probe tests passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "flint probe test failed: " << e.what() << '\n';
    return 1;
  }
}
