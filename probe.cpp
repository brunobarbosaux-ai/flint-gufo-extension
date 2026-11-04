#include "probe.hpp"

#include <filesystem>
#include <optional>
#include <string>

#include "src/runtime/config.hpp"

namespace gufo::models::flint {

bool ProbeFlintGguf(const core::GgufReader& reader) {
  const auto architecture = reader.GetMetadataString("general.architecture");
  return architecture.has_value() && *architecture == "flint";
}

namespace {

// Resolves the single GGUF a probe was pointed at: the path itself when it is a
// file, otherwise the lone *.gguf in the directory. Ambiguity is an error, not
// a silent pick.
std::optional<std::filesystem::path> ResolveArtifact(
    std::string_view model_root) {
  std::error_code ec;
  const std::filesystem::path root(model_root);
  if (std::filesystem::is_regular_file(root, ec)) {
    return root;
  }
  if (!std::filesystem::is_directory(root, ec)) {
    return std::nullopt;
  }
  std::optional<std::filesystem::path> found;
  for (const auto& entry : std::filesystem::directory_iterator(root, ec)) {
    if (!entry.is_regular_file() || entry.path().extension() != ".gguf") {
      continue;
    }
    if (found.has_value()) {
      return std::nullopt;  // More than one artifact: not ours to claim.
    }
    found = entry.path();
  }
  return found;
}

}  // namespace

bool ProbeFlint(std::string_view model_root) {
  const auto artifact = ResolveArtifact(model_root);
  if (!artifact.has_value()) {
    return false;
  }
  std::string error;
  auto reader = core::GgufReader::OpenFile(artifact->string(), &error);
  if (reader == nullptr) {
    return false;
  }
  return ProbeFlintGguf(*reader);
}

}  // namespace gufo::models::flint
