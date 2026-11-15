#include "src/gpu/engine.hpp"

namespace gufo::models::flint::gpu {

struct ModelState {};
struct Session::Impl {};

Session::Session() = default;
Session::~Session() = default;
bool Session::Append(std::span<const std::int32_t>, std::string* error_msg) {
  if (error_msg != nullptr)
    *error_msg = "Flint GPU path not compiled in";
  return false;
}
bool Session::AppendAll(std::span<const std::int32_t>, std::string* error_msg) {
  if (error_msg != nullptr)
    *error_msg = "Flint GPU path not compiled in";
  return false;
}
void Session::Rewind(std::uint32_t) noexcept {}
bool Session::SetHiddenCapture(std::span<const std::uint32_t>,
                               std::string* error_msg) {
  if (error_msg != nullptr)
    *error_msg = "Flint GPU path not compiled in";
  return false;
}
std::span<const std::uint32_t> Session::HiddenCaptureLayers() const noexcept {
  return {};
}
std::uint32_t Session::HiddenRingRows() const noexcept {
  return 0;
}
const float* Session::CapturedHiddenDevice(std::uint32_t,
                                           std::uint32_t) const noexcept {
  return nullptr;
}
bool Session::HasHidden(std::uint32_t) const noexcept { return false; }
std::span<const float> Session::CapturedHidden(std::uint32_t, std::uint32_t) {
  return {};
}
std::span<const float> Session::Logits() const noexcept {
  return {};
}
std::span<const float> Session::Logits(std::uint32_t) const noexcept {
  return {};
}
std::uint32_t Session::Position() const noexcept {
  return 0;
}
std::uint32_t Session::Capacity() const noexcept {
  return 0;
}

Model::~Model() = default;
std::shared_ptr<Model> Model::Load(const std::string&, const EngineOptions&,
                                   std::string* error_msg) {
  if (error_msg != nullptr)
    *error_msg = "Flint GPU path not compiled in";
  return nullptr;
}
std::unique_ptr<Session> Model::CreateSession(std::uint32_t, std::string*) {
  return nullptr;
}
const Config& Model::config() const noexcept {
  static const Config empty;
  return empty;
}
std::size_t Model::ResidentBytes() const noexcept {
  return 0;
}
std::uint64_t Model::FreeDeviceBytes() noexcept {
  return 0;
}

}  // namespace gufo::models::flint::gpu
