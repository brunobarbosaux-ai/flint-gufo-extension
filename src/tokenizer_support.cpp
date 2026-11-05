// Flint-owned definition for the vendored Qwen tokenizer. The vendored
// tokenizer.cpp reports the Ornith template profile through
// QwenChatTemplate::IsOrnithTemplate, but the chat-template translation unit
// is not vendored here. Flint never renders Ornith-profile prompts, so the
// flag only needs the same SHA-256 identity check that Gufo performs.

#include <string_view>

#include "gufo/ext/sha256.hpp"
#include "hip/vendor/qwen/qwen_chat_template.hpp"

namespace gufo::vendor::qwen::tokenization {
namespace {

std::string TemplateSha256(std::string_view value) {
  const auto* begin = reinterpret_cast<const std::uint8_t*>(value.data());
  return crypto::Sha256Hex({begin, value.size()});
}

}  // namespace

bool QwenChatTemplate::IsOrnithTemplate(std::string_view raw_template) {
  if (TemplateSha256(raw_template) == OrnithTemplateSha256())
    return true;
  return raw_template.ends_with("\n") &&
         TemplateSha256(raw_template.substr(0, raw_template.size() - 1)) ==
             OrnithTemplateSha256();
}

}  // namespace gufo::vendor::qwen::tokenization
