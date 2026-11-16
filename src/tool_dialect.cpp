#include "src/tool_dialect.hpp"

#include <exception>
#include <mutex>
#include <unordered_map>

#include <gufo/ext/json.hpp>

#include "src/chat_template.hpp"

namespace gufo::models::flint {
namespace {

namespace json = ::gufo::json;

// --- Declared parameter types ------------------------------------------------

// For each remembered tool, which parameters are strings. A parameter that is
// not listed (unknown tool, unknown parameter) is read as a string.
using ParameterKinds = std::unordered_map<std::string, bool>;

std::mutex& SchemaMutex() {
  static std::mutex mutex;
  return mutex;
}

std::unordered_map<std::string, ParameterKinds>& Schemas() {
  static std::unordered_map<std::string, ParameterKinds> schemas;
  return schemas;
}

// Whether a property schema admits a string: `type` string, or a list holding
// it, or one `anyOf` / `oneOf` branch that does. No type information (including
// a `$ref`) reads as a string, which is how the model writes most values.
bool AdmitsString(const json::Value& property) {
  if (!property.is_object())
    return true;
  if (const json::Value* type = property.find("type")) {
    if (type->is_string())
      return type->str() == "string";
    if (type->is_array()) {
      for (const auto& item : type->items()) {
        if (item.is_string() && item.str() == "string")
          return true;
      }
      return false;
    }
  }
  for (const char* key : {"anyOf", "oneOf"}) {
    const json::Value* branches = property.find(key);
    if (branches == nullptr || !branches->is_array())
      continue;
    for (const auto& branch : branches->items()) {
      if (AdmitsString(branch))
        return true;
    }
    return false;
  }
  return true;
}

ParameterKinds KindsOf(const std::string& schema_json) {
  ParameterKinds kinds;
  try {
    const json::Value schema = json::parse(schema_json);
    const json::Value* properties = schema.find("properties");
    if (properties != nullptr && properties->is_object()) {
      for (const auto& [name, property] : properties->members())
        kinds[name] = AdmitsString(property);
    }
  } catch (const std::exception&) {
    // A schema that does not parse declares nothing.
  }
  return kinds;
}

bool IsStringParameter(const std::string& tool, const std::string& parameter) {
  const std::lock_guard<std::mutex> lock(SchemaMutex());
  const auto found = Schemas().find(tool);
  if (found == Schemas().end())
    return true;
  const auto kind = found->second.find(parameter);
  return kind == found->second.end() ? true : kind->second;
}

// --- Scanner ------------------------------------------------------------------

bool IsSpace(char c) {
  return c == ' ' || c == '\t' || c == '\r' || c == '\n';
}

std::string_view Trim(std::string_view text) {
  while (!text.empty() && IsSpace(text.front()))
    text.remove_prefix(1);
  while (!text.empty() && IsSpace(text.back()))
    text.remove_suffix(1);
  return text;
}

std::size_t SkipSpaces(std::string_view text, std::size_t at) {
  while (at < text.size() && IsSpace(text[at]))
    ++at;
  return at;
}

bool StartsAt(std::string_view text, std::size_t at, std::string_view tag) {
  return text.substr(at).starts_with(tag);
}

// The text from `at` to the end is a proper prefix of `tag`: the tag may still
// complete.
bool IsPartialTagAt(std::string_view text, std::size_t at,
                    std::string_view tag) {
  const std::string_view rest = text.substr(at);
  return rest.size() < tag.size() && tag.starts_with(rest);
}

// Length of the longest suffix of `text` that is a proper prefix of `tag`.
std::size_t PartialTagSuffix(std::string_view text, std::string_view tag) {
  const std::size_t longest = std::min(text.size(), tag.size() - 1);
  for (std::size_t length = longest; length > 0; --length) {
    if (text.substr(text.size() - length) == tag.substr(0, length))
      return length;
  }
  return 0;
}

ext::ScannedToolArgument MakeArgument(const std::string& tool,
                                      const std::string& key,
                                      std::string_view raw) {
  if (IsStringParameter(tool, key))
    return {key, std::string(raw), true};
  return {key, std::string(Trim(raw)), false};
}

enum class CallEnd { kComplete, kIncomplete, kMalformed };

// Reads one call whose opening marker ends at `at`. On kComplete `*end` is the
// offset after the closing tag. On kIncomplete the text ended inside the call
// and `call` holds what was read so far.
CallEnd ReadCall(std::string_view text, std::size_t at,
                 ext::ScannedToolCall* call, std::size_t* end, bool partial,
                 bool* name_known) {
  *name_known = false;
  const std::size_t name_end = text.find('<', at);
  if (name_end == std::string_view::npos)
    return CallEnd::kIncomplete;  // the name may still grow
  const std::string_view name = Trim(text.substr(at, name_end - at));
  if (name.empty() || name.find_first_of(" \t\r\n") != std::string_view::npos)
    return CallEnd::kMalformed;
  call->name = std::string(name);
  *name_known = true;

  std::size_t q = name_end;
  while (true) {
    q = SkipSpaces(text, q);
    if (q >= text.size())
      return CallEnd::kIncomplete;
    if (StartsAt(text, q, kToolCallClose)) {
      *end = q + kToolCallClose.size();
      return CallEnd::kComplete;
    }
    if (StartsAt(text, q, kArgKeyOpen)) {
      const std::size_t key_begin = q + kArgKeyOpen.size();
      const std::size_t key_end = text.find(kArgKeyClose, key_begin);
      if (key_end == std::string_view::npos)
        return CallEnd::kIncomplete;
      const std::string key(Trim(text.substr(key_begin, key_end - key_begin)));
      if (key.empty())
        return CallEnd::kMalformed;
      std::size_t v = SkipSpaces(text, key_end + kArgKeyClose.size());
      if (v >= text.size() || IsPartialTagAt(text, v, kArgValueOpen))
        return CallEnd::kIncomplete;
      if (!StartsAt(text, v, kArgValueOpen))
        return CallEnd::kMalformed;
      const std::size_t value_begin = v + kArgValueOpen.size();
      const std::size_t value_end = text.find(kArgValueClose, value_begin);
      if (value_end == std::string_view::npos) {
        // A string value still being written: its text so far, less a tail
        // that may be the start of the closing tag, only grows.
        if (partial && IsStringParameter(call->name, key)) {
          const std::string_view so_far = text.substr(value_begin);
          call->unfinished = ext::ScannedToolArgument{
              key,
              std::string(so_far.substr(
                  0, so_far.size() - PartialTagSuffix(so_far, kArgValueClose))),
              true};
        }
        return CallEnd::kIncomplete;
      }
      call->arguments.push_back(MakeArgument(
          call->name, key, text.substr(value_begin, value_end - value_begin)));
      q = value_end + kArgValueClose.size();
      continue;
    }
    if (IsPartialTagAt(text, q, kToolCallClose) ||
        IsPartialTagAt(text, q, kArgKeyOpen))
      return CallEnd::kIncomplete;
    return CallEnd::kMalformed;
  }
}

}  // namespace

void RememberToolSchemas(
    const std::vector<std::pair<std::string, std::string>>& name_and_schema) {
  std::unordered_map<std::string, ParameterKinds> parsed;
  for (const auto& [name, schema] : name_and_schema)
    parsed[name] = KindsOf(schema);
  const std::lock_guard<std::mutex> lock(SchemaMutex());
  for (auto& [name, kinds] : parsed)
    Schemas()[name] = std::move(kinds);
}

void ForgetToolSchemas() {
  const std::lock_guard<std::mutex> lock(SchemaMutex());
  Schemas().clear();
}

void ScanFlintToolCalls(
    std::string_view text, std::span<const std::string_view> /*markers*/,
    std::vector<ext::ScannedToolCall>* calls,
    std::vector<std::pair<std::size_t, std::size_t>>* spans, bool partial) {
  std::size_t position = 0;
  while (true) {
    const std::size_t start = text.find(kToolCallOpen, position);
    if (start == std::string_view::npos)
      return;
    const std::size_t body = start + kToolCallOpen.size();
    ext::ScannedToolCall call;
    std::size_t end = 0;
    bool name_known = false;
    switch (ReadCall(text, body, &call, &end, partial, &name_known)) {
      case CallEnd::kComplete:
        call.valid = true;
        call.complete = true;
        calls->push_back(std::move(call));
        if (spans != nullptr)
          spans->emplace_back(start, end);
        position = end;
        break;
      case CallEnd::kIncomplete:
        // Only the final call can be unfinished, and only once its name is
        // known; a partial scan appends it, the final scan does not.
        if (partial && name_known) {
          call.valid = true;
          call.complete = false;
          calls->push_back(std::move(call));
        }
        return;
      case CallEnd::kMalformed:
        // Not a call: its markup stays text, and a later call is still read.
        position = body;
        break;
    }
  }
}

}  // namespace gufo::models::flint
