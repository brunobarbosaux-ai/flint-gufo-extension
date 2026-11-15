#ifndef GUFO_MODELS_FLINT_TOOL_DIALECT_HPP_
#define GUFO_MODELS_FLINT_TOOL_DIALECT_HPP_

#include <cstddef>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <gufo/ext/text.hpp>

namespace gufo::models::flint {

// Identity the runner returns from ToolDialectId() and the dialect registers
// under.
inline constexpr std::string_view kToolDialectId = "flint-glm";

// Registers the Flint tool dialect (grammar, markers, scanner) with the host.
// Call once during extension registration, before any runner is created.
void RegisterFlintToolDialect();

// Reads the tool calls Flint wrote:
//   <tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value>...</tool_call>
// See ext::ToolDialect for the contract (spans, `partial`, `unfinished`).
//
// The markup carries no value types, so an argument is a string or JSON text
// according to the declared parameter schema remembered by
// RememberToolSchemas(); an unknown tool or parameter is read as a string.
void ScanFlintToolCalls(
    std::string_view text, std::span<const std::string_view> markers,
    std::vector<ext::ScannedToolCall>* calls,
    std::vector<std::pair<std::size_t, std::size_t>>* spans, bool partial);

// Remembers, by tool name, which parameters of each declared tool are strings.
// `parameters_json` is the tool's JSON schema. The scanner cannot see the
// request, so the runner records the tools of every prompt it renders; a later
// declaration of the same name replaces the earlier one. Thread-safe.
void RememberToolSchemas(
    const std::vector<std::pair<std::string, std::string>>& name_and_schema);

// Forgets every remembered tool (tests).
void ForgetToolSchemas();

}  // namespace gufo::models::flint

#endif  // GUFO_MODELS_FLINT_TOOL_DIALECT_HPP_
