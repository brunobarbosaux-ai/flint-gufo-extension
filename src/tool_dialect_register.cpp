#include "src/tool_dialect.hpp"

#include <gufo/ext/text.hpp>

#include "src/chat_template.hpp"

namespace gufo::models::flint {

void RegisterFlintToolDialect() {
  using Syntax = sampling::JsonConstraint::ToolSyntax;
  using Tagged = sampling::JsonConstraint::TaggedArguments;
  Syntax syntax;
  syntax.open = std::string(kToolCallOpen);
  syntax.close = std::string(kToolCallClose);
  // The name runs up to the first tag: a tool whose name holds '<' cannot be
  // constrained.
  syntax.name_delimiters = "<";
  Tagged tagged;
  tagged.parameter_open = std::string(kArgKeyOpen);
  // One tag for raw and typed values: the model writes a string as is and any
  // other value as JSON, and the types come from each tool's schema.
  tagged.raw_value_open =
      std::string(kArgKeyClose) + std::string(kArgValueOpen);
  tagged.typed_value_open = tagged.raw_value_open;
  tagged.value_close = std::string(kArgValueClose);
  syntax.tagged = tagged;
  // Flint reasons before it answers and may open a call inside its reasoning.
  syntax.reasoning_header = true;

  ext::RegisterToolDialect({
      .markers = {kToolCallOpen},
      .reasoning_markers = {kToolCallOpen},
      .closers = {kToolCallClose},
      .scan_tool_calls = &ScanFlintToolCalls,
      .id = std::string(kToolDialectId),
      .syntax = syntax,
  });
}

}  // namespace gufo::models::flint
