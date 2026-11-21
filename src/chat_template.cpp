#include "src/chat_template.hpp"

#include <algorithm>
#include <exception>

#include <gufo/ext/json.hpp>

namespace gufo::models::flint {
namespace {

namespace json = ::gufo::json;

void ReplaceAll(std::string& text, std::string_view from, std::string_view to) {
  if (from.empty())
    return;
  std::size_t at = 0;
  while ((at = text.find(from, at)) != std::string::npos) {
    text.replace(at, from.size(), to);
    at += to.size();
  }
}

// flint_text(): the template's chain of `replace` filters, applied in the same
// order over the whole string (a later replacement sees the result of an
// earlier one): drop NUL and the reasoning tags, then escape the role tags.
std::string FlintText(std::string_view t) {
  std::string out(t);
  out.erase(std::remove(out.begin(), out.end(), '\0'), out.end());
  ReplaceAll(out, kThinkOpen, "");
  ReplaceAll(out, kThinkClose, "");
  static constexpr std::string_view kEscaped[][2] = {
      {"<|begin_of_text|>", "&lt;|begin_of_text|&gt;"},
      {"<|end_of_text|>", "&lt;|end_of_text|&gt;"},
      {"<|system|>", "&lt;|system|&gt;"},
      {"<|developer|>", "&lt;|developer|&gt;"},
      {"<|user|>", "&lt;|user|&gt;"},
      {"<|assistant|>", "&lt;|assistant|&gt;"},
      {"<|observation|>", "&lt;|observation|&gt;"},
  };
  for (const auto& pair : kEscaped)
    ReplaceAll(out, pair[0], pair[1]);
  return out;
}

// Jinja's `tojson` (Python json.dumps, ensure_ascii=False). Text that is
// not valid JSON is written as it came.
std::string ToJson(const std::string& text) {
  try {
    return json::parse(text).tojson();
  } catch (const std::exception&) {
    return text;
  }
}

std::string ToolDefinition(const ChatTool& tool) {
  if (!tool.definition_json.empty())
    return ToJson(tool.definition_json);
  std::string out = "{\"name\": " + json::Value(tool.name).tojson();
  if (!tool.description.empty())
    out += ", \"description\": " + json::Value(tool.description).tojson();
  out += ", \"parameters\": " +
         ToJson(tool.parameters_json.empty() ? "{}" : tool.parameters_json) +
         "}";
  return out;
}

std::string ToolPreamble(const std::vector<ChatTool>& tools) {
  std::string out =
      "<|system|>\n# Tools\n\nYou may call one or more functions to assist "
      "with the user query.\n\nYou are provided with function signatures "
      "within <tools></tools> XML tags:\n<tools>";
  for (const auto& tool : tools)
    out += ToolDefinition(tool);
  out +=
      "</tools>\n\nFor each function call, output the function name and "
      "arguments within the following XML format:\n"
      "<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key><arg_value>"
      "{arg-value-1}</arg_value>...</tool_call>\n";
  return out;
}

void AppendToolCalls(std::string& out, const std::vector<ToolCall>& calls) {
  for (const auto& call : calls) {
    out += kToolCallOpen;
    out += call.name;
    for (const auto& argument : call.arguments) {
      out += kArgKeyOpen;
      out += argument.name;
      out += kArgKeyClose;
      out += kArgValueOpen;
      out += argument.is_string ? argument.value : ToJson(argument.value);
      out += kArgValueClose;
    }
    out += kToolCallClose;
  }
}

}  // namespace

std::string RenderPrompt(const std::vector<ChatMessage>& messages,
                         const std::vector<ChatTool>& tools,
                         const ChatTemplateOptions& options) {
  (void)options;
  const std::string preamble = tools.empty() ? std::string() : ToolPreamble(tools);
  bool preamble_pending = true;
  std::string out;
  for (const auto& m : messages) {
    // The tool block precedes the first message that is not a system message.
    if (preamble_pending && m.role != "system") {
      out += preamble;
      preamble_pending = false;
    }
    if (m.role == "system") {
      out += kSystemTag;
      out += FlintText(m.content);
    } else if (m.role == "developer") {
      out += "<|developer|>";
      out += FlintText(m.content);
    } else if (m.role == "user") {
      out += kUserTag;
      out += FlintText(m.content);
    } else if (m.role == "assistant") {
      out += kAssistantTag;
      // Reasoning and content are written as they are: only the role tags of
      // user-controlled text (system, developer, user, tool) are escaped.
      if (!m.reasoning_content.empty()) {
        out += kThinkOpen;
        out += m.reasoning_content;
        out += kThinkClose;
      }
      out += m.content;
      AppendToolCalls(out, m.tool_calls);
    } else if (m.role == "tool") {
      out += "<|observation|><tool_response>";
      out += FlintText(m.content);
      out += "</tool_response>";
    }
  }
  if (preamble_pending)
    out += preamble;
  // add_generation_prompt: only the assistant tag.
  out += kAssistantTag;
  return out;
}

std::string RenderPrompt(const std::vector<ChatMessage>& messages,
                         const ChatTemplateOptions& options) {
  return RenderPrompt(messages, std::vector<ChatTool>{}, options);
}

std::string RenderPrompt(std::string_view system, std::string_view user,
                         const ChatTemplateOptions& options) {
  std::vector<ChatMessage> messages;
  if (!system.empty()) {
    messages.push_back({"system", std::string(system), "", {}});
  }
  messages.push_back({"user", std::string(user), "", {}});
  return RenderPrompt(messages, options);
}

}  // namespace gufo::models::flint
