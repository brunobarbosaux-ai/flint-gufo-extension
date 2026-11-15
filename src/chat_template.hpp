#ifndef GUFO_MODELS_FLINT_CHAT_TEMPLATE_HPP_
#define GUFO_MODELS_FLINT_CHAT_TEMPLATE_HPP_

#include <string>
#include <string_view>
#include <vector>

namespace gufo::models::flint {

// Flint control tokens as written in the artifact's chat template.
inline constexpr std::string_view kSystemTag = "<|system|>";
inline constexpr std::string_view kUserTag = "<|user|>";
inline constexpr std::string_view kAssistantTag = "<|assistant|>";
inline constexpr std::string_view kThinkOpen = "<think>";
inline constexpr std::string_view kThinkClose = "</think>";

// Native tool-call markup (GLM style): `<tool_call>NAME<arg_key>K</arg_key>
// <arg_value>V</arg_value>...</tool_call>`.
inline constexpr std::string_view kToolCallOpen = "<tool_call>";
inline constexpr std::string_view kToolCallClose = "</tool_call>";
inline constexpr std::string_view kArgKeyOpen = "<arg_key>";
inline constexpr std::string_view kArgKeyClose = "</arg_key>";
inline constexpr std::string_view kArgValueOpen = "<arg_value>";
inline constexpr std::string_view kArgValueClose = "</arg_value>";

struct ToolArgument {
  std::string name;
  // Raw text when `is_string`, JSON text otherwise (as a chat request carries it).
  std::string value;
  bool is_string{true};
};

struct ToolCall {
  std::string name;
  std::vector<ToolArgument> arguments;
};

struct ChatMessage {
  std::string role;  // system | developer | user | assistant | tool
  std::string content;
  std::string reasoning_content;
  // Calls the assistant made in this turn (rendered after its content).
  std::vector<ToolCall> tool_calls;
};

// One declared function. `definition_json` is the complete OpenAI function
// object (it keeps optional fields such as `strict`); when it is empty the
// definition is rebuilt from the three other fields.
struct ChatTool {
  std::string definition_json;
  std::string name;
  std::string description;
  std::string parameters_json{"{}"};
};

struct ChatTemplateOptions {
  bool enable_thinking{false};
};

// Renders what the official Jinja template (tests/golden/chat_template.jinja)
// renders for these messages and tools with add_generation_prompt, byte for
// byte: the `# Tools` system block comes before the first non-system message
// (or at the end when there is none), assistant turns carry their reasoning,
// content and tool calls, and tool results are `<|observation|>` blocks.
[[nodiscard]] std::string RenderPrompt(const std::vector<ChatMessage>& messages,
                                       const std::vector<ChatTool>& tools,
                                       const ChatTemplateOptions& options = {});

// Same without tools.
[[nodiscard]] std::string RenderPrompt(const std::vector<ChatMessage>& messages,
                                       const ChatTemplateOptions& options = {});

// Convenience: system + single user turn.
[[nodiscard]] std::string RenderPrompt(std::string_view system,
                                       std::string_view user,
                                       const ChatTemplateOptions& options = {});

}  // namespace gufo::models::flint

#endif  // GUFO_MODELS_FLINT_CHAT_TEMPLATE_HPP_
