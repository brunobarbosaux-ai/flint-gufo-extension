// G2 probe: render the packed flint fixtures with strycast's Flint template
// and compare the rendered bytes to the oracle's /apply-template output, then
// tokenize and compare ids to the oracle's /tokenize output.
//
// Cases file layout: see tools/flint/g2_cases.py.
//
// Usage: template_probe MODEL.gguf CASES.bin [MAX_REPORTS]
//   exits 0 when every supported fixture matches, 1 on mismatch.
//
// Coverage note: strycast's flint template
// (`src/models/flint/chat_template.cpp`) models system/developer/user/
// assistant/tool turns, reasoning content and control-token escaping, but it
// does NOT implement the tool preamble (`<tools>…` block, `<tool_call>`
// argument rendering) or the trailing-assistant generation-prompt
// suppression. Fixtures that depend on those are reported as SKIPPED with the
// reason rather than counted as passes.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/flint/chat_template.hpp"
#include "src/models/qwen/tokenizer.hpp"

namespace {

bool ReadPod(std::ifstream& in, void* out, std::size_t bytes) {
  in.read(static_cast<char*>(out), static_cast<std::streamsize>(bytes));
  return in.gcount() == static_cast<std::streamsize>(bytes);
}

bool ReadString(std::ifstream& in, std::string* out) {
  std::uint32_t size = 0;
  if (!ReadPod(in, &size, sizeof(size))) {
    return false;
  }
  out->assign(size, '\0');
  return size == 0 || ReadPod(in, &(*out)[0], size);
}

std::string Readable(std::string_view text, std::size_t limit = 120) {
  std::string out;
  for (const char c : text) {
    if (out.size() >= limit) {
      out += "...";
      break;
    }
    if (c == '\n') {
      out += "\\n";
    } else if (c == '\r') {
      out += "\\r";
    } else {
      out += c;
    }
  }
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3 || argc > 4) {
    std::fprintf(stderr, "usage: %s MODEL.gguf CASES.bin [MAX_REPORTS]\n",
                 argv[0]);
    return 2;
  }
  const int max_reports = argc == 4 ? std::atoi(argv[3]) : 10;

  std::string error;
  const auto reader = gufo::core::GgufReader::OpenFile(argv[1], &error);
  if (reader == nullptr) {
    std::fprintf(stderr, "open failed: %s\n", error.c_str());
    return 2;
  }
  const auto tokenizer =
      gufo::tokenization::QwenTokenizer::CreateFromGguf(*reader, &error);
  if (tokenizer == nullptr) {
    std::fprintf(stderr, "tokenizer load failed: %s\n", error.c_str());
    return 2;
  }

  std::ifstream in(argv[2], std::ios::binary);
  if (!in) {
    std::fprintf(stderr, "cannot open cases: %s\n", argv[2]);
    return 2;
  }
  std::uint32_t fixture_count = 0;
  if (!ReadPod(in, &fixture_count, sizeof(fixture_count))) {
    std::fprintf(stderr, "truncated cases header\n");
    return 2;
  }

  std::size_t render_failures = 0;
  std::size_t token_failures = 0;
  std::size_t skipped = 0;
  std::size_t compared = 0;
  std::size_t reported = 0;
  for (std::uint32_t index = 0; index < fixture_count; ++index) {
    std::string name;
    std::string expected_rendered;
    std::vector<std::uint32_t> expected_ids;
    if (!ReadString(in, &name) || !ReadString(in, &expected_rendered)) {
      std::fprintf(stderr, "truncated fixture %u\n", index);
      return 2;
    }
    std::uint32_t id_count = 0;
    if (!ReadPod(in, &id_count, sizeof(id_count))) {
      return 2;
    }
    expected_ids.resize(id_count);
    if (id_count != 0 &&
        !ReadPod(in, &expected_ids[0], id_count * sizeof(std::uint32_t))) {
      return 2;
    }

    // Tools (parsed for layout parity; strycast has no tool preamble).
    std::uint32_t tool_count = 0;
    if (!ReadPod(in, &tool_count, sizeof(tool_count))) {
      return 2;
    }
    for (std::uint32_t t = 0; t < tool_count; ++t) {
      std::string ignore;
      if (!ReadString(in, &ignore) || !ReadString(in, &ignore) ||
          !ReadString(in, &ignore)) {
        return 2;
      }
    }

    std::uint32_t message_count = 0;
    if (!ReadPod(in, &message_count, sizeof(message_count))) {
      return 2;
    }
    std::vector<gufo::models::flint::ChatMessage> messages;
    bool has_tool_calls = false;
    for (std::uint32_t m = 0; m < message_count; ++m) {
      std::uint8_t role = 0;
      std::string content;
      std::string thought;
      std::uint32_t call_count = 0;
      if (!ReadPod(in, &role, sizeof(role))) {
        return 2;
      }
      if (!ReadString(in, &content) || !ReadString(in, &thought) ||
          !ReadPod(in, &call_count, sizeof(call_count))) {
        return 2;
      }
      for (std::uint32_t c = 0; c < call_count; ++c) {
        std::uint32_t arg_count = 0;
        std::string ignore;
        if (!ReadString(in, &ignore) ||
            !ReadPod(in, &arg_count, sizeof(arg_count))) {
          return 2;
        }
        for (std::uint32_t a = 0; a < arg_count; ++a) {
          std::uint8_t is_string = 1;
          if (!ReadString(in, &ignore) ||
              !ReadPod(in, &is_string, sizeof(is_string)) ||
              !ReadString(in, &ignore)) {
            return 2;
          }
        }
        has_tool_calls = true;
      }
      static const char* kRoles[] = {"system", "developer", "user", "assistant",
                                     "tool"};
      if (role >= 5) {
        std::fprintf(stderr, "bad role %u in fixture %u\n", role, index);
        return 2;
      }
      messages.push_back({kRoles[role], content, thought});
    }

    if (tool_count != 0 || has_tool_calls) {
      ++skipped;
      if (reported < static_cast<std::size_t>(max_reports)) {
        ++reported;
        std::printf(
            "SKIPPED %s: strycast's flint template has no tool preamble / "
            "<tool_call> rendering\n",
            name.c_str());
      }
      continue;
    }
    if (!messages.empty() && messages.back().role == "assistant") {
      ++skipped;
      if (reported < static_cast<std::size_t>(max_reports)) {
        ++reported;
        std::printf(
            "SKIPPED %s: continue mode (trailing assistant turn) is not "
            "modelled by strycast's RenderPrompt\n",
            name.c_str());
      }
      continue;
    }

    ++compared;
    const auto rendered = gufo::models::flint::RenderPrompt(messages);
    if (rendered != expected_rendered) {
      ++render_failures;
      if (reported < static_cast<std::size_t>(max_reports)) {
        ++reported;
        std::size_t diff = 0;
        while (diff < rendered.size() && diff < expected_rendered.size() &&
               rendered[diff] == expected_rendered[diff]) {
          ++diff;
        }
        std::printf("BYTES DIFF %s (first at %zu, lengths %zu vs %zu)\n",
                    name.c_str(), diff, rendered.size(),
                    expected_rendered.size());
        std::printf(
            "  expected: %s\n",
            Readable(std::string_view(expected_rendered).substr(diff)).c_str());
        std::printf("  actual  : %s\n",
                    Readable(std::string_view(rendered).substr(diff)).c_str());
      }
      continue;
    }

    gufo::tokenization::TokenizerOptions options;
    options.parse_special_tokens = true;
    const auto ids = tokenizer->Encode(rendered, options);
    if (ids != expected_ids) {
      ++token_failures;
      if (reported < static_cast<std::size_t>(max_reports)) {
        ++reported;
        std::size_t diff = 0;
        while (diff < ids.size() && diff < expected_ids.size() &&
               ids[diff] == expected_ids[diff]) {
          ++diff;
        }
        std::printf("ID DIFF %s at %zu (%zu vs %zu ids)\n", name.c_str(), diff,
                    expected_ids.size(), ids.size());
      }
    }
  }

  std::printf(
      "G2: %zu/%zu compared fixtures match (%zu render failures, %zu token "
      "failures); %zu skipped (template feature not modelled)%s\n",
      compared - render_failures - token_failures, compared, render_failures,
      token_failures, skipped,
      (render_failures == 0 && token_failures == 0) ? "" : " — FAILED");
  return (render_failures == 0 && token_failures == 0) ? 0 : 1;
}
