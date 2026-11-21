// Tests the Flint tool-call scanner (src/tool_dialect.cpp): complete calls,
// typed and untyped values, malformed markup, unfinished calls while streaming,
// and the rule that streamed arguments only grow.

#include <cstddef>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "src/tool_dialect.hpp"

namespace {

using gufo::ext::ScannedToolCall;
using gufo::models::flint::ForgetToolSchemas;
using gufo::models::flint::RememberToolSchemas;
using gufo::models::flint::ScanFlintToolCalls;

int g_failures = 0;
int g_checks = 0;

#define CHECK(...)                                                          \
  do {                                                                      \
    ++g_checks;                                                             \
    if (!(__VA_ARGS__)) {                                                   \
      ++g_failures;                                                         \
      std::cout << "FAIL " << __FILE__ << ":" << __LINE__ << "  "           \
                << #__VA_ARGS__ << "\n";                                    \
    }                                                                       \
  } while (false)

struct Scan {
  std::vector<ScannedToolCall> calls;
  std::vector<std::pair<std::size_t, std::size_t>> spans;
};

Scan Run(std::string_view text, bool partial = false) {
  Scan scan;
  ScanFlintToolCalls(text, {}, &scan.calls, &scan.spans, partial);
  return scan;
}

std::string Call(const std::string& name,
                 const std::vector<std::pair<std::string, std::string>>& args) {
  std::string out = "<tool_call>" + name;
  for (const auto& [key, value] : args)
    out += "<arg_key>" + key + "</arg_key><arg_value>" + value + "</arg_value>";
  return out + "</tool_call>";
}

void TestComplete() {
  ForgetToolSchemas();
  const std::string text = Call("get_weather", {{"city", "Rennes"}, {"unit", "c"}});
  const Scan scan = Run(text);
  CHECK(scan.calls.size() == 1);
  CHECK(scan.calls[0].name == "get_weather");
  CHECK(scan.calls[0].valid && scan.calls[0].complete);
  CHECK(scan.calls[0].arguments.size() == 2);
  CHECK(scan.calls[0].arguments[0].name == "city");
  CHECK(scan.calls[0].arguments[0].value == "Rennes");
  CHECK(scan.calls[0].arguments[0].is_string);
  CHECK(scan.spans.size() == 1 && scan.spans[0] == std::make_pair<std::size_t, std::size_t>(0, text.size()));
}

void TestTextAroundAndParallel() {
  ForgetToolSchemas();
  const std::string a = Call("a", {{"x", "1"}});
  const std::string b = Call("b", {});
  const std::string text = "ok " + a + " and " + b + " bye";
  const Scan scan = Run(std::string_view(text).substr(3));  // host passes text from the first marker
  CHECK(scan.calls.size() == 2);
  CHECK(scan.calls[0].name == "a" && scan.calls[1].name == "b");
  CHECK(scan.calls[1].arguments.empty());
  CHECK(scan.spans.size() == 2);
  CHECK(scan.spans[0].first == 0 && scan.spans[0].second == a.size());
  CHECK(scan.spans[1].first == a.size() + 5 && scan.spans[1].second == a.size() + 5 + b.size());
}

void TestTypedValues() {
  RememberToolSchemas({{"search", R"({"type":"object","properties":{
      "query":{"type":"string"},"limit":{"type":"integer"},"flag":{"type":"boolean"},
      "tags":{"type":"array","items":{"type":"string"}},"opts":{"type":"object"},
      "maybe":{"type":["null","string"]},"either":{"anyOf":[{"type":"integer"},{"type":"string"}]},
      "free":{}}})"}});
  const Scan scan = Run(Call("search", {{"query", " a b "}, {"limit", " 5 "}, {"flag", "true"},
                                        {"tags", "[\"a\", \"b\"]"}, {"opts", "{\"k\": 1}"},
                                        {"maybe", "x"}, {"either", "7"}, {"free", "7"},
                                        {"undeclared", "9"}}));
  CHECK(scan.calls.size() == 1);
  const auto& args = scan.calls[0].arguments;
  CHECK(args.size() == 9);
  CHECK(args[0].is_string && args[0].value == " a b ");  // strings are kept as written
  CHECK(!args[1].is_string && args[1].value == "5");     // JSON values are trimmed
  CHECK(!args[2].is_string && args[2].value == "true");
  CHECK(!args[3].is_string && args[3].value == "[\"a\", \"b\"]");
  CHECK(!args[4].is_string && args[4].value == "{\"k\": 1}");
  CHECK(args[5].is_string);   // type list holding "string"
  CHECK(args[6].is_string);   // anyOf with a string branch
  CHECK(args[7].is_string);   // no type information
  CHECK(args[8].is_string);   // undeclared parameter
  ForgetToolSchemas();
}

void TestUnknownToolIsStrings() {
  ForgetToolSchemas();
  const Scan scan = Run(Call("mystery", {{"n", "5"}}));
  CHECK(scan.calls.size() == 1 && scan.calls[0].arguments[0].is_string);
}

void TestWhitespaceTolerance() {
  ForgetToolSchemas();
  const std::string text =
      "<tool_call> get_weather \n <arg_key> city </arg_key>\n<arg_value>Rennes</arg_value>\n"
      "</tool_call>";
  const Scan scan = Run(text);
  CHECK(scan.calls.size() == 1);
  CHECK(scan.calls[0].name == "get_weather");
  CHECK(scan.calls[0].arguments.size() == 1 && scan.calls[0].arguments[0].name == "city");
  CHECK(scan.spans[0].second == text.size());
}

void TestValuesWithMarkup() {
  ForgetToolSchemas();
  // A string value may hold angle brackets and even the closing tag of the call.
  const Scan scan = Run(Call("write", {{"content", "if a<b && c>d { </tool_call> }"}}));
  CHECK(scan.calls.size() == 1);
  CHECK(scan.calls[0].arguments[0].value == "if a<b && c>d { </tool_call> }");
}

void TestMalformed() {
  ForgetToolSchemas();
  // No value tag, no key close, empty name, stray text: none is a call, and the
  // later well-formed call is still read.
  const std::string good = Call("ok", {{"k", "v"}});
  for (const std::string& bad : {
           std::string("<tool_call>f<arg_key>k</arg_key>oops</tool_call>"),
           std::string("<tool_call><arg_key>k</arg_key><arg_value>v</arg_value></tool_call>"),
           std::string("<tool_call>f junk<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>"),
           std::string("<tool_call>f<arg_key></arg_key><arg_value>v</arg_value></tool_call>"),
           std::string("<tool_call>f stray text</tool_call>")}) {
    const Scan scan = Run(bad + good);
    CHECK(scan.calls.size() == 1);
    CHECK(!scan.calls.empty() && scan.calls[0].name == "ok");
    CHECK(scan.spans.size() == 1 && scan.spans[0].first == bad.size());
  }
}

void TestPartial() {
  RememberToolSchemas({{"search", R"({"type":"object","properties":{
      "query":{"type":"string"},"limit":{"type":"integer"}}})"}});
  // The name is not final until a tag follows it.
  CHECK(Run("<tool_call>sea", true).calls.empty());
  CHECK(Run("<tool_call>search", true).calls.empty());
  {
    const Scan scan = Run("<tool_call>search<", true);
    CHECK(scan.calls.size() == 1 && scan.calls[0].name == "search");
    CHECK(!scan.calls[0].complete && scan.calls[0].valid);
    CHECK(scan.spans.empty());
  }
  // A string value being written, minus a tail that may open the closing tag.
  {
    const Scan scan = Run("<tool_call>search<arg_key>query</arg_key><arg_value>hel", true);
    CHECK(scan.calls.size() == 1 && scan.calls[0].unfinished.has_value());
    CHECK(scan.calls[0].unfinished->name == "query" && scan.calls[0].unfinished->value == "hel");
  }
  {
    const Scan scan = Run("<tool_call>search<arg_key>query</arg_key><arg_value>hel</arg_v", true);
    CHECK(scan.calls.size() == 1 && scan.calls[0].unfinished->value == "hel");
  }
  {
    const Scan scan = Run("<tool_call>search<arg_key>query</arg_key><arg_value>a<b</arg_valu", true);
    CHECK(scan.calls[0].unfinished->value == "a<b");
  }
  // A JSON value waits until it is closed.
  {
    const Scan scan = Run("<tool_call>search<arg_key>limit</arg_key><arg_value>1", true);
    CHECK(scan.calls.size() == 1 && !scan.calls[0].unfinished.has_value());
    CHECK(scan.calls[0].arguments.empty());
  }
  // Closed arguments are kept while the next one is unfinished.
  {
    const Scan scan = Run("<tool_call>search<arg_key>query</arg_key><arg_value>x</arg_value><arg_key>li", true);
    CHECK(scan.calls.size() == 1 && scan.calls[0].arguments.size() == 1);
    CHECK(scan.calls[0].arguments[0].value == "x" && !scan.calls[0].unfinished.has_value());
  }
  // The final scan does not append an unfinished call.
  CHECK(Run("<tool_call>search<arg_key>query</arg_key><arg_value>he", false).calls.empty());
  // A complete call followed by the start of another.
  {
    const std::string done = Call("a", {{"k", "v"}});
    const Scan scan = Run(done + "<tool_call>b<arg_key>q</arg_key><arg_value>par", true);
    CHECK(scan.calls.size() == 2 && scan.calls[0].complete && !scan.calls[1].complete);
    CHECK(scan.spans.size() == 1);
    CHECK(scan.calls[1].unfinished && scan.calls[1].unfinished->value == "par");
  }
  ForgetToolSchemas();
}

// Streaming: for every prefix of a text, the partial scan must extend what the
// previous prefix produced (same calls, arguments only grow, the unfinished
// value only grows) and the final scan of the whole text is complete.
void TestStreamingMonotonic() {
  RememberToolSchemas({{"write", R"({"type":"object","properties":{
      "path":{"type":"string"},"content":{"type":"string"},"mode":{"type":"integer"}}})"}});
  const std::string text =
      "<tool_call>write<arg_key>path</arg_key><arg_value>/tmp/a.txt</arg_value>"
      "<arg_key>content</arg_key><arg_value>line one\n<b>line</b> two</arg_value>"
      "<arg_key>mode</arg_key><arg_value> 420 </arg_value></tool_call>"
      "<tool_call>write<arg_key>path</arg_key><arg_value>/tmp/b.txt</arg_value></tool_call>";
  Scan previous;
  for (std::size_t length = 0; length <= text.size(); ++length) {
    const Scan now = Run(std::string_view(text).substr(0, length), true);
    CHECK(now.calls.size() >= previous.calls.size());
    for (std::size_t i = 0; i < previous.calls.size() && i < now.calls.size(); ++i) {
      CHECK(now.calls[i].name == previous.calls[i].name);
      CHECK(now.calls[i].arguments.size() >= previous.calls[i].arguments.size());
      for (std::size_t a = 0; a < previous.calls[i].arguments.size(); ++a)
        CHECK(now.calls[i].arguments[a].value == previous.calls[i].arguments[a].value);
      if (previous.calls[i].unfinished && !now.calls[i].complete &&
          now.calls[i].arguments.size() == previous.calls[i].arguments.size()) {
        CHECK(now.calls[i].unfinished.has_value());
        if (now.calls[i].unfinished)
          CHECK(now.calls[i].unfinished->value.starts_with(previous.calls[i].unfinished->value));
      }
    }
    previous = now;
  }
  const Scan full = Run(text, false);
  CHECK(full.calls.size() == 2);
  CHECK(full.calls[0].arguments.size() == 3);
  CHECK(full.calls[0].arguments[1].value == "line one\n<b>line</b> two");
  CHECK(!full.calls[0].arguments[2].is_string && full.calls[0].arguments[2].value == "420");
  ForgetToolSchemas();
}

void TestNoMarker() {
  CHECK(Run("just text, no call", true).calls.empty());
  CHECK(Run("", true).calls.empty());
  CHECK(Run("<tool", true).calls.empty());
}

}  // namespace

int main() {
  TestComplete();
  TestTextAroundAndParallel();
  TestTypedValues();
  TestUnknownToolIsStrings();
  TestWhitespaceTolerance();
  TestValuesWithMarkup();
  TestMalformed();
  TestPartial();
  TestStreamingMonotonic();
  TestNoMarker();
  std::cout << (g_failures == 0 ? "all passed" : "failures") << ": "
            << g_checks - g_failures << "/" << g_checks << " checks\n";
  return g_failures == 0 ? 0 : 1;
}
