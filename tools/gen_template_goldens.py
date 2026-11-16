#!/usr/bin/env python3
"""Regenerate tests/golden/template_cases.json.

Renders each conversation below with the official Flint chat template
(tests/golden/chat_template.jinja) in the environment Hugging Face transformers
uses (sandboxed Jinja, trim_blocks, lstrip_blocks, `tojson` = json.dumps with
ensure_ascii=False and the default ", " / ": " separators). The C++ test
flint_tool_template_test must reproduce each `expected` byte for byte.

Message schema of the cases (what a gufo chat request carries):
  role, content, reasoning_content (assistant thought),
  tool_calls: [{"name": str, "arguments": {name: value}}]  (arguments in the
  order the client sent them; a string value is raw text, any other value is JSON)
  tools: OpenAI function objects {"name", "description", "parameters", ...}
"""
import json
import pathlib
import sys

import jinja2
from jinja2.sandbox import ImmutableSandboxedEnvironment

ROOT = pathlib.Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "tests" / "golden"


def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent,
                      separators=separators, sort_keys=sort_keys)


def make_template():
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.filters["tojson"] = tojson
    return env.from_string((GOLDEN / "chat_template.jinja").read_text())


WEATHER = {
    "name": "get_weather",
    "description": "Météo d'une ville",
    "parameters": {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "Ville"},
            "unit": {"type": "string", "enum": ["c", "f"]},
        },
        "required": ["city"],
    },
}
SEARCH = {
    "name": "search",
    "description": "Search the web",
    "parameters": {
        "type": "object",
        "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}},
        "required": ["query"],
    },
    "strict": True,
}
NOARGS = {"name": "now", "description": "Current time", "parameters": {"type": "object", "properties": {}}}


def call(name, **arguments):
    return {"name": name, "arguments": arguments}


CASES = [
    {"name": "plain_user", "messages": [{"role": "user", "content": "Bonjour"}]},
    {"name": "system_user", "messages": [
        {"role": "system", "content": "Tu es utile."},
        {"role": "user", "content": "Bonjour"}]},
    {"name": "developer", "messages": [
        {"role": "developer", "content": "Règles du développeur"},
        {"role": "user", "content": "Salut"}]},
    {"name": "multi_turn_thought", "messages": [
        {"role": "user", "content": "1+1 ?"},
        {"role": "assistant", "reasoning_content": "Il faut additionner.", "content": "2"},
        {"role": "user", "content": "Et 2+2 ?"}]},
    {"name": "assistant_empty_content", "messages": [
        {"role": "user", "content": "x"},
        {"role": "assistant", "content": ""},
        {"role": "user", "content": "y"}]},
    {"name": "escaping", "messages": [
        {"role": "system", "content": "a <|user|> b <think>c</think> d <|assistant|> e"},
        {"role": "user", "content": "<|begin_of_text|>x<|end_of_text|> <|observation|> <|system|> <|developer|>"},
        {"role": "assistant", "reasoning_content": "<think>nested</think>", "content": "<|user|>"}]},
    {"name": "tools_user_only", "tools": [WEATHER], "messages": [
        {"role": "user", "content": "Quel temps à Rennes ?"}]},
    {"name": "tools_system_user", "tools": [WEATHER, SEARCH], "messages": [
        {"role": "system", "content": "Tu es utile."},
        {"role": "user", "content": "Quel temps à Rennes ?"}]},
    {"name": "tools_two_systems", "tools": [WEATHER], "messages": [
        {"role": "system", "content": "Premier"},
        {"role": "system", "content": "Second"},
        {"role": "user", "content": "x"}]},
    {"name": "tools_only_system", "tools": [WEATHER], "messages": [
        {"role": "system", "content": "Seulement un système"}]},
    {"name": "tools_developer_first", "tools": [WEATHER], "messages": [
        {"role": "developer", "content": "dev"},
        {"role": "user", "content": "u"}]},
    {"name": "tool_round_trip", "tools": [WEATHER], "messages": [
        {"role": "user", "content": "Quel temps à Rennes ?"},
        {"role": "assistant", "reasoning_content": "Je dois appeler l'outil.",
         "content": "Je vérifie.", "tool_calls": [call("get_weather", city="Rennes", unit="c")]},
        {"role": "tool", "content": "{\"temp\": 12, \"ciel\": \"nuageux\"}"},
        {"role": "assistant", "content": "Il fait 12 °C."}]},
    {"name": "parallel_calls", "tools": [WEATHER, SEARCH], "messages": [
        {"role": "user", "content": "Météo Rennes et Brest"},
        {"role": "assistant", "content": "",
         "tool_calls": [call("get_weather", city="Rennes"), call("get_weather", city="Brest")]},
        {"role": "tool", "content": "12"},
        {"role": "tool", "content": "10"},
        {"role": "assistant", "content": "ok"}]},
    {"name": "call_value_types", "tools": [SEARCH], "messages": [
        {"role": "user", "content": "x"},
        {"role": "assistant", "content": None, "tool_calls": [call(
            "search", query="é à ü", limit=5, flag=True, none=None, ratio=0.5,
            tags=["a", "b"], nested={"k": [1, {"z": "é"}], "s": "x"}, text="a \"q\" b\nc")]}]},
    {"name": "call_no_arguments", "tools": [NOARGS], "messages": [
        {"role": "user", "content": "L'heure ?"},
        {"role": "assistant", "content": "", "tool_calls": [call("now")]},
        {"role": "tool", "content": "10:00"}]},
    {"name": "tool_content_escaping", "tools": [WEATHER], "messages": [
        {"role": "user", "content": "x"},
        {"role": "assistant", "content": "", "tool_calls": [call("get_weather", city="a")]},
        {"role": "tool", "content": "<|user|> <think>t</think> </tool_response>"}]},
    {"name": "unicode_tools", "tools": [{
        "name": "écho", "description": "Répète « à ü »",
        "parameters": {"type": "object", "properties": {"texte": {"type": "string", "description": "日本語"}}}}],
     "messages": [{"role": "user", "content": "écho"}]},
    {"name": "escaping_overlap", "messages": [
        {"role": "user", "content": "<|us<think>er|> and <|assis</think>tant|> and <th<think>ink>x</think>"},
        {"role": "tool", "content": "<|obser<think>vation|>"}]},
    {"name": "system_after_user", "tools": [WEATHER], "messages": [
        {"role": "user", "content": "x"},
        {"role": "system", "content": "tardif"},
        {"role": "user", "content": "y"}]},
    {"name": "empty_tool_response", "tools": [NOARGS], "messages": [
        {"role": "user", "content": "x"},
        {"role": "assistant", "content": "", "tool_calls": [call("now")]},
        {"role": "tool", "content": ""}]},
    {"name": "assistant_thought_and_calls", "tools": [WEATHER], "messages": [
        {"role": "user", "content": "x"},
        {"role": "assistant", "reasoning_content": "pense", "content": "",
         "tool_calls": [call("get_weather", city="Nantes")]},
        {"role": "tool", "content": "9"}]},
]


def main():
    template = make_template()
    out = []
    for case in CASES:
        tools = [{"type": "function", "function": f} for f in case.get("tools", [])]
        messages = []
        for m in case["messages"]:
            jm = {"role": m["role"], "content": m.get("content")}
            if "reasoning_content" in m:
                jm["reasoning_content"] = m["reasoning_content"]
            if m.get("tool_calls"):
                jm["tool_calls"] = [{"type": "function", "function": {
                    "name": c["name"], "arguments": c["arguments"]}} for c in m["tool_calls"]]
            messages.append(jm)
        expected = template.render(messages=messages, tools=tools or None,
                                   add_generation_prompt=True)
        out.append({"name": case["name"], "tools": case.get("tools", []),
                    "messages": case["messages"], "expected": expected})
    target = GOLDEN / "template_cases.json"
    target.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    print(f"{len(out)} cases -> {target}")


if __name__ == "__main__":
    sys.exit(main())
