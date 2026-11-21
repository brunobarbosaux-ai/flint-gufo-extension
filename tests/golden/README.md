# Golden prompts for the Flint chat template

`chat_template.jinja` is the official template of `RecursiveSI/flint` (Apache-2.0), revision `2a247be0c0aec895ccb0ac7ca92229ec6df37288`,
copied unchanged. `template_cases.json` holds conversations and the prompt that this template renders for each
one; `tools/gen_template_goldens.py` regenerates it (Python, jinja2, with the `tojson` filter and the
whitespace settings of Hugging Face transformers). The C++ test `flint_tool_template_test` renders the same
conversations with `RenderPrompt` and compares byte for byte.
