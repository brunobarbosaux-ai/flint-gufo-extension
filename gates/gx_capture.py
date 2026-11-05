#!/usr/bin/env python3
"""GX greedy-exactness capture for Flint Phase 7.

The serve API does not expose token ids, so GX compares the full greedy
generation text (content + reasoning_content) for the five fixed prompts
from drafter_batt.py at temperature 0. This is the same comparable surface
G4/G8 use. Run once per step; compare with `gx_compare.py`.

Usage:
  gx_capture.py --url http://127.0.0.1:8016/v1/chat/completions \
      --model flint-q6 --out gx-7.0.json [--max-tokens 256]
"""
import argparse
import json
import time
import urllib.request

PROMPTS = [
    ("tech", "Explain the difference between TCP and UDP."),
    ("code", "Write a Python function that checks whether a string is a "
             "palindrome. Include a few test cases."),
    ("prose", "Describe the water cycle step by step."),
    ("enum", "List the planets of the solar system in order from the sun, "
             "comma-separated, nothing else."),
    ("math", "What is 17 multiplied by 23? Give just the number."),
]


def post(url, payload, timeout=900):
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    rows = []
    for name, prompt in PROMPTS:
        start = time.time()
        data = post(args.url, {
            "model": args.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": 1.0,
            "top_k": 0,
            "seed": 1234,
        })
        message = data["choices"][0]["message"]
        content = message.get("content") or ""
        reasoning = message.get("reasoning_content") or ""
        rows.append({
            "name": name,
            "prompt": prompt,
            "content": content,
            "reasoning_content": reasoning,
            "full": reasoning + content,
            "seconds": round(time.time() - start, 2),
        })
        print(f"  [{name}] {len(content)} chars content, "
              f"{len(reasoning)} chars reasoning, {rows[-1]['seconds']}s")

    with open(args.out, "w") as handle:
        json.dump({"gate": "GX", "model": args.model,
                   "settings": {"temperature": args.temperature,
                                "max_tokens": args.max_tokens},
                   "rows": rows}, handle, indent=1)
    print(f"GX written: {args.out}")


if __name__ == "__main__":
    main()
