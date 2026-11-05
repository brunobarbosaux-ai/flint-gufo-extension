import json, urllib.request, sys, time, argparse

# Fixed battery: mixed content types, identical across drafter quants.
# temp 1.0 / top-p 0.95 (server defaults), so acceptance has +-0.15 run noise.
PROMPTS = [
    ("tech",  "Explain the difference between TCP and UDP.", 512),
    ("code",  "Write a Python function that checks whether a string is a palindrome. Include a few test cases.", 512),
    ("prose", "Describe the water cycle step by step.", 512),
    ("enum",  "List the planets of the solar system in order from the sun, comma-separated, nothing else.", 1024),
    ("math",  "What is 17 multiplied by 23? Give just the number.", 1024),
]

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:8087/v1/chat/completions")
ap.add_argument("--model", default="flint")
ap.add_argument("--temperature", type=float, default=1.0)
ap.add_argument("--top-p", type=float, default=0.95)
ap.add_argument("--out", default=None)
args = ap.parse_args()

# throwaway warmup (graph compile, KV alloc)
body = json.dumps({"model": args.model, "max_tokens": 8,
                   "messages": [{"role": "user", "content": "hi"}]}).encode()
urllib.request.urlopen(urllib.request.Request(
    args.url, data=body,
    headers={"Content-Type": "application/json"}), timeout=300)

rows = []
for name, q, mt in PROMPTS:
    body = json.dumps({"model": args.model, "max_tokens": mt,
                       "temperature": args.temperature, "top_p": args.top_p,
                       "messages": [{"role": "user", "content": q}]}).encode()
    req = urllib.request.Request(args.url,
                                 data=body, headers={"Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(req, timeout=300))
    t = d["timings"]; ch = d["choices"][0]
    da = t.get("draft_n_accepted", 0); dn = t.get("draft_n", 0)
    rows.append((name, t["predicted_per_second"], da, dn, t["predicted_n"], ch["finish_reason"]))
    print("%-5s decode %6.1f t/s | draft %4d/%4d (%.3f) | gen %4d | %s" %
          (name, t["predicted_per_second"], da, dn, (da / dn if dn else 0.0),
           t["predicted_n"], ch["finish_reason"]))

ta = sum(r[2] for r in rows); td = sum(r[3] for r in rows)
tt = sum(r[4] / r[1] for r in rows); tk = sum(r[4] for r in rows)
if td:
    print("AGG: acceptance %.3f (%d/%d) | aggregate decode %.1f t/s (%d tok / %.2f s)" %
          (ta / td, ta, td, tk / tt, tk, tt))
else:
    print("AGG: (no drafter) | aggregate decode %.1f t/s (%d tok / %.2f s)" %
          (tk / tt, tk, tt))

if args.out:
    with open(args.out, "w") as f:
        json.dump({"url": args.url, "model": args.model,
                   "temperature": args.temperature, "top_p": args.top_p,
                   "rows": [{"name": r[0], "decode_tps": r[1], "draft_accepted": r[2],
                             "draft_n": r[3], "gen": r[4], "finish": r[5]} for r in rows],
                   "agg_decode_tps": (tk / tt), "agg_acceptance": (ta / td if td else None)},
                  f, indent=2)
