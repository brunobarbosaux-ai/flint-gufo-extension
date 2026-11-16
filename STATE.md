# M1c State

## Current phase
A — Selector batch regression test

## Decisions
- Follow `docs/PLAN-M1C-ACCEPTANCE.md` phases A→G in order.
- D runs only if C verdict is B.
- Never loosen gates (1e-4 stage, 5e-6 selector).
- One GPU model process at a time.

## Environment
- Extension repo: /home/bruno/Projects/flint-gufo-extension
- Build dir: /home/bruno/AI/Gufo/build/host-test
- Tool binaries: /home/bruno/AI/Gufo/build/host-test/extensions/build/flint/build/
- Target: /home/bruno/AI/GGUF/flint-UD-Q4_K_XL.gguf
- Drafters: /home/bruno/AI/GGUF/mtp/Flint-DFlash2-{BF16,Q6_K}.gguf
- Oracle: /home/bruno/AI/llama/llama-b11804-rocm/llama-server
- Upstream: /tmp/opencode/dflash-upstream (pinned 07ebd93)

## Open questions
- B0: layer_types full vs sliding attention (suspect).

## B0 findings (drafter config facts)

| field | extension | source of truth | value there | match |
|---|---|---|---|---|
| target_layer_ids | [5,15,26,36,46] | GGUF dflash.target_layers=[6,16,27,37,47] (layer-inputs) + HF config | [5,15,26,36,46] (outputs) | YES (equivalent: input il == output il-1) |
| layer_types | ConfigJson writes all "full_attention" | HF config + GGUF sliding_window_pattern=[1,1,1,1,1] | all "sliding_attention" | NO (harness-only; extension kernel always applies window = correct) |
| input_embedding_scale | absent -> 1.0 | HF config (no dflash_config key) | 1.0 | YES |
| mask_token_id | GGUF tokenizer.ggml.mask_token_id | GGUF | 150019 | YES |
| rope_theta | GGUF | GGUF dflash.rope.freq_base | 10000.0 | YES |
| rms_norm_eps | GGUF | GGUF | 1e-6 | YES |
| head_dim | GGUF | GGUF | 128 | YES |
| block_size | GGUF | GGUF | 8 | YES |
| sliding_window | GGUF | GGUF | 2048 | YES |
| selector_rank/top_k | GGUF | GGUF | 256/16 | YES |

Suspect #1: ConfigJson layer_types=full_attention vs truth sliding_attention. Harness-only; affects parity only past 2048. Fix in Phase E.

p_min (llama.cpp common/speculative.cpp ~1349): per-position softmax prob of the argmax over the selector's top_k transition scores; break when 1/sum < p_min. Extension greedy confidence is always 1.0 -> cannot reproduce without kernel changes. Expected: "p_min 0.5 not reproduced".

llama.cpp reads target_layers directly as t_layer_inp[il] (input of layer il = output of il-1). Extension normalizes GGUF [6,16,27,37,47] -> [5,15,26,36,46]. Equivalent.

EOS ids (HF config): [150001, 150003, 150005, 150018]. bos 150000. vocab 150144.
