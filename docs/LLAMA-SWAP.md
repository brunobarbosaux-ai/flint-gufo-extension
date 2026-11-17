# Running Flint with speculation under llama-swap

Flint's DFlash2 speculative decoding is opt-in. Enable it with environment
variables on the **process that runs Gufo**. Under llama-swap that process is
the `cmd:` of a model entry, so the variables go on that command.

## Settings

| Variable | Default | Meaning |
| --- | --- | --- |
| `FLINT_SPEC` | off | `dflash` enables DFlash2; `off` or unset disables; `null` is a test mode (no-op drafter) |
| `FLINT_DRAFT` | `Flint-DFlash2-Q6_K.gguf` | path to the drafter GGUF, inside the process's filesystem |

The backend is also selectable per request with `--speculative dflash2`
(`gufo serve`), and `--draft-tokens`, `--min-draft-tokens` and `--draft-policy`
(`fixed` or `adaptive`, default `adaptive`) tune it. See `README.md`.

## Example llama-swap entry (Docker, official runtime image)

This is the shape of the existing `flint-ext` entry with speculation enabled.
The drafter must be mounted into the container, because `FLINT_DRAFT` is read
inside it:

```yaml
  flint-ext-spec:
    name: "Flint 40B + DFlash2 speculation"
    cmd: >
      docker run --name ${MODEL_ID} --rm
      --device=/dev/kfd --device=/dev/dri
      --group-add 992
      --ulimit memlock=-1
      -e FLINT_SPEC=dflash
      -e FLINT_DRAFT=/models/mtp/Flint-DFlash2-Q6_K.gguf
      -p ${PORT}:8080
      -v gufo-extensions:/home/gufo/.local/share/gufo/extensions
      -v /home/bruno/AI/GGUF:/models:ro
      ghcr.io/gufo-org/toolboxes/gufo-runtime:latest
      gufo serve --host 0.0.0.0 --port 8080 llm
      -m /models/flint-UD-Q4_K_XL.gguf
      --served-model-name flint-ext-spec
      -j 1
      -c 262144
    cmdStop: docker rm -f ${MODEL_ID}
```

Set `-e` flags on `docker run`, not on llama-swap itself: llama-swap only
launches the child, and the child is what reads `FLINT_SPEC`.

Keep `-j 1`. Flint serializes concurrent requests (`physical_width 1`), so
speculation targets single-stream latency. That matches llama-swap's one-model-
at-a-time model: only one Flint process holds the GPU, and a second request
waits for the first.

## VRAM

- Each speculative session adds about **0.44 GB** to its claim (hidden ring
  about 320 MiB plus context scratch). The drafter weights (Q6_K 336 MB, Q4_K_M
  240 MB, BF16 811 MB) are resident once per runner. Together, one speculative
  session costs about **0.7 to 1.2 GB** more than spec-off.
- The 262144-context KV cache and the Q4_K_XL weights are unchanged by
  speculation; only the per-session figure above is added. Check
  `rocm-smi --showmemuse` after the first speculative request.
- The base-session VRAM figure is undercounted by about 600-666 MiB (pre-
  existing, see `DEFERRED.md`), so leave headroom beyond the figures above.

## Mounting the drafter

Copy or mount a drafter GGUF where the process can read it. Here the drafter
lives in the local `mtp/` folder, mounted at `/models/mtp` in the example.
The Q4_K_M drafter saves about 96 MB of VRAM versus Q6_K, and it measured
slightly faster (decode speedup 1.345 vs 1.263 at depth 0). The GX gate passes
for all three quants.

## Caveats

- Greedy output (`temp 0`) is byte-identical with speculation on or off.
- At `temp > 0`, a fixed seed does not guarantee identical output with
  speculation on (accept/reject resampling). Turn speculation off where seed
  reproducibility is needed. See `README.md` and `docs/QUALITY.md`.
- Adaptive speculation is faster at temp 0 and 0.7 but about 3-6% slower at
  temp 1.0 in the M4 measurements. Keep it off for temp-1.0 workloads if
  throughput matters there.
