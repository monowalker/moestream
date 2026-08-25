# S44 — llama.cpp has its own MoE offload. How does it compare?

*2026-08-24. Ornith-1.0-35B-UD-IQ4_NL, ctx 32768, ubatch 1024, N_PARALLEL=1,
two warm-up requests, median of three 200-token generations.*

## Why this needed measuring

Upstream added `-cmoe` / `--cpu-moe` ("keep all Mixture of Experts weights in the
CPU") and `-ncmoe N` / `--n-cpu-moe N` (the first N layers' worth). That is the
closest thing llama.cpp has to what this project does, so it is the comparison
worth publishing — not another baseline to beat, but the honest question of
whether the patch is still worth applying.

## Measured

| | decode | device memory (GTT+VRAM) |
|---|---:|---:|
| plain llama.cpp | **42.7 ms/tok** | 17.78 GiB |
| upstream `--cpu-moe` | 71.8 ms/tok | 18.02 GiB |
| upstream `--n-cpu-moe 20` | 80.2 ms/tok | 17.98 GiB |
| upstream `--n-cpu-moe 30` | 77.0 ms/tok | 17.97 GiB |
| **MoEStream `frac=0.25`** | **63.7 ms/tok** | **7.93 GiB** |
| **MoEStream `frac=0.40`** | **54.6 ms/tok** | **10.09 GiB** |

On speed the comparison is clean: **MoEStream at `frac=0.25` is faster than
upstream's offload (63.7 against 71.8) and at `frac=0.40` it is a third faster
(54.6).** Both were measured in the same session with the same protocol.

## The memory half is not confirmed, and should not be quoted

The device-memory column says `--cpu-moe` frees nothing — 18.02 GiB against
17.78 plain. That is a strong claim about someone else's feature and **this
measurement does not support making it.**

The probe is the GTT+VRAM counter used for every other number in this project.
It is the right instrument on a discrete GPU. On this machine it may not be: GTT
*is* system RAM made visible to the GPU, so "move the experts to CPU memory" and
"keep them in GTT" are not obviously different allocations. Two attempts to check
it against llama.cpp's own accounting failed — the server does not print its
buffer breakdown at the verbosity the container runs at, and container RSS is
useless because the weights are mapped rather than copied (166–228 MiB in every
configuration, including plain).

So the honest statement is: **on a unified-memory machine, upstream's MoE offload
did not reduce what this project measures as device memory, and we could not
determine whether that is a property of the feature or of the measurement.**
Anyone reproducing this on a discrete GPU would get a cleaner answer, and the
answer would probably be different.

## What it does establish

The speed comparison stands on its own, and it is the part that matters for
"should this patch exist". Upstream's offload pays about 1.7x for moving experts
off the GPU. MoEStream pays 1.28x at `frac=0.40` while holding 10.09 GiB instead
of 17.78 — and unlike `--cpu-moe`, it also starts models that do not fit at all,
which upstream's offload does not address on this machine (the weights still have
to be somewhere, and there is only one pool of RAM).
