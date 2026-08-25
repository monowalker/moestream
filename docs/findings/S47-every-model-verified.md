# S47 — Every model, every learn mode, verified end to end

*2026-08-25. Eight models, both learn passes, memory measured against the
baseline that exists rather than the file on disk.*

## Why this was necessary

`SPEC_DECODING=learn` was implemented, documented as working, and made the
launcher's default on the strength of one observation: the entrypoint printed
`measuring n_max=off` at startup. Nothing had ever checked that the loop closed —
that a row got written, that the next start read it, that the search advanced.

It did not. Two defects were sitting in the shipped path:

**`curl` is not in the image.** The recorder polled `/metrics` with `curl`, which
does not exist — `compose.yaml`'s healthcheck says so in a comment and uses
`python3` for exactly that reason. The recorder never fired, so no candidate was
ever recorded and the search could never move past its first one.

**State files were written mode 600 by root.** `mktemp` creates 600; the C++ side
writes 644. After the first fix the rows were being written and were unreadable
from the host, which made the fix look like it had not worked.

Neither would have been found by reading the code, running the launcher, or
checking that the docs matched. They needed the feature to be *used*.

## Memory, against a real baseline

The earlier tables compared against the size of the GGUF on disk. That is the
only reference the three largest models have — plain llama.cpp cannot start them
— but it is the wrong one wherever a baseline exists. Both sides started in the
same session, ctx 16384, `N_PARALLEL=1`, KV `q8_0`:

| | plain llama.cpp | MoEStream | |
|---|---:|---:|---|
| Qwen3.5-4B (dense) | 3.95 GiB | **2.75 GiB** | 70% |
| Qwen3.8-27B (dense) | 16.25 GiB | **7.57 GiB** | **47%** |
| gemma-4-31B (dense) | 19.04 GiB | **8.44 GiB** | **44%** |
| Ornith-1.0-35B (MoE) | 17.55 GiB | **7.71 GiB** | **44%** |
| Ornith-1.5-35B (MoE) | 20.40 GiB | **7.80 GiB** | **38%** |
| Qwen3-Coder-Next (MoE) | does not start | 13.18 GiB | 36% of the file |
| Laguna-S-2.1 (MoE) | does not start | 21.09 GiB | 39% of the file |
| gpt-oss-120b (MoE) | does not start | 20.57 GiB | 35% of the file |

**Every model tested drops, from 4B to 120B.** Output was correct on all eight
and every figure reproduced across two starts.

**Qwen3.5-4B had been written up as a model streaming does not help.** It does —
30%. The claim came from comparing 2.75 GiB against a 2.71 GiB *file*. A model
file is not a memory baseline: the runtime adds compute buffers, a KV cache and
allocator slack on top, and on a 4B model that overhead is a third of the total.

## The learn loops, closed

| | verified by |
|---|---|
| `MOESTREAM_CACHE_FRAC=learn` | recommended 0.50 from measured reuse distances, wrote `tuning.tsv`, next start used it |
| `UBATCH=learn` | measured 1024 at 17.4 tok/s prefill, wrote `ubatch.tsv` |
| `SPEC_DECODING=learn` | wrote `off`, advanced to `n_max=1` on the next start, and the rates carry the right sign — dense 1.4 → 1.9 tok/s, MoE 9.8 → 9.2 |
| keyed by concurrency | one start at `N_PARALLEL=1` and one at `4` produced two rows, `…/p1` and `…/p4` |

That last one matters because speculation is worth 2.08x at one request and
costs 34% at four ([S45](S45-speculation-and-batching.md)). Without the key a
value learned at one concurrency would be applied at another, and the learn loop
could not notice — it never re-measures a row it already has.

## A behaviour users have to know about

**Learned settings are written from an `atexit` handler**, so they survive
SIGTERM and are lost to SIGKILL:

```
make down / docker compose down / docker stop   → saved
docker kill / docker rm -f                      → lost
```

This is sharper than it sounds. Each start tries exactly one candidate. If the
row is never written, the next start measures the same candidate again and the
search never advances. The verification harness used `docker rm -f` between runs
and threw away every measurement it took — sixteen starts, nothing recorded, and
it read as a product bug until the harness was the suspect.

Documented in [`USAGE.md`](../USAGE.md) §4.

## One model that cannot be used at all

`Qwen3.8-27B-Q4_0_ROCMFP4_STRIX.gguf` fails in llama.cpp before MoEStream sees
it: `tensor 'output.weight' has invalid ggml type 101, should be in [0, 43)`. A
vendor FP4 format upstream does not implement. Plain llama.cpp will not load it
either; there is nothing here to fix.
