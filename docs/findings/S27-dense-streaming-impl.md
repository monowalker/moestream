# Finding S27 — dense FFN streaming, implemented and measured

| | |
|---|---|
| Target | making finding S18's argument real: stream a dense model's FFN through the arena |
| Date | 2026-08-23 |
| Verdict | **works.** −56% device memory at bit-identical output; prefill free above ubatch 1024 |
| Measured on | `research/spikes/s27_dense_streaming/`, Qwen3.8-27B-IQ4_NL and Qwen3.5-4B-UD-Q4_K_XL |

## What was built

`MOESTREAM_DENSE_FRAC` keeps the first *f* of a dense model's layers resident
and streams the rest of their FFN weights through an arena, one layer at a time.
**Accuracy is untouched** — the same bytes are used, they are just not all
resident at once, which is the trade this project exists to make.

Three things had to be true, and only the first was obvious from S18:

1. **The slab does not transfer.** It works by shrinking `ne[2]`, the expert
   dimension, so `mul_mat_id` runs against a smaller array. A dense FFN weight
   is 2D and goes through plain `mul_mat`. Only the arena transfers.
2. **The model's own tensors must be stubs.** Allocating them at full size and
   merely not loading them leaves the whole model resident and saves nothing.
   They are allocated one row wide and substituted at graph build.
3. **The tail is streamed, not the head.** Which layers are chosen does not
   change bytes per token — a dense model uses every layer exactly once — but
   with the tail streamed the resident head's compute is time the reads hide
   behind.

Whether a model is dense or MoE is read from the GGUF (`expert_count`), and
`auto` streams the least that still fits, so a model that already fits streams
nothing and behaves exactly like plain llama.cpp. Nothing is selected by hand.

## Result

> **[Corrected 2026-08-23]** The decode figures below were later re-measured
> against a baseline held in the same configuration (finding S34). The memory
> and prompt-processing results are unchanged; the *cost* is 1.94–2.86x
> depending on batching and speculation, and never less. Any ratio in this
> document computed against a differently-configured baseline is superseded by
> S34's matrix.

| | resident | prefill | decode |
|---|---:|---:|---:|
| Qwen3.8-27B, plain llama.cpp | 15.58 GiB | 67.5 tok/s | 211.0 ms/tok |
| Qwen3.8-27B, `frac=0.50` | 11.51 GiB | 66.8 | 452.7 |
| **Qwen3.8-27B, `frac=0.00`** | **6.90 GiB (−56%)** | **66.0 (−2.2%)** | **675.9** |
| Qwen3.5-4B, plain llama.cpp | 3.30 GiB | — | 44.5 |
| **Qwen3.5-4B, `frac=0.00`** | **2.10 GiB (−36%)** | — | 106.2 |

The 4B saves less because less of it is FFN (1.31 of 3.30 GiB); the mechanism is
the same and needed no per-model work.

**Cost is linear**: about 55 ms per GiB of FFN moved out, flat across the range.
Unlike the expert cache there is no knee — the miss-ratio curve has an interior
optimum, this does not — so `auto` streaming the least that fits is the whole
policy, and `learn` measures the exchange rate rather than searching for a
setting.

## Accuracy: three independent checks

| | |
|---|---|
| greedy output | byte-identical on both models here — but **not guaranteed**: on gemma-4-31B (S33) two of three prompts matched over 1100 characters and the third diverged once the model entered a degenerate repetition loop, where a near-tie makes argmax sensitive to accumulation order. The arena changes memory alignment, so this is the §8.2 mechanism, not corrupted weights |
| **perplexity** | **4.2000 at `frac` 1.00 / 0.40 / 0.00 — identical to four decimals** |
| runtime guards | no `[BUG]` lines, no slot exhaustion |

## Prefetch hides about half, and there is no free zone *at one request*

> **[Corrected 2026-08-23]** The heading used to end at "no free zone". That is
> true at K=1 and false in general: finding S37 measures the cost falling from
> 3.15x to **1.20x** between one and sixteen concurrent requests, because the
> window below scales with the tokens in a pass and the reads do not.

Reads are issued one layer ahead into a double-buffered arena. Unlike every
predictive scheme this project rejected (N2 / S10 / S11 / §10.14), there is
nothing to predict: **layer 33 is always layer 33.**

It hides consistently ~50% — 8.96 GiB at the 9.6 GB/s S19 measured is 1002 ms of
reads, and the observed increment is 480 ms — but not more, and the reason is
structural:

```
one layer's read    143 MiB / 9.6 GB/s = 15.6 ms
one layer's compute 211 ms / 65 layers =  3.3 ms      -> reads outrun compute 4.7x
```

Buffering deeper does not help: to hide the reads behind the resident head's
compute you must hold what you prefetched, and that is as much memory as the
streaming saved. Measured, more buffers are simply worse (690.8 → 704.7 → 721.4
ms at 2 → 3 → 4).

**The free zone exists only where a pass carries many tokens**, which is why
prefill is free and decode is not — and why batching, below, is the lever that
matters.

## Prefill is free, but not at every ubatch

S18 predicted break-even at ubatch ≈ 218. Measured:

| ubatch | `frac=1.00` | `frac=0.00` | |
|---:|---:|---:|---|
| 128 | 71.4 | 43.6 | −39% |
| 256 | 75.4 | 39.2 | **−48%** |
| 512 | 75.3 | 66.4 | −12% |
| 1024 | 66.8 | 66.0 | **−1%** |
| 2048 | 62.9 | 64.3 | +2% |

**The prediction was wrong: break-even is between 512 and 1024, not 218.** The
"prefill is free" claim holds only at ubatch ≥ 1024 and must be stated with that
condition. Had this been left at the single ubatch=1024 point S27 first measured,
the claim would have been published unqualified and been false for anyone using
a smaller micro-batch.

## Caveats

- FFN only. Attention and SSM are not streamed; see `S30-dense-attention.md` for
  why extending this needs a different design rather than a wider filter.
- Both dense models available here are attention/SSM hybrids, so the FFN share
  (and therefore the saving) is not representative of a plain transformer.
- Verified on llama.cpp `3581ba0cf` and on `master`; output identical on both.

> Reproduce: `research/spikes/s27_dense_streaming/measure.sh`
> Related: `S18-dense-streaming.md`, `S21-dense-ceiling.md`, `S29-dense-tuning.md`
