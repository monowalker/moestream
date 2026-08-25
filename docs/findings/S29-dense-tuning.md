# Finding S29 — tuning dense streaming: batching beats everything, and two predictions were wrong

| | |
|---|---|
| Target | seven open questions about S27's dense streaming, all environment knobs |
| Date | 2026-08-23 |
| Verdict | **`N_PARALLEL=4` reaches 190.8 ms/token in 7.29 GiB — faster and lighter than plain llama.cpp** |
| Measured on | `research/spikes/s29_dense_tuning/`, Qwen3.8-27B-IQ4_NL |

## The headline

> **[Corrected 2026-08-23]** This section originally read *"half the memory and
> slightly faster"*, comparing streaming at four concurrent requests against
> plain llama.cpp at **one**. Measured at four on both sides (finding S34), plain
> reaches 99.1 ms/token and streaming 194.1 — **streaming is 1.96x slower, not
> faster.** Batching helps both; it helps streaming more, which is the real
> result and the only one this section should have claimed.

| both sides at K=4 | resident | aggregate |
|---|---:|---:|
| plain llama.cpp | 15.98 GiB | **99.1 ms/tok** |
| dense `frac=0.00` | 7.29 GiB | 194.1 ms/tok |

Batching does move the exchange rate a long way — 2.86x at one request down to
1.96x at four — A dense
pass reads the same bytes whatever the token count, so four concurrent sequences
divide the streamed bytes per token by four. Nothing is discarded: unlike
speculation, every token is real.

| K | decode (aggregate) |
|---:|---:|
| 1 | 729.2 ms/tok |
| 2 | 469.3 |
| 4 | **190.8** |

This does not make one sequence faster. It makes the machine produce more tokens
for the same reads, which is exactly the quantity streaming is short of.

## MTP helps, and stacks badly with batching

| | no MTP | + MTP |
|---|---:|---:|
| `frac=1.00` (nothing streamed) | 212.4 ms | 135.5 (1.57x) |
| `frac=0.40` | 516.4 | 256.6 (2.01x) |
| `frac=0.00` | 696.1 | **329.6 (2.11x)** |

**The speed-up grows with how much is streamed** — 1.57x when nothing streams,
2.11x when everything does — while acceptance stays 0.571 throughout. The extra
is precisely the I/O being amortised, which is the prediction S28 made.

> **[Corrected 2026-08-23]** That MTP narrows the gap is real; the size quoted
> elsewhere was not. Comparing streaming-with-MTP against plain-*without* gave
> "1.56x". With MTP on both sides it is **2.27x** (151.9 vs 344.3 ms), against
> 2.86x with MTP on neither. See S34.

But combined with batching it reverses:

| | no MTP | + MTP |
|---|---:|---:|
| `N_PARALLEL=2` | 469.3 | 317.2 |
| `N_PARALLEL=4` | **190.8** | 238.0 |

**Both are the same lever — tokens per pass.** Once batching supplies enough,
MTP only adds the 43% of drafted tokens that get rejected. `n_max` sweeps 1/3/5/7
to 394.7 / 329.6 / 302.3 / 291.9 ms, so the shipped default of 3 is not optimal
either, but none of it is worth having alongside `N_PARALLEL=4`.

## I/O threads were tuned for the wrong shape

`dense_read_range` inherited `g_nthreads` from the `[io]` auto-tuner, which
optimises against **scattered 1.4 MiB expert reads**. A dense layer is a single
143 MiB contiguous read.

| threads | decode |
|---:|---:|
| 1 | 855.8 ms |
| 2 | 776.4 |
| **4** | **656.8** |
| 8 | 674.4 |
| 16 | 676.3 |

**−23% from the inherited value.** Two read patterns sharing one tuned constant
was a real cost, and nothing in the code said so.

## More arena buffers are worse

| buffers | decode |
|---:|---:|
| 2 | 690.8 ms |
| 3 | 704.7 |
| 4 | 721.4 |

Deeper lookahead cannot help when a layer's read (15.6 ms) outruns a layer's
compute (3.3 ms) by 4.7x, and the buffers cost memory that was the point of
streaming. Predicted in S27, confirmed here.

## Two predictions that were wrong

**S18's break-even ubatch.** Predicted 218; measured between 512 and 1024.

| ubatch | `frac=1.00` | `frac=0.00` |
|---:|---:|---:|
| 128 | 71.4 | 43.6 (−39%) |
| 256 | 75.4 | 39.2 (**−48%**) |
| 512 | 75.3 | 66.4 (−12%) |
| 1024 | 66.8 | 66.0 (−1%) |
| 2048 | 62.9 | 64.3 (+2%) |

"Prefill is free" is true only at ubatch ≥ 1024. S27 measured one point and the
claim was about to be published without the condition.

**Batching and speculation composing.** Expected to multiply; they substitute.

## Perplexity and the stock runtime

```
dense_frac 1.00 / 0.40 / 0.00   PPL = 4.2000 / 4.2000 / 4.2000
```

Identical to four decimals — the accuracy claim rests on this, not only on
matching greedy output.

```
stock llama.cpp master   16.69 GiB   219.2 ms/tok
MoEStream (3581ba0cf)    15.58 GiB   211.0 ms/tok
```

Same model, streaming off: the pinned commit is not behind master.

## Caveat

Aggregate throughput at `N_PARALLEL=K` is measured by issuing K concurrent
requests and dividing wall time by total tokens. Per-request latency is
unchanged or worse; this is a throughput result, not a latency one.

> Reproduce: `research/spikes/s29_dense_tuning/measure.sh` and `part2.sh`
> Related: `S27-dense-streaming-impl.md`, `S28` (in `S24-skiprank-verdict.md` family), `S18-dense-streaming.md`
