# Finding S37/S38 — the free zone does exist, above a batch size, and only where the read set is constant

| | |
|---|---|
| Target | S27's conclusion that dense decode has no free zone |
| Date | 2026-08-23 |
| Verdict | **S27 was right about K=1 and wrong as a general claim.** Dense streaming closes from 3.15x to **1.20x** between one and sixteen concurrent requests. MoE improves too, but far less — 1.60x to 1.32x |
| Measured on | `research/spikes/s37_batch_freezone/`, `s38_moe_batch/`, Qwen3.8-27B-IQ4_NL and Ornith-1.0-35B-UD-IQ4_NL |

## What S27 got wrong

S27 measured no free zone and explained it:

```
one layer's read    143 MiB / 9.6 GB/s = 15.6 ms
one layer's compute 211 ms / 65 layers =  3.3 ms      -> reads outrun compute 4.7x
```

Correct, and correct only for one token per pass. **Compute per pass scales with
the tokens in it; the reads do not.** At K sequences the window is ~3.3K ms
against the same 15.6 ms, so the ratio crosses 1 somewhere around K = 5.

## Dense: it converges

| K | plain llama.cpp | streaming `frac=0.00` | cost |
|---:|---:|---:|---:|
| 1 | 231.1 ms/tok | 727.2 | **3.15x** |
| 2 | 144.7 | 449.5 | 3.11x |
| 4 | 100.6 | 191.0 | 1.90x |
| 8 | 82.7 | 134.0 | 1.62x |
| **16** | **76.3** | **91.4** | **1.20x** |

Memory stays where it was: 17.47 → 8.78 GiB at K=16, **−50%**.

> **With enough concurrent requests, dense streaming is close to free.**
> At one request it costs 3x. That is a difference between deployments, not a
> difference between models, and nothing in the earlier write-up said so.

## MoE: it improves much less, for the reason that keeps recurring

| K | plain llama.cpp | expert streaming | cost |
|---:|---:|---:|---:|
| 1 | 50.1 ms/tok | 80.0 | **1.60x** |
| 4 | 32.4 | 50.7 | 1.56x |
| 8 | 29.4 | 44.4 | 1.51x |
| 16 | 21.0 | 27.8 | **1.32x** |

A MoE pass reads the *union* of what its tokens want:

```
union(K) = 256 x (1 - (1 - 8/256)^K)      per token: union(K)/K
K=1   8.0    8.0        K=8   57.4   7.2
K=4  30.5    7.6        K=16 101.6   6.4
```

Reads per token fall from 8.0 to 6.4 across the whole sweep, against dense's
`total/K`. So batching amortises almost nothing here — exactly the mechanism
that made MTP lose on MoE and win on dense (S31/S32).

> **One property decides all of it: does a pass's read volume grow with the
> number of tokens in it?** Dense: no, so batching and speculation both pay.
> MoE: yes, so neither does much.

> **[Qualified 2026-08-24 — see [S45](S45-speculation-and-batching.md).]** The law
> holds only while the pass has room. Batching and speculation are both ways to
> fill it, and they do not add: on a streamed dense model speculation is worth
> 2.08x at one request and **costs 34%** at four, because the batch has already
> taken what there was to take. "Does this pay?" has to become "does this pay
> *given what is already in the pass?*" 

**A prediction that was wrong.** This was expected to be flat; 1.60 → 1.32x is a
real improvement. `union(K)/K` falling from 8.0 to 6.4 accounts for part of it;
the expert cache absorbing repeat references across concurrent sequences is the
likely rest, and was not measured.

## Does dense overtake MoE?

On the ratio, at K=16, yes: 1.20x against 1.32x. **On the thing that matters, no.**

| K=16 | memory | absolute speed |
|---|---:|---:|
| MoE, streaming | 8.04 GiB (−55%) | **27.8 ms/tok** |
| dense, streaming | 8.78 GiB (−50%) | 91.4 ms/tok |

MoE is **3.3x faster in absolute terms** at comparable memory. The better ratio
belongs to the model family with the worse starting point. Reporting the ratio
alone would have inverted the conclusion.

## No slot exhaustion, and the output holds

`union(16) = 101.6` against 64 slots at `frac=0.25` looked like guaranteed slot
exhaustion. **It did not occur** — zero exhaustion lines at every K — because a
batch past the arena threshold takes the arena path, which holds a whole layer.

Correctness was checked at the batch sizes the timings were taken at
(`s39_batch_correctness`), plain against streaming **at the same K**, since batch
composition changes matmul tiling and therefore the output even in plain
llama.cpp (§10.6):

| | result |
|---|---|
| dense, K=4 | **byte-identical**, 1881 bytes over 4 prompts |
| dense, K=16 | **byte-identical**, 1907 bytes |
| MoE, all K | identical, zero exhaustion |

That check was added after the fact: S37 measured only time. Two speed numbers
in this project have already turned out to come from broken configurations
(§10.8, and the 218 ms dense run that emitted garbage), so a timing without a
correctness check beside it is not a result.

## What it changes

`MOESTREAM_DENSE_FRAC=auto` streams the least that fits, which is right when
concurrency is unknown. It is conservative for a server that batches: at K=16
the memory could come down further at almost no cost. Nothing currently tells
the runtime how concurrent its traffic will be, and `N_PARALLEL` at start-up is
a ceiling rather than a measurement.

> Reproduce: `research/spikes/s37_batch_freezone/measure.sh`,
> `s38_moe_batch/measure.sh`, `s39_batch_correctness/measure.sh`
> Related: `S27-dense-streaming-impl.md`, `S34-like-for-like.md`, `S31-moe-mtp.md`
