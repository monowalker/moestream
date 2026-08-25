# Finding S34 — the like-for-like matrix, and three reported numbers that were not

| | |
|---|---|
| Target | every comparison in this project's dense reporting, measured with the baseline in the same configuration |
| Date | 2026-08-23 |
| Verdict | **dense streaming costs 1.94–2.86x depending on configuration, never less.** Three previously reported figures compared against a baseline left in a different configuration, and every one of them flattered streaming |
| Measured on | `research/spikes/s34_like_for_like/measure.sh`, Qwen3.8-27B-IQ4_NL, ctx 16384 |

## Why this exists

Three numbers were reported during this work that should not have been:

| reported | what it compared | honest figure |
|---|---|---|
| "N_PARALLEL=4 is *faster* than plain llama.cpp" | streaming at K=4 against plain at **K=1** | plain at K=4 is **1.96x faster** than streaming at K=4 |
| the same claim again, in a summary table after being corrected once | — | — |
| "MTP brings the cost down to **1.56x**" | streaming **with** MTP against plain **without** it | **2.27x** with MTP on both sides |

All three moved in the same direction. That is not sampling error, it is a habit:
**the improvement was applied to one side and the baseline was left alone.** The
fix has to be structural, so this spike measures the baseline in every
configuration the streamed side is measured in and prints them adjacent.

## The matrix

`aggregate` is wall time over total tokens across K sequences (throughput).
`per-seq` is wall time over one sequence's tokens (latency).

| | plain llama.cpp | streaming `frac=0.00` | memory | cost |
|---|---:|---:|---:|---:|
| K=1, no MTP | 15.58 GiB / **256.8 ms** | 6.90 GiB / 735.2 ms | −56% | **2.86x** |
| K=1, MTP | 16.63 GiB / **151.9 ms** | 7.82 GiB / 344.3 ms | −53% | **2.27x** |
| **K=4, no MTP** | 15.98 GiB / **99.1 ms** | 7.29 GiB / **194.1 ms** | **−54%** | **1.96x** |
| K=4, MTP | 18.37 GiB / 125.7 ms | 9.55 GiB / 243.3 ms | −48% | 1.94x |

Per-sequence latency, which throughput figures hide:

| | plain | streaming |
|---|---:|---:|
| K=1, no MTP | 256.8 ms | 735.2 |
| K=4, no MTP | 396.4 ms | 776.5 |

**Batching does not make a sequence faster.** It makes the machine emit more
tokens for the same reads, and each individual reply gets slower.

## What the matrix says

**1. The exchange rate is 1.94–2.86x, and it never inverts.** Every technique
that improves streaming improves the baseline too. Batching moves streaming from
2.86x to 1.96x, but only because plain llama.cpp gains less from it — plain is
compute-bound at K=1 already, streaming is read-bound and has more slack.

**2. The best rate is K=4 without MTP**: −54% of memory for 1.96x. Adding MTP on
top costs both sides (99.1 → 125.7 plain, 194.1 → 243.3 streamed) — the same
"two ways of putting more tokens in a pass are substitutes, not complements"
that S29 found for streaming alone. It holds for the baseline as well.

**3. MTP's honest contribution is smaller than reported.** With MTP on both
sides the cost is 2.27x, against 2.86x without. Real, and a third of what
"1.56x" implied.

## The correct one-line summary

> **Dense streaming halves the memory and costs roughly twice the time.**
> Batching and speculation change the absolute speeds a great deal and the
> ratio very little.

## Caveats

- One model. Ratios on the plain transformer (gemma-4-31B) were only measured at
  K=1 without MTP, where the cost is 2.96x — consistent, but the matrix was not
  repeated there.
- `n_predict=40` per request. Short generations weight the per-request overhead
  more heavily than a long chat would.
- K=4 with MTP peaks at 18.37 GiB for the baseline, close enough to this
  machine's 23.5 GiB GTT limit that it is worth noting the measurement was not
  memory-starved but was not far off either.

> Reproduce: `research/spikes/s34_like_for_like/measure.sh`
> Related: `S27-dense-streaming-impl.md`, `S29-dense-tuning.md`, `S28`/`S31`
