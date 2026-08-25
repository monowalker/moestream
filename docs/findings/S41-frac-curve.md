# S41 — Does `MOESTREAM_CACHE_FRAC=learn` actually pick the fastest slot count?

> **[Corrected 2026-08-24.]** The curve below was measured with a single
> 30-token warm-up, which is not enough for a MoE decode measurement on this
> machine — MoE decode samples spread by ~8% until two full warm-up requests have
> run, where dense samples spread by 0.03%. Re-measured properly on Ornith-1.0
> (200-token generations, two warm-ups, median of three):
>
> | frac | device memory | decode |
> |---|---:|---:|
> | 0.15 | 6.46 GiB | 70.1 ms/tok |
> | 0.25 | 7.93 GiB | 60.6 ms/tok |
> | 0.40 | 10.09 GiB | 57.6 ms/tok |
> | 0.50 | 11.56 GiB | 52.2 ms/tok |
> | 0.60 | 13.03 GiB | 49.6 ms/tok |
> | `MOESTREAM=0` | 17.78 GiB | 42.6 ms/tok |
>
> **The curve is monotonic.** More slots is always faster, approaching the
> resident baseline. The claim below that it "turns over at 0.60" was an artefact
> of the warm-up, and so is the conclusion drawn from it — that a hit-rate proxy
> is structurally blind to a slot count worse than a smaller one. There is no
> such point to be blind to. `learn`'s monotonic assumption is sound.
>
> What survives: the top of the curve is flat in the sense that matters, and
> memory buys speed at a steeply diminishing rate. 0.15 → 0.25 costs 1.5 GiB and
> buys 9.5 ms; 0.50 → 0.60 costs 1.5 GiB and buys 2.6 ms.

*2026-08-23. Ornith-1.0-35B-UD-IQ4_NL, ctx 32768, ubatch 1024, one request,
200-token generations, every point measured in the same configuration.*

## Why ask

`learn` recommends a slot count from measured reuse distances, using a fixed
exchange rate: keep buying slots while a GiB still buys at least
`MOESTREAM_PT_PER_GIB` (2.5) percentage points of hit rate. Hit rate is a proxy.
The question is whether the proxy lands on the same place the clock does.

On this machine `learn` had settled on **frac = 0.50**. Whether that was right
had never been checked against a measured curve.

## The curve

| frac | slots/layer | device memory | generation | vs. the best |
|---|---|---|---|---|
| 0.15 | 38 | 5.97 GiB | 71.6 ms/tok | +29.5% |
| 0.20 | 51 | 6.70 GiB | 65.1 ms/tok | +17.7% |
| 0.25 | 64 | 7.44 GiB | 60.7 ms/tok | +9.8% |
| 0.30 | 77 | 8.17 GiB | 58.9 ms/tok | +6.5% |
| 0.40 | 102 | 9.59 GiB | 56.5 ms/tok | +2.2% |
| **0.50** | **128** | **11.06 GiB** | **55.3 ms/tok** | — |
| 0.60 | 154 | 12.54 GiB | 56.4 ms/tok | +2.0% |

Output was byte-identical at every point.

## What it says

**`learn` picked the measured optimum.** 0.50 is the fastest point on the curve,
and the hit-rate proxy found it without ever looking at a clock.

**The optimum is not the interesting part of the curve.** From 0.40 to 0.60 the
whole range is within 2.2% of the best, across 3 GiB of memory. Meanwhile 0.25
gives up 9.8% of the speed to free 3.6 GiB. Anyone who wants the memory back can
have most of it for very little, and the flat top means `learn` being slightly
off would not matter much either.

**The curve turns over.** 0.60 is slower than 0.50, at 1.5 GiB more. Past some
point a larger slab stops paying for itself — the misses it removes were being
served from page cache anyway (S19), while the slab itself competes for the same
memory. This is the one place the hit-rate proxy is structurally blind: hit rate
rises monotonically with slots, so nothing in the rule can represent a slot count
that is worse than a smaller one. Here the memory cap happened to stop it in the
right place. On a machine with more memory it would not, and `learn` would
recommend past the peak.

## What was not changed

Nothing. The recommendation is right on this machine and the failure mode above
is unobserved, not merely unmeasured — but it is a known blind spot, and a
future version that keeps a timing sample per frac would close it. Recorded here
so that the next person to see `learn` recommend a large slab has the curve to
check it against, and knows to check.
