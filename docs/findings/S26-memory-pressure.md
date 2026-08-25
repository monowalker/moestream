# Finding S26 — the freed memory is real, and about 8 of the 9.8 GiB is usable for free

| | |
|---|---|
| Target | the project's central claim — "runs, and leaves room for everything else" |
| Date | 2026-08-22 |
| Verdict | **the saving holds.** A co-tenant can take ~8 GiB at a 3% cost; past that there is a cliff, and decode goes to +63% |
| Measured on | `research/spikes/s26_memory_pressure/measure.sh`, Ornith-1.0-35B-UD-IQ4_NL, `frac=0.25`, host 30.6 GiB |

## Why this was missing

Two numbers were already solid: plain llama.cpp holds **17.49 GiB** of GTT,
MoEStream holds **7.65 GiB**, so **9.8 GiB is freed**. GTT is a hard driver
allocation, so that difference is genuinely available to other processes.

Then S19 found that 98.7% of decode reads are served from the page cache — which
lives in *the same free memory that was just handed to other applications*. So:

> "9.8 GiB freed" was measured.
> "9.8 GiB freed **and still 58 ms/token**" was not — that was only ever
> measured with the 9.8 GiB sitting idle.

Every published figure for this project was taken on an otherwise empty machine.
The trade the project exists to make had never actually been made.

## Method

A ballast container occupies N GiB and touches every page so it stays resident,
under a hard `--memory` cap so Docker kills the ballast rather than the OOM
killer choosing a victim among the machine's real services. Decode is the median
of 3 × 100 tokens after warm-up. Device reads come from the per-process kernel
counter used in S19 — what genuinely reaches the SSD.

## Result

| ballast | decode | vs idle | page cache | device reads |
|---:|---:|---:|---:|---:|
| 0 GiB | **57.71 ms** | — | 17.2 GiB | 0.0 MiB/tok |
| 4 GiB | 57.92 ms | +0.4% | 17.3 GiB | ~0 |
| **8 GiB** | **59.62 ms** | **+3.3%** | 13.8 GiB | 1.5 MiB/tok |
| 12 GiB | 93.97 ms | **+63%** | 10.1 GiB | 48.4 MiB/tok |
| 16 GiB | 98.08 ms | +70% | 6.6 GiB | 62.7 MiB/tok |

**There is a cliff between 8 and 12 GiB**, and the device-read column says
exactly what it is: the expert set is 14.48 GiB, and once the page cache falls
below that, reads start reaching the SSD. 0 → 1.5 → 48.4 MiB/token is the page
cache failing to hold the working set, precisely as S19 predicts.

## What this means for the claim

**The claim survives, with a number attached to it that it did not have before.**

```
freed by MoEStream            9.8 GiB   (hard GTT, unconditional)
usable at negligible cost   ~ 8   GiB   (+3.3% decode)
usable at 63% slower        ~12   GiB
```

So "leaves room for everything else" is not marketing: a co-tenant really can
take 8 GiB and the model barely notices. What was never said is that the last
~2 GiB of the saving is expensive, and that the boundary is set by the expert
set fitting in whatever page cache is left, not by anything MoEStream controls.

**On a smaller host this arrives sooner.** The cliff is at
`page cache < expert set`. On a 16 GiB machine, Ornith-1.0 would be past it
before any co-tenant started.

## The comparison is not the one the script originally claimed

The script's closing note said plain llama.cpp "cannot be run against a ballast
above ~5 GiB". **That is wrong, and in the direction that flatters MoEStream.**
Plain llama.cpp keeps every weight in GTT and needs no page cache at all, so it
is *insensitive* to a co-tenant — right up until it stops fitting:

| ballast | plain llama.cpp | MoEStream |
|---:|---|---|
| 8 GiB | 17.49 + 8 = 25.5 GiB — fits, ~41.9 ms | 15.7 GiB, 59.6 ms |
| 12 GiB | 29.5 GiB — at the edge of a 30.6 GiB host | 19.7 GiB, 94.0 ms |
| 16 GiB | 33.5 GiB — **does not fit** | 23.7 GiB, 98.1 ms |

So between 0 and ~12 GiB of co-tenant, **plain llama.cpp is both faster and
degrades less**; MoEStream's advantage is that it is still running at 16 GiB and
beyond, where plain llama.cpp is not. That is the same "running versus not
running" argument the README already makes for the three oversized models, and
it is the honest shape of the trade — not "same speed, less memory".

**These crossover rows are arithmetic, not measurements.** Only the MoEStream
column was measured. Running the plain-llama.cpp ballast sweep is the obvious
follow-up and was not done.

## Caveats

- One model on one host. The cliff position is
  `expert set vs page cache`, so it moves with both.
- The ballast is a single process touching anonymous memory. A real co-tenant
  with its own file-backed working set would compete for page cache differently.
- Decode only. Prefill uses the arena and reads far more per pass; its behaviour
  under pressure is unmeasured.

> Reproduce: `research/spikes/s26_memory_pressure/measure.sh`
> Related: `S19-pagecache-share.md`, `RESULTS.md` §2, §12.5
