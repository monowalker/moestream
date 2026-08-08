# Finding S11 — Prefetching via fadvise does not work (a negative result)

| | |
|---|---|
| Target | implementing the P2 predictor plus prefetch, following findings S9/S10 |
| Date | 2026-08-06 |
| Verdict | **not adopted. 19% worse when measured. Even with free prediction, only 3.6% better** |
| Measured on | Laguna-S-2.1 / `MOESTREAM_CACHE_FRAC=0.20` / 253 tokens |

## What was done

Implemented after S9 established "where I/O is 75% of decode there is about 2x
of headroom for prefetching" and S10 established "the P2 predictor is 71%
accurate".

```
in layer L's remap op:
  1. the usual work (read the experts that missed)
  2. predict layer L+1's experts from the top-k of W_{L+1} · h_L
  3. issue posix_fadvise(WILLNEED) for those not in the slab
the intent being that by the time layer L+1's remap runs, they are in page cache
```

**Writing into the slab from a background thread was deliberately avoided.**
`ExpertCache` assumes a single thread, and breaking its refcount invariants
would produce the same class of quietly corrupting bug as N4 and S7. `fadvise`
touches no cache structure at all.

## Result

| | prefetch off | prefetch on |
|---|---:|---:|
| decode | **361.8 ms/tok** | **429.3 ms/tok** |
| I/O time (253 tok) | 57.783 s | 54.498 s |
| effective bandwidth | 3.02 GB/s | 3.20 GB/s |

```
I/O saved        13.0 ms/token   (+6% bandwidth)
prediction cost -69.4 ms/token
net             -56.4 ms/token   -> 19% worse
```

**Even with free prediction the improvement would be 3.6%.**

## Two errors

### Error 1: the prediction cost was underestimated 12x

Estimated 5.8 ms, measured 69.4 ms/token.

Only the arithmetic was counted (256 experts × 3072 dims × 47 layers = 37M MAC).
**Reading the 141 MiB router matrix every token** was not. That is
**memory-bandwidth bound**, not compute bound (2 GB/s effective).

Parallelising would improve it (9–15 ms with 8 threads), but that is moot in
light of error 2.

### Error 2: fadvise does not produce continuous queueing

The S9 spike showed "burst + gap 2.10 GB/s → continuous queueing 4.48 GB/s", but
that figure was for **maintaining your own queue**. `posix_fadvise(WILLNEED)` is
merely a hint to the kernel, and measured +6%.

Candidate causes, none of which `fadvise` can control:

- fadvise is a hint; neither its timing nor its priority is guaranteed
- the prefetch window is one layer, about 2.8 ms
- page cache is about 15 GiB against a 54.7 GiB model, so prefetched pages can
  be evicted before use

> **S9's "2.13x of headroom" was not wrong.** The error was believing `fadvise`
> could capture it. The headroom was specifically "maintain your own queue",
> which cannot be delegated to the kernel.

## The remaining option, and its ceiling

Prefetching into the slab from a dedicated thread (S9's "option C") is the only
route to S9's 2.13x — but **the ceiling is not 2.13x**.

At 71% accuracy, the 29% of wasted transfers **compete for the same SSD
bandwidth**:

```
transfer needed          658 MiB/token
issued at 71% accuracy   658 / 0.71 = 926 MiB/token   (40% more)

assuming perfect overlap:
  prefetched  926 MiB / 4.48 GB/s = 207 ms   (exceeds 133 ms of compute, so not hidden)
  missed      0.29 × 658 MiB      =  43 ms   (stalls in place)
  decode ~ 250 ms

  361.8 -> 250 ms  ~= 1.45x
```

**S9's 2.13x assumed 100% accuracy; the real ceiling is around 1.45x.**

## Lesson

> **"Headroom" measured in a spike means nothing without the means to capture
> it.**

S9 correctly measured that continuous queueing reaches 4.48 GB/s. But that is
the performance of **an implementation that maintains its own queue**, not of
**prefetching as a concept**. The moment the implementation was made safe
(fadvise), the measured headroom was gone.

Proceeding in stages was right. Stage 1 (the decision logic) worked, stage 2
produced this result, and **we stopped before the dangerous option C.**
