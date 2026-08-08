# Finding S9 — Where the headroom for prefetching is (an estimate, before implementing)

| | |
|---|---|
| Target | re-evaluating predictive prefetch; how regime-dependent N2/N3's rejection was |
| Date | 2026-08-06 |
| Verdict | **N2/N3's rejection is regime-specific. Under current conditions there is about 2x of headroom** |
| Code | `research/spikes/s9_prefetch_bw/main.c` |
| Measured on | Laguna-S-2.1 (54.7 GiB / 3 shards) / `MOESTREAM_CACHE_FRAC=0.20` |

## Why re-evaluate

`RESULTS.md` §7 concludes that all three prefetch schemes are net-negative. But
that was measured **where I/O is 1% of decode**.

```
                          Ornith-35B        Laguna-S-2.1
decode (with I/O)            54.6 ms           361.8 ms
decode (no I/O, NOOP=3)      54.1 ms            89.4 ms
cost of I/O                  0.53 ms (1%)      272.4 ms (75%)
```

**What there is to hide grew 500x.** With the premise changed, the conclusion
needs re-testing.

## The current breakdown (measured)

From `report()` (Laguna / frac=0.20 / 253 tokens):

```
hit rate 67.53%  (hit 80288 / miss 38612)
I/O 115836 reads, 166383.09 MiB in 57.783 s (3.02 GB/s)
bytes/token 657.64 MiB
remap total 57.89 s (read 57.78 s / upload 0.00 s)   <- 99.8% is waiting on reads
zero-copy 115836 / 115836 (100%)
```

| | Value |
|---|---:|
| decode | 361.8 ms/token |
| of which I/O | 229 ms |
| of which everything else | 133 ms |
| miss transfer | 657.64 MiB/token |
| **effective bandwidth** | **3.02 GB/s** |
| device limit (S2, sequential) | 4.48 GB/s |

That is **458 scattered reads averaging 1.44 MiB per token** (152.6 expert
misses × 3 tensors). Per layer, it repeats a serial I/O 4.87 ms → compute
2.83 ms cycle 47 times.

## Test: does continuous queueing raise bandwidth on scattered reads?

Half the value of prefetching is keeping the SSD busy. But S2's 4.48 GB/s was
measured on **sequential large blocks**, with no guarantee that 1.29 MiB
scattered reads reach the same figure. This was settled before implementing.

Reproducing the real pattern exactly (20 tokens × 47 layers × 10 reads =
11.83 GiB, O_DIRECT):

| Method | Bandwidth |
|---|---:|
| **(A) reproducing the current behaviour** — burst per layer + 2.83 ms compute gap | **2.10 GB/s** |
| (B) removing only the compute gaps | **4.09 GB/s** |
| **(C) continuous queueing** | **4.48 GB/s** |

Effect of thread count on (C):

| Threads | Bandwidth |
|---:|---:|
| 4 | 4.48 GB/s |
| 8 | 4.48 GB/s |
| 16 | 4.46 GB/s |
| 32 | 4.46 GB/s |

**Scattered reads do reach the device limit.** Four threads saturate it; no deep
queue is needed.

### Where the loss is

```
(A) 2.10 -> (B) 4.09 GB/s   filling the compute gaps   1.95x  <- most of it
(B) 4.09 -> (C) 4.48 GB/s   making layer bursts continuous  1.10x
```

**Most of the improvement comes from not leaving the SSD idle during compute.**
No sophisticated continuous scheduler is needed; simply issuing layer L+1's
reads before layer L's compute reaches (B).

## Expected benefit

If the device limit of 4.48 GB/s were reached:

```
I/O    = 657.64 MiB / 4.48 GB/s = 154 ms/token
compute = 133 ms (overlapping the I/O)
decode = max(154, 133) = 154 ms

361.8 -> 154 ms   ~= 2.35x
```

Folding in 81.4% prediction accuracy (N2's measurement), the 18.6% that miss
stall in place:

```
stalls      0.186 × 229 = 42.6 ms
prefetched  0.814 × 154 = 125 ms -> hidden behind 133 ms of compute
decode ~= 133 + 42.6 = 176 ms   ~= 2.06x
```

## Only P2 is usable

| Predictor | Verdict | Reason |
|---|---|---|
| P1 previous-token reuse | **impossible in principle** | reused experts are by definition already cached. It does not come back in a different regime (`RESULTS.md` §7) |
| **P2 layer lookahead `W_{L+1}·h_L`** | **candidate** | 81.4% accurate. It was rejected on cost, and that cost is now relatively smaller |
| P5 cross-layer co-activation | no | 18.5–40.0% accurate |

P2's cost (measured in N2):

| Component | Value | vs current decode |
|---|---:|---:|
| matrix multiply | 3.3 ms/token | 0.9% |
| fetching probs/hidden GPU→CPU | 7.5 ms/token | 2.1% |
| callback scheduler cost | ~22 ms/token | 6.1% |
| total | **~33 ms** | **9%** |

Then, it paid 33 ms out of a 54.6 ms decode to hide 0.5 ms. Now it would pay
33 ms out of 361.8 ms to hide about 190 ms. **The balance has inverted.**

## Unverified, and risks

| # | Item | Impact |
|---|---|---|
| 1 | the spike uses O_DIRECT; the real system goes through page cache. (A) gives 2.10 against a real 3.02 GB/s | the real system already benefits from page cache, so the improvement may be smaller than the spike's 2.13x ratio |
| 2 | the 81.4% accuracy is **from Ornith**; Laguna is unmeasured | lower accuracy means more stalls |
| 3 | pipelining requires making the remap op asynchronous | it currently waits synchronously in `run_reads_parallel`. Risk of re-encountering the graph-ordering problems from N4/S7 |
| 4 | this finding measures Laguna (54.7 GiB, outside the intended range) | on Ornith-class models I/O is 1%, so **prefetching is worth almost nothing**. Do not misapply the conditions |

## Conclusion

**"Where the hit rate is low there is about 2x of headroom for prefetching" is
supported by measurement.**

With conditions:

- **only where I/O dominates decode.** Meaningless on Ornith-class (1% I/O)
- only P2 is usable; P1 is impossible in principle
- most of the improvement comes from not leaving the SSD idle during compute, so
  **the pipelining itself matters more than sophisticated prediction**

> N2/N3's conclusion that predictive prefetch is pointless **is not withdrawn**.
> It is correct for the conditions measured. The error was detaching it from
> those conditions and treating it as a general rule.
