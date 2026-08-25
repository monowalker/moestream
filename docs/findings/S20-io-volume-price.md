# Finding S20 — what a byte of decode I/O is actually worth, and why layer index is not a proxy for volume

| | |
|---|---|
| Target | the linearity assumption behind every "reduce bytes by X%" estimate in `V3-idea-analysis.md` |
| Date | 2026-08-22 |
| Verdict | **decode I/O costs 11.82 ms/token, and the return on removing it is strongly non-linear.** Removing 22% of reads buys 10% of that cost; removing 63% buys 98% |
| Measured on | `research/spikes/s20_io_volume_curve/`, Ornith-1.0-35B-UD-IQ4_NL, `frac=0.25`, `ub=1024`, `ctx=32768` |

## Why

S17 and B1 were both costed the same way: multiply the share of read bytes
removed by the measured I/O share of decode. That assumes decode time is linear
in read volume, and nobody had checked.

`MOESTREAM_DROP_FROM=N` is an existing diagnostic — at layers ≥ N a miss is
pointed at the null expert instead of being fetched — so the sweep needs no new
code. Output below the layer count is deliberately corrupt; this measures speed
only.

## The bounds reproduce

| configuration | decode | docs (§10.12) |
|---|---:|---:|
| `MOESTREAM=0`, no streaming | **41.87 ms** | 41.96 |
| `NOOP=3`, cache lookup with zero I/O | **46.23 ms** | 46.30 / 46.46 |
| `drop_from=0`, every miss dropped | 46.33 ms | — |
| normal, full reads | **58.15 ms** | 58.75 |

`NOOP=3` and `drop_from=0` agree to 0.1 ms by two independent routes, which is
a good control. So on this machine today:

```
sync + cache management = 46.23 - 41.87 =  4.36 ms   (docs: 4.50)
I/O                     = 58.15 - 46.33 = 11.82 ms   (docs: 12.45)
```

## Layer index is not read volume

The first reading of the sweep assumed `drop_from=30` removes 25% of the reads,
because it removes 25% of the layers. **It does not.** Replaying the traces
(`layer_miss_share.py`) gives the real mapping, and the distribution is
extremely skewed:

```
miss rate by layer (en, deciles):  52% 28% 11% 16% 14% 11% 15% 16% 19% 14%
```

**Layer 0 misses on more than half its lookups** while the rest of the model
sits near 11–19%. Early layers route far less repetitively, so the first tenth
of the model carries a disproportionate share of all reads. Corrected:

| `drop_from` | layers removed | **reads removed** (en / ja / code) |
|---:|---:|---:|
| 30 | 25% | **21.9 / 18.2 / 19.5%** |
| 20 | 50% | **42.6 / 39.2 / 39.7%** |
| 10 | 75% | **62.9 / 60.7 / 60.1%** |
| 0 | 100% | 100% |

## The price curve

Using the en mapping against the measured timings:

| reads removed | decode | ms saved | share of the 11.82 ms |
|---:|---:|---:|---:|
| 0% | 58.15 | — | — |
| **21.9%** | 56.94 | 1.21 | **10.2%** |
| **42.6%** | 53.74 | 4.41 | **37.3%** |
| **62.9%** | 46.58 | 11.57 | **97.9%** |
| 100% | 46.33 | 11.82 | 100% |

**Strongly convex, with a knee between 43% and 63%.** The first fifth of the
reads removed is nearly worthless; past the knee almost the entire I/O cost
disappears, and the last 37% of reads is worth 2% of the cost.

The most likely mechanism is the one §10.12 already identified when its two
instruments disagreed by 1.6x: the cost of a read is not only its transfer but
the cache and memory pressure it leaves behind for the compute that follows.
Halve the traffic and the survivors get cheaper as well as fewer. S19 makes this
more plausible still — on this model 98.7% of reads are served from the page
cache, so what is being relieved is memory-subsystem pressure, not disk queueing.

## What this does to the estimates it was built to check

| proposal | reads removed | linear estimate | **from this curve** |
|---|---:|---:|---:|
| S17 `SKIP_RANK=7` | 19.6% | −2.3 ms | **≈ −1.1 ms (−1.9%)** |
| S17 `SKIP_RANK=6` | 36.8% | −4.3 ms | **≈ −3.4 ms (−5.9%)** |
| B1 shape (b), half the bytes | ~50% | −5.9 ms | **≈ −6.5 ms (−11%)** |
| B1 shape (a), ~90% fewer bytes | ~90% | −10.6 ms | **≈ −11.8 ms (−20%)** |

**Small byte reductions are worth less than the linear model said; large ones
are worth slightly more.** That is bad news for S17 at conservative thresholds
and good news for B1, and it sharpens the ranking between them: this curve says
there is little point shaving 20% off the reads, and a large payoff for
anything that clears the knee.

## Caveats

- The byte mapping comes from the recorded bench traces (en/ja/code corpora),
  while the timings were taken on a different prompt. The per-layer miss
  distribution is consistent across all three traces, so the shape is
  trustworthy, but the exact percentages are not from the same workload.
- `drop_from` removes reads **by layer**, so this curve conflates "which bytes"
  with "how many". The knee could in principle be a property of *which* layers
  were spared rather than of volume. `SKIP_RANK` (finding S24) removes bytes on
  a different axis entirely and is the cleaner test of the same question.
- Four points. The knee's location is bracketed to somewhere in 43–63%, not
  located.
- A first attempt to read the runtime's own `bytes/token` counter per
  configuration failed: `SIGUSR1` triggers `mrc_report()`, and `bytes/token` is
  printed by `report()`, which only runs at exit. The re-run was abandoned
  because a concurrent build had contaminated its timings, and the offline
  trace mapping answers the same question without the GPU.

> Reproduce: `research/spikes/s20_io_volume_curve/measure.sh` then
> `layer_miss_share.py`.
> Related: `S17-miss-weight-skip.md`, `S19-pagecache-share.md`,
> `S24-skiprank.md`, `RESULTS.md` §10.12
