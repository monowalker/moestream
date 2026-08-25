# Finding S17 — Skipping low-weight experts is 3.5x cheaper if you only skip misses

| | |
|---|---|
| Target | DESIGN.md §11.6 `soft` mode, re-evaluated against the post-inversion bottleneck |
| Date | 2026-08-22 |
| Verdict | **worth implementing, pending PPL.** 10–12% of decode read bytes for 0.6–0.8% of router weight mass |
| Measured on | `research/spikes/s17_miss_weight/analyze.py` against `research/bench/{en,ja,code}.trace` (Ornith-1.0-35B, 20,000 tokens each) |

## Why this was reopened

Finding M0-2 measured the router weight distribution and concluded that §11.6's
weight-threshold skip was near-inert: `τ=0.02` touches 0.1% of selections, and
even `turbo`'s `τ=0.06` costs 2.1–2.9% of the weight mass. `soft` mode was never
implemented, and `RESULTS.md` §14 still lists it as "room for redesign".

**That evaluation applied the threshold to every selection.** It predates
§10.12, where the dominant decode cost turned out to be I/O — and I/O is paid
**only on a cache miss**. A hit costs nothing, so there is no reason to skip one.
The quantity that matters is therefore not `P(w < τ)` but the joint

```
P(w < τ  AND  the selection missed the cache)
```

and the two are not the same, because rarely-chosen experts are both
lower-weighted *and* less likely to be resident.

## Method

Per-layer independent LRU replayed over the recorded traces, which yields the
exact miss stream. Every miss costs the same 1.422 MiB (gate+up+down), so the
share of misses removed **is** the share of read bytes removed. Weight mass is
normalised per token and layer, so "mass lost" is directly comparable to M0-2's
figures.

LRU rather than S3-FIFO: it is what `analyze_trace.py` already uses, and it
understates the hit rate by ~8 points (§10.15), which makes every number below
**conservative** — a better policy leaves fewer misses, and the exchange rate
improves as the hit rate rises (see the frac=0.40 rows).

## Result

Misses are systematically lower-weighted than hits, in every domain:

| trace | frac | hit rate | mean weight of a hit | mean weight of a miss | ratio |
|---|---:|---:|---:|---:|---:|
| en | 0.25 | 81.55% | 0.1290 | 0.1074 | 0.83x |
| ja | 0.25 | 83.91% | 0.1281 | 0.1059 | 0.82x |
| code | 0.25 | 80.47% | 0.1281 | 0.1045 | 0.80x |

At `frac=0.25` (64 slots, the shipped default):

| τ | bytes saved (en / ja / code) | weight mass lost | bytes per unit mass |
|---:|---:|---:|---:|
| 0.02 | 0.36 / — / 0.31% | 0.008% | 44–47x |
| 0.06 | **11.9 / 10.0 / 10.4%** | **0.62–0.82%** | **13.6–16.2x** |
| 0.08 | 32.2 / 31.2 / 29.9% | 2.56–2.95% | 10.2–12.2x |
| 0.10 | 56.3% (en) | 6.15% (en) | 9.2x |

Against the blanket rule M0-2 evaluated (skip regardless of residency), at the
same τ and therefore the same skipped-expert set:

| τ=0.06 | bytes saved | mass lost | bytes per unit mass |
|---|---:|---:|---:|
| blanket (M0-2) | 10.0–11.9% | 2.12–2.86% | 3.5–4.7x |
| **miss-only** | 10.0–11.9% | **0.62–0.82%** | **13.6–16.2x** |

**Identical byte saving, one third to one quarter of the quality cost.** The
skipped hits in the blanket rule cost weight mass and save nothing, because
their bytes were already resident.

The exchange rate improves as the cache gets better, which is the right
direction — at `frac=0.40` it reaches 25.7–34.4x at τ=0.06.

## Two reasons the quality cost is an upper bound, not an estimate

1. **Renormalisation.** Dropping expert *i* and renormalising leaves
   `Σ_{j≠i} w_j/(1−w_i)·E_j(x)`. The error against the original is
   `w_i·(E_i(x) − <E_j(x)>)`, not `w_i·E_i(x)`. Experts are correlated, so the
   realised error is strictly smaller than the mass figure.
2. **The shared expert always runs.** On Ornith it contributes on every token
   regardless, as M0-2 already noted.

Neither is a substitute for measuring perplexity, which is the next step and is
**mandatory before this is enabled by default** (QR-2: PPL degradation ≤ 1%).

## What this is worth end to end

Read bytes are not the whole of decode. Applying the τ=0.06 saving to the
measured I/O share:

| model | I/O share of decode | 11% of it | end-to-end |
|---|---:|---:|---:|
| Ornith-35B (frac 0.25) | 12.45 ms of 58.75 (21%) | 1.4 ms | **−2.3%** |
| Laguna-S-2.1 (frac 0.15) | 63.7% | ~7% | **−7%** |

At τ=0.08 the Laguna figure is roughly −20%, at a mass cost of 2.6–3.0% that
almost certainly needs to be an opt-in mode rather than a default.

**This is not a headline win on Ornith.** It matters on the models where I/O
dominates, which is exactly the class the project exists for, and it costs no
memory at all — the one currency this system is short of.

## Where it should attach

`remap_exec` already has the hook: the branch at `il >= g_drop_from` maps a miss
to the zero slot without fetching (the `MOESTREAM_DROP_FROM` diagnostic). The
production form is the same edit with a weight predicate instead of a layer
predicate, plus renormalisation of the surviving weights — which the current
code does not do, because `MOESTREAM_DROP_FROM` is a diagnostic and does not
care.

The router weights are **not currently visible** to the remap op; only the ids
tensor is passed. Wiring them in is the real cost of this change, and it is the
same plumbing `soft` mode would need anyway.

## Open

- perplexity at τ ∈ {0.04, 0.06, 0.08}, with and without renormalisation
- whether the deadline condition from §11.6 (skip only when the read would
  actually stall) beats a fixed τ. It should, on the models where the window
  measurement in §10.14 showed reads outrunning compute 82% of the time
- re-run against S3-FIFO rather than LRU, to confirm the conservative direction

## Provenance of the traces, and why the live model is a harder case

`research/bench/*.trace` was recorded on **Ornith-1.0-35B-UD-IQ4_NL**
(40 layers, 1.448 MiB/expert). The instance running on this machine is
**Ornith-1.5-35B-Q4_K_M** (41 layers, 1.819 MiB/expert, 18.65 GiB of experts),
and its live S3-FIFO hit rate at `frac=0.40` is **84.82%** over 34,002 tokens —
below the 91.05% LRU figure this spike got at the same frac on the bench trace.

A lower hit rate means more misses, so the **absolute** byte saving on the live
model is larger than the table above; the exchange rate sits between this
document's `frac=0.25` and `frac=0.40` rows, so roughly 12–13% of misses at
τ=0.06. Against ~90.8 MiB/token of reads at a measured 3.72 GB/s, that is about
3 ms of a 98–125 ms token.

Re-tracing on Ornith-1.5 is cheap (`MOESTREAM_ORDER_TRACE`) and should be done
before any implementation, because the router weight distribution is a property
of the trained model, not of the runtime, and 1.0 and 1.5 are different
trainings.

> Reproduce: `python3 research/spikes/s17_miss_weight/analyze.py` (needs numpy).
> Related: `M0-2-expert-distribution.md`, `V3-idea-analysis.md`,
> `RESULTS.md` §10.12, §10.15, §14.
