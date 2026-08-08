# Finding N2 — Parallel I/O works; all three prefetch schemes failed

| | |
|---|---|
| Target | DESIGN.md §22 (prefetch) / §23 (asynchronous I/O) |
| Date | 2026-08-03 |
| Setup | patched llama.cpp + MoEStream, Radeon 780M (Vulkan), Crucial P310 |
| Reproduce | `MOESTREAM=1 MOESTREAM_IO_THREADS=8 ms_generate ...` |

## Results (steady state, second half of a 150-token generation, 3 runs each)

| Configuration | ms/token | tok/s | vs baseline |
|---|---:|---:|---:|
| baseline (all resident) | 42.65 ± 0.04 | 23.44 | 100% |
| **MoEStream 38% (final)** | **66.0 ± 3.0** | **15.2** | **65%** |
| MoEStream 38% (before, sequential I/O) | 85.64 | 11.68 | 50% |

**Improvement: 85.64 → 66.0 ms/token (−23%), 11.68 → 15.2 tok/s (+30%)**

## Decomposing the bottleneck

Measured by disabling features stage by stage.

| Stage | ms/token | Increment |
|---|---:|---:|
| (1) baseline (no callback) | 42.29 | — |
| (2) install the eval callback only | 46.04 | +3.75 |
| (3) + round-tripping ids to the GPU (get/set) | 48.28 | +2.24 |
| (4) + cache lookup (no I/O) | 48.92 | +0.64 |
| (5) + real I/O (sequential) | 85.64 | **+36.7** |
| (6) + parallel I/O (8 threads) | 66.0 | −19.6 |

**The instrumentation costs only +6.6 ms in total; I/O dominates.**
(The initial guess that instrumentation was the main cost was wrong — the result
of concluding before separating the stages.)

## ★ What worked: batching intra-layer misses and reading them in parallel

Changed to collect a layer's misses and read them in parallel.

```
I/O threads = 1 : 85.64 ms/token
I/O threads = 8 : 66.0  ms/token   (−23%)
```

Consistent with S2's measurement that bandwidth saturates at QD=8 (4.42 GB/s).

**An implementation lesson**: the first version used a persistent thread pool
with a condition variable and deadlocked (CPU 0%, all 7 threads in state S).
Static analysis did not find the cause, so it was replaced with **a simple
`std::thread` per batch**, which resolved it. A read takes about 1 ms and thread
creation about 20 µs; there was no reason to bring in complex synchronization.

## ★ What failed: prefetching (all three schemes net-negative)

| Predictor | Accuracy | Best ms/token | Verdict |
|---|---:|---:|---|
| none (demand only) | — | **66.0** | ✅ best |
| P1 previous-token reuse (§22.3) | — (zero issued) | 62–66 | ❌ ineffective |
| P2 layer lookahead (§22.6) | **81.4%** | 97.0 | ❌ slower |
| P5 cross-layer co-activation (§22.4) | 18.5–40.0% | 69.9 | ❌ not accurate enough |

### Why P1 was ineffective

**Whatever P1 predicts is already in the cache.** An expert used on the previous
token is obviously still among the 97 slots, so it only "hits things that would
hit anyway" and produces nothing to prefetch (zero issued). The real misses are
the 62% newly selected, which P1 cannot predict.

> M0-2's measured "38% previous-token reuse" **has no value as a predictor when
> the cache is comfortably larger than the working set.** §22.3's description of
> it as "the cheapest and earliest predictor" is withdrawn.

### P2 is accurate, but fetching the hidden state is fatal

Capture the router weights W and hidden state h from the ancestors of
`ffn_moe_probs` and predict from the top-k of `W_{L+1} · h_L`. It reached
**81.4% accuracy**, confirming the design document's §22.6 hypothesis that the
residual stream changes only gradually across layers.

Yet it was slower. The breakdown:

| Component | Cost |
|---|---:|
| matrix multiply (256×2048×40 layers) | 3.3 ms/token (after vectorising; 60 ms initially) |
| fetching probs/hidden GPU→CPU | 7.5 ms/token |
| **scheduler cost of requesting `probs` via callback** | **~22 ms/token** |

**On Vulkan, merely requesting a tensor from within the graph via a callback
costs about 0.5 ms of synchronization per layer.** That is what killed P2.

### P5 has zero synchronization cost but insufficient accuracy

Predicting layer L+1 from layer L's ids alone (which are read anyway) needs no
extra synchronization. A co-activation table (40×256×8 = 160 KiB) was built from
a 60,000-token real trace. But it reached only **18.5–40%**, far short of P2's
81.4%.

## The structural reason prefetching did not help

```
86% hit rate -> about 1 miss per layer
prefetch window = 1 layer = about 1.6 ms
a read = about 1 ms
overhead = extra acquire (cache churn) + a thread per layer (40 per token)
```

**At a prefetch depth of D=1, the overhead is too large against the 1.6 ms
available.**

The design document's §22.7 called for **D=4**. This implementation is D=1 and
does not meet the design. Deeper prefetching needs the separation of "software
queue (256 entries)" from "device queue (2–4)" introduced in finding S2, plus a
**persistent I/O pipeline** — which a thread per layer cannot provide.

> **Prefetching is incomplete as of this finding.** P2 (81.4%) looks promising
> as a predictor, so the next design would have to satisfy "P2 + D=4 + a
> persistent I/O pipeline + avoiding the probs synchronization" all at once.
> Avoiding that synchronization might be possible by inserting a prediction node
> into the graph with `ggml_map_custom1` (noted as an open item in §32).

## What remains of the difference (66.0 − 42.65 = 23.4 ms)

| Component | Approximate |
|---|---:|
| eval callback + ids round trip (instrumentation) | 6.6 ms |
| exposed demand I/O | 16.8 ms |

The 6.6 ms of instrumentation can be removed by putting the remap node into the
graph with `ggml_map_custom1`. The 16.8 ms of I/O can only be hidden once
prefetching works. **With both, this lands at 42.65 + α and the design target
(80%) is within reach.**
