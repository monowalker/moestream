# Finding S12 — The arena did not need to read every expert

| | |
|---|---|
| Target | how much the prefill staging arena (finding S7) reads |
| Date | 2026-08-06 |
| Verdict | **adopted. 40–53% faster on small prefill deltas. Output bit-identical** |
| Code | `src/llama-moestream.cpp` (`load_layer_arena` / `ms_pfload_fn`) |

## What was wasted

`load_layer_arena()` **always read all 256 experts**.

But everything above the threshold (11 on Ornith) takes the arena path, and that
includes **the 38–500 token deltas that dominate agent use**. On a small delta,
only a fraction of the experts is actually referenced.

Measured before the change (Ornith / frac=0.25 / UBATCH=1024):

| Delta tokens | prefill |
|---:|---:|
| 6 (slab path) | 0.68 s |
| 11 | 2.06 s |
| 16 | 2.26 s |
| 26 | 1.99 s |
| 41 | 3.02 s |
| **201** | **4.80 s** |
| **401** | **4.65 s** |

**201 and 401 cost almost the same.** The fixed cost of reading all 12.9 GiB
dominated, and the token count barely mattered.

## The change

Build the set of **experts actually referenced by this batch** from the `ids`
that `ms_pfload_fn` receives, and read only those.

```cpp
for (each token, each top_k) { e = ids[...]; need[e] = 1; }
load_layer_arena(il, &need);
```

Because the arena is reused across layers, **what is currently loaded has to be
tracked** (the layer index alone is insufficient — the same layer may have been
filled with a different set). The set is reset when the layer changes.

Slots that were not filled retain the previous layer's data, but `need` is built
from **the very same tensor** that `mul_mat_id` will use as `ids`, so anything
referenced is guaranteed to have been loaded. It is safe by construction.

## Results

| Delta tokens | Before | After | Reduction |
|---:|---:|---:|---:|
| 6 (slab path) | 0.68 s | 0.69 s | — (different path, unchanged) |
| 11 | 2.06 s | **1.25 s** | **−39%** |
| 16 | 2.26 s | 1.42 s | −37% |
| 26 | 1.99 s | 1.20 s | −40% |
| 41 | 3.02 s | **1.75 s** | **−42%** |
| 201 | 4.80 s | **2.24 s** | **−53%** |

Bytes read, measured:

```
Ornith (10 small deltas) : 19489 of 81920 experts read -> 76% saved
Laguna (sharded GGUF)    :  1186 of 11776 experts read -> 90% saved
```

### Why it beat the estimate

The prior estimate used the union formula
`u(n) = n_expert × (1 − (1 − top_k/n_expert)^n)`, giving −27% at 41 tokens and
**saturation at 0%** by 201.

Measurement gave −42% at 41 and −53% at 201.
**Real expert selection is more skewed than the formula, concentrating on fewer
experts.** Assuming uniform selection was too conservative.

## Correctness

| | Result |
|---|---|
| reproducibility (same request twice) | **identical** |
| greedy match against plain llama.cpp | **identical** |
| slot exhaustion | **zero** |
| Laguna (sharded GGUF + leading dense layer) | normal |

## The statistics were lying, reporting "0% saved"

The first implementation displayed `19489 / 19489 = 0% saved` — while the
measurement was 40–50% faster.

The denominator counted "how many are in the union", making `asked == read` and
therefore **always 0%**. It was corrected to "how many would have been read
without the union" (all experts).

> **It was noticed only because the measurement (fast) contradicted the
> statistic (0%).** Looking at the statistic alone would have concluded "no
> effect".

## Residual risks (low, not zero)

| # | Item | Handling |
|---|---|---|
| 1 | **an out-of-range expert_id** would not be in `need` and would be referenced unloaded | ids were passed straight through before this change too, so it is not a new hole — but it is still a hole. It is now **detected and warned about once** |
| 2 | early return when everything is already loaded skipped the I/O accounting | fixed, because it made `[ub]` underestimate I/O |

## Where it helps

In agent use, the per-turn delta falls in exactly this range (production logs
show `f_sim=0.998`, with 38–41 tokens most common). **TTFT drops by about 40%
every turn.**

On long prompts the union saturates and the effect fades — but that is the region
where `UBATCH` tuning (finding S7) is doing the work instead.
