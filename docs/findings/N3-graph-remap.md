# Finding N3 — In-graph remap is performance-neutral; the instrumentation cost survives either route

| | |
|---|---|
| Target | DESIGN.md §10.4 / §22 / the open item from finding N2 |
| Date | 2026-08-03 |
| Reproduce | `MOESTREAM_GRAPH_REMAP=1` (default) / `=0` for the eval callback route |

## Motivation

Finding N2 established that of the 23.4 ms gap to baseline, **6.6 ms was
instrumentation** (the eval callback forcing node-by-node graph execution, plus
round-tripping ids to the GPU). To remove it, the remap node was inserted into
the graph with `ggml_map_custom1`.

Expected benefits:
- no eval callback → the graph executes in batches
- runs on the CPU backend → touches tensor data directly, no ids round trip

## Implementation

One node inserted into `build_moe_ffn` in `llama-graph.cpp`:

```
cb(selected_experts, "ffn_moe_topk", il);
ggml_tensor * ms_ids = moestream::build_remap(ctx0, selected_experts, il);
```

**An important caveat**: `selected_experts` is **also used to fetch the weights**
via `ggml_get_rows(probs, selected_experts)`. Turning it into slot_ids there
would corrupt the weights, so the original tensor is kept and only the ids passed
to `build_lora_mm_id` are substituted (4 sites).

## Results

| Route | ms/token | tok/s | misses |
|---|---:|---:|---:|
| baseline | 42.51 | 23.52 | — |
| eval callback | 64.00 | 15.63 | 7,029 |
| **in-graph remap** | 62.71–64.56 | 15.5–16.0 | **6,595** |

**Performance is effectively identical (the difference is within noise). The
hoped-for 6.6 ms was not recovered.**

The reason: inserting a CPU custom op into a Vulkan graph creates a backend
boundary (split) per layer, and the resulting synchronization and transfer costs
came to roughly what node-by-node execution under the eval callback costs.

## A bug squashed on the way: double execution

The first measurement was much worse at 114 ms/token, with misses rising 2.4x
from 7,029 to 16,622. The cause was that **the eval callback's `on_topk` and the
in-graph `ms_remap_fn` were both running**, so acquire and SSD reads happened
twice.

Fixed by making the routes mutually exclusive via `want_topk()`. It was only
after adding a counter (`remap called 2040 times = 51 steps × 40 layers`) that
"the call count is right but acquires are doubled" could be separated out.

## Decision: make in-graph remap the default

Performance is neutral, but `MOESTREAM_GRAPH_REMAP=1` is the default for these
reasons:

| Aspect | in-graph remap | eval callback |
|---|---|---|
| performance | equal | equal |
| misses | 6,595 | 7,029 (slightly more) |
| occupies `cb_eval` | **no** (users can have it) | yes |
| consistency with the design | self-contained in the graph, as §10.4 intended | bolted-on instrumentation |

## Conclusion and open items

**The ~6 ms of instrumentation cannot be avoided by either route.** The root
cause is that the router's choices have to be seen on the CPU, and any CPU
intervention point in a GPU-executed graph implies synchronization.

To genuinely remove it:
- do the remap on the GPU (look up the slot table in a Vulkan shader) — needs
  kernel changes
- or make the slot table a GPU-resident tensor read with `ggml_get_rows`

The latter **may be possible without modifying any kernel** (expressing
ids → slot_table[ids] via `ggml_get_rows`). The CPU still needs a point at which
to issue I/O, so it would require splitting "remap on the GPU, I/O decisions on
the CPU". Noted as the next item.

---

## Follow-up: passing `cur` as a second input (trying to run P2 with zero synchronization)

Finding N2 saw P2 (81.4% accurate) killed by the ~22 ms/token of synchronization
needed to fetch `probs`. The hypothesis here was that passing `cur` as **the
second input** to a custom op that already exists should add no backend splits
(`ggml_map_custom1` → `ggml_map_custom2`).

### Result: the hypothesis was rejected

| Configuration | ms/token |
|---|---:|
| `map_custom1` (ids only) + prefetch off | **62.7–64.6** |
| `map_custom2` (ids + cur) + prefetch off | **77.0** |
| `map_custom2` + P2 prefetch top-8 | 89.1 |

**Merely adding `cur` as an input costs +13 ms/token.**
`cur` is only [n_embd, n_tokens] = 8 KiB, but bringing it to the CPU takes about
0.35 ms per layer — 14 ms across 40 layers. It is not the transfer volume; **it
is the synchronization itself.**

### The prefetch barely issued anything either

Over 150 tokens, only **34–39 prefetches** were issued. Most of what P2 predicts
is already in the cache, so nothing needs prefetching — the same structure as P1
in finding N2.

| top-N | Issued | Actually useful |
|---:|---:|---:|
| 2 | 36 | 4 (11%) |
| 4 | 39 | 7 (18%) |
| 8 | 34 | 2 (6%) |

## Summary: every route that brings the hidden state to the CPU failed

| Method | Cost of fetching hidden | Verdict |
|---|---|---|
| requesting `probs` via the eval callback | ~22 ms/token | ❌ |
| `cur` as the second input to `map_custom2` | ~14 ms/token | ❌ |

**P2 is a good predictor (81.4%) but unusable unless some way exists to extract
the hidden state to the CPU on the Vulkan backend.**

Remaining possibilities:
1. **Predict on the GPU** — add layer L+1's router as an extra node in the GPU
   graph and pass only the result (top-k ids) to the CPU. ids are 32 B so the
   transfer is trivial, but one synchronization is still needed for the CPU to
   read it. If it can be read at the same point as the existing remap, the extra
   cost could be zero.
2. **Give up prefetching and raise the cache ratio** — measured hit rate is
   86.7% at frac=0.38. Raising frac reduces misses but works against the memory
   goal.

## Final configuration (defaults)

```
MOESTREAM=1
MOESTREAM_CACHE_FRAC=0.38
MOESTREAM_IO_THREADS=8
MOESTREAM_GRAPH_REMAP=1      # ggml_map_custom1; cur is not passed
MOESTREAM_PREFETCH=0         # prefetch off by default (all three schemes net-negative)
```

| | ms/token | tok/s |
|---|---:|---:|
| baseline (3 runs) | 42.45 / 42.44 / 42.28 | 23.6 |
| **MoEStream 38% (3 runs)** | **64.42 / 69.99 / 64.29** | **14.3–15.6** |

**65% of baseline (median). Memory −53%.**
