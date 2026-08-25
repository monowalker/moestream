# Finding S24/S25 — S17 implemented, measured, and rejected

| | |
|---|---|
| Target | finding S17's proposal, built and run end to end |
| Date | 2026-08-22 |
| Verdict | **rejected. +10.2% perplexity for −3.2% decode.** The quality budget (QR-2) is 1.0% |
| Measured on | `research/spikes/s24_skiprank/`, `s25_decode_ppl/`, Ornith-1.0-35B-UD-IQ4_NL, `frac=0.25` |

## What was built

S17 argued for skipping a cache **miss** whose expert contributes little, on the
grounds that a hit costs nothing so there is no reason to skip one. A follow-up
found the implementation could be far cheaper than S17 assumed:
`ggml_argsort_top_k` returns experts in **descending probability order** —
verified on 100.00% of adjacent pairs across all three traces — so the position
within `top_k` *is* the rank. The rule becomes a predicate on `k`:

```c
if (!hit && g_skip_rank > 0 && k >= g_skip_rank) { ... zero slot, no fetch ... }
```

No new tensor, no graph reordering, no extra CPU↔GPU sync. About ten lines. The
weight-based version S17 proposed would have needed the router weights, which
llama.cpp only finalises well after the `ffn_moe_topk` anchor the patch attaches
to — i.e. a graph-order change, which this project has been bitten by before
(§9). The patch is kept as `research/spikes/s24_skiprank/skiprank.patch`; it is
**not** in `src/`.

## The first quality measurement was measuring nothing

Run at `-ub 512`, perplexity came out **5.1040 for every setting** — no skip,
rank 7, rank 6, rank 5 — identical to four decimals, while the generated text
visibly degraded. Both cannot be true.

```c
if (!g_pf_ready || n_tokens <= g_pf_threshold)  -> slab path (remap_exec)
else                                            -> arena path
```

The threshold is 6. At `-ub 512` every evaluation took the **arena** path, which
never enters `remap_exec` and so never reaches the predicate. The numbers were
real; they measured a code path the change does not touch.

Re-run at `-ub 4`, below the threshold, with the path proven in the logs:

```
graph builds  prefill-path 0 / slab-path 6120
[S17] rank skip >= 7: 56162 misses skipped of 1639040 demands (3.43%)
```

> **Note for `RESULTS.md` §8.** The published perplexity figures use
> `-c 512 -b 512 -ub 512`. By the same argument they exercised the arena path,
> so they do not speak to the slab/decode path's quality. §8's decode evidence
> is the token-identity check against plain llama.cpp, not those PPL numbers.
> This is not a claim that §8 is wrong — only that it measures less than its
> placement suggests.

## Result

Speed on an idle machine, perplexity through the decode path:

| | decode | vs off | PPL | vs off |
|---|---:|---:|---:|---:|
| `MOESTREAM=0` | 41.87 ms | — | 5.0946 | — |
| off | 58.93 ms | — | 5.0926 | — |
| `SKIP_RANK=7` (−20% reads) | 57.07 ms | **−3.2%** | 5.6115 | **+10.2%** |
| `SKIP_RANK=6` (−37% reads) | 55.13 ms | −6.4% | 6.4860 | +27.4% |
| `SKIP_RANK=5` (−52% reads) | 54.16 ms | −8.1% | 7.5269 | +47.8% |

**Every 1% of speed costs 3.2–5.9% of perplexity.** QR-2 allows 1.0% total. The
cheapest setting overshoots the entire quality budget by 10x while returning
3% of decode.

## Why the offline estimate was so wrong

S17 predicted the cost from **router weight mass**: rank ≥ 7 loses 2.05% of the
mass, and the document argued this was an *upper* bound because dropping expert
*i* and renormalising leaves an error of `w_i·(E_i − <E_j>)`, not `w_i·E_i`.

Measured, 2.05% of weight mass produced **+10.2%** perplexity — five times the
supposed upper bound. Two reasons, both foreseeable:

1. **Renormalisation was never implemented.** The patch maps the expert to the
   zero slot and leaves the other weights alone, so the FFN output is scaled
   down by the dropped weight. The bound assumed a correction that is not there.
2. **Weight mass is not a proxy for output error.** Perplexity is a log
   likelihood; a small perturbation applied at every layer of every token
   compounds in a way a linear share of a softmax does not capture.

> **Lesson: an offline proxy for quality needs to be validated against the real
> metric once before it is used to rank proposals.** S17 ranked this idea above
> several alternatives on the strength of a bytes-per-weight-mass ratio that
> turned out to be 5x optimistic.

## Where this leaves the idea

Renormalising would improve it, and was the missing half. But the ceiling is
set by S20: removing 20% of the reads is worth 3% of decode, so even a perfect
version competes for a very small prize. **Combined with the fact that any
version trades accuracy for speed — which is the trade this project exists to
avoid — the direction is closed, not merely unfinished.**

> Reproduce: apply `skiprank.patch` to `src/llama-moestream.cpp`, rebuild, then
> `research/spikes/s24_skiprank/measure.sh` and `s25_decode_ppl/measure.sh`.
> Related: `S17-miss-weight-skip.md`, `S20-io-volume-price.md`,
> `M0-2-expert-distribution.md`, `RESULTS.md` §8, §9
