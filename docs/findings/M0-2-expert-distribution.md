# Finding M0-2 — Measuring expert activation distribution, and the go/no-go decision

| | |
|---|---|
| Target | DESIGN.md §32.1 (the largest risk) / §34.2 (go/no-go) / §12.5 / §12.6 / §21.5 / §26.4 / §32.2 |
| Date | 2026-08-03 |
| **Verdict** | **GO** |
| Model | `Ornith-1.0-35B-UD-IQ4_NL.gguf` (arch `qwen35moe`, 40 layers / 256 experts / top-8) |
| Method | capture `ffn_moe_topk-*` / `ffn_moe_weights_*-*` via `llama_context_params.cb_eval`, **without patching llama.cpp** |
| Sample | 3 domains × 20,000 tokens = 60,000 tokens / **19.2 million activation events** |
| Hardware | Radeon 780M (Vulkan), Crucial P310 |
| Reproduce | `research/tools/analysis/expert_trace` → `research/tools/analysis/analyze_trace.py`, `analyze_cross_domain.py` |

## Criteria (published in advance in §34.2)

```
GO    : h(0.38) >= 0.50
NO-GO : h(0.38) < 0.30 and predictor P2 accuracy < 50%
```

The required hit rate, from measured values (finding S1: `B_act=455 MiB/token`,
`t_c=43.7 ms`, `BW=4.46 GB/s`):

```
to stay within a 20% slowdown, h >= 51.0%
```

## Results

### Miss ratio curve by domain (38% cache = 3,880 of 10,240 slots)

| Domain | Zipf exponent `s` | oracle (static hot set) | **LRU (dynamic)** | Verdict |
|---|---:|---:|---:|---|
| code | 0.473 | 87.7% | **88.7%** | GO |
| ja | 0.340 | 87.9% | **91.5%** | GO |
| en | 0.283 | 82.1% | **90.2%** | GO |

**Against a requirement of 51.0%, the worst domain gives 88.7% — about 1.7x of
headroom.**

### The full MRC

| Cache ratio | code | ja | en |
|---:|---:|---:|---:|
| 5% | 45.7% | 37.6% | 31.2% |
| 10% | 60.4% | 52.3% | 46.0% |
| 20% | 74.7% | 70.5% | 63.7% |
| **38%** | **87.7%** | **87.9%** | **82.1%** |
| 50% | 92.5% | 93.4% | 89.4% |
| 75% | 98.2% | 98.5% | 97.5% |

(Oracle values. LRU at 38% gives code 88.7 / ja 91.5 / en 90.2.)

---

## ★ Finding 1: global frequency skew is weak, but temporal locality is strong

**On `ja` and `en`, LRU beat the static oracle hot set.**

| Domain | Oracle | LRU | Difference |
|---|---:|---:|---:|
| code | 87.7% | 88.7% | +1.0 pt |
| ja | 87.9% | 91.5% | **+3.6 pt** |
| en | 82.1% | 90.2% | **+8.1 pt** |

The Zipf exponents are also below the `s ≳ 0.45` assumed in §21.5 — 0.283 for
`en` and 0.340 for `ja`. That indicates the Qwen family's load-balancing
auxiliary loss (the concern raised in §12.6) is genuinely effective.

**The conclusion does not change**, because even with a flat static frequency
distribution, **temporal locality within a sequence is strong**. Real data
supports the hypothesis stated in §12.6:

> "Global frequency skew is likely to be weak. Per-sequence and per-domain skew,
> however, is expected to be strong."

### → Design change 1: lower the PINNED ratio α from 0.20 to 0.05

§12.6 specified a PINNED set (static residency from calibration statistics) at a
default of 20%, but measurement shows **a static hot set is worse than dynamic
LRU**. Whatever is allocated to PINNED shrinks the dynamic cache, so **a large α
is actively harmful**.

| Change | Old | New |
|---|---|---|
| `--pin-ratio` default | 0.20 | **0.05** |
| PINNED's purpose | a global hot set | **warm-up only** (naturally replaced within a few thousand steps) |
| `--pin-strategy` default | `auto` | **`warmup-only`** |

PINNED residency of the shared expert (40 of them, 76 MiB) is kept, since it is
used on every token.

---

## ★ Finding 2: the direction for per-layer quotas is settled (§12.5's prediction held)

Routing entropy per layer (maximum 8.00 bits):

| Domain | Mean | L0 | L10 | L20 | L30 | L39 | Minimum (most specialised) |
|---|---:|---:|---:|---:|---:|---:|---|
| code | 6.41 | 7.53 | 6.56 | 5.82 | 6.21 | 6.36 | L20 (5.82) |
| ja | 6.63 | 7.54 | 6.92 | 6.29 | 6.74 | 5.94 | L36 (5.74) |
| en | 6.98 | 7.73 | 7.08 | 6.68 | 6.99 | 6.89 | L37 (6.63) |

**All three domains show "shallow layers diffuse, middle and deep layers
specialised"**, exactly as §12.5 predicted — so ghost-list-driven per-layer quota
adjustment **should converge toward allocating more to deep layers**.

> **[Corrected in finding N1]** It converged the other way: **more goes to
> high-entropy (shallow) layers**, because that is where the marginal utility of
> another slot is actually highest. The rule is right; the predicted direction
> was wrong.

Worth noting that L0 is always the highest entropy (7.5–7.7 bits, nearly
uniform). **Caching barely works at layer 0**, leaving room for optimisation —
a smaller `quota_min` for layer 0, or treating its experts separately (future
work).

---

## ★ Finding 3: cross-domain interference is real but not fatal

The unverified risk in §32.2: does mixing domains across sessions flatten the
distribution?

### Hot set overlap (38% cache)

| Pair | Shared | Jaccard |
|---|---:|---:|
| code ∩ ja | 55.5% | 0.384 |
| code ∩ en | 56.8% | 0.397 |
| ja ∩ en | **70.4%** | 0.543 |
| **union of all three** | **1.63x** larger | |

Three domains at once need **1.63x** the cache of one, not 3x. The natural
languages (ja/en) share 70%; only code is somewhat separate.

### LRU hit rate when mixed

| Cache ratio | single-domain mean | domains in sequence | **fully interleaved** | Difference |
|---:|---:|---:|---:|---:|
| 20% | 76.9% | 76.9% | 64.6% | −12.3 pt |
| **38%** | 89.9% | 90.0% | **82.4%** | **−7.6 pt** |
| 50% | 94.0% | 94.1% | 89.4% | −4.7 pt |

**Even the worst case (three domains fully interleaved) gives 82.4%**, far above
the required 51.0%. The interference is real but not fatal.

---

## ★ Finding 4: affinity batching works, but the switching granularity was wrong

Measuring K in §26.4's "run the same cluster for K steps before switching"
(38% cache, three domains mixed):

| K (consecutive tokens) | Hit rate |
|---:|---:|
| 1 | 82.4% |
| 4 | 82.4% |
| 16 | **81.1%** ← the default. Slightly worse |
| 64 | 85.7% |
| 256 | 89.0% |
| **1024** | **89.8%** (nearly recovering the 90.0% of sequential execution) |

### → Design change 2: raise affinity batching's K from 16 to 256

§26.4's default of `K = 16` is **not merely ineffective but slightly harmful**,
presumably because it switches before the working set has turned over.
**It is meaningless below K = 256.**

| Change | Old | New |
|---|---|---|
| `--affinity-switch-tokens` default | 16 | **256** |
| ceiling | — | 1024 (beyond which gains saturate and TBT fairness degrades) |

A larger K does lengthen other sessions' waiting time, so **K must be forced
down while an `interactive`-class session is waiting** (combined with §26.6's
priority control).

---

## ★ Finding 5: router weight distribution — is a weight threshold skip justified?

Nearly identical across the three domains:

| Rank | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mean weight | 0.25 | 0.17 | 0.13 | 0.11 | 0.10 | 0.09 | 0.08 | 0.07 |

| Threshold | Selections affected | **Weight mass lost** |
|---|---:|---:|
| `w < 0.02` | 0.1% | **0.01–0.02%** |
| `w < 0.06` | 5.3–7.4% | **2.1–2.9%** |

§11.6's `soft` mode default of `τ = 0.02` **skips essentially nothing** (0.1%).
Even `turbo`'s `τ = 0.06` loses under 3% of the weight mass.

> **Note**: the bottom two weights sum to about 15% of the total, so the design
> document's §11.6 assumption that "the bottom 2–3 weights sum to a few percent"
> was **somewhat optimistic** — it is actually 15%. The shared expert always
> contributes, so the effective impact is smaller still. `soft` mode's quality
> impact (QR-2: PPL degradation ≤ 1%) needs separate perplexity verification.

---

## ★ Finding 6: §21.5's analytical model was far too pessimistic

§21.5 estimated `h(f) ≈ f^(1−s)` from the Zipf distribution, predicting
`h(0.38) ≈ 51%` at `s = 0.3`.

**Measured (en, s=0.283): 90.2% with LRU — 1.8x the prediction.**

The cause: that formula looks only at **the static frequency distribution** and
**ignores temporal locality entirely**. A real cache holds what is being used
now, so a flat distribution still gives a high hit rate.

> **§21.5's Zipf model should be deleted and replaced with the measured MRC.**
> The analytical model was useful for early orientation, but keeping it now that
> measurements exist would be misleading.

---

## Upper bounds on predictors (the basis for §22)

| Predictor | Upper bound (measured) |
|---|---|
| **P1 temporal locality** (the previous token's experts in the same layer) | **37.6–42.8%** correct |
| P0 certain information | 100% (by definition) |

P1 at about 38% means **four in ten prefetches would be right just by copying
the previous token's choices**. That costs almost nothing, so §22.3's
characterisation of it as "the cheapest and earliest predictor" is supported.

> **[Corrected in finding N2]** This is not so. What P1 predicts is by
> definition already in the cache, so it produces nothing to prefetch (zero
> issued). A 38% reuse rate says nothing about a predictor's value.

Upper bounds for P2 (layer lookahead), P5 (co-activation) and P3 (a learned
predictor) are unmeasured. Next up.

---

## Summary

| Risk listed in §32.1 | Measured outcome |
|---|---|
| global frequency skew is insufficient | **it happened** (s = 0.28–0.47) |
| → hit rate falls short of target | **it did not**. Temporal locality compensates, giving 82–91% |
| flattening across sessions (§32.2) | **partly happened** (−7.6 pt), but 82.4% still exceeds the target |

**None of the five mitigations prepared in §32.1 need to be activated.** The
design proceeds to the implementation phase unchanged.

## Next

| ID | Item | Priority |
|---|---|---|
| M0-3 | measure predictor P2 (layer lookahead) accuracy | high |
| M0-4 | compare S3-FIFO against LRU (confirm the benefit before implementing) | medium |
| S2 | io_uring bandwidth: can it beat QD=8's 4.46 GB/s? | high |
| Q-1 | measure `soft` mode perplexity (verifying QR-2) | medium |
