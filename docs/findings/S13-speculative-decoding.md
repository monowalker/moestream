# Finding S13 — Speculative decoding backfires when I/O bound

| | |
|---|---|
| Target | using llama.cpp's built-in speculative decoding (`--spec-type`) |
| Date | 2026-08-06 |
| Verdict | **not adopted. −35% with ngram, −79% with a draft model** |
| Measured on | Ornith-1.0-35B / `MOESTREAM_CACHE_FRAC=0.25` / UBATCH=1024 |

## The reasoning that motivated it

Decode is slow because of **per-token fixed costs**:

```
every token passes through 40 layers
  -> paying 4.97 ms of synchronization + 13.06 ms of I/O + weight reads each time
```

Speculative decoding verifies K tokens in one pass, which should amortise that
fixed cost by a factor of K. And `n_tokens=4` stays within the slab path's
threshold of 11, with union(4) ≈ 30 experts fitting in 64 slots.

**MoEStream should benefit more than an ordinary model**, the thinking went:
normally compute dominates and speculation's gain is limited, whereas here fixed
costs dominate.

## Result

| Configuration | Memory | decode |
|---|---:|---:|
| **baseline (no speculation)** | 6.59 GiB | **57.8 ms/tok (17.30 tok/s)** |
| ngram-cache (no extra memory) | 6.59 GiB | 77.9 ms/tok (12.84 tok/s) **−35%** |
| draft-simple (Qwen3.5-4B) | 9.80 GiB | 281.0 ms/tok (3.56 tok/s) **−79%** |

## Why

### 1. The acceptance rate is low

```
draft acceptance = 0.34615 (9 accepted / 26 generated), mean len = 2.50
```

Four tokens speculated, 0.87 accepted on average. **The verification cost of the
three rejected tokens is entirely wasted.**

### 2. Waste costs more here than usual

The verification pass has `n_tokens=4`, so it reads union(4) ≈ 30 experts, but
only one token's worth is accepted. **I/O per token goes up.**

```
normal decode          : union(1) = 8 experts per token
speculative (0.87 acc) : union(4)/0.87 ~= 34 experts per token  -> about 4x
```

Speculation is a technique for environments with compute to spare where
serialisation is the bottleneck. MoEStream is the opposite: **I/O dominates**
(13.06 of 59.82 ms on Ornith; 65.8% of decode on Laguna). Speculation pushes
that I/O up.

### 3. A draft model takes memory

Loading Qwen3.5-4B (2.71 GiB) takes memory from 6.59 to 9.80 GiB, squeezing the
slab and lowering the hit rate. **Any technique that needs extra memory is
doubly disadvantaged in a system where memory is the constraint.**


> **[2026-08-23]** MTP could not be tested here — Ornith-1.0 has no
> `nextn_predict_layers`. Finding S31 tests it on Ornith-1.5 and confirms this
> section's mechanism at **88.7% acceptance**, which removes the objection that
> the 0.346 draft acceptance above was the real cause. It also shows the sign
> flips for dense models, where a pass has no union to widen.

## Generalisation

> **In an I/O-bound system, every technique of the form "do more computation to
> reduce serialisation" backfires.**

The same root as finding S11 (predictive prefetch):

| | What it tried | Result |
|---|---|---|
| S11 prefetch | **hide** I/O | no means to hide it; only the prediction cost grew |
| S13 speculation | **add** compute to reduce serialisation | I/O went up 4x |

Both assume compute is in surplus. In MoEStream it is not compute that is
scarce — **it is SSD bandwidth**.

## Aside: vocabulary compatibility

Speculation requires matching vocabularies. On this machine:

| Model | arch | Vocab | Draft available? |
|---|---|---:|---|
| Ornith-1.0-35B | qwen35moe | 248320 | Qwen3.5-4B / Qwen3.6-27B work |
| Qwen3-Coder-Next | qwen3next | 151936 | none |
| Laguna-S-2.1 | laguna | 100352 | none |

For most models it is not usable at all.
