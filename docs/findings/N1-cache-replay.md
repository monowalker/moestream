# Finding N1 — Replaying real traces through the expert cache, and A/B results

| | |
|---|---|
| Target | DESIGN.md §11.1–§11.3 / §12.3–§12.6 |
| Date | 2026-08-03 |
| Implementation | `src/expert_cache.{hpp,cpp}` (C++17, ~260 lines) |
| Verification | replaying M0-2's real trace (20,000 tokens × 40 layers × top-8) |
| Reproduce | `make spike SPIKE=n1_cache_replay` |

## Is the implementation correct?

| | Python simulation | **the C++ implementation** |
|---|---:|---:|
| en, 38% cache, LRU-equivalent | 90.2% | **90.47%** |

**It agrees with an independent Python implementation**, so `ExpertCache` is
correct.

## Measured with real SSD I/O (en, 38%, pin=0.05)

```
hit rate       : 90.47%
bytes/token    : 43.34 MiB   (9.5% of the 455.0 MiB of full residency)
real I/O       : 1,828,860 preads at 1.43 GB/s
I/O time/token : 31.82 ms
```

Against `t_c = 43.7 ms`, `t_io = 31.82 ms` — **within the region where I/O hides
entirely behind compute**. And this I/O is **synchronous pread at QD=1**
(1.43 GB/s); per finding S2, QD=2 reaches 4.48 GB/s, so making it asynchronous
should cut `t_io` to about a third (≈10 ms). **This number is a conservative
lower bound.**

## A/B results (hit rate at a 38% cache)

| Condition | en | code |
|---|---:|---:|
| pin=0.05 / adaptive per-layer quota ON | 90.47% | 90.75% |
| **pin=0.00 / adaptive ON** | **90.53%** | 90.74% |
| pin=0.05 / adaptive **OFF** (even split) | 89.82% | 89.84% |
| pin=0.20 (the design's old default) | **88.80%** | 90.65% |

## ★ Decision 1: PINNED is disabled by default (α = 0)

| pin_ratio | en | code |
|---:|---:|---:|
| 0.00 | **90.53%** | 90.74% |
| 0.05 | 90.47% | **90.75%** |
| 0.20 | 88.80% | 90.65% |

**PINNED is neutral at best and harmful at worst** — −1.73 pt on `en` at 0.20.
M0-2 already lowered α from 0.20 to 0.05; **measurement supports going to 0.**

| Change | History |
|---|---|
| `--pin-ratio` default | 0.20 (design) → 0.05 (M0-2) → **0.00** (N1) |
| PINNED's role | a global hot set → warm-up only → **TTFT at startup only, off by default** |

Residency of the shared expert (all 40, used on every token) is kept separately.

> **The PINNED set concept in design doc §12.6 is effectively rejected.** The
> value of calibration statistics is limited to "what to read first at startup".

## ★ Decision 2: adaptive per-layer quotas help (+0.7–0.9 pt) — but in the opposite direction to the prediction

| | en | code |
|---|---:|---:|
| adaptive ON | 90.47% | 90.75% |
| adaptive OFF (even) | 89.82% | 89.84% |
| **gain** | **+0.65 pt** | **+0.91 pt** |

Estimating marginal utility from ghost lists **does work**. §12.5's mechanism is
confirmed.

**But it converges opposite to §12.5's prediction.**

```
converged quotas (en): L0=186  L10=107  L20=71  L30=87  L39=93
                       ^most               ^fewest
```

§12.5 predicted "shallow layers are diffuse, deep layers specialise → **allocate
more to deep layers**". M0-2's entropy measurements did confirm that shallow
layers are more diffuse. Nevertheless, **adaptation allocated more to the
shallow (high-entropy) layers.**

The reason: ghost hits are frequent in layers that are *starved* of cache. Higher
entropy means a larger working set, so the marginal utility of another slot is
genuinely higher there. **The rule was right; my prediction of its direction was
wrong.**

> §12.5's "more should go to deep layers" is deleted and corrected to
> **"more goes to high-entropy layers"**. Whether giving L0 186/256 = 73% is
> genuinely optimal is unverified (it may be a local optimum). Tuning
> `quota_eta` / `quota_period` remains open.

## ★ Decision 3: S3-FIFO's SMALL segment works, but contributes little

```
hits originating in SMALL : 0.1% of all hits
SMALL -> MAIN promotions  : 3,409 of 606,537 evictions
```

SMALL performs its intended role of quickly discarding experts used only once
(scan resistance), but **its direct contribution to the hit count is small**.

The initial implementation put demand misses straight into MAIN, contrary to
§12.4, leaving SMALL entirely unused (zero promotions). Fixing it left the
overall hit rate unchanged at 90.47%.

> **SMALL's real value is resistance to the prefill sweep (§12.4), which this
> replay (decode only) does not measure.** It needs re-evaluating on a trace
> that includes prefill.

## Open items

| ID | Item |
|---|---|
| N2 | add asynchronous I/O (QD=2–4) and measure `t_io`. It should fall from 31.82 to about 10 ms |
| N3 | measure SMALL's value on a trace that includes the prefill sweep |
| N4 | tune `quota_eta` / `quota_period`. Is 73% for L0 really optimal? |
| N5 | add predictor P1 (38% previous-token reuse) to lower the stall rate |
