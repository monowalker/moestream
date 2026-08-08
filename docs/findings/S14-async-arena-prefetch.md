# Finding S14 — Asynchronous arena prefetch (the best strategy inverts between models)

| | |
|---|---|
| Target | overlapping the prefill staging arena's (S7) loads with GPU compute |
| Date | 2026-08-06 |
| Verdict | **adopted, on by default. Ornith +10.6% / Laguna +72.5%. Output unchanged** |
| Code | `src/llama-moestream.cpp` (`pf_kick` / `pf_wait` / `ms_pfload_fn`) |

## What was wasted

Arena loading was **synchronous**. There was a `MOESTREAM_PREFILL_BUFS=2`
option, but it only added a second buffer; the load still waited.

```
layer L: [wait on I/O] -> [GPU compute] -> layer L+1: [wait on I/O] -> [GPU compute] -> ...
                          ^ the SSD is idle here
```

Measured by `[ub]` (per pass):

| | I/O | compute | serial | overlapped (theoretical) |
|---|---:|---:|---:|---:|
| Ornith | 0.97 s | 4.08 s | 5.05 s | 4.08 s (−19%) |
| Laguna | 13.37 s | 11.49 s | 24.86 s | 13.37 s (−46%) |

## Why this is safe, unlike decode prefetch (S11)

| | S11 decode prefetch | this finding |
|---|---|---|
| destination | **the ExpertCache slab** | **the arena (a plain buffer)** |
| refcount / eviction | yes → breaking it corrupts output | **none** |
| prediction | required (71% accurate) | **not required** (just read the next layer's experts) |
| effect of a wrong guess | a stall | **none** (see below) |

All that is needed is waiting for completion. There is always exactly one thread,
and state is handed over across a `join()`.

## Implementation

```
layer L's op:
  1. join the background read for layer L
  2. build this batch's union from ids
  3. * synchronously read anything not present   <- correctness is guaranteed here
  4. record this union and start prefetching layer L+1 in the background
```

**Because of step 3, whatever the prefetch read, the output is unchanged.**
`g_arena_have` tracks what is currently loaded, so nothing already present is
re-read.

## ★ The prefetch strategy inverts between models

At prefetch time the next layer's ids do not exist, so what to read must be
chosen. Two strategies were measured.

**Strategy A: read every expert** — no prediction needed, but only about 70% are
actually required, so it over-reads.

**Strategy B: read the previous pass's union for the same layer** — reads less;
anything missing is topped up in step 3.

How well the previous pass's union covers the next was measured first (640–736
samples):

| | needed now | previous union | shortfall | total read | vs all experts |
|---|---:|---:|---:|---:|---:|
| Ornith | 176.6 | 177.9 | **11.9** | 189.7 | 74% |
| Laguna | 177.4 | 179.0 | **8.4** | 187.4 | 73% |

**The previous pass's union nearly covers the next** (the shortfall is 5–7% of
what is needed).

Measuring prefill speed on top of that, **the better strategy inverts between
models**:

| | synchronous (before) | A: read all | B: union |
|---|---:|---:|---:|
| **Ornith** | 224.75 tok/s | **246.90 (+9.9%)** | 233.76 (+4.0%) |
| **Laguna** | 46.81 tok/s | 62.54 (+33.6%) | **80.12 (+71.2%)** |

- **Ornith** — its 14.5 GiB of experts fits in page cache and reads cheaply
  (9 GB/s). **Waiting on the synchronous top-up costs more than over-reading
  saves** → read all is fastest
- **Laguna** — its 50.7 GiB does not fit, so reads come from the real SSD
  (3.6 GB/s). **Reading less far outweighs the waiting** → union is fastest

> Exactly the same shape as the I/O thread count (Ornith 4 / Laguna 16).
> **Whether it fits in page cache inverts the optimum.**

## Automatic switching

The existing `[prefetch]` decision (total expert bytes vs projected page cache)
is reused directly.

```
expert bytes <= projected page cache -> I/O is cheap -> read all (create no waits)
expert bytes >  projected page cache -> I/O is dear  -> union   (read less)
```

It appears in the startup log:

```
moestream: [prefetch]   prefill read strategy = read all (create no waits)
moestream: [prefetch]   prefill read strategy = union (read less)
```

## Final result (automatic switching, on by default)

| | before (synchronous) | automatic | Improvement | Extra memory |
|---|---:|---:|---:|---:|
| **Ornith** | 224.75 tok/s | **248.59 tok/s** | **+10.6%** | +0.49 GiB |
| **Laguna** | 46.81 tok/s | **80.73 tok/s** | **+72.5%** | +1.47 GiB |

Output matches the synchronous version (greedy). Prefetch hit rate is 94.5% on
Ornith and 87.5% on Laguna.

## Built to fail safe

- if the second arena buffer cannot be allocated, it **continues with fewer**
  (running synchronously beats giving up the arena entirely)
- `MOESTREAM_PREFILL_ASYNC=0` disables it
- `MOESTREAM_PREFILL_BUFS=1` also falls back to synchronous

## Follow-up: does async change the optimal UBATCH? → no

If asynchrony hides I/O, the value of a large UBATCH should fall (the knee moving
to `ub0 × I/C` ≈ 1200). **Wrong.**

| | Memory | prefill |
|---|---:|---:|
| UBATCH=1024 | 18.49 GiB | 80.13 tok/s |
| UBATCH=8096 | 22.26 GiB | **115.99 tok/s (+45%)** |

The error was **treating I as fixed per pass**. With union loading, raising
UBATCH does enlarge one pass's union, but **the pass count falls faster**:

```
UBATCH=1024: 13.6 passes x 69% union  = effectively 9.4 full model reads
UBATCH=8096:  1.7 passes x 100% union = effectively 1.7            -> 5.5x fewer
```

Async prefetch only removes the waiting *within* a pass, which is **independent
of reducing the pass count**. Both are effective, and `[ub]`'s recommendation
formula needs no change.

> A by-product: UBATCH=8096 spends 3.77 GiB on prefill speed. Redirecting that
> memory to the slab would add about 57 slots and take the hit rate from 67% to
> about 93%. **Which of prefill and decode to favour depends on the workload**,
> and can be decided by comparing `[mrc]` and `[ub]` in `make stats`.

## A mistake made along the way

Adding `getenv("MOESTREAM_PREFILL_ASYNC")` **failed silently** (the python
`str.replace` had no assert). The environment variable was set, yet the path
never ran — `hit 0 / miss 0` — and it nearly got written up as "no effect".

**The startup log now prints `prefill async prefetch = on/off`.** Without being
able to see that a setting took effect, the same thing happens again.
