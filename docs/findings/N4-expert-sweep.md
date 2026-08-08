# Finding N4 — Expert Sweep does not work under llama.cpp's graph execution

| | |
|---|---|
| Target | DESIGN.md §20.2 / v2 option B |
| Date | 2026-08-03 |
| Verdict | **not adopted** (cause identified; fixing it needs changes in ggml itself) |
| Default | **disabled** (`n_passes` always returns 1) |

## What it tried to do

When the union of experts referenced by one `mul_mat_id` exceeds the slab
capacity, split the FFN into P passes, mapping out-of-pass experts to the zero
slot and summing:

```
y = Σ_p FFN(x, ids_p)      out of pass -> zero_slot -> contributes 0
P = ceil(union(n_tokens) / n_slot)
```

For decode (n_tokens=1) P is 1 and the path is identical to before; only prefill
splits.

**The speed was there**: prefill 59.1 → 173.7 tok/s (2.9x), memory unchanged at
7.9 GiB.
**The output broke**: PPL 4.44 → 520801.

## Root cause: graph buffer aliasing

**Inserting a mathematically identity operation changes the result
substantially.** That cannot happen in a correct program.

| Configuration | PPL |
|---|---:|
| P=1 (reference) | **4.4461** |
| P=2, plain | 520801.58 |
| P=2 + identity CPU op (`ms_marker_fn`) | **185.26** |
| P=2 + `ggml_cont` | 536.23 |

A later pass's intermediate tensor is reusing — and overwriting — an earlier
pass's output buffer. `ggml_backend_sched` creates a split per CPU custom op
(40 layers × P passes), and under this structure the lifetime of tensors
crossing a split is not preserved.

## Suspects eliminated (all by measurement)

| Suspect | Method | Result |
|---|---|---|
| the mathematics (split addition) | Spike S4: verify `FFN(ids₀)+FFN(ids₁) == FFN(ids)` minimally | **difference 0.000e+00** (Q4_K/IQ4_NL × CPU/Vulkan) |
| the zero slot | Spike S3: decode a zero-filled block and read it back at run time | **exactly 0** (all four types) |
| id generation | run-time dump | pass0 real=2140/zero=1956, pass1 exactly complementary |
| cache eviction | slots=256 (no eviction) vs 165 (eviction) | **PPL identical** (520801.5849) |
| execution order / races | order trace via marker op, same conditions twice | remap0→FFN0→remap1→FFN1. **PPL identical** |
| kernel fusion | `GGML_VK_DISABLE_FUSION=1` | 288508 → 288916 (no change) |
| **graph aliasing** | inserting an identity op | **PPL moves 2800x → guilty** |

## A real breakdown found along the way

Decomposing "prefill is 4.9x slower" (a by-product of this finding):

```
baseline ub=512   295.1 tok/s
baseline ub=8      74.9 tok/s   <- shrinking ubatch alone costs 3.9x
MoEStream ub=8     59.7 tok/s   <- MoEStream itself costs only 1.25x
```

**MoEStream's overhead is just 1.25x.** What dominates is the structural
constraint that a small slab cannot run a large ubatch.

## Real bugs fixed on the way (independent of Expert Sweep)

| # | Bug | Impact |
|---|---|---|
| 6 | writes `sid=0` (the wrong expert) on slot exhaustion | PPL 4.97 → 2365 |
| 7 | no ceiling on `freq`, so S3-FIFO's second chance never expires | eviction stops → exhaustion |
| 8 | `n_passes` calls `finalize()` during graph construction | GGML_ASSERT (tensor not allocated) |
| 9 | `n_passes` missing the "slots ≥ n_expert → 1" branch | pointless splitting |
| 10 | slab sized by `ub × top_k` (worst case) instead of the union | 30% too much memory |

All merged into the main path; **the current P=1 path is healthy at PPL +0.11%**.

## If this is revisited

1. **Introduce a notion of "ops with side effects" into ggml** — the real fix,
   but a large upstream change
2. **Remap without a CPU custom op** — keep the slot table as a GPU-resident
   tensor and read it with `ggml_get_rows`. No splits, hence no aliasing.
   Synchronization for "the CPU decides which experts to read" is still needed
   separately
3. **Keep all experts resident during prefill only** — trading memory back for
   speed (a variant of option A)

## Conclusion

Expert Sweep is **disabled by default**, left available for experiments via
`MOESTREAM_FORCE_PASSES`. The code is not deleted, and the diagnostic machinery
(`MOESTREAM_SWEEP_TEST`, `MOESTREAM_ORDER_TRACE`) is kept for another attempt.
