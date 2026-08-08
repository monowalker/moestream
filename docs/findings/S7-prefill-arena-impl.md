# Finding S7 — Implementing and verifying the prefill staging arena

| | |
|---|---|
| Target | implementing findings S5 / S6 (the final form of v2 option A) |
| Date | 2026-08-04 |
| Verdict | **works. Prefill 4.5x (UBATCH=1024), +0.9 GiB, output bit-identical** |
| Code | `src/llama-moestream.cpp` (`load_layer_arena` / `ms_pfload_fn`) |
| Environment | Ryzen 7 8745HS / Radeon 780M (RADV) / Crucial P310 |
| Conditions | ctx=32768 / 100000, np=1, KV q8_0, FA on, ngl 99, Ornith-1.0-35B-UD-IQ4_NL |

## Implementation

During prefill (`n_tokens > threshold`), computation reads from **a staging
arena holding one layer's full expert set**. A single arena is allocated and
reused layer by layer.

```
decode  (n_tokens <= 11) : the 97-slot slab as before, ids remapped to slot_id
prefill (n_tokens >  11) : an arena with all 256 experts, ids left as expert_id
```

The threshold is computed: the largest `n` satisfying
`union(n) * 1.15 + top_k <= n_slot` — **the most the slab can handle without
exhaustion**. On this machine it is **11 tokens**. With that, the slab no longer
constrains ubatch and `-ub 512` becomes usable, and `MOESTREAM_MAX_UBATCH` no
longer has to match `UBATCH`.

Filling the arena is done as a **CPU custom op** whose output is used as
`mul_mat_id`'s `ids`, which establishes "fill the arena → mul_mat_id" as a graph
dependency. It is the same mechanism as the decode-side remap op — a route with
a track record.

The arena's actual size is **498 MiB**. (The original 330 MiB estimate assumed a
uniform 450560 B per expert; under UD quantization sizes differ by layer, and the
largest layer is 498 MiB.)

## Results

| Configuration | Memory | prefill 479 tok | 3097 tok | 13877 tok | decode |
|---|---:|---:|---:|---:|---:|
| plain llama.cpp (ub=512) | 17.31 GiB | 173.1 | 267.1 | 254.8 tok/s | 41.6–45.8 ms |
| MoEStream, no arena (ub=8) | 8.22 GiB | 41.3 | 49.3 | 46.0 tok/s | 59.8–63.0 ms |
| **MoEStream, arena (ub=512)** | **8.85 GiB** | **103.4** | **133.4** | **141.3 tok/s** | **50.7–76.6 ms** |

- **prefill 3.07x** (46.0 → 141.3 tok/s at 13877 tokens)
- **memory +0.63 GiB** (8.22 → 8.85 GiB)
- decode unchanged (measured warm separately: 53.8 → 54.2 ms/tok)

### Output correctness

Greedy (temperature=0, top_k=1), across all three prompt lengths:

| | reproducibility (same request twice) | vs plain llama.cpp (ub=512) |
|---|---|---|
| short (479 tok) | **identical** | **identical** |
| mid (3097 tok) | **identical** | **identical** |
| long (13877 tok) | **identical** | **identical** |

**Generated tokens match plain llama.cpp with every expert resident, exactly.**

## Two bugs hit during implementation

### Bug 1: imported host memory slows decode 19x even when unused

Importing anonymous memory via `VK_EXT_external_memory_host` as S5/S6 designed
made prefill faster but took **decode from 53.8 to 1017.9 ms/token**.

Isolating it:

| Configuration | decode |
|---|---:|
| arena disabled / ub=8 | 53.84 ms |
| arena disabled / ub=512 | 53.80 ms | ← ubatch is innocent |
| arena enabled (import) / ub=512 | **1017.88 ms** |
| **arena allocated but never used** (threshold=100000) | **1023.65 ms** |

**Slow merely from being allocated, never used, never even in the graph.**
Neither the arena count (1/2/4/8), nor `GGML_VK_MAX_NODES_PER_SUBMIT=1`, nor
`GGML_VK_DISABLE_ASYNC=1` made any difference.

The apparent cause is amdgpu **revalidating userptr BOs on every command
submission**. Scanning 498 MiB = 127,488 pages each time, across dozens of
submissions per decoded token, easily reaches a second.

**The fix: stop importing and let ggml-vulkan allocate normally.** On UMA it
returns `DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT` (heap 1), so the existing
zero-copy path (`ggml_backend_vk_buffer_host_ptr`) can `pread` straight into it.
It is also faster for the GPU to read than heap 0 imported memory (S5: 80.6 vs
31.0 GB/s).

> **S5/S6's premise of "host pointer import" turned out to be unusable in
> practice.** S6's result — that `mul_mat_id` works bit-identically on an arena —
> still holds, but the product implementation uses no import at all.
> ADR-0016 ("on UMA, allocate via host pointer import") should be **rejected**.

### Bug 2: `selected_experts` is sometimes a view

Passing ids through in `ms_pfload_fn` used a flat
`memcpy(dst->data, a->data, ggml_nbytes(a))`. But `a` can be a non-contiguous
view, in which case **the wrong expert_ids** reach `mul_mat_id`.

The symptom is hard to read. The output looks grammatical and sensible, yet
**the same request returns something different every time**:

```
nbuf=1   reproducibility: differs     nbuf=2   reproducibility: differs
nbuf=4   reproducibility: differs     nbuf=8   reproducibility: differs
```

Running plain llama.cpp (ub=512) three times as a control gave identical output,
establishing that the non-determinism was ours.

**The fix:** access with the `a->nb[0]` / `a->nb[1]` strides, as
`ms_remap_fn`'s passthrough already did. Reproducibility and bit-identity both
returned.

> Lesson: **"the output looks plausible" is not evidence of correctness.**
> Not everything breaks as spectacularly as N4 (PPL 520801). Always check greedy
> reproducibility and agreement with a reference implementation.

## Follow-up: sweeping UBATCH (2026-08-04)

The arena **reads every expert (12.9 GiB) once per pass**, so the number of reads
is `passes = prompt length / UBATCH`. Raising UBATCH reduces passes and therefore
total I/O directly.

For a 67k-token prompt:

| UBATCH | Passes | Total read |
|---:|---:|---:|
| 512 | 131 | 1.69 TB |
| 1024 | 66 | 851 GiB |
| 2048 | 33 | 426 GiB |

Measured (13877-token prompt; prefill tok/s / device memory):

| UBATCH | ctx=32768 | ctx=100000 (the real setup) |
|---:|---|---|
| 512 | 140.3 tok/s / 8.84 GiB | 110.2 tok/s / 9.65 GiB |
| **1024** | **194.2 tok/s / 8.98 GiB** | **188.6 tok/s / 9.92 GiB** |
| 2048 | 199.3 tok/s / 9.26 GiB | 198.0 tok/s / 10.45 GiB |
| 4096 | 195.7 tok/s / 9.82 GiB | — |

**It plateaus at 1024**, which means the constraint moved from I/O to
**compute**. Plain llama.cpp (ub=1024) is 244.4 tok/s / 17.44 GiB, so the arena
lands at **77% of the speed for 57% of the memory** (it was 45% of the speed
before).

2048 buys +5% for double the memory. **1024 is the default.**

Output was confirmed bit-identical to plain llama.cpp under greedy decoding at
512 / 1024 / 2048 / 4096.

> **[Corrected 2026-08-07] "It plateaus because I/O-bound became compute-bound"
> is not the right explanation.** Re-sweeping on current code (§10.11) shows 1024
> is a **peak**, not a knee: 2048 is 10% *slower*. Per-token compute grows
> super-linearly with UBATCH, and the optimum is where that balances against
> amortising I/O.

### What it means for the real setup (ctx=100000)

| | prefill | re-prefilling 67k tokens |
|---|---:|---:|
| before the arena (ub=8) | ~42 tok/s | ~27 min |
| arena (ub=512) | 110.2 tok/s | ~10.6 min |
| **arena (ub=1024)** | **188.6 tok/s** | **~5.9 min** |

**4.5x overall.**

## Open items

| # | Item | Detail |
|---|---|---|
| 1 | ~~decode cold start~~ | **Corrected 2026-08-06: not a problem. No action needed.**<br/>It was originally recorded as "94.6–102.8 ms/tok cold, 54.2 ms/tok warm, about 1.8x". That was wrong. Those figures were **single measurements** taken during the UBATCH sweep, averaged over 700 generated tokens. A phenomenon that resolves in 25 tokens cannot raise a 700-token average by 1.8x.<br/>**Measured properly (ten runs of 25 tokens from a fresh start)**: 74.6 → 53.5 → 53.9 → … **steady from the second**. Total cost about **525 ms, once per server start**.<br/>Decode at a 14k context is also 54.8–57.4 ms/tok, no different from a short context, so the 94 ms figures are not explained by context length either. Given the ±20 ms spread seen in the frac sweep, the difference between 94.6 and 54.2 was within noise from the start.<br/>Touching the `acquire()`/`release()` refcount path for 525 ms is not worth risk 1 from S11 (refcount leak → eviction stops → corrupt output). |
| 2 | I/O volume | every forward pass reads all experts (12.9 GiB). Page cache makes the effective rate higher than 4.48 GB/s, but it depends on RAM pressure |
| 3 | double buffering | currently a single buffer, so I/O and compute serialise. Since UBATCH=1024 already moved the constraint to compute, **the benefit looks small** (paying +498 MiB is questionable) — *superseded by finding S14, which measured +10.6% to +72.5%* |
| 4 | other architectures | verified only on Ornith (qwen35moe) |

## Conclusion

**0.9 GiB more memory buys 4.5x prefill (ctx=100000, UBATCH=1024), and the
output is bit-identical under greedy decoding to plain llama.cpp with every
expert resident.**

The reduction against the 17.44 GiB baseline is **−48.5%** (ctx=32768,
UBATCH=1024) — a different kind of trade from the rejected `MAX_UBATCH=32`
option (13.5 GiB, −22%).

It sits at **77% of the speed for 57% of the memory**. Before the arena it was
45% of the speed.
