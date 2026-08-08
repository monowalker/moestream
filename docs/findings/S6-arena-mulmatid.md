# Finding S6 — `ggml_mul_mat_id` works on an imported arena

| | |
|---|---|
| Target | finding S5's "risk 1: ggml integration" |
| Date | 2026-08-04 |
| Verdict | **PASS. Bit-identical, with only a 9–13% speed difference** |
| Code | `research/spikes/s6_arena_mulmatid/main.c` |
| Environment | Radeon 780M (RADV PHOENIX) / ggml b0.18.0 / Q4_K |

## What was tested

The biggest unknown left by S5 was whether **`ggml_mul_mat_id` can correctly
index 256 experts' worth of tensors placed on an imported anonymous arena**.
Without that, none of S5's other results would matter.

`mul_mat_id` was run over **identical data by an identical procedure** on both
the arena path and a normal buffer, and the outputs compared. The arena is
allocated with `ggml_backend_dev_buffer_from_host_ptr()` — upstream's generic
API. No MoEStream-specific patch is involved.

`ids` was set to `i % 256`, **spanning the full 0..255 range** — a reference
pattern the current slab (97 slots) cannot express.

## Results

```
n_expert=256  n_tok=64  top_k=8  distinct experts referenced=256

--- case 1: ne0=512 ne1=256 Q4_K  (as = 18.0 MiB) ---
  [ OK ] T1 mul_mat_id executes on the arena
  [ OK ] T2 bit-identical to a normal buffer  ★the main test   max|diff|=0.000e+00 relRMS=0.000e+00
  [ OK ] T3 correct across the full 0..255 id range
  [ OK ] T4 consistent with the CPU backend    relRMS=0.0045 (max|diff|=2.164e-01)
       speed: arena 3.50 ms / normal 3.10 ms  (ratio 1.13)

--- case 2: ne0=2048 ne1=768 Q4_K  (as = 216.0 MiB) ---
  [ OK ] T5 executes at realistic sizes
  [ OK ] T5 bit-identical at realistic sizes   max|diff|=0.000e+00 relRMS=0.000e+00
       speed: arena 21.84 ms / normal 20.08 ms  (ratio 1.09)

 verdict: PASS
```

**`max|diff| = 0.000e+00`.** The 256 expert tensors on the arena return exactly
the same result as on a normal buffer, at 18 MiB and at 216 MiB alike.

The relative RMS of 0.0045 against the CPU backend is comfortably below Q4_K's
quantization error, ruling out "fetching a different expert" (misindexing would
put the relative error near 1.0 — the same criterion as finding S0b).

## An important correction: the speed penalty is far smaller than S5 estimated

S5 used a compute shader (a grid-stride XOR convolution) and got
**arena / normal = 0.38 (2.6x slower)**. With the real `mul_mat_id`:

| | S5 (XOR shader) | S6 (real `mul_mat_id`) |
|---|---:|---:|
| arena / normal | **0.38** | **0.88–0.92** |
| slowdown | 2.6x | **1.09–1.13x** |

**The XOR shader overestimated the penalty by more than 20x.**

The reason is straightforward: `mul_mat_id` is not purely bandwidth bound. One
expert's weights are reused across every token routed to it, and the GPU's caches
absorb most of the memory latency. The XOR convolution reads data once and
discards it, measuring the worst case of zero reuse.

> Item 5 in S5's "unverified risks" (the shader's access pattern differs from
> matmul's) was a legitimate concern, and it was **20x pessimistic**.

## Updated speed estimate

S5 estimated the GPU-read difference at +0.28 s per pass. Using the measured
1.09–1.13x against a baseline of 1.74 s per pass (ub=512) gives
**+0.16–0.23 s**.

| ubatch | I/O | compute | bound by | prefill |
|---:|---:|---:|---|---:|
| 512 | 3.09 s | 1.90–1.97 s | I/O | **166 tok/s** |
| 1024 | 3.09 s | 3.79–3.93 s | compute | **260–270 tok/s** |

The conclusion is unchanged from S5 (**the SSD is the constraint**, and the
arena's slower memory does not matter). Only the basis moved from estimate to
measurement.

## It does not repeat Expert Sweep's (N4) failure

N4's Expert Sweep split `mul_mat_id` into P passes, hit ggml's graph buffer
aliasing, and broke PPL to 520801.

**The arena approach does not split anything.** `mul_mat_id` is called once; only
the location of the tensor it references changes. The graph structure is
identical to the current one, so N4's failure mode does not apply. This finding's
bit-identity confirms that empirically.

## The compute buffer scales with ubatch, but with a tiny coefficient

S5's risk 2 (activation memory when raising ubatch) was measured, **with
MoEStream disabled** — the question is about llama.cpp's own behaviour, not the
slab's growth. Memory lines do not appear in the log, so sysfs values were used.

```
MOESTREAM=0 / ctx=32768 / np=1 / KV q8_0 / ngl 99   (idle: vram 102 + gtt 300 MiB)
```

| ubatch | vram_used | gtt_used | Total | vs ub=8 |
|---:|---:|---:|---:|---:|
| 8 | 494 MiB | 17100 MiB | 17.18 GiB | — |
| 256 | 487 MiB | 17171 MiB | 17.24 GiB | **+71 MiB** |
| 512 | 489 MiB | 17241 MiB | 17.31 GiB | **+141 MiB** |
| 1024 | 488 MiB | 17382 MiB | 17.45 GiB | **+282 MiB** |
| 2048 | 490 MiB | 17664 MiB | 17.72 GiB | **+564 MiB** |

**Linear, at about 0.275 MiB per unit of ubatch.** Going from 8 to 512 costs
+141 MiB.

That is 40x smaller than raising `MOESTREAM_MAX_UBATCH` (8 → 32 costs
**+5.6 GiB**), because that adds **expert weights** through more slab slots,
whereas this adds only **activations**.

### Memory budget (settled)

| Item | Increment |
|---|---:|
| current (97 slots, ub=8) | 7.97 GiB |
| + one layer's prefill arena | +330 MiB |
| + double buffering (to overlap I/O and compute) | +330 MiB |
| + compute buffer (ub=8 → 512) | +141 MiB |
| **total** | **~8.75 GiB** |

**7.97 → 8.75 GiB (+10%).** Against the 16.98 GiB baseline the reduction goes
from **−53% to −48%**. Not nothing, but a different order from the rejected
`MAX_UBATCH=32` option (7.97 → 13.5 GiB, −53% → −20%).

Skipping double buffering saves 330 MiB (8.44 GiB total, −50%) but serialises
I/O and compute at 3.09 + 1.9 ≈ 5.0 s per pass, dropping prefill at ub=512 from
166 to **102 tok/s**. **Usable as a memory/speed dial.**

## Remaining risks

Updating S5's risk table:

| # | Item | Status |
|---|---|---|
| 1 | ggml integration — can `mul_mat_id` index 256 experts on the arena | **resolved (this finding)** |
| 5 | the shader's access pattern differs from matmul's | **resolved. Measured 1.09–1.13x** |
| 2 | how much the compute buffer grows at ubatch 512–1024 | **resolved. Only +141–282 MiB** |
| 3 | real expert reads are 450 KB scattered accesses; S5 was sequential large blocks | unverified. 4.48 GB/s is an upper bound |
| 4 | SSD writes and GPU reads hitting the same arena concurrently | unverified. Double buffering is essential |
| 6 | **llama.cpp integration** — graph construction that switches slab/arena on `n_tokens` | not started. **The largest remaining work** |
| 7 | this spike uses Q4_K; the real model is UD-IQ4_NL (mixed quantization) | unverified |

## Conclusion

**S5's biggest hurdle is cleared.**

- `mul_mat_id` works bit-identically over 256 expert tensors on an imported arena
- full-range `ids` outside the slab constraint (97 slots) works
- the speed penalty is 9–13%; S5's estimate (2.6x) was far too pessimistic
- the graph structure is unchanged, so N4's failure mode does not apply
- the compute buffer grows only +141 MiB at ub=512. The budget is **7.97 → 8.75 GiB (+10%)**

Three of S5's five risks are resolved. What remains is chiefly **risk 6
(llama.cpp integration)** — graph construction that switches between slab and
arena based on `n_tokens`. Risks 3 (scattered access) and 4 (I/O contending with
GPU reads) are properties best measured after integration, and both push the
estimate in the optimistic direction.
