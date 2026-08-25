# MoEStream — Measurements

A record of running a 35B-class MoE model on a **single unified-memory machine
(integrated GPU)**, with device memory brought from **17.3 GiB down to 7.4 GiB
(−57%)**.

Everything here is measured and reproducible. Estimates and design targets are
labelled as such. The `Finding` references at the end of each section point at
the primary sources in `docs/findings/`.

---

## 0. Summary

Ornith-1.0-35B-UD-IQ4_NL / `frac=0.25` / `UBATCH=1024` / `CTX_SIZE=32768`
(re-measured 2026-08-24 on llama.cpp `b0539c43`; conditions in §12.4b)

| Metric | Baseline | MoEStream 0.25 | |
|---|---:|---:|---:|
| **device memory** | 17.77 GiB | **7.93 GiB** | **−55%** |
| decode | 23.4 tok/s | **16.5 tok/s** | **−29%** |
| prefill (13877-token real document) | 256.2 tok/s | **239.4 tok/s** | **−7%** |
| TTFT on a continuing turn | 1.06 s | **1.34 s** | 79% |
| quality (PPL, wiki2) | 4.4400 | **4.4494** | **+0.21%** |

Prefill was originally the one weak point, at 4.9x worse (59.7 tok/s). The
**arena (S7) plus union reads (S12) plus asynchronous prefetch (S14) recovered
it from 46.0 to 242.5 tok/s — 5.3x** — at a cost of 0.9 GiB. Generated tokens are
bit-identical under greedy decoding to plain llama.cpp with every expert
resident (though token identity is a weaker claim than numerical identity; §8.3).

**Only Ornith can be compared against plain llama.cpp.** Qwen3-Coder-Next
(36.5 GiB), Laguna-S-2.1 (54.7 GiB) and gpt-oss-120b (58.5 GiB) exceed this
machine's 23.5 GiB GTT limit and **do not start**. Under MoEStream they run in
13.44 GiB / 14.8 tok/s, 18.79 GiB / 3.1 tok/s and 14.91 GiB / 3.8 tok/s
respectively (§12.4b).

The most reusable finding in this project is not a performance number. It is
this:

> **The bottleneck moved as the implementation improved, and we only noticed by
> re-measuring.**
> The first measurement (2026-08-04) gave synchronization **11.4 ms** against
> I/O **0.5 ms**, and for a long time this project's headline was "the
> bottleneck is synchronization, not I/O". With union reads, async prefetch and
> zero-copy all in place, the same measurement now gives **synchronization
> 2.2 ms against I/O 7.8–12.5 ms** — **inverted** (§10.12).
>
> The original claim is still true of the code it was measured on. What was
> wrong was **leaving the headline unchanged after the implementation changed.**

Published work in this area (Klotski, ProMoE, MoE-Infinity) is built on the
premise that SSD bandwidth is short and clever prefetching is therefore the
answer. At 30B class on unified memory, that premise does not hold (§4, §7).

> **[Added 2026-08-23]** Two things in this document were overtaken by a later
> round of measurement, and both are worth knowing before reading further.
>
> **"I/O" is two different costs sharing one name.** On Ornith-1.0, **98.7% of
> decode reads never reach the SSD** — they are served from the page cache, and
> the cost is a copy into GPU-visible memory rather than a device read. On a
> model whose experts do not fit in RAM the same code pays the NVMe. Effective
> read bandwidth differs 2.5x between the two, with no code difference
> (`findings/S19-pagecache-share.md`).
>
> **Dense models are supported, and the claim that they could not be was wrong.**
> `B_act = total size` holds per forward *pass*, not per token, so prefill
> amortises it away. Measured on Qwen3.8-27B: **16.49 → 7.81 GiB (−53%) with
> byte-identical output, perplexity identical to four decimals, and prompt
> processing within 2.2% of plain llama.cpp.** Generation costs 3.2x, which is
> the honest shape of the trade (`findings/S27-dense-streaming-impl.md`).

---

## 1. Measurement environment

| Item | Value |
|---|---|
| CPU | AMD Ryzen 7 8745HS (8C/16T) |
| GPU | Radeon 780M (integrated, RDNA3) / Vulkan RADV |
| Memory | 32 GiB DDR5 (UMA — shared between CPU and GPU) |
| SSD | Crucial P310 (PCIe 4.0 NVMe) |
| OS | Linux 7.0.0-28-generic |
| How it runs | Docker (`docker compose up -d`) — nothing installed on the host |

### Model

| Item | Value |
|---|---|
| File | `Ornith-1.0-35B-UD-IQ4_NL.gguf` |
| Architecture | `qwen35moe` |
| Parameters | 34.66 B |
| Layers | 40 |
| Experts per layer | 256 (routed) + 1 (shared) |
| top-k | 8 |
| Quantization | gate/up = IQ3_S, down = IQ4_NL, some Q6_K |
| File size | 16.86 GiB |

### Tensor classification (measured)

All 733 tensors, classified by `research/tools/analysis/gguf_inspect.py`:

| Class | Tensors | Size | Share |
|---|---:|---:|---:|
| **STREAMED** (routed experts) | 120 | **14.48 GiB** | 85.9% |
| **ROW_LOOKUP** (token_embd) | 1 | 515.31 MiB | 3.0% |
| **RESIDENT** (attention, norm, shared expert, …) | 612 | 1.88 GiB | 11.1% |
| Total | 733 | 16.86 GiB | |

**86% of the model is routed experts.** That is the basis for the whole design —
move the experts out and the reduction is dramatic — and it worked out that way
(resident RESIDENT + ROW_LOOKUP comes to 2.4 GiB).

> Finding: `S1-real-expert-stream.md`

---

## 2. Memory

`MOESTREAM_CACHE_FRAC` is the fraction of experts kept resident.

| frac | slots/layer | device memory | decode | Note |
|---:|---:|---:|---:|---|
| 1.00 | 256 | 16.98 GiB | 42.66 ms | equivalent to no streaming |
| 0.60 | 154 | ~11 GiB | 53.1 ms | worse again above 0.60 |
| **0.38** | **97** | **7.97 GiB** | **53.64 ms** | **recommended at the time** |
| 0.15 | 39 | 4.6 GiB | 90.6 ms | hit rate falls and it starts to break down |

Approximation: **device memory ≈ 2.4 GiB + 14.5 GiB × frac**

38% was chosen because **0.38 and 0.60 decode at almost the same speed
(53.6 vs 53.1 ms)**. Raising frac does not make it faster — which was the first
sign that I/O was not the constraint.

### Why above 0.60 it gets slower

A larger slab raises `as->ne[2]` in `mul_mat_id`, which raises the setup cost of
Vulkan's indirect dispatch. The point where that increase outweighs the I/O
saved is around frac ≈ 0.6.

---

## 3. Speed

### 3.1 Decode (steady state)

The second half of a 150-token generation, median of 3 runs each:

| Configuration | ms/token | tok/s | Ratio |
|---|---:|---:|---:|
| baseline (all resident) | 42.66 ± 0.04 | 23.44 | 100% |
| **MoEStream 38% (final)** | **53.64** | **18.64** | **79.5%** |
| MoEStream 38% (sequential I/O) | 85.64 | 11.68 | 50% |
| MoEStream 15% | 90.6 | 11.0 | 47% |

**Sequential I/O to 8-thread parallel `pread` took it from 85.64 to 66.0 ms
(−23%); later optimizations reached 53.64 ms.**

### 3.2 Where the 11.0 ms difference goes

The 11.0 ms between `42.66` and `53.64 ms` was separated by disabling stages
with `MOESTREAM_NOOP`:

| Configuration | ms/token | Increment at this stage |
|---|---:|---:|
| baseline | 42.66 | — |
| remap node only (I/O fully disabled, `NOOP=3`) | **54.11** | **+11.45** |
| full (remap + real I/O) | 54.64 | **+0.53** |

> These are consecutive measurements within one session at a fixed
> cache_frac=0.38. The 53.64 ms in §3.1 is a median of 3 from a different
> session, hence the ~1 ms difference.

**I/O costs only 0.53 ms/token. All 11.45 ms is CPU↔GPU synchronization.**

> **[Added 2026-08-07] This measurement is correct for the code it was taken on
> and does not describe the current code.** With union reads (S12), async
> prefetch (S14) and zero-copy complete, the same `NOOP=3` difference fell from
> 11.45 to **4.5 ms**, and I/O rose from 0.53 to **7.8–12.5 ms**. **I/O is now
> 3.5–5.6x larger than synchronization.** See §10.12.

Breakdown of the synchronization cost:

```
per layer  0.285 ms  (fetch the router's ids GPU->CPU, remap them to slots,
                      write them back to the GPU)
x 40 layers = 11.4 ms
```

The dependency `router_L -> remap_L -> ffn_L -> attn_{L+1}` is strictly serial,
so **that 11.4 ms was a structural floor for the architecture at the time.**
(In the current code, synchronization is about 2.2 ms; §10.12.) Hiding it would
require doing the remap on the GPU — that is, adding slot indirection to ggml.

> Findings: `N2-speed-optimization.md`, `N3-graph-remap.md`, `V2-idea-analysis.md`

### 3.3 Prefill

| Configuration | ubatch | prefill tok/s |
|---|---:|---:|
| baseline | 512 | 295.1 |
| baseline | 8 | 74.9 |
| MoEStream 38% | 8 | 59.7 |
| MoEStream 38% | 24 | 75.0 |
| MoEStream 38% | 32 | 81.1 |

**The decomposition matters.** Of the 4.9x degradation from 295.1 to
59.7 tok/s:

- **3.9x came from dropping ubatch from 512 to 8**
- **only 1.25x came from MoEStream itself**

ubatch was small in order to respect the slab constraint: *the number of
distinct experts within one micro-batch must not exceed the slot count*. So the
prefill penalty was not from streaming — **it came from the slab size limit.**

Raising ubatch recovers it, but needs more slots and therefore more memory.
**That trade was declined** (going from −53% to −20% of memory defeats the
purpose):

| MOESTREAM_MAX_UBATCH | slots/layer | Memory | Prefill |
|---:|---:|---:|---:|
| **8** | 97 | **7.9 GiB** | 59.7 tok/s |
| 24 | 165 | 11.7 GiB | 75.0 tok/s |
| 32 | 195 | 13.5 GiB | 81.1 tok/s |

> Findings: `N4-expert-sweep.md`, `V2-idea-analysis.md`

### 3.4 The prefill staging arena — removing the slab constraint

A way to **lift the ubatch limit without adding memory**.

For the duration of prefill only, computation reads from an **arena holding one
layer's full expert set**. One arena is allocated (498 MiB measured) and reused
layer by layer. The resident slab (97 slots, for decode) does not grow at all.

```
decode  (n_tokens <= 11) : slab as before, ids remapped to slot_id
prefill (n_tokens >  11) : arena with all 256 experts, ids left as expert_id
```

The threshold of 11 is computed: the largest n satisfying
`union(n) × 1.15 + top_k ≤ n_slot` — that is, **the most the slab can serve
without exhaustion**. With that, the slab no longer constrains ubatch, and
`-ub 512` became usable.

Measured (13877-token prompt, np=1, KV q8_0):

| Configuration | ctx=32768 | ctx=100000 (the real setup) |
|---|---|---|
| plain llama.cpp (ub=1024) | 244.4 tok/s / 17.44 GiB | — |
| MoEStream, no arena (ub=8) | 46.0 tok/s / 8.22 GiB | ~42 tok/s |
| MoEStream, arena (ub=512) | 140.3 tok/s / 8.84 GiB | 110.2 tok/s / 9.65 GiB |
| **MoEStream, arena (ub=1024)** | **194.2 tok/s / 8.98 GiB** | **188.6 tok/s / 9.92 GiB** |
| MoEStream, arena (ub=2048) | 199.3 tok/s / 9.26 GiB | 198.0 tok/s / 10.45 GiB |

**Prefill 4.2–4.5x, for 0.76–0.9 GiB.** That puts it at 79% of the speed for
51% of the memory.

It plateaus at `UBATCH=1024` because **the constraint moved from I/O to
compute**. The arena reads every expert (12.9 GiB) once per pass, so the number
of passes (prompt length / UBATCH) is the I/O volume. By 1024 the I/O is faster
than the compute and there is nothing left to gain (2048 buys +5% for double the
memory).

Correctness was checked with greedy decoding (temperature=0). Across three
prompt lengths, **identical requests reproduce identically**, and **generated
tokens are bit-identical to plain llama.cpp with all experts resident**.

Two bugs were hit on the way. Both generalise:

1. **Imported host memory slows decode 19x even when unused.**
   A 498 MiB BO imported via `VK_EXT_external_memory_host` took decode from
   53.8 to 1023 ms/token without ever entering the graph or being read.
   → Stop importing; let ggml-vulkan allocate normally (on UMA it returns
   heap 1 `DEVICE_LOCAL|HOST_VISIBLE`, and the existing zero-copy path works
   unchanged).
2. **`selected_experts` is sometimes a view.** A flat `memcpy` of the ids
   passes wrong expert_ids, and **the output looks correct while differing on
   every run**.
   → Access with the `nb[0]`/`nb[1]` strides.

#### The arena did not need to read every expert (Finding S12)

`load_layer_arena()` originally always read all experts. But the arena path is
what handles the **38–500 token deltas that dominate agent use**, where most of
that read is wasted.

Before the change, **201 and 401 tokens cost almost the same (4.80 / 4.65 s)** —
the fixed cost of reading all 12.9 GiB dominated everything.

Reading only the experts actually referenced by `ids`:

| Delta tokens | Before | After | Reduction |
|---:|---:|---:|---:|
| 11 | 2.06 s | **1.25 s** | −39% |
| 41 | 3.02 s | **1.75 s** | −42% |
| 201 | 4.80 s | **2.24 s** | −53% |

Bytes read fell **76% on Ornith** and **90% on Laguna**. Output is bit-identical
to plain llama.cpp under greedy decoding.

The prior estimate (from the union formula) predicted −27% at 41 tokens and 0%
at 201. Measurement beat it: **real expert selection is more skewed than the
formula assumes.**

#### Overlapping arena loads with GPU compute (Finding S14)

Arena loads were synchronous. Reading layer L+1 while layer L computes overlaps
them.

**Unlike decode prefetch (§7, rejected), this is safe.** The destination is a
**plain buffer**, not the `ExpertCache` — no refcounts, no eviction. No
prediction is needed either. And because anything not already present is read
synchronously once the ids are known, **whatever the prefetch read, the output
is unchanged.**

At prefetch time the next layer's ids do not exist yet, so two strategies were
measured:

| | synchronous (before) | read all | union |
|---|---:|---:|---:|
| **Ornith** | 224.75 tok/s | **246.90 (+9.9%)** | 233.76 (+4.0%) |
| **Laguna** | 46.81 tok/s | 62.54 (+33.6%) | **80.12 (+71.2%)** |

**The better strategy inverts between models.** Ornith fits in page cache and
reads cheaply (9 GB/s), so waiting on the synchronous top-up costs more than
over-reading. Laguna reads from the real SSD (3.6 GB/s), so reading less beats
waiting. **The same shape as the I/O thread count.**

The existing `[prefetch]` decision is reused to switch automatically:

| | before | auto-switching | Improvement | Extra memory |
|---|---:|---:|---:|---:|
| Ornith | 224.75 tok/s | **248.59 tok/s** | **+10.6%** | +0.49 GiB |
| Laguna | 46.81 tok/s | **80.73 tok/s** | **+72.5%** | +1.47 GiB |

> Findings: `S5-prefill-arena.md`, `S6-arena-mulmatid.md`, `S7-prefill-arena-impl.md`, `S12-arena-union-load.md`, `S14-async-arena-prefetch.md`

---

## 4. SSD bandwidth — refuting the "prefetching is necessary" premise

### 4.1 What the device can actually do

Measured by `research/spikes/s1_real_expert_stream` and
`research/spikes/s2_iouring_bw`.
**Figures taken through the page cache are a measurement trap; these were
re-taken with O_DIRECT.**

| Method | Bandwidth |
|---|---:|
| through page cache | 14.23 GB/s ← **not the disk. This number is unusable** |
| O_DIRECT synchronous pread QD=1 | 1.55 GB/s |
| O_DIRECT QD=2 | 2.85 GB/s |
| O_DIRECT QD=4 | 4.28 GB/s |
| **O_DIRECT QD=8** | **4.42 GB/s** ← saturates here |
| O_DIRECT QD=32 | 4.46 GB/s |
| **io_uring (best)** | **4.49 GB/s (+1.5%)** |

### 4.2 io_uring is worth only +1.5% in bandwidth

| Method | Bandwidth |
|---|---:|
| synchronous pread QD=1 | 1.55 GB/s |
| parallel pthread pread QD=8 | 4.42 GB/s |
| io_uring | 4.49 GB/s (**+1.5%**) |

**4.48 GB/s was the device's limit, not the API's.** The design had io_uring as
a requirement; measurement showed parallel `pread` is enough. The current
implementation uses 8 pread threads.

That is not to say io_uring is worthless — its value is **latency control, not
bandwidth**:

| QD | Bandwidth | p99 latency |
|---:|---:|---:|
| 2 | 4.48 GB/s | 1.19 ms |
| 8 | 4.46 GB/s | 3.39 ms |
| 128 | 4.41 GB/s | **95.02 ms** |

**Same bandwidth, but going from QD 2 to 128 multiplies p99 by 80.** Registered
buffers improve p99 from 5.18 to 4.42 ms (−15%).

### 4.3 4 KiB alignment in `.msp` does not improve bandwidth

Design doc §18.4 proposed paying 2.7% more disk for 2 MiB alignment. Measured,
the read-around overhead of reading GGUF directly is only **0.27%**, and the
bandwidth difference was within noise (+0.1%). **That design decision was
dropped.**

The other `.msp` advantages remain valid (one expert = one `readv`, against
three for GGUF, so a third of the IOPS).

> Findings: `S1-real-expert-stream.md`, `S2-iouring-bandwidth.md`

---

## 5. Expert activation distribution and caching

### 5.1 Required hit rate

The go/no-go condition, derived from measured values:

```
B_act (expert bytes touched per token)  = 455 MiB/token
t_c   (compute time per token)          = 43.7 ms
BW    (effective bandwidth)             = 4.46 GB/s

-> to stay within a 20% slowdown, hit rate h >= 51.0%
```

### 5.2 Measured hit rate (38% cache)

From real traces across three domains (code / Japanese / English):

| Domain | Zipf exponent `s` | oracle (static hot set) | **LRU (dynamic)** | Verdict |
|---|---:|---:|---:|---|
| code | 0.473 | 87.7% | **88.7%** | GO |
| ja | 0.340 | 87.9% | **91.5%** | GO |
| en | 0.283 | 82.1% | **90.2%** | GO |

**Against a requirement of 51.0%, the worst domain gives 88.7% — about 1.7x of
headroom.**

### 5.3 Miss ratio curve

| Cache ratio | code | ja | en |
|---:|---:|---:|---:|
| 5% | 45.7% | 37.6% | 31.2% |
| 10% | 60.4% | 52.3% | 46.0% |
| 20% | 74.7% | 70.5% | 63.7% |
| **38%** | **87.7%** | **87.9%** | **82.1%** |
| 50% | 92.5% | 93.4% | 89.4% |
| 75% | 98.2% | 98.5% | 97.5% |

(Oracle values. LRU at 38% gives code 88.7 / ja 91.5 / en 90.2.)

**The curve has a clear knee around 38%**, which is where the recommendation
came from.

### 5.4 Slots can be divided evenly across layers (2026-08-06)

Slots are distributed evenly (`slab_slots()` does not look at the layer index).
The design document's §12.5 provided for dynamic per-layer quota reallocation,
which is disabled. **How much that costs** was measured per layer from real
workload reuse distances.

Holding the total slot count fixed and allocating optimally by taking the
**upper concave hull** of each layer's cumulative curve:

| Model | Even split | Optimal upper bound | Difference | Slots when optimal |
|---|---:|---:|---:|---|
| Ornith-35B (64 slots) | 82.28% | 83.30% | **+1.02 pt** | 44–126 (even: 64) |
| Laguna-S-2.1 (51 slots) | 65.96% | 67.06% | **+1.11 pt** | 41–79 (even: 51) |

**About one point, and that is the linear-relaxation upper bound** — integer
allocation cannot reach it. In time that is roughly 0.2 ms/tok on Ornith and
3 ms/tok on Laguna. Not worth making `slab_slots()` layer-aware and re-verifying
threshold computation and slot-exhaustion safety per layer. **The even split
stays.**

Disabling §12.5 turned out to have been the right call.

> Note: the first implementation was a naive greedy allocator, and it produced
> the impossible result that **the optimization was 11.69 pt worse than the even
> split**. Reuse-distance histograms are not monotonically decreasing, so
> marginal gain is non-monotonic and greedy loses optimality (it was leaving
> layers with zero slots). Fixed by taking the upper concave hull first.
> **It was noticeable only because "the optimum is worse than the baseline" is
> physically impossible.**

### 5.4c S3-FIFO already beats LRU (2026-08-06)

The MRC derives an exact **LRU-equivalent** hit rate from reuse distance. Since
the real cache is S3-FIFO, comparing both on the same run shows **whether there
is anything left in the replacement policy**.

| | real cache (S3-FIFO) | LRU-equivalent from MRC | Difference |
|---|---:|---:|---:|
| Ornith (64 slots) | **82.88%** | 81.95% | **+0.93 pt** |
| Laguna (51 slots) | **67.91%** | 66.38% | **+1.53 pt** |

**S3-FIFO is already ahead of LRU.** There is nothing to gain from tuning the
policy (`FREQ_MAX`, `ghost_mult`, segment ratios).

> A side effect: the `[mrc]` recommendation is **slightly conservative**. The
> real cache runs 1–1.5 pt better than LRU, so the table is pessimistic. That
> only errs toward safety, so it is not corrected.

### 5.4b Dynamic caching beats a static PINNED set

Design doc §12.6 specified a 20% statically pinned set built from calibration
statistics. Measurement said the opposite:

| Pin ratio | Hit rate |
|---:|---:|
| **0.00** | **90.53%** |
| 0.05 | 90.47% |
| 0.20 (the old design default) | **88.80%** |

**More pinning is strictly worse.** Dynamic LRU/S3-FIFO wins throughout; a
static hot set merely freezes what the dynamic cache would have found anyway,
and loses whatever adaptivity it removes. **The §12.6 default was changed from
20% to 5%.**

In some domains LRU even beats the oracle — a static hot set chosen with
knowledge of the future:

| Domain | Oracle | LRU | Difference |
|---|---:|---:|---:|
| code | 87.7% | 88.7% | +1.0 pt |
| ja | 87.9% | 91.5% | **+3.6 pt** |
| en | 82.1% | 90.2% | **+8.1 pt** |

The Zipf exponent for `en` is a shallow 0.283, which leaves a static hot set
little advantage.

### 5.5 Mixed domains

A coding agent moves between Japanese, English and code. The worst case was
measured:

| Cache ratio | single domain | **three domains fully interleaved** | Difference |
|---:|---:|---:|---:|
| 20% | 76.9% | 64.6% | −12.3 pt |
| **38%** | 89.9% | **82.4%** | **−7.6 pt** |
| 50% | 94.0% | 89.4% | −4.7 pt |

**Even at worst, 82.4% — far above the 51.0% required.**

Hot set overlap (38% cache):

| Pair | Jaccard |
|---|---:|
| code ∩ ja | 55.5% |
| code ∩ en | 56.8% |
| ja ∩ en | **70.4%** |

Natural languages share 70% with each other; only code is somewhat separate.

> Findings: `M0-2-expert-distribution.md`, `N1-cache-replay.md`

---

## 6. In practice — multi-turn

**This is the measurement in this repository that bears most directly on real
use.**

`research/bench/agent_turns.py`: five questions asked while holding roughly
2,000 tokens of code in context — how a coding agent is actually used.
llama-server's prompt cache (`--cache-ram`) applies.

### MoEStream 38% (final configuration, measured)

| Turn | Tokens evaluated | prefill (s) | decode tok/s | Total (s) |
|---:|---:|---:|---:|---:|
| 1 | 2254 | 41.22 | 17.98 | 44.10 |
| 2 | **18** | **0.87** | 19.19 | 3.37 |
| 3 | **16** | **0.59** | 19.04 | 3.11 |
| 4 | **19** | **0.93** | 19.25 | 3.43 |
| 5 | **17** | **0.86** | 18.86 | 3.41 |

### Against the baseline

| Metric | Baseline | MoEStream 38% | Ratio |
|---|---:|---:|---:|
| first TTFT (turn 1) | 9.96 s | 41.2 s | 4.1x slower |
| **continuing TTFT (turn 2 on)** | **1.06 s** | **1.34 s** | **1.27x** |
| decode | 23.4 tok/s | 18.6 tok/s | 1.26x |

> This section was measured **before** the prefill arena (§3.4). With the arena,
> the 4.1x on the first TTFT shrinks to about **1.8x**, so the cost of "waiting
> once at the start of a session" is much smaller than it appears here. For
> coding agents that paste large tool outputs, evaluated tokens still reach the
> thousands even with the prompt cache working, so this path matters often.

**When the prompt cache hits, evaluated tokens drop from 2254 to 16–19.** With
a hundredth of the prefill to do, MoEStream's prefill penalty (4.1x) all but
vanishes on continuing turns, and the perceived difference converges on the
decode gap (1.26x).

**Wait an extra 31 seconds once at the start of a session, and after that it is
20% slower for half the memory.** That is the conclusion for 30B-class MoE on a
unified-memory machine.

> Configured by `CACHE_RAM=2048` (MiB) in `.env`. llama.cpp's default of 8192 is
> too large for a 24 GB-class machine.

---

## 7. Predictive prefetch — "conditions where it helps" and "a means that helps" are different claims

This is the most winding chapter in the project. The conclusion first:

> **Predictive prefetch is not used.**
> Not because prediction does not work. The conditions where it would help are
> real (I/O-bound models), and the accuracy was sufficient (71–81%).
> **There was no means to exploit it** — nothing implementable safely could
> capture the headroom.

It was investigated in three stages.

---

### 7.1 Stage 1: all three schemes are net-negative on Ornith-35B (findings N2/N3)

| Predictor | Accuracy | Best ms/token | Verdict |
|---|---:|---:|---|
| P1 previous-token reuse | — | — | ❌ **never issued a single fetch** |
| P2 layer lookahead `W_{L+1}·h_L` | **81.4%** | 97.0 | ❌ slower than baseline |
| P5 cross-layer co-activation | 18.5–40.0% | 69.9 | ❌ not accurate enough |

**Why P1 never issued anything**

An expert used on the previous token is **by definition already cached** — that
is what reuse means. The actual misses are the 62% newly selected, which P1
cannot predict in principle. M0-2's "38% previous-token reuse" did not imply a
useful predictor.

> This is structural and independent of conditions. **It does not come back in
> a different regime.**

**Why P2 is slow at 81.4% accuracy**

| Cost component | Value |
|---|---:|
| matrix multiply | 3.3 ms/token |
| fetching probs/hidden GPU→CPU | 7.5 ms/token |
| scheduler cost of requesting `probs` via callback | ~22 ms/token |
| total | **~33 ms/token** |

Decode at the time was 54.6 ms, of which **I/O was 0.53 ms** (§3.2). **Paying
33 ms to hide 0.5 ms** cannot win.

---

### 7.2 Stage 2: change the conditions and the premise changes (findings S9/S10)

§7.1 was measured where **I/O is 1% of decode**. Once the model no longer fits
in host RAM, that premise collapses.

| | Ornith-35B | Laguna-S-2.1 (54.7 GiB) |
|---|---:|---:|
| decode (with I/O) | 54.6 ms | 361.8 ms |
| decode (no I/O, `MOESTREAM_NOOP=3`) | 54.1 ms | 89.4 ms |
| **cost of I/O** | **0.53 ms (1%)** | **272 ms (75%)** |

**What there is to hide grew 500x.** So two things were re-measured.

**Where the headroom is (S9)**

A spike reproducing the real pattern (458 scattered reads averaging 1.44 MiB per
token):

| Method | Bandwidth |
|---|---:|
| (A) current — burst per layer, with compute gaps | 2.10 GB/s |
| (B) remove only the compute gaps | 4.09 GB/s |
| (C) continuous queueing | **4.48 GB/s** |

Scattered reads still reach the device limit. **95% of the improvement comes
from not leaving the SSD idle during compute** (A→B is 1.95x, B→C only 1.10x).

**What about accuracy (S10)**

Re-measured on Laguna:

```
P2  W_(L+1)·h_L -> layer L+1 : 71.05%   (18x random)
P5  layer L's picks -> L+1   :  3.97%   (random is 3.91% — zero information)
```

P2 holds up (65–82% in the middle layers).
**P5 was not "insufficiently accurate" — it had no information at all**, which
is worse than the N2 record suggested.

The estimate at this point was **1.8x**.

---

### 7.3 Stage 3: implemented, and 19% worse (finding S11)

Implemented as the P2 predictor plus `posix_fadvise(WILLNEED)`. Writing
directly into the slab from a background thread was **deliberately avoided**: it
breaks the `ExpertCache` refcount invariants, which is the same class of quietly
corrupting bug as N4 and S7.

| | prefetch off | prefetch on |
|---|---:|---:|
| decode | **361.8 ms/tok** | **429.3 ms/tok** |
| effective bandwidth | 3.02 GB/s | 3.20 GB/s (+6%) |
| prediction cost | — | 69.4 ms/token |

**Even if prediction were free, the gain would be 3.6%.** Two errors:

1. **The prediction cost was underestimated 12x.** Only the arithmetic (37M MAC)
   was counted; **reading the 141 MiB router matrix every token** was not. That
   is memory-bandwidth bound, not compute bound.
2. **`fadvise` does not produce continuous queueing.** S9's 4.48 GB/s was the
   figure for *maintaining your own queue*; a hint to the kernel gave +6%.

> S9's "2.13x of headroom" was not wrong. The error was **believing `fadvise`
> could capture it.** The headroom was specifically "keep your own queue full",
> which cannot be delegated to the kernel.

**The one remaining route — a dedicated thread writing into the slab — has a low
ceiling too.** At 71% accuracy, the 29% of wasted transfers compete for the same
SSD bandwidth:

```
transfer needed              658 MiB/token
issued at 71% accuracy       658 / 0.71 = 926 MiB/token   (40% more)
perfectly overlapped, decode ~ 250 ms  ->  1.45x from 361.8
```

**S9's 2.13x assumed 100% accuracy; the real ceiling is about 1.45x.** Against
that stand three "corrupts quietly" risks: refcount leaks, slot reallocation
races, and partial writes. **Not worth it, so it was not adopted.**

The implementation was deleted. Only the decision logic (is this model I/O
bound?) remains, for diagnostics.

---

### 7.5 The same root, a different failure: speculative decoding (finding S13)

llama.cpp's built-in speculative decoding was tried too. It **made things
worse**.

| Configuration | Memory | decode |
|---|---:|---:|
| baseline | 6.59 GiB | **57.8 ms/tok** |
| ngram-cache (no extra memory) | 6.59 GiB | 77.9 ms/tok (−35%) |
| draft-simple (Qwen3.5-4B) | 9.80 GiB | 281.0 ms/tok (−79%) |

Acceptance rate 34.6% (0.87 accepted on average out of 4 speculated). The
verification of the rejected tokens is wasted, and **I/O per token roughly
quadruples**.

> **In an I/O-bound system, every technique of the form "do more computation to
> reduce serialisation" backfires.**
>
> | | What it tried to do | Result |
> |---|---|---|
> | S11 prefetch | **hide** I/O | no means to hide it |
> | S13 speculation | **add** compute to reduce serialisation | I/O went up 4x |
>
> Both assume compute is in surplus. In MoEStream what is scarce is not compute
> but **SSD bandwidth**.

> Finding: `S13-speculative-decoding.md`

### 7.4 What to take from this chapter

> **"There are conditions where it helps" and "there is a means that helps" are
> different claims.**

§7.1's conclusion (net-negative) is correct for the conditions measured. The
error was detaching it from those conditions and treating it as a general rule.

§7.2's conclusion (2x of headroom) is correct as a measurement. The error was
assuming **any** means could capture it. The spike measured "the performance of
self-managed queueing", not "the performance of prefetching as a concept".

Proceeding in stages was right. Decision logic → accuracy measurement →
implementation meant we **stopped before starting on the most dangerous option**
(concurrent writes into the cache).

> Findings: `N2-speed-optimization.md`, `N3-graph-remap.md`,
> `S9-prefetch-headroom.md`, `S10-p2-accuracy.md`, `S11-prefetch-fadvise.md`

---

## 8. Quality

`llama-perplexity` / wikitext-2:

| Configuration | PPL | Difference |
|---|---:|---:|
| baseline | 4.4400 | — |
| MoEStream 38% | 4.4494 | **+0.21%** |

### 8.0b Re-verified on the newer llama.cpp, and on the dense path (2026-08-24)

Everything below was measured on `3581ba0cf`. After the default moved to
`b0539c43` — and after a whole dense streaming path was added — quality was
measured again, same corpus and flags as §8.1 (`research/bench/ppl.txt`, 30
chunks, `-c 512 -b 512 -ub 512`, `ngl 99`):

| | PPL |
|---|---:|
| Ornith-1.0, `MOESTREAM=0` | **5.0919** |
| Ornith-1.0, `frac=0.25` | **5.1040** |
| Qwen3.8-27B dense, `MOESTREAM=0` | **4.2000** |
| Qwen3.8-27B dense, whole FFN streamed | **4.2000** |

**The MoE figures reproduce exactly** — the same four decimals as before. Three
weeks of upstream llama.cpp changed speed and memory (§12.4b) and did not move
quality by a digit.

**The dense figures close a hole the documentation had left open.**
[`USAGE.md` §5](USAGE.md) listed the streaming path's perplexity as "not
separately measured", and the dense claims elsewhere rested on that gap. It is
measured now: streaming the entire feed-forward block changes perplexity by
nothing at four decimal places.

---

### 8.1 Re-testing — noise, or a real difference? (2026-08-07)

To settle whether that +0.21% is measurement noise or systematic, the same
conditions were run **multiple times** to establish the measurement's own
variance.

Conditions: Ornith-1.0-35B-UD-IQ4_NL / `research/bench/ppl.txt`, 30 chunks /
`-c 512 -b 512 -ub 512` / `ngl 99`. (A different corpus from wikitext-2, so the
absolute PPL differs from the table above.)

| Configuration | Run 1 | Run 2 | Run 3 | vs baseline |
|---|---:|---:|---:|---:|
| baseline (`MOESTREAM=0`) | 5.0919 | 5.0919 | 5.0919 | — |
| MoEStream `frac=0.25` (64 slots) | 5.1040 | 5.1040 | — | **+0.238%** |
| MoEStream `frac=1.00` (256 slots) | 5.1040 | — | — | **+0.238%** |

**Two conclusions.**

**(1) The measurement variance is zero.** Three runs under identical conditions
agreed to four decimal places. So the +0.24% is **a reproducible systematic
difference, not noise.** It cannot be waved through as "within error".

> This re-test caused the previously documented "MoEStream (separate
> measurement) +0.11%" to be **deleted**. If the measurement reproduces this
> exactly, +0.21% and +0.11% cannot both occur under the same conditions. That
> line was two differently conditioned measurements placed side by side without
> saying so.

**(2) Degradation from streaming itself measures as zero.**
At `frac=1.00` the slot count equals `n_expert`, so **eviction cannot happen**.
That configuration and `frac=0.25`, where eviction happens constantly, produced
**exactly the same PPL**.

```
frac=0.25 (misses)     5.1040
frac=1.00 (no misses)  5.1040   <- identical
```

**Fetching experts from the SSD does not change the result at all.** The cache
mechanism is lossless.

### 8.2 So where does the +0.24% come from?

The slab's `ne[2]` changes from `n_expert` (256) to `n_slot + 1`, which changes
**the floating-point reduction order** inside `mul_mat_id`. Not one bit of any
expert weight is altered — no requantization, no approximation.

Supporting evidence:

- **The same difference appears at `frac=1.00`, where nothing is ever evicted**
  (§8.1). With zero misses, neither I/O nor the cache can be the cause.
- Slot remap alone is verified **bit-identical** on three paths: CPU, Vulkan and
  a real GGUF.

**The PPL difference is therefore numerical error from reduction order, not
accuracy lost to cache misses.**

### 8.3 Even so, this is not written up as "zero degradation"

Knowing the cause is numerical error is not the same as there being no
difference. Precisely stated, in two sentences:

> **Degradation from SSD streaming measures as zero** (frac 0.25 and 1.00 agree
> exactly).
> **But there is a +0.24% systematic difference from plain llama.cpp that
> reproduces 100% of the time and does not go away.**

Note also that "generated tokens are bit-identical under greedy decoding" is
**not proof of numerical identity**. `argmax` does not change ranking under tiny
numerical differences, so token identity is a **weaker** claim. In fact both
hold at once: identical tokens, and a 0.24% PPL difference.

### What actually does break quality — slot exhaustion

If the number of distinct experts within one micro-batch exceeds the slot count,
some expert cannot be loaded. Getting this wrong is catastrophic:

| Situation | PPL |
|---|---:|
| normal | 4.44 |
| writing the wrong expert (slot 0) on exhaustion **(a bug)** | 2365 |
| Expert Sweep (P=2) enabled | 520801 |

The current implementation writes the **zero slot** on exhaustion and warns.

Furthermore, **because the prefill arena (§3.4) is on by default, any ubatch
beyond what the slab can safely handle (measured: 11 tokens) is routed to the
arena automatically.** So `UBATCH` and `MOESTREAM_MAX_UBATCH` no longer have to
agree. Only if the arena is explicitly disabled
(`MOESTREAM_PREFILL_ARENA=0`) does `src/entrypoint.sh` verify the match at
startup and refuse to run.

> Finding: `N4-expert-sweep.md`

---

## 9. Expert Sweep — did not work; cause identified and disabled

The headline feature of design doc §20.2. The idea was to run experts in
multiple passes, mapping out-of-pass experts to the zero slot, and exploit the
commutativity of `Σ wᵢEᵢ(x)` to lift the slab constraint.

**The speed was there**: prefill 59.1 → 173.7 tok/s (2.9x), memory unchanged at
7.9 GiB.
**The output was not**: PPL 4.44 → **520801**.

Rather than stop at "it got faster", the cause was traced to the end.

### Elimination

| Hypothesis | How it was tested | Result |
|---|---|---|
| the zero slot is not actually zero | `research/spikes/s3_zero_slot`: CPU dequantize + Vulkan `mul_mat_id` for IQ3_S/IQ4_NL/Q6_K/Q4_K | **cleared**. Exactly zero for every type |
| split addition is not mathematically valid | `research/spikes/s4_sweep_math`: minimal case, `FFN(ids₀)+FFN(ids₁) == FFN(ids)` | **cleared**. Holds on CPU and Vulkan |
| cache eviction | slots=256 (no eviction) vs 165 | **cleared**. PPL identical (520801.5849) |
| execution order / races | order trace via marker op, same conditions twice | **cleared**. remap0→FFN0→remap1→FFN1, PPL identical |
| **graph aliasing** | insert an identity op | **guilty** |

### What settled it

**Inserting an operation that does nothing moved PPL by a factor of 2800:**

| Configuration | PPL |
|---|---:|
| P=2 (plain) | 520801 |
| P=2 + identity CPU op (`ms_marker_fn`) | **185.26** |
| P=2 + `ggml_cont` | 536 |

**If an operation that cannot change the result changes the result, then ggml's
graph buffer allocator (`ggml_gallocr`) is reusing a tensor that is still going
to be read.**

Using the same weight tensor in several `mul_mat_id` calls within one graph is
outside what llama.cpp expects, and fixing it needs changes in ggml core.
**That is beyond this project's scope, so it is disabled by default**
(experiment with `MOESTREAM_FORCE_PASSES`).

> Finding: `N4-expert-sweep.md`

---

## 10. Bugs found in the implementation (10)

Chasing Expert Sweep turned up 10 bugs on the main path. **All fixed.** That
debugging is what pushed main-path quality down to +0.11% PPL at the time.

| # | Bug | Symptom |
|---:|---|---|
| 1 | `finalize()` never called, slab uninitialised | output like "are are are…" |
| 2 | slot numbers colliding across layers (`slot % g_slots`) | wrong expert referenced |
| 3 | `release()` not called, refcount leak | eviction stopped |
| 4 | eval callback and graph op running simultaneously | duplicate I/O, 2.4x misses, 114 ms/token |
| 5 | ignoring `ggml_top_k`'s view stride | heap corruption |
| 6 | writing `sid=0` (the wrong expert) on slot exhaustion | PPL 4.97 → 2365 |
| 7 | no ceiling on `freq`, so S3-FIFO's second chance never expires | eviction stops → exhaustion |
| 8 | `n_passes` calling `finalize()` during graph construction | `GGML_ASSERT` (tensor not allocated) |
| 9 | `n_passes` missing the "slots ≥ n_expert → 1" branch | pointless splitting |
| 10 | slab sized by `ub × top_k` (worst case) instead of the union | 30% too much memory |

**Fix #10 (union estimate)** directly lowered the recommended configuration's
memory:

```
u    = n_expert × (1 − (1 − top_k/n_expert)^ub)     <- expected distinct experts
need = min(n_expert, u × 1.15 + top_k)              <- 15% safety margin
```

---

## 10.5 Model generality — confirmed across models

| Model | Size | Layers | Experts | top_k | Sharded | Leading dense |
|---|---:|---:|---:|---:|---|---|
| Ornith-1.0-35B | 16.87 GiB | 40 | 256 | 8 | — | — |
| Qwen3-Coder-Next | 36.54 GiB | 48 | 512 | 10 | — | — |
| Laguna-S-2.1 | 54.7 GiB | 48 | 256 | 10 | **3 shards** | **1 layer** |
| gpt-oss-120b | 58.46 GiB | 36 | 128 | 4 | **2 shards** | — |

**Running on four models without one line of architecture-specific branching.**
The only dependency is llama.cpp's tensor naming convention
(`blk.<il>.ffn_{gate,up,down}_exps.weight`); top_k, layer count, expert count,
sharding and leading dense layers all come from the GGUF automatically.

Each new model still broke an assumption, though. Two on Laguna:

1. **Reading from the wrong file in a sharded GGUF** — `llama_tensor_weight::idx`
   was ignored. Laguna's first shard is metadata only (zero tensors), so before
   the fix every expert was read from the wrong place. Now every expert's
   position is verified at startup and streaming is disabled if any is off.
2. **Statistics stopped on a model whose first layer is dense** — the boundary
   was detected by `il == 0`, and Laguna's layer 0 has no experts. Inference
   kept working; only the measurement died.

Both were the kind of defect where **inference keeps running**.

### Laguna-S-2.1 measured (54.7 GiB / 3 shards / 48 layers / 256 experts / top-10)

At `MOESTREAM_CACHE_FRAC=0.20` (51 slots, hit rate 66.47%):

| UBATCH | Memory | prefill | decode |
|---:|---:|---:|---:|
| 1024 | 16.60 GiB | 38.2 tok/s | 418 ms/tok |
| **8192** | **20.05 GiB** | **83.8 tok/s** | 440 ms/tok (2.27 tok/s) |

The automatic `[ub]` recommendation (8192, predicted 77.8 tok/s) matched the
measured 83.8 tok/s, **7% conservative**.

**But this model is too large for this machine.** The MRC says a 90% hit rate
needs 25.35 GiB resident, and GTT is 24 GiB. It settles at 66% and 2.27 tok/s
decode instead. That is beyond MoEStream's intended range (models of 15–35 GiB).

> Finding: `S8-model-generality.md`

---

## 10.6 `N_PARALLEL=1` was an unverified assumption (2026-08-06)

`.env` said for a long time that this should stay at 1 because the cache assumes
a single thread. **That was a design-time assumption that had never been
measured.**

Measured, it is both safe and faster:

| | 2 requests |
|---|---:|
| sequential (equivalent to N_PARALLEL=1) | 18.6 s |
| **concurrent (N_PARALLEL=2)** | **13.0 s (1.43x)** |

Zero slot exhaustions. The reason is simple: **llama.cpp's parallel slots batch
multiple sequences into one ubatch processed on the same thread**, so the cache
remains single-threaded. What is parallel is request admission, not cache
access.

It is faster because the per-token fixed costs (4.97 ms of CPU↔GPU
synchronization, weight reads) are amortised over two sequences. The same
amortisation as speculative decoding (§7.5), except **both tokens here are real,
so nothing is wasted**.

> Output can differ from the sequential case, but that is **unrelated to
> MoEStream**. The control (`MOESTREAM=0`) shows the same difference. Changing
> batch composition changes matrix-multiply tiling and accumulation order; that
> is stock llama.cpp behaviour.

**It does not help when running a single agent** (there are no concurrent
requests). It is only worth it if you genuinely run several. The default stays
at 1.

---

## 10.7 Code review (2026-08-06)

Static analysis with `-Wall -Wextra -Wshadow -Wconversion`, plus manual review
of resource handling and bounds checks. **Zero memory-safety warnings.** Three
substantive issues, all fixed.

| # | Issue | Impact | Fix |
|---|---|---|---|
| 1 | **staging buffers allocated unconditionally** | 41 MiB wasted on Ornith, **124 MiB** on Laguna | allocate lazily |
| 2 | `ExpertCache` / fd never released | only at process exit (harmless in practice) | release in `atexit` |
| 3 | unused parameters, `Seg` name collision | warnings only | fixed |

**The first is the largest.** Staging buffers for non-zero-copy environments
(discrete GPUs and the like) were allocated at `g_maxb × 3 × 32` always, but on
UMA zero-copy covers 100% of reads and **they are never used**. Changed to
allocate when actually needed.

Checked and found sound:

- every array index is guarded (`g_cache[il]`, `g_arena[bi]`, `g_mrc_last[il]`, …)
- the arena's degraded path (when the second buffer cannot be allocated) keeps array sizes consistent
- `finalize()` is re-entrant (early return on `g_cache.empty()`)
- there is always exactly one async prefetch thread; `pf_kick` always calls `pf_wait()` first
- layers L and L+1 always use different buffers (`il % 2`); prefetch of L+2 happens after FFN(L) completes

### Code size

| | Lines |
|---|---:|
| `src/llama-moestream.{h,cpp}` | 3227 |
| `src/expert_cache.{hpp,cpp}` | 425 |
| `src/apply.py` (the llama.cpp patch) | 298 |
| `src/entrypoint.sh` | 175 |
| `launcher.sh` (the interactive start-up) | 386 |
| `src/spec_probe.cpp` (asks llama.cpp what a model supports) | 30 |
| `research/tools/*.sh` | ~325 |
| **product total** | **4541** |
| spikes (verification, not shipped) | 6031 |
| documentation | 15583 |

> File layout was reorganised on 2026-08-06 and again on 2026-08-08.
> `expert_cache.{hpp,cpp}` was product code sitting in an early prototype's
> `nano/` directory and was moved into `src/`; what remained of `nano/` (the N1
> replay harness and its data) went to `research/spikes/n1_cache_replay/`. The
> reference clone in `vendor/` (233 MB) was deleted — the Dockerfile fetches
> llama.cpp at a pinned commit during the build.

**What is inserted into llama.cpp is 4 files, 5 blocks, about 95 lines.**
Documentation outnumbering code by 3.7x says something about the nature of this
project.

---

## 10.8 Fix #1 from §10.7 destroyed the output (2026-08-06, found and fixed the same day)

**The lazy allocation introduced in §10.7 as "reduce wasted memory" silently
swallowed every SSD read on the decode path.** It was caught and fixed the same
day, but it is **the most dangerous way this repository has broken**, so it gets
its own section.

### What happened

```cpp
const size_t cap = g_maxb ? g_stage.size() / g_maxb : 0;
for (; p < pend.size() && si + 3 <= cap; ++p)     // <- here
```

`cap` bounds the number of *staging buffer slots*. With lazy allocation, `g_stage`
stays empty on UMA, so `cap == 0`, so `si + 3 <= 0` is false from the start, so
**the inner loop never runs, `jobs` is empty, and it breaks out.**

The result: **not one byte of a missed expert is read from SSD, yet the slot is
recorded as holding it.** Uninitialised weights go into the computation.

The error was that `cap` had also become the bound on *how many reads may be
issued*, constraining even the case where zero-copy needs no staging at all.

### Why it was dangerous — **a broken implementation produces better numbers**

Skipping I/O entirely makes **decode faster**.

| | broken | correct | previously recorded |
|---|---:|---:|---:|
| Ornith (frac 0.25) | 46.9 ms/tok | **58.7 ms/tok** | 53.6 ms/tok (frac 0.38) |
| Qwen3-Coder-Next | 54.6 ms/tok | **101.7 ms/tok** | 106.4 ms/tok |
| Laguna-S-2.1 | 85.9 ms/tok | **319.2 ms/tok** | 291–310 ms/tok |

Laguna looked **3.7x faster**. It very nearly got written up as "union reads
(S12) and async prefetch (S14) paying off". The availability of a coherent
story is the worst part.

The prefill path (the arena) was **untouched**, because `load_layer_arena` takes
a different route with `j.direct = true` always. So the prefill numbers stayed
correct, which made it even harder to notice.

### How it was found

By **output**, not speed. On Laguna:

```
The capital of France is  ->  ' Paris. ernaernernREPLREPLR2555 }R2 5 }R2 '
```

Correct for exactly one token, then collapse. Ornith and Qwen returned
`The capital of France is Paris, a city renowned for...` — **plausible
sentences**. **Without Laguna, all three would have passed as "faster".**

### The fix

1. `cap` now bounds only **the number of staging buffers actually needed**
2. A watchdog prints `[BUG] discarded N pending reads. Output will be corrupt`
   to stderr if even one read is dropped (`research/tools/ms-bench.sh` always
   shows this count)

### The lesson

> **A measurement that looks only at speed gives its best numbers when the
> implementation is broken.**

After §9 (Expert Sweep, PPL 520801) and §13.3 (bug 2, non-determinism), this is
**the third time the same lesson has come up**. The previous two broke *visibly*
— slower, or obviously wrong. This one **got faster**. Be most suspicious when
speed moves in the favourable direction.

To prevent a recurrence, `research/tools/ms-bench.sh` is designed to always
report output correctness and the `[BUG]` line count **alongside** speed (§12.1).

---

## 10.9 Could decode use the arena approach too? — considered and rejected (2026-08-07)

Since the prefill staging arena worked, the obvious question was whether
**applying the same approach to decode would improve memory and speed further**.
It was measured and rejected.

### The motivation, and the trap in it

The arena approach means "hold no resident slots; read each expert as needed".
Applied to decode, the 3.62 GiB slab would disappear entirely, leaving only the
0.97 GiB arena.

**But the arena did not work on prefill *because it is an arena*.** Prefill
references nearly all 256 experts within one ubatch, which means **reading all of
them wastes nothing, and prefetching needs no prediction**. Those conditions
were what made it work.

Decode references 8 of 256. Those conditions do not hold. And further:

> **Which experts layer L+1 will call cannot be known, in principle, until layer
> L finishes computing.**
> So a decode arena cannot prefetch, and every layer becomes a serial
> read → wait → compute. On top of that, losing the cache multiplies I/O by the
> reciprocal of the miss rate.

### Measured — lower the cache ratio and extrapolate

Lowering `MOESTREAM_CACHE_FRAC` lowers the hit rate, approaching "read every
time" — the arena approach. Ornith-35B / ctx=8192 / UBATCH=1024:

| Slots | Hit rate | Miss rate | decode | Device memory |
|---:|---:|---:|---:|---:|
| 64 (default) | 81.87% | 18.13% | **55.5 ms** | 7.02 GiB |
| 38 | 69.14% | 30.86% | 61.6 ms | 5.54 GiB |
| 26 | 59.36% | 40.64% | 72.8 ms | 4.86 GiB |

Specifying `frac=0.05` **bottoms out at 26 slots**, identical to `frac=0.10`
(72.8 / 72.6 ms), because of the slab's lower bound
`union(g_max_ubatch=2) × 1.15 + top_k = 26`.

**The cost per point of miss rate accelerates:**

```
64->38 slots : misses +12.73 pt / decode + 6.1 ms  -> 0.479 ms/pt
38->26 slots : misses + 9.78 pt / decode +11.2 ms  -> 1.145 ms/pt   (2.4x worse)
```

Because rising misses also degrade page-cache locality. Extrapolating the final
slope to a 100% miss rate (no cache — the arena approach):

| | Device memory | decode |
|---|---:|---:|
| current (64 slots) | 7.02 GiB | 55.5 ms |
| decode arena (extrapolated) | **3.40 GiB** | **141 ms or worse** |

Memory breakdown: non-expert + KV 2.43 GiB + arena 0.97 GiB = 3.40 GiB.

### Verdict — the exchange rate is 6x worse

```
cutting via the frac knob :  8.0 ms / GiB
decode arena approach     : 46.6 ms / GiB
```

**For the same memory saved, the existing frac knob is 6x cheaper. Not worth
implementing.** And since the curve is accelerating, 141 ms is a lower bound;
reality would be worse.

### One region where it does open something

The slab has a **floor of 26 slots (4.86 GiB)** that the frac knob cannot go
below. The arena approach could reach 3.40 GiB. That only matters under a
constraint of "4.86 GiB is still too much" — and under that constraint, choosing
a smaller model is probably the better answer.

**The floor itself may be lowerable more cheaply (see the open items in §14).**

### A by-product — the best slot count moves with the workload

The same measurement produced an MRC recommendation of **102 slots (frac 0.40)**,
above the default 64.

| Slots | Hit rate | Marginal cost |
|---:|---:|---:|
| 102 (40%) | 90.77% | 3.14 pt/GiB |
| 64 (25%, default) | 81.87% | **7.00 pt/GiB** |
| 38 (15%) | 69.14% | 14.41 pt/GiB |

Long-form generation (two 700-token essays) spreads references over more
experts, which puts 64 slots below the knee. **The §12.5 measurement — mostly
short replies — was fine at 0.25.**

> **The best `MOESTREAM_CACHE_FRAC` moves with the workload, not just the
> model.** Which is exactly why the MRC is measured continuously and the
> recommendation comes from production logs. The default is a starting point,
> not an answer.

---

## 10.10 Three alternatives derived from §10.9 (2026-08-07)

Since the decode arena did not pay, three other ways of reaching **the same goal
(less memory)** were considered and all measured.

### Option 1: lower the slab's floor — **worth adopting**

The slab has a floor. `slab_slots()` computes the count as
`max(frac × n_expert, union(g_max_ubatch) × 1.15 + top_k)`, and on Ornith
`g_max_ubatch = 2` means it **cannot go below 26 slots (4.86 GiB)**.

But with `N_PARALLEL=1`, decode's `n_tokens` is always 1; anything larger is
routed to the prefill arena automatically. Lowering the floor to one token's
worth gives `union(1) × 1.15 + top_k = 17 slots`.

Measured with `MOESTREAM_MAX_UBATCH=1` (Ornith / ctx=8192 / UBATCH=1024):

| Configuration | Slots | Device memory | decode |
|---|---:|---:|---:|
| `frac=0.05` default | 26 (floor) | 4.86 GiB | 76.3 ms |
| `frac=0.05` + `MAX_UBATCH=1` | **17** | **4.35 GiB** | 80.9 ms |

**0.51 GiB saved for +4.6 ms (+6%) of decode: 9.0 ms/GiB** — about the same as
the frac knob (8.0 ms/GiB) and **a fifth** of the decode arena (46.6 ms/GiB).

> **Part of the arena's goal is reachable through existing machinery, with no
> arena.** But `g_max_ubatch` *is* the guarantee that a micro-batch fits in the
> slab; getting its relationship with `N_PARALLEL` wrong leads directly to slot
> exhaustion and corrupt output. The minimal safe automation would be to drop it
> to 1 only when `N_PARALLEL == 1`. **Measured only; not implemented.**

### Option 2: use one prefill arena instead of two — **costs 10% of prefill**

The arena defaults to two buffers (498 MiB × 2). The second exists for async
prefetch and **neither is used during decode**. Of 7.02 GiB, 0.97 GiB sits idle
while decoding.

Measured with `MOESTREAM_PREFILL_BUFS=1` (ctx=32768):

| Configuration | Device memory | prefill (13877 tokens, twice) | decode |
|---|---:|---:|---:|
| 2 buffers (default) | 7.36 GiB | 243.4 / 248.9 tok/s | 57.6 ms |
| **1 buffer** | **6.87 GiB** | **220.4 / 224.8 tok/s** | 58.1 ms |

**0.49 GiB saved, prefill −9.7%, decode unchanged.** The loss is exactly the
async prefetch no longer working.

For decode-only use (mostly turn 2 onward, where the prompt cache hits) that is
nearly free memory. **Not enough reason to change the default, but a reasonable
memory-first profile.** Measured only.

### Option 3: learn the recommended slot count from the workload — **implemented**

§10.9 established that the best slot count moves with the workload (the same
Ornith wants 0.25 for short replies and 0.40 for long-form). The MRC already
emits a recommendation while running, but **a human had to read the log, edit
`.env`, and restart**.

`MOESTREAM_CACHE_FRAC=learn` was added: the recommendation is recorded in
`./state/tuning.tsv` and applied automatically on the next start.

```
1st start:   [learn] no tuning recorded yet for this model; starting conservatively at frac=0.15 and measuring
             [mrc] recommended 102 slots (0.40)
             [learn] recorded frac=0.40 (102 slots) for this model; the next start will use it
2nd start:   [learn] using frac=0.40 learned from previous runs (/state/tuning.tsv)
             40 layers x 256 experts -> 102 slots/layer (40%)
other model: [learn] no tuning recorded yet for this model; starting conservatively at frac=0.15
             48 layers x 512 experts -> 128 slots/layer (25%)
```

Safety properties of the design:

- **Records are keyed by model identity** (file name + expert count + bytes per
  slot), so swapping models never applies the previous model's value. The same
  reasoning as never writing derivable values into `.env`.
- **Written via a temporary file and `rename`**, so a crash never leaves a
  corrupt file.
- **A number specified directly wins over the learned value.** `learn` is an
  explicit opt-in.
- Nothing is written if the recommendation moves by less than 0.02 (no write per
  report).
- **With no record, the first run starts at 0.15.** The memory cap cannot be
  computed until weights and KV are loaded, so an initial value cannot be
  guaranteed to fit. Starting high and failing to start defeats the entire point
  of an unattended feature. Erring low is not a startup risk, so it errs low. In
  fact the production machine (Laguna / ctx=100000) had only 0.58 GiB free, and
  starting at 0.25 would have failed.

### Reinforcing option 3: cap the recommendation by memory

Originally the recommendation came from **pt/GiB (marginal hit rate) alone**,
which is dangerous: **recommending a slot count that does not fit makes the next
start fail** — the worst possible outcome for a feature whose purpose is to work
unattended.

So device memory limits and current usage are read from sysfs
(`mem_info_gtt_total` / `mem_info_vram_total` / `*_used`) and the recommendation
is capped at what fits.

```
slots that fit = (limit - reserve - non-slab usage) / bytes per slot
```

"Non-slab" includes weights, KV and the arena, plus **whatever other processes
are using on the GPU**. Treating that as fixed is the conservative choice, and
it matches this project's stated value of coexisting with other things. The
reserve defaults to 1.0 GiB (`MOESTREAM_MEM_RESERVE_GIB`).

Three measured cases:

| Condition | Ideal recommendation | Slots that fit | Actual recommendation |
|---|---:|---:|---:|
| Ornith (room to spare, 7.0/24.0 GiB) | 102 | 346 | **102** (passes through) |
| Laguna (tight, 20.3/24.0 GiB) | 51 | 77 | **51** (passes through) |
| Ornith + 15 GiB reserve (artificially tight) | 102 | 98 | **98** (capped) |

```
[mrc] recommended 98 slots (0.38) -- below this, hit-rate loss exceeds 2.5 pt/GiB
[mrc]   device memory 7.0 / 24.0 GiB used; at most 98 slots fit
[mrc]   ** capped by memory: 102 slots would be better for hit rate,
[mrc]      but would not fit. Recommending 98. **
```

**What gets recorded is the capped value** (0.38). Recording the ideal would
make the next start fail, so this has to be a value that actually works.

As a side check, **Laguna's recommendation of 0.20 matched the optimum measured
independently in §12.5.** The MRC arrived at the same answer on its own.

### Where the three stand

| Option | Memory saved | Cost | Exchange rate | Verdict |
|---|---:|---|---:|---|
| decode arena (§10.9) | −3.62 GiB | decode +85 ms | 46.6 ms/GiB | **rejected** |
| 1: lower the slab floor | −0.51 GiB | decode +4.6 ms | 9.0 ms/GiB | promising (not implemented) |
| 2: one arena buffer | −0.49 GiB | prefill −9.7% | — | conditional (not implemented) |
| 3: learn it | 0 | none | — | **implemented** |

---

## 10.11 Re-sweeping UBATCH — the predictor was recommending the wrong value (2026-08-07)

The conclusion "UBATCH=1024 is the knee" was based on measurements taken
**before** union reads (S12) and async prefetch (S14). Both cut the I/O side
substantially, so the constraint — and therefore the optimum — may have moved.
It was the one item whose premise had changed without being re-measured, so it
was swept again on current code.

Conditions: Ornith-35B / frac=0.25 / ctx=32768 / 13877-token real document /
best of 3 runs each.

| UBATCH | Device memory | prefill |
|---:|---:|---:|
| 512 | 7.22 GiB | 212.1 tok/s |
| **1024** | **7.36 GiB** | **252.7 tok/s** |
| 2048 | 7.64 GiB | 226.8 tok/s |
| 4096 | 8.20 GiB | 211.7 tok/s |

**1024 is not a knee — it is a peak.** The old measurement had 2048 at +5%; on
current code 2048 is **10% slower** than 1024. The trend inverted.

### The built-in measurement showed why

`[ub]` separates a pass into I (I/O) and C (compute):

| UBATCH | I/O per pass | compute per pass | Factor vs previous | compute/token |
|---:|---:|---:|---:|---:|
| 512 | 1.53 s | 0.82 s | — | 1.60 ms |
| 1024 | 1.61 s | 2.40 s | **2.93x** | 2.34 ms |
| 2048 | 1.59 s | 7.32 s | **3.05x** | 3.57 ms |
| 4096 | 1.54 s | 16.68 s | **2.28x** | 4.07 ms |

**I/O is constant at about 1.55 s regardless of UBATCH** (as designed). But
**compute rises 2.3–3.0x when ubatch doubles. That is not linear** — attention
is quadratic in the number of tokens within a micro-batch.

Composing total time from I and C reproduces the measurements well:

```
ub= 512 : 28 passes x (1.53+ 0.82) = 65.8 s -> 210.9 tok/s  (measured 212.1)
ub=1024 : 14 passes x (1.61+ 2.40) = 56.1 s -> 247.2 tok/s  (measured 252.7)
ub=2048 :  7 passes x (1.59+ 7.32) = 62.4 s -> 222.5 tok/s  (measured 226.8)
ub=4096 :  4 passes x (1.54+16.68) = 72.9 s -> 190.4 tok/s  (measured 211.7)
```

### The predictor's bug — a recommendation guaranteed to make things slower

`ub_report()` extrapolated assuming **`C ∝ ubatch` (linear)**. Under that
assumption prefill speed approaches `ub0/C` monotonically, so the recommendation
is **always "go higher"**. What it actually emitted:

| UBATCH at run time | Predictor's recommendation | Correct answer, measured |
|---:|---:|---:|
| 512 | 8192 | 1024 |
| 1024 | 4096 | 1024 |
| 2048 | 4096 | 1024 |

**Following the recommendation to 4096 is 16% slower than 1024.** A feature
meant to improve performance was reliably instructing users to degrade it.

### The fix — stop extrapolating

A single run **observes C at exactly one point**, so the growth exponent cannot
be fitted. Guessing the exponent and extrapolating anyway would be building yet
another plausible-looking instrument, which had already failed several times that
day. **The extrapolation was removed and only measured facts are reported.**

```
[ub] prefill cost split (measured over 25 passes, UBATCH=1024)
[ub]   I/O per pass         1.52 s  <- fixed cost, independent of UBATCH
[ub]   compute per pass     2.53 s  (2.47 ms per token)
[ub]   raising UBATCH amortizes the 1.52 s I/O over fewer passes,
[ub]   but per-token compute grows with UBATCH (attention is
[ub]   quadratic within a micro-batch), so higher is not always
[ub]   better. There is an optimum and it cannot be extrapolated
[ub]   from a single UBATCH -- measure it
```

> **Lesson: extrapolation is only permissible within the range where the model
> has been confirmed.** This predictor measured I and C correctly. The only
> thing wrong was the **unverified assumption** that C is proportional to
> ubatch. The measurement was right; the model was wrong. The eighth entry for
> the list in §13.3.

### Conclusion

**UBATCH=1024 is correct as a default**, but not for the reason originally
given. It is not "the point where I/O-bound becomes compute-bound and it
plateaus" — it is **the point where amortising I/O balances against
super-linearly growing compute**. Which is why going past 1024 does not flatten
out; it **gets worse**.

---

## 10.12 Re-measuring the decode breakdown — the headline claim was stale (2026-08-07)

To decide whether there was anything left on the engine side, decode overhead
was decomposed on current code. **The result: this project's headline claim —
"the bottleneck is CPU↔GPU synchronization, not I/O" — no longer holds for the
current code.**

### Method

Four stages, separated by diagnostic switches. Ornith-35B / ctx=8192 /
UBATCH=1024 / median of 5–7 runs each.

| Configuration | Includes | Meaning of the difference |
|---|---|---|
| A `MOESTREAM=0` | compute only | baseline |
| B `NOOP=1` (remap passed through) | + CPU↔GPU sync | B−A = sync |
| C `NOOP=3` (cache lookup only) | + cache management | C−B = management |
| D normal | + SSD reads | D−C = I/O |

`NOOP=3` still calls `mark_resident()`, so **cache behaviour is identical to
normal** and only the reads disappear. A valid control.

### Results — reproduced at two fracs

| | frac=0.38 | frac=0.25 |
|---|---:|---:|
| A plain llama.cpp | 41.88 ms | 41.96 ms |
| B sync only | 44.12 | — |
| C cache lookup only | 46.38 | 46.30 / 46.46 |
| D normal | 54.20 | 58.75 / 60.76 |
| **sync (B−A)** | **2.24 ms (4.1%)** | ~2.2 ms |
| **cache management (C−B)** | **2.26 ms (4.2%)** | ~2.3 ms |
| **I/O (D−C)** | **7.82 ms (14.4%)** | **12.45 ms (21.2%)** |

**Sync is a steady ~2.2 ms. I/O is 3.5–5.6x that.**

### Contradiction with what was written

`README.md` and §13.1 said:

> Per decoded token, I/O is **0.5 ms** and synchronization is **11.4 ms** — a
> **23x** difference.

**The current code is the other way round.** The old measurement predates union
reads (S12), async prefetch (S14) and zero-copy, when the same `NOOP=3`
difference was 11.4 ms. It is now 4.5 ms (sync 2.2 + management 2.3). Our own
improvements moved the bottleneck.

> **The headline number was not updated when the implementation changed.**
> While insisting in §12 that measurement conditions must be stated, the most
> frequently quoted figure was itself from an old measurement.

### Effect on the conclusions — the rejections stand

**The argument** "synchronization is the bottleneck, therefore cleverness about
I/O is wasted" **collapses**. But predictive prefetch, fadvise and io_uring were
rejected **on direct measurement, not on that argument** (net-negative, −19%,
+1.5% respectively). So **the conclusions stand and only the rationale is
corrected.**

Note also that S11's fadvise was measured on Laguna, where I/O is 75% of decode,
and still lost by 19%. A high I/O share does not imply prefetching wins.

### Derived options measured — all came up empty

With I/O established as the main cost, options for **reducing it without adding
memory** were measured.

**Option A: change the decode I/O thread count.** The built-in measurement (time
blocked on reads) and the difference method (effective cost) disagreed by about
1.6x, suggesting reader threads were competing with compute for memory
bandwidth. If true, fewer threads should be faster.

| Threads | decode |
|---:|---:|
| 1 | 58.92 ms |
| 2 | 61.72 ms |
| 4 | 58.85 ms |
| 8 | 60.56 ms |
| auto | 60.15 ms |

**A 5% spread, comparable to the ~3% run-to-run variation of a fixed
configuration. No significant difference.** The contention hypothesis was not
supported. **Rejected.**

> An earlier record claimed "1 thread is fastest for Ornith decode (56.29 vs
> 56.96 at 4)" and dropped it from the candidates over a 1.2% difference. Seven
> runs here confirm there is no significant difference. **That call was right by
> accident.**

### Fixing the instrument

The 1.6x gap between the built-in measurement (`[prefetch]`) and the difference
method means **neither is wrong**:

- built-in = **time spent inside** `run_reads_parallel` (blocked on reads)
- difference method = **the effective cost including what reads cause**
  (each miss writes ~1.4 MiB into GPU-visible memory, and the slowdown of
  subsequent compute from memory and cache pressure appears after the timer
  stops)

**Both are correct about what they measure. The label was wrong.**
`measured I/O share` reads as "total I/O cost" when it is really blocking time.

```
old: [prefetch] measured I/O share 11.7% (2.9 s of decode, of which 0.3 s waiting on reads)
new: [prefetch] decode spends 11.7% blocked on reads (0.3 s of 2.9 s)
     [prefetch]   end-to-end I/O cost is roughly 1.6x that;
                  measure it with MOESTREAM_NOOP=3 as the control
```

> Lesson: **when two instruments disagree, it does not follow that one is
> broken.** They may be measuring different things under the same name.

---

## 10.13 Batching syscalls (preadv / io_uring) — a 0.6% ceiling, rejected (2026-08-07)

With I/O established as the bulk of decode overhead (§10.12), one last option
for **cutting I/O cost without adding memory** was examined.

Decode issues about 6 `pread` calls per layer (1–2 missed experts ×
gate/up/down), so about **240 per token** across 40 layers. Batching them should
reduce syscall overhead.

### First: `preadv` does not apply

`preadv(fd, iov, iovcnt, offset)` scatters **from one file offset into several
buffers**; it **cannot gather from different offsets**. An expert's gate/up/down
live in separate tensors at distant offsets, so it does not apply.

`io_uring` can batch different offsets, but **finding S2 already measured it at
+1.5% over parallel pthread pread**. That was on prefill-sized reads, though, and
decode's smaller reads have a different fixed-cost ratio — so it was worth
checking.

### Measurement (`research/spikes/s15_syscall_overhead`)

With everything in page cache, performing the same count and size of reads as
decode:

| | Time | Per call |
|---|---:|---:|
| (a) 240 preads (0.43 MiB each) | 10.388 ms | 43.28 µs |
| (b) the same bytes in one call | 12.406 ms | 51.69 µs equivalent |
| **(c) 240 preads (4 KiB each)** | **0.348 ms** | **1.45 µs** ← the syscall floor |

**Comparing (a) with (b) failed.** (a) reuses a 3.4 MiB buffer that stays in
L2/L3; (b) fills a cold 103 MiB buffer. That measures **destination locality,
not syscall cost** — which is why (b) came out slower, an impossible result
otherwise.

The correct approach is (c): **keep the call count and minimise the data**, so
only the fixed cost remains.

### Conclusion

```
fixed cost per call 1.45 µs x 240 calls = 0.348 ms/token
                                        = 0.60% of a 58 ms decode
```

**Batching can win at most 0.6%**, and io_uring does not eliminate syscalls
entirely, so the real share is smaller. **Rejected.**

This also confirms S2's "+1.5%" holds for decode. **Read time is almost entirely
data movement, not call count.**

---

## 10.14 Re-evaluating cross-layer prefetch — rejected on economics (2026-08-07)

Learning that Pulsar (NVIDIA/CUDA, 295B–1T) **does use cross-layer prediction**
prompted a re-evaluation of the P2 predictor that MoEStream rejected in findings
N2/S11 — because **the basis for that rejection had moved by 5x in §10.12**.

| | At rejection (2026-08-04) | Now (2026-08-07) |
|---|---:|---:|
| I/O available to hide | 0.5 ms | **7.8–12.5 ms** |
| remap synchronization | 11.4 ms | **2.2 ms** |

### Measurement 1 — the predictor's synchronization cost

Without implementing a predictor, an op that merely **carries the hidden state
(the router's input) to the CPU each layer** was inserted and the decode
increment measured (`MOESTREAM_PROBE_HIDDEN=1`). That is the synchronization P2
would have to pay.

| Configuration | decode |
|---|---:|
| baseline | 60.23 ms |
| probe enabled | **64.06 ms** |
| baseline (re-measured) | 59.93 ms |

**Synchronization cost: 3.98 ms.** The two baselines differ by 0.30 ms, so this
is 13x the variation — a real difference. That is far below the 14–22 ms at the
time of rejection, and **on the balance sheet alone the sign had flipped.**

### Measurement 2 — room to overlap (this is what settled it)

For prefetch to work there must be **compute running behind the reads being
hidden**. The window is the time from one layer's remap finishing to the next
layer's remap starting. It was instrumented as `[overlap]`.

| | Ornith (frac 0.25) | Laguna (frac 0.15) |
|---|---:|---:|
| compute per layer (the window) | 1.282 ms | 2.897 ms |
| reads per layer | 0.187 ms | **5.032 ms** |
| **ratio** | **6.8x** | **0.6x** |
| layers where reads outran the window | 29 / 23,959 (0.1%) | **9,611 / 11,703 (82.1%)** |
| I/O share of decode | 11.9% | 63.7% |

**There is no room for prefetch at either end, for opposite reasons.**

```
small I/O (Ornith) : the window is 6.8x, but there is only 0.187 ms to hide
                     -> ceiling 3%, three risks. Not worth it
large I/O (Laguna) : plenty to hide, but the window is 0.6x and compute
                     finishes first in 82% of layers
                     -> even with perfect prefetch, at most 58% can be hidden
```

### Correcting an earlier claim — S9's conclusion was over-generalised

`README.md` and §7 said:

> Where I/O accounts for 75%, there is **about 2x of headroom** (S9)

**That is wrong.** S9 measured whether continuous queueing raises SSD bandwidth.
It **did not measure room to overlap with compute**. Measuring that directly
shows that where I/O dominates, **compute finishes first and there is nowhere to
hide**.

> **"A high I/O share means prefetching helps" does not hold.**
> What is required is not an I/O share but a **window-to-read ratio comfortably
> above 1**, and the two are not the same. As the I/O share rises, the window
> ratio falls.

### Even the safe design does not work out

S14 (async arena prefetch) avoided the risks by **never touching ExpertCache**.
The same shape would eliminate S11's three risks (refcount leaks, slot
reallocation races, partial writes): the prefetch thread reads into a dedicated
buffer and the main thread does the slab update.

**But on UMA that does not work.**

```
memcpy effective bandwidth   18.0 GB/s (measured)
misses per token             82 MiB
added cost of the relay copy 4.77 ms/token
```

Reads currently go **straight from SSD into GPU-visible memory** via zero-copy,
so adding a relay adds 4.77 ms without removing a single byte.

| Design | Balance |
|---|---|
| safe (S14 style) | gain 5.8 − sync 3.98 − relay 4.77 = **negative** |
| risky (S11 style) | gain 5.8 − sync 3.98 = **under +1.8 ms (3%)** + three risks |

**Removing the risk removes the gain.** On a discrete GPU the PCIe transfer is a
separate place to hide things, which lowers the relative cost of the relay copy —
which is where Pulsar's version works.

### Conclusion

**Not implemented.** There was no need to go as far as quantifying the risk
(simulating collision counts); **the economics failed before that.**

> Lesson: **"the premise changed, so re-evaluate" is right, but a re-evaluation
> is for re-measuring the facts, not for overturning the earlier conclusion.**
> Here the balance sheet had genuinely flipped (sync 14–22 → 3.98 ms). But the
> balance sheet is only part of the condition, and the deciding factor was
> **room to overlap** — a quantity that had never been measured at all.

---

## 10.15 S3-FIFO vs LFU — comparing against Pulsar's admission policy (2026-08-07)

Pulsar uses **LFU (touch counts)** for admission into its VRAM hot set.
MoEStream's S3-FIFO **had been compared against LRU** (findings N1/M0-2) but
**never against LFU**. A policy change costs no memory at all, so if LFU won it
would be a free improvement.

### Method (`research/spikes/s16_policy_compare`)

Replay the real traces collected in M0-2 (English / code / Japanese, 20,000
tokens each, 40 layers × top-8 × 256 experts) through three policies.

**S3-FIFO is linked directly from the product implementation
(`src/expert_cache.cpp`)** — reimplementing it would let implementation
differences distort the comparison. Only LFU and LRU are written in the spike.

### Result — S3-FIFO wins all 15 points

| Trace | Slots | S3-FIFO | LFU | LRU | LFU − S3 |
|---|---:|---:|---:|---:|---:|
| English | 26 | **62.24%** | 45.86% | 60.32% | **−16.38 pt** |
| English | 38 | **71.11%** | 55.43% | 68.98% | −15.68 pt |
| English | 64 | **82.93%** | 69.96% | 81.49% | −12.97 pt |
| English | 102 | **91.64%** | 83.26% | 91.03% | −8.38 pt |
| English | 128 | **94.78%** | 89.15% | 94.42% | −5.64 pt |
| code | 64 | **82.13%** | 79.30% | 80.41% | −2.83 pt |
| Japanese | 64 | **85.22%** | 76.89% | 83.86% | −8.34 pt |

**LFU loses even to LRU** (except on the code trace).

### Why it loses

LFU decides what to keep by cumulative reference count, so **experts that were
called heavily early on stay long after the context has moved**. At the same
time, newly needed experts have low counts and are **not admitted** — a side
effect of touch-count admission.

**The gap widens as slots shrink** (−16.38 pt at 26 slots, −5.64 pt at 128).
MoEStream operates around 25%, which is where the gap is largest.

The domain differences are consistent too. **The small gap on code (−2.83 pt)**
is presumably because the set of experts in use is stable there. In natural
language, where the topic moves, LFU is dragged by the past.

### Why LFU presumably works for Pulsar

Pulsar runs an 8 GB VRAM hot set against a 211 GB model — a cache ratio of about
4%. Where the working set vastly exceeds the cache, LFU's "keep only the truly
frequent" property plausibly helps. That is a different regime from MoEStream's
25%.

> **The same shape as the P2 predictor in §10.14.** Change the scale and the
> best method inverts. Both implementations are right in their own regime.

### What was adopted from Pulsar — nothing

| Mechanism | Verdict | Reason |
|---|---|---|
| cross-layer prediction | rejected | window/read ratio fails at both ends (§10.14) |
| LFU admission | rejected | loses to S3-FIFO on all 15 points (this section) |
| CPU lane | not applicable | UMA has no PCIe; there is no transfer to avoid |
| io_uring QD32 + staged upload | not applicable | zero-copy means there is no upload stage |
| residency on a spare GPU | not applicable | there is one GPU |
| `.warm` temperature persistence | already equivalent | `MOESTREAM_CACHE_FRAC=learn` (§10.10) |

**Nothing was adoptable, which is not the same as wasted.**
The choice of S3-FIFO is now supported against LFU as well, **closing a gap
where it had only ever been compared with LRU.**

---


## 10.16 frac and UBATCH are not independent — two-stage learning (2026-08-08)

### How it started

`MOESTREAM_CACHE_FRAC` and `UBATCH` had been treated as separate settings, each
swept for an optimum and each automated with `learn`. But **the sweeps had been
done at different fracs**, and re-measuring under matched conditions reversed the
ranking.

Ornith-35B (256E, ctx=32768), prefill tok/s:

| frac | UB=1024 | UB=2048 | Winner |
|---|---|---|---|
| 0.25 | **249.1** | 226.8 | 1024 (+9.8%) |
| 0.15 | 220.0 | **224.3** | 2048 (+2.0%) |

**The best UBATCH inverts with frac.**

### Why it inverts

Raising UBATCH re-reads each expert fewer times, but attention within a
micro-batch is quadratic so per-token compute rises. The optimum is where those
balance (§10.11).

Raising frac enlarges the slab, so **fewer misses, which weakens the "fewer
re-reads" benefit**. The balance point then moves toward the compute side, and a
smaller UBATCH wins. Lower the frac and misses dominate again, favouring a
larger UBATCH.

In other words **the two settings sit on opposite sides of the same balance.**
Learning one without fixing the other collects measurements from different
conditions and **converges on a combination that does not exist.**

### The fix — stamp each ubatch measurement with its frac

Every measurement records **the frac it was taken at**.

```
tuning.tsv   <model file>  <frac>   <n_expert:bytes_per_slot>
ubatch.tsv   <model file>  <frac>   <ubatch>  <prefill tok/s>
```

At startup, only rows matching the frac this run will use are considered.
**When frac moves, old rows simply stop matching and are measured again.** No
version numbers, no generation tracking. Rows for 0.15 are not deleted, so
moving back reuses them.

### The two stages

| Stage | Decides | Basis | Recorded in |
|---|---|---|---|
| 1 | frac | reuse distance (MRC) against a pt/GiB threshold | `tuning.tsv` |
| 2 | UBATCH | measured prefill tok/s, compared | `ubatch.tsv` |

Stage 1 needs no restarts because **one run yields the hit rate at every slot
count** (reuse distance does not depend on cache size). Stage 2 cannot be
predicted — per-token compute is super-linear, so extrapolation is wrong
(§10.11) — so each candidate is measured once and compared.
**Predict what can be predicted; measure only what cannot.** The order follows
from the dependency: frac first, UBATCH second.

### Verified on the machine (2026-08-08)

With frac fixed at 0.15, one candidate per start, converging on the fifth:

```
start 1  measuring 1024 at frac=0.15   -> 218.4 tok/s
start 2  measuring 2048 at frac=0.15   -> 237.1
start 3  measuring 4096 at frac=0.15   -> 227.4
start 4  measuring 8192 at frac=0.15   -> 228.2
start 5  using 2048 at frac=0.15 (237.1 tok/s)  -> re-measured 238.6
```

Re-measurement differs by 0.6% (237.1 → 238.6), which cannot change the ranking.

Changing frac to 0.25 matched none of the four rows for 0.15 and measured 1024
again (245.7 tok/s) — 1.4% from the earlier 249.1, reproducing the direction of
the inversion.

The two stages chaining:

```
[mrc] recommended 128 slots (0.50)
[learn] frac=0.50 recorded; the next start uses it
  |  next start
UBATCH=learn: measuring 1024 at frac=0.50
40 layers x 256 experts -> 128 slots/layer (50%)
```

### Three defects found while implementing it

All of the "looks like it is working" variety, found only by looking at values.

**1. The device memory reading at exit was meaningless.**
The frac recommendation is capped by memory, because pt/GiB alone will happily
recommend a value that cannot start. But the recording happened at exit, by
which point the buffers were already released and it read
`device memory 1.1 / 24.0 GiB used`. **The cap was effectively passing
everything.** Changed to keep the maximum of readings taken while running (first
at 32 tokens, then every 2000), which reads 5.9 GiB correctly under the same
conditions.

**2. Recording of the ubatch result was tied to frac being learned.**
With `MOESTREAM_CACHE_FRAC` pinned to a number and `UBATCH=learn`, the recording
branch was gated on `g_frac_learn` and never fired, so **the same candidate was
measured forever.** The condition is now "did the entrypoint choose the ubatch".

**3. On a first start with no state file, awk exited with status 2.**
"Cannot open the file" was indistinguishable from "already measured", so the
path that picks an unmeasured candidate was skipped and it fell through to the
default. The resulting value happened to be the same, for a different reason.
Now a missing file is read as `/dev/null` (the state directory is not necessarily
writable, so it is not created).

The shell side is covered by 11 unit tests requiring no container — invalidation
on frac change, not retrying rate-0 candidates, ignoring other models' rows,
surviving corrupt rows, and an unreadable state directory.

### A fourth, found by testing on other models — large UBATCH was always discarded (2026-08-08)

A defect that never surfaced on Ornith appeared the moment conditions changed on
Qwen3-Coder:

```
[learn] UBATCH=4096 at frac=0.20 -> 0.0 tok/s prefill
[learn] UBATCH=8192 at frac=0.20 -> 0.0 tok/s prefill
```

Passes are accounted **when the next one starts** — the interval is defined from
one pass beginning to the next, which is what excludes partial passes and idle
gaps. But **the last pass of a run is never closed.**

A run with only one prefill pass therefore has nothing to account. And one pass
is exactly what happens when **the UBATCH can swallow the whole prompt** — which
**larger candidates are more likely to do.**

The code at the time recorded a rate of 0 as "tried, nothing measurable" and
never retried that candidate. Combined:

> **On a short-prompt workload, large UBATCH values are all permanently retired
> and the learner necessarily picks a small one.**

Nothing in the log looks wrong. It is invisible until you see the 0.0. It never
surfaced on Ornith because a long prompt divided by 1024 gives several passes,
so losing the last one still left others.

Two fixes were applied at the time:

1. **Close the still-open pass at exit** (`pf_close_pass`), using the last arena
   activity rather than `now()` — `now()` would include the idle gap between the
   request finishing and the process stopping.
2. **Do not record a rate of 0.** The candidate stays unmeasured and is tried
   again.

**Fix 1 turned out to be wrong. See below.**

**The lesson is also about the measurement procedure.** This defect was hit
because the request count and generation length were not kept the same when the
model changed. Changing the conditions is arguably what exposed it, but
**never compare numbers until they have been re-measured under matched
conditions.** In fact, on Laguna, raising frac from 0.15 to 0.20 made prefill
*look* like it fell from 77.4 to 68.8 tok/s — but the first was three requests
(warm-up excluded) and the second was one (no exclusion possible). That is a
difference of conditions, not an effect of frac.

### The fifth — the "close the last pass" fix was itself wrong (2026-08-08)

The fix applied for the fourth problem — `pf_close_pass`, which closes a pass
left open at exit and uses the time of the last arena activity as its end — was
**itself wrong.** Sweeping the four candidates under identical conditions, the
values stopped being monotonic and fell apart:

| UBATCH | Recorded | Full intervals | What was actually happening |
|---|---:|---:|---|
| 1024 | 108.8 | 2 | correct (warm-up excluded) |
| 2048 | 47.6 | 1 | could not exclude, so cold time is included (the true value is ~110) |
| 4096 | 163.5 | 0 | `pf_close_pass`'s value — **about 50% too high** |
| 8192 | 164.9 | 0 | likewise |

163 tok/s is not physically possible against a true 110. The cause was the end
timestamp: **the asynchronous arena prefetch runs ahead of compute**, so the last
read completes well before the pass does. "Last arena activity" is not usable as
an approximation of a pass boundary.

### Independent corroboration — cross-checking against llama.cpp's own prompt eval

Which figure is right cannot be settled by internal reasoning; it needs **a
second measurement**. llama.cpp's `prompt eval` is "whole prompt divided by
prefill time", a definition that does not move with UBATCH. Both were captured
in the same run:

| Condition | llama.cpp | MoEStream | Gap |
|---|---:|---:|---:|
| REQS=1, ub=1024, 2466 tok | 97.09 (one cold pass only) | 109.0 | explained by definition |
| REQS=8, ub=1024, 2466 tok | 97.94 → 109.2–111.4 | 113.5 | +3% vs the warm mean |
| REQS=2, ub=1024, 22828 tok | 114.19 / 116.09 | 118.0 | +2% |

**Where two or more full passes exist, MoEStream's measurement is correct.**
MoEStream reads slightly high because it excludes the cold pass and the partial
tail, which is a difference of definition.

### The rule that was settled on

> **Record only when at least two full passes were seen and the cold first one
> could be excluded.**

The only case that fails is "the prompt is shorter than twice the UBATCH". And
in that case llama.cpp clamps the ubatch to the prompt length anyway, so
**4096 and 8192 behave identically and there is nothing to distinguish**. That
is not a measurement failure — it is the answer that this UBATCH is unusable for
this workload — so a 0 is recorded and the candidate retired. Deleting
`state/ubatch.tsv` restarts the search.

Two situations also have to be distinguished, or a transient accident gets
recorded as a permanent property:

| Observation | Meaning | Behaviour |
|---|---|---|
| zero full passes (no prefill ran at all) | rejected request, health check only, early stop | nothing recorded, tried again |
| exactly one full pass | prompts are shorter than twice the UBATCH | record 0, retire the candidate |

### The sixth — per-pass measurement cannot rank UBATCH at all (2026-08-08)

Sweeping all four candidates under the rule above, with a 22828-token prompt
long enough for every candidate to qualify, and comparing against llama.cpp:

| UBATCH | llama.cpp | per-pass MoEStream | Gap |
|---|---:|---:|---:|
| 1024 | 115.0 | 117.9 | +2.5% |
| 2048 | 128.4 | 132.0 | +2.8% |
| 4096 | **135.1** ← fastest | 142.4 | +5.4% |
| 8192 | 133.5 | **152.7** ← fastest | **+14.4%** |

**The gap grows monotonically and the winner disagrees.** llama.cpp makes 4096
fastest with 8192 slightly behind; the per-pass measurement picks 8192.

Even with the two-full-passes rule, the four values did not agree with
llama.cpp, and the gap grew with UBATCH — because the exclusions themselves are
UBATCH-dependent. At ub=8192, 6444 of 22828 tokens (28%) fall into the partial
tail and are discarded, leaving one pass to speak for the whole request.

> **An exclusion added to remove bias was producing a new bias.**

### The fix — make the request the unit

A request's definition does not move with UBATCH, and a request is what the user
waits for. It is also exactly what llama.cpp reports as `prompt eval`.

The start time is taken at the first prefill pass and the interval is closed
**when decode begins** — detected through an existing hook (`nt == 1`). Nothing
inside is discarded: full passes, the partial tail, and the gaps between them all
count. Warm-up exclusion applies to the first *request*, not the first pass.

Re-measured under the same conditions:

| UBATCH | llama.cpp (warm mean) | MoEStream | Gap |
|---|---:|---:|---:|
| 1024 | 114.38 | 114.4 | +0.02% |
| 2048 | 127.96 | 128.1 | +0.11% |
| 4096 | **135.97** ← fastest | **136.3** ← fastest | +0.24% |
| 8192 | 134.46 | 135.2 | +0.55% |

**The gap is flat at 0.02–0.55% and the winner agrees**, reproducing even the
fact that 8192 is slightly behind 4096.

**This removed a limitation.** "UBATCH values above half the prompt length
cannot be explored" was specific to per-pass measurement. Per request, any
prompt length compares every candidate. The recording condition became **two
completed requests** rather than two full passes — a matter of traffic, not
prompt length.

### Why this was not done from the start

The per-pass measurement **already existed for a different purpose**: separating
I/O time from compute time (§10.12), where the pass *is* the right unit. It was
reused for a different question without checking that the unit fit.

Then §10.11 established that partial passes made 1024 read 10% low, and an
exclusion was added. That observation is correct. But from then on, every problem
was fixed **locally, inside the frame**, and the higher-order problem — that the
exclusion rate is itself UBATCH-dependent — went unnoticed. The fifth defect,
`pf_close_pass`, is part of the same pattern.

**The cross-check against an independent instrument came far too late.**
llama.cpp's `prompt eval` had been in the log from the beginning. It was compared
only at the end, and one experiment settled it.

> **Fix your own instrument to an external one before building it.**

### Remaining limitations

- **Traffic differences between runs remain in the measurement.** Measuring
  candidate A on a quiet start and candidate B on a busy one distorts the
  comparison. Warm-up exclusion only works once two intervals exist, and
  differences in prompt-length distribution still leave a residue. In production,
  where one start serves many requests, the effect is small — but **not zero**.
- Stage 1's MRC counts **decode accesses only** (prefill goes through the arena
  and never touches the slab). At 200,000 samples that means **about 625
  generated tokens** before frac is recorded. Until then it runs at 0.15 —
  slower, but better than failing to start.
- Four restarts are needed to finish the candidate sweep. Ordinary use gets
  there in a few days; it is not instant.
- The search only moves within the candidate set. A true optimum between 1024
  and 2048 will not be found. Finer steps would only lengthen the search, so the
  coarseness is deliberate.

---

## 10.17 gpt-oss-120b — a fourth architecture and a second quantization format (2026-08-08)

### Result

**A 58.46 GiB model ran on a machine with 23.5 GiB of GTT.** Plain llama.cpp
does not even start.

```
36 layers x 128 experts -> 19 slots/layer (15%)
device memory : 14.48 GiB      (-75%)
prefill       : 106.8 tok/s
decode        : 259.9 ms/tok (3.85 tok/s)
output        : ' Paris.'
[BUG]         : 0 lines
load time     : 45 s
```

**Not one line of architecture-specific branching was added.** It is the same
code as the other three models.

### What was new about it

| | The other three | gpt-oss-120b |
|---|---|---|
| vendor | Qwen family / Laguna | **OpenAI** |
| expert quantization | IQ4_NL (type 20) | **MXFP4 (type 39)** |
| top_k | 8–10 | **4** |
| experts | 256 / 512 | 128 |
| bytes per slot | 57.9–202.8 MiB | **454 MiB** |

**That a different quantization format works is meaningful.**
MoEStream **places expert bytes into slots and rewrites ids**; it never
interprets the quantized data. ggml does the decoding. So format independence is
not luck — it is **structural**, and a second format confirms it.

The conditions are worth stating, though. It holds for formats that store
experts as **contiguous slices** of `[n_embd, n_ff, n_expert]`. A format that
interleaves experts would need separate verification.

### top_k matters more than total size

gpt-oss is **3.8 GiB larger** than Laguna yet uses 4 GiB less memory and is
faster at both decode and prefill.

| | Laguna-S-2.1 | gpt-oss-120b |
|---|---:|---:|
| file | 54.7 GiB | 58.46 GiB |
| top_k | 10 | **4** |
| device memory | 18.51 GiB | **14.48 GiB** |
| decode | 3.13 tok/s | **3.85 tok/s** |
| prefill | 79.7 tok/s | **106.8 tok/s** |

Because it reads fewer than half as many experts per token.
**"Bigger models are slower" does not hold. What matters is top_k.**

### The header alone settled compatibility, before the download finished

Tensor names and layout were checked from the GGUF header (at the start of the
file) without waiting for the download. One thing needed checking.

llama.cpp accepts experts in both a **fused** form (`ffn_gate_up_exps`) and a
**split** form (`ffn_gate_exps` / `ffn_up_exps`), looking for the fused one first
and falling back to split. MoEStream's `parse_name` only recognises the three
split names, so **a fused GGUF would not engage at all**.

gpt-oss fuses `gate_up_proj` on the Hugging Face side, so this was a real risk —
but this GGUF is split, with all 36 layers × 3 roles = 108 tensors present.

> Discovering "the format is wrong, it does not work" after downloading 62.8 GB
> is expensive. **The header is at the front of the file, so it can be read
> mid-download.**

Supporting fused models would mean adding a fourth role to `parse_name` and
fixing everywhere that assumes "one layer = three tensors". Not implemented.

### The measurement method was off by 20x

The first measurement gave prefill at **4.34 tok/s**. The real figure is
**106.8 tok/s**.

The cause was a prompt of only **12 tokens**. Prefill reads, per layer, the
experts referenced within the micro-batch. At 12 tokens × top-4 that is up to 48
experts per layer, about 20 GiB across 36 layers. **That fixed cost is divided
among 12 tokens.**

The same shape as the trap in §12.2, where `llama-bench`'s default of 512 tokens
was too short and made MoEStream look like 113% of plain llama.cpp. Opposite
direction, same cause: **prompt length changes what gets amortised.**

> Lesson: **when you see a prefill number, first ask how many tokens the prompt
> was.**

---

## 11. Design decisions overturned by measurement

**Of 32 ADRs, 12 were revised or rejected by measurement.** All are kept as
revision history in `DESIGN.md`. The main ones:

| Design-time decision | Measured | Outcome |
|---|---|---|
| io_uring is essential for bandwidth | **+1.5%** over parallel pthreads | **rejected**; 8 pread threads adopted |
| 2 MiB alignment in `.msp` improves bandwidth | 0.27% extra read, **+0.1%** bandwidth | **rejected** |
| PINNED (static residency) at 20% | pin=0.20 is **1.7 pt worse** | **changed to 5%** |
| predictive prefetch is the key | all three schemes net-negative | **rejected** |
| Expert Sweep lifts the slab constraint | PPL 520801 (a ggml bug) | **disabled by default** |
| hide I/O by executing in arrival order | only **0.5 ms** of I/O to hide | **ceiling too low; not adopted** |
| `soft` mode weight threshold τ=0.02 | skips **0.1%** | essentially inert; τ needs redesign |
| "the bottom 2–3 weights sum to a few %" | measured **15%** | the design was optimistic |
| residency uses BIOS-allocated VRAM | RADV reports the GTT heap as DEVICE_LOCAL | BIOS settings turn out to be irrelevant to performance |
| import an mmapped GGUF into Vulkan | RADV refuses with `VkResult -13` | **rejected**; cause confirmed as "file-backed VMAs are refused" (S5) |
| on UMA, allocate via host pointer import (ADR-0016) | an imported BO takes decode from 53.8 to 1023 ms/token **even unused** | **rejected** (S7); let ggml-vulkan allocate normally |

### Zero-copy did work

On UMA machines ggml-vulkan already allocates with
`eDeviceLocal | eHostVisible | eHostCoherent`. Reading straight into that with
`pread` **removes an intermediate buffer and one memcpy**:

| Configuration | Effect |
|---|---|
| zero-copy on (`MOESTREAM_ZEROCOPY=1`) | **−5.2%** |
| parallel I/O (1 → 8 threads) | +1.2% (decode) / about 1.4x on prefill |

**100% of reads take the zero-copy path.**

> Note: these were once reported as "+30%" and "−11%". That measurement was
> taken while another llama-server on the host was contending for the GPU. The
> table above is a re-measurement in a clean environment.

---

## 12. Reproducing this

```bash
git clone <repo> && cd moestream
cp .env.example .env
# set MODEL_DIR / MODEL_FILE / MS_PORT in .env
make up
# Web UI: http://localhost:8091
```

### 12.1 Measuring in one command — `research/tools/ms-bench.sh`

```bash
research/tools/ms-bench.sh --baseline           # also A/B against MOESTREAM=0
research/tools/ms-bench.sh --frac 0.25 --ub 1024
```

**This script never reports speed on its own.** Alongside it, it always reports:

| Item | Why it cannot be separated from speed |
|---|---|
| CPU / RAM / **GTT limit** / SSD | without the GTT limit, "fits / does not fit" is not reproducible. §12.4's "plain llama.cpp will not start" rests on this value |
| **the actual slot count** (not frac) | `0.25` is 64 slots at 256E and 128 at 512E. frac alone does not reproduce |
| **page cache state** | prefill swings more than 20% between cold and warm. On 2026-08-06 an identical baseline moved 242.8 → 295.6 tok/s (+22%) |
| **output correctness** (`The capital of France is`) | as §10.8 shows, **a broken implementation gives the best numbers** |
| **the `[BUG]` line count** | non-zero if even one read was discarded. A measurement with a non-zero count is void |

### 12.2 Using llama.cpp's own `llama-bench`

It is included in the image (`/opt/llama.cpp/build/bin/llama-bench`), but there
are **two traps** when pointing it at MoEStream.

**Trap 1: without an explicit `MOESTREAM_GGUF`, it quietly runs under different
conditions.** Unlike llama-server, llama-bench has no path to hand the model path
to MoEStream. Without it you get `GGUF path unknown`, and **top_k cannot be read
from metadata either and falls back to a default of 8**. No error; the numbers
come out anyway.

```bash
docker run --rm --device /dev/dri:/dev/dri --group-add "$(getent group render|cut -d: -f3)" \
  -v /path/to/models:/models:ro \
  -e MOESTREAM=1 -e MOESTREAM_CACHE_FRAC=0.25 \
  -e MOESTREAM_GGUF=/models/MODEL.gguf \
  --entrypoint /opt/llama.cpp/build/bin/llama-bench moestream/server:local \
  -m /models/MODEL.gguf -ngl 99 -ub 1024 -b 1024 -p 4096 -n 128 -r 2
```

**Trap 2: the default `-p 512` overstates MoEStream.** The arena's cost is
proportional to the number of passes (prompt length / UBATCH), so at `ub=1024`
a 512-token prompt makes one pass and the cost is never counted.

Measured (Ornith-1.0-35B / frac=0.25 / ub=1024, 2026-08-06):

| | plain llama.cpp | MoEStream | Ratio |
|---|---:|---:|---:|
| `pp512` (llama-bench default) | 281.2 t/s | **318.5 t/s** | **113%** ← "streaming is faster" |
| `pp4096` | 324.6 t/s | 247.1 t/s | 76% |
| `pp13312` | 279.9 t/s | 209.7 t/s | 75% |
| **real document, 13877 tokens** | 295.6 tok/s | 242.5 tok/s | **82%** |
| `tg128` | 23.84 t/s | 17.56 t/s | 74% |
| **real document, decode** | 24.10 t/s | 17.04 t/s | **71%** |

**Conclusions:**

- **decode measures correctly with llama-bench** (74% against 71% measured). Use
  it as a standard tool
- **for prefill, use `-p 4096` or more. Discard `-p 512` results**
- `pp13312` (75%) sits 7 pt below the real document (82%) because tokens
  generated by `std::rand() % n_vocab` have no expert locality. **That errs
  low**, so it is safe to publish but does not represent real use
- plain llama.cpp's speed does not depend on input content, so its llama-bench
  figures can be used directly as the baseline

### 12.3 A/B against the baseline

```bash
# MoEStream off (plain llama.cpp)
sed -i 's/^MOESTREAM=1/MOESTREAM=0/' .env && make up

# MoEStream on
sed -i 's/^MOESTREAM=0/MOESTREAM=1/' .env && make up
```

### 12.4 Individual measurements

| Target | Command |
|---|---|
| multi-turn (§6) | `python3 research/bench/agent_turns.py 8091 "label" 5` |
| SSD bandwidth (§4) | `research/spikes/s2_iouring_bw` |
| zero-slot verification (§9) | `research/spikes/s3_zero_slot` |
| split-addition verification (§9) | `research/spikes/s4_sweep_math` |
| expert distribution (§5) | `research/tools/analysis/expert_trace` → `research/tools/analysis/analyze_trace.py` |
| I/O disabled (§3.2) | `MOESTREAM_NOOP=3` |

### 12.4b Re-measured on a newer llama.cpp (2026-08-24)

Everything in §12.5 below was measured against `3581ba0cf`. When the Dockerfile's
default moved to `b0539c43`, all of it was measured again — **same machine, same
script, same conditions, only the commit differs.** That makes this a clean
upstream A/B rather than a re-run.

| | | `3581ba0cf` | `b0539c43` | |
|---|---|---:|---:|---|
| **Ornith-1.0** | plain memory | 17.29 GiB | 17.77 GiB | +0.48 |
| | plain decode | 41.5 ms/tok | 42.8 | +3.1% |
| | plain prefill | 295.6 tok/s | **256.2** | **−13.3%** |
| | streamed memory | 7.44 GiB | 7.93 GiB | +0.49 |
| | streamed decode | 58.7 ms/tok | 60.7 | +3.4% |
| | streamed prefill | 242.5 tok/s | 239.4 | −1.3% |
| **Qwen3-Coder-Next** | memory | 13.15 GiB | 13.44 GiB | +0.29 |
| | decode | 101.7 ms/tok | **67.8** | **−33.3%** |
| | prefill | 150.8 tok/s | 149.2 | −1.1% |
| **Laguna-S-2.1** | memory | 18.51 GiB | 18.79 GiB | +0.28 |
| | decode | 319.2 ms/tok | 326.7 | +2.3% |
| | prefill | 79.7 tok/s | 76.0 | −4.6% |
| **gpt-oss-120b** | memory | 14.48 GiB | 14.91 GiB | +0.43 |
| | decode | 259.9 ms/tok | 262.1 | +0.8% |
| | prefill | 106.8 tok/s | **92.2** | **−13.7%** |

Three things come out of this, and none of them are about MoEStream.

**Memory rose by 0.3–0.5 GiB everywhere, on both sides.** It rises by the same
amount with `MOESTREAM=0`, so it is upstream's, not this patch's. It is why every
percentage saving in this document moved a point or two: the baseline moved too.

**`qwen3next` decode got 33% faster upstream.** Qwen3-Coder-Next went from 101.7
to 67.8 ms/token with no change here. This is the largest single movement in the
table and it belongs entirely to llama.cpp.

**Prefill regressed 13–14% on two models.** Ornith's *plain* prefill fell from
295.6 to 256.2 tok/s while the streamed side barely moved (242.5 → 239.4). That
makes MoEStream's prefill look better — 93% of plain rather than 82% — and it
would be dishonest to present it that way. The streamed side did not improve;
the baseline got worse.

#### Dense, measured the same way

The dense figures elsewhere in this document came from ad-hoc scripts. These come
from the same `ms-bench.sh` run as the table above — **`ctx 32768`, like the MoE
rows**, which is why the memory differs from §12.4c below (`ctx 16384`):

| | plain llama.cpp | MoEStream (whole FFN) | |
|---|---:|---:|---|
| **Qwen3.8-27B** memory | 16.83 GiB | **8.15 GiB** | −52% |
| prefill | 69.1 tok/s | 67.0 tok/s | −3% |
| decode | 214.5 ms/tok | 681.0 ms/tok | 3.17x |
| **gemma-4-31B** memory | 19.76 GiB | **9.16 GiB** | −54% |
| prefill | 55.5 tok/s | **56.7 tok/s** | **+2%** |
| decode | 232.4 ms/tok | 704.6 ms/tok | 3.03x |

gemma-4's prefill is marginally *faster* streamed than resident. The difference
is within run-to-run spread and should not be read as a gain; what it does
establish is that prompt processing is free on a plain transformer as well as on
the hybrids, which had not been shown at these conditions before.

All ten measurements reported zero `[BUG]` lines and correct output.

---

### 12.4c Every model, against the baseline that exists (2026-08-25)

The tables elsewhere quote three of the four large models against **the size of
the file on disk**, because plain llama.cpp cannot start them and there is no
other reference. That is honest for those rows and misleading for the small ones,
where a real baseline does exist and was not used. Measured here with both sides
started in the same session, ctx 16384, `N_PARALLEL=1`, KV `q8_0`:

| | plain llama.cpp | MoEStream | |
|---|---:|---:|---|
| Qwen3.5-4B (dense, FFN streamed) | 3.95 GiB | **2.75 GiB** | −30% |
| Qwen3.8-27B (dense, FFN streamed) | 16.25 GiB | **7.57 GiB** | **−53%** |
| gemma-4-31B (dense, FFN streamed) | 19.04 GiB | **8.44 GiB** | **−56%** |
| Ornith-1.0-35B (MoE, `frac=0.25`) | 17.55 GiB | **7.71 GiB** | **−56%** |
| Ornith-1.5-35B (MoE, `frac=0.25`) | 20.40 GiB | **7.80 GiB** | **−62%** |
| Qwen3-Coder-Next (MoE, `frac=0.25`) | does not start | 13.18 GiB | −64% of the file |
| Laguna-S-2.1 (MoE, `frac=0.25`) | does not start | 21.09 GiB | −61% of the file |
| gpt-oss-120b (MoE, `frac=0.25`) | does not start | 20.57 GiB | −65% of the file |

**Qwen3.5-4B had been reported as a model streaming does not help.** It does:
3.95 → 2.75 GiB, a 30% cut. That claim came from comparing 2.75 GiB against the
2.71 GiB *file* rather than against the 3.95 GiB plain llama.cpp actually uses. A
model file is not a memory baseline — the runtime adds compute buffers, a KV
cache and an allocator's slack on top of it, and on a 4B model that overhead is a
third of the total.

Output was correct on all eight, and every figure reproduced across two starts.

---

### 12.5 Four-model summary (re-measured 2026-08-06 to 08-08)

**Measurement conditions** — always quote these alongside the table.

| | |
|---|---|
| CPU | AMD Ryzen 7 8745HS (8 cores / 16 threads) |
| GPU | AMD Radeon 780M (RADV PHOENIX), Vulkan, **UMA** |
| RAM | 30.6 GiB / **GTT limit 23.5 GiB** ← the fits/does-not-fit boundary |
| SSD | Crucial P310 (PCIe 4.0 NVMe), 4.48 GB/s effective |
| llama.cpp | `3581ba0cf` (build 10230) — see §12.4b for the same table on `b0539c43` |
| common settings | `CTX_SIZE=32768` / `UBATCH=1024` / `N_PARALLEL=1` / KV `q8_0` / FlashAttention on / `ngl 99` |
| prefill | a **13877-token real document** (`research/bench/prompt_long.txt`), `cache_prompt=false` |
| decode | **median** of 3 × 100 tokens after warm-up. A single run is unreliable (§13.3) |
| page cache | warm (the same model was read immediately before) |
| correctness | `The capital of France is`, 24 tokens greedy, plus zero `[BUG]` lines |

| | **Ornith-1.0-35B** | **Qwen3-Coder-Next** | **Laguna-S-2.1** | **gpt-oss-120b** |
|---|---:|---:|---:|---:|
| file size | 16.87 GiB | 36.54 GiB | 54.7 GiB (3 shards) | **58.46 GiB** (2 shards) |
| structure | 40 layers / 256E / top-8 | 48 layers / 512E / top-10 | 48 layers / 256E / top-10<br/>(first layer dense) | 36 layers / 128E / **top-4** |
| architecture | qwen35moe | qwen3next | laguna | **gpt-oss** |
| expert quantization | IQ4_NL | IQ4_NL | IQ4_NL | **MXFP4** |
| **plain llama.cpp** | 17.29 GiB<br/>41.5 ms/tok (24.10 tok/s)<br/>295.6 tok/s | **will not start**<br/>(exceeds 23.5 GiB GTT) | **will not start**<br/>(exceeds 23.5 GiB GTT) | **will not start**<br/>(exceeds 23.5 GiB GTT) |
| `MOESTREAM_CACHE_FRAC` | **0.25** (→ 64 slots) | **0.25** (→ 128 slots) | **0.20** (→ 51 slots) | **0.15** (→ 19 slots) |
| **device memory** | **7.44 GiB** (−57%) | **13.15 GiB** (−64%) | **18.51 GiB** (−66%) | **14.48 GiB** (**−75%**) |
| **decode** | **58.7 ms/tok (17.04 tok/s)** | **101.7 ms/tok (9.83 tok/s)** | **319.2 ms/tok (3.13 tok/s)** | **259.9 ms/tok (3.85 tok/s)** |
| **prefill** | **242.5 tok/s** | **150.8 tok/s** | **79.7 tok/s** | **106.8 tok/s** |
| vs plain llama.cpp | decode 71% / prefill 82% | — (nothing to compare) | — (nothing to compare) | — (nothing to compare) |
| output correctness | **exact match** with plain llama.cpp | normal | normal | normal |
| auto: arena | 498 MiB × 2 | — | 1494 MiB × 2 | 1614 MiB × 2 |
| auto: prefetch strategy | read all | read all | **union** | **union** |

**gpt-oss-120b is larger than Laguna yet uses less memory and is faster.**
Because of top-4: it references fewer than half as many experts per token as
Laguna (top-10), so it reads proportionally less. **top_k matters more than
total model size.** Its −75% is the largest reduction of the four.

`prompt_long.txt` counts as **13876 tokens** under gpt-oss's tokenizer (13877
for the other three). A different tokenizer could easily have differed by
hundreds; it happened to land within one, so this column is directly comparable.

**How to read this table.**

- **Only Ornith can be compared against plain llama.cpp**; the others do not fit
  on this machine. The question is not "is MoEStream faster" but **"does
  something that cannot run, run usably"**.
- Ornith: **43% of the memory, 71% of decode and 82% of prefill**. Quality is
  PPL +0.21% with greedy bit-identity (§8).
- Qwen3-Coder-Next, an 80B-class model, at **13 GiB and 9.8 tok/s** — a
  practical speed.
- Laguna at 54.7 GiB gives **3.1 tok/s**: it runs, but is not recommended for
  constant use. It is included as a record of what happens outside the intended
  range (15–35 GiB).
- **decode does not degrade linearly with model size** (16.9 → 36.5 GiB is 1.8x
  the memory and 1.7x the decode time) because what governs it is whether the
  total expert bytes fit in page cache. Laguna (50.7 GiB of experts against a
  projected 14.8 GiB of page cache) is the first to become genuinely I/O bound.

---

## 13. What is new here

### 13.1 Redefining the bottleneck

> **[2026-08-23]** This section's framing survives, with one correction that
> changes what "I/O" refers to. See §14's index and
> `findings/S19-pagecache-share.md`: on a model whose experts fit in RAM the
> reads are page-cache copies, not device reads, and the two behave differently
> enough that a conclusion drawn on one does not transfer to the other.

Existing MoE offloading work (Klotski / ProMoE / MoE-Infinity) starts from
**"SSD bandwidth is insufficient"**. That is correct at 700B class — about 11 GB
must be read per token, which no SSD can serve.

It does not hold at 35B class on a unified-memory machine:

| | 700B class (regime A) | **35B class (regime B)** |
|---|---:|---:|
| `B_act` (bytes/token) | ~11 GB | **455 MiB** |
| `t_c` (compute/token) | — | **43.7 ms** |
| bandwidth needed | tens of GB/s | **~10 GB/s (before caching)** |
| real I/O after a 38% cache | — | **43.34 MiB/token** |
| **actual I/O cost** | dominant | **0.5 ms/token** |
| **actual bottleneck** | I/O | **CPU↔GPU sync, 11.4 ms** |

**In regime B, I/O hides completely behind compute.** What needs optimising is
neither bandwidth nor prediction accuracy but **the number of synchronizations.**

> **[Added 2026-08-07] This conclusion applies to the code it was measured on.**
> After union reads, async prefetch and zero-copy, I/O rose to 7.8–12.5 ms and
> synchronization fell to 2.2 ms — the ordering inverted (§10.12). The regime
> distinction between A and B still holds; what changed is where the bottleneck
> sits within regime B.

### 13.2 Practical implications

1. **Predictive prefetch is better left unimplemented in this regime.**
   Even an 81.4%-accurate predictor is net-negative: the synchronization needed
   to get its inputs off the GPU costs more than the I/O it can hide.

2. **A static hot set (PINNED) loses to a dynamic cache.**
   In some domains plain LRU beats an oracle hot set chosen by calibration.

3. **The prompt cache determines how it feels in practice.**
   Prefill is 4.1x slower, but from the second turn evaluated tokens fall by
   two orders of magnitude and the gap narrows to 1.27x. Judging "unusable" from
   a synthetic benchmark alone would be wrong — and nearly was.

4. **Measure the cost of shrinking ubatch separately from the cost of
   streaming.** Of a 4.9x prefill degradation, 3.9x came from ubatch and only
   1.25x from MoEStream.

### 13.3 Measurements we got wrong (kept as a record)

> **Added 2026-08-06: an instrument and a measurement can point opposite ways.**
> Adding 1 (no threads at all) to the I/O thread candidates was considered.
> Ornith's decode measured fastest at 1 thread (56.29 vs 56.96 ms at 4), but
> **the auto-tuner's bandwidth measurement ranked 1 last** (8.26 vs 9.80 GB/s at
> 6). The 1.2% difference is within measurement variation (±1 ms) and is not
> explained by bandwidth. The original explanation — "thread creation costs more
> than the read" — was also wrong. **Adding a candidate on weak evidence only
> adds the risk of choosing wrongly**, so it was dropped. (On Laguna, 1 thread
> is 2.3x slower: 655.4 vs 285.3 ms at 16.)

> **Added 2026-08-06: a single measurement created "the biggest open item".**
> S7 recorded "decode cold start is 1.8x" and made it the top priority. Measured
> properly, it **resolves within 25 tokens, at a total cost of about 525 ms once
> per start**. The 94.6–102.8 ms it rested on were single measurements taken
> during a UBATCH sweep, attributing ordinary variation to cold start.
> **The same error was made in the frac sweep** (a single measurement reported a
> "+33.1 ms cliff"; five paired runs gave +8.9 ms). Do not use single
> measurements.

To keep the numbers here trustworthy, our own mistaken measurements are recorded:

| Error | Correct value |
|---|---|
| reported SSD bandwidth as 10.6 GB/s | that was page cache. O_DIRECT gives **1.51 GB/s** |
| compared baseline at ub=512 with MoEStream at ub=8 | neither PPL nor speed compares unless ubatch matches |
| parallel I/O +30% / zero-copy −11% | another process was contending for the GPU. Correct: **+1.2% / −5.2%** |
| Expert Sweep "prefill 2.9x" | output quality was never checked (PPL 520801) |
| "unsuitable for coding agents" | the prompt cache had not been measured. **It is suitable** |
| headline "I/O 0.5 ms vs sync 11.4 ms" | correct for the code of the time; the current code is the reverse (§10.12) |
| per-pass prefill measurement | the exclusion rate itself depends on UBATCH; the request is the correct unit (§10.16) |

---

## 14. Open items

> **[2026-08-23] Several rows below were settled by a later measurement round.**
> The findings are the primary sources; this table is the index.
>
> | | outcome |
> |---|---|
> | `soft` mode τ | **rejected on measurement.** Implemented as a rank predicate and measured: +10.2% perplexity for −3.2% decode (S17 / S24 / S25) |
> | quantization tiers (Q2_K) | **rejected on thesis.** Every form pays for memory with accuracy, which is the trade this project exists to avoid. A Q2_K run also produced the session's fastest decode from a broken configuration — §10.8 recurring (V3) |
> | dense models | **overturned.** Dense FFN streaming implemented: −54 to −58% memory with perplexity identical to four decimals, and prompt processing free above ubatch 1024. Generation costs **1.94–2.86x** depending on batching and speculation — measured with the baseline in the same configuration, after three earlier figures were found to have compared against a differently-configured one (S18 / S21 / S27 / S29 / S30 / S33 / S34) |
> | "the bottleneck is I/O" | **needs qualifying.** 98.7% of Ornith-1.0's decode reads never reach the SSD; "I/O" is a page-cache copy on models that fit and a device read on models that do not (S19) |
> | the memory saving | **priced.** ~8 of the 9.8 GiB freed is usable by a co-tenant at +3.3%; past that there is a cliff to +63% (S26) |
> | comparison discipline | **three dense figures were reported against a baseline left in another configuration**, each flattering streaming. Corrected in S34, which measures the baseline in every configuration the streamed side is measured in |
> | speculative decoding | **sign depends on the machine and the model together, and is now learned rather than assumed.** Two costs decide it, both measured directly: a MoE verification pass gets 3.5x dearer from width 1 to 6 while a dense one is flat, and one drafted token costs ~53% of a MoE forward pass against ~11% of a dense one. Result: 2.29x on a streamed dense model; on MoE it loses, streamed and resident alike (48.6 → 51.1 ms/tok at the best setting, −65% at llama.cpp's defaults). `SPEC_DECODING=learn` tries one draft size per start with "off" among the candidates (S42 supersedes S31 / S32) |
> | sizing a model from its header | **three architectures were being mis-sized.** Per-layer GQA, sliding windows, and hybrid recurrence each break the obvious KV formula; gemma-4 was offered 4096 tokens where 32768 fits in 19.03 GiB. A MoE model's resident size is a property of the machine, not the file, and cannot be quoted from the header at all (S40) |
> | speculative decoding, revisited | **now enabled automatically where it pays.** Which kinds a model supports is decided by llama.cpp's own `common_speculative_types_from_gguf()`, called from a 30-line tool that links upstream's `common` — the rule is not copied here. Worth **2.71x** on a streamed dense model with a head; the model-free `ngram-*` kinds do nothing when streamed and cost 16% when not, because speculation only pays at an acceptance rate that covers verification (S42) |
> | remaining speed work | **none within the thesis.** Byte reduction only pays in large steps (22% removed buys 10% of the I/O cost), and large steps require giving up accuracy (S20 / V3) |

| Item | Status | Prospects |
|---|---|---|
| CPU↔GPU synchronization | now ~2.2 ms; was a structural floor at 11.4 | removable by adding slot indirection to ggml |
| Expert Sweep | disabled by ggml graph aliasing | needs a fix in ggml core |
| ~~prefill 59.7 tok/s~~ | **solved (S7 / S12 / S14)** | staging arena + `UBATCH=1024` + union reads + async prefetch. 46.0 → **242.5 tok/s (5.3x)**, +0.76 GiB, output bit-identical to plain llama.cpp |
| quantization tiers (Q2_K for streamed experts) | **rejected 2026-08-23** | off-thesis in every form; see `V3-idea-analysis.md` |
| `soft` mode τ | **rejected 2026-08-23** | built and measured: +10.2% PPL for −3.2% decode (`S24-skiprank-verdict.md`) |
| fused-expert GGUFs (`ffn_gate_up_exps`) | not supported | needs a fourth role in `parse_name` (§10.17) |
| other architectures (DeepSeek / GLM / Mixtral) | adapter designed, unverified | — |
| dense models | **supported 2026-08-23** | FFN streaming, −56%, no accuracy cost (`S27-dense-streaming-impl.md`). Attention needs a different design (`S30-dense-attention.md`) |
| a test suite beyond logic checks | `make test` covers logic only | performance and output correctness need real hardware |
| does `learn` pick the fastest frac? | **yes, and the curve is monotonic** | re-measured 2026-08-24 with a proper warm-up: 0.15 → 0.60 runs 70.1 → 49.6 ms/tok with no turning point, approaching the 42.6 resident baseline. The earlier claim that it turned over at 0.60 was a warm-up artefact, and so was the worry that a hit-rate proxy is blind to it (`S41-frac-curve.md`) |
| how noisy is a MoE measurement? | **eight times a dense one, and it decays rather than scatters** | three claimed improvements were reported and withdrawn in one session, all on MoE. Dense samples spread 0.03%, MoE 8.1%, and the first sample is always the slow one — so whatever is measured second looks better (`S43-warm-up.md`) |
| upstream's own MoE offload | **measured, published with a caveat** | `--cpu-moe` runs 71.8 ms/tok against MoEStream's 63.7 at frac 0.25. Our GTT+VRAM probe shows it freeing no device memory (18.02 GiB against 17.78 plain), but llama.cpp does not print its buffer breakdown at this verbosity, so that half is unconfirmed and is not quoted outside this table (`S44-upstream-cpu-moe.md`) |
| upstream tracking | **default moved to `b0539c43` 2026-08-23** | built and verified on MoE and dense, all 5 patch blocks still attach. Alternating warm-cache A/B against the `3581ba0c` measurement baseline: decode within 1.2%, output identical, +0.51 GiB — the same +0.51 GiB appears with `MOESTREAM=0`, so it is upstream's |
| every model, every learn mode | **verified end to end 2026-08-25** | eight models, both passes: memory drops on all of them (4B to 120B, 30–65% against a real baseline), output correct everywhere, and all three learn loops close — including the concurrency key that keeps a draft size learned at one `N_PARALLEL` from being applied at another. Two shipped defects were found only by *using* the feature: the recorder called `curl`, which is not in the image, and wrote state files unreadable from the host (`S47-every-model-verified.md`) |
| every other llama.cpp knob | **measured, none help** | `--poll`, `--threads-batch`, `--no-op-offload`, `--mlock`, `--cache-reuse`, KV at `q4_0` traded for slots, and upstream's `--cpu-moe` — twelve settings, no movement on either family. The knobs that matter were already the ones with `learn` on them (`S46-parameters-that-do-nothing.md`) |
| measurement tooling | **two bugs fixed 2026-08-24** | `ms-bench.sh` let `.env` overwrite a model named on the command line, so a whole table could be measured on the wrong model with no visible failure; it now takes `--model` / `--ctx` / `--dense-frac` / `--spec`. And `make bench` did not read `.env.launcher`, so anyone set up by the launcher was benchmarking a different configuration from the one running |
| launcher memory planning | **corrected 2026-08-23** | the KV estimate now reads per-layer GQA, sliding windows, and `full_attention_interval` (`S40-header-sizing.md`) |

---

## Appendix: index of key figures

| Symbol | Meaning | Measured |
|---|---|---|
| `B_act` | expert bytes touched per token | 455 MiB/token |
| `t_c` | compute time per token | 43.7 ms |
| `BW` | SSD effective bandwidth (O_DIRECT, saturated) | 4.46–4.49 GB/s |
| `h_req` | hit rate needed to stay within a 20% slowdown | 51.0% |
| `h(0.38)` | measured hit rate at a 38% cache | 82.4–91.5% |
| `t_io` | real I/O time after caching | **0.5 ms/token** (2026-08-04) → **7.8–12.5 ms** (2026-08-07, §10.12) |
| `t_sync` | CPU↔GPU synchronization | **11.4 ms/token** (2026-08-04) → **2.2 ms** (2026-08-07, §10.12) |
| one expert | gate + up + down combined | 1.422 MiB |
| I/O after caching | at the 38% configuration | 43.34 MiB/token (9.5% of full) |