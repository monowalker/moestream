# MoEStream v3 — What is left, re-derived from the current bottleneck

An idea catalogue in the same form as `V2-idea-analysis.md`, but grounded in the
**post-inversion** measurements rather than the 2026-08-04 ones V2 rested on.

V2's rejections were reached when decode was believed to be 11.4 ms of
synchronization against 0.5 ms of I/O. §10.12 inverted that: synchronization is
2.2 ms and I/O is 7.8–12.5 ms. §10.14 re-examined **one** consequence of the
inversion (cross-layer prefetch) and correctly rejected it on the overlap
window. The rest of Category 1 — eleven items rejected because "I/O is not the
constraint" — has not been revisited, and this document does that.

## The budget being spent against

Ornith-1.0-35B, `frac=0.25`, from §10.12:

| | ms | share |
|---|---:|---:|
| compute (plain llama.cpp) | 41.96 | 71% |
| CPU↔GPU synchronization | 2.2 | 4% |
| cache management | 2.3 | 4% |
| **I/O** | **12.45** | **21%** |
| total | 58.75 | |

Laguna-S-2.1, `frac=0.15`: I/O is **63.7%** of decode. Two very different
regimes, and only the second is the one the project's headline models live in.

**Everything worth doing now either removes bytes or removes the cost of moving
them.** Nothing else is large enough to matter.

---

## C1 — Where do the reads actually come from? ★ do this first

The project's own numbers do not add up, and the discrepancy is load-bearing.

```
misses/token at frac 0.25   = 320 fetches x 18.13%   = 58.0 experts
bytes/token                 = 58.0 x 1.422 MiB       = 86.5 MB
measured I/O (D-C, §10.12)  = 12.45 ms
                            -> 6.95 GB/s
measured device ceiling (§4.1, O_DIRECT saturated)   -> 4.46-4.49 GB/s
```

**The read path is running at 1.55x the speed the device can deliver.** The
built-in `[prefetch]` blocking-time instrument makes it worse, not better: 11.9%
of decode is ~7 ms, i.e. 12.4 GB/s. Both instruments say the same thing.

There is only one explanation: on Ornith, **most reads never reach the SSD.**
The expert data is 14.5 GiB, the host has 30 GiB, `pread` is buffered
(`open(..., O_RDONLY)`, no `O_DIRECT`), and the kernel is serving the majority of
misses out of the page cache. The source already anticipates this — the
`[prefetch]` decision comment reasons about exactly this fit — but the comment
says "Ornith → I/O is 1% of decode" while §10.14 measures 11.9%. One of those is
stale.

Consequences, all of them significant:

- **The 12.45 ms is not disk time.** It is largely `copy_to_user` into
  GPU-visible memory. It is bounded by memory bandwidth and by the write path
  into host-visible device-local memory, not by the NVMe.
- **`O_DIRECT` would make Ornith slower**, not faster. Worth stating explicitly
  so nobody tries it.
- **Every "reduce bytes" estimate below is conservative for Laguna and
  optimistic for Ornith**, because Ornith's bytes are cheaper than the model
  assumes.
- **The memory claim needs a sentence.** Ornith's honest figure is 7.44 GiB of
  hard residency **plus** a working set the kernel keeps in reclaimable page
  cache. That is still a real win — reclaimable is categorically different from
  a GTT allocation — but the docs currently do not say it, and a reader on a
  16 GiB machine will get different numbers.

**Cost to settle: one afternoon.** `/proc/self/io`'s `read_bytes` counts only
block-device I/O; compare it against bytes requested across a decode run. Or
drop the page cache and re-measure. No design change either way.

### The live instance is already the control group

The measurements above are all from **Ornith-1.0-35B-UD-IQ4_NL**. The instance
running on this machine today is **Ornith-1.5-35B-Q4_K_M**, which is a different
model in exactly the way that matters:

| | Ornith-1.0 UD-IQ4_NL | Ornith-1.5 Q4_K_M |
|---|---:|---:|
| layers | 40 | 41 |
| streamed experts | 14.48 GiB (85.9%) | **18.65 GiB (92.3%)** |
| bytes per expert | 1.448 MiB | **1.819 MiB (+26%)** |
| total | 16.86 GiB | 20.21 GiB |
| fits the page-cache headroom? | 14.5 vs ~17 GiB — **yes** | 18.7 vs 17.0 GiB — **no** |

And the runtime's own bandwidth auto-tuner reports, on the live workload:

```
moestream: [io] settled on 6 threads (measured 3.72 GB/s)
moestream: [prefetch]   experts total 18.7 GiB / page cache headroom 17.0 GiB
```

**3.72 GB/s — below the 4.48 GB/s device ceiling, not above it.** Same code, same
machine, same instrument family; the effective read bandwidth differs by roughly
3x between the two builds, and the thing that differs is whether the expert set
fits in RAM.

That is the C1 hypothesis observed rather than inferred, and it settles the
framing without any new measurement: **"I/O" in this system is two different
costs wearing one name.** Under the page-cache ceiling it is a copy into
GPU-visible memory; over it, it is the NVMe. Which one you are paying decides
which of the items below is worth doing.

It also means the running configuration is genuinely I/O bound at the device:

```
misses/token = 41 layers x 8 x 15.2%  = 49.9 experts
bytes/token  = 49.9 x 1.819 MiB       = 90.8 MiB
at 3.72 GB/s                          = 25.6 ms/token blocked on reads
observed decode                       = 98-125 ms/token (8-10 tok/s)
```

Roughly a quarter of decode is *blocking* read time, and §10.12's difference
method puts the effective cost ~1.6x higher again. Every "reduce bytes" item
below is worth more on this model than on the one the docs measured — which is
the opposite of the usual direction for a follow-up, and worth saying out loud.

---

## C2 — Is the write into GPU-visible memory the real cost? ★ largest single unknown

Follows directly from C1. If the bytes are already in RAM, then what the 12.45 ms
buys is a copy into `eDeviceLocal|eHostVisible|eHostCoherent` memory, which on
AMD is mapped **write-combined**. Measured elsewhere in this project, a plain
`memcpy` between ordinary buffers runs at **18.0 GB/s** (§10.14). The read path
achieves 6.95–12.4 GB/s into the slab.

If that gap is the WC write path rather than the SSD, it is worth **up to
6 ms/token on Ornith (10%)** and it costs no memory.

Spike: with a Vulkan buffer allocated the way the slab is (reuse
`s0b_backend_slab`), compare, with the source page-cache-hot,

1. `pread` into `malloc`'d memory
2. `pread` into the slab's host pointer
3. `memcpy` from a page-cache-hot `mmap` into the slab's host pointer
4. the same with non-temporal stores

If (1) ≫ (2), the destination is the limiter and (3)/(4) are the fix. If they
match, the kernel copy is already optimal and this whole line closes. Either
outcome is worth having, and neither needs a model.

---

## A — Reduce the cost of moving bytes

### A1 — Intra-layer read/compute split (defer `down`) — **not yet evaluated, and it should be**

The remap CPU op reads `gate`, `up` and `down` for every miss and only then
returns, so the FFN cannot start until all three have landed. But `down` is not
needed until after the gate/up GEMM has run.

This is idea **R** ("load/compute pipeline") from V2, rejected there with
"arrival-order execution has a ceiling of 0.5 ms". **That ceiling is now
12.45 ms on Ornith and ~205 ms on Laguna.** The reason for the rejection is off
by 25–400x and the idea has never been re-costed.

It is also the one prefetch shape that escapes §10.14's objection. §10.14 killed
the safe (S14-style) design because it needs a 4.77 ms relay copy: a cross-layer
prefetcher does not know the ids yet, so it cannot know which slot to write, and
must stage. **Here the ids are already known and the slot is already acquired**,
so the background thread reads straight into its final GPU-visible destination.
No relay, no prediction, and no concurrent access to `ExpertCache` — the slot is
reserved before the thread starts and released after it joins.

Rough size: `down` is roughly 35–40% of expert bytes. What can be hidden is
bounded by the gate/up GEMM, not by the whole inter-remap window (1.282 ms on
Ornith, 2.897 ms on Laguna) — and that GEMM has never been timed separately.

**Spike before code**: add a counter for the gate/up-only interval, the way
`[overlap]` was added before the prefetch work. If it is shorter than the
`down` read, the idea is dead and costs nothing to have asked. The risk that
kills it is ggml's graph splitting forcing a sync at the CPU node boundary,
which would serialise what this is trying to overlap.

### A2 — Weight-aware miss skipping — **measured; see finding S17**

At `frac=0.25`, τ=0.06 removes **10–12% of decode read bytes for 0.62–0.82% of
router weight mass** — 13.6–16.2x, against 3.5–4.7x for the blanket rule M0-2
rejected. Same bytes, a quarter of the quality cost, because skipping a *hit*
loses weight mass and saves nothing.

Worth −2.3% on Ornith and ~−7% on Laguna at τ=0.06; ~−20% on Laguna at τ=0.08.
Costs no memory. **Blocked on perplexity**, which is mandatory before it can be
a default.

### A3 — A persistent read thread pool — small, real, cheap

`run_reads_parallel` spawns threads per call: 40 layers x n threads per token.
§10.12's own thread sweep prices a spawn at ~6 µs (1 thread 58.92 ms vs 8
threads 60.56 ms across 280 extra spawns), so at 4 threads this is ~0.7 ms/token,
about **1.2%**. The comment records that a condvar pool deadlocked once; a
latch-per-batch is a much smaller thing to get right than what was tried.

Low priority, but it is free memory-wise and the measurement is already done.

---

## B — Reduce the bytes themselves

### B1 — A low-precision streaming tier — **rejected: it is off-thesis, and the measurement went wrong**

`RESULTS.md` §14 has listed this as untested since the beginning, and the
arithmetic makes it the largest thing on the board. It is still the wrong thing
to build, for a reason that has nothing to do with its size.

**The project's claim is "you no longer have to pay for memory with accuracy".**
Every shape of this idea pays for memory with accuracy:

| shape | what it approximates |
|---|---|
| (a) uniform low precision | every weight |
| (b) two-tier residency: slab bit-exact, streaming source low | every expert that misses (~18% of lookups) |

(b) was drafted here as "the version that keeps the central claim intact". It
does not. `every weight is bit-identical to the original` is false the moment
any expert is served from an approximated copy. There is no shape of B1 that
is not a quantisation trade wearing different clothes.

**Combined with S20, this closes the byte axis entirely.** S20's curve says only
large byte reductions pay (removing 22% of reads buys 10% of the I/O cost;
removing 63% buys 98%). The only ways to remove 60% of the bytes are to
approximate them or to not need them. Lossless compression of already-quantised
weights recovers a few percent, nowhere near the knee.

> **There is no way to cut decode read volume by the amount that would matter
> without giving up accuracy. Within the project's own thesis, the byte axis is
> finished.**

**What the measurement showed before it was abandoned.** A Q2_K requantisation
was built and run before the above was thought through properly:

| | decode | memory | PPL |
|---|---:|---:|---:|
| IQ4_NL, frac 0.25 | 57.85 ms | 7.46 GiB | 5.0919 |
| Q2_K, frac 0.25 | 48.26 ms | 5.26 GiB | — |
| Q2_K, frac 0.50 | 39.63 ms | 7.97 GiB | **322.03** |
| Q2_K, no streaming | — | — | 8.7086 |

Q2_K alone costs **+71% perplexity** (5.09 → 8.71), which settles it on quality
alone. But note the last two rows: streaming is supposed to be lossless, and
5.09 → 8.71 → **322** is not a quantisation artefact, it is a **bug**. And the
configuration that produced it was also the fastest measured all session, at
39.63 ms.

> That is §10.8's lesson recurring exactly: **a broken configuration produced
> the best speed number in the session.** It is recorded here so that 39.63 ms
> is never quoted. The cause was not investigated, because the direction was
> rejected on thesis grounds regardless.

The artefacts (the requantised GGUF, the spike, the image that built it) were
deleted rather than kept, because an off-thesis branch left lying around reads
as drift in what the project is for.

**B3 — a second NVMe.** The 4.48 GB/s ceiling is one device. The reason on
record for dismissing this is "the second drive is a Windows disk" — a fact about
the verification machine, not a principle. For Laguna-class models the SSD
genuinely is the wall, so RAID0 across two NVMe is a real deployment answer and
belongs in `USAGE.md` as such, not in the rejected pile.

---

## D — Dense models — **the standing claim is wrong; see finding S18**

NG-4 says streaming "cannot work in principle" for dense because
`B_act = total size`. That is true per **pass**, not per **token**, and prefill
puts U tokens through one pass.

Measured from `Qwen3.8-27B-IQ4_NL` (15.22 GiB, 65 layers):

- streaming the body through 2 one-layer arenas: **2.13 GiB resident, −86%** —
  a *larger* reduction than any MoE result here (gpt-oss-120b is −75%)
- **prefill is free above ubatch ≈ 105**: the compute ceiling is 32.4 tok/s and
  the existing async arena hides the I/O completely from ubatch 128 upward
- **decode is where it actually fails**: 13.57 GiB/token → 3.25 s/token
  (0.31 tok/s)

So the accurate statement is not "no benefit" but **"the largest memory
reduction available, paid for entirely in decode"** — the same trade the README
already sells for gpt-oss-120b, at a much worse point on the curve.

And on unified memory the usual counter-argument does not apply: `--n-gpu-layers`
moves a layer from GPU to CPU, but both read the same DDR5, so it reduces total
footprint by **zero bytes**. On a UMA box, SSD streaming is the only mechanism
that reduces the number at all. The dense case is *stronger* here than the design
document allows.

Three positions, in ascending order of difficulty: **D1** prefill-bound dense
workloads (real today; needs only `parse_name` to accept plain `ffn_*` and a
whole-layer arena), **D2** dense models that do not fit at all, at ~0.3 tok/s,
**D3** activation sparsity, where MoEStream's slot table and id remap already are
the right substrate — slice the FFN into G groups and each group *is* an expert —
but which needs an offline-trained predictor per model and is genuine research.
Details, including the density each target rate demands, are in S18.

---

## What is not worth revisiting

Unchanged from V2, and the inversion does not rescue any of them:

| | Why it stays rejected |
|---|---|
| cross-layer predictive prefetch | §10.14. Not the balance sheet — the **overlap window**. Ornith has only 0.187 ms/layer to hide; Laguna's compute finishes first in 82% of layers |
| `io_uring` / `preadv` batching | §10.13. Fixed cost is 0.348 ms/token, a 0.6% ceiling. Read time is data movement, not call count |
| speculative decoding | S13. Raises bytes per accepted token ~4x. In an I/O-bound system, spending compute to buy serialisation is the wrong direction |
| a static PINNED hot set | §5.4b. Loses to dynamic by 8.1 points |
| LFU admission | §10.15. S3-FIFO wins all 15 points |
| Expert Sweep | ggml graph buffer aliasing. Needs a fix in ggml core, not here |

---

## Order of work — as it turned out

Everything on the original list was measured. The result is that **nothing on
the speed side is worth building.**

| | item | outcome |
|---|---|---|
| C1 | page cache vs device | **answered (S19).** 98.7% of Ornith-1.0's decode reads never reach the SSD. "I/O" is two costs sharing a name |
| C2 | write-combined write path | not run. S19 makes it moot for models that fit; for those that do not, the device is the wall |
| A2 | weight-aware miss skipping | **implemented and rejected (S17/S24/S25).** +10.2% PPL for −3.2% decode. The offline weight-mass proxy under-predicted the real PPL cost by 5x |
| B1 | low-precision streaming tier | **rejected on thesis** (above). Off-concept in every shape |
| D1 | dense prefill support | **not speed — a memory feature.** −86%, no accuracy cost. The one thing here still worth building |
| A1 | intra-layer split (defer `down`) | **not measured.** The only lossless speed idea left; S20 caps it at a few percent |
| A3 | persistent read thread pool | not measured; ~1.2% |
| D3 | activation sparsity | research; unchanged |

### The conclusion this session actually reached

```
S20:  only large byte reductions pay      (22% removed -> 10% of the I/O cost)
B1:   large byte reductions require giving up accuracy
      => within the thesis, the byte axis is closed
S19:  and on models that fit in RAM, those bytes were never expensive anyway
```

**The remaining speed levers are A1 and A3, worth a few percent between them,
and neither has been measured.** That is the honest state: the project is
substantially finished for speed, and what looked like a rich seam of untried
ideas was mostly ideas that had been rejected for the right reason under a
stale rationale, plus one (A2) that was worth testing and failed.

The memory axis is a different story — see D, and finding S26, which measures
what the freed memory is actually worth when a co-tenant uses it.

