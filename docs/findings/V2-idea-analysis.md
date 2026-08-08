# MoEStream v2 — Evaluating and selecting ideas

An evaluation grounded in measurement. The criteria are the values measured in
this session, not guesses.

> **[Note added 2026-08-07]** The measurements this document rests on are from
> 2026-08-04. **The bottleneck has since inverted** (`RESULTS.md` §10.12): with
> union reads, async prefetch and zero-copy in place, synchronization fell to
> 2.2 ms and I/O rose to 7.8–12.5 ms. The rejections below still stand — they
> lost on direct measurement, not on the reasoning — but the reasoning that
> "I/O is not the constraint" no longer describes the current code.

## The measurements this rests on

| Observation | Value | Source |
|---|---|---|
| decode I/O cost | **0.5 ms/token** | N3 follow-up |
| decode CPU↔GPU synchronization | **11.4 ms/token** (40 layers × 0.285 ms) | N3 follow-up |
| decode speed | 55.0 ms vs baseline 43.7 ms (**79%**) | here |
| memory | 7.9 GiB vs 16.9 GiB (**−53%**) | N3 |
| PPL degradation | **+0.11%** | here |
| prefill (ub=512) baseline | 295.1 tok/s | here |
| **prefill (ub=8) baseline** | **74.9 tok/s** | here |
| prefill (ub=8) MoEStream | 59.7 tok/s | here |

### The decomposition that matters most

```
prefill 4.9x slower  =  3.9x (ubatch forced to 8)  x  1.25x (MoEStream itself)
```

**MoEStream's own overhead is only 1.25x.** What dominates is the structural
constraint that a small slab cannot run a large ubatch.

### Stating that constraint precisely

```
distinct experts referenced by one mul_mat_id call  <=  slab slot count

union(ubatch) ~= n_expert x (1 - (1 - top_k/n_expert)^ubatch)
  ub=8 -> 58    ub=32 -> 176    ub=128 -> 250    ub=512 -> 256
```

It arises because llama.cpp processes all tokens with "one layer = one
`mul_mat_id`". Every expert in the batch must be resident **simultaneously**.

---

## Evaluating the proposed ideas

### Category 1: already refuted by measurement (11 items)

Decode's I/O cost is only **0.5 ms/token**. Proposals that reduce I/O waiting
target a bottleneck that does not exist.

| ID | Idea | Verdict |
|---|---|---|
| C | expert prediction prefetch | **implemented and rejected** (P1/P2/P5 all net-negative, N2) |
| D | context-aware expert cache | same (equivalent to P4; what it predicts is already cached) |
| M | adaptive expert cache | same |
| AH | speculative expert loading | same |
| AS | background expert loader | same (a persistent thread was implemented and rejected) |
| N | mmap/page cache optimization | I/O is not the constraint. But **useful on the prefill path** (below) |
| O | io_uring streaming | **implemented and rejected** (+1.5% over parallel pread, S2) |
| P | GPU Direct Storage | meaningless in principle on UMA (§13.7); zero-copy already solves it |
| Q | NVMe parallel streaming | bandwidth is not the constraint, and the second drive is a Windows disk |
| R | load/compute pipeline | arrival-order execution has a ceiling of **0.5 ms**. Not worth the effort |
| AN | compute-aware streaming | same |

> **These have already been eliminated by measurement. Do not reimplement them.**

### Category 2: aimed at decode's 11.4 ms of synchronization

The dependency `router_L → remap_L → ffn_L → attn_{L+1}` is strictly serial,
leaving no room to overlap. As long as the CPU decides, 40 synchronizations are
unavoidable.

| ID | Verdict |
|---|---|
| AL / AM (MoE virtual memory / paging) | conceptually right but **GPUs have no page faults**, so it cannot be implemented |
| AZ (separate attention/expert pipelines) | serial dependency; they cannot be separated |
| AD / AE (per-layer strategies) | fewer synchronizations but more memory. Estimated 1.1 GiB for 1.4 ms — **not worth it** |

> **Decode at 79% is the ceiling of the current architecture.**

### Category 3: aimed at prefill's structural constraint ★the real target

| ID | Idea | Assessment |
|---|---|---|
| **B** | Expert Sweep prefill | **implementable. The strongest candidate** (below) |
| **A** | prefill/decode separation | **implementable. Simpler than B and more effective** (below) |
| S | token batch reordering | **impossible with a static graph**. `mul_mat_id` already groups by expert internally; the problem is simultaneous residency, not ordering |
| U / AY | dynamic batch control / prefill chunking | exactly what is already being done (the ub limit) |
| J / K | low-precision experts for prefill | valid; shrinks the slab (below) |

### Category 4: changing the model itself (research topics)

| ID | Assessment |
|---|---|
| G / H / I / AI / AJ (distillation / deltas / factorisation) | the real answer for memory, but **requires rebuilding the model**. Violates NG-6 (no requantization) and needs separate quality verification. Beyond v2's scope |
| E / AV (two-stage router / semantic routing) | requires training the model. Not something a runtime can do |
| F (expert clustering) | same |

### Category 5: agent-specific

| ID | Assessment |
|---|---|
| V / W (better prompt caching) | **already in llama.cpp** (context checkpoints). Orthogonal to MoEStream. Effective, but nothing to build |
| AT / AC / AB / AU | applications of the above. Exhaust V first |

---

## The three selected for v2

### Proposal 1: A — separating the prefill and decode paths ★highest priority

**Rationale**: 3.9 of the 4.9x prefill slowdown comes from being forced to
ubatch=8. Lifting the slab constraint for prefill alone should approach the
baseline's 295 tok/s.

**Approach**:

```
decode (n_tokens <= 8) : the current 97-slot slab (device memory, 7.9 GiB)
prefill (n_tokens > 8) : reference host-visible tensors mmapped from the GGUF
```

On UMA, host-visible memory is readable by the device (already demonstrated by
the zero-copy path). Mmapped pages are **page cache** — "soft" memory the kernel
can reclaim.

| | Memory | prefill |
|---|---|---|
| current | 7.9 GiB (fixed) | 59.7 tok/s |
| with A | 7.9 GiB + page cache (reclaimable) | **≈ 250–295 tok/s** |

**Effort**: a change in llama.cpp to hold two expert tensors per layer and choose
between them at graph construction based on n_tokens. Close to the existing
`build_remap` branch. **Difficulty: medium.** Note that depending on page cache
partially revives an approach rejected in §13.2, so the position — "respect the
memory ceiling for decode, but use soft memory during prefill" — needs stating
clearly.

> **Outcome**: rejected. RADV refuses to import file-backed VMAs
> (`VkResult -13`, confirmed in finding S5). The **anonymous staging arena**
> variant was adopted instead (findings S5/S6/S7).

### Proposal 2: B — Expert Sweep prefill (multi-pass mul_mat_id)

**Rationale**: a direct approach that works even where A cannot (discrete GPUs,
no mmap).

**Approach**: split `mul_mat_id` into P passes, mapping out-of-pass experts to
the **zero slot** and summing the results.

```
y = Σ_p  mul_mat_id(slab, x, ids_p)      ids_p: out of pass -> zero_slot -> contributes 0
P = ceil(union(n_tokens) / n_slot)
```

`P` is **determined from n_tokens at graph construction**, so decode
(n_tokens=1) automatically gets P=1 and **decode performance is unaffected**.

| | expected prefill | Memory |
|---|---|---|
| current (ub=8) | 59.7 tok/s | 7.9 GiB |
| B (ub=512, P=3) | **≈ 98 tok/s** | 7.9 GiB |
| B (ub=512, P=2, slots=128) | **≈ 150 tok/s** | 9.7 GiB |

The cost is P times the FFN compute (the zero slots are multiplied too). It is
still faster than a small ubatch because GPU utilisation recovers.

**Effort**: add a pass loop to `build_moe_ffn` and pass the pass index to
`build_remap`. **Difficulty: medium to high.** The zero slot already exists, so
the foundation is there.

> **Outcome**: implemented, and it broke. PPL went to 520801 through graph buffer
> aliasing in ggml. Disabled by default (finding N4).

### Proposal 3: J/K/AQ — a low-precision expert tier for prefill

**Rationale**: written into design doc §18.6 but never verified. Lowering the
bytes per slot **allows more slots in the same memory** → the union constraint
relaxes → a larger ubatch becomes usable.

```
current: IQ3_S/IQ4_NL, 1.42 MiB/expert,  97 slots = 7.9 GiB
tier:    Q2_K-ish,     0.72 MiB/expert, 192 slots = 7.9 GiB   <- ub=32 becomes possible
```

Prefill mainly compresses context and may be less precision-sensitive than
decode (needs verification). The combination would be **low precision for
prefill, current precision for decode**.

**Effort**: a requantization tool equivalent to `moestream pack`, plus per-tier
slabs. **Difficulty: medium.** It requires reversing NG-6 (no requantization),
and **perplexity verification is mandatory.**

> **Outcome**: still unverified. Listed as an open item in `RESULTS.md` §14.

---

## Why the others were not selected

| ID | Reason |
|---|---|
| S (token batch reordering) | `mul_mat_id` already groups by expert internally. The problem is simultaneous residency, not ordering, and a static graph cannot regroup the token set dynamically |
| E (two-stage router) | requires retraining the model. Outside a runtime's scope |
| G/H/I (distillation/deltas/factorisation) | large potential but requires rebuilding the model. **A separate project in scale** |
| AL/AM (virtual memory/paging) | GPUs have no demand paging, so it cannot be implemented |

---

## Novelty and impact

| Proposal | Novelty as OSS or a paper |
|---|---|
| A | low (an implementation technique). But **the practical impact is the largest** |
| B | **medium to high**. "Multi-pass expert sweep under a static graph constraint" is not covered by existing MoE offloading papers (Klotski/ProMoE/MoE-Infinity), which all assume dynamic graphs |
| J/K | medium. There is prior work (MoQE and others), but **varying precision between prefill and decode** is new |

### The more publishable results from this session

The **measurements themselves** may be worth more than any of the three
proposals.

1. **"The bottleneck in MoE offloading is CPU↔GPU synchronization, not I/O"**
   (0.5 ms vs 11.4 ms) — nearly all existing papers assume I/O is the
   constraint, making this refutation genuinely novel
2. **All three predictive prefetch schemes are net-negative** — P2 is slower even
   at 81.4% accuracy
3. **Formalising the "batch union ≤ slot count" constraint** under static graph
   execution

> Which also suggests that what could have Colibri-level impact here is not the
> implementation but **the measurement**.

> **[Note added 2026-08-07]** Point 1 was itself overturned by re-measurement.
> The current code has synchronization at 2.2 ms and I/O at 7.8–12.5 ms
> (`RESULTS.md` §10.12). Ironically, that makes the more durable lesson
> **"re-measure when the implementation changes"** rather than the specific
> ordering.
