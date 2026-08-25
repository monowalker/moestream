# Finding S18 — "Dense models get no benefit" is wrong; they get the largest memory reduction and pay for it entirely in decode

| | |
|---|---|
| Target | DESIGN.md NG-4, README "MoEStream is inert on dense models — it will run them, with no benefit" |
| Date | 2026-08-22 |
| Verdict | **the claim is false as written.** Dense streams to **−86%** memory, and prefill is free above ubatch ≈ 105. Only decode fails, and it fails badly |
| Measured on | `research/spikes/s18_dense_stream/analyze.py` against `Qwen3.8-27B-IQ4_NL.gguf` (15.22 GiB, 65 layers, dense) |

## The claim on record

> NG-4 — **optimizing dense models**: for dense, `B_act = total size` and
> streaming cannot work in principle. Compatibility only.

`B_act = total size` is true **per forward pass**. It is not true per token.
A prefill pass puts `U` tokens through one set of weights, so the streamed bytes
per token are `total / U`. The whole argument rests on silently reading "pass"
as "token", and that only holds at `U = 1` — decode.

## The byte budget of a real dense model

| | | share |
|---|---:|---:|
| FFN (gate / up / down) | 9.10 GiB | 59.8% |
| attention + SSM | 4.41 GiB | 29.0% |
| embeddings / output | 1.64 GiB | 10.8% |
| per layer | mean 213.7 MiB, max 253.5 MiB | |

Streaming the body through the **existing** prefill-arena scheme — N arenas of
one layer each, exactly what `MOESTREAM_PREFILL_ARENA` already does — gives:

| arenas | resident | vs 15.22 GiB |
|---:|---:|---:|
| 1 | 1.88 GiB | **−88%** |
| **2** (async prefetch, the shipped default) | **2.13 GiB** | **−86%** |
| 4 | 2.63 GiB | −83% |

**−86% is a larger reduction than anything measured on an MoE model in this
project** (gpt-oss-120b, the best case, is −75%). It is larger for the obvious
reason: a dense model has no resident expert slab to keep, only one layer in
flight.

## Prefill is free

> **[Superseded 2026-08-22 by finding S21 — measure, do not extrapolate.]**
> The compute ceiling below was extrapolated from Ornith's **MoE** prefill at
> 1.77 TFLOP/s. Qwen3.8-27B has since been run on this machine, and the real
> figure is **3.68 TFLOP/s — 2.1x higher**, giving a measured ceiling of
> **67.3 tok/s**, not 32.4.
>
> **The conclusion survives; the threshold doubles.** Faster compute leaves less
> time to hide the same 3.25 s of I/O behind, so break-even moves from
> **ubatch 105 to ubatch 218**. Prefill is still free at ubatch 512 and 1024 —
> the range this project runs in — but the claim that 128 suffices was wrong.
>
> Measured decode for the same model, fully resident, is **211.96 ms/token
> (4.72 tok/s)**, against the ~233 ms that 15.22 GiB at this machine's ~70 GB/s
> of DDR5 predicts: decode is memory-bandwidth bound and lands within 10% of the
> physical bound. Streamed it becomes ~3.46 s/token = **0.29 tok/s**, against
> the 0.31 estimated below — so the decode half of this document held up.
>
> Details: `S21-dense-ceiling.md`. The table below is left as written, with the
> corrected numbers alongside.

Empirical constants, both derived from measurements already in `RESULTS.md`
rather than guessed: effective prefill throughput of the Radeon 780M is
**1.77 TFLOP/s** (Ornith 295.6 tok/s at 3 B active params) — **measured at 3.68
on a dense model, see above** — and saturated SSD bandwidth is
**4.48 GB/s** (§4.1).

At 27.32 B params the compute ceiling is 32.4 tok/s (30.9 ms/token). One
streamed pass of the 13.57 GiB body costs 3.25 s of I/O, which amortises:

| ubatch | I/O ms/token | serial tok/s | overlapped tok/s | % of ceiling |
|---:|---:|---:|---:|---:|
| 64 | 50.81 | 12.2 | 19.7 | 61% |
| **128** | 25.41 | 17.8 | **32.4** | **100%** |
| 512 | 6.35 | 26.9 | 32.4 | 100% |
| 1024 | 3.18 | 29.4 | 32.4 | 100% |

**Break-even is ubatch ≥ 105** as extrapolated, **≥ 218 as measured** (S21).
Above it, the async arena (S14) hides the I/O completely and streamed prefill
runs at the compute ceiling — the same speed as having the whole model resident,
in 2.13 GiB instead of 15.22 GiB.

S21 also confirms the README's claim that **MoEStream is inert on a dense
model**: with `MOESTREAM=1` the same model measured 67.2 tok/s prefill and
212.48 ms/token decode at identical device memory. It loads, finds no expert
tensors, and gets out of the way.

This is the *best* case for the arena, not a marginal one. Everything that made
the arena work on MoE prefill (§10.9 — "reading all of them wastes nothing, and
prefetching needs no prediction") is **exactly true by construction** for dense:
there is no union to estimate and no expert that goes unused.

## Decode is where it genuinely fails

| | bytes/token | I/O | rate |
|---|---:|---:|---:|
| everything streamed | 13.57 GiB | 3252 ms | **0.31 tok/s** |
| attention resident, FFN streamed | 9.10 GiB | 2182 ms | 0.46 tok/s |

NG-4's conclusion is correct **for decode**, and by a wide margin. It is the
reasoning that is wrong, and the reasoning is what got generalised into
"no benefit".

## The comparison that actually matters on UMA

The obvious objection is that llama.cpp already handles an oversized dense model
with `--n-gpu-layers`, keeping the rest on the CPU — and CPU inference reads from
DDR5 at ~70 GB/s, 15x faster than the SSD. On a discrete-GPU machine that is the
right answer and streaming is pointless.

**On unified memory it is not an answer at all.** CPU and GPU share the same
DDR5. Moving a layer off the GPU does not reduce the machine's memory use by one
byte; it only changes which engine reads it. On a UMA box, SSD streaming is the
**only** mechanism that reduces total footprint — which makes the dense case
*stronger* here than the docs allow, not weaker.

## Three positions worth writing down

**D1 — prefill-bound dense workloads. Real today, no research needed.**
Reranking, embedding, classification, scoring, perplexity evaluation, and
long-prompt/short-answer summarisation are all prefill-dominated. A dense 27B
would serve them at 100% of the compute ceiling in 2.13 GiB.

> **[Corrected 2026-08-22] "No new mechanism" was wrong.** Reading the runtime
> rather than reasoning about it turns up a structural obstacle this document
> originally missed.
>
> The slab works by **resizing `ne[2]`** — the expert dimension — from
> `n_expert` down to the slot count, which is what lets `mul_mat_id` run
> unchanged against a smaller array. A dense FFN tensor is **2D**
> (`[n_embd, n_ff]`, `ne[2] = 1`) and is consumed by plain `ggml_mul_mat`,
> not `mul_mat_id`. There is no expert dimension to shrink, so **the slab path
> does not apply to dense at all** — only the arena path does.
>
> That makes the change larger than "a `parse_name` case":
>
> | | what is needed |
> |---|---|
> | `parse_name` | recognise `blk.N.ffn_{gate,up,down}.weight`, `n_expert = 1` |
> | model loader | allocate the model's own FFN tensors as **stubs**, not at slot-pool size — otherwise the full model is still resident and the saving is zero |
> | arena | size to one layer's FFN (253 MiB here) rather than one layer's expert set (498 MiB) |
> | graph patch | a **new anchor in `build_ffn`**, not `build_moe_ffn`; substitute the arena-backed tensors and insert a load op ahead of them |
> | slab / remap | must be disabled for dense, not merely unused |
>
> Roughly 200–300 lines plus a fifth patch block, against the 5 blocks the
> project currently ships. Still ordinary engineering with no research in it,
> but not the afternoon the original wording implied.

**D1b — partial residency: it is a continuum, not all-or-nothing.**
Keep K of the 65 layers resident and stream the rest. Memory and speed then move
together exactly the way `MOESTREAM_CACHE_FRAC` moves them for MoE
(`research/spikes/s18_dense_stream/partial.py`, using S21's measured 211.96 ms
of compute and S19's two read regimes):

| resident | streamed | memory | reads | serial | **overlapped** |
|---:|---:|---:|---:|---:|---:|
| 65 | 0 | 15.22 GiB | — | 212 ms | **4.72 tok/s** |
| 56 | 9 | 13.76 GiB | 217 ms | 429 ms | **4.61 tok/s** |
| 48 | 17 | 12.09 GiB | 410 ms | 622 ms | 2.44 tok/s |
| 32 | 33 | 8.75 GiB | 795 ms | 1007 ms | 1.26 tok/s |
| 0 | 65 | 2.07 GiB | 1567 ms | 1779 ms | 0.64 tok/s |

(reads at the 9.3 GB/s page-cache rate S19 measured; at the 4.48 GB/s device
rate every streamed row roughly halves.)

**Which layers you stream does not change the bytes** — a dense model uses every
layer exactly once, unlike MoE where layer 0 misses on 52% of lookups and the
rest on 11–19% (S20). But it changes how much can be **hidden**: stream the
*tail*, and the entire forward pass through the resident head is time the reads
can happen in.

> **This is the one place dense beats MoE outright.** Every predictive prefetch
> scheme in this project failed (N2, S10, S11, §10.14) for the same reason: you
> cannot know which experts layer L+1 wants until layer L has run. **For a dense
> model there is nothing to predict — layer 33 is always layer 33.** Perfect
> prefetch, zero prediction cost, no accuracy risk.

That creates a genuinely free zone. Reads are fully hidden while
`streamed ≤ compute × bandwidth`:

```
page-cache served (9.3 GB/s) : 1.84 GiB = 9 layers  -> 1.42 GiB saved for free
device served    (4.48 GB/s) : 0.88 GiB = 4 layers  -> 0.47 GiB saved for free
```

So a few GiB comes off at **no speed cost at all**, and beyond that the price is
linear and steep. Note the arena itself costs 0.42 GiB, which is why the free
saving is smaller than the free streaming budget.

These rows are arithmetic over two measured constants, not an end-to-end run —
nothing dense has been executed through the arena yet.

**D2 — dense models that do not fit at all, at ~0.3 tok/s.**
The same trade the README already sells for gpt-oss-120b at 3.9 tok/s, at a much
worse point on the curve. Whether 0.3 tok/s is worth having is the user's call,
not the design document's — but it should be stated as a number, not as
"cannot work in principle".

**D3 — activation sparsity. The research lever, and structurally a good fit.**
With attention resident (6.54 GiB total footprint) and only the *active* FFN
rows streamed, the density the FFN would have to reach is:

| target | density needed @4.48 GB/s | @10 GB/s (page-cache served) |
|---:|---:|---:|
| 1 tok/s | 46% | no sparsity needed |
| 3 tok/s | 15% | 34% |
| 5 tok/s | 9% | 20% |

Published SwiGLU sparsification brackets that range — CATS reports roughly 50%
density with little loss, and the relufied families (ProSparse, TurboSparse,
PowerInfer) reach 10–20%. So 1–3 tok/s is not obviously out of reach.

Structurally **MoEStream is already the runtime for this**: slice `ffn_gate` and
`ffn_up` column-wise and `ffn_down` row-wise into G groups, and each group *is*
an expert. The slot table, the id remap and `mul_mat_id` all apply unchanged —
the same trick, on a different axis.

The blocker is the predictor, and it is the same wall §3 of HOW-IT-WORKS
describes: you cannot know which rows are active without computing the gate, and
the gate is the thing you are streaming. The literature's answer is a small
offline-trained predictor per model (Deja Vu / PowerInfer). That is genuine
research, it reverses "no architecture-specific work", and it is the only item
in this document that is not straightforward engineering.

Prior art this project's docs do not currently cite: **"LLM in a flash"
(Apple, 2023)** is exactly dense-model flash streaming via predicted FFN
sparsity, and is more relevant to D3 than Klotski / ProMoE / MoE-Infinity are.

## Cheapest next step for D3

No training and no runtime change: capture FFN intermediate activations over a
corpus from a dense GGUF, and measure realised density against magnitude
threshold, plus the PPL cost of thresholding. Purely offline, and it decides
go/no-go before any predictor work starts.

## Caveats

- Every figure here is **arithmetic from the tensor index plus two measured
  constants**, not an end-to-end run. Nothing dense has actually been executed
  under MoEStream. The prefill claim in particular assumes the async arena
  behaves on a 253 MiB layer as it does on a 498 MiB expert set; that is likely
  but unverified.
- The compute ceiling is extrapolated from Ornith's MoE prefill. A dense model
  has different kernel shapes and the real number will differ.
- `Qwen3.8-27B` is a hybrid attention/SSM architecture, so the 29% attention
  share is not representative of a plain transformer, where it would be lower
  and the FFN share correspondingly higher — which makes D1 and D3 slightly
  better, not worse.

> Reproduce: `python3 research/spikes/s18_dense_stream/analyze.py <dense.gguf>`.
> Related: `S7-prefill-arena-impl.md`, `S14-async-arena-prefetch.md`,
> `RESULTS.md` §10.9, DESIGN.md §5.1 NG-4.
