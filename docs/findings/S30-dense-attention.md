# Finding S30 — why streaming attention and SSM needs a different design, not a wider filter

| | |
|---|---|
| Target | extending S27's dense streaming from the FFN to every large per-layer weight |
| Date | 2026-08-23 |
| Verdict | **abandoned.** Stubbing cannot survive code that reads a weight's shape. On this model the recoverable remainder is ~0.95 GiB, which does not pay for the redesign |
| Cost | six build/debug cycles across three designs, no shipped change |

## The idea, and why it looked right

S27 streams the FFN only — 8.96 of Qwen3.8-27B's 15.22 GiB. The obvious next
step is the other 4.3 GiB of per-layer weight matrices.

Grepping the architecture showed something better than a per-callsite patch:

```
qwen35.cpp:237   build_lora_mm(wqkv,    input)     attention
qwen35.cpp:270   build_lora_mm(wq,      cur)       attention
qwen35.cpp:464   build_lora_mm(ssm_out, final)     SSM
llama-graph.cpp  build_lora_mm(up/gate/down, cur)  FFN
```

**Every per-layer weight matrix reaches `ggml_mul_mat` through one function.**
One anchor there would cover FFN, attention and SSM with no architecture-specific
code, preserving ADR-0036. The claim rule was widened to "any 2D `blk.N.*.weight`
over 8 MiB that is not an expert tensor", keyed by tensor pointer.

## What actually happened

**Failure 1 — the ggml context was sized for three weights per layer.**
A leftover from the FFN-only version. `GGML_ASSERT(obj_new)` during load. Fixed
by counting the real number of registered weights.

**Failure 2 — the one that ends the idea.**

```
GGML_ASSERT(a->ne[d] == b->ne[d]) failed
  ggml_concat <- build_conv_state <- build_layer_attn_linear
```

A streamed weight is allocated as a one-row stub and substituted at graph build.
That is sufficient **only if the weight is exclusively multiplied by**. Qwen3.5's
SSM path reads `attn_qkv`'s *shape* to size the conv state, and sees the stub.

> **Substituting at the multiply covers every use of a weight's data, and no use
> of its shape.** There is no filter that fixes this, because whether a given
> weight's shape is read somewhere is not a property of its name or size.

**Failure 3 — the wider hook broke the case that worked, and why.**
With the claim narrowed back to the FFN, `build_lora_mm` still failed with
`ggml_can_mul_mat` while the identical claim set through the old `build_ffn`
anchor worked. A later diagnostic build settled it, and there are two distinct
problems, neither of which is the anchor itself:

**(a) The pointer-indexed lookup returned the wrong weight.** The failing version
keyed `g_dn_bypointer[stub] = {layer, index-within-layer}` and read
`g_dn_view[layer].w[index].t`. Rebuilt with a lookup by name and role instead —
same anchor, same claim set — it does not crash at all, and every one of 32
layers substitutes with correct shapes. Handing `ffn_up`'s multiply the tensor
belonging to `ffn_down` produces exactly `ggml_can_mul_mat` on `[9216,2560]`
against `[2560, n]`.

**(b) `build_lora_mm` cannot order the load, even when it substitutes correctly.**
`cur` is a by-value parameter, so wrapping it inside `build_lora_mm` never
reaches the caller. `build_ffn` calls it three times with *its own* `cur`, and
only the first multiply ends up depending on the load op. With the load op
enabled the diagnostic boots and still produces garbage
(`' the capital of the capital of…'`), which is that missing edge.

> **`build_ffn` was the right anchor for a reason that only became visible from
> the inside: it is the level at which `cur` can be rebound for all three
> weights at once.** A choke point that sees every weight is not automatically a
> choke point where ordering can be expressed.

## What it would be worth here anyway

| tensor | layers | size | streamable? |
|---|---:|---:|---|
| `attn_qkv` | 48 | 1.61 GiB | no — SSM reads its shape |
| `ssm_out` | 48 | 0.97 GiB | no |
| `attn_gate` | 48 | 0.79 GiB | no |
| `attn_q/k/v/output` | 17 | **0.95 GiB** | probably |

Qwen3.8-27B is **48 SSM layers to 17 attention layers**. The safe remainder is
0.95 GiB: 6.90 → ~6.0 GiB, moving −56% to −61%. That is not worth an
implementation cycle.

**A plain transformer would be a different calculation** — attention is roughly
30% of such a model and none of it goes through the SSM path. Both dense models
available here are hybrids, so this is untested.

## The design proposed here was then tried, and it does not work either

This section originally said: *do not stub — allocate the model's tensors at
their real shape but bind them to arena memory, so any code reading the shape
sees the truth.* That was built (`ggml_backend_tensor_alloc` into arena buffers
during `create_tensor`, the graph keeping llama.cpp's own tensors, one load op
per layer for ordering). **It fails twice, and the second failure invalidates
the premise.**

| | result |
|---|---|
| Qwen3.8-27B `frac=0.00` | boots, streams 13.12 GiB, decode 1125 ms — output `'\xa0\xa0 \xa0 ，， ， ，'` |
| Qwen3.8-27B `frac=0.50` | boots — output `'                '` |
| Qwen3.5-4B `frac=0.00` | boots — output `' vibr vibr vibrasanasanasan…'` |

**1. It corrupts silently.** No crash, no `[BUG]` line, reads demonstrably
happening (decode 211 → 1125 ms). Exactly the hazard predicted before building
it: with the shape correct, a wrong weight is indistinguishable from a right one
until you read the text. Layers *L* and *L+2* share a buffer, and the single
ordering edge per layer is evidently not sufficient.

**2. The memory does not come back, which ends the idea.**

```
plain llama.cpp        15.58 GiB
real-shape binding     14.50 GiB     <- streaming 13.12 GiB, saving 1.08
```

The reason is **not** that llama.cpp refuses to skip pre-placed tensors. Reading
`ggml_backend_alloc_ctx_tensors_from_buft_impl` afterwards shows it plainly does:

```c
if (t->data == NULL && t->view_src == NULL) {   // a bound tensor contributes 0
    this_size = GGML_PAD(ggml_backend_buft_get_alloc_size(buft, t), alignment);
}
```

13.12 GiB was bound successfully — the runtime's own accounting says so — and by
that code the model buffer should have shrunk by the same amount. It did not,
and **why is not known.** Something else is still holding the memory, and the
candidates (a second allocation path in the loader, buffer sizing taken from the
GGUF rather than the context, the measurement counting a buffer twice) were not
narrowed down.

> **An earlier revision of this section stated "llama.cpp's allocator does not
> skip tensors the runtime pre-allocated". That was inferred from the symptom
> and is wrong** — the skip is right there in the source. The honest statement is
> that the memory did not come back and the cause is unidentified.

### Where the 13 GiB went: nowhere, because it was never measured

The mmap route in `llama-model.cpp:1566` builds the model buffer from a
**file-offset span** `[first, last)`, which cannot shrink when tensors are
pre-allocated. That looked like the answer. It was tested, and it is not:

| | memory |
|---|---:|
| binding design, mmap (default) | **3.10 GiB** |
| binding design, `--no-mmap` | 3.59 GiB |
| stub design, same model | 2.10 GiB |

`--no-mmap` makes it *worse*, so the mmap span is not what held the memory. And
3.10 GiB is not 14.50 GiB.

**The 14.50 GiB figure came from Qwen3.8-27B and the 2.10 GiB reference from
Qwen3.5-4B.** They are different models, five times apart in size, and the
"13 GiB went missing" conclusion was drawn by putting them side by side. On the
same model the binding design uses 3.10 GiB against the stub design's 2.10 —
**1 GiB more, not 13.**

> **The premise was an artefact of an unmatched comparison, and so was the
> hypothesis built on top of it.** This is the same failure as the three
> comparison errors recorded in S34: a number from one configuration set against
> a number from another. It produced two paragraphs of confident mechanism —
> first blaming llama.cpp's allocator, then blaming its mmap path — for an
> effect that was 1 GiB and had nothing to do with either.

What remains true about the binding design is the part that was measured
directly: **it corrupts output silently** (garbage text at full speed, no crash,
no `[BUG]` line), and it costs about 1 GiB more than stubbing on the same model.
Those are reasons enough not to ship it. The memory question that dominated this
section did not exist.


> Related: `S27-dense-streaming-impl.md`, `S18-dense-streaming.md`, ADR-0036
