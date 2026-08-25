# S42 — Which speculative decoding a model can do, and whether it is worth doing

*2026-08-23. Qwen3.8-27B-IQ4_NL and gemma-4-31B-it-IQ4_NL, one request at a
time, ctx 32768, ubatch 1024, 200-token generations. Every cell has its baseline
measured in the same configuration.*

## The question

[S29](S29-dense-tuning.md) found speculative decoding worth about 2x on a streamed dense
model, and [S31](S31-moe-mtp.md)/[S32](S32-moe-mtp-nmax.md) found it *costs* a
streamed MoE model. The launcher was not using either result, which left
the worst case this project has — a dense model, one request at a time, 3x
slower — paying full price with the best available remedy switched off.

Two things had to be settled before switching it on:

1. **Which models can self-speculate**, and how to know without guessing.
2. **Whether the model-free kinds help**, since only a minority of GGUFs carry a
   speculation head. If `ngram-*` paid, every model would benefit.

## 1. What a model supports is llama.cpp's question

llama.cpp has eleven speculative types. `draft-simple`, `draft-eagle3`,
`draft-mtp`, `draft-dflash` and `draft-dspark` all need something from the
model; the five `ngram-*` variants need nothing at all.

The first implementation here read `<arch>.nextn_predict_layers` from the GGUF
metadata. **That is the wrong test.** Upstream's own rule, in
`common_speculative_types_from_gguf()`, is the presence of the tensor
`blk.<block_count-1>.nextn.eh_proj.weight`. The difference is not academic — of
the nine GGUFs on this machine:

| model | `nextn_predict_layers` | `eh_proj` tensor | llama.cpp says |
|---|---|---|---|
| `Ornith-1.5-35B-Q4_K_M` | 1 | yes | `draft-mtp` |
| `Qwen3.8-27B-IQ4_NL` | 1 | yes | `draft-mtp` |
| **`Qwen3.8-27B-Q4_0_ROCMFP4_STRIX`** | **0** | **no** | — |
| `gemma-4-31B`, `Ornith-1.0`, `Qwen3.5-4B`, `Qwen3-Coder-Next`, `Laguna-S-2.1`, `gpt-oss-120b` | 0 | no | — |

The same model at a different quantization loses the head. A rule copied into
the launcher would have to be kept in step with upstream by hand, and would
drift silently the first time upstream added a type.

So it is not copied. `src/spec_probe.cpp` links llama.cpp's own `common` library
and calls `common_speculative_types_from_gguf()` directly. Thirty lines, no rule
of ours, and it follows upstream on the next build.

## 2. Is it worth doing? It depends on one thing, and it is not the model family

The first version of this finding said MoE models lose because a verification
pass fetches the *union* of the experts its tokens want. That explanation is
wrong, and the measurement that shows it was already on the page: the loss is
**larger on plain llama.cpp than with streaming**, and an expert-union cost paid
at the SSD cannot behave that way.

Measuring properly says something simpler.

### The arguments dominate, and llama.cpp's defaults are the worst case

Ornith-1.5-35B, plain llama.cpp, no streaming. Reference: **36.9 ms/token**.

| | decode | vs reference |
|---|---:|---:|
| `--spec-type draft-mtp`, everything else default | 60.8 ms/tok | **+65%** |
| defaults + draft KV forced to `q8_0` | 60.7 ms/tok | +64% |
| defaults + `--no-spec-draft-backend-sampling` | 61.5 ms/tok | +67% |
| `--spec-draft-n-max 3 --spec-draft-p-min 0.7` | 40.8 ms/tok | +11% |
| **`--spec-draft-n-max 1`** | **37.0 ms/tok** | **break-even** |

`p-min` is the single biggest lever: 0.0 → 0.7 takes 60.8 ms to 40.8. llama.cpp
defaults it to 0.0, so a bare `--spec-type draft-mtp` drafts regardless of how
unsure the head is, and every rejected draft is paid for.

The draft context's KV type was the first suspect — llama.cpp fixes it at F16
regardless of the main setting (`speculative.cpp:2340`), so a model running
`q8_0` gets a second cache at twice the precision. Forcing it to `q8_0` changed
nothing (60.7 against 60.8). Reading the source made it look like the answer;
measuring said it was not.

### How many tokens to draft, per configuration

`--spec-draft-p-min 0.7` throughout, streamed, 300-token generations:

| | dense Qwen3.8-27B (FFN streamed) | MoE Ornith-1.5 (frac 0.40) |
|---|---:|---:|
| no speculation | 679.5 ms/tok | **48.6 ms/tok** |
| `n_max=1` | 521.2 (1.30x) | **51.1 (0.95x — worse)** |
| `n_max=3` | 308.3 (2.20x) | — |
| **`n_max=5`** | **296.5 (2.29x)** | — |
| `n_max=8` | 388.4 (1.75x) | — |

**The dense optimum is interior** — 5 is best and 8 is worse than 3, so "larger is
better" is not a rule that can be hardcoded.

The MoE column needed measuring twice. A first pass with one short warm-up put
`n_max=1` slightly *ahead* of no speculation (52.3 against 54.3) and this document
said so. With two warm-up requests the baseline settles at 48.6 and `n_max=1`
measures 51.1 — **speculation loses on MoE here, streamed as well as resident.**
The earlier reading was inside the noise: MoE decode samples on this machine
spread by about 8% until fully warm, where dense samples spread by 0.03%.

### What actually decides it — measured, not reasoned

Two explanations were written into this project before this was measured, and
**both were half right**, which is why neither survived contact with the data on
its own.

Speculation is a trade: guess several tokens cheaply, then check them all in one
pass. It pays when **checking many at once costs little more than checking one**,
and when **guessing is cheap relative to checking**. Those two are measurable
separately, and neither needs speculation to measure.

**How much does a wider pass cost?** Run K concurrent requests and one decode
pass advances all K streams by a token, so per-stream ms/token *is* the pass
time. No drafting involved:

| pass width K | MoE resident | MoE streamed | dense streamed |
|---:|---:|---:|---:|
| 1 | 36.6 ms | 51.3 ms | 697.6 ms |
| 2 | 55.0 (1.50x) | 73.2 (1.43x) | 830.2 |
| 4 | 90.3 (2.47x) | 117.1 (2.28x) | 687.2 |
| 6 | 130.3 (**3.56x**) | 168.4 (3.28x) | 764.8 (**±10%, no trend**) |

**A dense pass costs the same at width 6 as at width 1.** It reads the whole
feed-forward block either way. A MoE pass does not: four tokens want four
different sets of experts, so the pass touches most of the union of them — 3.51x
by width 6. (The idealised union ratio is 5.53x; the measured 3.51x is lower
because some of the pass is width-independent.) This is the first direct
measurement of the law [S37](S37-batch-freezone.md) inferred from batch scaling.

**How much does guessing cost?** Back it out of the same numbers. On MoE
resident at `n_max=3` and 0.90 acceptance, a step is one width-4 pass (91.4 ms)
plus three drafts, and produces 3.7 tokens at a measured 40.8 ms each — so the
three drafts cost 60 ms, **20 ms each against a 38 ms full pass**. Checked
against `n_max=1`: a width-2 pass (55.2) plus one draft (20) over 1.9 tokens
predicts 39.6 ms/token; measured 37.0.

On dense streamed at `n_max=5` and 0.57 acceptance: a width-6 pass (751 ms, flat)
plus five drafts, 3.85 tokens at 296.5 ms each — **78 ms per draft against a 681
ms pass**.

So:

| | wider pass costs | one draft costs | outcome |
|---|---|---|---|
| **dense** | **nothing extra** | **11%** of a pass | both favourable → **2.29x** |
| **MoE** | **2.4x at width 4** | **53%** of a pass | both unfavourable → loses |

MoE is penalised twice, and neither penalty is "MoE is slow" or "MoE is fast".
The pass gets more expensive as it widens *because* only `top_k` experts run per
token — the same property that makes a MoE forward pass cheap in the first place,
which is also what makes a fixed drafting cost large beside it.

This resolves the contradiction with published reports of MTP winning on MoE
models. Those run much larger models on much larger GPUs, where a single decode
pass is far more expensive — so the draft cost shrinks against it, and the wider
pass has compute to spare. Same arithmetic, other end of the same line. **Which
side you land on is a property of the machine and the model together**, which is
why nothing here is hardcoded.

### The model-free kinds still do not help

`ngram-*` needs nothing from the model, so it looked like the way to give every
GGUF the benefit:

| | generation |
|---|---:|
| Qwen3.8-27B streamed | 713.8 ms/tok |
| Qwen3.8-27B streamed + `ngram-cache` | 713.5 ms/tok |
| gemma-4-31B streamed | 704.8 ms/tok |
| gemma-4-31B streamed + `ngram-cache` | 697.2 ms/tok |
| gemma-4-31B **plain** | 236.0 ms/tok |
| gemma-4-31B **plain + `ngram-cache`** | **274.3 ms/tok** |

Nothing when streamed, 16% worse when not. A drafted token only pays if it is
accepted, and an n-gram guess taken from recent text is not accepted often enough
on prose the model has not yet repeated.

**It was worth re-asking for MoE, and the answer is the same.** A near-free draft
looked like it might exploit a wide pass that a costly draft could not, and a
first measurement showed `ngram-cache` 24% ahead on Ornith-1.5. It did not
survive: with two warm-up requests instead of one,

| Ornith-1.5 | no speculation | + `ngram-cache` |
|---|---:|---:|
| streamed, pass 1 | 49.9 ms/tok | 49.5 |
| streamed, pass 2 | 49.0 ms/tok | 49.0 |
| resident | 36.7 ms/tok | 37.1 |

The 24% was the baseline's first sample being slow, dragging its cumulative
throughput down. All five `ngram-*` variants were tried; none helps either family.

## What was changed

Nothing about this is hardcoded, because no value here survives being moved to
another configuration. `SPEC_DECODING=learn` (the launcher's default when the
probe reports a type) tries one draft size per start from `off 1 2 3 5` and keeps
the fastest, keyed by model *and* streaming configuration.

The number it learns from is llama.cpp's own
`llamacpp:predicted_tokens_seconds`, read from the server's metrics endpoint.
That matters: the runtime here sees batches, not acceptances, so anything it
measured itself would count rejected draft tokens as work done and would rank a
wasteful setting as a fast one. **`off` is one of the candidates**, so "do not
speculate" can win on its own merits — which, on a fast MoE model, it does.
