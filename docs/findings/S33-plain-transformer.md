# Finding S33 — dense streaming on a plain transformer, and the limit of "byte-identical"

| | |
|---|---|
| Target | the one case S27 could not test: a dense model with no SSM or linear-attention layers |
| Date | 2026-08-23 |
| Verdict | **−58% memory, prompt processing unaffected, perplexity identical.** And it is where the byte-identity claim broke |
| Measured on | `research/spikes/s33_plain_transformer/measure.sh`, gemma-4-31B-it-IQ4_NL (60 layers, no `expert_count`, no `ssm.*`, no `nextn`) |

## Why it mattered

Both dense models available until now — Qwen3.5-4B and Qwen3.8-27B — are
attention/SSM hybrids, so S27's −56% was a figure for that family. A plain
transformer answers two things a hybrid cannot: how much of a dense model is FFN
when nothing is spent on SSM state, and whether streaming attention is worth
implementing when every layer's attention goes through `build_attn`.

Finding one took searching. **The Qwen 3.x line is hybrid throughout** — 3.5 and
3.8 use SSM, 3.6 uses Gated DeltaNet. Plain transformers are becoming the
exception rather than the default, which is itself worth recording: S30's
consolation that "a plain transformer would be a different calculation" applies
to a shrinking set of models.

## Result

| | plain llama.cpp | `frac=0.50` | `frac=0.00` |
|---|---:|---:|---:|
| device memory | 18.30 GiB | 13.21 GiB | **7.70 GiB (−58%)** |
| prompt processing | 53.7 tok/s | 54.8 | **55.3 (no loss)** |
| generation | 228.1 ms/tok | 466.4 | 675.0 (2.96x) |

**The FFN share is higher, as predicted**, and the saving follows it:

| | FFN | attn (+SSM) | embd |
|---|---:|---:|---:|
| gemma-4-31B (plain) | **67.7%** | 27.6% | 4.6% |
| Qwen3.8-27B (hybrid) | 59.8% | 29.0% | 10.8% |

**Prompt processing tolerates a smaller ubatch than the hybrid does.** S29 found
Qwen3.8 losing 48% at ubatch 256; here the loss is 16%:

| ubatch | gemma-4-31B | Qwen3.8-27B |
|---:|---:|---:|
| 256 | 45.1 tok/s (−16%) | 39.2 (−48%) |
| 1024 | 55.3 (+3%) | 66.0 (−1%) |
| 2048 | 55.5 | 64.3 |

Plausibly because full attention on every layer makes each token more expensive
to compute, widening the window the reads hide behind — but that is an
explanation offered, not measured.

## Where byte-identity broke

Perplexity is **240.0322 at `frac=1.00` and at `frac=0.00`** — identical to four
decimals, so the weights are correct. (The absolute value is high because
`ppl.txt` suits this model's tokenizer poorly; only the agreement is meaningful,
and the reference is the non-streaming run of the same model.)

Greedy output, three prompts at 120 tokens each:

| prompt | result |
|---|---|
| "Write a detailed technical explanation of the Raft consensus algorithm." | **identical**, 496 chars |
| "The history of the Roman Empire began" | **identical**, 610 chars |
| `def quicksort(arr):` | **diverges at character 79** |

The divergence lands exactly where the model enters a degenerate repetition:

```
ref : ...// This is a bit of trickyy own own own own own...
strm: ...// This is the same as the same as the same as...
```

In that state several tokens sit at nearly equal probability, and the arena
changes memory alignment — which changes how the Vulkan matmul tiles, and so the
floating-point accumulation order. It is the mechanism `RESULTS.md` §8.2 already
documents for the MoE path's +0.24%, surfacing here because a near-tie amplifies
it into a different token.

> **The claim has to be "no accuracy loss", not "byte-identical output".**
> Byte-identity was observed on every earlier test and reported as if it were a
> property. It is not one: it holds when no near-tie occurs, and nothing
> guarantees that. `RESULTS.md` §8.3 makes exactly this distinction — token
> identity is a weaker claim than numerical identity — and it was worth more
> attention than it got.

## What it changes for attention streaming

S30 rejected extending streaming to attention partly because Qwen3.8 has only 17
ordinary-attention layers out of 65, leaving 0.95 GiB recoverable. **gemma-4 has
no SSM at all**: 4.44 GiB of attention across 60 layers, every one of them
through `build_attn`. On this model the ceiling would be roughly 7.70 → 3.5 GiB.

That does not revive the idea on its own — S30's three failed designs are about
mechanism, not about how much is available — but it removes the "not worth the
cycle" half of the argument for this class of model.

## Caveats

- Measured at K=1 without MTP only. The like-for-like matrix (S34) was run on the
  hybrid; the 2.96x here is not directly comparable to S34's cells.
- Perplexity 240 means the corpus and this model are a poor match. The comparison
  is internally valid and the absolute number should not be quoted.

> Reproduce: `research/spikes/s33_plain_transformer/measure.sh`
> Related: `S27-dense-streaming-impl.md`, `S30-dense-attention.md`, `S34-like-for-like.md`
