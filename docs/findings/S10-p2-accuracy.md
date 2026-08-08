# Finding S10 — P2 predictor accuracy, and P5 carrying no information at all

| | |
|---|---|
| Target | verifying the premises for predictive prefetch (between findings S9 and S11) |
| Date | 2026-08-06 |
| Verdict | **P2 = 71.05% (18x random). P5 = 3.97% (indistinguishable from random — no information)** |
| Code | `research/spikes/s10_p2_accuracy/{main.cpp, analyze.py}` |
| Measured on | Laguna-S-2.1 (48 layers / 256 experts / top-10) / 150 tokens / 7050 layer samples |

## What was measured

| Predictor | Definition |
|---|---|
| **P2** | multiply layer L's hidden state `h_L` by layer L+1's router matrix `W_{L+1}` and take the top-k as the prediction for L+1 |
| **P5** | predict that experts chosen at layer L will also be chosen at L+1 (cross-layer co-activation) |

N2 recorded P2 = 81.4% and P5 = 18.5–40.0% on Ornith-35B, but whether that
holds on another model had never been checked.

## Method

**Without patching llama.cpp at all**, capture via
`llama_context_params.cb_eval`:

```
ffn_norm-<L>        ... layer L's router input h_L   [n_embd]
ffn_moe_logits-<L>  ... layer L's router output      [n_expert]
```

The router matrix `blk.<L>.ffn_gate_inp.weight` is stored as F32 in the GGUF, so
the accuracy computation is done in Python (numpy).

### ★ A self-check came first

Before looking at any accuracy figure, confirm the captured tensors are what we
think they are:

```
(1) self-check   W_L·h_L vs logits_L : 100.00%  (70500 samples)
    -> the pipeline is correct
```

A mismatch would mean we captured something else, so the design aborts without
producing any further numbers. It passed at 100%.

### A trap in the measurement conditions

The first collection showed `slot exhaustion events 9228`. At `n_ubatch=512`,
prefill overflowed the slab (38 slots) and **the trace was of a degraded state
where experts had been substituted with zeros**. It was re-collected at
`n_ubatch=1` after confirming `exhaustion 0`.

## Results

```
(2) P2  W_(L+1)·h_L -> layer L+1 : 71.05%   (69000 samples)
(3) P5  layer L's picks -> L+1   :  3.97%
    random equivalent            :  3.91%
```

### P5 was not "insufficiently accurate" — it has no information

**3.97% is indistinguishable from random (3.91% = 10/256).**

N2 recorded P5 as "18.5–40.0%, not accurate enough". On Laguna it carries
**nothing at all**. The hypothesis that the same experts tend to be chosen
across layers simply does not hold for this model.

### P2 does hold (with large variation by layer)

```
L2:30%  L3:59%  L4:52%  L5:54%  L6:64%  L7:72%  L8:66%  L9:68%
L10:73% L11:70% L12:71% L13:68% L14:73% L15:77% L16:73% L17:66%
L18:74% L19:68% L20:74% L21:72% L22:73% L23:74% L24:74% L25:73%
L26:75% L27:73% L28:72% L29:78% L30:79% L31:75% L32:77% L33:75%
L34:74% L35:78% L36:81% L37:79% L38:77% L39:77% L40:81% L41:82%
L42:81% L43:79% L44:74% L45:73% L46:64% L47:46%
```

**The middle and later layers (L7–L45) are a stable 65–82%**; only the ends are
poor:

- **L2 = 30%** — the preceding layer 0 is dense, so the hidden state has
  different properties
- **L47 = 46%** — the final layer is close to the output, where the residual
  changes more

The design document's §22.6 hypothesis that the residual stream changes only
gradually across layers is supported on Laguna too, **excluding the ends**.

## Compared with Ornith

| | Ornith-35B (N2) | Laguna-S-2.1 (here) |
|---|---:|---:|
| experts / top_k | 256 / 8 | 256 / 10 |
| **P2** | **81.4%** | **71.05%** |
| **P5** | 18.5–40.0% | **3.97% (no information)** |

P2 is 10 points lower but the mechanism holds on both models. P5 can be not
merely weak but absent, depending on the model.

## How this result was used

This was measured as a premise for implementing prefetch. Folding in 71%
accuracy gave an expectation of 1.8x; **the implementation came out 19% worse**
(finding S11). Accuracy was not the problem — the **means** of prefetching was.

> Measuring accuracy was the right order of operations. Had it been 20% rather
> than 71%, no implementation would have been needed. It was sufficient, so we
> proceeded — and hit a different wall there.
