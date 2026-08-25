# Finding S31 — MTP loses on MoE and wins on dense, and the reason is not draft quality

> **[Corrected 2026-08-24 — read [S42](S42-speculation-by-model.md) instead.]**
> Two things below do not survive re-measurement.
>
> **The numbers came from 40-token generations**, which is short enough that
> warm-up dominates — and a MoE decode measurement on this machine keeps drifting
> for far longer than a dense one ([S43](S43-warm-up.md)). Properly warmed, the
> streamed MoE case is not a 6% loss and not break-even either: **48.6 ms/token
> without speculation against 51.1 with `n_max=1`**, which is a loss.
>
> **The mechanism given below is half the story.** It says a verification pass
> fetches the union of the experts its tokens want — true, and now measured
> directly: a MoE decode pass costs 3.51x more at width 6 than at width 1, while
> a dense one is flat. What it misses is that the drafting head is itself
> expensive relative to a MoE forward pass (20 ms against 38), so MoE is
> penalised twice. It also cannot explain what this document reported: that the
> loss is *larger* without streaming, which the second penalty does explain.
>
> The practical conclusion has changed too. Speculation is no longer switched off
> for MoE on principle; `SPEC_DECODING=learn` measures it, with "off" as one of
> the candidates.


| | |
|---|---|
| Target | resolving what S28 left open: does MTP help a streaming MoE model? |
| Date | 2026-08-23 |
| Verdict | **it loses, at 88.7% acceptance.** −11% with everything resident, −6.3% with expert streaming. S13's mechanism confirmed, and this time draft quality cannot be blamed |
| Measured on | `research/spikes/s31_moe_mtp/measure.sh`, Ornith-1.5-35B-Q4_K_M, ctx 8192 |

## Correcting S28 first

S28 reported `acceptance = 0.00000 (0 accepted / 3 generated)` on this model and
concluded the measurement had not run. **That conclusion was wrong.** The figure
came from a 40-token measurement whose last acceptance line described a three-token
warm-up request — a sample far too small to mean anything. MTP was working the
whole time.

> Reading a counter without checking the sample size behind it produced a
> confident "inconclusive" where the answer was available. The fix was to
> generate 200 tokens instead of 40.

## Result

| | memory | decode | acceptance |
|---|---:|---:|---:|
| `MOESTREAM=0`, no MTP | 20.02 GiB | **36.6 ms/tok** | — |
| `MOESTREAM=0` + MTP | 20.91 GiB | 40.7 ms/tok (**+11%**) | 0.887 (55/62) |
| `frac=0.40`, no MTP | 10.13 GiB | **48.9 ms/tok** | — |
| `frac=0.40` + MTP | 10.71 GiB | 52.0 ms/tok (**+6.3%**) | 0.887 (55/62) |

**MTP is net-negative both ways, at 88.7% acceptance.**

That acceptance rate is what makes this decisive. S13 rejected speculation on
this project with a draft acceptance of **0.346** — 0.87 tokens accepted of 4 —
which left the obvious objection that the draft model was simply bad. Here
almost nine drafted tokens in ten are accepted and it still loses. **The
mechanism, not the draft, is the problem.**

## The mechanism, and why it predicts both signs

S13 stated it: a verification pass with K tokens references the *union* of the
experts those K tokens want, not `top_k`.

```
union(K) = n_expert x (1 - (1 - top_k/n_expert)^K)

Ornith, 256 experts, top-8:   union(1) = 8.0    union(4) = 30.5    -> 3.8x
```

So a 4-token pass moves 3.8x the expert bytes. At 2.5 accepted tokens per pass
that is ~8.7 expert-fetches per accepted token against 8 — worse, before the MTP
head's own compute. Which is what the table shows.

**A dense pass has no union.** It reads the same weights whatever the token
count, so K tokens per pass divides the bytes per token by the acceptance count.
Measured on Qwen3.8-27B (S29): 696.1 → 329.6 ms, **2.11x faster**, and the
speed-up *grows* with how much is streamed.

| | read volume per pass | MTP |
|---|---|---:|
| dense | independent of token count | **2.11x faster** |
| MoE | grows with token count (union) | **1.06–1.11x slower** |

> **Speculation pays exactly to the extent that a pass's read volume does not
> depend on how many tokens are in it.** Same code, same machine, opposite sign,
> and the sign is predictable in advance from that one property.

Note the streaming case loses *less* than the resident one (6.3% vs 11%). The
expert cache absorbs part of the widened union — the extra experts a 4-token
pass pulls in are often already resident — so streaming is slightly more
tolerant of speculation than having everything in RAM. That is the opposite of
what S13's reasoning alone would suggest and was not predicted.

## Practical consequence

For the stock llama.cpp server on this machine, which runs
`--spec-type draft-mtp --spec-draft-n-max 3` in production: **that setting is
right for its dense model and would be wrong for a MoE one.** Nothing about the
flag says so.

## Caveats

- One MoE model. Ornith-1.0 has no `nextn_predict_layers`, so the model S13 used
  and every documented figure in `RESULTS.md` cannot be tested this way at all.
- `n_max=3` only. S29 found `n_max` matters on dense (394.7 → 291.9 ms from 1 to
  7); the MoE side was not swept, and a larger draft widens the union further, so
  it should get worse rather than better.
- ctx 8192 to let the 20.21 GiB model fit alongside its KV cache with streaming
  off. The documented Ornith figures use different contexts and are not
  comparable to these absolute numbers.

> Reproduce: `research/spikes/s31_moe_mtp/measure.sh`
> Related: `S13-speculative-decoding.md`, `S29-dense-tuning.md`, `S27-dense-streaming-impl.md`
