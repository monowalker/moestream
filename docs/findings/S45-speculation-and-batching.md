# S45 — Speculation and batching do not add. They collide.

*2026-08-24. Qwen3.8-27B-IQ4_NL, FFN fully streamed, ctx 32768, ubatch 1024, KV
q8_0, two warm-up rounds per stream, median of three 200-token generations.*

## The question

Both techniques work the same way: put more tokens into one forward pass. On a
dense model a pass reads the same weights whatever it carries, so both are worth
a lot — [S37](S37-batch-freezone.md) measured batching, [S42](S42-speculation-by-model.md)
measured speculation.

The launcher enabled speculation whenever the model supported it, whatever
concurrency the user had chosen. That is only correct if the two compose.

## They do not

| | per stream | total throughput | device memory |
|---|---:|---:|---:|
| K=1, no speculation | 687.5 ms/tok | 1.45 tok/s | 8.16 GiB |
| K=1, speculation on | 329.9 ms/tok | **3.03 tok/s (2.08x)** | 9.55 GiB |
| K=4, no speculation | 681.9 ms/tok | **5.87 tok/s (4.05x)** | 8.50 GiB |
| **K=4, speculation on** | 1025.4 ms/tok | **3.90 tok/s** | 12.08 GiB |

**At four concurrent requests, turning speculation on makes the server 34%
slower and costs 3.6 GiB.** Not "adds less than it did" — actively worse than
leaving it off.

The reason is visible in the widths. At K=4 with `n_max=5`, each pass carries
4 × 6 = 24 token positions, and 20 draft passes run per step to produce them. The
"a dense pass is flat with width" result from [S42](S42-speculation-by-model.md)
was measured to width 6; there is no reason to expect it to hold to 24, and the
drafting cost scales with streams × draft depth while the saving does not.

The two are competing for the same resource — room in a pass — and batching gets
it for free while speculation pays a head cost per drafted token.

## What changed

**The launcher no longer enables speculation above two concurrent requests**, and
says why rather than silently omitting it.

**`SPEC_DECODING=learn` now keys its rows on `N_PARALLEL` as well as the
streaming configuration.** Without that, a draft size learned at one request
would be applied at eight, which is precisely the mistake this finding
documents — and the learn loop would have had no way to notice, because it never
re-measures a row it already has.

## The general shape

Anything that pays by widening a pass is competing with everything else that
pays by widening a pass. This project now has three of them — batching,
speculation, and the prefill arena's union reads — and the useful question is not
"does this help?" but "does this help *given what is already in the pass?*"

[S37](S37-batch-freezone.md)'s law needs the same qualification. It says a
technique pays to the extent a pass's read volume does not grow with the tokens
in it. True — but only while the pass has room. Past that, the compute does grow,
and the technique that got there first has taken the benefit.
