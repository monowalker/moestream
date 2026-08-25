# Finding S21 — the dense compute ceiling, measured: S18 was out by 2.1x

| | |
|---|---|
| Target | the one extrapolated number finding S18's prefill argument rests on |
| Date | 2026-08-22 |
| Verdict | **S18 underestimated dense prefill by 2.1x.** The conclusion survives; the break-even ubatch doubles from 105 to 218 |
| Measured on | `research/spikes/s21_dense_baseline/measure.sh`, Qwen3.8-27B-IQ4_NL (15.22 GiB, 65 layers, dense), plain llama.cpp, ngl 99 |

## Why this needed measuring

S18 argued that a dense model streams for free during prefill above ubatch ≈ 105.
Every input to that was measured except one: the effective compute throughput,
which was **extrapolated from Ornith's MoE prefill** at 1.77 TFLOP/s.

Ornith is MoE. Its kernel shapes are not a dense model's. Extrapolating across
that boundary is precisely the move this project keeps catching itself making
(§10.11, §10.14, §13.3), and Qwen3.8-27B is dense, on the disk here, and fits in
23.5 GiB of GTT — so there was no excuse for not measuring it.

## Result

| configuration | device memory | prefill | decode |
|---|---:|---:|---:|
| plain llama.cpp, ub=1024 | 15.79 GiB | **67.3 tok/s** | **211.96 ms/tok** (4.72 tok/s) |
| plain llama.cpp, ub=2048 | 16.15 GiB | 62.8 tok/s | 212.88 ms/tok |
| MOESTREAM=1, ub=1024 | 15.79 GiB | 67.2 tok/s | 212.48 ms/tok |

**Predicted 32.4 tok/s, measured 67.3.** The effective figure is
**3.68 TFLOP/s**, not 1.77 — Ornith's MoE prefill was a poor proxy, and in the
direction that makes streaming look better than it is.

The third row also confirms the README's claim that **MoEStream is inert on a
dense model**: 67.2 vs 67.3 tok/s and 212.48 vs 211.96 ms/tok, i.e. within
noise, with identical device memory. It loads, finds no expert tensors, and gets
out of the way.

## What it does to S18

| | S18 (extrapolated) | measured |
|---|---:|---:|
| compute ceiling | 32.4 tok/s | **67.3 tok/s** |
| compute per token | 30.9 ms | **14.9 ms** |
| I/O per streamed pass | 3.25 s | 3.25 s |
| **break-even ubatch** | **105** | **218** |

**The conclusion holds, the threshold doubles.** Faster compute means less time
to hide the same I/O behind, so a larger batch is needed before streaming is
free. At ubatch 512 or 1024 — the range this project actually runs in — prefill
is still fully hidden. At 128, which S18 claimed was already enough, it is not.

## The decode figure is a useful cross-check

211.96 ms/token for a fully resident dense 27B is not a mystery: 15.22 GiB read
once per token at the ~70 GB/s this machine's DDR5 delivers is 233 ms. Decode is
**memory-bandwidth bound**, and the measurement lands within 10% of the physical
bound. That is a good sign the rest of the numbers are trustworthy.

It also prices dense streaming honestly. Streamed, decode becomes
`3.25 s of I/O + 0.21 s of compute ≈ 3.46 s/token = 0.29 tok/s`, against S18's
estimate of 0.31. **Streaming costs a dense model 16x on decode** — and that
part of S18 was right.

For comparison on the same machine: Ornith-35B MoE decodes at 24 tok/s fully
resident. A dense 27B manages 4.72. That 5x is the whole reason this project is
about MoE.

## Caveats

- Qwen3.8-27B is a hybrid attention/SSM architecture; a plain transformer will
  have different kernel mix and a different ceiling. 3.68 TFLOP/s should be
  treated as "measured on this one dense model", not as a machine constant —
  which is exactly the error being corrected here.
- Prefill numbers on this machine ran ~15% below `RESULTS.md` §12.5 for the
  baseline generally (251 vs 296 tok/s on Ornith) because other containers were
  live. The dense ceiling is therefore, if anything, **understated**, which
  makes the break-even ubatch an over-estimate.
- Nothing dense has yet been run *through* MoEStream's arena. This measures the
  ceiling that streaming would have to hide I/O behind, not streaming itself.
- The script errored on its final line: a `flock` guard was added to it **while
  it was running**, and bash re-reads a script from a byte offset. All three
  configurations had completed and their output is intact, but editing a
  running script is a real hazard and it was avoidable.

> Reproduce: `research/spikes/s21_dense_baseline/measure.sh`
> Related: `S18-dense-streaming.md`, `V3-idea-analysis.md` D
