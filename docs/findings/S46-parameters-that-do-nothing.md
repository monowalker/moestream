# S46 — Twelve llama.cpp settings that change nothing here

*2026-08-24. Two warm-up requests, three measured 300-token generations, median.
Ornith-1.5-35B (experts streamed, `frac=0.40`) and Qwen3.8-27B (FFN streamed),
N_PARALLEL=1, ctx 32768, ubatch 1024, KV q8_0.*

## Why this is written down

Every flag `llama-server` accepts that plausibly touches performance and that
this project does *not* already set was tried. None helped. A list of things that
do not work is worth as much as the list that does — it is the difference between
"we did not think of that" and "we measured that".

## MoE — Ornith-1.5, experts streamed

| | median | samples |
|---|---:|---|
| reference | 49.4 ms/tok | 52.9 / 49.4 / 48.6 |
| `--poll 1` | 48.5 | 49.1 / 48.4 / 48.5 |
| `--poll 50` | 48.6 | 48.7 / 48.6 / 48.6 |
| `--threads-batch 16` | 48.8 | |
| `--no-op-offload` | 48.7 | |
| `--mlock` | 49.1 | |
| `--cache-reuse 256` | 49.4 | |
| MTP `n_max=1` | **51.1** | worse |

`--poll 1` looks like a 1.8% gain and is not one. Every configuration lands
between 48.4 and 49.1; only the *reference* has a slow first sample. The
reference is always measured first, which is exactly the trap
[S43](S43-warm-up.md) is about.

## Dense — Qwen3.8-27B, FFN streamed

| | median | samples |
|---|---:|---|
| reference | **690.2 ms/tok** | 690.2 / 690.1 / 690.4 |
| `--poll 1` | 695.6 | 695.7 / 695.6 / 695.5 |
| `--threads-batch 16` | 695.9 | |
| `--no-op-offload` | 695.3 | |

All three consistently a little worse, and the samples are reproducible to
**0.03%** — which is what a measurement looks like when there is nothing to argue
about.

## Also tried and rejected elsewhere

| | where |
|---|---|
| all five `ngram-*` speculative types, MoE and dense | [S42](S42-speculation-by-model.md) |
| `--spec-draft-type-k/v q8_0` (matching the draft KV to the main one) | [S42](S42-speculation-by-model.md) |
| `--no-spec-draft-backend-sampling` | [S42](S42-speculation-by-model.md) |
| `--cpu-moe`, `--n-cpu-moe` (upstream's own offload) | [S44](S44-upstream-cpu-moe.md) |
| KV at `q4_0`, spending the saving on expert slots | below |

## KV precision against expert slots

A question this project had never asked: at a fixed memory budget, is `q8_0` KV
plus fewer slots better than `q4_0` KV plus more? `.env.example` picks `q8_0` on
quality grounds without ever pricing the alternative.

| Ornith-1.0 | decode | device memory |
|---|---:|---:|
| KV `q8_0`, `frac=0.25` | **63.3 ms/tok** | 7.93 GiB |
| KV `q4_0`, `frac=0.25` | 65.7 ms/tok | 7.78 GiB |
| KV `q4_0`, `frac=0.30` | 57.8 ms/tok | 8.51 GiB |
| KV `q4_0`, `frac=0.40` | 55.8 ms/tok | 9.93 GiB |

At matched memory `q8_0` wins: 7.93 GiB at 63.3 ms against 7.78 GiB at 65.7.
Halving the KV precision frees only **0.15 GiB**, because Ornith is a hybrid —
10 of its 40 layers hold a KV cache at all — and `q4_0` costs 2.4 ms per token
outright.

**The limit of this result:** it says nothing about a model with a large cache.
`gemma-4-31B` holds 1.74 GiB of KV at ctx 32768 against Ornith's 0.40, and a long
context moves the balance further. On such a model the trade is worth re-pricing;
on this one there is nothing to trade.

## What this leaves

The knobs that matter were already the ones with `learn` attached to them:
the slot count, the micro-batch, and now the speculative draft size. Nothing
else llama.cpp exposes moved this workload on this machine.
