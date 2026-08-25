# Finding S32 — a larger MTP draft makes MoE worse, and dense better

> **[Corrected 2026-08-24 — read [S42](S42-speculation-by-model.md) instead.]**
> The direction below is right — a larger draft does make MoE worse — but the
> explanation is only half of it, and the arguments used were not llama.cpp's
> best. Everything here ran with `--spec-draft-p-min` left at its default of 0.0,
> which drafts regardless of how unsure the head is; setting it to 0.7 moves the
> same model from +65% to +11%. And a dense model turns out to have an *interior*
> optimum (`n_max=5` beats both 3 and 8), so "larger is worse for MoE, better for
> dense" is too simple in both directions.


| | |
|---|---|
| Target | the half S31 left open — does raising `--spec-draft-n-max` rescue MTP on MoE? |
| Date | 2026-08-23 |
| Verdict | **no. Monotonically worse, and memory grows with it.** The mirror image of dense |
| Measured on | `research/spikes/s32_moe_nmax/measure.sh`, Ornith-1.5-35B-Q4_K_M, `frac=0.40`, ctx 8192 |

## Result

| | memory | decode | acceptance |
|---|---:|---:|---:|
| no MTP | 10.13 GiB | **51.6 ms/tok** | — |
| `n_max=1` | 10.58 GiB | 51.0 | 0.917 |
| `n_max=3` | 10.71 GiB | 52.9 | 0.887 |
| `n_max=5` | 10.83 GiB | 54.4 | 0.887 |
| `n_max=7` | 10.95 GiB | 54.2 | 0.887 |

Against dense, from S29, on the same knob:

| `n_max` | dense (`frac=0.00`) | MoE (`frac=0.40`) |
|---:|---:|---:|
| 1 | 394.7 ms | 51.0 ms |
| 7 | **291.9 ms (better)** | **54.2 ms (worse)** |

**The same flag moves the two model families in opposite directions, and keeps
moving them.** `n_max=1` on MoE is the only setting that is not a loss, and it is
within noise of not using MTP at all.

## Why, in one line of arithmetic

```
union(K) = n_expert x (1 - (1 - top_k/n_expert)^K)        Ornith: 256 experts, top-8

K = 1 ->  8.0 experts      K = 4 -> 30.5      K = 8 -> 57.4
```

Per pass, on 41 layers at 1.819 MiB per expert:

| | bytes read | tokens produced | per token |
|---|---:|---:|---:|
| no MTP | 597 MiB | 1 | **597 MiB** |
| `n_max=3` | 2274 MiB | 3.66 | **621 MiB** |

3.8x the bytes for 3.66x the tokens — a 4% loss, against 2.5% measured. A dense
pass reads the same bytes at any K, so the same arithmetic gives a straight
division and S29's 2.11x speed-up.

> **Speculation amortises reads only when a pass's read set does not grow with
> the number of tokens in it.** Dense: it does not. MoE: it does, and faster than
> the tokens do.

## An important limit on this conclusion

**This is a statement about a machine where reading is expensive, not about MoE
in general.** This host is UMA: CPU and GPU share one ~70 GB/s DDR5 pool, so the
widened union is paid at memory-bus prices and dominates.

On a discrete GPU the arithmetic can invert. VRAM bandwidth is far higher and
batch-1 decode is closer to fixed-cost bound, so the extra experts are cheap
while the amortisation of per-token overhead is the same. **Reports of MTP
helping MoE on dGPUs are not in conflict with this.** As with §4 and §7, the
conclusion belongs to the regime, not to the technique.

## Caveats

- One MoE model. Ornith-1.0, which every figure in `RESULTS.md` uses, has no
  `nextn_predict_layers` and cannot be tested this way at all.
- Acceptance is flat at 0.887 from `n_max=3` upward, so the losses above are
  the union widening, not the draft degrading.
- Streaming is *more* tolerant than full residency (S31: −6.3% vs −11%), because
  the expert cache already holds some of what the widened union pulls in.

> Reproduce: `research/spikes/s32_moe_nmax/measure.sh`
> Related: `S31-moe-mtp.md`, `S13-speculative-decoding.md`, `S29-dense-tuning.md`
