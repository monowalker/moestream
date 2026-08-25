# S43 — A MoE decode measurement is eight times noisier than a dense one

*2026-08-24. Found the hard way: three separate "improvements" reported and
retracted in one session, all of them on MoE models.*

## What happened

In a single session, these were measured, believed, and then withdrawn:

| claim | first measurement | properly warmed |
|---|---|---|
| `ngram-cache` speeds up MoE | **+24%** | 49.9 → 49.5 ms/tok (nothing) |
| `--poll 1` speeds up MoE | −1.8% | inside one cluster with every other flag |
| MTP `n_max=1` helps streamed MoE | 54.3 → 52.3 (a gain) | 48.6 → 51.1 (**a loss**) |
| MoE batches 3.2x at 6 concurrent | 3.2x | **1.83x** |
| the frac curve turns over at 0.60 | 0.50 faster than 0.60 | monotonic; 0.60 is fastest |
| a MoE pass at width 2 is free | 94.3 → 94.3 ms | 51.3 → 73.2 (**1.43x**) |

Every one was on a MoE model. Not one dense measurement had to be withdrawn.

## Why

Sample spread, same protocol, same machine, three consecutive 300-token
generations after two warm-up requests:

| | samples | spread |
|---|---|---|
| dense Qwen3.8-27B, FFN streamed | 690.2 / 690.1 / 690.4 | **0.03%** |
| MoE Ornith-1.5, experts streamed | 52.9 / 49.4 / 48.6 | **8.1%** |

And the MoE spread is not symmetric noise — it is a **decaying transient**. The
first sample is slow and each subsequent one is faster, which is what makes it so
good at producing false positives: the *reference* is usually measured first, so
whatever is measured second looks better.

The cause is the expert cache. A dense model streams the same bytes every token,
so it reaches steady state within one request. A MoE model's S3-FIFO pool fills
according to which experts the text happens to want, and a few hundred tokens of
one prompt do not exercise it the way a few thousand do. `docs/RESULTS.md` §7
already recorded the effect on a cold start (76.6 → 50.7 ms/token as it
converges); what was not appreciated is that it is still converging well after
the point a dense model has settled.

## What to do about it

**Two full warm-up requests before measuring, three measured samples, report the
median** — and even that is not always enough: the reference row above still
shows 52.9 on its first measured sample. When a MoE result is within 5% of its
baseline, it is not a result.

The project's own `research/tools/ms-bench.sh` was already doing the right thing
by accident: it drives a 13877-token prefill and a 200-token generation before it
measures anything. Every number it produced this session held up. Every number
produced by an ad-hoc script with one short warm-up did not.

So the rule is not "warm up more", it is **use `ms-bench.sh`**, and if a
one-off script is unavoidable, copy its warm-up.

## The uncomfortable part

The three retracted claims were not caught by a check. They were caught because
someone asked why the number looked odd. A measurement that agrees with what you
hoped is the one least likely to be measured again — and on this machine, on a
MoE model, roughly one in ten such measurements will be wrong by enough to matter.

`docs/RESULTS.md` §10.8 records an earlier version of the same failure: an
"optimization" that made a broken build look fast. That one was found because the
output was checked. This one needed the *speed* checked twice.
