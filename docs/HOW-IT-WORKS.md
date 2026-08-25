# How MoEStream works

> Written to be readable without prior knowledge. Every number is measured;
> the sources are `docs/RESULTS.md` and `docs/findings/`.

---

## 0. In one sentence

**It runs large AI models on a computer that does not have enough memory for
them.**

Other approaches to the same goal exist. What differs is **what gets sacrificed**.

| Approach | How it shrinks | What it costs you |
|---|---|---|
| Quantization (Q4 → Q2 → Q1) | make the numbers coarser | **intelligence** |
| **MoEStream** | leave weights on the SSD, fetch what is needed | **speed** |

Measured: memory **17.3 GB → 7.4 GB (a 57% cut)** while keeping **70–80%** of
the speed.

On intelligence, the honest answer splits in two (§9 has the detail):

- **Degradation from the streaming itself measured as zero.** With few slots
  (fetching from SSD constantly) and with slots for everyone (never fetching at
  all), the quality metric agrees **to four decimal places**.
- **But there is a +0.24% difference from plain llama.cpp, and it does not go
  away.** The cause is arithmetic ordering, not lost information — **which is
  still not zero.**

And beyond that: **models that plain llama.cpp cannot even start now run.** On
this machine (23.5 GB reachable by the GPU) a 36.5 GB or 58.5 GB model does not
load at all; under MoEStream they run in 13.2 GB and 14.5 GB (§9).

---

## 1. Why models are large

A model is, underneath, an **enormous table of numbers** called weights.

`Ornith-1.0-35B`, used throughout here, holds **34.6 billion** of them.
Compressed (quantized) to about half a byte each, that is **16.9 GB**.

To generate text, **the whole table has to be in memory**. Which is why a 16 GB
machine cannot run it.

### The usual fix, and its limit

The standard response is **quantize harder** — reduce the precision of each
number so the table shrinks.

```
Q8  : 8 bits each   ->  large / smart
Q4  : 4 bits each   ->  the usual balance
Q2  : 2 bits each   ->  small / visibly dumber
Q1  : 1 bit each    ->  substantially broken
```

Coarser numbers do make the table smaller, and **the model does get dumber**.
It is one-way: the discarded information is not coming back.

**MoEStream attacks the problem along a different axis.**

---

## 2. What MoE is — not everyone works

This is the starting point.

### An ordinary (dense) model

Every token uses **all** the weights. All 34.6 billion numbers get read. Hence
all of them must be in memory.

> **They still have to be read — but not all at once.** MoEStream streams a
> dense model's FFN one layer at a time, which cuts memory by 56% at no cost to
> accuracy. Generation pays 3.2x for it; prompt processing pays nothing, because
> a forward pass reads the same weights whether it carries one token or a
> thousand. See [Dense models](../README.md#dense-models-too).

```
generate 1 token -> use all 34.6 billion numbers
```

### MoE (Mixture of Experts)

MoE is different. The model contains **many experts**, and **only a few of them
work on any given token**.

For this model:

```
40 layers, each holding 256 experts
                     |
   per token, only 8 of them work in each layer
                     |
        248 of 256 (97%) are idle
```

Picture a 256-person company where each job calls in 8 people. The other 248 are
not involved in that job.

### Which raises an obvious question

> **Do the 248 idle ones need to be in memory at all?**

That question is the whole project.

In this model **86% of the bytes are expert weights** (14.5 GB); everything else
comes to just 1.9 GB. Get the experts out and the reduction is dramatic.

---

## 3. Why "fetch only what you need" is hard

Simple to say. Two walls in the way.

### Wall 1: you do not know who is needed until the last moment

A small mechanism called the **router** decides, and **its answer does not exist
until it is computed**.

```
input -> [router] -> "this time: #3, #17, #88, ... 8 of them"
                        ^ only known here
```

So you **cannot prepare in advance**. The moment you know, you have to fetch
from SSD.

And there are 40 layers, each router depending on the previous layer's output.
It is strictly sequential.

### Wall 2: the program assumes everyone is present

MoEStream is built on llama.cpp, whose code is written around:

```
"there is an array of 256 experts in memory; use #3"
```

If **only 64 are resident**, that assumption breaks.

Rewriting llama.cpp would solve it, and is exactly what we want to avoid —
every upstream release would then have to be merged by hand.

---

## 4. The core idea — rewrite the roster

Getting past wall 2 is the neatest part of this project.

**Provide only 64 seats, and swap names for seat numbers.**

```
Memory holds 64 seats (not all 256 experts)

  seat0  seat1  seat2  ...  seat63
 +-----++-----++-----+     +-----+
 | #17 || #3  || #88 |  …  | #42 |    <- who is sitting there now
 +-----++-----++-----+     +-----+

The router says "use #3 and #88"
      |
Look up: #3 is in seat 1, #88 is in seat 2
      |
Rewrite the instruction to "use seat 1 and seat 2", then hand it to llama.cpp
```

From llama.cpp's point of view **the roster simply got smaller**. Told to use
seat 1, it uses seat 1. **Not one line of its code changed.**

This is the **Slot Table + ID Remap**.

If someone called for is not seated, they are fetched from SSD into a free seat.
If no seat is free, whoever has gone longest without being called gives theirs up
(cache eviction).

### Why this works at all

**Because the same experts get called repeatedly.**

Measured: with seats for 25% of them (64 seats), the probability that a called
expert is already seated — the hit rate — is **75–88%**. Seven to nine calls out
of ten need no SSD access.

The break-even hit rate worked out to 51%, so there is **more than 1.5x of
headroom**. (Going to 38% raises the hit rate to 82–91%, but costs 1.5 GB for
almost no speed, which is why the default is 25%.)

---

## 5. The most surprising finding

This is the part with the most reuse value outside this project.

### The received wisdom

Published work on the same problem (Klotski, ProMoE, MoE-Infinity) shares an
assumption:

> SSDs are slow. So predict who will be called next and prefetch cleverly.

Which sounds entirely reasonable.

### What measuring showed

**The assumption did not hold.**

Breaking down the time to generate one token:

```
reading data from SSD          0.5 ms
CPU and GPU waiting on each other  11.4 ms   <- 23x more
```

**The bottleneck was not the SSD. It was CPU/GPU synchronization.**

### And later, that inverted

That measurement is from 2026-08-04. After the improvements in §6 (fetch only
who is needed, fetch while working) the same measurement gave:

```
                          2026-08-04    2026-08-07
reading from SSD             0.5 ms  ->  7.8-12.5 ms
CPU/GPU synchronization     11.4 ms  ->  2.2 ms
```

**We removed the synchronization ourselves, and the SSD became the dominant cost again.**

The original measurement was not wrong. The problem was that **the
implementation changed and the most-quoted number was not updated**. It took
re-measuring to notice.

The inversion does not change the conclusion below — that all the prefetching
was wasted — because those schemes **lost on direct measurement, not on the
reasoning**. What changed is the rationale, not the result.

Why synchronization is needed at all: the seat lookup runs on the CPU.

```
GPU computes "expert #3 and #88"
   | tell the CPU            (sync #1)
CPU rewrites to "seat 1, seat 2"
   | hand back to the GPU    (sync #2)
GPU continues
```

Times 40 layers, at 0.285 ms per layer: 11.4 ms.

### Why this contradicts the published work

**The models are a different size.**

```
700B class : 11 GB read per token   -> the SSD genuinely cannot keep up
35B class  : 455 MB read per token  -> the SSD has room to spare
```

That work targets much larger models, where "the SSD is the wall" is correct.
At 35B it is not. **The same-looking problem was a different problem.**

### What follows from that

If I/O is not the constraint, **every trick that makes I/O smarter is wasted**.
It was.

| What was tried | Result |
|---|---|
| Predict the next experts and prefetch (3 schemes) | **all three were slower than not doing it** |
| Push prediction accuracy to 81.4% | getting the state it needs off the GPU (14–22 ms) exceeded the I/O it could hide (0.5 ms) |
| `io_uring` for faster I/O | **+1.5%** over parallel `pread`. 4.48 GB/s was the device's ceiling |
| Pin the frequently used experts | **lost** to plain eviction (−1.7 points) |

The lesson: **measuring correctly comes before having clever ideas.**

---

## 6. The other problem — reading a prompt calls everyone

Everything so far was about **generating** text (decode). There is also
**reading** it (prefill).

### Why reading is worse

Generating is one token at a time, so only 8 experts are called. Reading
processes tokens **in batches**. Batch 512 of them:

```
512 tokens x 8 experts = 4096 calls
        | the same experts recur, so after removing duplicates
      very nearly all 256 are called
```

**64 seats is nowhere near enough.** Too few seats corrupts the output.

### The first fix, and what it cost

The only lever was to shrink the batch. How many tokens fit is computable:

| Batch size | Total calls | Distinct experts | Seats to provide |
|---:|---:|---:|---:|
| 6 tokens | 48 | ~44 | **59** ← fits in 64 |
| 8 tokens | 64 | ~57 | 74 |
| 512 tokens | 4096 | ~256 | 302 |

"Seats to provide" exceeds "distinct experts" because of a **safety margin**.
The actual count varies per batch, so provisioning for the average leaves some
batches short — and short means corrupt output. The figure is 1.15x.

So the batch had to shrink to single digits, which is desperately inefficient.
**Reading got 5x slower.**

Measured at the time (97 seats, batches of 8):

```
plain llama.cpp, batch 512   295 tok/s
     | shrink to batch 8
plain llama.cpp, batch 8     74.9 tok/s   <- a 4x loss
     | plus MoEStream's own cost
MoEStream                    59.7 tok/s   <- only 1.25x more
```

Note what that says: **80% of the slowdown came from shrinking the batch**, not
from streaming.

### The fix — borrow a big room temporarily

Adding seats fixes it, but seats are exactly the resident memory we are trying
to save. Doubling them took memory from 8 GB to 13.5 GB, which defeats the
purpose.

Hence the **staging arena**:

```
[permanent workspace]  64 seats                  <- for decode. Not enlarged.
[temporary big room]   256 seats (one layer)     <- for prefill
```

The key is that **the big room only ever holds one layer**.

```
layer 1: bring layer 1's 256 experts into the room -> compute
layer 2: empty it, bring layer 2's 256 -> compute
layer 3: same room again … (40 times)
```

Reused each layer, the extra memory is **one layer's worth: 498 MB**. Providing
all 40 layers would need 19.5 GB; rotating one costs 0.5 GB.

That removed the seat constraint, so the batch could go back from 8 to 512.

There is another step. The room **re-reads every expert once per batch**, so
batching more tokens directly reduces the number of re-reads:

```
batch 512  ->  67000 / 512  = 131 refills  ->  1.69 TB read in total
batch 1024 ->  67000 / 1024 =  66 refills  ->   851 GB read in total
```

Going 512 → 1024 made reading **1.7x faster** for 0.14 GB more memory. Going to
2048 adds only 5%, which is the signal that **the cost moved from the SSD to
compute**. That is where the current ceiling is.

### Two more improvements

**Call only who is needed.** The room was being filled with all 256 experts
every time, though only some get used — especially for short inputs (the ~40
tokens an agent sends each turn), where only 30% are. Restricting the fill made
that case **40–50% faster**.

**Fill while working.** The GPU used to wait while the room filled. **Filling
one layer ahead** removes the wait. Worth +11% to +73% depending on the model.

Neither needs configuring. **Which strategy wins inverts between models**, so it
is decided at startup (models whose experts fit in page cache prefer "fill
everything"; those that do not prefer "fill selectively").

And importantly, **the generated text matches**. Not one character differs from
plain llama.cpp with all weights resident.

That is not the same as "the numbers are identical". When generating, the model
picks **the highest-scoring candidate**, so scores can differ slightly without
changing which token wins. Matching text is strong evidence but **not proof of
numerical identity** (§9).

### Where that landed

```
before the arena (8 tokens at a time)   46.0 tok/s
     | big room + batch 1024 + selective fill + fill-ahead
MoEStream (current)                    242.5 tok/s   <- 5.3x
     (for reference) plain llama.cpp   295.6 tok/s   <- 82% of it
```

**It never exceeds plain llama.cpp.** A measurement once suggested otherwise;
that was an artifact of a short prompt. The room refills per batch, so longer
prompts mean more refills. A short prompt only refills once, and that cost never
gets counted.

### You do not have to choose the settings — two-stage auto-tuning

Two numbers have come up: **seat count** (how many to keep on hand) and **batch
size** (how many tokens to read at once). Both move memory and speed
substantially, and **the best value differs per model with nothing worth
copying**.

Measured optima for batch size:

| Model | Best |
|---|---|
| Ornith-35B | 1024 |
| Qwen3-Coder-Next | 4096 |
| Laguna-S-2.1 | 8096 |

**An 8x spread.** Not the kind of value to make a user pick. So both are
measured and remembered. In `.env` (these are the defaults in `.env.example`):

```bash
MOESTREAM_CACHE_FRAC=learn   # stage 1: seat count
UBATCH=learn                 # stage 2: batch size
```

What is learned is recorded under `./state/`, **per model**.

**Stage 1 (seats) does not need measuring by trial.** Recording "how many
*distinct* experts were touched since this one was last used" yields, **from a
single run, the hit rate at every possible seat count at once**. No repeated
attempts at different sizes. From that it computes how many points of hit rate
another GiB buys, and recommends where that stops being worth it. It is capped
by measured device memory, because **hit rate alone will cheerfully recommend a
seat count that cannot start**.

**Stage 2 (batch size) can only be measured.** A larger batch means fewer
refills but more compute per token, and the growth is not linear, so **one
measured point cannot be extrapolated from**. Each start measures one candidate;
after four, the fastest is kept.

**The two are not independent.** More seats means fewer refills, which weakens
the case for a large batch, so **the best batch size moves down as seats go up**.
Measured on Ornith: at 25% seats, 1024 wins; at 15%, 2048 does. Each batch-size
measurement therefore records how many seats were in play at the time. When the
seat count moves, old measurements stop matching and are simply taken again.

---

## 7. The claim in this document that turned out to be wrong

For most of this project's life, the answer to "does it help an ordinary dense
model?" was **no**, and the reasoning looked airtight. An MoE model is mostly
experts, and only a few experts run per token, so most of the file is idle at
any moment and can live on disk. A dense model has no idle part. Every weight
participates in every token. There is nothing to leave behind.

The reasoning is correct. The conclusion drawn from it was not.

### What was actually being assumed

"Nothing to leave behind" quietly means "nothing can be **absent**". But
streaming does not require a weight to be absent from the *computation* — only
from *memory*, and only until the moment it is needed. A dense model's
feed-forward block is used once per layer, in a known order, one layer at a
time. Layer 40's feed-forward weights are not needed while layer 3 is running.

So they can be read then. The model computes exactly the same thing; the bytes
simply arrive later and leave sooner.

Built and measured, on a 27B dense model: **memory 16.18 GiB → 7.50 GiB**, with
perplexity identical to four decimal places. Not "close". Identical — because
nothing about the arithmetic changed.

### What it costs, and the law underneath

Generation went from 208 to 647 milliseconds per token — about **3x**. That is a
serious price, and much worse than the MoE case, which is about 1.3x.

The reason those two numbers differ turns out to be the single most useful thing
this project learned, and it is one sentence:

> **A technique pays exactly to the extent that a pass's read volume does not
> grow with the number of tokens in it.**

Work through what that means for each kind of model.

- **Dense.** A pass over the feed-forward weights reads the same bytes whether
  you are computing one token or sixteen. Put sixteen requests in the batch and
  the read cost is *divided by sixteen*. Measured: the penalty falls from 3.15x
  at one request to **1.20x at sixteen**.
- **MoE.** A pass reads whichever experts the tokens in it happen to want. Two
  tokens usually want different experts, so the bytes read **grow** with the
  batch. The same trick barely helps: 1.60x to 1.32x.

The same law explains something that had been filed as a separate mystery.
Speculative decoding — guessing several tokens and checking them in one pass —
*costs* an MoE model while *gaining* a dense model 2.29x. It is not two
phenomena. It is one, seen from both sides: speculation puts more tokens in a
pass, and only the dense model's read volume refuses to grow with them.

### Why the wrong conclusion survived so long

Nobody measured it. The argument was clean, it matched the folklore, and it
pointed away from work rather than toward it. A claim shaped like that is the
most dangerous kind, because there is no failing test to trip over — the feature
simply never gets built, and the reason it was not built never gets checked.

The correction is written into the project's own record: `docs/RESULTS.md` §14
carries the row **"dense models — overturned"**, next to the rows for ideas that
were tried and rejected. Both kinds of entry cost the same to make. Only one of
them is comfortable to write.

---

## 8. A record of failures — the most useful part

This repository keeps **everything that did not work**. These were all actually
hit, and all of them generalise.

### Failure 1: celebrating a speed-up without checking output

A technique that made reading 2.9x faster (Expert Sweep) was implemented and
believed. **Output quality was never checked.**

When measured, the quality metric (PPL) had gone from **4.44 to 520801**. Five
orders of magnitude. Completely broken.

The cause was intermediate results being overwritten by another computation. It
was pinned down by inserting an operation that is **mathematically a no-op** and
watching the result change by 2800x. In a correct program that cannot happen.

> Lesson: **a speed improvement means nothing without a quality check beside it.**

### Failure 2: nearly fooled by plausible output

When the arena was first implemented, the output looked grammatical and sensible.
On a hunch, **the same question was asked twice — and gave different answers.**

The cause was copying expert ids. They are sometimes **strided rather than
contiguous**, and the code assumed contiguous and copied in one block. The
result: **the wrong experts were being called.**

The insidious part is that wrong experts still produce plausible-looking text.
The failure is quiet enough to survive a glance.

What settled it was **running plain llama.cpp three times and confirming exact
agreement**, which established that the variation was ours.

> Lesson: **"the output looks plausible" is not evidence of correctness.**
> Check against a reference implementation, and check reproducibility on the
> same input.

### Failure 3: 19x slower merely by existing

The arena's memory was allocated by a particular mechanism (host pointer
import). Reading got faster — and **generation went from 53.8 to 1017.9
ms/token**.

Investigation showed **it was slow even when the memory was never used**.
Allocated, never referenced by any computation. Still slow.

The apparent cause is the GPU driver revalidating that entire region (120,000
pages) on every command submission.

The fix was to stop using that mechanism. Ordinary allocation is faster, and as
a bonus gives memory the GPU reads 2.6x faster.

> Lesson: **"it is unused, so it cannot matter" is sometimes false.**

### Failure 4: measuring the wrong thing entirely

The SSD measured **14.23 GB/s**, which was taken as fast. It was not the SSD —
it was **the OS's copy in RAM**. The real figure is **4.48 GB/s**. A threefold
misunderstanding.

Every SSD measurement since bypasses the OS cache explicitly.

> Lesson: **measurement starts with confirming what is being measured.**

### Failure 5: an optimization that broke everything ★the worst kind

While reading the code, **a buffer that was never used** turned up — a fallback
staging buffer with no role on this machine, yet always allocated, wasting 124 MB
on large models. It was changed to allocate only when needed. An obvious
improvement.

That change stopped **every single SSD read from happening**.

The reason is mundane. A limit meaning "how many staging buffers may be used"
had quietly also become "how many reads may be issued at once". No buffers
allocated → limit 0 → **no reads**. And the slot was still recorded as
populated, so **garbage was fed into the computation.**

The frightening part: **it looked faster.**

| | broken | correct |
|---|---:|---:|
| 35B model | 46.9 ms/tok | 58.7 ms/tok |
| 80B model | 54.6 ms/tok | 101.7 ms/tok |
| 120B-class model | 85.9 ms/tok | 319.2 ms/tok |

Of course it was fast — it was not reading the SSD. And for a 3.7x speed-up
there was **a perfectly coherent explanation available**: "the prefetch we just
added must be working". It came very close to being recorded as an improvement.

It was caught by looking at **output**, not speed:

```
The capital of France is  ->  ' Paris. ernaernernREPLREPLR2555 }R2 5 }R2 '
```

Correct for one word, then collapse. But of three models, only one broke that
visibly. The other two returned `Paris, a city renowned for...` — **plausible
sentences**. **Without that one model, all three would have passed as "faster".**

Two fixes: the limit was corrected, and **a watchdog now logs "output will be
corrupt" if even one read is discarded**.

> Lesson: **a broken implementation produces the best speed numbers.**
> Failures 1 and 2 taught the same thing, but those broke *visibly* — slower, or
> obviously wrong. This one **lied in the favourable direction**. An unexpected
> improvement is the moment to be most suspicious.

### Failure 6: the standard benchmark tool lied

llama.cpp ships `llama-bench`. It reported MoEStream at **113% of plain
llama.cpp** — that reading from SSD is faster than not. Impossible.

The cause was **prompt length**. The tool defaults to 512 tokens, so the arena
refills once and the refill cost is never counted. At 4096 and above it settles
at 76%, close to the 82% measured on real prompts.

| Prompt length | vs plain llama.cpp |
|---|---:|
| 512 (the tool's default) | **113%** ← false |
| 4096 | 76% |
| 13312 | 75% |
| a real document, 13877 | 82% |

Generation speed measures correctly at the default (74%, against 71% measured).
**Only the prompt-processing measurement was broken.**

> Lesson: **even a standard tool has to be checked against your own
> implementation before you trust its numbers.**

---

### Failure 7: a ruler that stretched depending on what it measured

To decide the batch size automatically, a speed measurement was written for the
purpose. Comparing four candidates with it:

| Batch size | llama.cpp's own figure | our ruler | gap |
|---|---:|---:|---:|
| 1024 | 115.0 | 117.9 | +2.5% |
| 2048 | 128.4 | 132.0 | +2.8% |
| 4096 | **135.1** ← fastest | 142.4 | +5.4% |
| 8192 | 133.5 | **152.7** ← fastest | **+14.4%** |

**The gap grows to the right, and the winner disagrees.** llama.cpp says 4096;
our ruler picks 8192.

The cause was the exclusions built into the ruler. Measuring per batch, the
first batch (nothing cached yet, so extremely slow) and the trailing partial
batch both have to be excluded. But **the share excluded grows with the batch
size**. At 8192, 28% of the prompt was discarded as "trailing partial", leaving a
single batch to speak for the whole thing.

> **An exclusion added to remove bias was generating a new bias.**

The fix was to change the unit: **measure per request, not per batch**. A
request's definition does not move with the batch size, and a request is what
the user is actually waiting for. Nothing is discarded; only the first request
is excluded. Re-measured, the gap flattened to 0.02–0.55% and the winner agreed.

**Why not do that from the start.** The per-batch measurement already existed
**for a different purpose** — separating I/O time from compute time — where the
batch *is* the right unit. It was reused for a different question without
checking that the unit fit. And llama.cpp's own figure **had been in the log all
along**; it was only cross-checked at the very end, where one experiment settled
it.

> Lesson: **check against an external ruler before building your own.**
> And when reusing a measurement, confirm its unit matches the new question.

---

## 9. So what is the actual trade-off

Back to the table at the top.

```
[quantize harder]
  16.9 GB -> 7.4 GB
  discards information; the model is definitely dumber, permanently

[MoEStream]
  16.9 GB -> 7.4 GB
  discards not one bit (the weights are intact on the SSD)
  quality metric differs by +0.24% (arithmetic ordering; not zero)
  costs about 1.4x in speed
```

**Neither is better; they cut different things.**

- If intelligence is not critical, quantization is simpler and faster
- **If you want the memory saving without the intelligence loss, this is the
  only option here**

And speed is recoverable by waiting. Intelligence is not. That asymmetry is why
this approach exists.

### Is the intelligence really intact?

Rather than hand-wave, this was measured and separated.

The quality metric (PPL, lower is better), **three runs each under identical
conditions**. The first surprise was that **the variance was zero**:

```
plain llama.cpp   5.0919  /  5.0919  /  5.0919   <- identical all three times
MoEStream         5.1040  /  5.1040              <- also identical
```

Not one movement in the fourth decimal. Which means the +0.24% gap is **not
chance — it is systematic**. It cannot be waved away as noise.

### Locating the difference

So it was measured again with **every expert resident**. With seats for all of
them, nothing is ever evicted — **zero SSD fetches**.

```
64 seats  (fetching constantly)   5.1040
256 seats (never fetching)        5.1040   <- identical
```

**Identical.** Fetching or not fetching changes nothing at all.

That is:

> **Fetching from SSD does not cost any accuracy.**
> The difference comes from somewhere else entirely.

### What does differ — the order of addition

When summing expert outputs, floating-point arithmetic **gives a slightly
different last digit depending on the order of the additions**. Unlike mental
arithmetic, `(a+b)+c` and `a+(b+c)` are not strictly equal.

Changing the number of seats changes the memory layout, which changes the
summation order. That is exactly why the difference persisted with no fetching.

**Not one bit of any expert weight was altered.** Nothing was made coarser and
nothing was skipped. No information was discarded.

### The conclusion, stated in two parts

Saying "zero degradation" is half right and half wrong. Precisely:

> **(1) Degradation from SSD streaming measured as zero.**
> Reducing the seats does not change the result (confirmed by experiment).
>
> **(2) But there is a +0.24% difference from plain llama.cpp that does not go
> away.** The cause is arithmetic ordering, not lost information. It is still
> not zero.

"The generated text is character-for-character identical" is weaker evidence
than (1). Generation picks **the highest-scoring candidate**, so scores can
differ without changing the ranking. Indeed both hold at once: **identical text
and a 0.24% metric difference.**

### Where it is the wrong tool

| Situation | Use |
|---|---|
| Memory is sufficient (24 GB free for a 35B) | **plain llama.cpp / vLLM** |
| You want a 700B class model and do not care about speed | **other approaches** — I/O is the constraint there, which is a different problem |
| **30–120B class, in a fraction of the memory, at a usable speed** | **MoEStream** |

### The goal was coexistence; "it runs at all" turned out to matter as much

The design goal was to turn **"runs, but takes the whole machine"** into
**"runs, and leaves room for everything else"**. A 35B runs fine in 17 GB, so
that case is not about *whether* it runs.

Measurement made the second effect look at least as valuable.

This machine gives the GPU 23.5 GB. Against that:

| Model | Size | plain llama.cpp | MoEStream |
|---|---:|---|---|
| 35B | 16.9 GB | runs | 7.4 GB |
| 80B class | 36.5 GB | **does not start** | **13.2 GB / 9.8 tok/s** |
| 115B class | 54.7 GB | **does not start** | **18.5 GB / 3.1 tok/s** |
| 120B (gpt-oss) | 58.5 GB | **does not start** | **14.5 GB / 3.9 tok/s** |

The bottom three **do not start**. Not "slower" — they do not fit, so they never
come up. Making them run is a separate kind of value from coexistence.

### Is 3.9 tok/s slow?

Yes, though not for this class of system.

```
3.9 tok/s -> about 25 seconds for a 100-word reply. Usable if you are patient.
9.8 tok/s -> practical. Fine behind an agent.
```

**There are stages between "it runs" and "it is usable".** A 120B model at
3.9 tok/s sits at "not for constant use, but genuinely usable". An 80B at
9.8 tok/s is fine day to day.

And what matters most: **neither of them starts on this machine otherwise.**

---

## 10. The numbers

All measured on Ryzen 7 8745HS + Radeon 780M + PCIe4 NVMe, 30.6 GB RAM,
**23.5 GB reachable by the GPU**; speed and memory re-measured 2026-08-24 on
llama.cpp `b0539c43`, perplexity carried over from the earlier commit.

### On the 35B model

| Metric | plain llama.cpp | MoEStream | |
|---|---:|---:|---|
| memory | 17.77 GB | **7.93 GB** | **−55%** |
| generation | 23.4 tok/s | 16.5 tok/s | −29% |
| prompt processing | 256.2 tok/s | 239.4 tok/s | −7% |
| quality (PPL, lower is better) | 4.4400 | 4.4494 | **+0.21%** |
| generated text | — | not one character differs | |

*(Re-measured 2026-08-24 on llama.cpp `b0539c43`. Prompt processing reads 93% of
plain where it read 82% on the older commit — because plain llama.cpp's own
prefill fell 13%, not because streaming improved.)*

Quality was re-tested on a separate corpus (§9):

| | PPL | Difference |
|---|---:|---:|
| plain llama.cpp (identical three times) | 5.0919 | — |
| MoEStream, 64 seats (constant fetching) | 5.1040 | +0.24% |
| MoEStream, 256 seats (no fetching) | **5.1040** | +0.24% |

The point is that the bottom two agree exactly. **Streaming costs nothing**;
what remains is a **+0.24% difference from plain llama.cpp** caused by
arithmetic ordering. With zero measurement variance, that difference is not
noise.

### Across four MoE models

| | 35B | 80B class | 115B class | **120B** |
|---|---:|---:|---:|---:|
| model size | 16.9 GB | 36.5 GB | 54.7 GB | **58.5 GB** |
| plain llama.cpp | runs (17.8 GB) | **does not fit** | **does not fit** | **does not fit** |
| **MoEStream memory** | **7.9 GB** | **13.4 GB** | **18.8 GB** | **14.9 GB** |
| generation | **16.5 tok/s** | **14.8 tok/s** | 3.1 tok/s | 3.8 tok/s |
| prompt processing | 239 tok/s | 149 tok/s | 76 tok/s | 92 tok/s |

Only the 35B can be compared, because the others **do not start under plain
llama.cpp** (they exceed the GPU's 23.5 GB). So the question here is not "is it
faster" but **"does something that cannot run, run usably"**.

The 80B at 13 GB and 9.8 tok/s is the most practically useful result. The 120B
is the most striking: it is **larger** than the 115B-class model yet uses less
memory and runs faster, because it routes to 4 experts per token instead of 10.
**What costs you is `top_k`, not model size.**

### On dense models

Added later, after §7 established that the "dense cannot benefit" reasoning was
wrong. Measured on Qwen3.8-27B, one request at a time:

| Metric | plain llama.cpp | MoEStream | |
|---|---:|---:|---|
| memory | 16.49 GB | **7.81 GB** | **−53%** |
| prompt processing | 69.1 tok/s | 67.0 tok/s | **−3%** |
| generation | 4.69 tok/s | 1.28 tok/s | −73% |
| generation, both sides speculating | 8.93 tok/s | 3.79 tok/s | −58% |
| quality (PPL) | identical to four decimals | | |

And the same generation cost with more requests in flight, which is what decides
whether dense streaming is cheap or expensive:

| requests at once | 1 | 4 | 8 | 16 |
|---|---:|---:|---:|---:|
| cost vs. plain llama.cpp | 3.15x | 1.90x | 1.62x | **1.20x** |

### The main breakdown

| | |
|---|---|
| share of the model that is experts | 86% (14.5 GB of 16.9 GB) |
| seats (share of experts kept resident) | 25% (64 of 256) |
| probability of already being seated (hit rate) | 75–88% |
| hit rate actually required | 51% (over 1.5x of headroom) |
| CPU↔GPU synchronization per token | 2.2 ms (11.4 originally; reduced) |
| SSD reads per token | 7.8–12.5 ms (0.5 originally; the ratio inverted) |
| effective SSD speed | 4.48 GB/s |
| product code | 4541 lines |
| code added to llama.cpp | 4 files, 5 blocks, ~95 lines |

---

## 11. Where to read more

| Document | Contents |
|---|---|
| [`RESULTS.md`](RESULTS.md) | the full measurements, including failed approaches and our own mistaken measurements |
| [`DESIGN.md`](DESIGN.md) | design document (35 chapters, 34 decisions), with the 14 overturned by measurement kept as history |
| [`USAGE.md`](USAGE.md) | running and configuring it |
| `findings/` | primary sources per experiment — what was measured and how |

### Particularly worth reading

- `findings/N4-expert-sweep.md` — the "faster but broken" story, with the process of eliminating suspects by measurement left intact
- `findings/S7-prefill-arena-impl.md` — the arena implementation, and the detail behind failures 2 and 3
- `RESULTS.md` §10.8 — **the full record of failure 5**, where an "improvement" broke everything
- `RESULTS.md` §12.2 — **failure 6 in detail**, and how to use the standard tool correctly
- `RESULTS.md` §10.16 — **the full record of failure 7** and the two-stage auto-tuning: five fixes inside the same frame before the frame itself was questioned
- `RESULTS.md` §13.3 — **a record of our own measurement mistakes**

---

## Appendix: terms

| Term | Meaning |
|---|---|
| **token** | the unit of text a model handles; roughly a word in English |
| **weights** | the enormous table of numbers that is the model |
| **quantization** | making the numbers coarser to shrink the table; Q4 is 4 bits each |
| **MoE** | mixture of experts; only some experts run per token |
| **expert** | a component inside MoE; 256 per layer here |
| **router** | the small mechanism that picks which experts to call |
| **prefill** | the phase that reads the given prompt |
| **decode** | the phase that generates text one token at a time |
| **PPL (perplexity)** | a quality metric; lower is better |
| **hit rate** | the share of calls where the expert was already in memory |
| **UMA / unified memory** | CPU and GPU sharing the same RAM — an integrated GPU |
