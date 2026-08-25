# MoEStream

![MoEStream architecture — a router picks top-k experts; an id remap turns each expert_id into the slot_id it occupies in a small pool of GPU-resident slots; ggml's mul_mat_id then runs unchanged against that smaller array. Experts not in a slot are read from the NVMe SSD into one.](imgs/moestream.jpeg)

> # You no longer have to pay for memory with accuracy.

When a model does not fit, the usual answer is to quantize harder and accept
that it gets dumber. MoEStream spends **speed** instead. The weights stay on the
NVMe SSD and stream in only as they are needed, so nothing is discarded and
nothing is approximated — every weight that reaches the arithmetic is the one the
file holds, and perplexity is identical to four decimal places.

Streaming weights from storage is not a new idea; what is measured here is what
it costs, and where the cost actually lands, at this scale.

**Mixture-of-Experts and ordinary dense models are both supported**, and you do
not have to say which — MoEStream reads the file and picks the path. It is a
patch on upstream llama.cpp, not a fork, so you keep its server, its Web UI and
its model support.

[**Quick start**](#quick-start) · [What it costs](#what-it-costs-measured) ·
[Dense models](#dense-models-too) · [Will it run here?](#will-it-run-on-your-machine) ·
[Docs](#documentation)

### A 120B model on a mini PC with no graphics card

```
  machine    AMD Ryzen 7 8745HS, Radeon 780M (integrated, ~24 GB GTT)
             Crucial P310 NVMe · Linux + Docker · no graphics card
  model      gpt-oss-120b, MXFP4, 58.46 GiB of weights

             plain llama.cpp    will not start  (weights exceed GTT)
             MoEStream          runs in 14.91 GiB   (−74%)
                                3.8 tok/s generation
                               92.2 tok/s prompt processing
```

---

## How it works, in one minute

An MoE model does not use all of its weights on any given token. A router picks
a handful of "experts" per layer — 4 out of 128 in the model above — and the rest
sit idle. They still have to be in memory, though, because the code assumes every
expert is present.

MoEStream removes that assumption. It keeps a small pool of expert-sized **slots**
in GPU memory and leaves the full set on the SSD. When the router asks for expert
`#93`, the runtime fetches it into some free slot `#7` and **rewrites the id the
matrix multiply will use**, from 93 to 7. The kernel never learns anything
changed — it multiplies against a smaller array with the right data in it.

That is the whole idea. Everything else — which experts to keep, how much to read
at once, when to read ahead — is bookkeeping around it, and all of it was
measured rather than guessed.

A dense model has no experts, so there is no router to intercept — but its
feed-forward weights are used one layer at a time, in a known order, and the same
trick works on those. Same idea, different tenant: see
[Dense models](#dense-models-too).

Because the runtime moves *bytes* and never decodes them, **it does not need to
know your quantization format.** IQ4_NL and MXFP4 are verified with no
format-specific code between them; the others are untested rather than known to
fail.

**→ The same idea explained at length, for a general reader:
[`docs/HOW-IT-WORKS.md`](docs/HOW-IT-WORKS.md)**

---

## Will it run on your machine?

| You need | Why |
|---|---|
| **Linux with Docker** | everything builds and runs in a container; nothing is installed on the host |
| **A GPU llama.cpp can use via Vulkan** | integrated is fine and is what this was built for — a discrete card works too |
| **Unified memory (integrated GPU), or enough VRAM for the slots** | the zero-copy read path needs GPU-visible host memory. Tested on AMD UMA (Radeon 780M); other UMA setups should behave the same, but are unverified |
| **An NVMe SSD** | weights are read from it on every token — experts on an MoE model, feed-forward blocks on a dense one. A SATA SSD works and is slow; a hard disk is not usable |
| **A GGUF model, MoE or dense** | both are supported and neither needs configuring: MoEStream reads the expert count and picks the path. MoE is what this was built for and is the cheaper of the two — see [Dense models](#dense-models-too) for what dense costs |

Verified on AMD Radeon 780M (RADV, Vulkan, UMA). **NVIDIA and Apple Silicon are
untested.** Nothing in the design is AMD-specific, but "untested" is the honest
word.

---

## Quick start

**Prerequisites**: Linux with Docker, a Vulkan-capable GPU
([details](#will-it-run-on-your-machine)), a GGUF model, and about 15 GB of disk
for the build. Nothing is installed on your host — the build fetches llama.cpp,
patches it and compiles it inside the image.

```bash
git clone <this repo> && cd moestream
cp .env.example .env
$EDITOR .env                        # one line: MODEL_DIR=/path/to/your/models
make launch                         # builds the image if missing (~25 min), then starts
```

`make launch` reads every GGUF header in `MODEL_DIR` and this machine's memory
limits, then asks only what it cannot work out on its own — which model, one
request at a time or several, how much context, and, when a model would fit
whole, whether you would rather have the memory or the speed. It shows what your
answers imply before starting anything:

```
   #  model                              size   type          fits?
   1  Ornith-1.0-35B-UD-IQ4_NL.gguf    16.87G  MoE 256e      yes
   4  Laguna-S-2.1-UD-IQ4_NL.gguf      54.70G  MoE 256e      NO — streaming is the point

  what this implies
  ─────────────────────────────────────────────────────────────
  weights on disk         16.87 GiB
  KV cache                 0.40 GiB   at ctx 32768 x 1
  cannot be streamed       4.37 GiB   (attention and embeddings)
```

**Everything else is derived at run time and has no setting** — MoE or dense,
how many experts stay resident, micro-batch, arena sizes and thresholds, read
threads, thread count, sampling defaults, GPU group IDs. Three of them keep
improving while you use it: how much of the model stays resident, how many prompt
tokens go through at once, and how far ahead to speculate. Each measures your own
traffic and remembers what it found, per model.

That is all. An OpenAI-compatible API and llama.cpp's Web UI are on
`http://localhost:8091`:

```bash
curl localhost:8091/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"hello"}]}'
```

`make down` stops it, `make logs` follows the log, `make test` runs the checks
that need no hardware.

**→ Filling in `.env` yourself, every setting, what is derived and why, tuning,
troubleshooting: [`docs/USAGE.md`](docs/USAGE.md)**

---

## What it costs, measured

The one **MoE** model here small enough to run both ways, so the comparison is
real rather than extrapolated:

| Ornith-1.0-35B (IQ4_NL) | plain llama.cpp | MoEStream |
|---|---:|---:|
| device memory | 17.77 GiB | **7.93 GiB (−55%)** |
| generation | 23.4 tok/s | 16.5 tok/s (−29%) |
| prompt processing | 256.2 tok/s | 239.4 tok/s (−7%) |
| quality (perplexity) | 4.4400 | 4.4494 (+0.21%) |
| generated tokens | — | identical to plain llama.cpp |

**The perplexity difference is not from streaming.** Measured against itself with
eviction made impossible, MoEStream scores exactly the same to four decimals —
fetching experts from SSD changes nothing. The +0.21% is floating-point reduction
order, reproducible and not noise
([`RESULTS.md` §8](docs/RESULTS.md)).

### The four MoE models tested

Three of them cannot be loaded at all without this.

| | Ornith-1.0-35B | Qwen3-Coder-Next | Laguna-S-2.1 | **gpt-oss-120b** |
|---|---:|---:|---:|---:|
| weights on disk | 16.87 GiB | 36.54 GiB | 54.7 GiB | **58.46 GiB** |
| experts | 256, top-8 | 512, top-10 | 256, top-10 | 128, **top-4** |
| plain llama.cpp | runs | **will not start** | **will not start** | **will not start** |
| **MoEStream** | **7.93 GiB** | **13.44 GiB** | **18.79 GiB** | **14.91 GiB** |
| generation | 16.5 tok/s | 14.8 tok/s | 3.1 tok/s | 3.8 tok/s |
| prompt processing | 239.4 tok/s | 149.2 tok/s | 76.0 tok/s | 92.2 tok/s |

**"Will not start" is not "slower".** On a machine with ~24 GB reachable by the
GPU, three of these four are simply unavailable without MoEStream.

Four architectures, two quantization formats, experts from 128 to 512, and **no
architecture-specific code**: the runtime moves expert bytes and rewrites ids
without ever looking inside a weight. The biggest model is not the slowest —
gpt-oss-120b is 3.8 GiB larger than Laguna yet runs in 3.9 GiB less memory and
generates faster, because it routes to 4 experts per token instead of 10.
**What costs you is `top_k`, not size.**

```bash
research/tools/ms-bench.sh --baseline    # reproduce this on your own machine
```

**→ Conditions, reproduction steps, and the same table on the older llama.cpp:
[`docs/RESULTS.md` §12](docs/RESULTS.md)**

---

## Dense models, too

Not just Mixture-of-Experts. An ordinary dense model has no experts to leave on
disk, but most of it is feed-forward weights — and those can be streamed a layer
at a time. **Half the memory, no loss of accuracy:**

| Qwen3.8-27B (dense) | plain llama.cpp | MoEStream |
|---|---:|---:|
| device memory | 16.49 GiB | **7.81 GiB (−53%)** |
| prompt processing | 69.1 tok/s | **67.0 tok/s (−3%)** |
| generation | 4.69 tok/s | 1.28 tok/s (−73%) |
| generation, **both sides speculating** | 8.93 tok/s | **3.79 tok/s (−58%)** |

Speculation is listed on both sides because plain llama.cpp can use it too.
Comparing a streamed model that speculates against a plain one that does not
would flatter this project, and three figures in it were reported that way before
being caught ([`S34`](docs/findings/S34-like-for-like.md)). With neither side
speculating, streaming costs **3.32x**; with both, **2.36x**.

*(One request at a time. `gemma-4-31B`, a plain transformer with no SSM layers:
**19.04 → 8.44 GiB (−56%)**. And on a model small enough that this was assumed
not to help, `Qwen3.5-4B`: **3.95 → 2.75 GiB (−30%)**. Every model tested drops,
4B to 120B — [`RESULTS.md` §12.4c](docs/RESULTS.md) has all eight.)*

**Reading prompts is free.** One pass over the weights serves the whole prompt, so
a long prompt costs no more to stream than a short one. Summarising, classifying,
reranking, scoring — all of it at full speed in half the memory.

**Generating gets cheaper the more you run at once**, because the same pass serves
every request in flight — 3.15x the time at one request, **1.20x at sixteen**
([`S37`](docs/findings/S37-batch-freezone.md)).

**And a single stream gets some of it back, if the model can self-speculate.**
Some models ship a head that predicts their own next tokens, and verifying
several guesses in one pass is nearly free for a streamed dense model — worth
**2.71x** here against **1.92x** on plain llama.cpp, which is why the gap narrows
from 3.32x to 2.36x.

`make launch` asks llama.cpp whether your model supports this, then measures how
large a draft pays, keeping "don't speculate" among the candidates — because on a
MoE model the same flag *loses*, and above two concurrent requests it loses on
dense too ([`S42`](docs/findings/S42-speculation-by-model.md),
[`S45`](docs/findings/S45-speculation-and-batching.md)).

**Accuracy is untouched** — perplexity identical to four decimal places, measured
through the streaming path itself, not inferred from the resident one.

**Nothing to configure.** MoEStream reads the file, sees there are no experts, and
takes the dense path. If the model already fits, it streams nothing and behaves
exactly like plain llama.cpp.

> **The fine print:** attention weights are never streamed
> ([`S30`](docs/findings/S30-dense-attention.md)); prompt processing is only free
> at micro-batch 1024 and above; generated text is usually byte-identical but not
> guaranteed to be, because memory alignment changes floating-point accumulation
> order.

**→ How it works and what else was tried:
[`S27`](docs/findings/S27-dense-streaming-impl.md)**

---

## Which llama.cpp

MoEStream is **not a fork**. It patches an unmodified llama.cpp checkout at build
time — 4 files, 5 blocks — so you keep upstream's Web UI, server, model support
and future fixes. This project adds one thing and takes nothing away.

| | |
|---|---|
| Default | `b0539c43` (2026-08-23), built and verified on MoE and dense |
| Previous default, kept for comparison | `3581ba0c` (2026-08-02) |
| Backend | Vulkan |

The two were measured against each other, alternating with a warm page cache so
neither got a cold-start advantage: output identical, decode within 1.2%, and
0.51 GiB more memory on the newer one — which it also uses with streaming turned
off, so that part is upstream's, not this patch's.

To track upstream yourself, set `LLAMA_COMMIT=master` and rebuild. If upstream
has moved code a patch attaches to, the build **stops and names the anchor**
rather than producing a quietly wrong binary.

**→ Full comparison and the reasoning behind the pin:
[`docs/RESULTS.md` §14](docs/RESULTS.md)**

---

## Which should you use?

| Your situation | Use |
|---|---|
| **The model does not fit at all** | **MoEStream** — this is the case it exists for |
| It fits, but leaves nothing for anything else | **MoEStream** — a 35B model in 7.9 GiB instead of 17.8 GiB, at about 1.4x the time |
| It fits and you have memory to spare | **plain llama.cpp**, or MoEStream with streaming turned off — same thing, and the launcher offers you the choice |
| A **dense** model on long prompts, short answers | **MoEStream** — half the memory at full prompt-processing speed ([Dense models](#dense-models-too)) |
| A **dense** model serving several requests at once | **MoEStream** — half the memory for 1.2x the time at 16 concurrent |
| A **dense** model, one request at a time, speed matters | **plain llama.cpp** — streaming costs 3.3x here, or 2.4x if the model can speculate. Batching is what makes it cheap |

The original goal was the second row: turning *"runs, but takes the whole
machine"* into *"runs, and leaves room for everything else"*. In practice the
first row turned out to matter just as much — three of the four MoE models tested
here are not slow without MoEStream, they are unavailable.

Is 3.8 tok/s slow? Yes. It is also the difference between having a 120B model
and not having one. For a chat you follow along with, or an agent that works
while you do something else, it is enough. For anything interactive and
impatient, it is not — and no amount of tuning will change that, because the
weights genuinely have to come off the SSD.

---

## What makes this different from other SSD-streaming work

The published work here (Klotski, ProMoE, MoE-Infinity) targets much larger
models on discrete GPUs. At 700B class a token can need ~11 GB read, and there
SSD bandwidth genuinely *is* the wall — so predicting which experts are coming
and fetching them early is the right answer.

At 35–120B on unified memory a token needs ~455 MiB, **the wall is somewhere else,
and several of those conclusions invert.** Every predictive prefetch scheme
measured here lost, one of them at 81.4% accuracy; `io_uring` beat parallel
`pread` by 1.5%; a hand-picked "hot set" loses to plain LRU by 8.1 points of hit
rate. None of that contradicts their work — it is a statement about the regime, not about them.

**→ The full comparison, and our own mistaken measurements alongside it:
[`docs/RESULTS.md`](docs/RESULTS.md)**

---

## Repository layout

```
Dockerfile        the image: fetches llama.cpp, patches it, builds it
compose.yaml      how it runs
.env.example      the one setting a human chooses, plus every override
launcher.sh       the interactive start-up (`make launch`)              386
Makefile          launch / up / down / logs / test / bench

src/              the product. 4155 lines, and that is all of it.
  llama-moestream.{h,cpp}   the runtime: slots, cache, reads, remap   3227
  expert_cache.{hpp,cpp}    S3-FIFO cache, one per layer                425
  apply.py                  where it attaches to llama.cpp              298
  entrypoint.sh             derives threads, batch size, GPU checks     175
  spec_probe.cpp            asks llama.cpp what a model can speculate     30

docs/             design, measurements, and the reasoning
research/         how everything above was arrived at
  tests/            checks that need no GPU (make test)
  tools/            measurement scripts (make bench)
  spikes/           one-off experiments, kept as evidence
  bench/            corpora and analysis summaries
  docker/           images used only for the above
state/            what `learn` recorded on this machine
```

**`src/` is the product; everything under `research/` is the paper trail.** The
spikes are not maintained code — they are the experiments that decided each
design question, kept so the claims in `docs/` can be checked rather than taken
on faith. The image only ever receives the handful of files in `src/` that the
patch needs.

There is no llama.cpp clone in this repository. The build fetches it.

---

## Documentation

| I want to… | Read |
|---|---|
| **run it** | [`docs/USAGE.md`](docs/USAGE.md) — starting it, every setting, tuning, troubleshooting |
| **understand it, no prior knowledge** | [`docs/HOW-IT-WORKS.md`](docs/HOW-IT-WORKS.md) — the problem, the solution, and what failed on the way |
| **check the numbers** | [`docs/RESULTS.md`](docs/RESULTS.md) — every measurement, including the ones that went against us |
| **see why a decision was made** | [`docs/DESIGN.md`](docs/DESIGN.md) — 32 ADRs, 12 revised or rejected by measurement |
| **read one experiment in full** | [`docs/findings/`](docs/findings/) — 44 write-ups, one per question asked |

Japanese versions sit beside each as `*.ja.md`.

### How it attaches to llama.cpp

`src/apply.py` edits **4 files, in 5 blocks**, against an unmodified checkout. If
upstream has moved any of them the build stops and names the anchor, so it can
never produce a half-patched binary —
`research/tests/test_apply_patch.py` proves that by deleting each anchor in turn.

---

## Status

**Working and measured.** MoE and dense streaming are both implemented and
verified end to end — every model tested drops memory, from 4B to 120B, with
output correct and perplexity unchanged. There is an interactive launcher,
tuning that improves while you use it, and a patch that still applies to today's
llama.cpp.

What is missing is **other people's hardware**. Everything here was verified on
one machine: an AMD Radeon 780M iGPU, Vulkan, unified memory. If you run it on
anything else — NVIDIA, Intel Arc, Apple Silicon, a discrete card — the result is
worth reporting either way.

Known limits: one machine and one backend; **attention is never streamed** on any
model, three designs for it having failed; Expert Sweep, the design's original
headline feature, does not work and is disabled; `make test` checks logic only.

**The failures are kept too**, because they were the expensive part: "dense
models cannot benefit" was wrong, prefetching was built and lost, three of our
own numbers flattered streaming before being caught, and a MoE speed measurement
on this machine turns out to be eight times noisier than a dense one — which
produced three "improvements" that did not survive re-measurement.

**→ All of it, with the numbers: [`docs/findings/`](docs/findings/) and
[`docs/RESULTS.md` §14](docs/RESULTS.md)**

---

## License

**MIT** ([`LICENSE`](LICENSE))

MoEStream is a patch, not a fork. No llama.cpp source is included here; the
build fetches it. llama.cpp and ggml are MIT-licensed, and the patched result
remains a derivative work of them.

## Acknowledgements

This project is a thin layer on top of
**[llama.cpp](https://github.com/ggml-org/llama.cpp)** and **ggml** by Georgi
Gerganov and its contributors. Everything hard — the model support, the
quantization formats, the Vulkan backend, the server, the tokenizers — is
theirs. MoEStream changes which bytes are resident and rewrites a few ids; the
inference is llama.cpp's, and it would not be worth doing on a weaker
foundation.

Thanks also to **[Unsloth](https://huggingface.co/unsloth)** for the GGUF
conversions the four MoE models tested here came from.
