# MoEStream

![MoEStream architecture — a router picks top-k experts; an id remap turns each expert_id into the slot_id it occupies in a small pool of GPU-resident slots; ggml's mul_mat_id then runs unchanged against that smaller array. Experts not in a slot are read from the NVMe SSD into one.](imgs/moestream.jpeg)

> # You no longer have to pay for memory with accuracy.

When a model does not fit, the usual answer is to quantize harder and accept
that it gets dumber. MoEStream spends **speed** instead. Expert weights stay on
the NVMe SSD and stream in only as the router calls for them, so nothing is
discarded and nothing is approximated — every weight is bit-identical to the
original.

Streaming weights from storage is not a new idea; what is measured here is what
it costs, and where the cost actually lands, at this scale.

### A 120B model on a mini PC with no graphics card

```
  machine    AMD Ryzen 7 8745HS, Radeon 780M (integrated, ~24 GB GTT)
             Crucial P310 NVMe · Linux + Docker · no graphics card
  model      gpt-oss-120b, MXFP4, 58.46 GiB of weights

             plain llama.cpp    will not start  (weights exceed GTT)
             MoEStream          runs in 14.48 GiB   (-75%)
                                3.9 tok/s generation
                              106.8 tok/s prompt processing
```

## How it works, in one minute

An MoE model does not use all of its weights on any given token. A router picks
a handful of "experts" per layer — 4 of 128 in the model above — and the rest sit
idle. They still have to be in memory, though, because the code assumes every
expert is present.

MoEStream removes that assumption. It keeps a small pool of expert-sized **slots**
in GPU memory and leaves the full set on the SSD. When the router asks for expert
`#93`, the runtime finds or fetches it into some slot `#7` and **rewrites the id
the matrix multiply will use**, from 93 to 7. The kernel never learns anything
changed: it multiplies against a smaller array, with correct data in it.

That is the whole idea. Everything else — which experts to keep, how much to read
at once, when to read ahead — is bookkeeping around it, and all of it is measured
rather than guessed.

**Because the runtime moves expert *bytes* and never decodes them, the
quantization format is not something it has to know about.** Two were verified
here — IQ4_NL and MXFP4 — with no format-specific code between them. The others
are simply untested, not known to fail; the requirement is that experts are
stored as contiguous per-expert slices, which is how GGUF lays them out today.

## Will it run on your machine?

| You need | Why |
|---|---|
| **Linux with Docker** | everything builds and runs in a container; nothing is installed on the host |
| **A GPU llama.cpp can use via Vulkan** | integrated is fine and is what this was built for — a discrete card works too |
| **Unified memory (integrated GPU), or enough VRAM for the slots** | the zero-copy read path needs GPU-visible host memory. Tested on AMD UMA (Radeon 780M); other UMA setups should behave the same, but are unverified |
| **An NVMe SSD** | expert weights are read from it on every token. A SATA SSD will work and will be slow; a hard disk will not be usable |
| **An MoE model in GGUF** | MoEStream is inert on dense models — it will run them, with no benefit |

Verified on AMD Radeon 780M (RADV, Vulkan, UMA). **NVIDIA and Apple Silicon are
untested.** Nothing in the design is AMD-specific, but "untested" is the honest
word.

---

## What it costs, measured

On the one model small enough to run *both* ways, so the comparison is real:

```
Ornith-1.0-35B-A3B (IQ4_NL)   device memory   17.29 GiB  ->  7.44 GiB    (-57%)
                              decode          41.5 ms/tok -> 58.7 ms/tok (71%)
                              prefill        295.6 tok/s  -> 242.5 tok/s (82%)
                              quality (PPL)   4.4400      -> 4.4494      (+0.21%)
                              generated tokens: identical to plain llama.cpp
```

That PPL difference is **not zero**, but it does not come from streaming.
On a separate corpus, measured three times each ([`docs/RESULTS.md` §8](docs/RESULTS.md)):

| | PPL | vs baseline |
|---|---:|---:|
| plain llama.cpp | 5.0919 | — |
| MoEStream, 64 slots (misses constantly) | 5.1040 | +0.24% |
| MoEStream, 256 slots (eviction impossible) | **5.1040** | +0.24% |

The last two rows are **identical to four decimals**: fetching experts from SSD
changes nothing at all. What remains is a reproducible +0.24% against plain
llama.cpp, caused by `ne[2]` changing the floating-point reduction order — not by
discarding information. Run-to-run variance was zero, so this is a systematic
difference, not noise.

**All figures measured** (2026-08-06, `MOESTREAM_CACHE_FRAC=0.25` /
`UBATCH=1024` / `CTX_SIZE=32768`) on an AMD Ryzen 7 8745HS with a Radeon 780M
(Vulkan/UMA, **GTT limit 23.5 GiB**) and a Crucial P310 (PCIe 4.0), against
**llama.cpp `3581ba0c` (build 10230, 2026-08-02)** — the commit the Dockerfile
pins. See [Which llama.cpp](#which-llamacpp).
Full measurement conditions and reproduction steps: [`docs/RESULTS.md` §12](docs/RESULTS.md).

### Models tested

Four models, four different architectures, on the machine above. All GGUFs came
from [Unsloth](https://huggingface.co/unsloth). **The code is the same for all
four — there are no per-architecture branches** — but the numbers are not: each
row uses a different `frac` (last row of the table), and memory and speed both
move with it. Read the table as four worked examples, not as a ranking.

| | Ornith-1.0-35B | Qwen3-Coder-Next | Laguna-S-2.1 | **gpt-oss-120b** |
|---|---:|---:|---:|---:|
| weights on disk | 16.87 GiB | 36.54 GiB | 54.7 GiB | **58.46 GiB** |
| architecture | qwen35moe | qwen3next | laguna | **gpt-oss** |
| experts | 256, top-8 | 512, top-10 | 256, top-10 | 128, **top-4** |
| expert quantization | IQ4_NL | IQ4_NL | IQ4_NL | **MXFP4** |
| plain llama.cpp | runs, 17.29 GiB | **does not start** | **does not start** | **does not start** |
| **MoEStream** | **7.44 GiB** (−56%) | **13.15 GiB** (−64%) | **18.51 GiB** (−66%) | **14.48 GiB** (−75%) |
| generation | **17.0 tok/s** | **9.8 tok/s** | 3.1 tok/s | 3.9 tok/s |
| prompt processing | **242.5 tok/s** | **150.8 tok/s** | 79.7 tok/s | 106.8 tok/s |
| experts kept resident | 25% | 25% | 20% | 15% |

Percentages are against **weights on disk**, which is the only baseline the three
larger models have — plain llama.cpp never gets far enough to measure. Ornith is
the one case with a real A/B: against plain llama.cpp's measured 17.29 GiB of
device memory, 7.44 GiB is **−57%**.

The last row is `MOESTREAM_CACHE_FRAC` — **the fraction of each layer's experts
that stays in GPU memory**, the one setting that decides the memory/speed
trade-off. 25% of 256 experts means 64 slots per layer; everything else is read
from SSD on demand. Lower it and memory falls, generation slows. **Every other
number in the table is a consequence of that choice**, so the same model at a
different `frac` gives a different row. You do not have to pick it: the default
learns it from your own traffic (see [Settings](#settings)).

**Three of these four do not start at all without MoEStream.** That is not
"slower" — it is the difference between running and not running.

Four architectures, two quantization formats, expert counts from 128 to 512, and
**not one architecture-specific branch in the code**. The runtime moves expert
bytes and rewrites ids; it never looks inside a weight, so the format is not its
concern ([`docs/RESULTS.md` §10.17](docs/RESULTS.md)).

The biggest model is not the slowest. gpt-oss-120b is **3.8 GiB larger** than
Laguna yet uses 4 GiB less memory and is faster at both ends, because it routes
to 4 experts per token instead of 10. **What costs you is `top_k`, not size.**

```bash
research/tools/ms-bench.sh --baseline    # reproduce the table above on your own machine
```

Conventionally, reducing memory means quantizing harder, which discards
information for good. MoEStream discards none: every weight stays intact on the
SSD. What it gives up instead is **speed**.

**→ How it works: [`docs/HOW-IT-WORKS.md`](docs/HOW-IT-WORKS.md)**
**→ Full measurements: [`docs/RESULTS.md`](docs/RESULTS.md)**

---

## Quick start

**Prerequisites**

- Linux with Docker (and the `docker compose` plugin)
- A GPU llama.cpp can reach via Vulkan — see [Will it run on your machine?](#will-it-run-on-your-machine)
- An MoE model in GGUF format, already downloaded
- About 15 GB of disk for the build, plus the model itself

Nothing else is installed on your host. The build fetches llama.cpp, patches it
and compiles it, all inside the image.

```bash
git clone <this repo> && cd moestream
cp .env.example .env
```

Edit three lines in `.env`:

```bash
MODEL_DIR=/path/to/your/models      # a directory on the host, mounted read-only
MODEL_FILE=your-moe-model.gguf      # for a split GGUF, name the first shard
MS_PORT=8091                        # the port to publish
```

Then:

```bash
make up            # builds the image if it is missing (~25 min), then starts it
make logs          # watch it come up
```

`make up` is `docker compose up -d` with the host's GPU group IDs filled in.
Compose builds the image on the first run because none exists yet; afterwards it
reuses it. **If you change anything under `src/`, ask for a rebuild explicitly:**

```bash
docker compose build && make up      # or: docker compose up -d --build
```

That is all. An OpenAI-compatible API and llama.cpp's Web UI are on
`http://localhost:8091`. Point any OpenAI-compatible client at
`http://localhost:8091/v1`.

```bash
curl localhost:8091/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"hello"}]}'
```

`make down` stops it. `make test` runs the checks that need no hardware.

> **Why `make up` and not `docker compose up -d`?** The render and video group
> IDs are facts about your host, not choices. `make up` reads them and passes
> them in. Put them in `.env` yourself and plain `docker compose up -d` works
> too — but a stale GID there means the GPU is silently not used, and the only
> symptom is that everything is mysteriously slow.

---

## Settings

Two values in `.env` decide everything. **Both default to `learn`, so you do not
have to pick them** — the runtime measures your own traffic and records what it
found under `./state/`, per model.

**`MOESTREAM_CACHE_FRAC` — how much of the model stays in GPU memory.**
The fraction of each layer's experts kept resident; the rest are read from SSD
as needed. This alone sets the memory/speed trade-off. Measured on
Ornith-1.0-35B (256 experts per layer):

| frac | slots per layer | device memory | generation |
|---:|---:|---:|---|
| 1.00 | 256 | 17.29 GiB | 41.5 ms/tok — equivalent to not streaming at all |
| 0.25 | 64 | 7.44 GiB | 58.7 ms/tok |
| 0.15 | 38 | 4.6 GiB | 90.6 ms/tok |

**`UBATCH` — how many prompt tokens are processed at once.**
Larger chunks re-read each expert fewer times, but per-token compute grows with
chunk size. Those two balance at a peak that sits in a different place for every
model swept so far — Ornith 1024, Qwen3-Coder 4096, Laguna 8096 — so there is no
value worth copying from someone else. (gpt-oss-120b was run at a fixed 1024 to
get it working; its peak has not been swept.)

**What `learn` does.**

| | What it learns | How |
|---|---|---|
| `MOESTREAM_CACHE_FRAC=learn` | the slot count | records reuse distance, which yields the hit rate at *every* possible size from one session — so it needs no restarts to compare |
| `UBATCH=learn` | the micro-batch | measures one candidate per start: per-token compute grows super-linearly, so a single run cannot be extrapolated from |

The two interact — a larger pool misses less, which weakens the reason to use a
large ubatch — so each ubatch measurement records the frac it was taken at, and
old measurements stop counting when frac moves.
Details: [`docs/USAGE.md`](docs/USAGE.md) and the comments in `.env.example`.

Everything else is derived: thread count, batch size, arena size, the sampling
defaults from the GGUF's own metadata. The prefill staging arena, reading only
the experts a batch actually references, and prefetch overlapped with GPU compute
are all on by default and need no configuration.

### Which llama.cpp

MoEStream is **not a fork**. It is a small set of patches applied to an
unmodified llama.cpp checkout at build time — 4 files, 5 blocks. You keep
upstream llama.cpp; you keep its Web UI, its server, its model support, its
future fixes. This project adds one thing to it and takes nothing away.

The Dockerfile pins a commit, which is the version everything in this README was
**verified against**:

| | |
|---|---|
| Verified commit | `3581ba0c` (build `b10230`, 2026-08-02) |
| Backend | Vulkan (`GGML_VULKAN=ON`) |

**The pin is a statement about what was measured, not a limit on what works.**
The patches attach to a handful of places in `llama-graph.cpp`,
`llama-model-loader.cpp` and `ggml-vulkan.cpp`; as long as upstream does not
restructure those, newer llama.cpp builds fine. To use one:

```bash
LLAMA_COMMIT=master        # in .env
docker compose build --no-cache && make up
```

If upstream *has* moved the code a patch attaches to, the build stops and names
the anchor that moved. It will not produce a silently wrong binary. Fixing it
means updating that one anchor in `src/apply.py`.

Speed may differ from the numbers here — those hold for the verified commit. A
scheduled CI job builds against llama.cpp `master` weekly
([`.github/workflows/upstream.yml`](.github/workflows/upstream.yml)), so
breakage shows up here before it shows up for you.

---

## Which should you use?

| Your situation | Use |
|---|---|
| The model fits in your GPU memory | **plain llama.cpp** — it will be faster, and MoEStream has nothing to add |
| The model fits, but leaves nothing for anything else | **MoEStream** — a 35B model in 7.4 GiB instead of 17.3 GiB |
| **The model does not fit at all** | **MoEStream** — this is the case it exists for |

The original goal was the middle row: turning *"runs, but takes the whole
machine"* into *"runs, and leaves room for everything else"*. In practice the
bottom row turned out to matter just as much. On a machine with ~24 GB reachable
by the GPU, three of the four models tested **do not start** under plain
llama.cpp. Under MoEStream they run.

Is 3.9 tok/s slow? Yes. It is also the difference between having a 120B model
and not having one. For a chat you follow along with, or an agent that works
while you do something else, it is enough. For anything interactive and
impatient, it is not — and no amount of tuning will change that, because the
weights genuinely have to come off the SSD.

---

## What makes this different from other SSD-streaming work

As far as I have been able to determine, the published work in this space
(Klotski, ProMoE, MoE-Infinity) is aimed at considerably larger models than
these, and at systems with discrete GPUs. At that end the arithmetic is
different: a 700B-class model can need on the order of 11 GB read per token, and
there SSD bandwidth genuinely is the wall — so predicting which experts are
coming and fetching them early is the reasonable answer.

At 35–120B on unified memory, a token needs around 455 MiB. **The wall is
somewhere else**, and several conclusions invert. Measured here:

- **Every predictive prefetch scheme measured here was net-negative.** A
  layer-lookahead predictor reached 81.4% accuracy and still lost, because
  reading the state it needs off the GPU costs more than the I/O it hides.
- **`io_uring` beat parallel `pread` by 1.5%.** The 4.48 GB/s ceiling was the
  device, not the API.
- **A static hot set loses to plain LRU** by 8.1 points of hit rate.
- **Streaming itself is lossless.** With constant misses and with no evictions
  possible, perplexity is identical to four decimals.

None of this contradicts that work. Those results hold at the scale and on the
hardware they were measured on; they stop holding here, which is a statement
about the regime, not about them. The full record — including the measurements
that turned out to be wrong, and how that was caught — is in
[`docs/RESULTS.md`](docs/RESULTS.md).

---

## Repository layout

```
Dockerfile        the image: fetches llama.cpp, patches it, builds it
compose.yaml      how it runs
.env.example      the settings a human chooses
Makefile          up / down / logs / test / bench

src/              the product. 3178 lines, and that is all of it.
  llama-moestream.{h,cpp}   the runtime: slots, cache, reads, remap   2498
  expert_cache.{hpp,cpp}    S3-FIFO cache, one per layer                425
  apply.py                  where it attaches to llama.cpp              255
  entrypoint.sh             derives threads, batch size, GPU checks

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
on faith. `.dockerignore` and the `Dockerfile` between them mean only `src/`
reaches the image.

There is no llama.cpp clone in this repository. The build fetches it.

## Documentation

| Document | Contents |
|---|---|
| **[`docs/HOW-IT-WORKS.md`](docs/HOW-IT-WORKS.md)** | **Technical explanation, no prior knowledge assumed** — what was solved, why that way, and what failed |
| **[`docs/RESULTS.md`](docs/RESULTS.md)** | **Full measurements** — speed, memory, quality, failed approaches, and mistaken measurements |
| [`docs/DESIGN.md`](docs/DESIGN.md) | Design document (35 chapters plus appendices, 32 ADRs), including the 12 decisions revised or rejected by measurement |
| [`docs/USAGE.md`](docs/USAGE.md) | Operating procedures |
| `docs/findings/` | Primary sources per experiment (S0 / S0b / S1 / S2 / S5–S14 / M0-2 / N1–N4 / V2) |

### Core of the implementation

- **§11.3 Slot Table and Slab / §10.4 ID Remap** — mapping the router's `expert_id` to a
  `slot_id` is enough to run MoE on a dynamic cache **without changing a single
  line of ggml's `mul_mat_id`**. Bit-identical results confirmed on three paths:
  CPU, Vulkan and a real GGUF.
- **Zero-copy** — on UMA machines ggml-vulkan already allocates with
  `eDeviceLocal|eHostVisible|eHostCoherent`, so `pread` can go straight from SSD
  into GPU-visible memory (100% of reads take this path, −5.2%).
- **S3-FIFO cache** — independent per layer. Measured hit rate 82.4–91.5% at a
  38% cache ratio.

### Exactly how it couples to llama.cpp

`src/apply.py` edits **4 files, in 5 blocks**, against an unmodified checkout:

| File | What it does |
|---|---|
| `src/CMakeLists.txt` | add two source files to the build |
| `src/llama-model-loader.cpp` | allocate expert tensors at slot-pool size, and skip loading their contents |
| `src/llama-graph.cpp` | insert the id-remap node and hand its output to `mul_mat_id` |
| `ggml/src/ggml-vulkan/ggml-vulkan.cpp` | expose the host pointer of a device buffer, and let host memory be imported as one (2 blocks) |

At run time it also depends on GGUF tensor naming and a few internal members of
ggml-vulkan.

Every edit is guarded by a named `assert` on its anchor text. If upstream moves the code an
edit attaches to, **the build stops and names it** — it cannot produce a
half-patched binary. `research/tests/test_apply_patch.py` verifies that guard by
deleting each anchor in turn and checking the build refuses.

### Written down honestly

- **Expert Sweep does not work** — the headline feature of the design. It
  achieved 2.9x prefill and broke PPL to 520801. The cause was traced to graph
  buffer aliasing in ggml, and it is disabled by default (`docs/RESULTS.md` §9).
- **12 design decisions rejected by measurement** (`docs/RESULTS.md` §11)
- **A record of our own mistaken measurements** — such as reading page-cache
  throughput as real bandwidth (`docs/RESULTS.md` §13.3), and an "optimization"
  that made a broken build look faster (§10.8)
- **Prefetching was implemented and it failed** — even where I/O accounted for
  75% of the time, the `fadvise` approach was 19% worse (`docs/findings/S11-*`)

---

## Status

Initial public release. The runtime works and is measured; what is missing is
other people's hardware. If you run it on something that is not an AMD iGPU —
NVIDIA, Intel Arc, Apple Silicon, a discrete card — the result is interesting
either way, and an issue saying "it worked" is as useful as one saying it did
not.

Known gaps, stated plainly:

- **Verified on one machine.** One CPU, one GPU, one SSD.
- **Expert Sweep, the design's original headline feature, does not work.** It
  reached 2.9x prefill and broke perplexity to 520801. Disabled by default; the
  cause is traced and written up ([`docs/RESULTS.md` §9](docs/RESULTS.md)).
- **`make test` covers logic, not performance or output correctness.** Those
  need a real machine and real models; the procedure is in
  [`docs/RESULTS.md` §12](docs/RESULTS.md).

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
conversions the four tested models came from.
