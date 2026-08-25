# MoEStream — Operating Guide

> **Note**: this is an experimental implementation. Read [Limitations](#5-limitations)
> before putting it in front of anything that matters.

## What this is

A small set of patches to llama.cpp — 4 files, 5 blocks — that stops model
weights from having to be resident in memory and **streams them from an SSD
instead**. On an MoE model it streams the experts; on a dense model it streams
the feed-forward weights. llama.cpp's CLI, server and OpenAI-compatible API all
work as they did.

```
Ornith-1.0-35B-UD-IQ4_NL (16.87 GiB)              (measured 2026-08-25)

  plain llama.cpp   device memory 17.77 GiB    23.4 tok/s
  MoEStream 0.25    device memory  7.93 GiB    16.5 tok/s   (-55% / -29%)

Models that plain llama.cpp cannot start at all here (GTT limit 23.5 GiB):

  Qwen3-Coder-Next (36.5 GiB)  ->  13.44 GiB / 14.8 tok/s
  Laguna-S-2.1     (54.7 GiB)  ->  18.79 GiB /  3.1 tok/s
  gpt-oss-120b     (58.5 GiB)  ->  14.91 GiB /  3.8 tok/s

Dense models, measured 2026-08-25 (Qwen3.8-27B-IQ4_NL, one request at a time):

  plain llama.cpp   device memory 16.49 GiB    4.69 tok/s
  MoEStream (FFN)   device memory  7.81 GiB    1.28 tok/s   (-53% / -73%)
```

Dense streaming frees more memory than MoE streaming does, and costs far more
speed for it. That is not an implementation weakness: a dense pass reads the
same bytes whatever it is asked, so there is no reuse to exploit. It becomes
cheap when several requests share a pass — 1.2x at 16 concurrent instead of 3x
at one — which is why the launcher asks about concurrency. See
[S37](findings/S37-batch-freezone.md).

---

## 1. Starting it

```bash
cp .env.example .env      # first time only
$EDITOR .env              # set one line: MODEL_DIR=/path/to/your/models
make launch               # first run builds the image (~25 min), then starts it
```

`make launch` is the normal way in. It reads every GGUF header in `MODEL_DIR`
and this machine's memory limits, then asks only what it cannot work out:

| it asks | because |
|---|---|
| which model | it cannot know which one you want |
| one request at a time, or several | a property of your workload, not the machine. On a dense model it is the single biggest speed lever |
| context length | it suggests the largest that fits, computed from the model's real attention geometry |
| memory or speed, **only when the model would fit whole** | genuinely a preference. Streaming is the default |

Everything else — MoE or dense, slot count, arena size and threshold, read
threads, micro-batch, sampling defaults, GPU group IDs — is derived at run time
and has no setting. What was chosen is printed at startup.

A full session looks like this:

```
  MoEStream launcher
  ─────────────────────────────────────────────────────────────
  GPU-reachable memory  24.0 GiB      host RAM  30.6 GiB

   #  model                                  size   type          fits?
   2  Ornith-1.0-35B-UD-IQ4_NL.gguf       16.87G   MoE 256e      yes
   4  Qwen3-Coder-Next-UD-IQ4_NL.gguf     36.54G   MoE 512e      NO — streaming is the point

  model number (1-9, or q to quit): 2
  How will you use it?
    1  chat / agent, one request at a time     (latency matters)
  choice [1]: 1
  context length [32768]:

  what this implies
  ─────────────────────────────────────────────────────────────
  weights on disk         16.87 GiB
  KV cache                 0.40 GiB   at ctx 32768 x 1
  cannot be streamed       4.37 GiB   (attention and embeddings)
  all resident would be   18.76 GiB of 24.0 -- it fits either way

  This model would also fit whole, so you get to choose what to spend.
    1  stream, freeing about 6 GiB for everything else   (about 1.3x slower)
    2  keep it all resident                             (fastest, uses the memory)
  choice [1]: 1

  start? [Y/n]: y
  loading the model....
  ready:  http://localhost:8080
```

It writes its answers to `.env.launcher`, which is layered over `.env` — so
`make up` afterwards reuses the same choices, and `.env` keeps whatever you set
by hand underneath.

### Or set it up by hand

```bash
$EDITOR .env              # MODEL_DIR, MODEL_FILE, MS_PORT, and anything in
                          # section 2 you want to pin
make up                   # docker compose up -d with the GPU group IDs filled in
make logs                 # watch it come up
```

- **Web UI** : http://localhost:8091
- **API**    : http://localhost:8091/v1 (OpenAI-compatible)

`make up` is `docker compose up -d` with the host's render and video group IDs
filled in. Those are facts about your machine, not choices, and a stale value in
`.env` means the GPU is silently unused — the only symptom being that everything
is mysteriously slow.

Stopping and cleaning up:

```bash
make down                         # stop and remove the container
docker compose down --rmi local   # also remove the image
```

Changing a setting in `.env` takes effect on the next `make up`. **Changing code
under `src/` does not** — Compose reuses the existing image unless you ask for a
rebuild:

```bash
docker compose build && make up   # or: docker compose up -d --build
```

## 2. Checking that streaming is actually on

These lines in `make logs` mean it is working:

On an MoE model:

```
moestream: enabled (cache_frac=learn)
moestream: [learn] using frac=0.50 from a previous run
moestream: GGUF = /models/Ornith-1.0-35B-UD-IQ4_NL.gguf
moestream: 40 layers x 256 experts -> 128 slots/layer (50%)
```

On a dense model:

```
moestream: [dense] 65 layers, keeping 0 resident, streaming 0..64 (frac=0.00)
moestream: [dense] arena ENABLED (2 x 143 MiB); streaming 8.96 GiB of FFN
                   across 64 layers, 8.96 GiB/token
```

On a dense model that fits, streaming correctly does nothing, and says so:

```
moestream: [dense] auto: model 15.22 GiB (FFN 9.10 + other 6.11), device 21.8 GiB free of 24.0
moestream: [dense] auto: the model fits; streaming nothing (identical to plain llama.cpp)
```

**If none of these appear, you are running plain llama.cpp** and memory will not
have gone down. The most common cause is `MOESTREAM=0` left in `.env.launcher`
by choosing speed over memory in the launcher.

## 3. Using it

The Web UI is just http://localhost:8091 in a browser.

Python (OpenAI SDK):

```python
from openai import OpenAI
c = OpenAI(base_url="http://localhost:8091/v1", api_key="dummy")
r = c.chat.completions.create(model="x", messages=[{"role":"user","content":"hello"}])
print(r.choices[0].message.content)
```

curl:

```bash
curl -s http://localhost:8091/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"x","messages":[{"role":"user","content":"hello"}],"max_tokens":200}'
```

## 4. Settings (`.env`)

| Variable | Default | What it does |
|---|---|---|
| `MOESTREAM` | `1` | `0` runs as plain llama.cpp, for A/B comparison |
| `MOESTREAM_CACHE_FRAC` | `learn` | fraction of experts kept resident. **The main memory/speed knob** |
| `UBATCH` | `learn` | micro-batch size. **The main prompt-processing knob** |
| `MODEL_DIR` / `MODEL_FILE` | — | where the model is. `MODEL_DIR` is bind-mounted read-only |
| `MS_PORT` | `8091` | published port. Named `MS_PORT` because `PORT` collides with a common host variable |
| `CTX_SIZE` | `32768` | context length. KV cache memory scales with it |
| `CACHE_RAM` | `2048` | prompt cache in host RAM (MiB). **This dominates how responsive it feels** |
| `N_PARALLEL` | `1` | concurrent requests |
| `RENDER_GID` / `VIDEO_GID` | detected | GPU access. `make up` reads them; set them only if you run compose directly |

Everything else is derived at run time — thread count, batch size, arena size,
the arena threshold, sampling defaults from the GGUF's own metadata. See
`.env.example` for the full list, including the diagnostic variables that exist
for investigating rather than tuning.

### Choosing `MOESTREAM_CACHE_FRAC`

| Value | Slots | Device memory | Speed | Use when |
|---:|---:|---:|---:|---|
| `MOESTREAM=0` | — | 17.78 GiB | 23.5 tok/s | no streaming at all |
| `0.60` | 154 | 13.03 GiB | 20.2 tok/s | speed matters more than memory |
| `0.50` | 128 | 11.56 GiB | 19.2 tok/s | |
| `0.40` | 102 | 10.09 GiB | 17.4 tok/s | |
| **`0.25`** | **64** | **7.93 GiB** | **16.5 tok/s** | a good default |
| `0.15` | 38 | 6.46 GiB | 14.3 tok/s | memory matters most |

Between no streaming and `0.25` you **give up 55% of the memory and keep 71% of
the speed**. The curve is monotonic, and memory buys speed at a steeply
diminishing rate: `0.15` → `0.25` costs 1.5 GiB and buys 2.2 tok/s, while `0.50`
→ `0.60` costs the same 1.5 GiB and buys 1.0
([`findings/S41`](findings/S41-frac-curve.md)).

**Rule of thumb**: `device memory ≈ 2.4 GiB of non-expert weights + 14.5 GiB × frac`
(the second term is this model's total expert bytes; substitute your own).

### The prefill staging arena

On by default. During prompt processing it uses an arena holding **one layer's
full expert set**, rotated layer by layer, instead of the resident slot pool.
The pool does not grow.

Measured on Ornith-35B (a sweep taken at frac=0.38; read it for the relative
shape, not the absolute numbers):

| | Memory | Prompt processing (13877 tokens) |
|---|---:|---:|
| disabled | 8.22 GiB | 46.0 tok/s |
| enabled, `UBATCH=512` | 8.84 GiB | 140.3 tok/s |
| **enabled, `UBATCH=1024`** | **8.98 GiB** | **194.2 tok/s** |
| enabled, `UBATCH=2048` | 9.26 GiB | 199.3 tok/s (only +5%) |

The current default (frac=0.25) measures **7.93 GiB / 239.4 tok/s**
(`docs/RESULTS.md` §12.4b). It is faster than the table because reading only the
referenced experts (S12) and asynchronous prefetch (S14) landed afterwards; the
relative shape across `UBATCH` is unchanged.

Output is bit-identical under greedy decoding to plain llama.cpp with every
expert resident.

---

### What is derived, and therefore has no setting

Leaving a value here that the runtime can work out is how one model's settings
get silently applied to the next one. So these are not settable:

| | how it is decided |
|---|---|
| MoE or dense path | the GGUF's expert count |
| `top_k` | GGUF metadata |
| slab slot count | `MOESTREAM_CACHE_FRAC` × the model's expert count |
| arena size and switch threshold | one layer's expert bytes, and the largest micro-batch the slab can serve |
| thread count | the host's physical cores (not `nproc` — counting SMT siblings is measurably slower) |
| dense read threads | fixed at 4. An auto-tuner picked 8 by read bandwidth and was 4.4% slower end to end, so it was removed (`docs/RESULTS.md` §10.12) |
| `BATCH` | raised to at least `UBATCH` |
| sampling defaults | the model's own values from the GGUF |
| render / video group IDs | detected by `make up` and `make launch` |

Whatever was chosen is printed at startup:

```bash
docker logs moestream 2>&1 | grep -a moestream
```

### How the learned settings are saved

Everything `learn` measures is written **when the server exits cleanly**, from an
`atexit` handler. `make down` and `docker compose down` send SIGTERM, so they
save it. **`docker kill` and `docker rm -f` do not** — the process dies before the
handler runs and that session's measurements are lost.

This matters more than it sounds. A learn candidate is only tried once per start;
if the row is never written, the next start measures the same candidate again and
the search never advances. If you are scripting restarts, stop the container, do
not kill it.

```bash
make down                 # SIGTERM — saves
docker compose down       # SIGTERM — saves
docker stop moestream     # SIGTERM — saves
docker rm -f moestream    # SIGKILL — loses this session's learning
```

The files live in `./state/`, one row per (model, configuration, candidate):

| file | what it holds |
|---|---|
| `tuning.tsv` | the slot count `MOESTREAM_CACHE_FRAC=learn` settled on |
| `ubatch.tsv` | prefill rate per micro-batch candidate |
| `spec.tsv` | generation rate per speculative draft size, keyed by concurrency too |

Deleting a file makes that dimension start over. Deleting `./state/` entirely
resets everything, which is the right thing to do after changing hardware.

## 4.4 Dense models

MoEStream reads the GGUF's expert count and takes the dense path on its own;
there is nothing to switch on. `MOESTREAM_DENSE_FRAC` controls how much of the
feed-forward weight is streamed:

| value | meaning |
|---|---|
| `auto` (default) | stream the least that still lets the model fit. A model that already fits streams **nothing** and behaves exactly like plain llama.cpp |
| `0.00` | stream the whole FFN — the most memory freed, the highest cost |
| `0.40` | keep 40% of the layers resident |
| `off` | never stream a dense model |
| `learn` | as `auto`, and report the measured cost per GiB on this machine |

`learn` here is **not** the expert cache's kind of learning. An expert cache has
a knee to find, because experts are reused; a dense pass reads the same bytes
every time, so the cost is linear in the bytes moved out and there is no optimum
to search for. What you choose is a point on a straight line, and the launcher
puts that choice to you directly.

Measured on `Qwen3.8-27B-IQ4_NL`, one request at a time:

| `MOESTREAM_DENSE_FRAC` | device memory | generation |
|---|---|---|
| `auto` (fits, streams nothing) | 16.18 GiB | 208.1 ms/token |
| `0.40` | 10.53 GiB | 512.5 ms/token |
| `0.00` | 7.50 GiB | 647.4 ms/token |

Two things make dense streaming far cheaper than that table suggests:

- **Concurrency.** One pass serves every request in the batch, and it reads the
  same bytes for one token as for sixteen. The cost falls from ~3x at one
  request to ~1.2x at sixteen ([S37](findings/S37-batch-freezone.md)).
- **Prompt processing is free** above `UBATCH=1024`, at any frac
  ([S27](findings/S27-dense-streaming-impl.md)).

Perplexity is unchanged to four decimal places. Nothing here trades accuracy.

## 4.5 Deciding `MOESTREAM_CACHE_FRAC` for your own setup

The 0.25 above was **measured on Ornith-35B**. A different model, or different
traffic, has a different optimum. There is machinery for finding yours.

### The easy way: let it learn

In `.env`:

```bash
MOESTREAM_CACHE_FRAC=learn
```

The first run starts conservatively at 0.15 and records what it measures into
`./state/tuning.tsv`. The next start applies it. No log-reading, no editing.

```
run 1: [learn] no frac recorded; starting at 0.15 (low on purpose: too high and the server may not start)
       [mrc] recommended 128 slots (0.50) -- below this, hit-rate loss exceeds 2.5 pt/GiB
       [mrc]   device memory 5.9 / 24.0 GiB used; at most 256 slots fit
       [learn] frac=0.50 recorded; the next start uses it

run 2: [learn] using frac=0.50 from a previous run
       40 layers x 256 experts -> 128 slots/layer (50%)
```

It needs about **625 generated tokens** before it can decide. The measurement
counts decode accesses only — prompt processing goes through the arena and never
touches the slot pool — and 200,000 samples are needed for the curve. Until then
it runs at 0.15: slower, but certain to start. Ordinary use gets there in a few
exchanges.

The recommendation is **capped by measured device memory**, because hit rate
alone will happily suggest a slot count that cannot be allocated:

```
[mrc]   ** capped by memory: 154 slots would be better for hit rate,
[mrc]      but would not fit. Recommending 96. **
```

How much headroom to leave is `MOESTREAM_MEM_RESERVE_GIB` (default 1.0 GiB).

**Why this is worth learning.** The best slot count moves with the *workload*,
not just the model. The same Ornith-35B recommended 0.25 under short replies and
0.40 under long-form generation (`docs/RESULTS.md` §10.9). The default is a
starting point, not an answer.

**Swapping models is safe.** Records are keyed by model identity — file name,
expert count, and bytes per slot — so a different model never inherits the
previous one's value. It measures again from 0.15. The bytes-per-slot part
matters: two of the models tested here both have 256 experts and are told apart
only by that.

Delete `state/tuning.tsv` to start over. A number written directly into
`MOESTREAM_CACHE_FRAC` always wins over the learned value.

### `UBATCH=learn` — the second stage

Once frac has settled, `UBATCH` is next. With `UBATCH=learn`, each start
measures one candidate; after four they are all in, and the fastest is used from
then on. Records go to `./state/ubatch.tsv`.

```
start 1  UBATCH=learn: measuring 1024 at frac=0.15
         [learn] UBATCH=1024 at frac=0.15 -> 218.4 tok/s prefill
start 2  UBATCH=learn: measuring 2048 at frac=0.15   -> 237.1
start 3  UBATCH=learn: measuring 4096 at frac=0.15   -> 227.4
start 4  UBATCH=learn: measuring 8192 at frac=0.15   -> 228.2
start 5  UBATCH=learn: using 2048 at frac=0.15 (237.1 tok/s prefill)
```

Why measure rather than derive: raising `UBATCH` re-reads each expert fewer
times, but per-token compute grows because attention within a micro-batch is
quadratic. They balance at a peak, and **a single run cannot be extrapolated
from** (§10.11).

What is measured is the **per-request** rate — prompt tokens divided by prompt
processing time, the same quantity llama.cpp reports as `prompt eval`. Because
that definition does not move with the micro-batch, candidates are comparable
(§10.16 has the cross-check against llama.cpp: 0.02–0.55% apart across all four
candidates, same winner). It needs **two completed requests** per start; the
first is discarded as warm-up. Prompt length does not matter.

**Why frac first.** The two are not independent. A larger pool misses less,
which weakens the reason to use a large micro-batch, so **the best `UBATCH`
moves down as frac goes up**. Measured on Ornith: 1024 wins at frac=0.25, 2048
wins at frac=0.15 (§10.16).

Each measurement therefore records the frac it was taken at. When frac moves,
old rows stop matching and the candidates are measured again; they are not
deleted, so moving back reuses them. If you set both to `learn`, expect the
`UBATCH` search to restart while frac is still settling.

Candidates are `1024 2048 4096 8192`, overridable with `UBATCH_CANDIDATES`. The
search only moves within that set — a true optimum between 1024 and 2048 will
not be found. Finer steps would only make the search longer, so the coarseness
is deliberate.

**There is no value worth copying from someone else.** The peak sits in a
different place for every model swept so far:

| model | best `UBATCH` |
|---|---:|
| Ornith-1.0-35B | 1024 |
| Qwen3-Coder-Next | 4096 |
| Laguna-S-2.1 | 8096 |

(`gpt-oss-120b` was run at a fixed 1024 to get it working; its peak has not been
swept.)

### You do not have to sweep frac by hand

MoEStream records **reuse distance** — how many *distinct* experts were touched
between two uses of the same one. That is a property of the access pattern and
has nothing to do with how large the cache is.

Which means **one ordinary session yields the hit rate for every possible slot
count at once**, whatever frac you happened to start with. There is no need to
run at 0.25, then 0.30, then 0.35.

### Procedure

```bash
# 1. Use it normally. frac can be anything.
make up

# 2. Give it some real work (a few thousand tokens)

# 3. Read the result
make stats
```

`make stats` (which is `research/tools/ms-stats.sh`) reports:

- current settings — model, top_k, slot count, arena
- `[stats]` hit rate
- `[mrc]` the slots-to-hit-rate curve and a recommended `MOESTREAM_CACHE_FRAC`
- `[ub]` predicted prompt processing speed by `UBATCH` and a recommendation
- measured device memory

For a differently named container, `make stats C=<name>`. For raw output,
`docker logs moestream 2>&1 | grep -a '\[mrc\]'`.

Example:

```
[mrc]   slots   frac   hit rate   memory   marginal   I/O upper
[mrc]    205     40%    93.32%   13.66 GiB    1.02 pt/GiB  (<17 ms)
[mrc]    154     30%    89.84%   10.26 GiB    2.24 pt/GiB  (<29 ms)
[mrc]    128     25%    85.96%    8.53 GiB    3.27 pt/GiB  (<40 ms)  <- current
[mrc]    102     20%    80.30%    6.80 GiB    5.19 pt/GiB  (<56 ms)
[mrc] recommended 128 slots (0.25) -- below this, hit-rate loss exceeds 2.5 pt/GiB
[mrc]   to apply: set MOESTREAM_CACHE_FRAC=0.25 in .env and restart
```

```bash
# 4. Apply it. With MOESTREAM_CACHE_FRAC=learn this is written to
#    ./state/tuning.tsv for you and used on the next start; set the value
#    in .env only if you want to pin it.
make up
```

**Re-measuring is only needed when the model changes, or when how you use it
changes.**

### How to read each column

| Column | Confidence | How to use it |
|---|---|---|
| hit rate | **high** — derived exactly from reuse distance; agrees with measurement within 1 pt | trust it |
| memory | **high** — computed from actual GGUF sizes | trust it |
| marginal (pt/GiB) | medium — a difference of hit rates, so model-free | **make the decision on this** |
| I/O upper (ms) | **low** — assumes no overlap, so it overstates | order of magnitude only; do not decide on it |

> Why the I/O column is unreliable: where slots are plentiful and misses are few,
> most I/O is hidden behind compute. Measured at 195 slots, the prediction was
> 6.9 ms against an actual 0.5 ms — 93% was hidden. As misses rise the hiding
> saturates, and by 102 slots the prediction is about right. That non-linearity
> is not in the formula.

### `UBATCH` gets a recommendation the same way

The arena re-reads a layer's experts once per pass, so:

```
time per pass = I (I/O, fixed, independent of UBATCH) + C (compute, proportional to UBATCH)
prompt speed  = ub / (I + C·ub/ub0)      -> asymptotic to ub0/C
```

`I` and `C` are separable at run time, so the speed at any `UBATCH` can be
predicted. **There is no knee — it is an asymptote**, and the larger the fixed
cost `I`, the larger the `UBATCH` needed to amortise it.

| Model | I/O per pass | Recommended `UBATCH` |
|---|---:|---|
| Ornith-35B (16.87 GiB) | 0.97 s | 2048 |
| Laguna-S-2.1 (54.7 GiB) | 13.37 s | 8192 |

**`UBATCH` does not affect output quality at all** — the same numbers, grouped
differently. The only cost is memory, measured at about 0.3 MiB per unit of
`UBATCH`.

### What is decided automatically

Everything other than `MOESTREAM_CACHE_FRAC` and `UBATCH` is measured at run time.

| Item | How it is decided | What it is worth |
|---|---|---|
| **I/O thread count** | cycles {2,4,6,8,12,16} and keeps the highest measured bandwidth | Ornith picks 4, Laguna picks 16. **The optimum inverts between models** (Laguna loses 6% at a fixed 8) |
| **What the arena reads** | only the experts a batch actually references (union) | **−40 to −53%** on small prompt deltas — which is exactly what an agent sends every turn |
| **Arena read strategy** | switches on whether total expert bytes fit in page cache | fits → read all / does not fit → union. **This inverts too** |
| **Async prefetch** | reads layer L+1 while the GPU computes layer L | Ornith **+10.6%** / Laguna **+72.5%** |

`make stats` shows the chosen values and each candidate under `[io]`:

```
[io] settled on 4 threads (measured 8.62 GB/s)
[io]    2 threads:  8.29 GB/s
[io]    4 threads:  8.62 GB/s   <- selected
[io]    8 threads:  8.86 GB/s
```

The startup log names the strategy:

```
moestream: prefill async prefetch = on
moestream: [prefetch]   prefill read strategy = union (read less)
```

There are diagnostic variables to disable them (`MOESTREAM_IO_THREADS`,
`MOESTREAM_PREFILL_ASYNC=0`). You should not normally need them.

### Adjusting the threshold

```bash
MOESTREAM_PT_PER_GIB=2.5    # default: how many hit-rate points a GiB may cost
```

- want memory → around `3.5`
- want speed → around `1.5`

**There is no strong justification for 2.5 itself.** It lines up with the cliff
in the measurements, and that is all. The reliable way to decide is to measure
decode at two settings and compare — **same prompt, at least five runs, paired**.
A single measurement disappears into the noise (±20 ms, measured).

### Turning the measurement off

```bash
MOESTREAM_MRC=0
```

The overhead is about 0.3% of decode (an O(n_expert) scan per access). Leaving
it on is fine.

---

## 4.6 Speculative decoding

Off unless the launcher turns it on, and it turns it on only where measurement
says it pays.

**Whether your model can do it is llama.cpp's question, not this project's.**
Some GGUFs carry a small head that predicts the model's own next tokens; most do
not, and the same model at a different quantization may differ. Rather than keep
a copy of upstream's rule, `moestream-spec-probe` links llama.cpp's `common`
library and calls `common_speculative_types_from_gguf()`. To ask it yourself:

```bash
docker run --rm -v /path/to/models:/models:ro \
  --entrypoint /usr/local/bin/moestream-spec-probe moestream/server:local \
  /models/your-model.gguf
# prints e.g. "draft-mtp", or nothing if the model cannot self-speculate
```

**Where it pays.** Verifying several guessed tokens in one pass costs a streamed
dense model nothing extra, because a dense pass reads the same weights whatever
it carries. Measured on Qwen3.8-27B, one request at a time:

| Qwen3.8-27B, FFN streamed | generation |
|---|---:|
| no speculation | 679.5 ms/tok |
| `n_max=1` | 521.2 ms/tok |
| `n_max=3` | 308.3 ms/tok |
| **`n_max=5`** | **296.5 ms/tok (2.29x)** |
| `n_max=8` | 388.4 ms/tok — past the optimum |

**Where it also costs: concurrency.** Batching fills the same pass speculation
would widen, and they do not add. Measured on a streamed dense model, speculation
is worth 2.08x at one request and **costs 34% and 3.6 GiB at four**
([`S45`](findings/S45-speculation-and-batching.md)). `make launch` leaves it off
above two concurrent requests, and `SPEC_DECODING=learn` keys what it learns on
`N_PARALLEL` so a value found at one concurrency is never reused at another.

**Where it costs.** On an MoE model the same flag can lose badly — 65% at
llama.cpp's own defaults on the model tested here. Two things go wrong at once: a
MoE verification pass gets **3.5x** dearer as it widens from 1 to 6 tokens
(they want different experts) where a dense one stays flat, and a drafted token
costs about half a MoE forward pass against a ninth of a dense one. On a machine
where a decode pass is much more expensive both penalties shrink, which is why
this is measured rather than assumed. The server prints a notice if you pin it on
for an MoE model.

**The model-free kinds do not help.** llama.cpp's `ngram-*` types need nothing
from the model, so they were the obvious way to give every model the benefit.
Measured, they do nothing when streamed (713.8 → 713.5 ms/tok) and cost 16% on
an unstreamed model. Speculation only converts read volume into speed at an
acceptance rate high enough to pay for the verification, and an n-gram guess is
not accepted often enough. Details: [`findings/S42`](findings/S42-speculation-by-model.md).

To set it by hand, `SPEC_DECODING` is passed straight through to `llama-server`:

```bash
SPEC_DECODING=--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.7
```

## 5. Limitations

### Things you must get right

| # | Requirement | Why |
|---|---|---|
| 1 | **Keep the GGUF on a bind mount**, not inside the image | read performance drops on overlayfs |
| 2 | `--device /dev/dri` and `--group-add` are required | without them it falls back to the CPU backend |
| 3 | Set `MODEL_FILE` correctly, or let `make launch` set it | the entrypoint derives `MOESTREAM_GGUF` from it |
| 4 | Do not run other GPU workloads alongside it *while measuring* | they contend for GTT and halve the numbers |

### Known constraints

- **MoE and dense.** The expert path and the dense FFN path are chosen from the
  GGUF's expert count. Attention and SSM weights are never streamed: three
  designs for it were built and all three failed
  ([S30](findings/S30-dense-attention.md)).
- **Verified on Vulkan.** CUDA, Metal and ROCm are untested.
- **One model per process.** Sharing a process between models will cross state.
- **No predictive prefetch.** All three schemes measured were net-negative
  (findings N2/N3, `docs/RESULTS.md` §7).
- **Decode right after prompt processing starts cold.** Prompt processing goes
  through the arena, so the S3-FIFO pool is not warmed by it and the first few
  dozen tokens are all misses (measured: 76.6 → 50.7 ms/tok as it converges).
  A known area for improvement (finding S7).
- **Concurrent requests are verified on the dense path, not the expert path.**
  Dense streaming was measured to 16 concurrent requests with output and
  perplexity checked ([S37](findings/S37-batch-freezone.md)). The expert cache
  is single-threaded and holds no locks; prefer `-np 1` with an MoE model.

### Before you rely on it

This started as a spike. The following have not been done:

- long-running stability (memory leaks, fd leaks)
- correctness under concurrent access (data races)
- recovery from I/O errors
- quantitative output-quality evaluation beyond perplexity (4.4400 → 4.4494
  measured on MoE; the dense streaming path measured separately at 4.2000 both
  ways, identical to four decimals)

**If you are going to run it, check the output by eye on a short task first.**
Four bugs of the "uninitialised slot produces nonsense" and "refcount leak stops
eviction" kind were found and fixed during development. There is no reason to
assume none remain.

---

## 6. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| no `moestream:` lines in the log | `MOESTREAM=1` is not reaching the container. If you used the launcher, check `.env.launcher` — choosing speed over memory writes `MOESTREAM=0` |
| the launcher's settings seem ignored | `.env.launcher` is layered over `.env` by `compose.yaml`. If you edited `compose.yaml`, check that both `env_file` entries are still there |
| a dense model streams nothing | that is `MOESTREAM_DENSE_FRAC=auto` doing its job: the model fits. Set `0.00` to stream it anyway |
| `GGUF path unknown` | set `MOESTREAM_GGUF` |
| `cannot open GGUF` | check the path *inside* the container (the `-v` target) |
| output is nonsense | streaming is broken. Compare against `MOESTREAM=0`; if that fixes it, please report it |
| memory did not go down | check the log for "N slots/layer" |
| half the expected speed | is something else using the GPU? (`docker ps`) |
| decode very slow | check `MOESTREAM_ZEROCOPY=1` took effect in the startup log |
| prompt processing slow | check for `prefill arena ENABLED` in the startup log |

### Confirming it is doing anything

```bash
# compare GPU memory before and after. Both counters matter on an APU:
# weights land in GTT, but some allocations are VRAM.
cat /sys/class/drm/card*/device/mem_info_gtt_used
cat /sys/class/drm/card*/device/mem_info_vram_used
```

Starting once with `MOESTREAM=1` and once with `MOESTREAM=0` and comparing is
the reliable check.

---

## 7. Removing it

```bash
docker compose down --rmi local   # container and image
make clean-docker                 # the research images too
```

No toolchain and no libraries were ever installed on the host.

## 8. A note on host environment variables

With `docker compose`, **your shell's environment overrides `.env`**. Common
names like `PORT` collide easily, which is why this project uses `MS_PORT`. If
it comes up on an unexpected port, `docker compose config` shows the values
actually in effect.
