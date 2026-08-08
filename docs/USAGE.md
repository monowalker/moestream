# MoEStream — Operating Guide

> **Note**: this is an experimental implementation. Read [Limitations](#5-limitations)
> before putting it in front of anything that matters.

## What this is

A small set of patches to llama.cpp — 4 files, 5 blocks — that stops MoE expert
weights from having to be resident in memory and **streams them from an SSD
instead**. llama.cpp's CLI, server and OpenAI-compatible API all work as they
did.

```
Ornith-1.0-35B-UD-IQ4_NL (16.87 GiB)              (measured 2026-08-06)

  plain llama.cpp   device memory 17.29 GiB    24.1 tok/s
  MoEStream 0.25    device memory  7.44 GiB    17.0 tok/s   (-57% / 71%)
  MoEStream 0.15    device memory  4.63 GiB    11.0 tok/s   (-73% / 46%)

Models that plain llama.cpp cannot start at all here (GTT limit 23.5 GiB):

  Qwen3-Coder-Next (36.5 GiB)  ->  13.15 GiB /  9.8 tok/s
  Laguna-S-2.1     (54.7 GiB)  ->  18.51 GiB /  3.1 tok/s
  gpt-oss-120b     (58.5 GiB)  ->  14.48 GiB /  3.9 tok/s
```

---

## 1. Starting it

```bash
cp .env.example .env      # first time only; edit MODEL_DIR / MODEL_FILE / MS_PORT
make up                   # first run builds the image (~25 min), then starts it
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

```
moestream: enabled (cache_frac=0.25)
moestream: GGUF = /models/Ornith-1.0-35B-UD-IQ4_NL.gguf
moestream: 40 layers x 256 experts -> 64 slots/layer (25%)
```

**If they are absent, you are running plain llama.cpp** and memory will not have
gone down.

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
| `1.00` | 256 | 17.29 GiB | 24.1 tok/s | effectively no streaming |
| `0.50` | 128 | ~11 GiB | ~19 tok/s | speed matters more than memory |
| **`0.25`** | **64** | **7.44 GiB** | **17.0 tok/s** | a good default |
| `0.15` | 38 | 4.63 GiB | 11.0 tok/s | memory matters most |
| `0.08` | 20 | ~3.4 GiB | ~8 tok/s | minimum viable |

Between `1.00` and `0.25` you **give up 57% of the memory and keep 71% of the
speed**. Below `0.15` the curve steepens, so if you need to cut, try `0.25`
first.

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

The current default (frac=0.25) measures **7.44 GiB / 242.5 tok/s**
(`docs/RESULTS.md` §12.5). It is faster than the table because reading only the
referenced experts (S12) and asynchronous prefetch (S14) landed afterwards; the
relative shape across `UBATCH` is unchanged.

Output is bit-identical under greedy decoding to plain llama.cpp with every
expert resident.

---

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
# 4. Write the value into .env and restart. It is never written for you.
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

## 5. Limitations

### Things you must get right

| # | Requirement | Why |
|---|---|---|
| 1 | **Keep the GGUF on a bind mount**, not inside the image | read performance drops on overlayfs |
| 2 | `--device /dev/dri` and `--group-add` are required | without them it falls back to the CPU backend |
| 3 | Set `MODEL_FILE` correctly in `.env` | compose builds `MOESTREAM_GGUF` from it |
| 4 | Do not run other GPU workloads alongside it *while measuring* | they contend for GTT and halve the numbers |

### Known constraints

- **MoE models only.** On a dense model nothing happens; it runs normally, with
  no benefit.
- **Verified on Vulkan.** CUDA, Metal and ROCm are untested.
- **One model per process.** Sharing a process between models will cross state.
- **No predictive prefetch.** All three schemes measured were net-negative
  (findings N2/N3, `docs/RESULTS.md` §7).
- **Decode right after prompt processing starts cold.** Prompt processing goes
  through the arena, so the S3-FIFO pool is not warmed by it and the first few
  dozen tokens are all misses (measured: 76.6 → 50.7 ms/tok as it converges).
  A known area for improvement (finding S7).
- **Concurrent requests are unverified.** The cache is single-threaded and holds
  no locks. Use `llama-server`'s default `-np 1`.

### Before you rely on it

This started as a spike. The following have not been done:

- long-running stability (memory leaks, fd leaks)
- correctness under concurrent access (data races)
- recovery from I/O errors
- quantitative output-quality evaluation beyond perplexity
  (4.4400 → 4.4494 measured; the arena path is verified by greedy bit-identity,
  its perplexity is not separately measured)

**If you are going to run it, check the output by eye on a short task first.**
Four bugs of the "uninitialised slot produces nonsense" and "refcount leak stops
eviction" kind were found and fixed during development. There is no reason to
assume none remain.

---

## 6. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| no `moestream:` lines in the log | `MOESTREAM=1` is not reaching the container |
| `GGUF path unknown` | set `MOESTREAM_GGUF` |
| `cannot open GGUF` | check the path *inside* the container (the `-v` target) |
| output is nonsense | streaming is broken. Compare against `MOESTREAM=0`; if that fixes it, please report it |
| memory did not go down | check the log for "N slots/layer" |
| half the expected speed | is something else using the GPU? (`docker ps`) |
| decode very slow | check `MOESTREAM_ZEROCOPY=1` took effect in the startup log |
| prompt processing slow | check for `prefill arena ENABLED` in the startup log |

### Confirming it is doing anything

```bash
# compare GPU memory before and after
cat /sys/class/drm/card1/device/mem_info_gtt_used
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
