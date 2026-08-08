# Finding S8 — Supporting sharded GGUFs and a leading dense layer (found on Laguna-S-2.1)

| | |
|---|---|
| Target | model generality. `src/llama-moestream.cpp` |
| Date | 2026-08-06 |
| Verdict | **two bugs identified and fixed. Verified working on a 54.7 GiB, 3-shard model** |
| Trigger | running an unsupported architecture, `laguna` (a 3-shard GGUF), on real hardware |

## Background

MoEStream **reads experts from the GGUF itself**, bypassing llama.cpp's loader.
That means it has to handle the model differences llama.cpp would otherwise
absorb. Two defects only surfaced when a new model was run.

## Bug 1: reading from the wrong file in a sharded GGUF

### Why this was dangerous

llama.cpp's `llama_tensor_weight` carries two things:

```cpp
struct llama_tensor_weight {
    uint16_t  idx;   // source file index
    size_t   offs;   // tensor data offset in the original file
```

`offs` is **an offset within that shard**. MoEStream ignored `idx` and read from
`offs` in the single file named by `MOESTREAM_GGUF`.

Laguna-S-2.1's layout:

| Shard | Size | Tensors | expert (gate) |
|---|---:|---:|---:|
| 1 | 3.51 MiB | **0** | 0 |
| 2 | 46.22 GiB | 704 | 40 layers |
| 3 | 8.49 GiB | 110 | 7 layers |

**Shard 1 is metadata only.** Every expert lives at `idx ≥ 1`. Before the fix,
all experts were read from shard 1 (3.51 MiB), so:

- out of range → `pread` returns 0 and **the expert stays zero** (its
  contribution vanishes)
- in range → **unrelated bytes are used as an expert**

Either way it **breaks quietly while producing plausible output** — a failure
mode this project had already hit twice (N4 and S7's bug 2).

### The fix

1. Add `file_idx` to `SlabInfo`, taken by `register_slab` (the llama.cpp side is
   one line in `apply.py`, passing `w->idx`)
2. `shard_paths()` parses the `-00001-of-00003.gguf` form and enumerates every
   file
3. `finalize()` opens all shards. **If even one is missing, MoEStream disables
   itself**
4. `do_read()` reads from `g_fds[file_idx]`
5. `gguf_topk()` and `gguf_bytes_per_slot()` scan all shards too

### ★ Position verification at startup

To prevent silent breakage, `finalize()` verifies every expert tensor:

```
file_offset + expert_bytes × n_expert  <=  that shard's actual size
```

If any fails, it prints "expert position verification failed; disabling, because
that is safer than reading the wrong data" and turns MoEStream off. A mistaken
`idx` is always caught here.

## Bug 2: statistics stopped entirely on a model with a leading dense layer

Laguna **has no experts in layer 0** (the first layer is dense); experts are in
layers 1–47.

MoEStream detected the statistics boundary with `il == 0`, so on this model none
of the following worked:

- `g_tokens` (the token counter) never incremented
- no `[stats]` hit-rate lines
- no periodic `[mrc]` / `[ub]` output
- `SIGUSR1` on-demand output did not respond

**Inference itself was fine**, so only the measurement machinery died quietly.

### The fix

Use `g_layers.begin()->first` (the lowest layer that has experts) as the
reference, and say so at startup:

```
moestream: first 1 layer is dense (no experts). Expert layers are 1-47
```

MoE models with a few leading dense layers are not unusual (the DeepSeek-V2/V3
family, for instance), so this is a generality problem, not a Laguna quirk.

## Verified on hardware

```
moestream: top_k = 10 (from GGUF metadata)
moestream: first 1 layer is dense (no experts). Expert layers are 1-47
moestream: GGUF = /models/Laguna-S-2.1-UD-IQ4_NL-00001-of-00003.gguf (3 shards)
moestream: 48 layers x 256 experts -> 51 slots/layer (20%)
moestream: prefill arena ENABLED (1 x 1494 MiB, threshold 3 tokens, single)
```

Output normal (`"The capital of France is"` → `" Paris."`), position verification
passed. **A configuration that would have been 100% broken before the fix.**

## What Laguna-S-2.1 can do (for reference)

| Item | Value |
|---|---|
| model | Laguna-S-2.1-UD-IQ4_NL (3 shards, 54.7 GiB total) |
| structure | 48 layers / 256 experts / top-10 / first layer dense |
| one layer's experts | 1494 MiB (largest layer) |

At `MOESTREAM_CACHE_FRAC=0.20` (51 slots):

| UBATCH | Memory | prefill | decode |
|---:|---:|---:|---:|
| 1024 | 16.60 GiB | 38.2 tok/s | 418 ms/tok |
| **8192** | **20.05 GiB** | **83.8 tok/s** | 440 ms/tok (2.27 tok/s) |

**UBATCH=8192 makes prefill 2.2x faster** for +3.45 GiB. The automatic `[ub]`
recommendation (8192, predicted 77.8 tok/s) matched the measured 83.8 tok/s,
**7% conservative**.

### Too large for this machine

The MRC shows why (computed from 283,420 reuse distances on a real workload):

| Slots | Share | Hit rate | Memory |
|---:|---:|---:|---:|
| 128 | 50% | 90.15% | 25.35 GiB |
| 77 | 30% | 77.86% | 15.25 GiB |
| **51** | **20%** | **66.47%** | **10.10 GiB** |
| 38 | 15% | 58.78% | 7.53 GiB |

A decent hit rate (90%) needs **25.35 GiB** resident, and this machine's GTT is
24 GiB. It settles at 66% and 2.27 tok/s decode instead.

**This exceeds MoEStream's intended range (30–70B class, 15–35 GiB models).**
Running this class at a practical speed would need stronger quantization
alongside it.

## Cross-model comparison (same machine)

| Model | Size | Layers | Experts | top_k | Sharded | Leading dense |
|---|---:|---:|---:|---:|---|---|
| Ornith-1.0-35B | 16.87 GiB | 40 | 256 | 8 | — | — |
| Qwen3-Coder-Next | 36.54 GiB | 48 | 512 | 10 | — | — |
| Laguna-S-2.1 | 54.7 GiB | 48 | 256 | 10 | **3 shards** | **1 layer** |

All three run **without one line of architecture-specific branching**. The only
dependency is llama.cpp's tensor naming convention
(`blk.<il>.ffn_{gate,up,down}_exps.weight`); everything else — top_k, layer
count, expert count, sharding, leading dense layers — is read from the GGUF
automatically.

## Lesson

> **Every new model that runs breaks one more assumption.**

This time it was "a GGUF is one file" and "layer 0 is MoE". Both held for the
first three models.

And both were the kind of defect where **inference keeps running**. Bug 1 read
the wrong experts; bug 2 killed only the measurements.
**Appearing to work is not evidence of being correct.**
