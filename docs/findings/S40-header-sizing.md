# S40 — What a GGUF header can and cannot tell you about memory

*2026-08-23. Measured on the launcher's own estimates against real starts.*

## Why this exists

The launcher's job is to ask the user as little as possible, which means it has
to work out the rest from the model file. The one number it cannot avoid
computing is the KV cache: it decides the context length it offers, and getting
it wrong is not a cosmetic error in either direction. Too high and the launcher
refuses a context that would have run; too low and it recommends a start that
dies.

The obvious formula — `layers × kv_heads × (key_len + value_len) × ctx` — is
wrong on three of the four model families in the test set, and wrong by a lot.

## What the obvious formula misses

**Per-layer GQA.** `attention.head_count_kv` is not always a scalar. On
`gemma-4-31B` it is a 60-element array, mostly 16 with a 4 every sixth layer.
Collapsing it to its maximum overcounts every one of those layers.

**Sliding windows.** `attention.sliding_window_pattern` marks which layers are
local. Those layers stop growing once the window is full — on gemma-4, at 1024
tokens — and they use `key_length_swa` / `value_length_swa`, which is 256 rather
than 512. Fifty of gemma-4's sixty layers are local. Treating them as
full-context makes the cache grow eight times faster than it does.

**Hybrid recurrence.** `full_attention_interval` says how rarely a hybrid model
does real attention. `Qwen3.8-27B` and `Ornith-1.0-35B` both set it to 4, so 16
of 65 layers and 10 of 40 layers respectively hold a KV cache at all. The rest
hold a fixed-size recurrent state that does not grow with context.

## The correction, measured

| model | ctx | old estimate | corrected estimate | measured total |
|---|---|---|---|---|
| `gemma-4-31B-it-IQ4_NL` | 4096 | 3.98 GiB KV | 0.24 GiB KV | 17.76 GiB |
| `gemma-4-31B-it-IQ4_NL` | 32768 | 31.8 GiB KV (refused) | 1.74 GiB KV | **19.03 GiB** |
| `Qwen3.8-27B-IQ4_NL` | 32768 | 4.32 GiB KV | 1.22 GiB KV | 16.18 GiB |
| `Ornith-1.0-35B` | 32768 | 1.33 GiB KV | 0.40 GiB KV | 17.56 GiB |

The gemma-4 row is the whole point. Under the old estimate the launcher offered
4096 tokens and reported that 32768 would need 31.8 GiB on a 24 GiB machine.
32768 was then started for real and used **19.03 GiB** against a corrected
prediction of 19.34 — the model had eight times more context available than the
launcher was willing to admit.

## The thing a header cannot tell you

How much an expert-streaming MoE model will hold. The slab is sized to the
memory that is left, so the answer is a property of the machine, not the file.
An early version of the launcher assumed 0.70 × the weights and told a user
starting `Qwen3-Coder-Next` (36.54 GiB) that even streamed it would need 27.6
GiB and would not fit. It then started and ran in **9.74 GiB**, because
`MOESTREAM_CACHE_FRAC=learn` begins at 0.15 on a model it has not seen.

So the launcher no longer quotes a resident figure for MoE. It quotes what
*cannot* be streamed — attention and embeddings, about a fifth of the file —
and says the slab takes what is left. That claim is true on every machine.

## What this cost to find

Nothing in the header is wrong or missing. Every key needed is present in all
four models. The error was reading only the keys that a plain transformer has,
and quietly assuming the rest of the architecture matched. The same mistake in a
memory planner is invisible until someone with a gemma-class model wonders why
their 24 GiB machine only offers 4096 tokens — and never reports it, because a
launcher that suggests a number looks like it knows.
