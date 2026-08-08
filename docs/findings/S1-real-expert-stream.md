# Finding S1 — MoE FFN runs bit-identically with expert streaming on a real model

| | |
|---|---|
| Target | DESIGN.md §10.4 / §11.3 / §13.4 / §13.6 / §2.3 |
| Date | 2026-08-02 |
| Verdict | **PASS** (with three design corrections) |
| Model | `Ornith-1.0-35B-UD-IQ4_NL.gguf` (16.87 GiB, arch=`qwen35moe`) |
| Hardware | Ryzen 7 8745HS / Crucial P310 (PCIe4) / ext4 / kernel 7.0 |
| Reproduce | `make spike SPIKE=s1_real_expert_stream` (needs `-v /models`) |

## The real model's structure (measured)

| Item | Value |
|---|---|
| architecture | `qwen35moe` |
| layers / hidden | 40 / 2048 |
| experts / top_k | **256 / 8** |
| expert FFN width / shared | 512 / 512 |
| full_attention_interval | **4** (10 attention layers / 30 SSM layers) |
| SSM | conv_kernel 4, state_size 128, group 16, inner 4096 |
| vocab / n_ctx_train | 248,320 / 262,144 |
| n_params | 34.66 B |

**Almost exactly what design doc Appendix B assumed for Qwen3.6-35B-A3B** —
hybrid structure, 256 experts and top-8 all as predicted.

## Residency classification, measured (§9.5)

| Class | Tensors | Size | Share |
|---|---:|---:|---:|
| **STREAMED** (routed experts) | 120 | **14.48 GiB** | 85.9% |
| **ROW_LOOKUP** (token_embd) | 1 | **515.31 MiB** | 3.0% |
| **RESIDENT** | 612 | 1.88 GiB | 11.1% |
| Total | 733 | 16.86 GiB | |

→ **Keeping RESIDENT + ROW_LOOKUP resident costs 2.4 GiB.** §17's budget is
sound.
→ The 515 MiB saved by ROW_LOOKUP is confirmed on real data.

## Test results

| # | Check | Result |
|---|---|---|
| T1 | expert slices are contiguous | **PASS**: gate 450,560 B / up 450,560 B / down 589,824 B, all multiples of 4 KiB |
| T2 | is O_DIRECT usable | **PASS** (it reads), but tensor start offsets are **4000 mod 4096** (unaligned) |
| T3 | **streaming MoE FFN matches the fully resident version** | **PASS, `max\|diff\| = 0.000e+00` (bit-identical)** |
| T4 | reduction in bytes read | **PASS**: 364.0 MiB → 19.91 MiB (**18.3x less**) |
| T5 | effective bandwidth | see below |
| T6 | mixed types under UD quantization | **PASS** (works with per-role slabs) |

### What T3 means

Layer 0's MoE FFN (`down @ (silu(gate@x) * (up@x))`, weighted over top-8) was run
two ways:

- **reference**: all 256 experts resident (364 MiB), `ids` = expert_id
- **MoEStream**: only the 14 needed experts `pread` from SSD (19.91 MiB),
  `ids` = slot_id

and **the outputs matched to the bit**. On real weights, real quantization types
and a real file. Decisive evidence that §10.4's Slot Table + ID Remap holds under
production conditions.

### T5: measured bandwidth (★ the most important result)

```
through page cache : 14.23 GB/s   <- a measurement trap. Not the real disk
O_DIRECT QD=1      :  1.55 GB/s   <- the limit of synchronous pread (latency bound)
O_DIRECT QD=2      :  2.85 GB/s
O_DIRECT QD=4      :  4.28 GB/s
O_DIRECT QD=8      :  4.42 GB/s   <- saturates here
O_DIRECT QD=16     :  4.20 GB/s
O_DIRECT QD=32     :  4.46 GB/s
```

**2.9x from parallelism, saturating near QD=8.**

### Regime determination from measured values

| Item | Design assumption | **Measured** |
|---|---:|---:|
| one expert | 1.945 MiB | **1.422 MiB** |
| `B_act` | 622.4 MiB/token | **455.0 MiB/token** |
| `t_c` | 40 ms (estimated) | **43.7 ms** (llama-server at 22.88 tok/s) |
| effective BW | 6.0 GB/s (assumed) | **4.46 GB/s** |
| **hit rate needed for ≤20% slowdown** | 55.9% | **51.0%** |

Assumption and measurement happened to land close together (lower bandwidth
offset by smaller experts). **§2.3's conclusion stands.**

## Design corrections needed

### Correction 1: the QD ceiling is 8–16, not 64 (§13.5)

The design said `QD = 64`; measurement **saturates at QD=8**, and beyond that
only p99 latency suffers. Presumably a characteristic of DRAM-less (HMB) drives.

> Change `IoGovernor`'s adaptive control (§13.5) to start at **QD=8** with a
> ceiling of 32. Having adaptive control at all was right; the search range was
> wrong.

### Correction 2: quantization types differ **per layer** too (violating a §10.4 constraint) ★unresolved

Under UD (Unsloth Dynamic) quantization, types vary not only by role but **by
layer**:

| Role | Type breakdown |
|---|---|
| `ffn_down_exps` | IQ4_NL × 37 layers, **Q6_K × 3 layers** |
| `ffn_gate_exps` | IQ3_S × 39 layers, **IQ4_NL × 1 layer** |
| `ffn_up_exps` | IQ3_S × 39 layers, **IQ4_NL × 1 layer** |

§10.4 requires all slots in a slab to share a type, so **one slab shared across
all layers does not work**.

Options (undecided → tracked as ADR-0018):

| Option | Description | Assessment |
|---|---|---|
| A | separate slab pools per type (4 pools in this case) | implementable; causes slot fragmentation |
| B | unify types across layers when packing `.msp` (requantize) | violates NG-6 (no requantization) |
| C | promote only the minority layers (3+1+1) to RESIDENT | **promising**. ~0.5 GiB extra residency, simplest to implement |

> C looks best. The layers with minority types are only 5/120 = 4% of the total,
> so making them resident barely affects the budget.

### Correction 3: reading GGUF directly means O_DIRECT is not zero-copy (§13.6 corroborated)

Tensor start offsets are consistently **4000 mod 4096**. Read-around makes
O_DIRECT usable (the extra read is a mild 0.82%), but **the destination is offset
from the slot boundary, so a bounce-buffer copy is unavoidable**.

→ §13.6's case for needing `.msp` is corroborated on real data.
→ But since the extra read is only 0.82%, direct-GGUF mode is more practical than
  expected (costing just one copy).

## Remaining work

| ID | Item | Priority |
|---|---|---|
| **S0b** | verify the `mul_mat_id` slab on the Vulkan backend | **high** (the integrated GPU is the real target) |
| **S2** | io_uring bandwidth: can it beat QD=8's 4.46 GB/s? | **high** |
| S3 | bandwidth with `.msp` (2 MiB aligned); the real gain over direct GGUF | medium |
| M0-2 | expert activation distribution → can the required 51% hit rate be met? | **highest** (go/no-go) |
